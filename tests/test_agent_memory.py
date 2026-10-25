import json

import pytest

from memory import (Experiment, ExperimentMemory, code_diff, code_hash, columns_in, mentioned_columns,
                    normalize_title, same_idea)


def exp(memory, title, status="rejected", **kw):
    return memory.add(Experiment(id=memory.next_id(), hypothesis=title, kind=kw.pop("kind", "features"),
                                 status=status, **kw))


def test_add_assigns_ids_and_rejects_unknown_status():
    memory = ExperimentMemory()
    exp(memory, "log area", "accepted", cv_score=41.2, gain=0.5)
    exp(memory, "freq encoding", "failed", error="KeyError: 'x'")
    assert [e.id for e in memory.experiments] == [1, 2]
    assert memory.tried_titles() == ["log area", "freq encoding"]
    assert memory.tried_kinds() == {"features": 2}
    with pytest.raises(ValueError):
        memory.add(Experiment(id=3, hypothesis="x", kind="features", status="maybe"))


def test_summary_mentions_status_score_and_error():
    memory = ExperimentMemory()
    assert memory.summary() == "No experiments yet."
    exp(memory, "log area", "accepted", cv_score=41.2346, gain=0.51)
    exp(memory, "freq encoding", "failed", error="Traceback...\nKeyError: 'x'")
    memory.add_note("Interactions help; encodings fail on NaN.")
    text = memory.summary()
    assert "#1 [accepted] features | log area | cv=41.235 (gain +0.51)" in text
    assert "#2 [failed] features | freq encoding | error: Traceback... KeyError: 'x'" in text
    assert "Critic notes:" in text and "Interactions help" in text


@pytest.mark.parametrize("max_chars", [200, 500, 1500])
def test_summary_respects_limit_and_keeps_latest(max_chars):
    memory = ExperimentMemory()
    for i in range(30):
        exp(memory, f"hypothesis number {i} with a fairly long descriptive title", cv_score=40 - i * 0.01,
            gain=0.01)
    text = memory.summary(max_chars=max_chars)
    assert len(text) <= max_chars
    assert "#30 " in text  # самая свежая запись не выкидывается


def test_dump_jsonl_roundtrip(tmp_path):
    memory = ExperimentMemory()
    exp(memory, "a", "accepted", cv_score=1.0, cv_std=float("nan"), gain=0.1)
    exp(memory, "b", "failed", error="boom")
    path = memory.dump_jsonl(tmp_path / "sub" / "experiments.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["hypothesis"] for r in rows] == ["a", "b"]
    assert rows[0]["cv_std"] is None and rows[1]["error"] == "boom"


def test_code_hash_ignores_blank_lines_and_trailing_spaces():
    assert code_hash("a = 1\n\n\nb = 2  \n") == code_hash("a = 1\nb = 2")
    assert code_hash("a = 1") != code_hash("a = 2")
    memory = ExperimentMemory()
    exp(memory, "x", code_hash=code_hash("a = 1"))
    assert memory.find_code(code_hash("a = 1\n")).id == 1 and memory.find_code("nope") is None


def test_normalize_title():
    assert normalize_title("  Log-transform of Floor_Area!! ") == "log transform of floor area"


def test_load_jsonl_roundtrip_new_and_old_format(tmp_path):
    memory = ExperimentMemory(columns=["a", "b"])
    exp(memory, "ratio a/b", "accepted", cv_score=1.0, gain=0.1, columns=["a", "b"], diff="+ df['r'] = df['a']",
        base_hash="abc")
    exp(memory, "boom", "failed", error="KeyError")
    path = memory.dump_jsonl(tmp_path / "experiments.jsonl")
    with path.open("a", encoding="utf-8") as fh:  # старый формат, лишний ключ и мусор
        fh.write('{"id": 3, "hypothesis": "old", "kind": "params", "status": "rejected", "extra": 1}\n')
        fh.write("not json\n")
        fh.write('{"id": 4, "hypothesis": "bad", "kind": "x", "status": "maybe"}\n')
    loaded = ExperimentMemory.load_jsonl(path, columns=["a", "b"])
    assert [e.hypothesis for e in loaded.experiments] == ["ratio a/b", "boom", "old"]
    assert loaded.experiments[0].columns == ["a", "b"] and loaded.experiments[0].base_hash == "abc"
    assert loaded.experiments[2].columns == [] and loaded.experiments[2].diff == ""
    assert loaded.columns == ["a", "b"] and loaded.next_id() == 4
    assert ExperimentMemory.load_jsonl(tmp_path / "missing.jsonl").experiments == []


BASE = 'def build_features(train, test, spec):\n    X = train.copy()\n\n    return X, test\n'


def test_code_diff_shows_only_changed_lines():
    new = BASE.replace("    return", '    X["a_b"] = X["a"] / X["b"].replace(0, np.nan)\n    return')
    assert code_diff(BASE, new) == '+    X["a_b"] = X["a"] / X["b"].replace(0, np.nan)'
    assert code_diff(BASE, BASE + "\n\n   \n") == ""  # пустые строки и хвостовые пробелы не изменение
    renamed = new.replace("X =", "frame =")
    assert code_diff(new, renamed).splitlines() == ["-    X = train.copy()", "+    frame = train.copy()"]


def test_code_diff_long_is_clipped():
    new = BASE.replace("    return", "".join(f'    X["f{i}"] = {i}\n' for i in range(40)) + "    return")
    lines = code_diff(BASE, new, max_lines=5).splitlines()
    assert len(lines) == 6 and lines[-1] == "... (+35 more changed lines)"
    assert len(code_diff(BASE, new, max_lines=0).splitlines()) == 40


def test_code_diff_clipping_prefers_added_lines():
    old = "def f():\n" + "".join(f"    old_{i} = {i}\n" for i in range(10))
    new = "def f():\n    added = 1\n"
    lines = code_diff(old, new, max_lines=3).splitlines()
    assert lines[:3] == ["-    old_0 = 0", "-    old_1 = 1", "+    added = 1"]  # «+» не потерялся, порядок исходный
    assert lines[3] == "... (+8 more changed lines)"


def test_columns_in_uses_only_string_literals_of_known_columns():
    code = """df["floor_area"] / df['year_built']; x = 'id'; s = "floor_area_log"; # floor_area in comment
spec.get("cat_cols"); df["it's"]"""
    known = ["floor_area", "year_built", "id", "facility_type", "it's"]
    assert columns_in(code, known) == ["floor_area", "year_built", "id", "it's"]
    assert columns_in("", known) == [] and columns_in(code, []) == []


COLUMNS = ["floor_area", "year_built", "Year_Factor", "State_Factor", "facility_type", "energy_star_rating",
           "sbytes", "dbytes", "proto", "city", "x1", "x2", "x3"]

# (kind, title_a, title_b, дубль?) — в т.ч. реальные названия из живых прогонов Qwen
SAME_IDEA = [
    ("encoding", "Frequency encoding of facility_type", "Count encoding facility_type", True),
    ("encoding", "Frequency encoding for high-cardinality categoricals",
     "Count encoding of high cardinality categorical columns", True),
    ("features", "Log1p transform floor_area", "Apply log to floor_area", True),
    ("features", "Ratio sbytes / dbytes", "sbytes divided by dbytes", True),
    ("features", "Group mean of floor_area by facility_type", "Mean floor_area per facility_type", True),
    ("missing", "Missing indicator for year_built", "NaN flag year_built", True),
    ("missing", "Impute year_built with median", "Fill missing year_built with median", True),
    ("params", "Increase lightgbm num_leaves", "Raise num_leaves of LightGBM", True),
    ("model", "Switch to CatBoost", "Use CatBoost model", True),
    ("target_transform", "Log1p target transformation", "Log transform of the target", True),
    ("feature_selection", "Drop redundant monthly temp cols", "Remove redundant monthly temperature columns", True),
    ("outliers", "Clip outliers of floor_area", "Winsorize floor_area at 1%", True),
    ("features", "Interaction x1*x2", "interaction: x1 * x2", True),
    ("features", "Ratio sbytes / dbytes", "Difference sbytes - dbytes", False),
    ("encoding", "Frequency encoding of facility_type", "Frequency encoding of State_Factor", False),
    ("features", "Log1p transform floor_area", "Log1p transform energy_star_rating", False),
    ("params", "Increase num_leaves", "Decrease num_leaves", False),
    ("params", "Raise learning_rate of lightgbm to 0.1", "Lower lightgbm learning_rate", False),
    ("params", "Increase learning rate", "Increase n_estimators", False),
    ("features", "Age of building (Year_Factor - year_built)", "Building floor area per year", False),
    ("features", "Create protocol-specific interaction features", "Add asymmetry and ratio features", False),
    ("encoding", "Count encoding of city", "Rare category grouping of city", False),
    ("features", "Interaction x1*x2", "Same interaction again", False),
    ("features", "Interaction x1*x2", "Interaction x1*x3", False),
    ("missing", "Missing indicator for year_built", "Missing indicator for energy_star_rating", False),
    ("features", "Ratio of heating to cooling degree days", "Frequency encoding of facility_type", False),
]


@pytest.mark.parametrize("kind, a, b, expected", SAME_IDEA)
def test_same_idea_table(kind, a, b, expected):
    assert same_idea(a, kind, b, kind, COLUMNS) is expected
    assert same_idea(b, kind, a, kind, COLUMNS) is expected  # симметрично


def test_same_idea_needs_same_kind():
    assert not same_idea("Frequency encoding of city", "encoding", "Frequency of city", "features", COLUMNS)


def test_mentioned_columns_whole_words_and_spaces():
    assert mentioned_columns("log of Floor Area and year_built_x", COLUMNS) == {"floor_area"}
    assert mentioned_columns("x1*x2 vs x10", COLUMNS) == {"x1", "x2"}


def test_find_similar_rejected_blocks_failed_allows_one_retry():
    memory = ExperimentMemory(columns=COLUMNS)
    exp(memory, "Frequency encoding of facility_type", "rejected", kind="encoding")
    exp(memory, "Ratio sbytes / dbytes", "failed", error="KeyError")
    assert memory.find_similar("Count encoding facility_type", "encoding").id == 1
    assert memory.find_similar("Count encoding facility_type", "features") is None
    assert memory.find_similar("sbytes divided by dbytes", "features") is None  # одно падение — можно повторить
    exp(memory, "sbytes / dbytes ratio", "failed", error="KeyError")
    assert memory.find_similar("sbytes divided by dbytes", "features").id == 3


def long_history(n=30):
    """n экспериментов разных видов: 3 принятых с диффами, падения с разными причинами, заметки критика."""
    memory = ExperimentMemory(columns=COLUMNS)
    kinds = ["features", "encoding", "params", "missing", "model", "feature_selection"]
    for i in range(1, n + 1):
        kind = kinds[i % len(kinds)]
        if i in (3, 11, 20):
            exp(memory, f"accepted idea {i} on floor_area", "accepted", kind=kind, cv_score=40 - i / 10, gain=0.1 * i,
                diff="\n".join(f'+        df["f{i}_{j}"] = df["floor_area"] * {j}' for j in range(6)))
        elif i % 7 == 0:
            exp(memory, f"crashing idea {i}", "failed", kind=kind,
                error="timeout: building features took longer than 300 s" if i % 2 else "KeyError: 'zzz'")
        else:
            exp(memory, f"rejected idea number {i} with a long descriptive title", kind=kind, cv_score=41.0,
                gain=-0.01, error="gain -0.01 <= threshold 0.05")
    for k in range(3):
        memory.add_note(f"critic note {k}: interactions of area help, encodings do not " * 3)
    return memory


@pytest.mark.parametrize("max_chars", [400, 800, 1500, 3000])
def test_summary_keeps_accepted_and_fits(max_chars):
    text = long_history().summary(max_chars=max_chars)
    assert len(text) <= max_chars
    assert "#30 " in text  # самый свежий не выкидывается
    if max_chars >= 800:
        assert all(f"#{i} [accepted]" in text for i in (3, 11, 20))


def test_summary_sections_on_long_history():
    text = long_history().summary(max_chars=1500)
    assert text.startswith("Accepted steps (already in the current best):")
    assert "By kind: " in text
    for kind in ["features", "encoding", "params", "missing", "model", "feature_selection"]:
        assert f"{kind}: " in text, kind
    assert "1 accepted (best gain +2)" in text  # #20 — params
    assert "failed (timeout)" in text or "failed (KeyError)" in text
    assert "Recent experiments (oldest first):" in text


def test_summary_with_room_shows_diffs_tried_and_notes():
    text = long_history().summary(max_chars=6000)
    assert '    + df["f3_0"] = df["floor_area"] * 0' in text and "... (+3 lines)" in text  # отступы схлопнуты
    assert "Already tried, do not repeat: rejected idea number 1 with" in text
    assert "Critic notes:" in text


def test_summary_drops_low_priority_first():
    memory = long_history()
    full, mid = memory.summary(max_chars=6000), memory.summary(max_chars=1500)
    assert "Critic notes:" in full and "Critic notes:" not in mid  # заметки режутся первыми
    assert "By kind: " in mid  # итог по видам — дольше всех после принятых


def test_summary_short_history_has_no_kind_totals():
    memory = ExperimentMemory()
    exp(memory, "a", "rejected", cv_score=1.0, gain=-0.1)
    assert "By kind" not in memory.summary() and "Recent experiments" in memory.summary()


KINDS = ("features", "encoding", "missing", "outliers", "target_transform", "params", "model", "feature_selection")


def test_coverage_untried_exhausted_untouched():
    memory = ExperimentMemory(columns=COLUMNS)
    for i in range(3):
        exp(memory, f"params try {i}", "rejected", kind="params")
    exp(memory, "Ratio sbytes / dbytes", "accepted", kind="features", columns=["sbytes", "dbytes"])
    exp(memory, "Count encoding of facility_type", "failed", kind="encoding")  # колонка из названия
    cov = memory.coverage(kinds=KINDS, task_type="multiclass")
    assert cov["untried_kinds"] == ["missing", "outliers", "model", "feature_selection"]  # без target_transform
    assert cov["exhausted_kinds"] == ["params"]
    assert {"sbytes", "dbytes", "facility_type"}.isdisjoint(cov["untouched_columns"])
    assert cov["n_untouched_columns"] == len(COLUMNS) - 3
    assert "target_transform" in memory.coverage(kinds=KINDS, task_type="regression")["untried_kinds"]
    assert len(memory.coverage(kinds=KINDS, max_columns=4)["untouched_columns"]) == 4


def test_planner_brief_appends_coverage_and_fits():
    memory = long_history()
    for max_chars in (600, 1500):
        text = memory.planner_brief(max_chars, kinds=KINDS, task_type="regression")
        assert len(text) <= max_chars
        assert "Kinds not tried yet: outliers, target_transform" in text
        assert text.rstrip().splitlines()[-1].startswith(("Columns no experiment", "Exhausted", "Kinds not"))
    assert "Exhausted (3+ tries, nothing accepted)" in memory.planner_brief(1500, kinds=KINDS)
    assert "features" in memory.planner_brief(1500, kinds=KINDS).split("Exhausted")[1]
    assert ExperimentMemory().planner_brief(kinds=KINDS) == "No experiments yet."
