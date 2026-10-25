import json

import pytest

import agent
import prompts_agent as P

BUDGET_TOKENS = 4000  # вход на один вызов, см. раздел 5 плана
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


@pytest.fixture
def regression_like():
    """Профиль размером с public-регрессию: 62 признака, русские заметки, длинный task.md."""
    names = ["Year_Factor", "State_Factor", "building_class", "facility_type", "floor_area", "year_built",
             "energy_star_rating", "ELEVATION"]
    names += [f"{m}_{s}_temp_c" for m in MONTHS for s in ("min", "avg", "max")]
    names += ["cooling_degree_days_c", "heating_degree_days_c", "precipitation_inches", "snowfall_inches",
              "snowdepth_inches", "avg_temp_c", "days_below_minus1C", "days_below_minus7C", "days_below_minus12C",
              "days_below_minus18C", "days_above_27C", "days_above_32C", "days_above_38C", "days_above_43C",
              "direction_max_wind_speed", "direction_peak_wind_speed", "max_wind_speed", "days_with_fog"]
    columns = []
    for name in names:
        if name in ("State_Factor", "building_class", "facility_type"):
            columns.append({"name": name, "dtype": "cat", "missing_pct": 0.0, "nunique": 60,
                            "top": ["Multifamily_Uncategorized", "Office_Uncategorized", "Education_Other"]})
        else:
            columns.append({"name": name, "dtype": "num", "missing_pct": 12.5, "nunique": 1234,
                            "min": -12.345678, "max": 98765.4321})
    profile = {"n_train": 53308, "n_test": 22449, "columns": columns,
               "target": {"name": "site_eui", "mean": 82.6, "std": 58.3, "min": 1.0, "max": 997.9, "skew": 3.1}}
    notes = {name: "Минимальная температура в январе месяце, градусы Цельсия" for name in names}
    task_text = ("Постройте регрессионную модель, которая по характеристикам здания и погодным показателям "
                 "оценивает его годовое удельное энергопотребление. ") * 40
    base_code = agent_like_code(3000)
    return profile, notes, task_text, base_code


def agent_like_code(chars: int) -> str:
    body = "".join(f'        df["f{i}"] = df["floor_area"] / (df["year_built"] + {i})\n' for i in range(200))
    code = ("import numpy as np\nimport pandas as pd\n\n\ndef build_features(train, test, spec):\n"
            "    X_train, X_test = train.copy(), test.copy()\n    for df in (X_train, X_test):\n" + body)
    return code[:chars] + "\n    return X_train, X_test\n"


def test_system_prompts_have_role_marker_and_render():
    plan = P.build_plan_messages(task_line="t", task_text="", profile="p", base_code="c", cv_text="cv",
                                 memory="m", n=5)
    code = P.build_code_messages(task_line="t", columns="c", insights="", base_code="x", cv_text="cv",
                                 hypothesis={"title": "h"}, memory="m", model_params=agent.MODEL_PARAMS,
                                 timeout=300)
    critic = P.build_critic_messages(task_line="t", memory="m")
    assert [m["role"] for m in plan] == ["system", "user"]
    assert plan[0]["content"].startswith("Role: planner.") and "up to 5 NEW hypotheses" in plan[0]["content"]
    assert code[0]["content"].startswith("Role: coder.") and "finish in 300 s" in code[0]["content"]
    assert "num_leaves" in code[0]["content"] and "depth" in code[0]["content"]
    assert critic[0]["content"].startswith("Role: critic.")


def test_few_shot_examples_are_valid():
    example = P.PLANNER_SYSTEM.format(n=6).split("Example answer:\n")[1]
    hypotheses = json.loads(example)["hypotheses"]
    assert all(h["kind"] in P.KINDS for h in hypotheses)
    coder = P.CODER_SYSTEM.format(params_help="x", timeout=300)
    assert agent.validate_code(agent.extract_code(coder.split("Example answer:\n")[1])) == []


def test_plan_prompt_fits_budget(regression_like):
    profile, notes, task_text, base_code = regression_like
    memory = "\n".join(f"#{i} [rejected] features | some hypothesis title number {i} | cv=41.2 (gain -0.01)"
                       for i in range(40))
    messages = P.build_plan_messages(
        task_line="type: regression | metric: rmse (lower is better) | target: site_eui",
        task_text=task_text, profile=P.render_profile(profile, notes, max_chars=5000), base_code=base_code,
        cv_text="CV rmse = 41.23", memory=memory, n=6)
    assert P.messages_tokens(messages) <= BUDGET_TOKENS, P.messages_tokens(messages)
    assert "site_eui" in messages[1]["content"] and "facility_type" in messages[1]["content"]


def test_code_and_repair_prompts_fit_budget(regression_like):
    profile, _, _, base_code = regression_like
    hypothesis = {"title": "Log of floor_area", "kind": "features", "rationale": "skewed " * 100}
    common = dict(task_line="type: regression | metric: rmse", columns=P.render_columns_brief(profile),
                  model_params=agent.MODEL_PARAMS, timeout=300)
    code = P.build_code_messages(**common, insights="x" * 2000, base_code=base_code, cv_text="CV rmse = 41.2",
                                 hypothesis=hypothesis, memory="m" * 5000)
    assert P.messages_tokens(code) <= BUDGET_TOKENS, P.messages_tokens(code)
    assert base_code.strip() in code[1]["content"] and "Log of floor_area" in code[1]["content"]

    error = "Traceback (most recent call last):\n" + "  frame\n" * 1000 + "KeyError: 'floor_area_x'"
    repair = P.build_repair_messages(**common, hypothesis=hypothesis, code=base_code, error=error,
                                     error_chars=1500)
    assert P.messages_tokens(repair) <= BUDGET_TOKENS, P.messages_tokens(repair)
    assert "KeyError: 'floor_area_x'" in repair[1]["content"] and "frame\n" * 300 not in repair[1]["content"]


def test_format_fix_appends_turns():
    base = [{"role": "system", "content": "Role: coder."}, {"role": "user", "content": "u"}]
    fixed = P.build_format_fix_messages(base, "blah " * 1000)
    assert [m["role"] for m in fixed] == ["system", "user", "assistant", "user"]
    assert len(fixed[2]["content"]) <= 1500 and "```python" in fixed[3]["content"]


def test_render_profile_truncates_many_columns():
    columns = [{"name": f"col_{i}", "dtype": "num", "missing_pct": 0, "nunique": 10, "min": 0, "max": 1}
               for i in range(600)]
    text = P.render_profile({"n_train": 10, "n_test": 5, "columns": columns}, max_chars=3000)
    assert len(text) <= 3000 and "more columns" in text and "col_0 " in text
    assert P.render_profile({"anything": "else"}, max_chars=50).startswith('{"anything"')


def test_render_columns_brief_groups_by_type():
    profile = {"columns": [{"name": "a", "dtype": "num"}, {"name": "b", "dtype": "cat", "nunique": 3},
                           {"name": "c", "dtype": "num"}]}
    assert P.render_columns_brief(profile) == "num: a, c\ncat: b (3 levels)"


def test_clip_and_estimate():
    assert P.clip("abc", 10) == "abc"
    assert P.clip("x" * 100, 30).endswith("...[truncated]...") and len(P.clip("x" * 100, 30)) <= 30
    assert P.clip("a" * 50 + "END", 30, tail=True).endswith("END")
    assert P.estimate_tokens("abcd" * 10) == 10 and P.estimate_tokens("абвг") == 2


def history(columns: list[str], n: int):
    """n экспериментов с реалистичными названиями, диффами и ошибками на колонках задачи."""
    from memory import Experiment, ExperimentMemory
    memory = ExperimentMemory(columns=columns)
    kinds = ["features", "encoding", "params", "missing", "feature_selection", "outliers"]
    for i in range(1, n + 1):
        a, b = columns[i % len(columns)], columns[(3 * i) % len(columns)]
        status = "accepted" if i % 5 == 0 else "failed" if i % 7 == 0 else "rejected"
        diff = "\n".join(f'+        df["{a}_x_{b}_{j}"] = df["{a}"] * df["{b}"] / (df["{b}"].abs() + {j})'
                         for j in range(8))
        memory.add(Experiment(
            id=i, hypothesis=f"Ratio of {a} to {b} with a smoothing term, variant {i}", kind=kinds[i % 6],
            status=status, cv_score=41.0 - i / 100, gain=0.05 if status == "accepted" else -0.01,
            error=("Traceback ... KeyError: 'no_such_col'" if status == "failed"
                   else "harness: PARAMS['is_unbalance'] is not allowed for lightgbm, ignored | gain -0.01 <= 0.05"),
            columns=[a, b], diff=diff))
    memory.add_note("Ratios of area to weather help a bit; encodings of facility_type do not. Try group stats. " * 3)
    return memory


@pytest.mark.parametrize("n", [12, 30])
def test_prompts_with_long_memory_fit_budget(regression_like, n):
    profile, notes, task_text, base_code = regression_like
    columns = [c["name"] for c in profile["columns"]]
    memory = history(columns, n)
    cfg = agent.AgentConfig()
    kinds = P.KINDS + (P.MODEL_SEARCH_KIND, P.TUNE_KIND)  # худший случай: все виды-стратегии доступны
    brief = memory.planner_brief(cfg.memory_chars, kinds=kinds, task_type="regression")
    plan = P.build_plan_messages(
        task_line="type: regression | metric: rmse (lower is better) | target: site_eui", task_text=task_text,
        profile=P.render_profile(profile, notes, max_chars=cfg.profile_chars), base_code=base_code,
        cv_text="CV rmse = 41.23", memory=brief, n=6, memory_chars=cfg.memory_chars,
        errors="e" * 2000, errors_chars=cfg.errors_chars, can_tune=True, can_search_models=True)
    code = P.build_code_messages(
        task_line="type: regression | metric: rmse", columns=P.render_columns_brief(profile), insights="x" * 500,
        base_code=base_code, cv_text="CV rmse = 41.2", hypothesis={"title": "Log of floor_area", "kind": "features"},
        memory=memory.summary(cfg.coder_memory_chars, recent=4), memory_chars=cfg.coder_memory_chars,
        model_params=agent.MODEL_PARAMS, timeout=300)
    assert P.messages_tokens(plan) <= BUDGET_TOKENS, P.messages_tokens(plan)
    assert P.messages_tokens(code) <= BUDGET_TOKENS, P.messages_tokens(code)
    assert brief in plan[1]["content"]  # сводка не обрезана сборщиком промпта
    assert "Accepted steps" in brief and "Kinds not tried yet: target_transform, model" in brief
    assert "Accepted steps" in code[1]["content"]
    print(f"\nn={n}: planner {P.messages_tokens(plan)} tok (memory {len(brief)} chars), "
          f"coder {P.messages_tokens(code)} tok")
