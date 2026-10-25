import numpy as np
import pandas as pd
import pytest
from p3_fixtures import Spec, classification_task, regression_task, write_task
from sklearn.model_selection import StratifiedKFold

import fallback as F
import metrics
import submission

FAST = dict(time_budget=60, threads=1, models=("lightgbm",), max_rounds=300)


def _cfg(**kw):
    return F.FallbackConfig(**{**FAST, **kw})


def test_regression_end_to_end(tmp_path):
    spec, y_test = regression_task(tmp_path, n_train=800, n_test=200)
    cand = F.run(spec, config=_cfg(models=("lightgbm", "catboost", "xgboost")))
    assert submission.check_file(spec) == []
    assert cand.oof.shape == (800,) and cand.test_pred.shape == (200,)
    assert len(cand.meta["fold_scores"]) == cand.meta["n_splits"] == 5
    assert cand.meta["folds"].shape == (800,) and cand.meta["classes"] is None
    assert {r["name"] for r in cand.meta["runs"]} >= {"lightgbm", "catboost", "xgboost"}
    constant = metrics.score("rmse", y_test, np.full(200, y_test.mean()))
    written = pd.read_csv(tmp_path / "submission.csv")["target"]
    assert metrics.score("rmse", y_test, written) < 0.5 * constant  # модель заметно лучше константы


def test_multiclass_string_labels(tmp_path):
    spec, y_test = classification_task(tmp_path, n_classes=4)
    cand = F.run(spec, config=_cfg())
    assert cand.meta["classes"] == ["alpha", "beta", "delta", "gamma"]
    assert cand.test_pred.shape == (300, 4) and np.allclose(cand.test_pred.sum(axis=1), 1)
    out = pd.read_csv(tmp_path / "submission.csv")
    assert set(out["label"]) <= set(cand.meta["classes"]) and submission.check_file(spec) == []
    assert cand.meta["cv_score_post"] >= cand.cv_score  # пост-обработка принимается, только если не хуже
    assert metrics.score("macro_f1", y_test, out["label"]) > 0.5


def test_binary_auc_writes_probabilities(tmp_path):
    spec, _ = classification_task(tmp_path, n_classes=2, labels=[0, 1], metric="roc_auc")
    cand = F.run(spec, config=_cfg())
    out = pd.read_csv(tmp_path / "submission.csv")
    assert out["label"].between(0, 1).all() and out["label"].nunique() > 10
    assert np.allclose(out["label"], cand.test_pred[:, 1])
    assert cand.meta["decision"] is None and 0.5 < cand.cv_score <= 1


def test_imbalanced_macro_f1_plans_balanced_variant(tmp_path):
    spec, _ = classification_task(tmp_path, n_classes=5, imbalance=True)
    train = pd.read_csv(tmp_path / "train.csv")
    train.drop(train.index[train["label"] == "eps"][20:]).to_csv(tmp_path / "train.csv", index=False)  # 20 объектов
    data = F.load(spec, _cfg())
    counts = np.bincount(data.y)
    assert counts.max() / counts.min() >= 5
    names = [name for name, _, _ in F._plan(data, F.FallbackConfig())]
    assert names == ["lightgbm", "lightgbm_balanced", "xgboost", "catboost"]


def test_failing_model_does_not_break_others(tmp_path, monkeypatch):
    spec, _ = regression_task(tmp_path)
    original = F._fit_predict

    def flaky(model, *args, **kw):
        if model == "catboost":
            raise RuntimeError("boom")
        return original(model, *args, **kw)
    monkeypatch.setattr(F, "_fit_predict", flaky)
    cand = F.run(spec, config=_cfg(models=("catboost", "lightgbm")))
    assert cand.meta["model"] == "lightgbm" and submission.check_file(spec) == []


def test_all_models_fail_gives_constant(tmp_path, monkeypatch):
    spec, _ = classification_task(tmp_path, n_classes=3)
    monkeypatch.setattr(F, "_fit_predict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no models")))
    cand = F.run(spec, config=_cfg())
    assert cand.name == "fallback_constant" and submission.check_file(spec) == []


def test_unknown_metric_and_broken_deadline(tmp_path):
    spec, _ = regression_task(tmp_path)
    spec.metric = "some exotic score"

    class Deadline:
        def left(self):
            return 10_000.0

        def stage(self, name):
            raise KeyError(name)
    cand = F.run(spec, Deadline(), config=_cfg())
    assert cand.meta["metric"] == "rmse" and submission.check_file(spec) == []


def test_short_deadline_is_respected(tmp_path):
    spec, _ = classification_task(tmp_path, n_train=3000, n_test=500, n_classes=3)

    class Deadline:
        def left(self):
            return 30.0  # 30 с − резерв 20 с → бюджет упирается в нижнюю границу 20 с
    import time
    start = time.monotonic()
    F.run(spec, Deadline(), config=F.FallbackConfig(threads=1))
    assert time.monotonic() - start < 45 and submission.check_file(spec) == []


def test_tiny_dataset_and_rare_class(tmp_path):
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"id": range(30), "x": rng.normal(size=30), "y": ["a"] * 14 + ["b"] * 15 + ["c"]})
    test = pd.DataFrame({"id": range(30, 40), "x": rng.normal(size=10)})
    write_task(tmp_path, train, test, pd.DataFrame({"id": test.id, "y": "a"}))
    spec = Spec(tmp_path, "id", "y", "multiclass", "macro_f1", True)
    cand = F.run(spec, config=_cfg())
    assert cand.test_pred.shape == (10, 3) and submission.check_file(spec) == []


def test_continuous_target_is_regression_whatever_hints_say(tmp_path):
    spec, _ = regression_task(tmp_path)
    spec.task_type, spec.metric = "multiclass", "macro_f1"  # ошибка понимания задачи
    cand = F.run(spec, config=_cfg())
    assert cand.meta["metric"] == "rmse" and cand.oof.ndim == 1 and submission.check_file(spec) == []
    y = pd.Series(np.random.default_rng(0).normal(size=500))
    assert F.infer_task_type(y, "accuracy") == "regression"
    assert F.infer_task_type(pd.Series([0, 1, 2] * 50), "accuracy") == "multiclass"


def test_folds_match_stratified_kfold_with_seed():
    y = np.array([0] * 50 + [1] * 30 + [2] * 20)
    folds = F.make_folds(y, "multiclass", 5, 42)
    for i, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(100), y)):
        assert (folds[valid] == i).all()


def test_guess_spec_from_files(tmp_path):
    regression_task(tmp_path / "reg")
    spec = F.guess_spec(tmp_path / "reg" / "task.md")
    assert (spec.id_col, spec.target, spec.task_type, spec.metric) == ("id", "target", "regression", "rmse")
    classification_task(tmp_path / "clf", n_classes=2, labels=["no", "yes"], metric="ROC AUC")
    spec = F.guess_spec(tmp_path / "clf")
    assert (spec.target, spec.task_type, spec.metric) == ("label", "binary", "roc_auc")


def test_log_target_tried_for_skewed_regression(tmp_path):
    spec, _ = regression_task(tmp_path)
    train = pd.read_csv(tmp_path / "train.csv")
    train["target"] = np.exp(train["target"] / 3)  # положительный скошенный таргет
    train.to_csv(tmp_path / "train.csv", index=False)
    cand = F.run(spec, config=_cfg(write=False))
    assert "lightgbm_log" in {r["name"] for r in cand.meta["runs"]}
    assert not (tmp_path / "submission.csv").exists()  # write=False — файл не трогаем


def test_multi_target_regression(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    df = pd.DataFrame({"id": range(n), "x": x, "t1": 2 * x + rng.normal(scale=.1, size=n), "t2": -x})
    write_task(tmp_path, df.iloc[:200], df.iloc[200:, :2], pd.DataFrame({"id": df.id[200:], "t1": 0.0, "t2": 0.0}))
    spec = F.guess_spec(tmp_path)
    assert spec.target == ["t1", "t2"] and spec.task_type == "regression"
    cand = F.run_multi(spec, config=_cfg())
    assert cand.test_pred.shape == (100, 2) and submission.check_file(spec) == []


def test_n_threads_is_positive():
    assert F.n_threads() >= 1
    assert F.model_threads("lightgbm", 3_000, 8) == 1 and F.model_threads("catboost", 3_000, 8) == 8
    assert F.model_threads("xgboost", 87_670, 8) == 4 and F.model_threads("lightgbm", 53_308, 8) == 4


@pytest.mark.parametrize("budget_left, expected", [(1000.0, 900.0), (100.0, 80.0), (5.0, 20.0)])
def test_budget_from_deadline(budget_left, expected):
    class Deadline:
        def left(self):
            return budget_left
    assert F._budget(Deadline(), F.FallbackConfig()) == pytest.approx(expected)


def test_period_column_detection():
    train = pd.DataFrame({"year": [1, 2, 3, 4, 5] * 20, "x": np.arange(100.0), "code": [1, 2, 3, 4, 5] * 20})
    test = pd.DataFrame({"year": [6] * 10, "x": np.arange(10.0), "code": [3] * 10})
    assert F.find_period_column(train, test) == "year"
    assert F.find_period_column(train[["code"]], test[["code"]]) is None  # значения test встречаются в train
    assert F.find_period_column(train[["x"]], test[["x"]] + 1000) is None  # непрерывная колонка — не период


def test_period_cv_groups_whole_years(tmp_path):
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"id": range(n), "year": rng.integers(1, 6, n), "x": rng.normal(size=n)})
    df["y"] = df.x * 2 + df.year + rng.normal(scale=.1, size=n)
    test = pd.DataFrame({"id": range(n, n + 100), "year": 6, "x": rng.normal(size=100)})
    write_task(tmp_path, df, test, pd.DataFrame({"id": test.id, "y": 0.0}))
    spec = Spec(tmp_path, "id", "y", "regression", "rmse")
    data = F.load(spec, _cfg())
    assert data.cv_scheme["kind"] == "period" and data.cv_scheme["column"] == "year"
    for f in range(data.n_splits):
        assert df.year[data.folds == f].nunique() == 1  # каждый фолд — целый год
    cand = F.run(spec, config=_cfg())
    assert "lightgbm_noperiod" in {r["name"] for r in cand.meta["runs"]} and cand.meta["cv_scheme"]["kind"] == "period"


def test_duplicate_aware_cv(tmp_path):
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"x": rng.integers(0, 30, 300), "z": rng.integers(0, 3, 300)})
    train = pd.concat([base] * 3, ignore_index=True)  # каждая строка трижды
    train["label"] = np.where(train.x > 15, "a", "b")
    train.insert(0, "id", range(len(train)))
    test = pd.DataFrame({"id": range(900, 1000), "x": rng.integers(100, 200, 100), "z": 1})  # близнецов в train нет
    write_task(tmp_path, train, test, pd.DataFrame({"id": test.id, "label": "a"}))
    data = F.load(Spec(tmp_path, "id", "label", "binary", "accuracy"), _cfg())
    assert data.cv_scheme["kind"] == "duplicates" and data.cv_scheme["twin_share"] == 0
    feats = train[["x", "z"]].astype(str).agg("|".join, axis=1)
    assert (pd.Series(data.folds).groupby(feats.to_numpy()).nunique() == 1).all()  # одинаковые строки в одном фолде
    test2 = train.iloc[:100].drop(columns="label")  # test из тех же строк — схема обычная
    write_task(tmp_path, train, test2, pd.DataFrame({"id": test2.id, "label": "a"}))
    assert F.load(Spec(tmp_path, "id", "label", "binary", "accuracy"), _cfg()).cv_scheme["kind"] == "random"


def test_label_shift_detected_and_used(tmp_path):
    rng = np.random.default_rng(2)

    def frame(n, prior):
        y = rng.choice(3, n, p=prior)
        return pd.DataFrame({"x1": rng.normal(size=n) + 1.5 * (y == 1), "x2": rng.normal(size=n) + 1.5 * (y == 2),
                             "label": np.array(["a", "b", "c"])[y]})
    train, test = frame(4000, [0.7, 0.2, 0.1]), frame(2000, [0.3, 0.3, 0.4])
    train.insert(0, "id", range(4000))
    test = test.drop(columns="label")
    test.insert(0, "id", range(4000, 6000))
    write_task(tmp_path, train, test, pd.DataFrame({"id": test.id, "label": "a"}))
    spec = Spec(tmp_path, "id", "label", "multiclass", "macro_f1", True)
    default = F.run(spec, config=_cfg())  # по умолчанию — только диагностика
    assert default.meta["label_shift"]["used"] is False and default.meta["label_shift"]["tv"] > 0.2
    cand = F.run(spec, config=_cfg(label_shift=True))
    shift = cand.meta["label_shift"]
    assert shift["used"] and shift["tv"] > 0.2 and abs(shift["test"]["c"] - 0.4) < 0.08
    assert submission.check_file(Spec(tmp_path, "id", "label", "multiclass", "macro_f1", True)) == []


def test_no_label_shift_when_same_distribution(tmp_path):
    spec, _ = classification_task(tmp_path, n_train=2000, n_test=1000, n_classes=3)
    cand = F.run(spec, config=_cfg())
    assert cand.meta["label_shift"]["used"] is False


def test_threads_from_environment(monkeypatch):
    monkeypatch.setenv("FALLBACK_THREADS", "3")
    assert F.n_threads() == 3
    monkeypatch.setenv("FALLBACK_THREADS", "zero")
    assert F.n_threads() >= 1


def test_cv_folds_match_run_and_skip_rows_without_target(tmp_path):
    spec, _ = regression_task(tmp_path)
    train = pd.read_csv(tmp_path / "train.csv")
    train.loc[[3, 7], "target"] = np.nan  # строки без таргета fallback выбрасывает — evaluator должен знать какие
    train.to_csv(tmp_path / "train.csv", index=False)
    folds, rows, scheme = F.cv_folds(spec, _cfg())
    cand = F.run(spec, config=_cfg())
    assert np.array_equal(folds, cand.meta["folds"]) and scheme["kind"] == cand.meta["cv_scheme"]["kind"]
    assert len(rows) == len(folds) == 598 and 3 not in rows and 7 not in rows
    assert np.array_equal(cand.meta["train_rows"], rows) and scheme["n_splits"] == 5


def _panel_task(tmp_path, n_entities=600, periods=5, seed=0):
    """Сущности с уникальным атрибутом наблюдаются по периодам; таргет = свой уровень сущности + шум. test — новый период."""
    rng = np.random.default_rng(seed)
    area = rng.uniform(1000, 90000, n_entities).round(1)
    kind = rng.choice(["office", "school", "home"], n_entities)
    level = rng.normal(80, 25, n_entities)
    rows = []
    for p in range(1, periods + 2):
        seen = rng.random(n_entities) < 0.8
        for e in np.flatnonzero(seen):
            rows.append({"year": p, "area": area[e], "kind": kind[e], "temp": rng.normal(10 + p, 1),
                         "y": level[e] + 2 * p + rng.normal(scale=3)})
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    train, test = df[df.year <= periods], df[df.year == periods + 1].drop(columns="y")
    write_task(tmp_path, train, test, pd.DataFrame({"id": test.id, "y": 0.0}))
    return Spec(tmp_path, "id", "y", "regression", "rmse")


def test_entity_detected_and_improves_cv(tmp_path):
    spec = _panel_task(tmp_path)
    data = F.load(spec, _cfg())
    assert data.cv_scheme["kind"] == "period" and data.entity is not None
    assert data.entity["identity"] == ["area"] and set(data.entity["columns"]) == {"area", "kind"}
    cand = F.run(spec, config=_cfg())
    runs = {r["name"]: r["cv"] for r in cand.meta["runs"]}
    assert runs["lightgbm_entity"] < 0.7 * min(runs["lightgbm"], runs.get("lightgbm_noperiod", np.inf))
    assert cand.meta["entity_key"]["identity"] == ["area"] and submission.check_file(spec) == []


def test_entity_features_use_only_past_and_never_own_row(tmp_path):
    spec = _panel_task(tmp_path, n_entities=200)
    data = F.load(spec, _cfg())
    f = 0
    train_idx, valid_idx = np.flatnonzero(data.folds != f), np.flatnonzero(data.folds == f)
    e_tr, e_va, e_te = F.entity_features(data, train_idx, valid_idx)
    key, order, y = data.entity["key_train"], data.period_order, data.y
    for pos in range(0, len(train_idx), 7):  # каждая проверяемая строка: статистика только по более ранним периодам
        r = train_idx[pos]
        src = train_idx[(order[train_idx] < order[r]) & (key[train_idx] == key[r])]
        assert e_tr[pos, 1] == len(src) and r not in src
        if len(src):
            assert np.isclose(e_tr[pos, 0], y[src].mean())
    assert np.all(e_va[:, 1] <= (order[train_idx] < order[valid_idx][0]).sum())
    assert e_te.shape == (len(data.entity["key_test"]), 3)


def test_no_entity_without_repeating_identities(tmp_path):
    spec, _ = regression_task(tmp_path)  # непрерывные признаки без повторов — сущностей нет
    assert F.load(spec, _cfg()).entity is None
    spec, _ = classification_task(tmp_path / "c", n_classes=3)  # мультикласс — не поддерживается
    assert F.load(spec, _cfg()).entity is None


def test_contract_used_by_team_solution_and_agent_eval(tmp_path):
    """Вызовы как в solution.py и agent_eval.LocalEvaluator (P4) из отправленного архива — контракт не ломать."""
    import time
    import types
    spec, _ = classification_task(tmp_path, n_classes=3)
    best = F.run(spec, config=_cfg())
    assert {"folds", "train_rows", "fold_scores", "classes", "decision"} <= set(best.meta)
    assert F.clearly_better(spec.metric, best, best) is False and np.isfinite(F.effective_cv(best))
    # agent_eval._trainer: _params(name, SimpleNamespace(y, task, metric), FallbackConfig(seed, early_stopping), ...)
    shape = types.SimpleNamespace(y=np.empty(900), task="classification", metric=spec.metric)
    cfg = F.FallbackConfig(seed=42, early_stopping=100)
    data = F.load(spec, cfg)
    x, x_test, cats = data.frame("lightgbm")
    for name in ("lightgbm", "xgboost", "catboost"):
        params = F._params(name, shape, cfg, 2, {})
        frame, frame_test, model_cats = data.frame(name)
        tr, va = data.folds != 0, data.folds == 0
        p_va, p_te, iters, present = F._fit_predict(name, False, params, frame[tr], data.y[tr], frame[va], data.y[va],
                                                    frame_test, model_cats, time.monotonic() + 60, 100)
        assert p_va.shape == (va.sum(), len(present)) and p_te.shape == (len(frame_test), len(present))
    reg_spec, _ = regression_task(tmp_path / "r")
    reg = F.run(reg_spec, config=_cfg())
    shape = types.SimpleNamespace(y=np.empty(600), task="regression", metric="rmse")
    assert F._params("lightgbm", shape, cfg, 2, {})["objective"] == "regression"
    assert F.clearly_better("rmse", reg, reg) is False
