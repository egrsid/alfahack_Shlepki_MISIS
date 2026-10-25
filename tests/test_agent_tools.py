"""Действия агента над рецептом fallback: drop / weights / target_stats, «быстрый рецепт», скрининг больших задач."""

import types

import numpy as np
import pandas as pd
import pytest

import agent
import agent_eval
import fallback
from mocks import MockLLM, SimpleDeadline, identity_features, make_regression_task
from test_agent_loop import FAST, experiments

ONE = fallback.FallbackConfig(models=("lightgbm",), try_log_target=False, bag_seeds=1)


@pytest.fixture
def task(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    return spec, fallback.run(spec, None, ONE)


def test_row_weights_schemes():
    prep = types.SimpleNamespace(X=pd.DataFrame({"a": [1, 1, 1, 2, 3, 4]}))
    data = types.SimpleNamespace(prep=prep, period_order=None, task="multiclass", classes=["a", "b"],
                                 y=np.array([0, 0, 0, 0, 0, 1]))
    dedup = agent_eval.row_weights(data, {"scheme": "dedup"})
    assert dedup.mean() == pytest.approx(1.0) and dedup[0] == pytest.approx(dedup[3] / 3)  # строка из тройки — 1/3
    balanced = agent_eval.row_weights(data, {"scheme": "class", "power": 1.0})
    assert balanced[5] == pytest.approx(5 * balanced[0])  # редкий класс 1:5
    with pytest.raises(ValueError, match="period"):
        agent_eval.row_weights(data, {"scheme": "recency"})
    data.period_order = np.array([1, 1, 2, 2, 3, 3])
    recency = agent_eval.row_weights(data, {"scheme": "recency", "power": 1.0})
    assert recency[4] == pytest.approx(3 * recency[0])


def test_drop_maps_raw_columns_to_fallback_matrix():
    columns = ["Year_Factor", "facility_type", "facility_type__freq", "x_1"]
    renamed = {"x_1": "x 1"}
    assert agent_eval.drop_columns(columns, ["facility_type", "x 1"], renamed) == (
        "facility_type", "facility_type__freq", "x_1")
    assert agent_eval.drop_columns(columns, [], renamed) == ()


def test_target_key_features_never_see_own_row_or_validation_fold():
    folds = np.array([0, 0, 1, 1, 2, 2])
    data = types.SimpleNamespace(y=np.array([1.0, 2.0, 10.0, 20.0, 100.0, 200.0]), folds=folds, period_order=None,
                                 task="regression", classes=None)
    key = np.zeros(6)  # один ключ на все строки
    train_idx, valid_idx = np.array([0, 1, 2, 3]), np.array([4, 5])
    tr, va, te = fallback.target_key_features(data, key, np.zeros(2), train_idx, valid_idx)
    assert tr[0, 0] == pytest.approx(15.0)  # строка фолда 0 видит только фолд 1 (внутри обучения)
    assert tr[2, 0] == pytest.approx(1.5)
    assert va[0, 0] == pytest.approx(8.25) and te[0, 0] == pytest.approx(8.25)  # валидация и test — весь train
    assert not np.isin([100.0, 200.0, 150.0], [*tr[:, 0], *va[:, 0]]).any()  # таргет валидации не участвует


def test_action_state_validates_and_accumulates():
    data = types.SimpleNamespace(prep=types.SimpleNamespace(X=pd.DataFrame({"a": [1, 2]})), period_order=None,
                                 task="regression", classes=None, y=np.array([1.0, 2.0]))
    cols = ["a", "b", "c"]
    state, error = agent.action_state("drop", {"drop": ["a"]}, {}, cols, data)
    assert error == "" and state == {"drop": ["a"]}
    state, error = agent.action_state("drop", {"drop": ["b"]}, state, cols, data)
    assert state == {"drop": ["a", "b"]}
    assert agent.action_state("drop", {"drop": ["zzz"]}, {}, cols, data)[1].startswith("drop: unknown columns")
    assert "period" in agent.action_state("weights", {"scheme": "recency"}, {}, cols, data)[1]
    state, error = agent.action_state("target_stats", {"keys": ["a", "b"]}, {}, cols, data)
    assert state == {"target_keys": [["a", "b"]]}
    assert "already used" in agent.action_state("target_stats", {"keys": [["b", "a"]]}, state, cols, data)[1]


def test_recipe_evaluates_actions(task):
    spec, best = task
    evaluate = agent_eval.RecipeEvaluator(best, threads=2)
    reference = evaluate(spec, identity_features)
    dropped = evaluate(spec, identity_features, {"recipe_state": {"drop": ["x1"]}})
    assert dropped.meta["recipe_state"] == {"drop": ["x1"]} and dropped.cv_score > 1.2 * reference.cv_score  # x1 важен
    keyed = evaluate(spec, identity_features, {"recipe_state": {"target_keys": [["city"]]}})
    assert np.isfinite(keyed.cv_score) and keyed.cv_score != reference.cv_score


def test_slow_low_weight_member_is_frozen():
    runs = {name: types.SimpleNamespace(name=name, seconds=sec, oof=np.zeros(3), test_pred=np.zeros(2), cv=1.0)
            for name, sec in (("lightgbm", 60), ("xgboost", 70), ("randomforest", 40), ("catboost", 240))}
    ctx = types.SimpleNamespace(runs=list(runs.values()), members={"lightgbm": 0.4, "xgboost": 0.2,
                                                                   "randomforest": 0.23, "catboost": 0.17},
                                fold_members=None, data=types.SimpleNamespace(n_splits=5, folds=np.zeros(3),
                                                                              cv_scheme={}, period_order=None))
    best = types.SimpleNamespace(meta={"context": ctx})
    assert agent_eval.RecipeEvaluator(best).frozen == {"catboost"}
    assert agent_eval.RecipeEvaluator(best, fast=False).frozen == set()


def test_agent_applies_tool_without_coder(task):
    spec, best = task
    action = {"title": "Drop x3", "kind": "drop", "rationale": "x3 is noise", "action": {"drop": ["x3"]}}
    bad = {"title": "Drop unknown", "kind": "drop", "rationale": "-", "action": {"drop": ["nope"]}}
    llm = MockLLM({"planner": [{"analysis": "-", "hypotheses": [action, bad]}]})
    agent.research(spec, best, llm, SimpleDeadline(1200), evaluate=agent_eval.RecipeEvaluator(best, threads=2),
                   config=agent.AgentConfig(**{**FAST, "max_iters": 2, "screen": False}))
    rows = experiments(spec)
    assert [r["kind"] for r in rows] == ["drop", "drop"] and llm.count("coder") == 0
    assert rows[0]["status"] in ("accepted", "rejected") and rows[0]["cv_score"] is not None
    assert rows[0]["diff"] == 'action drop: {"drop": ["x3"]}'
    assert rows[1]["status"] == "failed" and "unknown columns ['nope']" in rows[1]["error"]


def test_big_task_screen_subsamples_rows_and_uses_one_seed(task):
    spec, best = task
    recipe = agent_eval.RecipeEvaluator(best, threads=2)
    screen = agent_eval.ScreenEvaluator(recipe, max_train_rows=100, seed_seconds=0.0)
    assert screen.max_rows == 100 and screen.seeds == 1 and "≤ 100 train rows" in screen.name
    cand = screen(spec, identity_features, {"recipe_state": {"drop": ["x3"]}})
    assert np.isfinite(cand.cv_score)
    assert agent_eval.ScreenEvaluator(recipe).max_rows is None  # маленькая задача — все строки


def test_filter_notes_do_not_leak_into_the_next_experiment(task):
    spec, best = task
    ratio = 'df["x1_x2"] = df["x1"] * df["x2"]; df["x1_x2_copy"] = df["x1"] * df["x2"] * 2'
    from mocks import feature_code, fenced, plan
    llm = MockLLM({"planner": [plan("Product x1*x2")], "coder": [fenced(feature_code(ratio))]})
    evaluate = agent_eval.RecipeEvaluator(best, threads=2)
    action = {"title": "Drop x3", "kind": "drop", "rationale": "-", "action": {"drop": ["x3"]}}
    llm.script["planner"].append({"analysis": "-", "hypotheses": [action]})
    agent.research(spec, best, llm, SimpleDeadline(1200), evaluate=evaluate,
                   config=agent.AgentConfig(**{**FAST, "max_iters": 2, "screen": False}))
    rows = experiments(spec)
    assert "duplicates" in rows[0]["error"] and rows[1]["kind"] == "drop"
    assert "duplicates" not in rows[1]["error"]
