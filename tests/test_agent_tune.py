import sys
import types

import pytest

import agent
import prompts_agent as P
from mocks import (GOOD_CODE, MockLLM, SimpleDeadline, StubEvaluator, StubTuner, fenced, make_best,
                   make_regression_task, plan, useless_code)
from test_agent_loop import FAST, coder_prompts, experiments, planner_prompts

TUNED = {"n_estimators": 400, "learning_rate": 0.1}  # у StubEvaluator в тестах 60 деревьев — недообучение


@pytest.fixture
def reg_task(tmp_path):
    return make_regression_task(tmp_path / "reg")


def run(spec, llm, tuner, **cfg):
    evaluator = StubEvaluator(n_splits=3, n_estimators=60)
    return agent.research(spec, make_best(spec, evaluator), llm, SimpleDeadline(600),
                          config=agent.AgentConfig(**{**FAST, "tune_min_seconds": 1, **cfg}), evaluate=evaluator,
                          tuner=tuner)


def test_set_constant_replaces_or_appends():
    code = 'import x\n\n\ndef build_features(a, b, c):\n    return a, b\n\n\nPARAMS = {\n    "a": 1,\n}\nMODEL = "lightgbm"\n'
    replaced = agent.set_constant(code, "PARAMS", {"num_leaves": 63})
    assert "PARAMS = {'num_leaves': 63}\nMODEL = \"lightgbm\"\n" in replaced and '"a": 1' not in replaced
    assert agent.extract_constants(replaced)[0]["PARAMS"] == {"num_leaves": 63}
    appended = agent.set_constant("def build_features(a, b, c):\n    return a, b\n", "PARAMS", {})
    assert appended.endswith("\nPARAMS = {}\n") and agent.validate_code(appended) == []


def test_tune_hypothesis_is_accepted_without_coder(reg_task):
    tuner = StubTuner(TUNED)
    llm = MockLLM({"planner": [plan("Tune lightgbm hyperparameters", kind="tune")]})
    result = run(reg_task, llm, tuner)
    assert len(result) == 1 and llm.count("coder") == 0
    assert result[0].meta["params"] == TUNED and result[0].meta["kind"] == "tune"
    assert "PARAMS = {'n_estimators': 400, 'learning_rate': 0.1}" in result[0].meta["code"]
    cfg, left = tuner.calls[0]
    assert cfg["model"] == "lightgbm" and cfg["params"] == {} and cfg["n_trials"] == 30 and 0 < left <= 180
    assert "- tune: automatic hyperparameter search" in llm.calls[0][1][0]["content"]


def test_next_experiment_builds_on_tuned_params(reg_task):
    llm = MockLLM({"planner": [plan("Tune lightgbm hyperparameters", kind="tune"),
                               plan("Interaction x1*x2")],
                   "coder": [fenced(GOOD_CODE)]})
    run(reg_task, llm, StubTuner(TUNED))
    assert "PARAMS = {'n_estimators': 400, 'learning_rate': 0.1}" in coder_prompts(llm)[0]


def test_tuner_params_are_normalized(reg_task):
    tuner = StubTuner({"n_estimators": 400, "learning_rate": 5.0, "is_unbalance": True, "boosting": "dart"})
    llm = MockLLM({"planner": [plan("Tune lightgbm hyperparameters", kind="tune")]})
    run(reg_task, llm, tuner)
    row = experiments(reg_task)[0]
    assert "'learning_rate': 0.5" in row["diff"] and "is_unbalance" not in row["diff"]


def test_tuner_without_gain_is_rejected_and_failure_is_contained(reg_task):
    llm = MockLLM({"planner": [plan("Tune lightgbm hyperparameters", "Optuna search for lightgbm", kind="tune")]})
    result = run(reg_task, llm, StubTuner({}))  # подбор ничего не нашёл → код не изменился
    assert result == [] and experiments(reg_task)[0]["error"] == "no change vs the current best code"

    llm = MockLLM({"planner": [plan("Tune lightgbm hyperparameters", kind="tune"), plan("Interaction x1*x2")],
                   "coder": [fenced(GOOD_CODE)]})
    result = run(reg_task, llm, StubTuner(RuntimeError("optuna exploded")))
    rows = experiments(reg_task)
    assert rows[0]["status"] == "failed" and "optuna exploded" in rows[0]["error"]
    assert len(result) == 1 and rows[1]["status"] == "accepted"  # цикл пошёл дальше


def test_no_tuner_tune_becomes_params_and_is_not_offered(reg_task):
    llm = MockLLM({"planner": [plan("Tune lightgbm hyperparameters", kind="tune")], "coder": [fenced(GOOD_CODE)]})
    run(reg_task, llm, None)
    assert "- tune:" not in llm.calls[0][1][0]["content"]
    assert experiments(reg_task)[0]["kind"] == "params" and llm.count("coder") == 1


def test_stub_tuning_module_is_ignored(reg_task, monkeypatch):
    stub = types.ModuleType("tuning")
    stub.STUB = True
    stub.tune = StubTuner(TUNED)
    monkeypatch.setitem(sys.modules, "tuning", stub)
    assert agent._optional("tuning", "tune") is None
    real = types.ModuleType("tuning")
    real.tune = StubTuner(TUNED)
    monkeypatch.setitem(sys.modules, "tuning", real)
    assert agent._optional("tuning", "tune") is real.tune


def test_auto_tune_before_stopping_on_plateau(reg_task):
    tuner = StubTuner(TUNED)
    llm = MockLLM({"planner": [plan("u one", "u two", "u three")],
                   "coder": [fenced(useless_code(f"u{i}")) for i in range(3)]})
    result = run(reg_task, llm, tuner, max_no_improve=2)
    rows = experiments(reg_task)
    # плато после двух неудач → подбор; принятый подбор сбрасывает плато, и оставшаяся гипотеза тоже проверяется
    assert [r["kind"] for r in rows] == ["features", "features", "tune", "features"] and len(tuner.calls) == 1
    assert rows[2]["status"] == "accepted" and len(result) == 1
    assert "(auto, before stopping)" in rows[2]["hypothesis"]


def test_auto_tune_when_planner_runs_dry_and_only_once(reg_task):
    tuner = StubTuner(TUNED)
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    run(reg_task, llm, tuner)
    assert [r["kind"] for r in experiments(reg_task)] == ["features", "tune"] and len(tuner.calls) == 1
    assert "Kinds not tried yet:" in planner_prompts(llm)[-1] and "tune" not in planner_prompts(llm)[-1].split(
        "Kinds not tried yet:")[1].split("\n")[0]


def test_auto_tune_can_be_disabled(reg_task):
    tuner = StubTuner(TUNED)
    llm = MockLLM({"planner": [plan("u one", "u two")], "coder": [fenced(useless_code(f"u{i}")) for i in range(2)]})
    run(reg_task, llm, tuner, max_no_improve=2, auto_tune=False)
    assert tuner.calls == []


def test_grid_tuner_mock_picks_best(reg_task):
    evaluator = StubEvaluator(n_splits=3, n_estimators=60)
    tuner = StubTuner(evaluator=evaluator, grid=[{"n_estimators": 20}, {"n_estimators": 300}])
    found = tuner(reg_task, lambda tr, te, s: (tr, te), {"model": "lightgbm", "params": {}}, SimpleDeadline(60))
    assert found == {"n_estimators": 300}
