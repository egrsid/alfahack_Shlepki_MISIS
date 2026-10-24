import pytest

import agent
import prompts_agent as P
from mocks import (GOOD_CODE, MockLLM, SimpleDeadline, StubEvaluator, StubTuner, fenced, make_best,
                   make_regression_task, plan, useless_code)
from test_agent_loop import FAST, experiments, planner_prompts

SEARCH = dict(FAST, auto_model_search=True)


class BiasedEvaluator(StubEvaluator):
    """StubEvaluator, у которого выбранные модели искусственно лучше/хуже: bonus прибавляется к RMSE по фолдам."""

    def __init__(self, bonus: dict[str, float] | None = None, fail: tuple[str, ...] = ()) -> None:
        super().__init__(n_splits=3, n_estimators=60)
        self.bonus, self.fail, self.models = bonus or {}, fail, []

    def __call__(self, spec, build_features, params, deadline=None):
        model = (params or {}).get("model") or "lightgbm"
        self.models.append(model)
        if model in self.fail:
            raise RuntimeError(f"{model} exploded")
        cand = super().__call__(spec, build_features, params, deadline)
        shift = self.bonus.get(model, 0.0)
        cand.meta["fold_scores"] = [s + shift for s in cand.meta["fold_scores"]]
        cand.cv_score += shift
        return cand


@pytest.fixture
def reg_task(tmp_path):
    return make_regression_task(tmp_path / "reg")


def run(spec, llm, evaluator, tuner=None, **cfg):
    return agent.research(spec, make_best(spec, evaluator), llm, SimpleDeadline(600),
                          config=agent.AgentConfig(**{**SEARCH, **cfg}), evaluate=evaluator, tuner=tuner)


def test_translate_params_moves_shared_concepts():
    lgb = {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "class_weight": "balanced", "max_depth": -1}
    moved, dropped = agent.translate_params(lgb, "lightgbm", "catboost")
    assert moved == {"iterations": 500, "learning_rate": 0.03, "auto_class_weights": "Balanced"}
    assert sorted(dropped) == ["max_depth", "num_leaves"]
    moved, dropped = agent.translate_params(lgb, "lightgbm", "xgboost")
    assert moved == {"n_estimators": 500, "learning_rate": 0.03} and "class_weight" in dropped
    assert agent.translate_params({"depth": 6, "auto_class_weights": "SqrtBalanced"}, "catboost", "lightgbm")[0] == {
        "max_depth": 6, "class_weight": "balanced"}
    assert agent.translate_params({"num_leaves": 7}, "lightgbm", "lightgbm") == ({"num_leaves": 7}, [])


def test_model_search_accepts_better_model_without_coder(reg_task):
    evaluator = BiasedEvaluator(bonus={"xgboost": -1.0})
    llm = MockLLM({"planner": [plan("Compare boosting models", kind="model_search")]})
    result = run(reg_task, llm, evaluator)
    assert len(result) == 1 and llm.count("coder") == 0
    best = result[0]
    assert best.meta["model"] == "xgboost" and best.meta["kind"] == "model_search"
    assert "MODEL = 'xgboost'" in best.meta["code"] and agent.validate_code(best.meta["code"]) == []
    row = experiments(reg_task)[0]
    assert row["status"] == "accepted" and row["error"].startswith("models: lightgbm (current) cv=")
    assert "xgboost cv=" in row["error"] and "catboost cv=" in row["error"]
    assert "- model_search: compare lightgbm, catboost and xgboost" in llm.calls[0][1][0]["content"]


def test_model_search_rejected_when_current_model_is_best(reg_task):
    evaluator = BiasedEvaluator(bonus={"xgboost": 5.0, "catboost": 5.0})
    llm = MockLLM({"planner": [plan("Compare boosting models", kind="model_search")]})
    assert run(reg_task, llm, evaluator) == []
    row = experiments(reg_task)[0]
    assert row["status"] == "rejected" and ("MODEL = 'xgboost'" in row["diff"] or "MODEL = 'catboost'" in row["diff"])


def test_model_search_skips_expensive_and_survives_failures(reg_task, monkeypatch):
    monkeypatch.setitem(agent.MODEL_COST, "catboost", 1e6)
    evaluator = BiasedEvaluator(bonus={"xgboost": -1.0})
    llm = MockLLM({"planner": [plan("Compare boosting models", kind="model_search")]})
    run(reg_task, llm, evaluator)
    row = experiments(reg_task)[0]
    assert "catboost skipped" in row["error"] and "catboost" not in evaluator.models

    evaluator = BiasedEvaluator(fail=("xgboost", "catboost"))
    monkeypatch.setitem(agent.MODEL_COST, "catboost", 3.0)
    assert run(reg_task, MockLLM({"planner": [plan("Compare boosting models", kind="model_search")]}),
               evaluator) == []
    row = experiments(reg_task)[0]
    assert row["status"] == "failed" and "xgboost failed (RuntimeError: xgboost exploded)" in row["error"]


def test_auto_model_search_then_tune_before_stopping(reg_task):
    evaluator = BiasedEvaluator(bonus={"xgboost": 5.0, "catboost": 5.0})
    tuner = StubTuner({"n_estimators": 400, "learning_rate": 0.1})
    llm = MockLLM({"planner": [plan("u one", "u two", "u three")],
                   "coder": [fenced(useless_code(f"u{i}")) for i in range(3)]})
    run(reg_task, llm, evaluator, tuner=tuner, max_no_improve=2, tune_min_seconds=1)
    kinds = [r["kind"] for r in experiments(reg_task)]
    assert kinds == ["features", "features", "model_search", "tune", "features"] and len(tuner.calls) == 1
    assert "(auto, before stopping)" in experiments(reg_task)[2]["hypothesis"]


def test_model_search_disabled_becomes_model_kind(reg_task):
    llm = MockLLM({"planner": [plan("Compare boosting models", kind="model_search")], "coder": [fenced(GOOD_CODE)]})
    run(reg_task, llm, BiasedEvaluator(), model_search=False)
    assert experiments(reg_task)[0]["kind"] == "model" and llm.count("coder") == 1
    assert "- model_search:" not in llm.calls[0][1][0]["content"]


def test_model_search_runs_once_per_base_code(reg_task):
    evaluator = BiasedEvaluator(bonus={"xgboost": 5.0, "catboost": 5.0})
    llm = MockLLM({"planner": [plan("Compare boosting models", "Model selection among boosters", kind="model_search"),
                               plan("Try all gradient boosting libraries", kind="compare_models")]})
    run(reg_task, llm, evaluator)
    assert [r["kind"] for r in experiments(reg_task)] == ["model_search"]
    assert len(planner_prompts(llm)) >= 2  # второй план был, но его выбор модели — повтор от той же базы
