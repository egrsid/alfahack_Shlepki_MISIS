import pytest

import agent_eval
import tuning
from mocks import SimpleDeadline, identity_features, make_multiclass_task, make_regression_task


@pytest.fixture
def reg_task(tmp_path):
    return make_regression_task(tmp_path / "reg", n_train=600)


class CountingEvaluator(agent_eval.LocalEvaluator):
    def __init__(self, **kw):
        super().__init__(n_splits=4, n_estimators=40, threads=2, **kw)
        self.folds_used = []

    def __call__(self, spec, build_features, params, deadline=None):
        cand = super().__call__(spec, build_features, params, deadline)
        self.folds_used.append(cand.meta["n_splits"])
        return cand


def test_tune_uses_agent_evaluator_on_few_folds_and_returns_known_params(reg_task):
    evaluator = CountingEvaluator()
    cfg = {"model": "lightgbm", "params": {"class_weight": None}, "n_trials": 4, "seed": 1, "evaluate": evaluator}
    found = tuning.tune(reg_task, identity_features, cfg, SimpleDeadline(120))
    assert isinstance(found, dict) and found.get("class_weight", "x") is None  # стартовые ключи сохраняются
    assert evaluator.calls == 0  # оценивала копия с max_folds, оригинал агента не тронут
    import agent
    _, normalized, _, notes = agent.normalize_model_config({"MODEL": "lightgbm", "PARAMS": found}, "regression")
    assert normalized == found and not notes  # всё из белого списка агента


def test_tune_is_deterministic_by_seed(reg_task):
    cfg = {"model": "xgboost", "params": {}, "n_trials": 3, "seed": 7,
           "evaluate": agent_eval.LocalEvaluator(n_splits=3, n_estimators=30, threads=2)}
    assert tuning.tune(reg_task, identity_features, cfg, None) == tuning.tune(reg_task, identity_features, cfg, None)


def test_tune_without_time_returns_start_params(reg_task):
    start = {"num_leaves": 15}
    cfg = {"model": "lightgbm", "params": start, "n_trials": 50, "seed": 0,
           "evaluate": agent_eval.LocalEvaluator(n_splits=3, n_estimators=30, threads=2)}
    assert tuning.tune(reg_task, identity_features, cfg, SimpleDeadline(0.0)) == start


def test_tune_failing_evaluator_returns_start_params(reg_task):
    def broken(*args, **kw):
        raise RuntimeError("boom")
    cfg = {"model": "lightgbm", "params": {"num_leaves": 7}, "n_trials": 3, "seed": 0, "evaluate": broken}
    assert tuning.tune(reg_task, identity_features, cfg, SimpleDeadline(60)) == {"num_leaves": 7}


def test_tune_classification_and_catboost(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls")
    cfg = {"model": "catboost", "params": {}, "n_trials": 2, "seed": 0,
           "evaluate": agent_eval.LocalEvaluator(n_splits=3, n_estimators=30, threads=2)}
    found = tuning.tune(spec, identity_features, cfg, SimpleDeadline(120))
    assert isinstance(found, dict) and set(found) <= {"depth", "learning_rate", "l2_leaf_reg", "random_strength",
                                                      "bagging_temperature"}
