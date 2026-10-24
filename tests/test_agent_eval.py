import shutil

import numpy as np
import pytest

import agent
import agent_eval
from mocks import (GOOD_CODE, MockLLM, SimpleDeadline, fenced, identity_features, make_multiclass_task,
                   make_regression_task, plan)
from test_agent_loop import FAST, experiments


@pytest.fixture
def reg_task(tmp_path):
    return make_regression_task(tmp_path / "reg")


@pytest.mark.parametrize("raw, key", [("Macro-F1", "macro_f1"), ("f1_macro", "macro_f1"), ("F1 weighted", "weighted_f1"),
                                      ("RMSE", "rmse"), ("log_loss", "logloss"), ("AUC", "roc_auc"),
                                      ("mean_absolute_error", "mae")])
def test_metric_name_normalizes_aliases(raw, key):
    assert agent_eval.metric_name(raw) == key


def test_score_regression_and_classification():
    y = np.array([1.0, 2.0, 3.0])
    assert agent_eval.score("rmse", y, y + 1.0, regression=True) == pytest.approx(1.0)
    assert agent_eval.score("MAE", y, y - 2.0, regression=True) == pytest.approx(2.0)
    assert agent_eval.score("rmsle", y, y, regression=True) == pytest.approx(0.0)
    labels = np.array([0, 1, 2, 2])
    proba = np.eye(3)[[0, 1, 2, 1]]
    assert agent_eval.score("accuracy", labels, proba, regression=False) == pytest.approx(0.75)
    assert 0 < agent_eval.score("macro_f1", labels, proba, regression=False) < 1
    with pytest.raises(ValueError):
        agent_eval.score("smape", y, y, regression=True)


def test_local_evaluator_finds_nested_files_and_reports_classes(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls")
    nested = spec.task_dir / "multiclass" / "participants"
    nested.mkdir(parents=True)
    for name in ("train.csv", "test.csv"):
        shutil.move(spec.task_dir / name, nested / name)
    cand = agent_eval.LocalEvaluator(n_splits=3, n_estimators=50, threads=2)(spec, identity_features, {}, None)
    assert cand.meta["classes"] == ["high", "low", "mid"] and cand.oof.shape == (450, 3)
    assert len(cand.meta["fold_scores"]) == 3 and cand.meta["evaluator"] == "local"


class ShrinkingDeadline:
    """Первая проверка — много времени, дальше — почти ничего: оставшиеся фолды не успеть."""

    def __init__(self):
        self.calls = 0

    def left(self):
        self.calls += 1
        return 1000.0 if self.calls == 1 else 1e-6


def test_local_evaluator_stops_by_deadline(reg_task):
    evaluator = agent_eval.LocalEvaluator(n_splits=3, n_estimators=50, threads=2)
    with pytest.raises(TimeoutError):
        evaluator(reg_task, identity_features, {}, SimpleDeadline(0))
    with pytest.raises(TimeoutError, match="after 1/3 folds"):
        evaluator(reg_task, identity_features, {}, ShrinkingDeadline())


def test_research_without_evaluator_uses_local_evaluator(reg_task):
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    result = agent.research(reg_task, None, llm, SimpleDeadline(600), config=agent.AgentConfig(**FAST))
    assert len(result) == 1 and result[0].meta["evaluator"] == "local"
    assert experiments(reg_task)[0]["status"] == "accepted"
