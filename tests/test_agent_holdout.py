"""Защита от переобучения агента под CV: скрытый фолд, растущий порог, доля выигранных фолдов."""

import types

import numpy as np
import pytest

import agent
import agent_eval
import fallback
from mocks import GOOD_CODE, MockLLM, SimpleDeadline, fenced, make_regression_task, plan
from test_agent_loop import FAST, experiments


def data(kind="random", n=5, order=None):
    folds = np.repeat(np.arange(n), 4)
    return types.SimpleNamespace(n_splits=n, folds=folds, cv_scheme={"kind": kind},
                                 period_order=None if order is None else np.asarray(order, dtype=float))


def test_hidden_fold_is_latest_period_or_last_fold():
    years = np.repeat([3, 5, 1, 4, 2], 4)  # фолд 1 — самый поздний год, ближе всего к test
    assert agent_eval.holdout_fold(data("period", order=years)) == 1
    assert agent_eval.holdout_fold(data("random")) == 4
    assert agent_eval.holdout_fold(data("random", n=3)) is None  # без одного из трёх фолдов сравнивать не на чем


def cand(scores, name="c"):
    return agent.Candidate(name, float(np.mean(scores)), float(np.std(scores)), np.zeros(1), np.zeros(1),
                           {"fold_scores": list(scores)})


def test_decision_needs_most_folds_to_win():
    base = cand([10.0, 10.0, 10.0, 10.0, 10.0])
    lucky = cand([8.0, 8.5, 10.1, 10.1, 10.1])  # RMSE: в среднем заметно лучше, но только на двух фолдах из пяти
    cfg = agent.AgentConfig(min_fold_wins=0.6)
    assert agent.decide(lucky, base, False, agent.AgentConfig()).accept
    decision = agent.decide(lucky, base, False, cfg)
    assert not decision.accept and "better on 2/5 folds (too few)" in decision.reason
    assert agent.decide(cand([9.5, 9.6, 9.4, 9.7, 10.1]), base, False, cfg).accept


def test_view_without_hidden_fold():
    view = agent._without_fold(cand([1.0, 2.0, 3.0, 4.0, 100.0]), 4)
    assert view.cv_score == pytest.approx(2.5) and view.meta["fold_scores"] == [1.0, 2.0, 3.0, 4.0]
    same = cand([1.0, 2.0])
    assert agent._without_fold(same, 1) is same and agent._without_fold(same, None) is same


def research_stub(**cfg):
    loop = object.__new__(agent._Research)
    loop.cfg, loop.recipe, loop.greater, loop.hidden = agent.AgentConfig(**cfg), True, False, 4
    loop.data = types.SimpleNamespace(spec_dict={"metric": "rmse"})
    loop.full_evals = 0
    return loop


def test_threshold_grows_with_number_of_evaluations():
    loop = research_stub()
    assert loop._accept_cfg().accept_k == 1.0 and loop._accept_cfg().min_fold_wins == 0.6
    loop.full_evals = 25
    assert loop._accept_cfg().accept_k == pytest.approx(1.5)
    loop.full_evals = 1000
    assert loop._accept_cfg().accept_k == 2.0


def test_hidden_fold_check_keeps_only_steps_that_generalize(caplog):
    loop = research_stub()
    loop.best = cand([10.0] * 5, "reference")
    good, overfit = cand([9.0, 9.0, 9.0, 9.0, 9.5], "good"), cand([8.0, 8.0, 8.0, 8.0, 10.5], "overfit")
    loop.accepted, loop.current = [good, overfit], overfit
    with caplog.at_level("INFO"):
        loop._holdout_check()
    assert loop.accepted == [good] and loop.current is good
    assert "overfit rmse=10.5 vs reference 10 (-0.5) → dropped" in caplog.text
    loop.accepted, loop.current = [overfit], overfit
    loop._holdout_check()
    assert loop.accepted == [] and loop.current is loop.best  # результата агента нет — остаётся fallback


def test_real_improvement_survives_hidden_fold(tmp_path, caplog):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    best = fallback.run(spec, None, fallback.FallbackConfig(models=("lightgbm",), try_log_target=False, bag_seeds=1))
    evaluate = agent_eval.RecipeEvaluator(best, threads=2)
    assert evaluate.holdout_fold == 4
    llm = MockLLM({"planner": [plan("Interaction x1*x2")], "coder": [fenced(GOOD_CODE)]})
    with caplog.at_level("INFO"):
        result = agent.research(spec, best, llm, SimpleDeadline(1200), evaluate=evaluate,
                                config=agent.AgentConfig(**{**FAST, "max_iters": 1}))
    assert result and experiments(spec)[0]["status"] == "accepted"
    assert "hidden fold 4: agent_01_features" in caplog.text and "→ kept" in caplog.text
