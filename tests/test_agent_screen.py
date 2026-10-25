"""Скрининг гипотез (agent_eval.ScreenEvaluator): быстрая оценка одним LightGBM рецепта перед полной оценкой."""

import numpy as np
import pytest

import agent
import agent_eval
import fallback
from mocks import GOOD_CODE, MockLLM, SimpleDeadline, feature_code, fenced, identity_features, make_regression_task, plan
from test_agent_loop import FAST, experiments

ONE = fallback.FallbackConfig(models=("lightgbm",), try_log_target=False, bag_seeds=1)
NOISE_CODE = feature_code('df["noise"] = np.random.default_rng(len(df)).normal(size=len(df))')


@pytest.fixture
def task(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    return spec, fallback.run(spec, None, ONE)


def with_x1_x2(train, test, spec):
    return tuple(frame.assign(x1_x2=frame["x1"] * frame["x2"]) for frame in (train, test))


def test_fixed_rounds_without_early_stopping(task):
    spec, best = task
    ctx = best.meta["context"]
    run = fallback.cv_model(ctx.data, "fixed", "lightgbm", ctx.config, 1e18, extra={"n_estimators": 37}, threads=1,
                            early_stopping=0)
    assert run.iterations == [37] * ctx.data.n_splits and np.isfinite(run.fold_scores).all()


def test_screen_is_deterministic_and_sees_useful_columns(task):
    spec, best = task
    screen = agent_eval.ScreenEvaluator(agent_eval.RecipeEvaluator(best, threads=2), seeds=2)
    assert screen.run.name in best.meta["context"].members and screen.rounds >= 50
    first, again = screen(spec, identity_features), screen(spec, identity_features)
    assert first.meta["fold_scores"] == again.meta["fold_scores"]  # общие сиды: эталон воспроизводится точно
    better = screen(spec, with_x1_x2)
    assert better.meta["new_columns"] == ["x1_x2"] and better.cv_score < 0.8 * first.cv_score  # RMSE ↓
    assert agent.decide(better, first, False, agent.AgentConfig(accept_k=3.0)).accept


def _research(spec, best, coder, **cfg):
    evaluate = agent_eval.RecipeEvaluator(best, threads=2)
    llm = MockLLM({"planner": [plan("Hypothesis")], "coder": [fenced(coder)] * 3})
    result = agent.research(spec, best, llm, SimpleDeadline(1200), evaluate=evaluate,
                            config=agent.AgentConfig(**{**FAST, "max_iters": 1, **cfg}))
    return result, evaluate, llm


def test_screen_rejects_without_full_evaluation_or_repair(task):
    spec, best = task
    result, evaluate, llm = _research(spec, best, NOISE_CODE, screen_k=3.0)
    row = experiments(spec)[0]
    assert row["status"] == "rejected" and row["error"].startswith(agent.SCREEN_REJECT) and row["gain"] is not None
    assert evaluate.calls == 1  # только эталон рецепта: отсеянная гипотеза полным рецептом не оценивалась
    assert [role for role, _, _ in llm.calls].count("coder") == 1  # отказ скрининга — не сбой: кода не чинят
    assert result == []


def test_screen_passes_useful_hypothesis_to_full_evaluation(task):
    spec, best = task
    result, evaluate, _ = _research(spec, best, GOOD_CODE, screen_k=3.0)
    row = experiments(spec)[0]
    assert row["status"] == "accepted" and "screen" in row["error"] and "passed" in row["error"]
    assert evaluate.calls == 2 and result and result[0].meta["screen_gain"] > 0


def test_screening_can_be_off(task):
    spec, best = task
    _, evaluate, _ = _research(spec, best, NOISE_CODE, screen=False)
    assert evaluate.calls == 2 and not experiments(spec)[0]["error"].startswith(agent.SCREEN_REJECT)
