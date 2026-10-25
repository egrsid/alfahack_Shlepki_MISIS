"""Агент «поверх fallback»: точка отсчёта — лучший кандидат fallback, код LLM только добавляет колонки."""

import numpy as np
import pytest

import agent
import agent_eval
import fallback
import prompts_agent as P
from mocks import GOOD_CODE, MockLLM, SimpleDeadline, fenced, identity_features, make_regression_task, plan
from test_agent_loop import FAST, experiments

# fallback из одного LightGBM без лог-таргета: признак x1*x2 он сам не построит — новая колонка честно помогает
# bag_seeds=1: рецепт переобучает один сид — сравниваем с fallback того же сида
ONE = fallback.FallbackConfig(models=("lightgbm",), try_log_target=False, bag_seeds=1)


@pytest.fixture
def task(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    return spec, fallback.run(spec, None, ONE)


def with_columns(**columns):
    def build_features(train, test, spec):
        out = []
        for frame in (train, test):
            frame = frame.copy()
            for name, fn in columns.items():
                frame[name] = fn(frame)
            out.append(frame)
        return tuple(out)
    return build_features


def test_reference_reproduces_fallback_best(task):
    spec, best = task
    cand = agent_eval.RecipeEvaluator(best, threads=2)(spec, identity_features)
    assert cand.cv_score == pytest.approx(fallback.effective_cv(best), rel=1e-3)
    assert cand.meta["new_columns"] == [] and cand.meta["evaluator"] == "recipe"
    assert cand.oof.shape == best.oof.shape and len(cand.meta["fold_scores"]) == len(best.meta["fold_scores"])


def test_new_columns_improve_and_changed_originals_are_ignored(task):
    spec, best = task
    evaluate = agent_eval.RecipeEvaluator(best, threads=2)
    reference = evaluate(spec, identity_features)
    better = evaluate(spec, with_columns(x1_x2=lambda f: f["x1"] * f["x2"]))
    assert better.meta["new_columns"] == ["x1_x2"] and better.cv_score < 0.8 * reference.cv_score  # RMSE ↓
    assert fallback.clearly_better(spec.metric, better, reference)
    changed = evaluate(spec, with_columns(x1=lambda f: f["x1"] * f["x2"]))  # перезапись исходной колонки
    assert changed.meta["new_columns"] == [] and changed.cv_score == pytest.approx(reference.cv_score, rel=1e-6)


def test_no_context_is_rejected():
    with pytest.raises(ValueError):
        agent_eval.RecipeEvaluator(fallback.Candidate("x", 0.0, 0.0, np.zeros(1), np.zeros(1), {}))


def test_agent_starts_from_fallback_and_adds_columns(task):
    spec, best = task
    llm = MockLLM({"planner": [plan("Interaction x1*x2"), plan("Tune lightgbm", kind="params")],
                   "coder": [fenced(GOOD_CODE)]})
    result = agent.research(spec, best, llm, SimpleDeadline(600), config=agent.AgentConfig(**FAST),
                            evaluate=agent_eval.RecipeEvaluator(best, threads=2))
    rows = experiments(spec)
    assert rows[0]["status"] == "accepted" and rows[0]["kind"] == "features"
    assert len(result) == 1 and result[0].meta["new_columns"] == ["x1_x2"]
    assert fallback.clearly_better(spec.metric, result[0], best)  # лучше самого fallback, а не своей базы
    planner = llm.calls[0][1]
    assert P.RECIPE_RULE in planner[0]["content"] and "- params:" not in planner[0]["content"]
    coder = [m for role, m, _ in llm.calls if role == "coder"][0]
    assert "PARAMS = {" not in coder[0]["content"] and "only new column names" in coder[0]["content"]
    assert all(r["kind"] in P.RECIPE_KINDS for r in rows)  # гипотеза params отброшена


def test_bagged_and_randomforest_members_are_resolved(tmp_path):
    spec = make_regression_task(tmp_path / "reg", n_train=500)
    best = fallback.run(spec, None, fallback.FallbackConfig(models=("lightgbm", "randomforest"), try_log_target=False))
    ctx = best.meta["context"]
    assert ctx.members and set(ctx.members) <= {r.name for r in ctx.runs}  # в т.ч. fallback_<имя>_bag3
    evaluate = agent_eval.RecipeEvaluator(best, threads=2)
    cand = evaluate(spec, with_columns(x1_x2=lambda f: f["x1"] * f["x2"], city2=lambda f: f["city"] + "_"))
    assert np.isfinite(cand.cv_score) and cand.meta["new_columns"] == ["x1_x2", "city2"]

    forest = fallback.run(spec, None, fallback.FallbackConfig(models=("randomforest",), try_log_target=False,
                                                              bag_seeds=1, entity=False))
    assert list(forest.meta["context"].members) == ["randomforest"]
    cand = agent_eval.RecipeEvaluator(forest, threads=2)(spec, with_columns(
        ratio=lambda f: f["x1"] / f["x3"], city2=lambda f: f["city"] + "_"))  # NaN и категория → числа для sklearn
    assert np.isfinite(cand.cv_score)
