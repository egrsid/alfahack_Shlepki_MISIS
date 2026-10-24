import numpy as np
import pandas as pd
import pytest

import agent
import postprocess
from agent_eval import LABEL_METRICS, label_score, score
from mocks import MockLLM, SimpleDeadline, StubEvaluator, make_best, make_multiclass_task, make_regression_task
from test_agent_loop import FAST, experiments

POST = dict(FAST, postprocess=True)


class MockCand:
    """Кандидат для decide: только оценки по частям OOF."""

    def __init__(self, folds):
        self.cv_score, self.cv_std, self.meta = float(np.mean(folds)), float(np.std(folds)), {"fold_scores": folds}


def skewed_proba(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """3 класса, редкий класс 2 (≈8%): модель его «узнаёт», но систематически занижает вероятность."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.6, 0.32, 0.08])
    logits = rng.normal(size=(n, 3)) + 2.5 * np.eye(3)[y]
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    proba[:, 2] *= 0.15
    return y, proba / proba.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("metric", LABEL_METRICS)
@pytest.mark.parametrize("k", [2, 4])
def test_fast_label_score_matches_sklearn(metric, k):
    rng = np.random.default_rng(k)
    y = rng.integers(0, k, 500)
    pred = np.where(rng.random(500) < 0.6, y, rng.integers(0, k, 500))
    pred[pred == k - 1] = 0  # класс, который ни разу не предсказан
    assert postprocess.fast_label_score(metric, y, pred, k) == pytest.approx(label_score(metric, y, pred, k))


def test_class_weights_lift_underpredicted_class():
    y, proba = skewed_proba()
    found = postprocess.fit_class_weights(y, proba, "macro_f1")
    assert found is not None and found.classes == ["0", "1", "2"]
    assert found.weights[2] > 2 * found.weights[0]  # редкому классу вернули вес
    assert np.mean(found.folds_after) > np.mean(found.folds_before) + 0.03
    decision = agent.decide(MockCand(found.folds_after), MockCand(found.folds_before), True, agent.AgentConfig())
    assert decision.accept


def test_class_weights_on_noise_give_no_real_gain():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 3000)
    proba = rng.dirichlet(np.ones(3), size=3000)  # вероятности не связаны с классом
    found = postprocess.fit_class_weights(y, proba, "macro_f1")
    assert abs(np.mean(found.folds_after) - np.mean(found.folds_before)) < 0.02
    decision = agent.decide(MockCand(found.folds_after), MockCand(found.folds_before), True, agent.AgentConfig())
    assert not decision.accept


def test_class_weights_skip_non_label_metrics_and_bad_input():
    y, proba = skewed_proba(500)
    for metric in ("rmse", "roc_auc", "logloss"):
        assert postprocess.fit_class_weights(y, proba, metric) is None
    assert postprocess.fit_class_weights(y, proba[:, :2], "macro_f1") is None          # не та форма
    assert postprocess.fit_class_weights(y, proba, "macro_f1", classes=["a", "b", "c"]) is None  # метки вне классов
    binary = postprocess.fit_class_weights(y > 0, proba[:, 0], "f1", classes=["False", "True"])
    assert binary is not None and len(binary.weights) == 2  # бинарная: один столбец вероятностей
    weighted = postprocess.apply_weights(proba, np.array([1.0, 1.0, 4.0]))
    assert np.allclose(weighted.sum(axis=1), 1.0)


class ShyEvaluator(StubEvaluator):
    """StubEvaluator, который занижает вероятность класса «high»: argmax его недопредсказывает."""

    def __init__(self) -> None:
        super().__init__(n_splits=3, n_estimators=60)

    def __call__(self, spec, build_features, params, deadline=None):
        cand = super().__call__(spec, build_features, params, deadline)
        k = cand.meta["classes"].index("high")
        for proba in (cand.oof, cand.test_pred):
            proba[:, k] *= 0.1
            proba /= proba.sum(axis=1, keepdims=True)
        y = np.searchsorted(cand.meta["classes"], pd.read_csv(spec.task_dir / "train.csv")[spec.target].astype(str))
        folds = cand.meta["fold_ids"]  # CV — по искажённым вероятностям, как у честного evaluator
        cand.meta["fold_scores"] = [score(spec.metric, y[folds == f], cand.oof[folds == f], False)
                                    for f in np.unique(folds)]
        cand.cv_score, cand.cv_std = float(np.mean(cand.meta["fold_scores"])), float(np.std(cand.meta["fold_scores"]))
        return cand


def test_research_postprocesses_best_multiclass(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls", n_train=1200)
    evaluator = ShyEvaluator()
    result = agent.research(spec, make_best(spec, evaluator), MockLLM(), SimpleDeadline(600),
                            config=agent.AgentConfig(**POST), evaluate=evaluator)
    assert result and result[0].name == "agent_post"
    post = result[0].meta["postprocess"]
    assert post["weights"]["high"] == max(post["weights"].values()) and post["gain"] > 0
    assert np.allclose(result[0].test_pred.sum(axis=1), 1.0)
    # оценка по фолдам evaluator: «после» парно сравнимо с другими кандидатами (например, с fallback P3)
    assert result[0].meta["fold_scores"] == post["folds_after"]
    assert result[0].cv_score == pytest.approx(np.mean(post["folds_after"]))
    row = experiments(spec)[-1]
    assert row["kind"] == "postprocess" and row["status"] == "accepted" and "weights high×" in row["error"]


def test_research_postprocess_off_and_regression(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls", n_train=1200)
    evaluator = ShyEvaluator()
    agent.research(spec, make_best(spec, evaluator), MockLLM(), SimpleDeadline(600),
                   config=agent.AgentConfig(**FAST), evaluate=evaluator)
    assert all(r["kind"] != "postprocess" for r in experiments(spec))

    reg = make_regression_task(tmp_path / "reg")
    evaluator = StubEvaluator(n_splits=3, n_estimators=60)
    agent.research(reg, make_best(reg, evaluator), MockLLM(), SimpleDeadline(600),
                   config=agent.AgentConfig(**POST), evaluate=evaluator)
    assert all(r["kind"] != "postprocess" for r in experiments(reg))


def test_postprocess_runs_when_llm_becomes_unavailable(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls", n_train=1200)
    evaluator = ShyEvaluator()
    llm = MockLLM({"planner": [agent.LLMUnavailable("Budget has been exceeded")]})
    result = agent.research(spec, make_best(spec, evaluator), llm, SimpleDeadline(600),
                            config=agent.AgentConfig(**POST), evaluate=evaluator)
    assert result and result[0].name == "agent_post"
