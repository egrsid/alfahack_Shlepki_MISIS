import numpy as np
import pytest
from sklearn import metrics as skm

import metrics as M

RNG = np.random.default_rng(0)
Y_REG = RNG.uniform(0, 10, 300)
P_REG = Y_REG + RNG.normal(size=300)
Y_BIN = RNG.integers(0, 2, 300)
P_BIN = np.clip(Y_BIN * 0.3 + RNG.random(300) * 0.7, 0, 1)
Y_MC = RNG.integers(0, 4, 400)
P_MC = RNG.dirichlet(np.ones(4), 400)
P_MC[np.arange(400), Y_MC] += 0.3
P_MC /= P_MC.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("name, expected", [
    ("rmse", skm.mean_squared_error(Y_REG, P_REG) ** 0.5),
    ("mse", skm.mean_squared_error(Y_REG, P_REG)),
    ("mae", skm.mean_absolute_error(Y_REG, P_REG)),
    ("medae", skm.median_absolute_error(Y_REG, P_REG)),
    ("r2", skm.r2_score(Y_REG, P_REG)),
    ("mape", skm.mean_absolute_percentage_error(Y_REG, P_REG)),
    ("rmsle", skm.mean_squared_log_error(Y_REG, np.clip(P_REG, 0, None)) ** 0.5),
])
def test_regression_matches_sklearn(name, expected):
    assert M.score(name, Y_REG, P_REG) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("name, expected", [
    ("roc_auc", skm.roc_auc_score(Y_BIN, P_BIN)),
    ("gini", 2 * skm.roc_auc_score(Y_BIN, P_BIN) - 1),
    ("logloss", skm.log_loss(Y_BIN, P_BIN)),
    ("f1", skm.f1_score(Y_BIN, P_BIN >= 0.5)),
    ("accuracy", skm.accuracy_score(Y_BIN, P_BIN >= 0.5)),
    ("mcc", skm.matthews_corrcoef(Y_BIN, P_BIN >= 0.5)),
    ("pr_auc", skm.average_precision_score(Y_BIN, P_BIN)),
])
def test_binary_matches_sklearn(name, expected):
    assert M.score(name, Y_BIN, P_BIN) == pytest.approx(expected, rel=1e-9)
    assert M.score(name, Y_BIN, np.column_stack([1 - P_BIN, P_BIN])) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("name, expected", [
    ("macro_f1", skm.f1_score(Y_MC, P_MC.argmax(1), average="macro")),
    ("weighted_f1", skm.f1_score(Y_MC, P_MC.argmax(1), average="weighted")),
    ("balanced_accuracy", skm.balanced_accuracy_score(Y_MC, P_MC.argmax(1))),
    ("logloss", skm.log_loss(Y_MC, P_MC)),
    ("roc_auc", skm.roc_auc_score(Y_MC, P_MC, multi_class="ovr")),
    ("qwk", skm.cohen_kappa_score(Y_MC, P_MC.argmax(1), weights="quadratic")),
])
def test_multiclass_matches_sklearn(name, expected):
    assert M.score(name, Y_MC, P_MC) == pytest.approx(expected, rel=1e-9)


def test_labels_and_string_targets():
    assert M.score("macro_f1", Y_MC, P_MC.argmax(1)) == pytest.approx(M.score("macro_f1", Y_MC, P_MC))
    labels = np.array(["a", "b", "c", "d"], dtype=object)[Y_MC]
    assert M.score("accuracy", labels, P_MC) == pytest.approx(M.score("accuracy", Y_MC, P_MC))
    predicted = np.array(["a", "b", "c", "d"], dtype=object)[P_MC.argmax(1)]  # строковые метки и в y_pred
    assert M.score("macro_f1", labels, predicted) == pytest.approx(M.score("macro_f1", Y_MC, P_MC))
    assert M.score("macro_f1", list(labels), list(predicted)) == pytest.approx(M.score("macro_f1", Y_MC, P_MC))


def test_fast_metric_matches_registry():
    pred = P_MC.argmax(1)
    for name in ("macro_f1", "weighted_f1", "balanced_accuracy"):
        assert M._fast_label_metric(name, 4)(Y_MC, pred) == pytest.approx(M.score(name, Y_MC, pred))
    assert M._fast_label_metric("f1", 2)(Y_BIN, (P_BIN >= .5).astype(int)) == pytest.approx(M.score("f1", Y_BIN, P_BIN))


@pytest.mark.parametrize("alias, canonical", [
    ("Macro F1", "macro_f1"), ("f1-macro", "macro_f1"), ("F1_MACRO", "macro_f1"), ("RMSE", "rmse"),
    ("ROC AUC", "roc_auc"), ("auc", "roc_auc"), ("log_loss", "logloss"), ("R²", "r2"), ("weighted-f1", "weighted_f1"),
])
def test_resolve_aliases(alias, canonical):
    assert M.resolve(alias) == canonical


def test_unknown_metric():
    with pytest.raises(M.UnknownMetric):
        M.get("pinball_whatever")
    assert M.default_for("multiclass") == "macro_f1"


@pytest.mark.parametrize("text, expected", [
    ("Основная метрика — Macro F1, дополнительно смотрим Accuracy.", "macro_f1"),
    ("Метрика качества: RMSE (root mean squared error).", "rmse"),
    ("We report accuracy too. The evaluation metric is weighted F1-score.", "weighted_f1"),
    ("Качество оценивается по MAE", "mae"),
    ("Evaluation: ROC-AUC on the hidden test set", "roc_auc"),
    ("Решения сравниваются по коэффициенту детерминации", "r2"),
    ("Описание без метрики", None),
])
def test_find_in_text(text, expected):
    assert M.find_in_text(text) == expected


def test_classes_order_and_encode():
    assert M.classes_of([10, 2, 1, 2]) == ["1", "2", "10"]
    assert M.classes_of([1.0, 0.0, np.nan]) == ["0", "1"]
    assert M.classes_of(["b", "a", "b"]) == ["a", "b"]
    assert M.classes_of([True, False]) == ["False", "True"]
    assert list(M.encode([10, 1, 2], ["1", "2", "10"])) == [2, 0, 1]
    with pytest.raises(ValueError):
        M.encode(["x"], ["a", "b"])


def test_score_rejects_bad_input():
    with pytest.raises(ValueError):
        M.score("rmse", Y_REG, P_REG[:-1])
    with pytest.raises(ValueError):
        M.score("rmse", Y_REG, np.where(np.arange(300) == 0, np.nan, P_REG))


def test_is_better_respects_direction():
    assert M.is_better("rmse", 1.0, 2.0) and not M.is_better("rmse", 2.0, 1.0)
    assert M.is_better("macro_f1", 0.6, 0.5) and M.is_better("macro_f1", 0.1, None)
    assert not M.is_better("macro_f1", float("nan"), 0.5)


def test_decision_improves_macro_f1_on_imbalanced():
    rng = np.random.default_rng(1)
    n, k = 4000, 4
    y = rng.choice(k, n, p=[0.7, 0.2, 0.07, 0.03])
    logits = rng.normal(size=(n, k)) + 1.2 * np.eye(k)[y] + np.log([0.7, 0.2, 0.07, 0.03])
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    decision = M.fit_decision("macro_f1", y, proba)
    assert decision is not None and decision["kind"] == "class_weights"
    assert M.score("macro_f1", y, proba, decision) > M.score("macro_f1", y, proba) + 0.01
    folds = np.arange(n) % 5
    fold_scores, final = M.crossfit_decision("macro_f1", y, proba, folds)
    assert len(fold_scores) == 5 and final is not None
    assert np.mean(fold_scores) > M.score("macro_f1", y, proba)  # выигрыш держится и на чужих фолдах


def test_binary_threshold_and_untunable_metrics():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.1).astype(int)
    p = np.clip(0.05 + 0.25 * y + rng.normal(scale=0.1, size=3000), 0, 1)
    decision = M.fit_decision("f1", y, p)
    assert decision["kind"] == "threshold" and decision["threshold"] < 0.5
    assert M.score("f1", y, p, decision) > M.score("f1", y, p)
    assert M.fit_decision("roc_auc", y, p) is None and M.fit_decision("recall", y, p) is None


def test_estimate_priors_recovers_label_shift():
    rng = np.random.default_rng(3)
    k, n = 3, 20_000
    means = np.array([[-2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])

    def sample(prior):
        y = rng.choice(k, n, p=prior)
        x = means[y] + rng.normal(size=(n, 2))
        logp = -0.5 * ((x[:, None, :] - means[None]) ** 2).sum(-1) + np.log(train_prior)
        return y, np.exp(logp) / np.exp(logp).sum(1, keepdims=True)  # точные апостериорные при train_prior

    train_prior = np.array([0.6, 0.3, 0.1])
    _, p_train = sample(train_prior)
    assert np.allclose(M.estimate_priors(p_train, train_prior), train_prior, atol=0.02)
    shifted = np.array([0.3, 0.3, 0.4])
    _, p_test = sample(shifted)
    assert np.allclose(M.estimate_priors(p_test, train_prior), shifted, atol=0.03)


def test_decision_with_sample_weight_follows_priors():
    rng = np.random.default_rng(4)
    y = (rng.random(5000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.4 * y + rng.normal(scale=0.15, size=5000), 0, 1)
    plain = M.fit_decision("f1", y, p)
    heavy = M.fit_decision("f1", y, p, sample_weight=np.where(y == 1, 5.0, 1.0))  # в test положительных в 5 раз больше
    assert heavy["threshold"] < plain["threshold"]
    scores, decision = M.crossfit_decision("f1", y, p, np.arange(5000) % 5, sample_weight=np.where(y == 1, 5.0, 1.0))
    assert len(scores) == 5 and decision["kind"] == "threshold"
