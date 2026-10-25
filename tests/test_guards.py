import types

import numpy as np

import guards

REG = types.SimpleNamespace(task_type="regression")
CLS = types.SimpleNamespace(task_type="multiclass")


def cand(oof, test):
    return types.SimpleNamespace(name="c", oof=np.asarray(oof, dtype=float), test_pred=np.asarray(test, dtype=float))


def test_regression_close_to_baseline_passes_and_drift_is_caught():
    rng = np.random.default_rng(0)
    base_test = rng.normal(10, 2, 500)
    base = cand(rng.normal(10, 2, 1000), base_test)
    good = cand(rng.normal(10, 2, 1000), base_test + rng.normal(0, 0.3, 500))
    assert guards.check_candidate(REG, good, base) == []
    shifted = cand(good.oof, base_test + 5)
    assert any("shifted" in p for p in guards.check_candidate(REG, shifted, base))
    noise = cand(good.oof, rng.normal(10, 2, 500))
    assert any("correlate" in p for p in guards.check_candidate(REG, noise, base))
    wild = cand(good.oof, np.where(np.arange(500) < 50, 1e6, base_test))
    assert any("outside" in p for p in guards.check_candidate(REG, wild, base))


def test_shape_and_nan():
    base = cand(np.zeros(10), np.zeros(5))
    assert "shape" in guards.check_candidate(REG, cand(np.zeros(10), np.zeros(4)), base)[0]
    assert "NaN" in guards.check_candidate(REG, cand(np.zeros(10), [np.nan] * 5), base)[0]
    assert guards.check_candidate(REG, cand(np.zeros(10), np.zeros(5)), None) == []


def proba(labels, k, confidence):
    p = np.full((len(labels), k), (1 - confidence) / (k - 1))
    p[np.arange(len(labels)), labels] = confidence
    return p


def test_classification_shares_and_leak_confidence():
    rng = np.random.default_rng(1)
    labels_oof, labels_test = rng.integers(0, 3, 900), rng.integers(0, 3, 300)
    base = cand(proba(labels_oof, 3, 0.7), proba(labels_test, 3, 0.7))
    assert guards.check_candidate(CLS, cand(proba(labels_oof, 3, 0.75), proba(labels_test, 3, 0.72)), base) == []
    one_class = cand(base.oof, proba(np.zeros(300, dtype=int), 3, 0.7))
    assert any("class shares" in p for p in guards.check_candidate(CLS, one_class, base))
    leak = cand(proba(labels_oof, 3, 0.99), proba(labels_test, 3, 0.6))
    assert any("leakage" in p for p in guards.check_candidate(CLS, leak, base))


def test_missing_oof_is_not_a_problem():
    base = cand(np.zeros(10), np.arange(5.0))
    no_oof = types.SimpleNamespace(name="c", oof=None, test_pred=np.arange(5.0) + 0.1)
    assert guards.check_candidate(REG, no_oof, base) == []
