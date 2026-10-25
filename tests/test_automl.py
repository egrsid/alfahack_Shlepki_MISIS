import sys
import time
import types

import numpy as np
import pandas as pd
import pytest

import automl
import fallback
from mocks import make_multiclass_task, make_regression_task

FAST = {"GBM": {"num_boost_round": 50}, "RF": {"n_estimators": 50}, "KNN": {}}


def fallback_like(spec, folds, rows=None, classes=None, metric=None):
    return fallback.Candidate("fallback", 0.0, 0.0, np.zeros(0 if folds is None else len(folds)), np.zeros(1),
                              {"folds": folds, "train_rows": rows, "classes": classes, "metric": metric or spec.metric})


def test_models_are_trained_on_our_folds(tmp_path):
    # признак g = номер фолда, таргет = g % 2: на чужих фолдах g «узнаётся», на наших (LeaveOneGroupOut) — никогда
    root = tmp_path / "t"
    root.mkdir()
    n = 800
    g = np.arange(n) % 4
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"id": np.arange(n), "g": g, "noise": rng.normal(size=n), "y": g % 2})
    frame.to_csv(root / "train.csv", index=False)
    frame.drop(columns="y").head(50).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": frame.id.head(50), "y": 0}).to_csv(root / "sample_submission.csv", index=False)
    (root / "task.md").write_text("Predict y. Metric: accuracy.", encoding="utf-8")
    spec = fallback.guess_spec(root / "task.md")
    cands = automl.run(spec, fallback_like(spec, g, classes=["0", "1"], metric="accuracy"), seconds=120,
                       models={"GBM": {"num_boost_round": 30}})
    assert cands, "AutoGluon не обучился"
    for cand in cands:
        assert cand.oof.shape == (n, 2) and cand.meta["fold_scores"] and len(cand.meta["fold_scores"]) == 4
        assert cand.cv_score < 0.8  # при случайных фолдах было бы ~1.0


def test_multiclass_classes_aligned_and_regression(tmp_path):
    spec = make_multiclass_task(tmp_path / "cls")
    folds = np.arange(450) % 3
    cands = automl.run(spec, fallback_like(spec, folds, classes=["high", "low", "mid"]), seconds=120, models=FAST)
    assert cands and all(c.name.startswith("automl_") for c in cands)
    best = cands[0]
    assert best.oof.shape == (450, 3) and best.test_pred.shape == (150, 3)
    assert np.allclose(best.oof.sum(axis=1), 1.0, atol=1e-6) and best.cv_score > 0.4
    assert not (spec.task_dir / ".agent_work" / "automl").exists()  # каталог данных и моделей удалён

    spec = make_regression_task(tmp_path / "reg")
    folds = np.arange(400) % 4
    cands = automl.run(spec, fallback_like(spec, folds), seconds=120, models=FAST)
    assert cands and cands[0].oof.shape == (400,) and cands[0].test_pred.shape == (150,)


def test_failures_return_empty(tmp_path, monkeypatch):
    spec = make_regression_task(tmp_path / "reg")
    assert automl.run(spec, fallback_like(spec, None)) == []  # нет фолдов
    assert automl.run(spec, fallback_like(spec, np.arange(10) % 2), models=FAST) == []  # фолды не той длины
    assert automl.run(spec, fallback_like(spec, np.arange(400) % 4), seconds=10) == []  # нет времени
    # процесс-воркер упал (ни одна модель не обучилась) → []
    assert automl.run(spec, fallback_like(spec, np.arange(400) % 4), seconds=60, models={"NO_SUCH_MODEL": {}}) == []


def test_hanging_worker_is_killed(tmp_path, monkeypatch):
    spec = make_regression_task(tmp_path / "reg")
    monkeypatch.setattr(automl, "_command", lambda job: [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(automl, "MIN_SECONDS", 1.0)
    monkeypatch.setattr(automl, "PREDICT_MARGIN", 1.0)
    start = time.monotonic()
    assert automl.run(spec, fallback_like(spec, np.arange(400) % 4), seconds=1.0) == []
    assert time.monotonic() - start < 30 and not (spec.task_dir / ".agent_work" / "automl").exists()


def test_no_pretrained_or_missing_models():
    config = automl._hyperparameters()
    assert not {"TABPFN", "TABPFNMIX", "FASTAI", "FT_TRANSFORMER"} & set(config) and {"GBM", "NN_TORCH"} <= set(config)


def test_missing_autogluon_is_skipped_without_worker(tmp_path, monkeypatch):
    spec = make_regression_task(tmp_path / "reg")
    monkeypatch.setattr(automl, "available", lambda: False)
    monkeypatch.setattr(automl, "_command", lambda job: pytest.fail("worker must not start"))
    assert automl.run(spec, fallback_like(spec, np.arange(400) % 4), seconds=120) == []
