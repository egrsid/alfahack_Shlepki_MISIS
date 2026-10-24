import numpy as np
import pandas as pd
import pytest
from p3_fixtures import classification_task, regression_task, write_task

import metrics
import submission as S


def _read(spec):
    return pd.read_csv(spec.task_dir / "submission.csv")


def test_regression_aligned_by_id_when_sample_is_shuffled(tmp_path):
    spec, _ = regression_task(tmp_path, shuffle_sample=True)
    test = pd.read_csv(tmp_path / "test.csv")
    pred = test["id"].to_numpy(dtype=float) * 2  # прогноз однозначно связан с id
    S.write(spec, pred)
    out, sample = _read(spec), pd.read_csv(tmp_path / "sample_submission.csv")
    assert list(out.columns) == ["id", "target"]
    assert out["id"].tolist() == sample["id"].tolist()
    assert np.allclose(out["target"], out["id"] * 2)
    assert not list(tmp_path.glob(".submission.csv.*"))  # временный файл убран


def test_nan_predictions_are_repaired(tmp_path):
    spec, _ = regression_task(tmp_path)
    pred = np.full(200, 5.0)
    pred[:3] = [np.nan, np.inf, -np.inf]
    S.write(spec, pred)
    out = _read(spec)
    assert np.isfinite(out["target"]).all() and out["target"].iloc[3] == 5.0


def test_wrong_length_raises_and_keeps_old_file(tmp_path):
    spec, _ = regression_task(tmp_path)
    S.write(spec, np.ones(200))
    before = (tmp_path / "submission.csv").read_text()
    with pytest.raises(S.SubmissionError):
        S.write(spec, np.ones(199))
    assert (tmp_path / "submission.csv").read_text() == before


def test_multiclass_string_labels_and_decision(tmp_path):
    spec, _ = classification_task(tmp_path, n_classes=3)
    classes = metrics.classes_of(pd.read_csv(tmp_path / "train.csv")["label"])
    assert classes == ["alpha", "beta", "gamma"]
    proba = np.tile([0.5, 0.3, 0.2], (300, 1))
    S.write(spec, proba, classes=classes)
    assert set(_read(spec)["label"]) == {"alpha"}
    S.write(spec, proba, classes=classes, decision={"kind": "class_weights", "weights": [1, 1, 5]})
    assert set(_read(spec)["label"]) == {"gamma"}
    assert S.output_kind(spec) == "labels"


def test_integer_labels_keep_integer_form(tmp_path):
    spec, _ = classification_task(tmp_path, n_classes=3, labels=[0, 1, 10])
    classes = metrics.classes_of(pd.read_csv(tmp_path / "train.csv")["label"])
    assert classes == ["0", "1", "10"]
    proba = np.zeros((300, 3))
    proba[:, 2] = 1
    S.write(spec, proba, classes=classes)
    assert (tmp_path / "submission.csv").read_text().splitlines()[1].endswith(",10")
    assert _read(spec)["label"].dtype.kind == "i"


def test_binary_auc_writes_positive_probability(tmp_path):
    spec, _ = classification_task(tmp_path, n_classes=2, labels=["no", "yes"], metric="roc_auc")
    assert S.output_kind(spec) == "proba"
    p_yes = np.linspace(0, 1, 300)
    S.write(spec, np.column_stack([1 - p_yes, p_yes]), classes=["no", "yes"])
    assert np.allclose(_read(spec)["label"], p_yes)
    S.write(spec, p_yes, classes=["no", "yes"])  # 1-D вероятность положительного класса тоже принимается
    assert np.allclose(_read(spec)["label"], p_yes)


def test_proba_columns_layout(tmp_path):
    spec, _ = classification_task(tmp_path, n_classes=3, metric="logloss")
    test = pd.read_csv(tmp_path / "test.csv")
    pd.DataFrame({"id": test.id, "gamma": 0.2, "alpha": 0.5, "beta": 0.3}).to_csv(
        tmp_path / "sample_submission.csv", index=False)
    assert S.output_kind(spec) == "proba_columns"
    proba = np.tile([0.6, 0.3, 0.1], (300, 1))  # порядок классов alpha, beta, gamma
    S.write(spec, proba, classes=["alpha", "beta", "gamma"])
    out = _read(spec)
    assert list(out.columns) == ["id", "gamma", "alpha", "beta"]
    assert np.allclose(out[["alpha", "beta", "gamma"]].iloc[0], [0.6, 0.3, 0.1])


def test_test_without_id_column_uses_row_order(tmp_path):
    spec, _ = regression_task(tmp_path)
    test = pd.read_csv(tmp_path / "test.csv").drop(columns="id")
    test.to_csv(tmp_path / "test.csv", index=False)
    S.write(spec, np.arange(200, dtype=float))
    assert np.allclose(_read(spec)["target"], np.arange(200))


def test_write_dummy_and_check_file(tmp_path):
    spec, _ = classification_task(tmp_path, n_classes=3)
    assert S.check_file(spec)  # файла ещё нет
    path = S.write_dummy(spec)
    assert path is not None and S.check_file(spec) == []
    assert S.write_dummy(spec) is None  # существующий файл не трогаем
    (tmp_path / "submission.csv").write_text("id,label\n1,unknown\n")
    problems = S.check_file(spec)
    assert any("rows" in p for p in problems) and any("unknown labels" in p for p in problems)


def test_regression_dummy_uses_median_for_mae(tmp_path):
    spec, _ = regression_task(tmp_path)
    spec.metric = "MAE"
    S.write_dummy(spec)
    y = pd.read_csv(tmp_path / "train.csv")["target"]
    assert _read(spec)["target"].iloc[0] == pytest.approx(y.median())


def test_bool_target_written_like_sample(tmp_path):
    train = pd.DataFrame({"id": range(6), "x": range(6), "flag": [True, False] * 3})
    test = pd.DataFrame({"id": range(6, 9), "x": range(3)})
    sample = pd.DataFrame({"id": range(6, 9), "flag": 0})
    write_task(tmp_path, train, test, sample)
    from p3_fixtures import Spec
    spec = Spec(tmp_path, "id", "flag", "binary", "accuracy", True)
    S.write(spec, np.array([0.9, 0.1, 0.7]), classes=["False", "True"])
    assert _read(spec)["flag"].tolist() == [1, 0, 1]
    assert S.check_file(spec) == []
