import time

import numpy as np
import pandas as pd
import pytest

from error_analysis import HEADER, analyze_errors, row_errors, worst_segments


@pytest.fixture
def regression_case():
    """Город c и большие x2 — сегменты с заложенной большой ошибкой; верх таргета недооценён."""
    rng = np.random.default_rng(0)
    n = 3000
    x = pd.DataFrame({"city": rng.choice(["a", "b", "c"], n, p=[0.45, 0.45, 0.1]), "x1": rng.normal(size=n),
                      "x2": rng.uniform(0, 1, n), "flag": rng.integers(0, 2, n)})
    y = 10 + 3 * x.x1 + rng.normal(scale=0.5, size=n)
    noise = np.where(x.city == "c", 8.0, 0.5) * rng.normal(size=n)
    pred = y + noise - np.where(y > np.quantile(y, 0.9), 2.0, 0.0)
    return x, y.to_numpy(), pred


@pytest.fixture
def multiclass_case():
    """Классы A/B/C; при x > 0.8 класс B путается с C."""
    rng = np.random.default_rng(1)
    n = 2000
    x = pd.DataFrame({"x": rng.uniform(0, 1, n), "proto": rng.choice(["tcp", "udp"], n)})
    y = rng.choice(["A", "B", "C"], n)
    proba = np.full((n, 3), 0.05)
    true = pd.Categorical(y, categories=["A", "B", "C"]).codes
    proba[np.arange(n), true] = 0.9
    hard = (x.x > 0.8).to_numpy() & (y == "B")
    proba[hard] = [0.05, 0.25, 0.7]
    return x, y, proba


def test_regression_finds_planted_segment_and_bias(regression_case):
    x, y, pred = regression_case
    text = analyze_errors(x, y, pred, "regression", cat_cols=["city"])
    lines = text.splitlines()
    assert lines[0] == HEADER and len(text) <= 800
    segments = text.split("segments with the highest error: ")[1]
    assert segments.startswith("city=c (n=")  # заложенный плохой сегмент — первый
    assert "target top 10%" in text and "underpredicted" in text and "bottom 10%" not in text  # слабое — не шум
    assert all(float(part.split(": ")[1].split("x")[0]) >= 1.2
                                          for part in segments.split("; "))
    assert "10% worst rows give" in text and "RMSE" in text


def test_multiclass_confusion_recall_and_segment(multiclass_case):
    x, y, proba = multiclass_case
    text = analyze_errors(x, y, proba, "multiclass")
    assert "B -> C" in text.split("top confusions")[1]
    assert "lowest recall: B 0.80 (n=" in text and "A 1.00" not in text
    assert "x in [" in text.split("segments with the highest error: ")[1].split(";")[0]


def test_binary_with_positive_probability_vector():
    rng = np.random.default_rng(2)
    x = pd.DataFrame({"g": rng.choice(["u", "v"], 500)})
    y = rng.choice(["no", "yes"], 500)
    p_yes = np.where(y == "yes", 0.8, 0.2)
    p_yes[(x.g == "v").to_numpy()] = 0.5
    text = analyze_errors(x, y, p_yes, "binary")
    assert "g=v" in text and "misclassified" in text


def test_explicit_class_order_is_respected():
    y = np.array(["b", "a", "b", "a"] * 10)
    proba = np.tile([[0.9, 0.1], [0.1, 0.9]], (20, 1))  # столбцы в порядке ["b", "a"]
    errors, _ = row_errors(y, proba, "multiclass", classes=["b", "a"])
    assert np.allclose(errors, 0.1)


@pytest.mark.parametrize("max_chars", [150, 300, 800])
def test_length_limit(regression_case, max_chars):
    x, y, pred = regression_case
    assert len(analyze_errors(x, y, pred, "regression", max_chars=max_chars)) <= max_chars


def test_garbage_inputs_return_empty():
    x = pd.DataFrame({"a": range(100)})
    y = np.arange(100.0)
    assert analyze_errors(x, y, None, "regression") == ""
    assert analyze_errors(x, y, np.zeros(50), "regression") == ""  # не та длина
    assert analyze_errors(x, y, np.zeros((100, 3)), "regression") == ""  # не та форма
    assert analyze_errors(x, np.array(["a", "b"] * 50), np.zeros((100, 3)), "multiclass") == ""  # 2 метки, 3 столбца
    assert analyze_errors(x, np.array(["a", "z"] * 50), np.zeros((100, 2)), "multiclass",
                          classes=["a", "b"]) == ""  # метка вне classes
    assert analyze_errors(pd.DataFrame(), y, y, "regression") == ""
    assert analyze_errors(x, pd.DataFrame({"t1": y, "t2": y}), y, "regression") == ""  # несколько таргетов


def test_nan_target_constant_and_mixed_columns():
    rng = np.random.default_rng(3)
    n = 400
    x = pd.DataFrame({"const": ["k"] * n, "mixed": [1, "a", None, 2.5] * (n // 4),
                      "num": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
                      "when": pd.date_range("2020-01-01", periods=n, freq="D")})
    y = rng.normal(size=n)
    y[:20] = np.nan
    pred = y + np.where(np.isnan(x.num), 3.0, 0.1)
    text = analyze_errors(x, y, pred, "regression")
    assert text.startswith(HEADER) and "num is missing" in text


def test_worst_segments_respects_min_size():
    x = pd.DataFrame({"g": ["rare"] * 5 + ["common"] * 995})
    errors = np.r_[np.full(5, 100.0), np.ones(995)]
    assert worst_segments(x, errors, min_segment=30) == []
    assert worst_segments(x, errors, min_segment=5)[0][1].startswith("g=rare (n=5)")


def test_speed_on_public_size():
    rng = np.random.default_rng(4)
    n, k = 90_000, 45
    x = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"c{i}" for i in range(k)])
    x["cat"] = rng.choice([f"l{i}" for i in range(300)], n)
    y = rng.choice([f"k{i}" for i in range(10)], n)
    proba = rng.dirichlet(np.ones(10), n)
    start = time.monotonic()
    assert analyze_errors(x, y, proba, "multiclass", cat_cols=["cat"]).startswith(HEADER)
    assert time.monotonic() - start < 5.0
