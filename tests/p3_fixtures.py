"""Маленькие задачи в формате платформы для тестов P3 (metrics, submission, preprocess, fallback); вне ZIP.

Spec повторяет поля TaskSpec из schemas.py (P1), пока его нет в main.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Spec:
    task_dir: Path
    id_col: str
    target: str | list[str]
    task_type: str
    metric: str
    greater_is_better: bool = False
    feature_notes: dict[str, str] = field(default_factory=dict)
    drop_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    date_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    raw_task_text: str = ""
    source: str = "heuristic"


def write_task(root: Path, train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame,
               text: str = "Задача.") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    sample.to_csv(root / "sample_submission.csv", index=False)
    (root / "task.md").write_text(text, encoding="utf-8")
    return root


def regression_task(root: Path, n_train: int = 600, n_test: int = 200, seed: int = 0,
                    shuffle_sample: bool = False) -> tuple[Spec, pd.Series]:
    """y = 3·x1 + x2² + эффект категории + шум; в данных пропуски, категория, дата, константа. Возвращает y теста."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame({
        "id": np.arange(n) + 10_000,
        "x1": rng.normal(size=n),
        "x2": rng.uniform(-2, 2, n),
        "city": rng.choice(["msk", "spb", "ekb", "nsk"], n),
        "built": pd.to_datetime("2000-01-01") + pd.to_timedelta(rng.integers(0, 7000, n), unit="D"),
        "const": 1,
    })
    y = 3 * df.x1 + df.x2 ** 2 + df.city.map({"msk": 2.0, "spb": 1.0, "ekb": 0.0, "nsk": -1.0}) \
        + rng.normal(scale=0.3, size=n) + 10
    df.loc[rng.random(n) < 0.1, "x1"] = np.nan
    df["built"] = df["built"].dt.strftime("%Y-%m-%d")
    train = df.iloc[:n_train].assign(target=y[:n_train].to_numpy())
    test = df.iloc[n_train:].reset_index(drop=True)
    sample = pd.DataFrame({"id": test.id, "target": 0.0})
    if shuffle_sample:
        sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    write_task(root, train, test, sample, "Предсказать target. Метрика качества: RMSE.")
    spec = Spec(root, "id", "target", "regression", "rmse", False)
    return spec, y[n_train:].reset_index(drop=True)


def classification_task(root: Path, n_train: int = 900, n_test: int = 300, seed: int = 0, n_classes: int = 3,
                        labels: list | None = None, metric: str = "macro_f1",
                        imbalance: bool = True) -> tuple[Spec, pd.Series]:
    """Классы зависят от x1, x2 и категории; метки — строки по умолчанию. Возвращает метки теста."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    labels = labels or ["alpha", "beta", "gamma", "delta", "eps"][:n_classes]
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    proto = rng.choice(["tcp", "udp", "icmp", "arp"], n, p=[0.6, 0.3, 0.08, 0.02])
    logits = np.column_stack([x1 * (k - 1) + x2 * ((k % 2) * 2 - 1) + (proto == "udp") * k
                              for k in range(len(labels))])
    if imbalance:
        logits = logits + np.log(np.linspace(1.0, 0.1, len(labels)))
    code = np.array([rng.choice(len(labels), p=np.exp(r) / np.exp(r).sum()) for r in logits * 2])
    df = pd.DataFrame({"id": np.arange(n), "x1": x1, "x2": x2, "proto": proto,
                       "note": rng.choice(["a", "b"], n)})
    y = pd.Series(np.asarray(labels, dtype=object)[code])
    train = df.iloc[:n_train].assign(label=y[:n_train].to_numpy())
    test = df.iloc[n_train:].reset_index(drop=True)
    sample = pd.DataFrame({"id": test.id, "label": labels[0]})
    write_task(root, train, test, sample, f"Предсказать label. Метрика: {metric}.")
    task_type = "binary" if len(labels) == 2 else "multiclass"
    spec = Spec(root, "id", "label", task_type, metric, True)
    return spec, y[n_train:].reset_index(drop=True)
