"""Постобработка вероятностей (P4): множители классов под метрики по меткам (macro F1, accuracy, ...).

Класс = argmax(p · w). Веса подбираются координатным подъёмом по лог-сетке на OOF-вероятностях.
Честная оценка: OOF-строки делятся на K частей (фиксированный сид), веса подбираются на K−1 частях и
проверяются на оставшейся; оценки «до» (w = 1) и «после» на одних и тех же частях идут в обычное
правило принятия агента. Только numpy/pandas; LLM не вызывается.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from agent_eval import LABEL_METRICS, metric_name

GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 9))  # множители веса класса за один шаг: 0.25 … 4


@dataclass
class PostResult:
    weights: np.ndarray         # итоговые множители по всем OOF
    classes: list[str]          # порядок столбцов вероятностей
    folds_before: list[float]   # метрика на частях OOF без весов ...
    folds_after: list[float]    # ... и с весами, подобранными на остальных частях


def fast_label_score(metric: str, y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """То же, что agent_eval.label_score (sklearn), но через матрицу ошибок: в подборе весов — сотни вызовов."""
    name = metric_name(metric)
    cm = np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    if name in ("accuracy", "micro_f1"):
        return float(tp.sum() / max(1, len(y)))
    if name == "balanced_accuracy":
        present = support > 0
        return float(np.mean(tp[present] / support[present]))
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    if name == "f1" and n_classes == 2:
        return float(f1[1])
    if name == "weighted_f1":
        return float((f1 * support).sum() / max(1, support.sum()))
    if name in ("macro_f1", "f1"):  # как в sklearn: классы, встречающиеся в y или в предсказаниях
        return float(f1[denom > 0].mean())
    raise ValueError(f"unknown label metric: {metric}")


def apply_weights(proba: Any, weights: np.ndarray) -> np.ndarray:
    """Перевзвешенные и нормированные вероятности: argmax по ним — класс с учётом весов."""
    scaled = np.asarray(proba, dtype=float) * weights
    total = scaled.sum(axis=1, keepdims=True)
    return np.divide(scaled, total, out=np.full_like(scaled, 1.0 / len(weights)), where=total > 0)


def _fit(y: np.ndarray, proba: np.ndarray, metric: str, rounds: int = 2) -> np.ndarray:
    """Координатный подъём: по очереди для каждого класса — лучший множитель из сетки, пока метрика растёт."""
    k = proba.shape[1]
    weights = np.ones(k)
    best = fast_label_score(metric, y, proba.argmax(axis=1), k)
    for _ in range(rounds):
        improved = False
        for cls in range(k):
            for factor in GRID:
                trial = weights.copy()
                trial[cls] *= factor
                value = fast_label_score(metric, y, (proba * trial).argmax(axis=1), k)
                if value > best + 1e-12:
                    best, weights, improved = value, trial, True
        if not improved:
            break
    return weights / weights.min()  # ≥ 1: во сколько раз поднят класс относительно наименее поднятого


def fit_class_weights(y: Any, proba: Any, metric: str, *, classes: Iterable | None = None, n_splits: int = 5,
                      seed: int = 42) -> PostResult | None:
    """Множители вероятностей классов под метрику по меткам; None — метрика не по меткам или данные не те.

    y — исходные метки; столбцы proba идут в порядке classes (по умолчанию — отсортированные строковые метки).
    """
    name = metric_name(metric)
    if name not in LABEL_METRICS or proba is None:
        return None
    labels = pd.Series(np.asarray(y).reshape(-1)).astype(str)
    names = [str(c) for c in classes] if classes is not None else sorted(labels.unique())
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1 and len(names) == 2:  # бинарная: вероятность положительного класса
        proba = np.column_stack([1.0 - proba, proba])
    if proba.ndim != 2 or proba.shape != (len(labels), len(names)) or len(names) < 2 or len(labels) < 10 * n_splits:
        return None
    true = pd.Categorical(labels, categories=names).codes.astype(int)
    if (true < 0).any() or not np.isfinite(proba).all():
        return None
    parts = np.random.default_rng(seed).permutation(len(true)) % n_splits
    before, after = [], []
    for part in range(n_splits):
        held = parts == part
        weights = _fit(true[~held], proba[~held], name)
        before.append(fast_label_score(name, true[held], proba[held].argmax(axis=1), len(names)))
        after.append(fast_label_score(name, true[held], (proba[held] * weights).argmax(axis=1), len(names)))
    return PostResult(_fit(true, proba, name), names, before, after)
