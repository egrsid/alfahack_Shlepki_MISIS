"""Анализ худших объектов (P4): где текущая лучшая модель ошибается сильнее всего.

По out-of-fold предсказаниям кандидата и таргету — короткая сводка для планировщика: концентрация ошибки,
сегменты признаков с повышенной ошибкой, смещение на краях таргета (регрессия), худшие классы и пары
путаницы (классификация). В сводку попадают только агрегаты по сегментам, никогда — строки данных.
Модуль не зависит от остальных модулей команды и не вызывает LLM.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

HEADER = "Error analysis of the current best model (out-of-fold):"


def _num(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.3g}" if abs(value) < 1e4 else f"{value:.2e}"


def row_errors(y: Any, oof: Any, task_type: str, classes: Iterable | None = None) -> tuple[np.ndarray, dict]:
    """Ошибка на каждом объекте: регрессия — |y − oof|, классификация — 1 − p(истинный класс).

    Для классификации столбцы oof идут в порядке classes, по умолчанию — отсортированных строковых меток.
    Неверная форма или метки вне classes → ValueError.
    """
    if task_type == "regression":
        target = pd.to_numeric(pd.Series(np.asarray(y).reshape(-1)), errors="coerce").to_numpy(float)
        pred = np.asarray(oof, dtype=float)
        if pred.ndim == 2 and pred.shape[1] == 1:
            pred = pred[:, 0]
        if pred.ndim != 1 or len(pred) != len(target):
            raise ValueError(f"oof shape {pred.shape} does not match {len(target)} rows")
        return np.abs(target - pred), {"y": target, "pred": pred}

    labels = pd.Series(np.asarray(y).reshape(-1)).astype(str)
    names = [str(c) for c in classes] if classes is not None else sorted(labels.unique())
    proba = np.asarray(oof, dtype=float)
    if proba.ndim == 1 and len(names) == 2:  # бинарная: вероятность положительного класса
        proba = np.column_stack([1.0 - proba, proba])
    if proba.ndim != 2 or proba.shape != (len(labels), len(names)):
        raise ValueError(f"oof shape {proba.shape} does not match {len(labels)} rows x {len(names)} classes")
    true = pd.Categorical(labels, categories=names).codes
    if (true < 0).any():
        raise ValueError("target labels are missing from classes")
    p_true = proba[np.arange(len(true)), true]
    return 1.0 - p_true, {"true": true, "pred": proba.argmax(axis=1), "classes": names}


def _segment_keys(series: pd.Series, column: str, is_cat: bool, max_levels: int) -> pd.Series:
    """Метка сегмента для каждой строки (NaN — строка не входит ни в один сегмент)."""
    if is_cat or series.nunique(dropna=True) <= 10:
        values = series.astype(object)
        keys = values.where(values.isna(), values.astype(str)).map(lambda v: f"{column}={v}", na_action="ignore")
        keys = keys.fillna(f"{column} is missing")
        top = keys.value_counts().index[:max_levels]
        return keys.where(keys.isin(top))
    numeric = pd.to_numeric(series, errors="coerce")
    bins = pd.qcut(numeric, 5, duplicates="drop")
    keys = bins.map(lambda b: f"{column} in [{_num(b.left)}, {_num(b.right)}]", na_action="ignore").astype(object)
    return keys.where(numeric.notna(), f"{column} is missing")


def worst_segments(x: pd.DataFrame, errors: np.ndarray, cat_cols: Iterable[str] = (), max_columns: int = 80,
                   min_segment: int = 30, max_levels: int = 20, top: int = 6,
                   min_ratio: float = 1.2) -> list[tuple[float, str]]:
    """Сегменты с ошибкой ≥ min_ratio × средней, по убыванию вклада (средняя ошибка сегмента − общая) × размер."""
    err = pd.Series(errors, index=x.index)
    mean_all = float(err.mean())
    if not mean_all > 0:
        return []
    cat_cols = set(cat_cols)
    min_size = max(min_segment, int(0.005 * len(x)))
    found: list[tuple[float, str]] = []
    for column in list(x.columns)[:max_columns]:
        series = x[column]
        is_cat = column in cat_cols or not (pd.api.types.is_numeric_dtype(series)
                                            and not pd.api.types.is_bool_dtype(series))
        try:
            keys = _segment_keys(series, str(column), is_cat, max_levels)
        except (TypeError, ValueError):
            continue
        stats = err.groupby(keys, observed=True).agg(["mean", "size"])
        for name, (mean, size) in stats.iterrows():
            if size >= min_size and mean >= min_ratio * mean_all:
                found.append(((mean - mean_all) * size, f"{name} (n={int(size)}): {mean / mean_all:.1f}x avg"))
    found.sort(key=lambda item: -item[0])
    return found[:top]


def _regression_lines(info: dict, errors: np.ndarray) -> list[str]:
    y, pred = info["y"], info["pred"]
    squared = np.sort(errors ** 2)[::-1]
    worst = max(1, len(squared) // 10)
    share = squared[:worst].sum() / squared.sum() if squared.sum() > 0 else 0.0
    lines = [f"10% worst rows give {share:.0%} of squared error; RMSE {_num(float(np.sqrt(np.mean(errors ** 2))))}, "
             f"MAE {_num(float(errors.mean()))}"]
    residual = y - pred
    low, high = np.quantile(y, [0.1, 0.9])
    parts = []
    for name, mask in (("top 10%", y >= high), ("bottom 10%", y <= low)):
        if mask.sum() >= 10:
            bias = float(residual[mask].mean())
            if abs(bias) < 0.2 * errors.mean():  # смещение в пределах обычной ошибки — не сигнал
                continue
            word = "underpredicted" if bias > 0 else "overpredicted"
            bound = f"y >= {_num(high)}" if name == "top 10%" else f"y <= {_num(low)}"
            parts.append(f"target {name} ({bound}): mean residual {bias:+.3g} ({word})")
    if parts:
        lines.append("; ".join(parts))
    return lines


def _classification_lines(info: dict, errors: np.ndarray, min_class: int = 10) -> list[str]:
    true, pred, names = info["true"], info["pred"], info["classes"]
    wrong = true != pred
    loss = np.sort(errors)[::-1]
    worst = max(1, len(loss) // 10)
    share = loss[:worst].sum() / loss.sum() if loss.sum() > 0 else 0.0
    lines = [f"misclassified {wrong.mean():.1%} of rows; 10% worst rows give {share:.0%} of the loss (1 - p_true)"]
    recalls = []
    for k, name in enumerate(names):
        mask = true == k
        if mask.sum() >= min_class:
            recalls.append((float((pred[mask] == k).mean()), name, int(mask.sum())))
    recalls = sorted(r for r in recalls if r[0] < 0.95)
    if recalls:
        lines.append("lowest recall: " + ", ".join(f"{name} {r:.2f} (n={n})" for r, name, n in recalls[:3]))
    pairs = pd.Series(list(zip(true[wrong], pred[wrong]))).value_counts().head(3) if wrong.any() else []
    if len(pairs):
        lines.append("top confusions (true -> predicted): "
                     + ", ".join(f"{names[a]} -> {names[b]} ({n})" for (a, b), n in pairs.items()))
    return lines


def analyze_errors(x: pd.DataFrame, y: Any, oof: Any, task_type: str, *, classes: Iterable | None = None,
                   cat_cols: Iterable[str] = (), max_chars: int = 800, max_columns: int = 80,
                   min_segment: int = 30) -> str:
    """Сводка ошибок для промпта планировщика; "" если анализ невозможен (нет oof, не та форма, мусор)."""
    try:
        if oof is None or not isinstance(x, pd.DataFrame) or len(x) == 0 or isinstance(y, pd.DataFrame):
            return ""
        errors, info = row_errors(y, oof, task_type, classes)
        keep = np.isfinite(errors)
        if keep.sum() < max(20, min_segment):
            return ""
        if not keep.all():
            errors = errors[keep]
            info = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in info.items()}
        x = x.reset_index(drop=True).loc[keep]
        lines = (_regression_lines(info, errors) if task_type == "regression"
                 else _classification_lines(info, errors))
        segments = [text for _, text in worst_segments(x, errors, cat_cols, max_columns, min_segment)]

        def render() -> str:
            body = [f"- {line}" for line in lines]
            if segments:
                body.append("- segments with the highest error: " + "; ".join(segments))
            return "\n".join([HEADER, *body])

        text = render()
        while len(text) > max_chars and segments:
            segments.pop()
            text = render()
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
    except Exception:  # анализ — подсказка, а не обязательный шаг: любой сбой → пустая сводка
        return ""
