"""Сабмит (P3): сверка с sample_submission.csv и запись. Единственная точка записи submission.csv.

write(spec, pred) принимает предсказания в порядке строк test.csv в «сырых» единицах (как Candidate.test_pred):
регрессия — значения, классификация — вероятности (n, k) по classes (порядок metrics.classes_of) или готовые коды.
Что именно попадёт в файл (значения / метки / вероятность положительного класса / колонки-вероятности по классам),
решает output_kind() по метрике и по образцу. Строки — в порядке sample_submission.csv, колонки — как в нём же.
Запись атомарная: временный файл рядом + os.replace, поэтому убитый процесс не оставляет битый submission.csv.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import metrics

log = logging.getLogger("submission")

FILENAME = "submission.csv"


class SubmissionError(ValueError):
    """Сабмит не удаётся привести к формату sample_submission.csv."""


# ---------------------------------------------------------------- файлы задачи

def task_file(spec: Any, name: str) -> Path:
    """Файл задачи: поле spec.<stem>_path, каталог задачи или его подкаталоги (как agent._find_file у P4)."""
    explicit = getattr(spec, f"{name.split('.')[0]}_path", None)
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    task_dir = Path(spec.task_dir)
    if (task_dir / name).is_file():
        return task_dir / name
    found = sorted((p for p in task_dir.rglob(name) if not any(part.startswith(".") for part in p.parts[-3:])),
                   key=lambda p: (len(p.parts), str(p)))
    if not found:
        raise FileNotFoundError(f"{name} not found in {task_dir}")
    return found[0]


def output_path(spec: Any) -> Path:
    return Path(spec.task_dir) / FILENAME


def targets_of(spec: Any) -> list[str]:
    target = spec.target
    return [str(t) for t in target] if isinstance(target, (list, tuple)) else [str(target)]


def _stamp(path: Path) -> tuple[str, float, int]:
    stat = path.stat()
    return str(path), stat.st_mtime, stat.st_size


@lru_cache(maxsize=8)
def _read_columns(stamp: tuple[str, float, int], columns: tuple[str, ...] | None, as_str: bool) -> pd.DataFrame:
    kw: dict[str, Any] = {"dtype": str, "keep_default_na": False} if as_str else {"low_memory": False}
    return pd.read_csv(stamp[0], usecols=list(columns) if columns else None, **kw)


def _sample(spec: Any) -> pd.DataFrame:
    return _read_columns(_stamp(task_file(spec, "sample_submission.csv")), None, True).copy()


def _train_target(spec: Any, column: str) -> pd.Series:
    return _read_columns(_stamp(task_file(spec, "train.csv")), (column,), False)[column].dropna()


def _test_ids(spec: Any, id_col: str) -> np.ndarray | None:
    path = task_file(spec, "test.csv")
    header = pd.read_csv(path, nrows=0).columns
    if id_col not in header:
        return None
    return _read_columns(_stamp(path), (id_col,), True)[id_col].map(_norm_id).to_numpy()


def _norm_id(value: Any) -> str:
    """Сравнение id без учёта форматирования: «12», « 12 », «12.0» — один id."""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() and abs(number) < 2 ** 53 else text


# ---------------------------------------------------------------- что пишем

def _value_columns(spec: Any, sample: pd.DataFrame) -> tuple[str | None, list[str]]:
    id_col = getattr(spec, "id_col", None)
    id_col = id_col if id_col in sample.columns else None
    if id_col is None and len(sample.columns) > 1 and targets_of(spec)[0] not in sample.columns[:1]:
        id_col = sample.columns[0]  # id в образце называется иначе, чем в spec: первая колонка — идентификатор
    return id_col, [c for c in sample.columns if c != id_col]


def _looks_like_proba(values: pd.Series) -> bool:
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.isna().any() or not len(numbers):
        return False
    return bool(numbers.between(0, 1).all() and not (numbers == numbers.round()).all())


def class_list(spec: Any, classes: list | None = None) -> list[str]:
    if classes is not None:
        return [metrics.label_str(c) for c in classes]
    return metrics.classes_of(_train_target(spec, targets_of(spec)[0]))


def output_kind(spec: Any, classes: list | None = None) -> str:
    """values (регрессия) | labels | proba (вероятность положительного класса) | proba_columns (колонка на класс)."""
    if getattr(spec, "task_type", "") == "regression":
        return "values"
    sample = _sample(spec)
    _, value_cols = _value_columns(spec, sample)
    targets = targets_of(spec)
    if len(targets) == 1 and len(value_cols) > 1:
        names = {metrics.label_str(c) for c in value_cols}
        if names == set(class_list(spec, classes)):
            return "proba_columns"
    try:
        needs_proba = metrics.get(spec.metric).needs_proba
    except (metrics.UnknownMetric, AttributeError, TypeError):
        needs_proba = all(_looks_like_proba(sample[c]) for c in value_cols)
    if needs_proba and len(targets) == 1 and len(class_list(spec, classes)) > 2:
        log.warning("метрика %s требует вероятностей, но в образце одна колонка на %d классов — пишем метки",
                    spec.metric, len(class_list(spec, classes)))
        return "labels"
    return "proba" if needs_proba else "labels"


# ---------------------------------------------------------------- сборка

def _row_order(spec: Any, sample: pd.DataFrame, id_col: str | None, n_pred: int) -> np.ndarray:
    """pos[i] — строка предсказаний (порядок test.csv) для i-й строки образца."""
    test_ids = _test_ids(spec, id_col) if id_col is not None and id_col == getattr(spec, "id_col", None) else None
    if test_ids is not None:
        if len(test_ids) != n_pred:
            raise SubmissionError(f"predictions have {n_pred} rows, test.csv has {len(test_ids)}")
        index = pd.Index(test_ids)
        if index.is_unique:
            pos = index.get_indexer(sample[id_col].map(_norm_id))
            if (pos >= 0).all():
                return pos
            log.warning("%d id из образца нет в test.csv", int((pos < 0).sum()))
    if len(sample) != n_pred:
        raise SubmissionError(f"predictions have {n_pred} rows, sample_submission.csv has {len(sample)}")
    log.warning("сопоставляю строки сабмита с test.csv по порядку, а не по id")
    return np.arange(n_pred)


def _fill(values: np.ndarray, fill: float, what: str) -> np.ndarray:
    bad = ~np.isfinite(values)
    if bad.any():
        log.warning("%s: %d нечисловых/бесконечных значений заменено на %.6g", what, int(bad.sum()), fill)
        values = np.where(bad, fill, values)
    return values


def _regression_frame(spec: Any, pred: np.ndarray, value_cols: list[str]) -> dict[str, np.ndarray]:
    targets = targets_of(spec)
    pred = pred.reshape(len(pred), -1).astype(float)
    if pred.shape[1] != len(value_cols):
        raise SubmissionError(f"predictions have {pred.shape[1]} columns, sample expects {len(value_cols)}")
    by_name = set(value_cols) == set(targets) and len(targets) == pred.shape[1]
    out = {}
    for j, col in enumerate(value_cols):
        k = targets.index(col) if by_name else j
        try:
            fill = float(pd.to_numeric(_train_target(spec, targets[k]), errors="coerce").median())
        except (ValueError, KeyError, FileNotFoundError):
            fill = 0.0
        out[col] = _fill(pred[:, k], fill if np.isfinite(fill) else 0.0, col)
    return out


def _proba_matrix(pred: np.ndarray, n_classes: int) -> np.ndarray:
    if pred.ndim == 1:
        if pred.dtype.kind in "iu" or (np.all(np.isfinite(pred)) and pred.max(initial=0) > 1):
            onehot = np.zeros((len(pred), n_classes))  # пришли коды классов — делаем «уверенные» вероятности
            onehot[np.arange(len(pred)), pred.astype(int)] = 1.0
            return onehot
        if n_classes != 2:
            raise SubmissionError(f"1-D probabilities for {n_classes} classes")
        pred = np.column_stack([1.0 - pred, pred])
    if pred.shape[1] != n_classes:
        raise SubmissionError(f"predictions have {pred.shape[1]} columns, task has {n_classes} classes")
    proba = pred.astype(float)
    bad = ~np.isfinite(proba).all(axis=1)
    if bad.any():
        log.warning("%d строк вероятностей с NaN/inf заменены равномерным распределением", int(bad.sum()))
        proba[bad] = 1.0 / n_classes
    return np.clip(proba, 0.0, 1.0)


def _label_text(labels: np.ndarray, sample_values: pd.Series) -> np.ndarray:
    """Метки в представлении образца: True/False ↔ 1/0, если образец числовой."""
    present = set(sample_values.astype(str).str.strip().unique())
    if set(labels) <= {"True", "False"} and present and present <= {"0", "1", "0.0", "1.0"}:
        return np.where(labels == "True", "1", "0")
    return labels


def build(spec: Any, pred: Any, *, classes: list | None = None, decision: dict | None = None) -> pd.DataFrame:
    """Таблица сабмита в формате образца (без записи на диск)."""
    sample = _sample(spec)
    id_col, value_cols = _value_columns(spec, sample)
    pred = np.asarray(pred)
    if pred.ndim == 0 or len(pred) == 0:
        raise SubmissionError("empty predictions")
    pos = _row_order(spec, sample, id_col, len(pred))
    pred = pred[pos]
    out = sample[[id_col]].copy() if id_col is not None else pd.DataFrame(index=sample.index)

    kind = output_kind(spec, classes)
    if kind == "values":
        for col, values in _regression_frame(spec, pred, value_cols).items():
            out[col] = values
        return out[sample.columns.tolist()]

    targets = targets_of(spec)
    if len(targets) > 1:  # мультилейбл: по колонке вероятность «1» на каждый таргет
        proba = pred.reshape(len(pred), -1).astype(float)
        if proba.shape[1] != len(value_cols):
            raise SubmissionError(f"predictions have {proba.shape[1]} columns, sample expects {len(value_cols)}")
        for j, col in enumerate(value_cols):
            p = np.clip(_fill(proba[:, j], 0.5, col), 0, 1)
            out[col] = p if kind == "proba" else (p >= 0.5).astype(int)
        return out[sample.columns.tolist()]

    names = class_list(spec, classes)
    if pred.ndim == 1 and pred.dtype.kind in "OUSb":  # пришли готовые метки в исходном виде
        pred = metrics.encode(pred, names)
    proba = _proba_matrix(pred, len(names))
    if kind == "proba_columns":
        for col in value_cols:
            out[col] = proba[:, names.index(metrics.label_str(col))]
    elif kind == "proba":
        out[value_cols[0]] = proba[:, -1]  # положительный класс — последний (pos_label sklearn по умолчанию)
    else:
        codes = metrics.to_labels(proba, decision)
        out[value_cols[0]] = _label_text(np.asarray(names, dtype=object)[codes].astype(str), sample[value_cols[0]])
    return out[sample.columns.tolist()]


# ---------------------------------------------------------------- проверка и запись

def validate(spec: Any, frame: pd.DataFrame, *, classes: list | None = None) -> list[str]:
    """Расхождения с sample_submission.csv; пустой список — файл можно отдавать."""
    problems: list[str] = []
    sample = _sample(spec)
    if list(frame.columns) != list(sample.columns):
        return [f"columns {list(frame.columns)} != sample {list(sample.columns)}"]
    if len(frame) != len(sample):
        problems.append(f"{len(frame)} rows, sample has {len(sample)}")
    id_col, value_cols = _value_columns(spec, sample)
    if id_col is not None:
        ids = frame[id_col].map(_norm_id)
        if ids.duplicated().any():
            problems.append(f"duplicated ids in {id_col}")
        if set(ids) != set(sample[id_col].map(_norm_id)):
            problems.append(f"ids in {id_col} differ from sample")
    for col in value_cols:
        values = frame[col]
        if values.isna().any() or (values.astype(str).str.strip() == "").any():
            problems.append(f"{col}: missing values")
    kind = output_kind(spec, classes)
    if kind in ("values", "proba", "proba_columns"):
        for col in value_cols:
            numbers = pd.to_numeric(frame[col], errors="coerce")
            if not np.isfinite(numbers.to_numpy(dtype=float)).all():
                problems.append(f"{col}: non-numeric or infinite values")
            elif kind != "values" and not numbers.between(0, 1).all():
                problems.append(f"{col}: probabilities outside [0, 1]")
    elif len(targets_of(spec)) == 1:
        allowed = set(class_list(spec, classes))
        labels = set(frame[value_cols[0]].map(metrics.label_str))
        if allowed <= {"True", "False"} and labels <= {"0", "1"}:
            labels = {"True" if v == "1" else "False" for v in labels}
        unknown = labels - allowed
        if unknown:
            problems.append(f"{value_cols[0]}: unknown labels {sorted(unknown)[:5]}")
    return problems


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write(spec: Any, pred: Any, *, classes: list | None = None, decision: dict | None = None,
          path: Path | None = None) -> Path:
    """Проверяет и атомарно пишет submission.csv рядом с task.md; расхождение с образцом → SubmissionError."""
    frame = build(spec, pred, classes=classes, decision=decision)
    problems = validate(spec, frame, classes=classes)
    if problems:
        raise SubmissionError("; ".join(problems))
    target = Path(path) if path is not None else output_path(spec)
    _atomic_write(frame, target)
    log.info("записан %s: %d строк, колонки %s", target, len(frame), list(frame.columns))
    return target


def write_candidate(spec: Any, cand: Any, *, path: Path | None = None) -> Path:
    """Сабмит из Candidate: классы и пост-обработка (порог/веса классов) берутся из cand.meta."""
    meta = getattr(cand, "meta", None) or {}
    return write(spec, cand.test_pred, classes=meta.get("classes"), decision=meta.get("decision"), path=path)


def write_dummy(spec: Any, *, overwrite: bool = False) -> Path | None:
    """Мгновенная страховка: константный прогноз (медиана/среднее или априорные вероятности классов).

    Не перезаписывает существующий файл, если overwrite=False. Любой сбой → None (страховка не должна ронять процесс).
    """
    target = output_path(spec)
    if target.exists() and not overwrite:
        return None
    try:
        sample = _sample(spec)
        id_col, _ = _value_columns(spec, sample)
        ids = _test_ids(spec, id_col) if id_col is not None and id_col == getattr(spec, "id_col", None) else None
        n = len(ids) if ids is not None else len(sample)
        targets = targets_of(spec)
        try:
            metric = metrics.resolve(spec.metric)
        except (metrics.UnknownMetric, AttributeError, TypeError):
            metric = ""
        if spec.task_type == "regression":
            columns = []
            for t in targets:
                y = pd.to_numeric(_train_target(spec, t), errors="coerce").dropna()
                center = y.median() if metric in ("mae", "medae", "mape", "wape", "smape") else y.mean()
                columns.append(np.full(n, float(center) if len(y) else 0.0))
            pred = np.column_stack(columns) if len(columns) > 1 else columns[0]
        elif len(targets) > 1:
            pred = np.column_stack([np.full(n, float(pd.to_numeric(_train_target(spec, t), errors="coerce").mean()))
                                    for t in targets])
        else:
            y = _train_target(spec, targets[0])
            names = metrics.classes_of(y)
            prior = np.bincount(metrics.encode(y, names), minlength=len(names)) / max(len(y), 1)
            pred = np.tile(prior, (n, 1))
        return write(spec, pred)
    except Exception as error:  # noqa: BLE001 — страховка не должна ронять запуск
        log.warning("константный сабмит не записан: %s: %s", type(error).__name__, error)
        return None


def check_file(spec: Any, path: Path | None = None) -> list[str]:
    """Проверка уже записанного файла (например, перед выходом из solution.py)."""
    target = Path(path) if path is not None else output_path(spec)
    if not target.is_file():
        return [f"{target} does not exist"]
    try:
        frame = pd.read_csv(target, dtype=str, keep_default_na=False)
    except Exception as error:  # noqa: BLE001
        return [f"{target} is unreadable: {error}"]
    return validate(spec, frame.replace("", np.nan))
