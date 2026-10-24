"""Свой evaluator агента (P4): K-fold оценка признаков, пока в сборке нет evaluator.py (P1).

Контракт тот же, что у evaluator.evaluate: (spec, build_features, cfg, deadline) -> Candidate,
cfg = {"model": "lightgbm|catboost|xgboost", "params": {...}, "target_transform": None|"log1p"}.
Сплиты и сид фиксированы, поэтому кандидаты сравнимы между собой (в том числе парно по fold_scores).
Метрику считает этот модуль, а не код LLM. Настоящий evaluator P1, когда он есть, приоритетнее.
sklearn и бустинги импортируются внутри функций: модуль импортируется и без них.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --- контракт команды (schemas.py у P1); пока его нет — локальный заменитель ---
try:
    from schemas import Candidate
except ImportError:
    @dataclass
    class Candidate:
        name: str
        cv_score: float
        cv_std: float
        oof: np.ndarray
        test_pred: np.ndarray  # в «сырых» единицах (вероятности для классификации)
        meta: dict = field(default_factory=dict)


def find_file(spec: Any, name: str) -> Path:
    """Файл задачи: поле spec.<stem>_path, каталог задачи или подкаталоги (в task.md пути бывают вложенные)."""
    explicit = getattr(spec, f"{name.split('.')[0]}_path", None)
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    task_dir = Path(spec.task_dir)
    if (task_dir / name).is_file():
        return task_dir / name
    found = sorted((p for p in task_dir.rglob(name) if ".agent_work" not in p.parts),
                   key=lambda p: (len(p.parts), str(p)))
    if not found:
        raise FileNotFoundError(f"{name} not found in {task_dir}")
    return found[0]


# ============================================================ метрики

_METRIC_ALIASES = {"f1_macro": "macro_f1", "macro_f1_score": "macro_f1", "f1_weighted": "weighted_f1",
                   "f1_micro": "micro_f1", "f1_score": "f1", "acc": "accuracy", "balanced_acc": "balanced_accuracy",
                   "log_loss": "logloss", "cross_entropy": "logloss", "auc": "roc_auc", "rocauc": "roc_auc",
                   "roc_auc_score": "roc_auc", "mean_squared_error": "mse", "root_mean_squared_error": "rmse",
                   "mean_absolute_error": "mae", "r2_score": "r2", "mean_absolute_percentage_error": "mape",
                   "root_mean_squared_log_error": "rmsle"}
LABEL_METRICS = ("macro_f1", "weighted_f1", "micro_f1", "f1", "accuracy", "balanced_accuracy")  # по argmax


def metric_name(metric: str) -> str:
    """Ключ метрики: «Macro-F1», «f1_macro», «F1 macro score» → macro_f1."""
    name = re.sub(r"[\s-]+", "_", str(metric).lower().strip())
    return _METRIC_ALIASES.get(name, name)


def label_score(metric: str, y: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Метрика по меткам классов (номера, как в y). f1 у бинарной — по второму из отсортированных классов."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    name = metric_name(metric)
    if name in ("macro_f1", "weighted_f1", "micro_f1"):
        return float(f1_score(y, labels, average=name.split("_")[0]))
    if name == "f1":
        return float(f1_score(y, labels) if n_classes == 2 else f1_score(y, labels, average="macro"))
    if name == "accuracy":
        return float(accuracy_score(y, labels))
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y, labels))
    raise ValueError(f"unknown label metric: {metric}")


def score(metric: str, y: np.ndarray, pred: np.ndarray, regression: bool) -> float:
    """Метрика задачи: регрессия — по предсказаниям, классификация — по вероятностям (n, k)."""
    from sklearn.metrics import (log_loss, mean_absolute_error, mean_absolute_percentage_error,
                                 mean_squared_error, r2_score, roc_auc_score)

    name = metric_name(metric)
    if regression:
        if name == "rmse":
            return float(mean_squared_error(y, pred) ** 0.5)
        if name == "mse":
            return float(mean_squared_error(y, pred))
        if name == "mae":
            return float(mean_absolute_error(y, pred))
        if name == "r2":
            return float(r2_score(y, pred))
        if name == "rmsle":
            return float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(y)) ** 2)))
        if name == "mape":
            return float(mean_absolute_percentage_error(y, pred))
        raise ValueError(f"unknown regression metric: {metric}")
    if name in LABEL_METRICS:
        return label_score(name, y, pred.argmax(axis=1), pred.shape[1])
    if name == "logloss":
        return float(log_loss(y, pred, labels=list(range(pred.shape[1]))))
    if name == "roc_auc":
        return float(roc_auc_score(y, pred[:, 1]) if pred.shape[1] == 2
                     else roc_auc_score(y, pred, multi_class="ovr"))
    raise ValueError(f"unknown classification metric: {metric}")


# ============================================================ оценка

def _to_model_frame(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строки/категории → category с общим словарём train и test, bool → int8."""
    x_train, x_test = x_train.copy(), x_test.copy()
    for col in x_train.columns:
        if x_train[col].dtype == object or isinstance(x_train[col].dtype, pd.CategoricalDtype):
            both = pd.concat([x_train[col], x_test[col]]).astype(object)
            dtype = pd.CategoricalDtype(sorted(both.dropna().astype(str).unique()))
            x_train[col] = x_train[col].astype(object).where(x_train[col].isna(), x_train[col].astype(str)).astype(dtype)
            x_test[col] = x_test[col].astype(object).where(x_test[col].isna(), x_test[col].astype(str)).astype(dtype)
        elif x_train[col].dtype == bool:
            x_train[col], x_test[col] = x_train[col].astype("int8"), x_test[col].astype("int8")
    return x_train, x_test


def _left(deadline: Any) -> float:
    left = getattr(deadline, "left", None) if deadline is not None else None
    try:
        return float(left()) if callable(left) else math.inf
    except Exception:
        return math.inf


def _target_name(spec: Any) -> str:
    target = spec.target
    if isinstance(target, str):
        return target
    target = list(target or [])
    if len(target) != 1:
        raise ValueError(f"LocalEvaluator supports one target column, got {target}")
    return target[0]


class LocalEvaluator:
    """evaluator.evaluate(spec, build_features, cfg, deadline) -> Candidate на фиксированных K фолдах.

    Классификация — StratifiedKFold, вероятности в столбцах по отсортированным строковым меткам (meta["classes"]).
    Дедлайн проверяется между фолдами: не успеваем по оценке оставшихся фолдов → TimeoutError.
    """

    def __init__(self, n_splits: int = 5, seed: int = 42, n_estimators: int = 300,
                 learning_rate: float = 0.05, threads: int | None = None) -> None:
        self.n_splits, self.seed = n_splits, seed
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.threads = threads or os.cpu_count() or 4
        self.calls = 0
        self._cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _load(self, spec: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = str(spec.task_dir)
        if key not in self._cache:
            self._cache[key] = (pd.read_csv(find_file(spec, "train.csv"), low_memory=False),
                                pd.read_csv(find_file(spec, "test.csv"), low_memory=False))
        return self._cache[key]

    def _model(self, name: str, params: dict, regression: bool) -> Any:
        if name == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            base = dict(iterations=self.n_estimators, learning_rate=self.learning_rate, random_seed=self.seed,
                        verbose=0, thread_count=self.threads, allow_writing_files=False)
            return (CatBoostRegressor if regression else CatBoostClassifier)(**{**base, **params})
        if name == "xgboost":
            from xgboost import XGBClassifier, XGBRegressor
            base = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate, random_state=self.seed,
                        n_jobs=self.threads, tree_method="hist", enable_categorical=True)
            return (XGBRegressor if regression else XGBClassifier)(**{**base, **params})
        from lightgbm import LGBMClassifier, LGBMRegressor
        base = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate, random_state=self.seed,
                    n_jobs=self.threads, verbose=-1)
        return (LGBMRegressor if regression else LGBMClassifier)(**{**base, **params})

    def __call__(self, spec: Any, build_features: Any, params: dict, deadline: Any = None) -> Candidate:
        from sklearn.model_selection import KFold, StratifiedKFold

        self.calls += 1
        if spec.task_type not in ("regression", "binary", "multiclass"):
            raise ValueError(f"LocalEvaluator does not support task_type={spec.task_type!r}")
        train, test = self._load(spec)
        target_name = _target_name(spec)
        drop = {spec.id_col, target_name, *(getattr(spec, "drop_cols", None) or [])}
        feats = [c for c in test.columns if c in train.columns and c not in drop]
        spec_dict = {"task_type": spec.task_type, "metric": spec.metric,
                     "cat_cols": list(getattr(spec, "cat_cols", None) or []),
                     "date_cols": list(getattr(spec, "date_cols", None) or []),
                     "text_cols": list(getattr(spec, "text_cols", None) or [])}
        x, x_test = build_features(train[feats].copy(), test[feats].copy(), spec_dict)
        x, x_test = _to_model_frame(x.reset_index(drop=True), x_test.reset_index(drop=True))

        name = (params or {}).get("model") or "lightgbm"
        model_params = dict((params or {}).get("params") or {})
        regression = spec.task_type == "regression"
        target = train[target_name]
        classes: list[str] = []
        if regression:
            y = target.to_numpy(dtype=float)
            log_target = (params or {}).get("target_transform") == "log1p" and y.min() > -1
            splitter, n_classes = KFold(self.n_splits, shuffle=True, random_state=self.seed), 0
            oof, test_pred = np.zeros(len(x)), np.zeros(len(x_test))
        else:
            classes = sorted(target.astype(str).unique())
            y = np.searchsorted(np.array(classes), target.astype(str).to_numpy())
            log_target, n_classes = False, len(classes)
            splitter = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
            oof, test_pred = np.zeros((len(x), n_classes)), np.zeros((len(x_test), n_classes))

        if name == "catboost":
            cats = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
            x, x_test = x.astype({c: str for c in cats}), x_test.astype({c: str for c in cats})
            model_params.setdefault("cat_features", cats)

        fold_scores: list[float] = []
        start = time.monotonic()
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y)):
            if deadline is not None:
                left = _left(deadline)
                need = (time.monotonic() - start) / fold * (self.n_splits - fold) if fold else 0.0
                if left <= 0 or need > left:
                    raise TimeoutError(f"evaluation stopped after {fold}/{self.n_splits} folds: "
                                       f"needs ~{need:.0f}s more, {left:.0f}s left")
            model = self._model(name, model_params, regression)
            y_fit = np.log1p(y[train_idx]) if log_target else y[train_idx]
            model.fit(x.iloc[train_idx], y_fit)
            if regression:
                p_valid, p_test = model.predict(x.iloc[valid_idx]), model.predict(x_test)
                if log_target:
                    p_valid, p_test = np.expm1(p_valid), np.expm1(p_test)
            else:  # колонки вероятностей по model.classes_, если в фолде не было какого-то класса
                cls = np.asarray(model.classes_).astype(int)
                p_valid, p_test = np.zeros((len(valid_idx), n_classes)), np.zeros((len(x_test), n_classes))
                p_valid[:, cls], p_test[:, cls] = model.predict_proba(x.iloc[valid_idx]), model.predict_proba(x_test)
            oof[valid_idx] = p_valid
            test_pred += p_test / self.n_splits
            fold_scores.append(score(spec.metric, y[valid_idx], p_valid, regression))

        meta = {"fold_scores": fold_scores, "model": name, "params": model_params,
                "target_transform": "log1p" if log_target else None, "n_splits": self.n_splits, "seed": self.seed,
                "evaluator": "local"}
        if classes:
            meta["classes"] = classes
        return Candidate(name=f"local_{name}", cv_score=float(np.mean(fold_scores)),
                         cv_std=float(np.std(fold_scores)), oof=oof, test_pred=test_pred, meta=meta)
