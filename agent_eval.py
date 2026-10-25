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
import types
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


def _team_metrics() -> Any:
    """metrics.py (P3) — реестр метрик команды, если он в сборке; иначе None."""
    try:
        import metrics
    except ImportError:
        return None
    return metrics if callable(getattr(metrics, "score", None)) else None


def _team_fallback() -> Any:
    """fallback.py (P3), если он в сборке: его обучение моделей (параметры, ранняя остановка, таймеры)."""
    try:
        import fallback
    except ImportError:
        return None
    return fallback if all(hasattr(fallback, a) for a in ("_fit_predict", "_params", "FallbackConfig")) else None


def score(metric: str, y: np.ndarray, pred: np.ndarray, regression: bool) -> float:
    """Метрика задачи: регрессия — по предсказаниям, классификация — по вероятностям (n, k), y — номера классов.

    Есть metrics.py (P3) — считает он (так же, как fallback и сабмит); иначе — свой набор метрик ниже.
    """
    team = _team_metrics()
    if team is not None:
        try:
            return float(team.score(metric, y, pred))
        except KeyError:  # UnknownMetric — подкласс KeyError: метрики нет в реестре P3, пробуем свой набор
            pass
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

    Фолды: заданные (folds — номер фолда для каждой используемой строки train, rows — позиции этих строк в
    train.csv; так их отдаёт fallback P3: честная схема по периодам/дублям) или (Stratified)KFold с сидом.
    Классификация — вероятности в столбцах по отсортированным строковым меткам (meta["classes"]).
    early_stopping > 0 и есть fallback.py (P3) — модели обучаются как у fallback (его параметры, ранняя остановка
    по фолду, таймер): кандидаты агента сравнимы с кандидатом fallback не только по фолдам, но и по силе моделей.
    Иначе — фиксированные n_estimators без ранней остановки (быстро, для тестов).
    Дедлайн проверяется между фолдами: не успеваем по оценке оставшихся фолдов → TimeoutError.
    """

    def __init__(self, n_splits: int = 5, seed: int = 42, n_estimators: int = 300,
                 learning_rate: float = 0.05, threads: int | None = None, *, folds: Any = None,
                 rows: Any = None, early_stopping: int = 0, max_folds: int | None = None) -> None:
        self.n_splits, self.seed, self.early_stopping = n_splits, seed, early_stopping
        self.max_folds = max_folds  # оценка только на первых фолдах тех же сплитов (быстрый подбор в tuning.py)
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.threads = threads or os.cpu_count() or 4
        self.folds = None if folds is None else np.asarray(folds, dtype=int)
        self.rows = None if rows is None else np.asarray(rows, dtype=int)
        self.calls = 0
        self._cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _trainer(self, name: str, model_params: dict, regression: bool, metric: str, n_rows: int,
                 cats: list[str]) -> Any:
        """Обучение фолда функциями fallback (P3) или None (его нет / early_stopping выключен)."""
        team = _team_fallback() if self.early_stopping > 0 else None
        if team is None:
            return None
        shape = types.SimpleNamespace(y=np.empty(n_rows), task="regression" if regression else "classification",
                                      metric=metric)
        cfg = team.FallbackConfig(seed=self.seed, early_stopping=self.early_stopping)
        params = team._params(name, shape, cfg, self.threads, {k: v for k, v in model_params.items()
                                                              if k != "cat_features"})

        def train(x_tr, y_tr, x_va, y_va, x_te, stop_at):
            return team._fit_predict(name, regression, params, x_tr, y_tr, x_va, y_va, x_te, cats, stop_at,
                                     self.early_stopping)
        return train

    def _splits(self, x: pd.DataFrame, y: np.ndarray, regression: bool) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.folds is not None:
            if len(self.folds) != len(x):
                raise ValueError(f"folds cover {len(self.folds)} rows, train has {len(x)}")
            splits = [(np.flatnonzero(self.folds != f), np.flatnonzero(self.folds == f))
                      for f in np.unique(self.folds)]
        else:
            from sklearn.model_selection import KFold, StratifiedKFold
            splitter = (KFold if regression else StratifiedKFold)(self.n_splits, shuffle=True, random_state=self.seed)
            splits = list(splitter.split(x, y))
        return splits[:self.max_folds] if self.max_folds else splits

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
        self.calls += 1
        if spec.task_type not in ("regression", "binary", "multiclass"):
            raise ValueError(f"LocalEvaluator does not support task_type={spec.task_type!r}")
        train, test = self._load(spec)
        if self.rows is not None:  # строки train, которые использует схема CV (индекс — исходные номера строк)
            train = train.iloc[self.rows]
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
            n_classes = 0
            oof, test_pred = np.zeros(len(x)), np.zeros(len(x_test))
        else:
            classes = sorted(target.astype(str).unique())
            y = np.searchsorted(np.array(classes), target.astype(str).to_numpy())
            log_target, n_classes = False, len(classes)
            oof, test_pred = np.zeros((len(x), n_classes)), np.zeros((len(x_test), n_classes))

        cats: list[str] = []
        if name == "catboost":
            cats = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
            x, x_test = x.astype({c: str for c in cats}), x_test.astype({c: str for c in cats})
        trainer = self._trainer(name, model_params, regression, spec.metric, len(x), cats)
        if name == "catboost" and trainer is None:
            model_params.setdefault("cat_features", cats)

        splits = self._splits(x, y, regression)
        fold_ids = np.zeros(len(x), dtype=int)  # номер фолда каждой строки — для честной постобработки по фолдам
        fold_scores: list[float] = []
        start = time.monotonic()
        for fold, (train_idx, valid_idx) in enumerate(splits):
            fold_ids[valid_idx] = fold
            if deadline is not None:
                left = _left(deadline)
                need = (time.monotonic() - start) / fold * (len(splits) - fold) if fold else 0.0
                if left <= 0 or need > left:
                    raise TimeoutError(f"evaluation stopped after {fold}/{len(splits)} folds: "
                                       f"needs ~{need:.0f}s more, {left:.0f}s left")
            y_fit = np.log1p(y[train_idx]) if log_target else y[train_idx]
            if trainer is not None:  # как у fallback P3: ранняя остановка по фолду и таймер на оставшееся время
                left = _left(deadline)
                stop_at = time.monotonic() + (left / (len(splits) - fold) if math.isfinite(left) else 1e9)
                y_valid = np.log1p(y[valid_idx]) if log_target else y[valid_idx]
                raw_valid, raw_test, _, present = trainer(x.iloc[train_idx], y_fit, x.iloc[valid_idx], y_valid,
                                                          x_test, stop_at)
                cls = np.asarray(present, dtype=int)
            else:
                model = self._model(name, model_params, regression)
                if not regression:  # класса может не быть в обучении фолда (группы дублей): XGBoost — метки 0..k-1
                    present = np.unique(y_fit)
                    model.fit(x.iloc[train_idx], np.searchsorted(present, y_fit))
                    cls = present[np.asarray(model.classes_).astype(int)]
                    raw_valid, raw_test = model.predict_proba(x.iloc[valid_idx]), model.predict_proba(x_test)
                else:
                    model.fit(x.iloc[train_idx], y_fit)
                    raw_valid, raw_test = model.predict(x.iloc[valid_idx]), model.predict(x_test)
            if regression:
                p_valid, p_test = raw_valid, raw_test
                if log_target:
                    p_valid, p_test = np.expm1(p_valid), np.expm1(p_test)
            else:  # колонки вероятностей по классам, которые были в обучении фолда
                p_valid, p_test = np.zeros((len(valid_idx), n_classes)), np.zeros((len(x_test), n_classes))
                if raw_valid is None:  # в обучении фолда один класс
                    p_valid[:, cls[0]], p_test[:, cls[0]] = 1.0, 1.0
                else:
                    p_valid[:, cls], p_test[:, cls] = raw_valid, raw_test
            oof[valid_idx] = p_valid
            test_pred += p_test / len(splits)
            fold_scores.append(score(spec.metric, y[valid_idx], p_valid, regression))

        meta = {"fold_scores": fold_scores, "fold_ids": fold_ids, "model": name, "params": model_params,
                "target_transform": "log1p" if log_target else None, "n_splits": len(splits), "seed": self.seed,
                "evaluator": "local", "folds_from": "given" if self.folds is not None else "kfold",
                "trainer": "fallback" if trainer is not None else "fixed"}
        if classes:
            meta["classes"] = classes
        return Candidate(name=f"local_{name}", cv_score=float(np.mean(fold_scores)),
                         cv_std=float(np.std(fold_scores)), oof=oof, test_pred=test_pred, meta=meta)


# ============================================================ оценка «поверх fallback»

def _safe_name(name: str) -> str:
    return "agent__" + (re.sub(r"[^\w]", "_", str(name)).strip("_") or "col")


class RecipeEvaluator:
    """evaluator.evaluate(spec, build_features, cfg, deadline) -> Candidate на рецепте лучшего кандидата fallback.

    best.meta["context"] (fallback.FallbackContext) — готовые матрицы fallback, фолды, обученные модели и веса
    членов лучшего кандидата. К матрицам добавляются ТОЛЬКО новые колонки из build_features (имена, которых нет во
    входе; префикс agent__), члены переобучаются fallback.cv_model с теми же параметрами, лог-таргетом,
    исключениями и признаками сущности, смешиваются с теми же весами, порог/веса классов — честно
    (fallback.make_candidate). cv_score и fold_scores — эффективные (как у fallback.clearly_better), поэтому без
    новых колонок кандидат воспроизводит лучший fallback, а агент улучшает именно его. cfg (модель/параметры кода
    LLM) не используется: модели — у рецепта. recipe_mode — признак для агента («только новые колонки»).
    """

    recipe_mode = True

    def __init__(self, best: Any, threads: int | None = None, decision_seconds: float = 20.0) -> None:
        ctx = (getattr(best, "meta", None) or {}).get("context")
        if ctx is None:
            raise ValueError("fallback candidate has no context")
        self.ctx, self.decision_seconds = ctx, decision_seconds
        self.runs = {r.name: r for r in ctx.runs}
        self.members = {name: float(w) for name, w in ctx.members.items() if name in self.runs and w > 0}
        if not self.members:
            raise ValueError("best fallback candidate has no trained members")
        self.threads = threads
        self.calls = 0
        self._inputs: tuple[pd.DataFrame, pd.DataFrame, dict] | None = None

    def _raw(self, spec: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Сырые train/test, которые получает build_features (строки — как у fallback, индекс — номер строки)."""
        if self._inputs is None:
            train = pd.read_csv(find_file(spec, "train.csv"), low_memory=False)
            test = pd.read_csv(find_file(spec, "test.csv"), low_memory=False)
            rows = self.ctx.data.rows
            if rows is not None:
                train = train.iloc[np.asarray(rows, dtype=int)]
            drop = {spec.id_col, _target_name(spec), *(getattr(spec, "drop_cols", None) or [])}
            feats = [c for c in test.columns if c in train.columns and c not in drop]
            spec_dict = {"task_type": spec.task_type, "metric": spec.metric,
                         "cat_cols": list(getattr(spec, "cat_cols", None) or []),
                         "date_cols": list(getattr(spec, "date_cols", None) or []),
                         "text_cols": list(getattr(spec, "text_cols", None) or [])}
            self._inputs = (train[feats], test[feats].reset_index(drop=True), spec_dict)
        return self._inputs

    def __call__(self, spec: Any, build_features: Any, params: dict | None = None, deadline: Any = None) -> Candidate:
        import copy as copy_module

        import fallback

        self.calls += 1
        ctx = self.ctx
        train, test, spec_dict = self._raw(spec)
        x_tr, x_te = build_features(train.copy(), test.copy(), spec_dict)
        new = [c for c in x_tr.columns if c not in train.columns and c in x_te.columns]
        extra_tr, extra_te = _to_model_frame(x_tr[new].reset_index(drop=True), x_te[new].reset_index(drop=True))
        names = {c: _safe_name(c) for c in new}
        extra_tr, extra_te = extra_tr.rename(columns=names), extra_te.rename(columns=names)
        cat_new = [c for c in extra_tr.columns if isinstance(extra_tr[c].dtype, pd.CategoricalDtype)]

        data = copy_module.copy(ctx.data)
        data.frames = {}
        for name in self.members:
            model = self.runs[name].model
            key = "lightgbm" if model in ("lightgbm", "xgboost") else model
            if key in data.frames:
                continue
            x, x_test, cats = ctx.data.frame(model)
            add_tr, add_te = extra_tr, extra_te
            if key == "catboost" and cat_new:
                add_tr, add_te = add_tr.astype({c: str for c in cat_new}), add_te.astype({c: str for c in cat_new})
            data.frames[key] = (pd.concat([x.reset_index(drop=True), add_tr], axis=1),
                                pd.concat([x_test.reset_index(drop=True), add_te], axis=1), [*cats, *cat_new])

        threads = self.threads or fallback.n_threads()
        oof = test_pred = None
        for i, (name, weight) in enumerate(self.members.items()):
            run = self.runs[name]
            left = _left(deadline)
            if left <= 0:
                raise TimeoutError(f"evaluation stopped before member {name}")
            stop_at = time.monotonic() + (left / (len(self.members) - i) if math.isfinite(left) else 1e9)
            fitted = fallback.cv_model(data, name, run.model, ctx.config, stop_at, transform=run.transform,
                                       extra=dict(run.params), threads=threads, exclude=run.exclude,
                                       entity=run.entity)
            oof = weight * fitted.oof if oof is None else oof + weight * fitted.oof
            test_pred = weight * fitted.test_pred if test_pred is None else test_pred + weight * fitted.test_pred
        scores = fallback._fold_scores(data, oof)
        cand = fallback.make_candidate(data, "agent_recipe", oof, test_pred, scores,
                                       {"model": "recipe", "members": dict(self.members)}, ctx.config,
                                       self.decision_seconds)
        eff = [float(s) for s in cand.meta["fold_scores_eff"]]
        cand.meta.update(fold_scores_raw=cand.meta["fold_scores"], fold_scores=eff, fold_ids=data.folds,
                         new_columns=new, evaluator="recipe")
        cand.cv_score, cand.cv_std = float(np.mean(eff)), float(np.std(eff))
        return cand
