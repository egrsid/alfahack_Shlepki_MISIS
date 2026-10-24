"""Реестр метрик (P3): имя → функция, направление, нужны ли вероятности. Единый score() для fallback, evaluator и агента.

Соглашения (общие с evaluator P1 и агентом P4):
- регрессия: y_pred — 1-D массив значений;
- классификация: y_true — коды классов 0..k-1 в порядке classes_of(), y_pred — матрица вероятностей (n, k);
  для бинарной допускается 1-D вероятность класса с кодом 1; 1-D целочисленный массив — уже готовые метки;
- метрики по меткам (F1, accuracy, ...) из вероятностей берут argmax либо decision — порог (бинарная)
  или веса классов (мультикласс), подобранные fit_decision() по OOF.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn import metrics as skm


class UnknownMetric(KeyError):
    """Имя метрики не найдено в реестре (см. names())."""


@dataclass(frozen=True)
class Metric:
    name: str
    func: Callable[[np.ndarray, np.ndarray], float]  # (y_true, y_pred) уже в нужной форме: значения / метки / вероятности
    greater_is_better: bool
    task: str               # "regression" | "classification"
    needs_proba: bool       # считается по вероятностям (AUC, logloss) → в сабмит идут вероятности
    tunable: bool = False   # метрика по меткам, которую стоит доводить порогом/весами классов (fit_decision)


# ---------------------------------------------------------------- регрессия

def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _msle(y: np.ndarray, p: np.ndarray) -> float:
    # отрицательные значения обрезаем нулём: так считают kaggle-подобные проверялки, иначе log1p(<-1) = nan
    return float(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2))


def _smape(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(y) + np.abs(p)
    ratio = np.divide(2.0 * np.abs(y - p), denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return float(np.mean(ratio))


def _wape(y: np.ndarray, p: np.ndarray) -> float:
    total = float(np.sum(np.abs(y)))
    return float(np.sum(np.abs(y - p)) / total) if total > 0 else float(np.sum(np.abs(y - p)))


# ---------------------------------------------------------------- классификация по вероятностям

def _proba_matrix(p: np.ndarray) -> np.ndarray:
    return np.column_stack([1.0 - p, p]) if p.ndim == 1 else p


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    if p.ndim == 1 or p.shape[1] == 2:
        return float(skm.roc_auc_score(y, p if p.ndim == 1 else p[:, 1]))
    # макро one-vs-rest только по классам, у которых в y есть и положительные, и отрицательные объекты
    scores = [skm.roc_auc_score(y == k, p[:, k]) for k in range(p.shape[1]) if 0 < np.sum(y == k) < len(y)]
    return float(np.mean(scores)) if scores else float("nan")


def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if p.ndim == 1 or p.shape[1] == 2:
        return float(skm.average_precision_score(y, p if p.ndim == 1 else p[:, 1]))
    scores = [skm.average_precision_score(y == k, p[:, k]) for k in range(p.shape[1]) if np.any(y == k)]
    return float(np.mean(scores)) if scores else float("nan")


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    proba = np.clip(_proba_matrix(p).astype(float), 1e-15, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(proba[np.arange(len(y)), y])))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    proba = _proba_matrix(p).astype(float)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1))) if proba.shape[1] > 2 else \
        float(np.mean((proba[:, 1] - onehot[:, 1]) ** 2))


# ---------------------------------------------------------------- классификация по меткам

def _f1(average: str) -> Callable[[np.ndarray, np.ndarray], float]:
    def func(y: np.ndarray, p: np.ndarray) -> float:
        return float(skm.f1_score(y, p, average=average, zero_division=0))
    return func


def _binary(func: Callable) -> Callable[[np.ndarray, np.ndarray], float]:
    # бинарные F1/precision/recall: положительный класс — код 1 (как pos_label по умолчанию в sklearn)
    def wrapped(y: np.ndarray, p: np.ndarray) -> float:
        if max(int(np.max(y, initial=0)), int(np.max(p, initial=0))) > 1:
            return float(func(y, p, average="macro", zero_division=0))
        return float(func(y, p, zero_division=0))
    return wrapped


REGISTRY: dict[str, Metric] = {m.name: m for m in [
    # регрессия
    Metric("rmse", _rmse, False, "regression", False),
    Metric("mse", lambda y, p: float(np.mean((y - p) ** 2)), False, "regression", False),
    Metric("mae", lambda y, p: float(np.mean(np.abs(y - p))), False, "regression", False),
    Metric("medae", lambda y, p: float(np.median(np.abs(y - p))), False, "regression", False),
    Metric("rmsle", lambda y, p: math.sqrt(_msle(y, p)), False, "regression", False),
    Metric("msle", _msle, False, "regression", False),
    Metric("mape", lambda y, p: float(skm.mean_absolute_percentage_error(y, p)), False, "regression", False),
    Metric("smape", _smape, False, "regression", False),
    Metric("wape", _wape, False, "regression", False),
    Metric("r2", lambda y, p: float(skm.r2_score(y, p)), True, "regression", False),
    # классификация по меткам
    Metric("accuracy", lambda y, p: float(np.mean(y == p)), True, "classification", False),
    Metric("balanced_accuracy", lambda y, p: float(skm.balanced_accuracy_score(y, p)), True, "classification", False,
           tunable=True),
    Metric("macro_f1", _f1("macro"), True, "classification", False, tunable=True),
    Metric("weighted_f1", _f1("weighted"), True, "classification", False, tunable=True),
    Metric("micro_f1", _f1("micro"), True, "classification", False),
    Metric("f1", _binary(skm.f1_score), True, "classification", False, tunable=True),
    Metric("precision", _binary(skm.precision_score), True, "classification", False),
    Metric("recall", _binary(skm.recall_score), True, "classification", False),
    Metric("mcc", lambda y, p: float(skm.matthews_corrcoef(y, p)), True, "classification", False, tunable=True),
    Metric("kappa", lambda y, p: float(skm.cohen_kappa_score(y, p)), True, "classification", False, tunable=True),
    Metric("qwk", lambda y, p: float(skm.cohen_kappa_score(y, p, weights="quadratic")), True, "classification", False,
           tunable=True),
    # классификация по вероятностям
    Metric("roc_auc", _roc_auc, True, "classification", True),
    Metric("gini", lambda y, p: 2.0 * _roc_auc(y, p) - 1.0, True, "classification", True),
    Metric("pr_auc", _pr_auc, True, "classification", True),
    Metric("logloss", _logloss, False, "classification", True),
    Metric("brier", _brier, False, "classification", True),
]}

# синонимы: ключ — имя без регистра, пробелов, дефисов и подчёркиваний
_ALIASES = {
    "rmse": "rmse", "rootmeansquarederror": "rmse", "rootmse": "rmse",
    "mse": "mse", "meansquarederror": "mse", "l2": "mse",
    "mae": "mae", "meanabsoluteerror": "mae", "l1": "mae",
    "medae": "medae", "medianabsoluteerror": "medae",
    "rmsle": "rmsle", "rootmeansquaredlogerror": "rmsle", "rootmeansquaredlogarithmicerror": "rmsle",
    "msle": "msle", "meansquaredlogerror": "msle",
    "mape": "mape", "meanabsolutepercentageerror": "mape",
    "smape": "smape", "symmetricmape": "smape", "symmetricmeanabsolutepercentageerror": "smape",
    "wape": "wape", "wmape": "wape",
    "r2": "r2", "r2score": "r2", "rsquared": "r2", "coefficientofdetermination": "r2",
    "accuracy": "accuracy", "acc": "accuracy", "accuracyscore": "accuracy",
    "balancedaccuracy": "balanced_accuracy", "balancedacc": "balanced_accuracy",
    "macrof1": "macro_f1", "f1macro": "macro_f1", "macrof1score": "macro_f1", "f1scoremacro": "macro_f1",
    "macroaveragedf1": "macro_f1", "macroavgf1": "macro_f1", "fmacro": "macro_f1",
    "weightedf1": "weighted_f1", "f1weighted": "weighted_f1", "weightedf1score": "weighted_f1",
    "microf1": "micro_f1", "f1micro": "micro_f1",
    "f1": "f1", "f1score": "f1", "fscore": "f1", "fmeasure": "f1", "binaryf1": "f1",
    "precision": "precision", "recall": "recall", "sensitivity": "recall",
    "mcc": "mcc", "matthewscorrcoef": "mcc", "matthewscorrelationcoefficient": "mcc",
    "kappa": "kappa", "cohenkappa": "kappa", "cohenskappa": "kappa",
    "qwk": "qwk", "quadraticweightedkappa": "qwk", "quadratickappa": "qwk",
    "rocauc": "roc_auc", "auc": "roc_auc", "aucroc": "roc_auc", "rocaucscore": "roc_auc", "areaundercurve": "roc_auc",
    "gini": "gini", "ginicoefficient": "gini", "normalizedgini": "gini",
    "prauc": "pr_auc", "averageprecision": "pr_auc", "aucpr": "pr_auc",
    "logloss": "logloss", "logarithmicloss": "logloss", "crossentropy": "logloss", "binarycrossentropy": "logloss",
    "multiclasslogloss": "logloss", "mlogloss": "logloss", "negloglikelihood": "logloss",
    "brier": "brier", "brierscore": "brier",
}

_DEFAULTS = {"regression": "rmse", "binary": "roc_auc", "multiclass": "macro_f1", "multilabel": "macro_f1"}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower().replace("²", "2"))


def resolve(name: str) -> str:
    """Каноническое имя метрики по любому написанию («Macro F1», «f1-macro», «ROC AUC», ...); иначе UnknownMetric."""
    if name in REGISTRY:
        return name
    canonical = _ALIASES.get(_key(name))
    if canonical is None:
        raise UnknownMetric(f"unknown metric {name!r}; known: {', '.join(REGISTRY)}")
    return canonical


def get(name: str) -> Metric:
    return REGISTRY[resolve(name)]


def names() -> list[str]:
    return list(REGISTRY)


def default_for(task_type: str) -> str:
    return _DEFAULTS.get(task_type, "rmse")


def greater_is_better(name: str) -> bool:
    return get(name).greater_is_better


def is_better(name: str, new: float, old: float | None, eps: float = 0.0) -> bool:
    """new лучше old с учётом направления метрики (old=None или не число → True)."""
    if old is None or not math.isfinite(old):
        return math.isfinite(new)
    if not math.isfinite(new):
        return False
    return new > old + eps if greater_is_better(name) else new < old - eps


# ---------------------------------------------------------------- метрика в тексте задачи

_TEXT_PATTERNS: list[tuple[str, str]] = [  # порядок важен только для одинаковых позиций
    ("rmsle", r"\brmsle\b|root[\s-]*mean[\s-]*squared[\s-]*log(arithmic)?[\s-]*error"),
    ("msle", r"\bmsle\b|mean[\s-]*squared[\s-]*log(arithmic)?[\s-]*error"),
    ("rmse", r"\brmse\b|root[\s-]*mean[\s-]*squared[\s-]*error|корень\s+из\s+средн\w*\s+квадрат\w*"
             r"|среднеквадратичн\w*\s+(ошибк|отклонени)\w*"),
    ("mse", r"\bmse\b|mean[\s-]*squared[\s-]*error|средн\w*\s+квадрат\w*\s+ошибк\w*"),
    ("smape", r"\bsmape\b|symmetric\s+mean\s+absolute\s+percentage"),
    ("mape", r"\bmape\b|mean[\s-]*absolute[\s-]*percentage[\s-]*error|средн\w*\s+абсолютн\w*\s+процентн\w*"),
    ("wape", r"\bw-?mape\b|\bwape\b"),
    ("medae", r"\bmedae\b|median[\s-]*absolute[\s-]*error"),
    ("mae", r"\bmae\b|mean[\s-]*absolute[\s-]*error|средн\w*\s+абсолютн\w*\s+ошибк\w*"),
    ("r2", r"\br\s?(\^\s?2|²|2)(\s?score)?\b|r-squared|коэффициент\w*\s+детерминации"),
    ("macro_f1", r"macro[\s_-]*(averaged[\s_-]*)?f[\s_-]?1|f[\s_-]?1[\s_-]*(score[\s_-]*)?\(?macro|макро[\s_-]*f[\s_-]?1"
                 r"|f[\s_-]?1[\s_-]*макро"),
    ("weighted_f1", r"weighted[\s_-]*f[\s_-]?1|f[\s_-]?1[\s_-]*(score[\s_-]*)?\(?weighted|взвешенн\w*\s+f[\s_-]?1"),
    ("micro_f1", r"micro[\s_-]*f[\s_-]?1|f[\s_-]?1[\s_-]*(score[\s_-]*)?\(?micro|микро[\s_-]*f[\s_-]?1"),
    ("f1", r"\bf[\s_-]?1([\s_-]*score)?\b|\bf[\s_-]*(мера|measure)\b"),
    ("balanced_accuracy", r"balanced[\s_-]*accuracy|сбалансированн\w*\s+(точност|accuracy)\w*"),
    ("accuracy", r"\baccuracy\b|доля\s+(правильн|верн)\w*\s+ответ\w*"),
    ("qwk", r"\bqwk\b|quadratic(ally)?[\s-]*weighted[\s-]*kappa"),
    ("kappa", r"cohen'?s?[\s-]*kappa|\bkappa\b"),
    ("mcc", r"\bmcc\b|matthews"),
    ("gini", r"\bgini\b|джини"),
    ("pr_auc", r"\bpr[\s_-]*auc\b|average[\s_-]*precision|auc[\s_-]*pr\b"),
    ("roc_auc", r"roc[\s_-]*auc|auc[\s_-]*roc|\bauc\b|area\s+under\s+(the\s+)?(roc\s+)?curve"),
    ("logloss", r"log[\s_-]*loss|cross[\s-]*entropy|кросс[\s-]*энтропи\w*|логарифмическ\w*\s+потер\w*"),
    ("brier", r"\bbrier\b"),
    ("precision", r"\bprecision\b"),
    ("recall", r"\brecall\b|\bполнот\w*"),
]
_PRIMARY = re.compile(r"основн\w*|главн\w*|целев\w*\s+метрик\w*|primary|main\s+metric|evaluat\w*|оценива\w*|метрик\w*|metric",
                      re.I)


def find_in_text(text: str) -> str | None:
    """Метрика из описания задачи: ближайшее упоминание после слова «основная/метрика/metric», иначе первое."""
    if not text:
        return None
    hits: list[tuple[int, int, str]] = []
    for name, pattern in _TEXT_PATTERNS:
        for found in re.finditer(pattern, text, re.I):
            hits.append((found.start(), found.end(), name))
    # убираем вложенные совпадения: «mean squared error» внутри «root mean squared error», «f1» внутри «macro f1»
    hits = [h for h in hits if not any(o != h and o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0])
                                       for o in hits)]
    if not hits:
        return None
    hits.sort()
    anchors = [m.end() for m in _PRIMARY.finditer(text)]
    for anchor in anchors:  # первое упоминание метрики в пределах 80 символов после якорного слова
        near = [h for h in hits if 0 <= h[0] - anchor <= 80]
        if near:
            return near[0][2]
    return hits[0][2]


# ---------------------------------------------------------------- классы и метки

def _natural_key(label: str) -> tuple:
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def label_str(value: Any) -> str:
    """Каноническая строка метки: 1, 1.0 и "1" → "1"; True → "True"."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def classes_of(y: Iterable) -> list[str]:
    """Классы в каноническом порядке: числовые метки — по значению, остальные — по строке (как np.unique в sklearn).

    Для бинарной задачи положительный класс — последний (так же sklearn выбирает pos_label по умолчанию).
    """
    values = pd.Series(list(y) if not isinstance(y, (pd.Series, np.ndarray)) else y).dropna()
    labels = {label_str(v) for v in pd.unique(values)}
    if all(_natural_key(label)[0] == 0 for label in labels):
        return sorted(labels, key=_natural_key)
    return sorted(labels)


def encode(y: Iterable, classes: list[str]) -> np.ndarray:
    """Метки → коды 0..k-1 по classes; неизвестная метка → ValueError."""
    labels = pd.Series(list(y) if not isinstance(y, (pd.Series, np.ndarray)) else y).map(label_str)
    codes = pd.Categorical(labels, categories=list(classes)).codes
    if (codes < 0).any():
        raise ValueError(f"labels outside of classes: {sorted(set(labels[codes < 0]))[:5]}")
    return codes.astype(np.int64)


def to_labels(y_pred: Any, decision: dict | None = None) -> np.ndarray:
    """Коды классов из вероятностей (argmax или decision) или из уже готовых меток."""
    pred = np.asarray(y_pred)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim == 1:
        if pred.dtype.kind in "iub":
            return pred.astype(np.int64)
        if np.all(np.isfinite(pred)) and np.all(pred == np.round(pred)) and pred.max(initial=0) > 1:
            return pred.astype(np.int64)  # float-метки 0., 2., 5.
        threshold = float((decision or {}).get("threshold", 0.5)) if (decision or {}).get("kind") == "threshold" else 0.5
        return (pred >= threshold).astype(np.int64)
    if decision and decision.get("kind") == "threshold" and pred.shape[1] == 2:
        return (pred[:, 1] >= float(decision["threshold"])).astype(np.int64)
    if decision and decision.get("kind") == "class_weights":
        weights = np.asarray(decision["weights"], dtype=float)
        if len(weights) == pred.shape[1]:
            return np.argmax(pred * weights, axis=1)
    return np.argmax(pred, axis=1)


def _true_codes(y_true: Any) -> np.ndarray:
    y = np.asarray(y_true)
    if y.dtype.kind in "iub":
        return y.astype(np.int64)
    if y.dtype.kind == "f" and np.all(np.isfinite(y)) and np.all(y == np.round(y)):
        return y.astype(np.int64)
    return encode(y, classes_of(y))


# ---------------------------------------------------------------- score

def score(name: str, y_true: Any, y_pred: Any, decision: dict | None = None) -> float:
    """Значение метрики. Формат y_true/y_pred — см. docstring модуля; несовпадение длины или NaN → ValueError."""
    metric = get(name)
    pred = np.asarray(y_pred)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if len(pred) != len(y_true):
        raise ValueError(f"{metric.name}: y_pred has {len(pred)} rows, y_true has {len(y_true)}")
    if pred.dtype.kind == "f" and not np.all(np.isfinite(pred)):
        raise ValueError(f"{metric.name}: y_pred contains NaN/inf")

    if metric.task == "regression":
        y = np.asarray(y_true, dtype=float)
        if pred.ndim != 1:
            raise ValueError(f"{metric.name}: regression expects 1-D predictions, got shape {pred.shape}")
        return float(metric.func(y, pred.astype(float)))

    if pred.ndim == 1 and pred.dtype.kind in "OUS":  # метки в исходном виде (например, из записанного сабмита)
        classes = classes_of(list(y_true) + list(pred))
        y_true, pred = encode(y_true, classes), encode(pred, classes)
    y = _true_codes(y_true)
    if metric.needs_proba:
        if pred.ndim == 1 and pred.dtype.kind in "iub" and pred.max(initial=0) > 1:
            raise ValueError(f"{metric.name}: needs probabilities, got labels")
        return float(metric.func(y, pred.astype(float)))
    return float(metric.func(y, to_labels(pred, decision)))


# ---------------------------------------------------------------- подбор порога / весов классов

def _confusion(y: np.ndarray, p: np.ndarray, k: int, w: np.ndarray | None = None) -> np.ndarray:
    return np.bincount(y * k + p, weights=w, minlength=k * k).reshape(k, k)


def _fast_label_metric(name: str, k: int) -> Callable[..., float]:
    """Быстрая версия метрики по матрице ошибок (для перебора); семантика как у sklearn.

    Третий аргумент — веса объектов (для оценки при других пропорциях классов); метрики без быстрой версии их игнорируют.
    """
    if name in ("macro_f1", "weighted_f1") or (name == "f1" and k > 2):
        def f1(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
            cm = _confusion(y, p, k, w)
            tp, support, predicted = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
            denom = support + predicted
            per_class = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
            if name == "weighted_f1":
                return float(np.sum(per_class * support) / max(support.sum(), 1))
            present = denom > 0  # sklearn усредняет по классам, встречающимся в y_true или y_pred
            return float(per_class[present].mean()) if present.any() else 0.0
        return f1
    if name == "f1":
        def f1_binary(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
            w = np.ones(len(y)) if w is None else w
            tp = np.sum(w * ((y == 1) & (p == 1)))
            denom = np.sum(w * (y == 1)) + np.sum(w * (p == 1))
            return float(2 * tp / denom) if denom else 0.0
        return f1_binary
    if name == "balanced_accuracy":
        def bal(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
            cm = _confusion(y, p, k, w)
            support = cm.sum(axis=1)
            present = support > 0
            return float(np.mean(np.diag(cm)[present] / support[present]))
        return bal
    func = REGISTRY[name].func
    return lambda y, p, w=None: float(func(y, p))


def fit_decision(name: str, y_true: Any, proba: Any, *, time_limit: float = 20.0,
                 max_rows: int = 300_000, seed: int = 0, sample_weight: Any = None) -> dict | None:
    """Порог (бинарная) или веса классов (мультикласс) под метрику по меткам; None — если не нужно/не помогло.

    Подбирается на переданных OOF-вероятностях; честный выигрыш оценивать по другим фолдам (crossfit_decision).
    sample_weight — веса объектов, например π_test(y)/π_train(y) при сдвиге пропорций классов (см. estimate_priors).
    """
    metric = get(name)
    if not metric.tunable:
        return None
    p = _proba_matrix(np.asarray(proba, dtype=float))
    y = _true_codes(y_true)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if len(y) > max_rows:
        idx = np.random.default_rng(seed).choice(len(y), max_rows, replace=False)
        y, p, w = y[idx], p[idx], (None if w is None else w[idx])
    k = p.shape[1]
    fast = _fast_label_metric(metric.name, k)
    sign = 1.0 if metric.greater_is_better else -1.0
    base = sign * fast(y, np.argmax(p, axis=1), w)
    stop_at = time.monotonic() + time_limit

    if k == 2:
        candidates = np.unique(np.quantile(p[:, 1], np.linspace(0.005, 0.995, 199)))
        best_t, best = 0.5, base
        for t in candidates:
            value = sign * fast(y, (p[:, 1] >= t).astype(np.int64), w)
            if value > best + 1e-12:
                best_t, best = float(t), value
        return {"kind": "threshold", "threshold": best_t} if best > base + 1e-9 else None

    grid = np.exp(np.linspace(-2.0, 2.0, 17 if k <= 30 else 9))
    weights, best = np.ones(k), base
    for _ in range(3 if k <= 30 else 1):  # покоординатный подъём по весам классов
        improved = False
        for c in range(k):
            if time.monotonic() > stop_at:
                break
            current = weights[c]
            for g in grid:
                weights[c] = current * g
                value = sign * fast(y, np.argmax(p * weights, axis=1), w)
                if value > best + 1e-12:
                    best, current, improved = value, weights[c], True
                weights[c] = current
        if not improved or time.monotonic() > stop_at:
            break
    weights = weights / np.exp(np.mean(np.log(weights)))
    return {"kind": "class_weights", "weights": [float(w) for w in weights]} if best > base + 1e-9 else None


def crossfit_decision(name: str, y_true: Any, proba: Any, folds: np.ndarray, *, time_limit: float = 60.0,
                      sample_weight: Any = None) -> tuple[list[float], dict | None]:
    """Честная оценка decision: для фолда f порог/веса подбираются на остальных фолдах OOF.

    Возвращает (оценки по фолдам с decision, decision, подобранный на всех OOF — для test).
    С sample_weight и подбор, и оценка идут с весами (качество при пропорциях классов test).
    """
    y = _true_codes(y_true)
    p = np.asarray(proba, dtype=float)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    fold_ids = np.unique(folds)
    per_fold_limit = time_limit / (len(fold_ids) + 1)
    k = _proba_matrix(p).shape[1]
    weighted = _fast_label_metric(get(name).name, k)
    scores = []
    for f in fold_ids:
        inside, outside = folds == f, folds != f
        decision = fit_decision(name, y[outside], p[outside], time_limit=per_fold_limit,
                                sample_weight=None if w is None else w[outside])
        if w is None:
            scores.append(score(name, y[inside], p[inside], decision))
        else:
            scores.append(weighted(y[inside], to_labels(p[inside], decision), w[inside]))
    return scores, fit_decision(name, y, p, time_limit=per_fold_limit, sample_weight=w)


# ---------------------------------------------------------------- сдвиг пропорций классов

def estimate_priors(proba: Any, train_priors: Any, *, max_iter: int = 200, tol: float = 1e-7) -> np.ndarray:
    """Пропорции классов в выборке по вероятностям модели, обученной при train_priors (EM, Saerens et al., 2002).

    Работает, если вероятности откалиброваны и меняются только пропорции классов, а не p(x | класс).
    """
    p = np.clip(_proba_matrix(np.asarray(proba, dtype=float)), 1e-12, 1.0)
    base = np.clip(np.asarray(train_priors, dtype=float), 1e-12, None)
    base = base / base.sum()
    priors = base.copy()
    for _ in range(max_iter):
        adjusted = p * (priors / base)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        new = adjusted.mean(axis=0)
        if np.max(np.abs(new - priors)) < tol:
            priors = new
            break
        priors = new
    return priors / priors.sum()
