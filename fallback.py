"""Fallback-пайплайн без LLM (P3): страховка решения и стартовый кандидат для агента.

run(spec, deadline) -> Candidate:
  константный сабмит в первые секунды → авто-предобработка (preprocess) → K-fold CV:
  LightGBM → [LightGBM на log1p-таргете] → CatBoost → [LightGBM с class_weight=balanced] → XGBoost
  (ранняя остановка по валидационному фолду, лимит времени на фолд) → жадный бленд по OOF →
  порог/веса классов для F1-подобных метрик (только если выигрыш подтверждён на других фолдах).
  submission.csv пишется после первой модели и перезаписывается только при улучшении CV.

Candidate.cv_score — без пост-обработки (argmax), чтобы сравнение с кандидатами агента/evaluator было честным;
CV с порогом/весами классов — в meta["cv_score_post"], сами они — в meta["decision"].

Автономный запуск (эвристический spec вместо understand_task): python3 fallback.py task.md
"""

from __future__ import annotations

import copy
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

import metrics
import preprocess
import submission

log = logging.getLogger("fallback")

# --- контракты команды (schemas.py у P1); пока его нет в main — локальные заменители ---
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


@dataclass
class HeuristicSpec:
    """Поля TaskSpec (schemas.py, P1) — для guess_spec, если schemas недоступен."""
    task_dir: Path
    id_col: str | None
    target: str | list[str]
    task_type: str
    metric: str
    greater_is_better: bool
    feature_notes: dict[str, str] = field(default_factory=dict)
    drop_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    date_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    raw_task_text: str = ""
    source: str = "heuristic"


@dataclass
class FallbackConfig:
    time_budget: float = 900.0       # потолок на весь fallback, с (план: fallback ≤ T+15 мин)
    reserve: float = 20.0            # не трогаем последние N секунд дедлайна
    n_splits: int = 5
    seed: int = 42                   # тот же сид, что у заглушки evaluator P4: при cv="random" сплиты совпадают
    cv: str = "auto"                 # auto — схема по данным (период / дубли / случайная), random — всегда (Stratified)KFold
    models: tuple[str, ...] = ("lightgbm", "xgboost", "catboost")  # CatBoost последним: на public медленнее и слабее
    max_rounds: int = 5000
    early_stopping: int = 100
    try_log_target: bool = True
    try_balanced: bool = True        # LightGBM с class_weight=balanced для F1-подобных метрик при дисбалансе
    blend: bool = True
    decision: bool = True            # порог / веса классов для F1-подобных метрик
    # учитывать сдвиг пропорций классов train → test (EM). По умолчанию выключено: на OOF public-мультикласса с
    # искусственным сдвигом коррекция ухудшила Macro F1 во всех сценариях (−0.004…−0.059) — EM не различает
    # похожие классы (DoS/Backdoor/Analysis). Оценка пропорций всё равно считается и пишется в meta как диагностика.
    label_shift: bool = False
    job_cap_units: float = 4.0       # потолок времени на модель после первой: N × время первого LightGBM
    entity: bool = True              # признаки по таргету повторяющейся сущности (регрессия/бинарная), берутся по CV
    decision_seconds: float = 40.0
    write: bool = True
    threads: int | None = None


# ---------------------------------------------------------------- окружение

def n_threads() -> int:
    """Доступные ядра с учётом affinity и квоты cgroup (в Docker os.cpu_count() видит все ядра хоста).

    FALLBACK_THREADS=<n> в окружении задаёт число вручную (перекалибровка без правки кода).
    """
    forced = os.environ.get("FALLBACK_THREADS", "")
    if forced.isdigit() and int(forced) > 0:
        return int(forced)
    n = os.cpu_count() or 1
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    quota = None
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if raw and raw[0] != "max":
            quota = float(raw[0]) / float(raw[1])
    except (OSError, ValueError, IndexError):
        try:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            quota = q / p if q > 0 and p > 0 else None
        except (OSError, ValueError):
            pass
    if quota:
        n = min(n, max(1, int(math.ceil(quota))))
    return max(1, n)


def _budget(deadline: Any, cfg: FallbackConfig) -> float:
    budget = cfg.time_budget
    if deadline is not None:
        try:
            budget = min(budget, float(deadline.left()) - cfg.reserve)
            stage = getattr(deadline, "stage", None)
            value = stage("fallback") if callable(stage) else None
            if isinstance(value, (int, float)) and math.isfinite(value):
                budget = min(budget, float(value) - cfg.reserve)
        except Exception as error:  # noqa: BLE001 — дедлайн чужой, его сбой не должен ронять fallback
            log.warning("дедлайн не прочитан (%s), бюджет %.0f с", error, budget)
    return max(budget, 20.0)


def read_table(path: Path) -> pd.DataFrame:
    """CSV с запасными кодировками и разделителем «;»."""
    frame = None
    for kw in ({}, {"encoding": "cp1251"}, {"encoding": "latin-1"}):
        try:
            frame = pd.read_csv(path, low_memory=False, **kw)
            break
        except UnicodeDecodeError:
            continue
    if frame is None:
        raise ValueError(f"cannot decode {path}")
    if frame.shape[1] == 1 and ";" in str(frame.columns[0]):
        frame = pd.read_csv(path, sep=";", low_memory=False, encoding=kw.get("encoding"))
    return frame


# ---------------------------------------------------------------- эвристический spec

def _continuous(y: pd.Series) -> bool:
    """Числовой таргет с дробными значениями и множеством уникальных — регрессия при любой подсказке."""
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    return bool(numeric and y.nunique() > 50 and not np.allclose(y, np.round(y)))


def infer_task_type(y: pd.Series, metric: str | None = None) -> str:
    y = y.dropna()
    if _continuous(y) or (metric is not None and metrics.get(metric).task == "regression"):
        return "regression"
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    n_unique = y.nunique()
    if metric is None and numeric and (n_unique > 20 or not np.allclose(y, np.round(y))):
        return "regression"
    return "binary" if n_unique <= 2 else "multiclass"


def guess_spec(task_path: str | Path) -> Any:
    """Спецификация задачи по файлам: id и таргет — из sample_submission.csv, метрика — из task.md, тип — по таргету.

    Нужна для автономного запуска и как запасной вариант, если understand_task (P2) упал.
    """
    path = Path(task_path)
    task_dir = path.parent if path.is_file() or path.suffix == ".md" else path
    md = path if path.is_file() else next(iter(sorted(task_dir.glob("*.md"))), None)
    text = md.read_text(encoding="utf-8", errors="replace") if md is not None and md.is_file() else ""
    probe = HeuristicSpec(task_dir, None, "", "regression", "rmse", False)
    sample_cols = list(pd.read_csv(submission.task_file(probe, "sample_submission.csv"), nrows=0).columns)
    train_cols = list(pd.read_csv(submission.task_file(probe, "train.csv"), nrows=0).columns)
    test_cols = list(pd.read_csv(submission.task_file(probe, "test.csv"), nrows=0).columns)

    id_col = sample_cols[0] if sample_cols[0] in test_cols and len(sample_cols) > 1 else None
    targets = [c for c in sample_cols if c != id_col and c in train_cols and c not in test_cols]
    if not targets:  # имя таргета в образце другое — берём колонки train, которых нет в test
        targets = [c for c in train_cols if c not in test_cols and c != id_col][:max(1, len(sample_cols) - 1)]
    if not targets:
        raise ValueError(f"cannot find target column in {task_dir}")

    metric = metrics.find_in_text(text)
    y = pd.read_csv(submission.task_file(probe, "train.csv"), usecols=targets)
    if len(targets) > 1:
        task_type = "regression" if any(infer_task_type(y[t], metric) == "regression" for t in targets) else "multilabel"
    else:
        task_type = infer_task_type(y[targets[0]], metric)
    if metric is None or (metrics.get(metric).task == "regression") != (task_type == "regression"):
        metric = metrics.default_for(task_type)
    fields = dict(task_dir=task_dir, id_col=id_col, target=targets[0] if len(targets) == 1 else targets,
                  task_type=task_type, metric=metric, greater_is_better=metrics.greater_is_better(metric),
                  feature_notes={}, drop_cols=[], cat_cols=[], date_cols=[], text_cols=[], raw_task_text=text,
                  source="heuristic")
    try:
        from schemas import TaskSpec
        spec = TaskSpec(**fields)
    except Exception:  # noqa: BLE001 — нет schemas или контракт другой
        spec = HeuristicSpec(**fields)
    log.info("spec (эвристика): target=%s, id=%s, тип=%s, метрика=%s", spec.target, spec.id_col, task_type, metric)
    return spec


# ---------------------------------------------------------------- данные задачи

@dataclass
class TaskData:
    spec: Any
    task: str                      # regression | binary | multiclass
    metric: str                    # каноническое имя
    y: np.ndarray                  # float (регрессия) или коды классов
    classes: list[str] | None
    prep: preprocess.Prepared
    folds: np.ndarray
    n_splits: int
    cv_scheme: dict = field(default_factory=dict)  # kind: random | period | duplicates (+ подробности)
    period_feature: str | None = None              # имя колонки-периода в prep.X (кандидат на исключение)
    label_shift: dict | None = None                # оценка пропорций классов в test (detect_label_shift)
    shift_weights: np.ndarray | None = None        # π_test(y)/π_train(y) для строк train, если сдвиг применяется
    rows: np.ndarray | None = None                 # позиции строк train.csv, участвующих в обучении (None — все)
    entity: dict | None = None                     # ключ «сущности» (find_entity) для признаков по её таргету
    period_order: np.ndarray | None = None         # порядок периода строки train (для «только прошлое»)
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]] = field(default_factory=dict)

    def frame(self, model: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        key = "lightgbm" if model in ("lightgbm", "xgboost") else model
        if key not in self.frames:
            x, cats = preprocess.model_frame(self.prep, self.prep.X, key)
            x_test, _ = preprocess.model_frame(self.prep, self.prep.X_test, key)
            self.frames[key] = (x, x_test, cats)
        return self.frames[key]


def make_folds(y: np.ndarray, task: str, n_splits: int, seed: int, groups: np.ndarray | None = None) -> np.ndarray:
    """Номер фолда для каждой строки: (Stratified)KFold, а при groups — (Stratified)GroupKFold (группа целиком в одном фолде)."""
    folds = np.zeros(len(y), dtype=np.int64)
    stratify = task != "regression" and np.bincount(y).max() >= n_splits
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed) if stratify else GroupKFold(n_splits)
    elif stratify:
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # «least populated class has only N members» — ожидаемо для редких классов
        for i, (_, valid) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            folds[valid] = i
    return folds


def find_period_column(train_x: pd.DataFrame, test_x: pd.DataFrame, max_levels: int = 50) -> str | None:
    """Колонка-«период»: целые коды, в test только значения, которых нет в train, и все по одну сторону от них.

    Пример — Year_Factor: train 1..5, test 6. Тогда test — «новый период», и честная валидация — по периодам.
    """
    found = []
    for col in train_x.columns:
        a, b = train_x[col], test_x[col]
        if pd.api.types.is_bool_dtype(a) or not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            continue
        ua, ub = np.unique(a.dropna()), np.unique(b.dropna())
        if not (3 <= len(ua) <= max_levels and 1 <= len(ub) <= max_levels) or np.intersect1d(ua, ub).size:
            continue
        if not np.allclose(np.r_[ua, ub], np.round(np.r_[ua, ub])):
            continue
        if ub.min() > ua.max() or ub.max() < ua.min():
            found.append((len(ua), col))
    return max(found)[1] if found else None  # больше уровней в train — больше фолдов


def duplicate_stats(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """(группа одинаковых строк train, доля строк train с дублем в train, доля строк test с точным близнецом в train)."""
    h_train = pd.util.hash_pandas_object(train_x, index=False).to_numpy()
    h_test = pd.util.hash_pandas_object(test_x, index=False).to_numpy()
    groups = pd.factorize(h_train)[0]
    dup_share = float(pd.Series(h_train).duplicated(keep=False).mean()) if len(h_train) else 0.0
    twin_share = float(np.isin(h_test, h_train).mean()) if len(h_test) else 0.0
    return groups, dup_share, twin_share


def identity_columns(train_x: pd.DataFrame, test_x: pd.DataFrame, exclude: set) -> list[str]:
    """Колонки-«идентичности» сущности: много уникальных значений, и у ≥ 30% строк test точное значение встречается
    в train редко (1–10 раз) — та же сущность в другой строке. Пример: floor_area здания (одно здание — разные годы)."""
    found = []
    for col in train_x.columns:
        if col in exclude or col not in test_x.columns:
            continue
        a = train_x[col]
        if a.nunique() < max(100, 0.01 * len(a)):  # мало уровней — это свойство, а не идентичность
            continue
        if test_x[col].map(a.value_counts()).between(1, 10).mean() >= 0.3:
            found.append(col)
    return found


def find_entity(train_x: pd.DataFrame, test_x: pd.DataFrame, exclude: set) -> dict | None:
    """Ключ сущности = колонки-идентичности + категориальные колонки (≤ 200 уровней). None — сущностей нет."""
    identity = identity_columns(train_x, test_x, exclude)
    if not identity:
        return None
    cats = [c for c in train_x.columns if c not in identity and c not in exclude
            and not pd.api.types.is_numeric_dtype(train_x[c]) and train_x[c].nunique() <= 200]
    cols = identity + cats
    key_train = pd.util.hash_pandas_object(train_x[cols].astype(str), index=False).to_numpy()
    key_test = pd.util.hash_pandas_object(test_x[cols].astype(str), index=False).to_numpy()
    return {"columns": cols, "identity": identity, "test_known": round(float(np.isin(key_test, key_train).mean()), 4),
            "key_train": key_train, "key_test": key_test}


ENTITY_FEATURES = ("entity__mean", "entity__count", "entity__median")


def _entity_stats(key_src: np.ndarray, y_src: np.ndarray, key_dst: np.ndarray) -> np.ndarray:
    if not len(key_src):
        return np.column_stack([np.full(len(key_dst), np.nan), np.zeros(len(key_dst)), np.full(len(key_dst), np.nan)])
    table = pd.DataFrame({"k": key_src, "y": y_src}).groupby("k", sort=False)["y"].agg(["mean", "count", "median"])
    out = table.reindex(key_dst).to_numpy(dtype=float)
    out[:, 1] = np.nan_to_num(out[:, 1])
    return out


def entity_features(data: "TaskData", train_idx: np.ndarray, valid_idx: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Среднее/число/медиана таргета той же сущности — без утечки: своя строка и валидационный фолд не участвуют.

    Есть порядок периодов (test — новый период): каждая строка видит только более ранние периоды, как test.
    Иначе — вложенный out-of-fold: строка обучающего фолда j видит остальные обучающие фолды.
    """
    key, y, order = data.entity["key_train"], data.y, data.period_order
    groups = order if order is not None else data.folds
    feat_tr = np.zeros((len(train_idx), len(ENTITY_FEATURES)))
    for g in np.unique(groups[train_idx]):
        rows = train_idx[groups[train_idx] == g]
        src = train_idx[groups[train_idx] < g] if order is not None else train_idx[groups[train_idx] != g]
        feat_tr[np.searchsorted(train_idx, rows)] = _entity_stats(key[src], y[src], key[rows])
    feat_va = np.zeros((len(valid_idx), len(ENTITY_FEATURES)))
    for g in np.unique(groups[valid_idx]):
        rows = valid_idx[groups[valid_idx] == g]
        src = train_idx[groups[train_idx] < g] if order is not None else train_idx
        feat_va[np.searchsorted(valid_idx, rows)] = _entity_stats(key[src], y[src], key[rows])
    feat_te = _entity_stats(key[train_idx], y[train_idx], data.entity["key_test"])  # для test весь train — прошлое
    return feat_tr, feat_va, feat_te


def choose_cv(train_x: pd.DataFrame, test_x: pd.DataFrame, y: np.ndarray, task: str, n_splits: int,
              seed: int, prep: preprocess.Prepared) -> tuple[np.ndarray, int, dict, str | None]:
    """Схема CV, похожая на то, как устроен test относительно train.

    period     — в test новый период (колонка-код вне диапазона train или дата позже train) → фолды по периодам;
    duplicates — в train много одинаковых строк, а в test близнецов из train заметно меньше → одинаковые строки
                 в одном фолде (иначе CV меряет запоминание дублей, которого на test не будет);
    random     — иначе (Stratified)KFold.
    """
    period = find_period_column(train_x, test_x)
    period_feature = None
    if period is not None:
        groups = pd.factorize(train_x[period])[0]
        renamed = {old: new for new, old in prep.report.get("renamed", {}).items()}
        period_feature = renamed.get(period, period) if renamed.get(period, period) in prep.X.columns else None
        k = min(n_splits, int(groups.max()) + 1)
        info = {"kind": "period", "column": period, "train_levels": int(groups.max()) + 1,
                "test_levels": sorted(np.unique(test_x[period].dropna()).tolist())[:10]}
        return make_folds(y, task, k, seed, groups), k, info, period_feature
    for col in [c for c in prep.X.columns if c.endswith("__ts")]:  # дата: test целиком позже train
        a, b = prep.X[col], prep.X_test[col]
        if a.notna().mean() > 0.9 and b.notna().any() and b.min() > a.max():
            groups = pd.qcut(a.rank(method="first"), n_splits, labels=False).fillna(0).astype(int).to_numpy()
            info = {"kind": "period", "column": col, "train_levels": n_splits, "test_levels": ["after train"]}
            return make_folds(y, task, n_splits, seed, groups), n_splits, info, None
    groups, dup_share, twin_share = duplicate_stats(train_x, test_x)
    info = {"dup_share": round(dup_share, 4), "twin_share": round(twin_share, 4)}
    if dup_share >= 0.05 and twin_share < 0.5 * dup_share:
        return make_folds(y, task, n_splits, seed, groups), n_splits, {"kind": "duplicates", **info}, None
    return make_folds(y, task, n_splits, seed), n_splits, {"kind": "random", **info}, None


def _resolve_task(spec: Any, y_raw: pd.Series) -> tuple[str, str]:
    task = str(getattr(spec, "task_type", "") or "")
    numeric = pd.api.types.is_numeric_dtype(y_raw) and not pd.api.types.is_bool_dtype(y_raw)
    if task == "regression" and not numeric:
        log.warning("spec: регрессия, но таргет не числовой — считаю классификацией")
        task = ""
    if task in ("binary", "multiclass") and _continuous(y_raw):
        log.warning("spec: %s, но таргет непрерывный (%d уникальных) — считаю регрессией", task, y_raw.nunique())
        task = "regression"
    if task not in ("regression", "binary", "multiclass"):
        task = infer_task_type(y_raw)
    if task != "regression":
        task = "binary" if y_raw.nunique() <= 2 else "multiclass"
    try:
        metric = metrics.resolve(getattr(spec, "metric", ""))
        if (metrics.REGISTRY[metric].task == "regression") != (task == "regression"):
            raise metrics.UnknownMetric(metric)
    except (metrics.UnknownMetric, TypeError):
        metric = metrics.default_for(task)
        log.warning("метрика %r не подходит к задаче %s — считаю по %s", getattr(spec, "metric", None), task, metric)
    return task, metric


def load(spec: Any, cfg: FallbackConfig) -> TaskData:
    train = read_table(submission.task_file(spec, "train.csv"))
    test = read_table(submission.task_file(spec, "test.csv"))
    target = submission.targets_of(spec)[0]
    if target not in train.columns:
        raise ValueError(f"target {target!r} not in train.csv")
    y_raw = train[target]
    keep = y_raw.notna().to_numpy()
    if pd.api.types.is_numeric_dtype(y_raw):
        keep &= np.isfinite(pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float, na_value=np.nan))
    if not keep.all():
        log.warning("строк без таргета в train: %d — убраны", int((~keep).sum()))
        train, y_raw = train[keep].reset_index(drop=True), y_raw[keep].reset_index(drop=True)
    task, metric = _resolve_task(spec, y_raw)

    if task == "regression":
        y, classes = y_raw.to_numpy(dtype=float), None
    else:
        classes = metrics.classes_of(y_raw)
        y = metrics.encode(y_raw, classes)

    targets = set(submission.targets_of(spec))
    drop = {getattr(spec, "id_col", None), *targets}
    features = [c for c in test.columns if c in train.columns and c not in drop]
    prep = preprocess.prepare(train[features], test[features], spec)
    n_splits = max(2, min(cfg.n_splits, len(y) // 20 if len(y) >= 40 else 2))
    if cfg.cv == "auto":
        folds, n_splits, scheme, period_feature = choose_cv(train[features], test[features], y, task, n_splits,
                                                            cfg.seed, prep)
    else:
        folds, scheme, period_feature = make_folds(y, task, n_splits, cfg.seed), {"kind": "random"}, None
    log.info("схема CV: %s", scheme)
    rows = None if keep.all() else np.flatnonzero(keep)
    period_order, entity = None, None
    if scheme.get("kind") == "period":
        column = scheme["column"]
        if column in train.columns:
            period_order = pd.to_numeric(train[column], errors="coerce").fillna(-np.inf).to_numpy()
        elif column in prep.X.columns:  # дата: те же интервалы времени, что и фолды
            period_order = pd.qcut(prep.X[column].rank(method="first"), n_splits, labels=False).fillna(0).to_numpy()
    if cfg.entity and task in ("regression", "binary"):
        try:
            entity = find_entity(train[features], test[features], {scheme.get("column")})
        except Exception as error:  # noqa: BLE001 — признаки сущности необязательны
            log.warning("поиск сущности пропущен: %s", error)
        if entity is not None:
            log.info("сущность: ключ %s (идентичности %s), в test узнаётся %.1f%% строк", entity["columns"],
                     entity["identity"], 100 * entity["test_known"])
    return TaskData(spec, task, metric, y, classes, prep, folds, n_splits, scheme, period_feature, rows=rows,
                    entity=entity, period_order=period_order)


def cv_folds(spec: Any, config: FallbackConfig | None = None) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Фолды ровно как у run() — для evaluator P1 и агента P4, чтобы все кандидаты мерились одной честной схемой.

    Возвращает (folds, rows, scheme): folds[i] — номер фолда i-й используемой строки train; rows — позиции этих строк
    в train.csv (None — используются все; иначе выброшены строки с пустым/бесконечным таргетом); scheme — описание
    схемы (kind: period | duplicates | random, n_splits и подробности). Модели не обучаются (чтение + предобработка).
    То же лежит в Candidate.meta["folds"] / ["train_rows"] / ["cv_scheme"] у кандидата fallback.
    """
    data = load(spec, config or FallbackConfig())
    return data.folds, data.rows, dict(data.cv_scheme, n_splits=data.n_splits)


# ---------------------------------------------------------------- модели

def _lr(n: int) -> float:
    return 0.1 if n > 300_000 else 0.05 if n > 3_000 else 0.03


def model_threads(model: str, n_rows: int, threads: int) -> int:
    """Потоки модели. LightGBM/XGBoost — не больше физических ядер (≈ половина vCPU) и «строк / 10 000».

    Замеры (Mac, fallback-LightGBM, одинаковый CV): 3 000 строк — 1 поток 1 с, 15 потоков 25 с;
    87 670 строк × 10 классов — 2/4/8 потоков: 184/152/202 с; 53 308 строк — 13/11/16 с. CatBoost масштабируется хорошо.
    """
    if model == "catboost":
        return threads
    return max(1, min(max(1, threads // 2), n_rows // ROWS_PER_THREAD))


ROWS_PER_THREAD = 10_000


def _params(model: str, data: TaskData, cfg: FallbackConfig, threads: int, extra: dict) -> dict:
    n = len(data.y)
    threads = model_threads(model, n, threads)
    regression = data.task == "regression"
    l1 = data.metric in ("mae", "medae", "mape", "smape", "wape")
    if model == "lightgbm":
        params = dict(n_estimators=cfg.max_rounds, learning_rate=_lr(n),
                      num_leaves=31 if n < 10_000 else 63 if n < 500_000 else 127,
                      min_child_samples=10 if n < 2_000 else 20, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, reg_lambda=1.0, n_jobs=threads, random_state=cfg.seed, verbose=-1)
        if regression:
            params["objective"] = "regression_l1" if l1 else "regression"
    elif model == "catboost":
        # ускорение без потери качества (public-мультикласс, фолд, 300 итераций): 69.6 → 18.1 с, logloss 0.4786 → 0.4751
        params = dict(iterations=cfg.max_rounds, learning_rate=min(0.15, _lr(n) * 1.6), depth=6,
                      random_seed=cfg.seed, thread_count=threads, verbose=0, allow_writing_files=False,
                      od_type="Iter", od_wait=cfg.early_stopping, use_best_model=True,
                      border_count=128, max_ctr_complexity=1, bootstrap_type="Bernoulli", subsample=0.8,
                      one_hot_max_size=16)
        if regression:
            params["loss_function"] = "MAE" if l1 else "RMSE"
    else:
        params = dict(n_estimators=cfg.max_rounds, learning_rate=_lr(n), max_depth=7, min_child_weight=1.0,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                      enable_categorical=True, max_cat_to_onehot=1, n_jobs=threads, random_state=cfg.seed,
                      early_stopping_rounds=cfg.early_stopping)
        if regression:
            params["objective"] = "reg:absoluteerror" if l1 else "reg:squarederror"
    params.update(extra)
    return params


def _lgb_timer(stop_at: float):
    import lightgbm as lgb

    def callback(env: Any) -> None:
        if time.monotonic() > stop_at:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    callback.order = 40  # type: ignore[attr-defined]
    return callback


class _CatBoostTimer:
    def __init__(self, stop_at: float) -> None:
        self.stop_at = stop_at

    def after_iteration(self, info: Any) -> bool:
        return time.monotonic() < self.stop_at


def _xgb_timer(stop_at: float):
    import xgboost as xgb

    class Timer(xgb.callback.TrainingCallback):
        def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
            return time.monotonic() > stop_at
    return Timer()


def _fit_predict(model: str, regression: bool, params: dict, x_tr: pd.DataFrame, y_tr: np.ndarray,
                 x_va: pd.DataFrame, y_va: np.ndarray, x_te: pd.DataFrame, cats: list[str],
                 stop_at: float, early_stopping: int) -> tuple[np.ndarray, np.ndarray, int, list]:
    """Обучение на фолде; классификация — коды 0..m-1 только присутствующих классов (иначе XGBoost падает)."""
    present = None
    if not regression:
        present = np.unique(y_tr)
        y_tr = np.searchsorted(present, y_tr)
        mask = np.isin(y_va, present)  # объекты классов, которых нет в обучении фолда, не годятся для early stopping
        x_es, y_es = x_va[mask], np.searchsorted(present, y_va[mask])
        if len(present) < 2:
            return None, None, 0, list(present)  # type: ignore[return-value]
    else:
        x_es, y_es = x_va, y_va

    if model == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if regression else lgb.LGBMClassifier
        est = cls(**params)
        est.fit(x_tr, y_tr, eval_set=[(x_es, y_es)],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False), _lgb_timer(stop_at)])
        iters = int(est.best_iteration_ or est.n_estimators)
    elif model == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        est = (CatBoostRegressor if regression else CatBoostClassifier)(**params)
        est.fit(x_tr, y_tr, cat_features=cats or None, eval_set=(x_es, y_es), callbacks=[_CatBoostTimer(stop_at)])
        iters = int(est.get_best_iteration() or est.tree_count_)
    else:
        from xgboost import XGBClassifier, XGBRegressor
        est = (XGBRegressor if regression else XGBClassifier)(**params, callbacks=[_xgb_timer(stop_at)])
        est.fit(x_tr, y_tr, eval_set=[(x_es, y_es)], verbose=False)
        iters = int(getattr(est, "best_iteration", None) or params["n_estimators"])

    if regression:
        return est.predict(x_va), est.predict(x_te), iters, []
    return est.predict_proba(x_va), est.predict_proba(x_te), iters, list(present)


@dataclass
class Run:
    name: str
    model: str
    oof: np.ndarray
    test_pred: np.ndarray
    fold_scores: list[float]
    seconds: float
    params: dict
    transform: str | None
    iterations: list[int]
    exclude: tuple[str, ...] = ()
    entity: bool = False

    @property
    def cv(self) -> float:
        return float(np.mean(self.fold_scores))


def cv_model(data: TaskData, name: str, model: str, cfg: FallbackConfig, stop_at: float, *,
             transform: str | None = None, extra: dict | None = None, threads: int = 1,
             exclude: tuple[str, ...] = (), entity: bool = False) -> Run:
    """K-fold с фиксированными фолдами: OOF, среднее по фолдам на test, оценка metrics.score на каждом фолде.

    entity — добавить признаки по таргету той же сущности (entity_features), считаются внутри каждого фолда.
    """
    start = time.monotonic()
    regression = data.task == "regression"
    x, x_test, cats = data.frame(model)
    if exclude:
        x, x_test = x.drop(columns=list(exclude), errors="ignore"), x_test.drop(columns=list(exclude), errors="ignore")
        cats = [c for c in cats if c not in exclude]
    params = _params(model, data, cfg, threads, extra or {})
    k = len(data.classes or [])
    oof = np.zeros(len(x)) if regression else np.zeros((len(x), k))
    test_pred = np.zeros(len(x_test)) if regression else np.zeros((len(x_test), k))
    scores, iterations = [], []
    for f in range(data.n_splits):
        now = time.monotonic()
        fold_stop = now + max(1.0, (stop_at - now) / (data.n_splits - f))
        train_idx, valid_idx = np.flatnonzero(data.folds != f), np.flatnonzero(data.folds == f)
        y_tr = data.y[train_idx]
        if transform == "log1p":
            y_tr = np.log1p(y_tr)
        x_tr, x_va, x_te = x.iloc[train_idx], x.iloc[valid_idx], x_test
        if entity and data.entity is not None:
            e_tr, e_va, e_te = entity_features(data, train_idx, valid_idx)
            x_tr = x_tr.assign(**dict(zip(ENTITY_FEATURES, e_tr.T)))
            x_va = x_va.assign(**dict(zip(ENTITY_FEATURES, e_va.T)))
            x_te = x_te.assign(**dict(zip(ENTITY_FEATURES, e_te.T)))
        p_va, p_te, iters, present = _fit_predict(
            model, regression, params, x_tr, y_tr, x_va,
            np.log1p(data.y[valid_idx]) if transform == "log1p" else data.y[valid_idx], x_te, cats, fold_stop,
            cfg.early_stopping)
        if regression:
            if transform == "log1p":
                p_va, p_te = np.expm1(p_va), np.expm1(p_te)
            oof[valid_idx], fold_test = p_va, p_te
        else:
            full_va, fold_test = np.zeros((len(valid_idx), k)), np.zeros((len(x_test), k))
            if p_va is None:  # в обучении фолда один класс
                full_va[:, present[0]], fold_test[:, present[0]] = 1.0, 1.0
            else:
                full_va[:, present], fold_test[:, present] = p_va, p_te
            oof[valid_idx] = full_va
        test_pred += fold_test / data.n_splits
        scores.append(metrics.score(data.metric, data.y[valid_idx], oof[valid_idx]))
        iterations.append(iters)
    return Run(name, model, oof, test_pred, scores, time.monotonic() - start, params, transform, iterations,
               tuple(exclude), bool(entity and data.entity is not None))


# ---------------------------------------------------------------- бленд и кандидаты

def _fold_scores(data: TaskData, oof: np.ndarray) -> list[float]:
    return [metrics.score(data.metric, data.y[data.folds == f], oof[data.folds == f]) for f in range(data.n_splits)]


def blend(data: TaskData, runs: list[Run], max_steps: int = 30) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Жадный отбор с возвращением (Caruana) по OOF; None — если отбор оставил одну модель.

    Отбор — по метрике задачи (на public-мультиклассе отбор по logloss отбрасывал balanced-модель, лучшую по Macro F1,
    и давал бленд хуже одиночных); веса подобраны на тех же OOF, поэтому в run бленд принимается только с выигрышем
    больше шума по фолдам, с порогом/весами классов.
    """
    if len(runs) < 2:
        return None
    select = data.metric
    counts = np.zeros(len(runs))
    current, best = None, None
    for step in range(max_steps):
        choice, choice_score = None, None
        for i, run in enumerate(runs):
            trial = run.oof if current is None else (current * step + run.oof) / (step + 1)
            value = metrics.score(select, data.y, trial)
            if choice_score is None or metrics.is_better(select, value, choice_score):
                choice, choice_score = i, value
        if best is not None and not metrics.is_better(select, choice_score, best):
            break
        counts[choice] += 1
        current = runs[choice].oof if current is None else (current * step + runs[choice].oof) / (step + 1)
        best = choice_score
    weights = counts / counts.sum()
    if (weights > 0).sum() < 2:
        return None
    oof = sum(w * r.oof for w, r in zip(weights, runs) if w > 0)
    test_pred = sum(w * r.test_pred for w, r in zip(weights, runs) if w > 0)
    return oof, test_pred, {r.name: round(float(w), 4) for w, r in zip(weights, runs) if w > 0}


def detect_label_shift(data: TaskData, run: Run, min_tv: float = 0.03, max_sanity_tv: float = 0.02,
                       apply: bool = False) -> dict:
    """Пропорции классов в test по калиброванной модели (EM). used=True — коррекция включена (apply), сдвиг заметный
    и калибровка в порядке (EM по OOF той же модели возвращает пропорции train). Иначе — только диагностика.
    """
    k = len(data.classes or [])
    train = np.bincount(data.y, minlength=k) / len(data.y)
    sanity = metrics.estimate_priors(run.oof, train)
    test = metrics.estimate_priors(run.test_pred, train)
    tv, sanity_tv = 0.5 * float(np.abs(test - train).sum()), 0.5 * float(np.abs(sanity - train).sum())
    info = {"train": dict(zip(data.classes, np.round(train, 4).tolist())),
            "test": dict(zip(data.classes, np.round(test, 4).tolist())),
            "tv": round(tv, 4), "sanity_tv": round(sanity_tv, 4), "model": run.name,
            "used": bool(apply and tv >= min_tv and sanity_tv <= max_sanity_tv)}
    if info["used"]:
        data.shift_weights = (test / np.clip(train, 1e-12, None))[data.y]
    return info


def make_candidate(data: TaskData, name: str, oof: np.ndarray, test_pred: np.ndarray, fold_scores: list[float],
                   meta: dict, cfg: FallbackConfig, decision_seconds: float) -> Candidate:
    meta = dict(meta, fold_scores=[float(s) for s in fold_scores], folds=data.folds, n_splits=data.n_splits,
                cv_scheme=data.cv_scheme, train_rows=data.rows,
                seed=cfg.seed, classes=data.classes, metric=data.metric, decision=None,
                cv_score_post=float(np.mean(fold_scores)))
    tunable = data.task != "regression" and metrics.get(data.metric).tunable
    w = data.shift_weights if tunable else None
    if w is not None:  # всё «пост»-качество — при пропорциях классов test, чтобы кандидаты сравнивались одинаково
        weighted = metrics._fast_label_metric(data.metric, len(data.classes))
        base = [weighted(data.y[data.folds == f], oof[data.folds == f].argmax(axis=1), w[data.folds == f])
                for f in range(data.n_splits)]
        meta.update(cv_score_post=float(np.mean(base)), label_shift=data.label_shift)
    else:
        base = fold_scores
    meta["fold_scores_eff"] = [float(v) for v in base]  # по фолдам — то, что сравнивается (с decision — ниже)
    if cfg.decision and tunable and decision_seconds > 1:
        try:
            post_scores, decision = metrics.crossfit_decision(data.metric, data.y, oof, data.folds,
                                                              time_limit=decision_seconds, sample_weight=w)
            if decision is not None and metrics.is_better(data.metric, float(np.mean(post_scores)),
                                                          float(np.mean(base))):
                meta.update(decision=decision, cv_score_post=float(np.mean(post_scores)),
                            fold_scores_post=[float(s) for s in post_scores],
                            fold_scores_eff=[float(s) for s in post_scores])
        except Exception as error:  # noqa: BLE001
            log.warning("порог/веса классов не подобраны: %s", error)
    return Candidate(name=name, cv_score=float(np.mean(fold_scores)), cv_std=float(np.std(fold_scores)),
                     oof=oof, test_pred=test_pred, meta=meta)


def clearly_better(metric: str, new: Candidate, old: Candidate, k: float = 1.0) -> bool:
    """Выигрыш по фолдам больше k стандартных ошибок парных разностей (как правило принятия у агента P4)."""
    a = np.asarray(new.meta.get("fold_scores_eff", new.meta.get("fold_scores", [])), dtype=float)
    b = np.asarray(old.meta.get("fold_scores_eff", old.meta.get("fold_scores", [])), dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return metrics.is_better(metric, effective_cv(new), effective_cv(old))
    diff = (a - b) * (1.0 if metrics.greater_is_better(metric) else -1.0)
    return bool(diff.mean() > k * diff.std(ddof=1) / np.sqrt(len(diff)))


def effective_cv(cand: Candidate) -> float:
    """CV того, что реально попадёт в сабмит: с порогом/весами классов, если они приняты."""
    return float((getattr(cand, "meta", None) or {}).get("cv_score_post", cand.cv_score))


def _constant_candidate(data: TaskData, cfg: FallbackConfig) -> Candidate:
    n_test = len(data.prep.X_test)
    if data.task == "regression":
        center = float(np.median(data.y)) if data.metric in ("mae", "medae", "mape", "wape") else float(np.mean(data.y))
        oof, test_pred = np.full(len(data.y), center), np.full(n_test, center)
    else:
        prior = np.bincount(data.y, minlength=len(data.classes)) / len(data.y)
        oof, test_pred = np.tile(prior, (len(data.y), 1)), np.tile(prior, (n_test, 1))
    return make_candidate(data, "fallback_constant", oof, test_pred, _fold_scores(data, oof),
                          {"model": "constant", "params": {}, "target_transform": None}, cfg, 0)


# ---------------------------------------------------------------- запуск

class _Writer:
    """Пишет сабмит только при улучшении CV с учётом пост-обработки (то, что реально уйдёт в файл)."""

    def __init__(self, spec: Any, metric: str, enabled: bool) -> None:
        self.spec, self.metric, self.enabled = spec, metric, enabled
        self.best: float | None = None

    def offer(self, cand: Candidate) -> None:
        score = effective_cv(cand)
        if not self.enabled or not metrics.is_better(self.metric, score, self.best):
            return
        try:
            submission.write_candidate(self.spec, cand)
            self.best = score
            log.info("submission.csv ← %s (cv=%.6g)", cand.name, score)
        except Exception as error:  # noqa: BLE001 — запись не должна ронять fallback; остаётся прежний файл
            log.error("submission.csv не записан (%s): %s: %s", cand.name, type(error).__name__, error)


def _plan(data: TaskData, cfg: FallbackConfig) -> list[tuple[str, str, dict]]:
    """Очередь заданий (имя, модель, доп. параметры) по приоритету."""
    jobs = []
    for model in cfg.models:
        jobs.append((model, model, {}))
        if model == "lightgbm" and data.task != "regression" and cfg.try_balanced and metrics.get(data.metric).tunable:
            counts = np.bincount(data.y)
            if counts[counts > 0].max() / counts[counts > 0].min() >= 5:
                jobs.append(("lightgbm_balanced", "lightgbm", {"class_weight": "balanced"}))
    # balanced идёт сразу за LightGBM: он дешёвый и на public-мультиклассе лучший по Macro F1.
    # Второй пресет LightGBM (глубже, lr 0.03, сильнее регуляризация) проверен на public и не взят: хуже базового на
    # обеих задачах (RMSE 45.59 против 44.72, Macro F1 0.601 против 0.604) при 80–138 с времени.
    return jobs


_COST = {"lightgbm": 1.0, "catboost": 2.5, "xgboost": 1.5}


def run(spec: Any, deadline: Any = None, config: FallbackConfig | None = None) -> Candidate:
    """Fallback-кандидат и submission.csv. Исключение — только если данные задачи не читаются вовсе."""
    cfg = config or FallbackConfig()
    start = time.monotonic()
    budget = _budget(deadline, cfg)
    end = start + budget
    threads = cfg.threads or n_threads()
    log.info("fallback: бюджет %.0f с, потоков %d", budget, threads)

    if cfg.write:
        submission.write_dummy(spec)
    data = load(spec, cfg)
    log.info("задача: %s, метрика %s, train %d строк, признаков %d, фолдов %d%s", data.task, data.metric,
             len(data.y), data.prep.X.shape[1], data.n_splits,
             f", классов {len(data.classes)}" if data.classes else "")
    writer = _Writer(spec, data.metric, cfg.write)
    decision_seconds = min(cfg.decision_seconds, 0.05 * budget)
    runs: list[Run] = []
    best: Candidate | None = None
    transform: str | None = None
    excluded: tuple[str, ...] = ()  # признаки, исключённые по CV (колонка-период)
    use_entity = False              # признаки сущности приняты по CV
    unit: float | None = None  # секунд на «единицу» LightGBM — для оценки остальных моделей

    def left() -> float:
        return end - time.monotonic()

    def consider(name: str, oof: np.ndarray, test_pred: np.ndarray, scores: list[float], meta: dict,
                 strict: bool = False) -> None:
        """Лучший — по качеству того, что уйдёт в файл (CV с порогом/весами классов); он же пишется на диск.

        strict — для бленда: его веса подобраны на тех же OOF, поэтому нужен выигрыш больше шума по фолдам.
        """
        nonlocal best
        cand = make_candidate(data, name, oof, test_pred, scores, meta, cfg, min(decision_seconds, left() / 4))
        if strict and best is not None and not clearly_better(data.metric, cand, best):
            log.info("%s: cv=%.6g не лучше %s с запасом больше шума — не берём", name, effective_cv(cand), best.name)
            return
        if best is None or metrics.is_better(data.metric, effective_cv(cand), effective_cv(best)):
            best = cand
            writer.offer(best)

    def accept(result: Run) -> None:
        runs.append(result)
        if (data.task != "regression" and data.label_shift is None
                and result.params.get("class_weight") is None):
            try:
                data.label_shift = detect_label_shift(data, result, apply=cfg.label_shift)
                log.info("пропорции классов train → test (EM по %s): сдвиг %.3f, калибровка %.3f, %s; test: %s",
                         result.name, data.label_shift["tv"], data.label_shift["sanity_tv"],
                         "учитываем" if data.label_shift["used"] else "только диагностика", data.label_shift["test"])
            except Exception as error:  # noqa: BLE001
                data.label_shift = {"used": False, "error": str(error)[:200]}
        consider(f"fallback_{result.name}", result.oof, result.test_pred, result.fold_scores,
                 {"model": result.model, "params": result.params, "target_transform": result.transform,
                  "excluded": list(result.exclude), "entity": result.entity, "iterations": result.iterations,
                  "seconds": round(result.seconds, 1)})

    def attempt(name: str, model: str, extra: dict, tf: str | None, share: float,
                exclude: tuple[str, ...] = (), entity: bool | None = None) -> Run | None:
        tail = 15.0 + decision_seconds
        if unit is not None and unit * _COST.get(model, 1.0) > 1.5 * max(0.0, left() - tail):
            log.info("%s пропущен: оценка %.0f с, осталось %.0f с", name, unit * _COST.get(model, 1.0), left())
            return None
        if left() - tail < 5:
            return None
        window = (left() - tail) * share
        if unit is not None:
            window = min(window, cfg.job_cap_units * unit)
        stop_at = time.monotonic() + window
        try:
            result = cv_model(data, name, model, cfg, stop_at, transform=tf, extra=extra, threads=threads,
                              exclude=exclude, entity=use_entity if entity is None else entity)
        except Exception as error:  # noqa: BLE001 — сбой одной модели не роняет остальные
            log.warning("%s упал: %s: %s", name, type(error).__name__, str(error)[:300])
            return None
        log.info("%s: cv=%.6g ± %.3g (%s, %.0f с, итераций ~%d)", name, result.cv, float(np.std(result.fold_scores)),
                 data.metric, result.seconds, int(np.mean(result.iterations)))
        return result

    jobs = _plan(data, cfg)
    for i, (name, model, extra) in enumerate(jobs):
        last = i == len(jobs) - 1
        result = attempt(name, model, extra, transform, 1.0 if last else 0.5, excluded)
        if result is None:
            continue
        unit = unit or result.seconds
        accept(result)  # сабмит пишется сразу после первой модели
        if model == "lightgbm" and not extra and data.period_feature and not excluded:
            # test — новый период, которого модель не видела: проверяем по CV, не мешает ли сам код периода
            without = attempt("lightgbm_noperiod", "lightgbm", {}, transform, 0.5, (data.period_feature,))
            if without is not None:
                accept(without)
                if metrics.is_better(data.metric, without.cv, result.cv):
                    excluded, result = (data.period_feature,), without
                    log.info("без %s лучше по CV — остальные модели учатся без него", data.period_feature)
        if model == "lightgbm" and not extra and data.entity is not None and not use_entity:
            with_entity = attempt("lightgbm_entity", "lightgbm", {}, transform, 0.5, excluded, entity=True)
            if with_entity is not None:
                accept(with_entity)
                if metrics.is_better(data.metric, with_entity.cv, result.cv):
                    use_entity, result = True, with_entity
                    log.info("признаки сущности лучше по CV — остальные модели учатся с ними")
        if (model == "lightgbm" and not extra and data.task == "regression" and transform is None
                and cfg.try_log_target and data.y.min() >= 0
                and (data.metric in ("rmsle", "msle", "mape") or pd.Series(data.y).skew() > 0.75)):
            logged = attempt("lightgbm_log", "lightgbm", {}, "log1p", 0.5, excluded)
            if logged is not None:
                accept(logged)
                if metrics.is_better(data.metric, logged.cv, result.cv):
                    transform = "log1p"
                    log.info("лог-таргет лучше по CV — остальные модели учатся на log1p")

    if cfg.blend and len(runs) >= 2 and left() > 5:
        try:
            mixed = blend(data, runs)
            if mixed is not None:
                oof, test_pred, weights = mixed
                scores = _fold_scores(data, oof)
                log.info("бленд %s: cv=%.6g", weights, float(np.mean(scores)))
                consider("fallback_blend", oof, test_pred, scores,
                         {"model": "blend", "members": weights, "params": {}, "target_transform": transform,
                          "excluded": list(excluded)}, strict=True)
        except Exception as error:  # noqa: BLE001
            log.warning("бленд упал: %s", error)

    if best is None:
        log.error("ни одна модель не обучилась — константный кандидат")
        best = _constant_candidate(data, cfg)
        writer.offer(best)
    best.meta.update(seconds=round(time.monotonic() - start, 1), preprocess=data.prep.report,
                     label_shift=data.label_shift,
                     entity_key=None if data.entity is None else {k: data.entity[k] for k in ("columns", "identity",
                                                                                               "test_known")},
                     runs=[{"name": r.name, "cv": r.cv, "std": float(np.std(r.fold_scores)),
                            "seconds": round(r.seconds, 1)} for r in runs])
    log.info("fallback готов за %.0f с: %s cv=%.6g (с порогом/весами: %.6g)", time.monotonic() - start, best.name,
             best.cv_score, best.meta.get("cv_score_post", best.cv_score))
    return best


def run_multi(spec: Any, deadline: Any = None, config: FallbackConfig | None = None) -> Candidate:
    """Несколько таргетов (мульти-регрессия / мультилейбл): по модели на таргет, общий бюджет делится поровну."""
    cfg = config or FallbackConfig()
    targets = submission.targets_of(spec)
    budget = _budget(deadline, cfg)
    sub_cfg = copy.copy(cfg)
    sub_cfg.write, sub_cfg.blend, sub_cfg.time_budget = False, False, budget / len(targets)
    sub_cfg.models = cfg.models[:1]
    oofs, tests, scores = [], [], []
    for t in targets:
        sub = copy.copy(spec)
        sub.target = t
        sub.task_type = "regression" if spec.task_type == "regression" else "binary"
        cand = run(sub, None, sub_cfg)
        positive = (lambda p: p) if sub.task_type == "regression" else (lambda p: p[:, -1])
        oofs.append(positive(cand.oof))
        tests.append(positive(cand.test_pred))
        scores.append(cand.cv_score)
    cand = Candidate("fallback_multi", float(np.mean(scores)), float(np.std(scores)),
                     np.column_stack(oofs), np.column_stack(tests), {"per_target": dict(zip(targets, scores))})
    if cfg.write:
        submission.write(spec, cand.test_pred)
    return cand


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    task_path = Path(argv[1] if len(argv) > 1 else "task.md").resolve()
    spec = guess_spec(task_path)
    runner = run_multi if isinstance(spec.target, list) else run
    cand = runner(spec)
    problems = submission.check_file(spec)
    print(f"{cand.name}: cv={cand.cv_score:.6g} ± {cand.cv_std:.3g}; submission: {problems or 'ok'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
