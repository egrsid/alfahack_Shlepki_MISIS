"""Авто-предобработка (P3): сырые признаки train/test → кадр для градиентных бустингов.

prepare(train_x, test_x, spec) определяет тип каждой колонки (число, категория, дата, текст, числа-строками),
удаляет бесполезные (константы, id-подобные, дубли, пустые в test) и строит признаки:
категории — pandas category (редкие уровни → "__rare__") + частота; даты — год/месяц/день/день недели/час/timestamp;
текст — длины + TF-IDF→SVD. Таргет не используется: всё, что считается по train+test вместе, — без утечки таргета.
model_frame(prep, X, model) приводит кадр к виду конкретной модели (lightgbm / xgboost / catboost / sklearn).
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("preprocess")

RARE = "__rare__"
NAN = "__nan__"
_ID_NAME = re.compile(r"(^|[_\s.-])(id|uuid|guid|index|idx|rowid|row_id)$|^(id|uuid|guid)([_\s.-]|$)|^unnamed:?\s*\d+$",
                      re.I)
_DATE_RE = re.compile(r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})"
                      r"([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?\s*(Z|[+-]\d{2}:?\d{2})?$")


@dataclass
class PrepConfig:
    min_count: int = 5              # уровни категории реже (в train) → RARE; на маленьких данных — 2
    max_cat_levels: int = 255       # больше уровней → для lightgbm/xgboost коды по частоте вместо category
    freq_min_levels: int = 10       # частотный признак для категорий с таким числом уровней и больше
    text_min_len: float = 25.0      # текст: средняя длина и число слов не меньше, много уникальных значений
    text_min_words: float = 4.0
    svd_components: int = 16
    max_text_rows: int = 400_000    # TF-IDF только на данных не больше этого
    max_text_cols: int = 5
    parse_share: float = 0.95       # доля значений, которая должна распознаться как число / дата
    float32_cells: int = 30_000_000  # при большем числе ячеек float64 → float32
    seed: int = 42


@dataclass
class Prepared:
    X: pd.DataFrame
    X_test: pd.DataFrame
    cat_cols: list[str]
    high_card: list[str]
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------- типы колонок

def _sample(values: pd.Series, n: int = 2000, seed: int = 0) -> pd.Series:
    values = values.dropna()
    return values.sample(n, random_state=seed) if len(values) > n else values


def _clean_numeric_text(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"[\s% ]", "", regex=True)
    has_dot = text.str.contains(".", regex=False)
    text = text.where(has_dot, text.str.replace(",", ".", regex=False))  # «12,5» → 12.5, «1,234.5» не трогаем
    return pd.to_numeric(text, errors="coerce")


def _numeric_like(values: pd.Series, share: float) -> bool:
    sample = _sample(values)
    return len(sample) > 0 and _clean_numeric_text(sample).notna().mean() >= share


def _parse_dates(values: pd.Series, share: float) -> pd.Series | None:
    sample = _sample(values, 500).astype(str).str.strip()
    if not len(sample) or sample.str.match(_DATE_RE).mean() < share:
        return None
    first = sample.iloc[0]
    dayfirst = bool(re.match(r"^\d{1,2}\.", first))  # 31.12.2020 — день впереди
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in ({"format": pd.tseries.api.guess_datetime_format(first, dayfirst=dayfirst)},
                   {"format": "mixed", "dayfirst": dayfirst}):
            if kw.get("format") is None:
                continue
            try:
                if pd.to_datetime(sample, errors="coerce", **kw).notna().mean() < share:
                    continue
                parsed = pd.to_datetime(values.astype(str).str.strip().where(values.notna()), errors="coerce", **kw)
            except (ValueError, TypeError):
                continue
            return parsed.dt.tz_localize(None) if getattr(parsed.dt, "tz", None) is not None else parsed
    return None


def _is_text(values: pd.Series, n_rows: int, cfg: PrepConfig) -> bool:
    sample = _sample(values).astype(str)
    if not len(sample):
        return False
    words = sample.str.split().str.len().mean()
    return (sample.str.len().mean() >= cfg.text_min_len and words >= cfg.text_min_words
            and values.nunique(dropna=True) > max(50, 0.05 * n_rows))


def detect_kind(values: pd.Series, n_rows: int, hints: dict[str, set], cfg: PrepConfig) -> str:
    """num | cat | date | text | numstr (числа, записанные строками)."""
    name = values.name
    if name in hints["text"]:
        return "text"
    if name in hints["date"] or pd.api.types.is_datetime64_any_dtype(values):
        return "date"
    if name in hints["cat"]:
        return "cat"
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
        return "num"
    if _numeric_like(values, cfg.parse_share):
        return "numstr"
    if _parse_dates(_sample(values, 500), cfg.parse_share) is not None:
        return "date"
    if _is_text(values, n_rows, cfg):
        return "text"
    return "cat"


# ---------------------------------------------------------------- построение признаков

def _date_features(name: str, dates: pd.Series) -> dict[str, pd.Series]:
    out = {f"{name}__year": dates.dt.year, f"{name}__month": dates.dt.month, f"{name}__day": dates.dt.day,
           f"{name}__dow": dates.dt.dayofweek, f"{name}__doy": dates.dt.dayofyear}
    if ((dates.dt.hour.fillna(0) != 0) | (dates.dt.minute.fillna(0) != 0)).any():
        out[f"{name}__hour"] = dates.dt.hour + dates.dt.minute / 60.0
    out[f"{name}__ts"] = (dates - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
    return {k: v.astype("float32") for k, v in out.items()}


def _text_features(name: str, text: pd.Series, n_svd: int, seed: int, use_svd: bool) -> dict[str, pd.Series]:
    filled = text.fillna("").astype(str)
    out = {f"{name}__len": filled.str.len().astype("float32"),
           f"{name}__words": filled.str.split().str.len().astype("float32"),
           f"{name}__digits": filled.str.count(r"\d").astype("float32")}
    out[f"{name}__len"][text.isna()] = np.nan
    if not use_svd:
        return out
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        matrix = TfidfVectorizer(min_df=3, max_features=50_000, ngram_range=(1, 2), sublinear_tf=True,
                                 dtype=np.float32).fit_transform(filled)
        k = min(n_svd, matrix.shape[1] - 1)
        if k >= 2:
            comps = TruncatedSVD(n_components=k, random_state=seed).fit_transform(matrix)
            for i in range(k):
                out[f"{name}__svd{i}"] = pd.Series(comps[:, i].astype("float32"), index=text.index)
    except Exception as error:  # noqa: BLE001 — текстовые признаки необязательны
        log.warning("TF-IDF для %s пропущен: %s", name, error)
    return out


def _category(values: pd.Series, n_train: int, min_count: int) -> tuple[pd.Series | None, pd.Series]:
    """(категория с редкими → RARE или None, если уровней < 2; частота значения по train+test)."""
    labels = values.astype(object).where(values.isna(), values.map(_label))
    freq = labels.map(labels.value_counts(dropna=True)).astype("float32") / len(labels)
    counts = labels.iloc[:n_train].value_counts(dropna=True)
    keep = counts[counts >= min_count].index
    merged = labels.where(labels.isna() | labels.isin(keep), RARE)
    levels = merged.iloc[:n_train].value_counts(dropna=True).index.tolist()  # по убыванию частоты: коды = ранг
    levels += [v for v in merged.iloc[n_train:].dropna().unique() if v not in set(levels)]  # RARE только в test
    if len(levels) < 2:
        return None, freq
    return pd.Series(pd.Categorical(merged, categories=levels), index=values.index), freq


def _label(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _safe_names(columns: list[str]) -> dict[str, str]:
    """LightGBM/XGBoost падают на символах JSON и [ ] < в именах признаков — заменяем их на «_»."""
    mapping, used = {}, set()
    for col in columns:
        base = re.sub(r"[^\w]", "_", str(col)).strip("_") or "col"
        name, i = base, 1
        while name in used:
            name, i = f"{base}_{i}", i + 1
        used.add(name)
        mapping[col] = name
    return mapping


def _drop_duplicates(frame: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    """Колонки, совпадающие с более ранней (по значениям), → {дубль: оригинал}."""
    seen: dict[int, list[str]] = {}
    dropped = {}
    for col in columns:
        key = int(pd.util.hash_pandas_object(frame[col], index=False).sum())
        twin = next((other for other in seen.get(key, []) if frame[other].equals(frame[col])), None)
        if twin is not None:
            dropped[col] = twin
        else:
            seen.setdefault(key, []).append(col)
    return dropped


def prepare(train: pd.DataFrame, test: pd.DataFrame, spec: Any = None, config: PrepConfig | None = None) -> Prepared:
    """Признаки train/test (без таргета и id) → Prepared. Строки и их порядок не меняются."""
    cfg = config or PrepConfig()
    n_train = len(train)
    columns = [c for c in train.columns if c in test.columns]
    full = pd.concat([train[columns], test[columns]], ignore_index=True)
    hints = {key: set(getattr(spec, f"{key}_cols", None) or []) & set(columns) for key in ("cat", "date", "text")}
    spec_drop = set(getattr(spec, "drop_cols", None) or [])
    min_count = cfg.min_count if n_train >= 1000 else 2

    kinds: dict[str, str] = {}
    dropped: dict[str, str] = {}
    out: dict[str, pd.Series] = {}
    cat_cols, texts, high_card = [], [], []
    for col in columns:
        values = full[col]
        if col in spec_drop:
            dropped[col] = "spec.drop_cols"
            continue
        train_part, test_part = values.iloc[:n_train], values.iloc[n_train:]
        if train_part.nunique(dropna=False) <= 1:
            dropped[col] = "constant"
            continue
        if test_part.isna().all() and len(test_part) and train_part.notna().mean() > 0.5:
            dropped[col] = "empty_in_test"
            continue
        kind = detect_kind(values, len(full), hints, cfg)
        kinds[col] = kind
        unique_in_train = train_part.nunique(dropna=True) == n_train and n_train > 20
        if unique_in_train and _ID_NAME.search(str(col)) and kind in ("num", "cat", "numstr"):
            dropped[col] = "id_like"
            continue
        if kind == "num":
            numbers = values.astype("float64") if pd.api.types.is_bool_dtype(values) else values
            out[col] = numbers.replace([np.inf, -np.inf], np.nan)
        elif kind == "numstr":
            out[col] = _clean_numeric_text(values).replace([np.inf, -np.inf], np.nan)
        elif kind == "date":
            dates = values if pd.api.types.is_datetime64_any_dtype(values) else _parse_dates(values, 0.5)
            if dates is None:  # подсказка spec.date_cols, но значения не даты — считаем категорией
                kinds[col] = kind = "cat"
            else:
                if getattr(dates.dt, "tz", None) is not None:
                    dates = dates.dt.tz_localize(None)
                out.update(_date_features(col, dates))
        if kind == "text":
            texts.append(col)
        if kind == "cat":
            if unique_in_train and values.astype(str).str.len().mean() >= 8:
                dropped[col] = "unique_strings"  # uuid, хэши, ФИО: уникальны в каждой строке, обобщать нечего
                continue
            category, freq = _category(values, n_train, min_count)
            if category is not None:
                out[col] = category
                cat_cols.append(col)
                if len(category.cat.categories) > cfg.max_cat_levels:
                    high_card.append(col)
            if values.nunique(dropna=True) >= cfg.freq_min_levels or category is None:
                out[f"{col}__freq"] = freq

    use_svd = len(full) <= cfg.max_text_rows
    for i, col in enumerate(texts):
        out.update(_text_features(col, full[col], cfg.svd_components, cfg.seed, use_svd and i < cfg.max_text_cols))

    frame = pd.DataFrame(out, index=full.index)
    derived_const = [c for c in frame.columns if c not in columns and frame[c].iloc[:n_train].nunique(dropna=False) <= 1]
    frame = frame.drop(columns=derived_const)
    duplicates = _drop_duplicates(frame, list(frame.columns))
    frame = frame.drop(columns=list(duplicates))
    dropped.update({col: f"duplicate_of:{twin}" for col, twin in duplicates.items() if col in columns})
    cat_cols = [c for c in cat_cols if c in frame.columns]
    high_card = [c for c in high_card if c in frame.columns]

    if frame.size > cfg.float32_cells:
        floats = frame.select_dtypes("float64").columns
        frame[floats] = frame[floats].astype("float32")

    names = _safe_names(list(frame.columns))
    frame = frame.rename(columns=names)
    cat_cols, high_card = [names[c] for c in cat_cols], [names[c] for c in high_card]

    x_train = frame.iloc[:n_train].reset_index(drop=True)
    x_test = frame.iloc[n_train:].reset_index(drop=True)
    report = {
        "n_input": len(columns), "n_features": frame.shape[1],
        "kinds": kinds, "dropped": dropped,
        "categorical": cat_cols, "high_card": high_card, "text": texts,
        "dates": [c for c, k in kinds.items() if k == "date"],
        "numeric_strings": [c for c, k in kinds.items() if k == "numstr"],
        "renamed": {new: old for old, new in names.items() if new != old},
        "duplicate_train_rows": int(train[columns].duplicated().sum()) if columns else 0,
    }
    log.info("признаки: %d входных → %d (категорий %d, дат %d, текстов %d), удалено %d: %s",
             len(columns), frame.shape[1], len(cat_cols), len(report["dates"]), len(texts), len(dropped),
             ", ".join(f"{c} ({r})" for c, r in list(dropped.items())[:8]))
    return Prepared(x_train, x_test, cat_cols, high_card, report)


# ---------------------------------------------------------------- адаптер под модель

def model_frame(prep: Prepared, X: pd.DataFrame, model: str) -> tuple[pd.DataFrame, list[str]]:
    """Кадр в виде, который ест модель, и список категориальных колонок для неё.

    lightgbm / xgboost — category как есть, высококардинальные → коды по частоте (float);
    catboost — категории строками (NaN → "__nan__"); прочие (sklearn) — все категории кодами.
    """
    frame = X.copy()
    if model == "catboost":
        for col in prep.cat_cols:
            frame[col] = frame[col].astype(object).where(frame[col].notna(), NAN).astype(str)
        return frame, list(prep.cat_cols)
    to_codes = prep.high_card if model in ("lightgbm", "xgboost") else prep.cat_cols
    for col in to_codes:
        frame[col] = frame[col].cat.codes.astype("float32").replace(-1, np.nan)
    return frame, [c for c in prep.cat_cols if c not in to_codes]
