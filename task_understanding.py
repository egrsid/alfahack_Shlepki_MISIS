"""Понимание задачи (роль P2; написано P4 — владелец может переписать).

    understand_task(task_path, llm | None) -> spec

Основа — эвристики fallback.guess_spec: таргет и id из sample_submission.csv, метрика из task.md, тип по таргету.
Их не меняем: образец сабмита надёжнее любого пересказа. Дальше — заметки по колонкам из data_dictionary.csv и,
если LLM доступна, один короткий запрос: метрика (только если в task.md её не нашли эвристикой), колонки дат,
свободного текста, категорий и утечек. Каждое поле ответа проверяется по данным (колонка есть в train и test,
метрика есть в metrics.py и подходит к типу таргета, дата парсится, текст длинный); не прошло — остаётся эвристика.
Любая ошибка LLM → spec эвристики. Ключ и адрес LLM — только из переменных окружения (через llm_client).
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

import fallback
import metrics
import submission

log = logging.getLogger("task_understanding")

TASK_CHARS = 3500        # task.md в промпте
DICT_CHARS = 2500        # словарь в промпте
SAMPLE_ROWS = 2000       # строк train для проверки ответа
MAX_DROP = 3             # LLM не может выкинуть больше стольких колонок
MAX_TOKENS = 1500

SYSTEM = ("Role: task_analyst. You are a careful data scientist. Read the ML task and the column summary, answer with ONE JSON object "
          "and nothing else.")
QUESTION = """Fill this JSON (use only column names from the list; empty lists are fine):
{"metric": "main evaluation metric as named in the task, e.g. rmse, mae, macro_f1, roc_auc, logloss",
 "task_type": "regression | binary | multiclass",
 "date_cols": [columns with dates or timestamps],
 "text_cols": [columns with free-form text (sentences), not short categories],
 "cat_cols": [string columns that are categories],
 "drop_cols": [columns the task says are leaks or must not be used; NOT ordinary features]}"""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def read_dictionary(spec: Any) -> dict[str, str]:
    """Заметки по колонкам из data_dictionary.csv: описание [+ единицы]. Нет словаря или формат другой → {}."""
    try:
        frame = pd.read_csv(submission.task_file(spec, "data_dictionary.csv"))
    except Exception as error:  # noqa: BLE001 — словаря может не быть
        log.info("data_dictionary.csv не прочитан: %s", error)
        return {}
    name_col = next((c for c in frame.columns if str(c).lower() in ("feature", "name", "column", "field")),
                    frame.columns[0])
    desc_col = next((c for c in frame.columns if "desc" in str(c).lower()), frame.columns[-1])
    unit_col = next((c for c in frame.columns if str(c).lower() == "unit"), None)
    notes: dict[str, str] = {}
    for _, row in frame.iterrows():
        note = str(row[desc_col])
        if unit_col is not None and pd.notna(row[unit_col]):
            note += f" [{row[unit_col]}]"
        notes[str(row[name_col])] = note
    return notes


def _columns_summary(train: pd.DataFrame, features: list[str]) -> str:
    lines = []
    for col in features:
        values = train[col]
        example = ", ".join(_clip(str(v), 40) for v in values.dropna().unique()[:3])
        lines.append(f"- {col}: {values.dtype}, {values.nunique()} unique, {values.isna().mean():.0%} missing; e.g. {example}")
    return _clip("\n".join(lines), 6000)


def _ask(llm: Any, spec: Any, train: pd.DataFrame, features: list[str], notes: dict[str, str]) -> dict:
    from agent import extract_json  # устойчивый разбор JSON из ответа Qwen (think-теги, ```json, хвосты)

    dictionary = _clip("\n".join(f"- {k}: {v}" for k, v in notes.items()), DICT_CHARS)
    user = (f"# Task description\n{_clip(spec.raw_task_text, TASK_CHARS)}\n\n"
            f"# Target: {spec.target}; id: {spec.id_col}\n\n# Columns (train sample)\n"
            f"{_columns_summary(train, features)}\n\n"
            + (f"# Data dictionary\n{dictionary}\n\n" if dictionary else "") + QUESTION)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user + "\n/no_think"}]
    kw = {"max_tokens": MAX_TOKENS, "temperature": 0.0}
    try:
        text = llm.chat(messages, **kw, extra_body={"reasoning": {"enabled": False}})
    except Exception as error:  # noqa: BLE001 — шлюз не принял подсказку «без рассуждения»: повтор без неё
        if getattr(error, "status_code", None) not in (400, 422):
            raise
        text = llm.chat(messages, **kw)
    answer = extract_json(text)
    if not isinstance(answer, dict):
        raise ValueError(f"expected a JSON object, got {type(answer).__name__}")
    return answer


def _names(value: Any, allowed: list[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v) in allowed]


def _is_date(values: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(values):
        return False
    sample = values.dropna().astype(str).head(500)
    if not len(sample):
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.9)


def _is_text(values: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(values):
        return False
    sample = values.dropna().astype(str)
    return bool(len(sample) and sample.str.len().mean() >= 20 and sample.str.split().str.len().mean() >= 3)


def apply_answer(spec: Any, answer: dict, train: pd.DataFrame, features: list[str]) -> list[str]:
    """Проверенные поля ответа LLM → spec (на месте); возвращает, что принято."""
    applied: list[str] = []
    text_metric = metrics.find_in_text(spec.raw_task_text or "")
    raw_metric = answer.get("metric")
    if text_metric is None and isinstance(raw_metric, str):  # метрику из текста эвристика уже нашла — она главнее
        try:
            name = metrics.resolve(raw_metric)
            regression = spec.task_type == "regression"
            if (metrics.get(name).task == "regression") == regression:
                if name != spec.metric:
                    applied.append(f"metric {spec.metric}→{name}")
                spec.metric, spec.greater_is_better = name, metrics.greater_is_better(name)
            else:
                log.info("понимание: метрика %s не подходит к типу %s — пропущена", name, spec.task_type)
        except (metrics.UnknownMetric, KeyError, TypeError):
            log.info("понимание: метрика %r не из реестра — пропущена", raw_metric)

    dates = [c for c in _names(answer.get("date_cols"), features) if _is_date(train[c])]
    texts = [c for c in _names(answer.get("text_cols"), features) if c not in dates and _is_text(train[c])]
    cats = [c for c in _names(answer.get("cat_cols"), features)
            if c not in dates and c not in texts and not pd.api.types.is_numeric_dtype(train[c])]
    mentioned = (spec.raw_task_text or "")
    drops = [c for c in _names(answer.get("drop_cols"), features) if c in mentioned][:MAX_DROP]
    for field, values in (("date_cols", dates), ("text_cols", texts), ("cat_cols", cats), ("drop_cols", drops)):
        merged = list(dict.fromkeys([*(getattr(spec, field, None) or []), *values]))
        if merged != list(getattr(spec, field, None) or []):
            setattr(spec, field, merged)
            applied.append(f"{field}={values}")
    return applied


def understand_task(task_path: str | Path, llm: Any = None) -> Any:
    spec = fallback.guess_spec(task_path)
    if isinstance(spec.target, list):  # мульти-таргет — только эвристики
        return spec
    base = copy.copy(spec)
    try:
        notes = read_dictionary(spec)
        spec.feature_notes = {**notes, **(getattr(spec, "feature_notes", None) or {})}
        if llm is None:
            return spec
        train = pd.read_csv(submission.task_file(spec, "train.csv"), nrows=SAMPLE_ROWS, low_memory=False)
        test_cols = set(pd.read_csv(submission.task_file(spec, "test.csv"), nrows=0).columns)
        features = [c for c in train.columns if c in test_cols and c not in (spec.id_col, spec.target)]
        answer = _ask(llm, spec, train, features, notes)
        applied = apply_answer(spec, answer, train, features)
        if applied:
            spec.source = "llm"
        log.info("понимание задачи (LLM): %s", "; ".join(applied) or "ничего нового")
        log.debug("ответ LLM: %s", json.dumps(answer, ensure_ascii=False)[:2000])
        return spec
    except Exception as error:  # noqa: BLE001 — любая ошибка → эвристика
        log.warning("понимание задачи через LLM не удалось (%s: %s) — эвристика", type(error).__name__,
                    str(error)[:300])
        base.feature_notes = getattr(spec, "feature_notes", None) or {}
        return base
