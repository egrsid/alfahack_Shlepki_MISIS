"""Промпты агента AutoResearch и сборщики сообщений с лимитами длины.

Роли — разные системные промпты одного агента: planner (аналитик + планировщик одним вызовом,
экономит токены), coder (код по шаблону; он же чинит код по ошибке), critic (разбор неудач).
Инструкции по формату — на английском (у Qwen стабильнее структурированный вывод), описание
задачи и заметки по признакам — как есть, обычно по-русски. В промпт идут только сводки
(профиль, память, код лучшего), никогда — сырые строки данных.
Первая строка каждого системного промпта — «Role: <имя>.»: по ней моки различают роли.
"""

from __future__ import annotations

import json
import math
from typing import Any

KINDS = ("features", "encoding", "missing", "outliers", "target_transform", "params", "model",
         "feature_selection")
LEVELS = ("low", "mid", "high")
TUNE_KIND = "tune"  # подбор гиперпараметров тюнером (P1); предлагается, только если тюнер есть
TUNE_KIND_LINE = ("- tune: automatic hyperparameter search for the current model on the current features "
                  "(no code is written; title like \"Tune lightgbm hyperparameters\"); best after features stop improving")
MODEL_SEARCH_KIND = "model_search"  # сравнить все бустинги на текущих признаках тем же evaluator, без кодера
RECIPE_KINDS = ("features", "encoding", "missing")  # режим «поверх fallback»: только новые колонки
RECIPE_RULE = ("A strong baseline (preprocessing, models, their blend, class weights) is fixed: an experiment may "
               "only ADD new columns (at most 8); changing or dropping existing columns has no effect. Read "
               "\"Baseline facts\": never rebuild what it already has; use importance to pick strong columns to "
               "combine; for drifting columns prefer ratios or values relative to a group.")
MODEL_SEARCH_KIND_LINE = ("- model_search: compare lightgbm, catboost and xgboost on the current features, keep the "
                          "best (no code; title like \"Compare boosting models\")")

# без enum: мелкие расхождения («feature», «medium») нормализует агент, а не ремонт JSON у клиента
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "title": {"type": "string"}, "kind": {"type": "string"},
                               "rationale": {"type": "string"}, "expected_gain": {"type": "string"},
                               "cost": {"type": "string"}},
                "required": ["title"],
            },
        },
    },
    "required": ["hypotheses"],
}


def estimate_tokens(text: str) -> int:
    """Грубая верхняя оценка: ~4 символа ASCII или ~2 символа кириллицы на токен."""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return math.ceil(ascii_chars / 4 + (len(text) - ascii_chars) / 2)


def messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m["content"]) + 4 for m in messages)


def clip(text: str, max_chars: int, tail: bool = False) -> str:
    """Обрезать до max_chars; tail=True — оставить конец (traceback)."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    marker = "...[truncated]..."
    keep = max(0, max_chars - len(marker))
    return marker + text[-keep:] if tail else text[:keep] + marker


def _num(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "nan" if value != value else f"{value:.4g}"
    return str(value)


def render_profile(profile: dict, notes: dict[str, str] | None = None, max_chars: int = 5000,
                   note_chars: int = 40) -> str:
    """Профиль данных → компактный текст: строки таблицы по колонкам, заметки из словаря данных."""
    notes = notes or {}
    columns = profile.get("columns") if isinstance(profile, dict) else None
    if not (isinstance(columns, list) and all(isinstance(c, dict) and "name" in c for c in columns)):
        return clip(json.dumps(profile, ensure_ascii=False, default=str), max_chars)

    head = [f"rows: train {profile.get('n_train', '?')}, test {profile.get('n_test', '?')}; "
            f"features: {len(columns)}"]
    target = profile.get("target") or {}
    if target:
        stats = ", ".join(f"{k} {_num(v)}" for k, v in target.items() if k not in ("name", "classes"))
        line = f"target {target.get('name', '')}: {stats}"
        if target.get("classes"):
            line += " classes: " + ", ".join(f"{k}={v}" for k, v in list(target["classes"].items())[:15])
        head.append(clip(line, 600))
    head.append("columns (name | type | missing% | unique | range or top values | note):")

    lines, used = [], sum(len(h) + 1 for h in head)
    for i, col in enumerate(columns):
        if "min" in col:
            values = f"{_num(col['min'])}..{_num(col['max'])}"
        else:
            values = ", ".join(clip(str(v), 18) for v in (col.get("top") or [])[:3])
        note = clip(notes.get(col["name"], ""), note_chars)
        line = (f"{col['name']} | {col.get('dtype', '?')} | {_num(col.get('missing_pct', 0))} | "
                f"{col.get('nunique', '?')} | {values}" + (f" | {note}" if note else ""))
        if used + len(line) + 40 > max_chars:
            lines.append(f"... and {len(columns) - i} more columns")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(head + lines)


def render_columns_brief(profile: dict, max_chars: int = 2500) -> str:
    """Для кодера: имена колонок по типам (без статистик)."""
    columns = profile.get("columns") if isinstance(profile, dict) else None
    if not isinstance(columns, list):
        return clip(json.dumps(profile, ensure_ascii=False, default=str), max_chars)
    groups: dict[str, list[str]] = {}
    for col in columns:
        name = col["name"]
        if col.get("dtype") in ("cat", "text"):
            name += f" ({col.get('nunique', '?')} levels)"
        groups.setdefault(col.get("dtype", "?"), []).append(name)
    return clip("\n".join(f"{dtype}: {', '.join(names)}" for dtype, names in groups.items()), max_chars)


def params_help(model_params: dict[str, dict]) -> str:
    return "\n".join(f"  {model}: {', '.join(keys)}" for model, keys in model_params.items())


# ============================================================ системные промпты

PLANNER_SYSTEM = """Role: planner.
You are a senior data scientist (data analyst + experiment planner) driving an automated tabular ML pipeline. \
The pipeline trains gradient boosting (lightgbm, catboost or xgboost) with fixed K-fold CV. Each experiment \
changes ONE thing in the current best solution and is kept only if CV improves beyond noise.

First study the data profile: column types, missing values, cardinality, skew, ranges and the domain meaning \
from the notes. Then propose up to {n} NEW hypotheses, ordered by expected gain / cost.

Hypothesis kinds:
- features: new columns from existing ones (ratios, differences, sums, group aggregates, date parts, counts)
- encoding: categorical handling (frequency/count encoding, grouping rare levels, combining columns)
- missing: missing-value indicators or imputation
- outliers: clipping or winsorizing extreme feature values
- target_transform: log1p of the target (regression with a non-negative skewed target only); it optimizes \
relative error, so it suits RMSLE/MAPE-like metrics and often hurts RMSE/MAE on the raw scale
- params: gradient boosting hyperparameters
- model: switch between lightgbm, catboost and xgboost
- feature_selection: drop noisy, constant, duplicated or ID-like columns

Rules:
- One small concrete change per hypothesis; name the columns it uses.
- Trees are invariant to monotonic transforms of one feature (log, sqrt, scaling, rank) and handle NaN \
natively: such changes and plain median/mean imputation give zero gain. Prefer interactions, ratios, \
differences, group statistics and counts across rows, which trees cannot build themselves.
- Use the domain meaning of columns (notes): physical rates and per-unit quantities, differences of related \
columns, a value relative to its group mean. Avoid blind products of unrelated columns.
- The target is NOT available to feature code: no target encoding, no target statistics.
- No external data or pretrained models (text: TF-IDF + SVD); never use row order or row index as a feature.
- Do not repeat what the memory says was already tried.
- If an error analysis is given, prefer hypotheses that target the worst segments, classes or target ranges.

Answer with ONE JSON object and nothing else:
{{"analysis": "<= 60 words: key data insights", "hypotheses": [{{"id": "h1", "title": "...", "kind": "features", \
"rationale": "...", "expected_gain": "low|mid|high", "cost": "low|mid|high"}}]}}

Example answer:
{{"analysis": "income is right-skewed with 5% missing; city has 300 levels with a long tail; age and tenure are \
strongly related.", "hypotheses": [{{"id": "h1", "title": "Ratio income / household_size", "kind": "features", \
"rationale": "Per-capita income is more informative than raw income", "expected_gain": "mid", "cost": "low"}}, \
{{"id": "h2", "title": "Frequency encoding of city", "kind": "encoding", "rationale": "Rare levels give noisy \
splits", "expected_gain": "low", "cost": "low"}}]}}"""

_LAST_KIND_LINE = "- feature_selection: drop noisy, constant, duplicated or ID-like columns"
assert _LAST_KIND_LINE in PLANNER_SYSTEM

CODER_SYSTEM = """Role: coder.
You write the Python module that a harness uses for one experiment of a tabular ML pipeline.

Contract of `build_features(train, test, spec)`:
- `train` and `test` are pandas DataFrames with the raw feature columns (no target, no id column).
- `spec` is a dict: task_type, metric, target (name only), cat_cols, date_cols, text_cols, feature_notes.
- Return `(X_train, X_test)`: DataFrames with the SAME number of rows and the SAME row order as the inputs \
and identical columns in both. Never sort, drop, filter or duplicate rows.
- The target is not available: no target encoding. Unsupervised statistics (counts, frequencies, group means of \
features) may be computed on train and test concatenated.
- String/object columns are fine (the harness makes them categorical); NaN is fine; never produce inf \
(replace division by zero with NaN).
- Allowed imports: numpy, pandas, scipy, sklearn (unsupervised transforms), math, re, collections, itertools, \
datetime. Forbidden: files, network, os/sys/subprocess, open/eval/exec, printing, top-level code other than \
imports, functions and constants. Never use the row index or row position as a feature.
- Vectorized pandas/numpy only (no Python loops over rows): it must finish in {timeout} s.
- Text columns (spec["text_cols"]): sklearn TfidfVectorizer (words or char n-grams) + TruncatedSVD fitted on \
train and test together, keyword flags, lengths. No pretrained models or embeddings.

Optional module-level constants (literal values only):
MODEL = "lightgbm"        # lightgbm | catboost | xgboost
PARAMS = {{}}               # hyperparameters; allowed keys:
{params_help}
TARGET_TRANSFORM = None   # or "log1p" (regression with target > -1 only)

Implement ONLY the given hypothesis on top of the current best module and keep everything else unchanged. \
Answer with exactly ONE ```python code block containing the COMPLETE module. No explanations.

Example answer:
```python
import numpy as np
import pandas as pd


def build_features(train, test, spec):
    X_train, X_test = train.copy(), test.copy()
    for df in (X_train, X_test):
        df["income_per_person"] = df["income"] / df["household_size"].replace(0, np.nan)
    return X_train, X_test


MODEL = "lightgbm"
PARAMS = {{"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 63}}
TARGET_TRANSFORM = None
```"""

CRITIC_SYSTEM = """Role: critic.
You review experiments of an automated tabular ML agent. In at most 120 words: what did not work and the likely \
reason, then what to try instead (concrete directions that differ from what was tried). Plain text, no code, \
no JSON."""


# ============================================================ сборщики сообщений

def _hypothesis_text(hypothesis: dict) -> str:
    text = f"[{hypothesis.get('kind', 'features')}] {hypothesis.get('title', '')}"
    if hypothesis.get("rationale"):
        text += f"\nWhy: {clip(hypothesis['rationale'], 300)}"
    return text


_PLANNER_INTRO = "with fixed K-fold CV. Each experiment changes ONE thing in the current best solution"
_RECIPE_DROP_KINDS = ("- outliers:", "- target_transform:", "relative error, so it suits", "- params:", "- model:",
                      "- feature_selection:")
assert _PLANNER_INTRO in PLANNER_SYSTEM


def _recipe_planner(system: str) -> str:
    """Планировщик «поверх fallback»: без видов гипотез про модель/параметры/удаление, с правилом про новые колонки."""
    lines = [line for line in system.split("\n") if not line.startswith(_RECIPE_DROP_KINDS)]
    return "\n".join(lines).replace("\nRules:\n", f"\nRules:\n- {RECIPE_RULE}\n", 1)


def build_plan_messages(*, task_line: str, task_text: str, profile: str, base_code: str, cv_text: str,
                        memory: str, n: int = 6, task_chars: int = 1200, profile_chars: int = 5000,
                        code_chars: int = 2000, memory_chars: int = 1500, errors: str = "",
                        errors_chars: int = 800, can_tune: bool = False,
                        can_search_models: bool = False, recipe: bool = False, signals: str = "",
                        signals_chars: int = 1600, retry: int = 0) -> list[dict]:
    if signals:  # сигналы рецепта дороже длинного профиля: профиль ужимается, чтобы вызов остался в бюджете
        profile_chars = min(profile_chars, 3400)
    user = (f"## Task\n{task_line}\n{clip(task_text, task_chars)}\n\n"
            f"## Data profile\n{clip(profile, profile_chars)}\n\n"
            + (f"## Baseline facts (what the fixed baseline already does) and signals\n"
               f"{clip(signals, signals_chars)}\n\n" if signals else "")
            + f"## Current best solution ({cv_text})\n```python\n{clip(base_code, code_chars)}\n```\n\n"
            + (f"## Where the current model errs\n{clip(errors, errors_chars)}\n\n" if errors else "")
            + f"## Memory\n{clip(memory, memory_chars)}\n\n"
            + (f"Attempt {retry + 1}: your previous plan only repeated ideas from Memory. Propose clearly "
               "DIFFERENT directions: columns no experiment has used yet, other operations (group statistics, "
               "differences of related columns, counts, ranks within a group), other combinations of strong "
               "columns.\n" if retry else "")
            + f"Propose up to {n} new hypotheses as ONE JSON object.")
    system = PLANNER_SYSTEM.format(n=n)
    if recipe:
        system = _recipe_planner(system)
    extra_kinds = ([MODEL_SEARCH_KIND_LINE] if can_search_models else []) + ([TUNE_KIND_LINE] if can_tune else [])
    if extra_kinds:
        system = system.replace(_LAST_KIND_LINE, "\n".join([_LAST_KIND_LINE, *extra_kinds]))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_CODER_CONSTANTS = CODER_SYSTEM[CODER_SYSTEM.index("Optional module-level constants"):
                                CODER_SYSTEM.index("Implement ONLY the given hypothesis")]
_CODER_EXAMPLE_CONSTANTS = CODER_SYSTEM[CODER_SYSTEM.index('\n\nMODEL = "lightgbm"\nPARAMS = {{"n_estimators"'):
                                        CODER_SYSTEM.rindex("\n```")]
_CODER_RETURN = "- Return `(X_train, X_test)`:"


def _coder_system(model_params: dict[str, dict], timeout: float, recipe: bool = False) -> str:
    system = CODER_SYSTEM
    if recipe:  # модель и параметры — у рецепта fallback; LLM только добавляет колонки
        system = (system.replace(_CODER_CONSTANTS, "").replace(_CODER_EXAMPLE_CONSTANTS, "")
                  .replace(_CODER_RETURN, f"- Keep every input column and ADD new ones: only new column names reach "
                                          f"the model (the originals are already used, preprocessed).\n{_CODER_RETURN}"))
    return system.format(params_help=params_help(model_params), timeout=int(timeout))


def build_code_messages(*, task_line: str, columns: str, insights: str, base_code: str, cv_text: str,
                        hypothesis: dict, memory: str, model_params: dict[str, dict], timeout: float,
                        columns_chars: int = 2500, memory_chars: int = 800, recipe: bool = False) -> list[dict]:
    user = (f"## Task\n{task_line}\n\n"
            f"## Columns of `train`/`test` (raw input of build_features)\n{clip(columns, columns_chars)}\n\n"
            + (f"## Data insights\n{clip(insights, 500)}\n\n" if insights else "")
            + f"## Current best module ({cv_text})\n```python\n{base_code.strip()}\n```\n\n"
            f"## Hypothesis to implement\n{_hypothesis_text(hypothesis)}\n\n"
            f"## Memory (already tried)\n{clip(memory, memory_chars)}\n\n"
            "Return the COMPLETE updated module in ONE ```python block.")
    return [{"role": "system", "content": _coder_system(model_params, timeout, recipe)}, {"role": "user", "content": user}]


def build_repair_messages(*, task_line: str, columns: str, hypothesis: dict, code: str, error: str,
                          model_params: dict[str, dict], timeout: float, error_chars: int = 1500,
                          columns_chars: int = 2500, recipe: bool = False) -> list[dict]:
    user = (f"## Task\n{task_line}\n\n"
            f"## Columns of `train`/`test` (raw input of build_features)\n{clip(columns, columns_chars)}\n\n"
            f"## Hypothesis\n{_hypothesis_text(hypothesis)}\n\n"
            f"## Your module\n```python\n{code.strip()}\n```\n\n"
            f"## It failed with\n```\n{clip(error, error_chars, tail=True)}\n```\n\n"
            "Fix the problem and keep implementing the hypothesis. "
            "Return the COMPLETE fixed module in ONE ```python block.")
    return [{"role": "system", "content": _coder_system(model_params, timeout, recipe)}, {"role": "user", "content": user}]


def build_format_fix_messages(messages: list[dict], bad_answer: str) -> list[dict]:
    return [*messages,
            {"role": "assistant", "content": clip(bad_answer, 1500)},
            {"role": "user", "content": "Your answer has no ```python block with `def build_features`. Reply with "
                                        "ONLY the complete module in ONE ```python block."}]


def build_critic_messages(*, task_line: str, memory: str, memory_chars: int = 1500) -> list[dict]:
    user = (f"## Task\n{task_line}\n\n## Experiments\n{clip(memory, memory_chars)}\n\n"
            "What did not work, why, and what to try instead (<= 120 words)?")
    return [{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": user}]
