"""AutoResearch-агент (P4): LLM-управляемый цикл улучшения лучшего кандидата.

research(spec, best, llm, deadline) -> list[Candidate]:
гипотеза (LLM) → код build_features (LLM) → песочница → оценка харнессом (evaluator) →
решение «принять/отклонить» по шуму CV → память → следующая итерация.

Агент никогда не пишет submission.csv и не роняет процесс: при любом сбое возвращает
уже принятых кандидатов (или []). Модули команды (schemas, llm_client, evaluator, sandbox,
eda, guards) подключаются через адаптеры ниже, поэтому agent.py импортируется и без них.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
import types
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

import prompts_agent as P
from agent_eval import Candidate, LocalEvaluator, find_file  # Candidate — из schemas.py (P1), если он есть
from error_analysis import analyze_errors
from memory import Experiment, ExperimentMemory, code_diff, code_hash, columns_in, normalize_title
from postprocess import apply_weights, fit_class_weights

log = logging.getLogger("agent")

# --- контракт llm_client.py (P2); пока его нет — локальный заменитель ---
try:
    from llm_client import LLMUnavailable
except ImportError:
    class LLMUnavailable(Exception):
        """LLM недоступна: сеть, исчерпаны ретраи, Budget has been exceeded."""


class LLMClient(Protocol):
    def chat(self, messages: list[dict], **kw: Any) -> str: ...
    def chat_json(self, messages: list[dict], schema: dict) -> dict: ...


class Deadline(Protocol):
    def left(self) -> float: ...


# evaluator.evaluate(spec, build_features, params, deadline) -> Candidate  (P1)
EvaluateFn = Callable[[Any, Callable, dict, Any], Candidate]
# sandbox.run_code(code, cwd, timeout) -> ExecResult(stdout, stderr, returncode, artifacts)  (P5)
RunCodeFn = Callable[[str, Path, float], Any]
# guards.check_candidate(spec, cand, baseline) -> list[str], пусто = ок  (P5)
GuardFn = Callable[[Any, Candidate, Candidate], list]
# tuning.tune(spec, build_features, cfg, deadline) -> dict гиперпараметров  (P1; tuning.py)
TuneFn = Callable[[Any, Callable, dict, Any], dict]


@dataclass
class AgentConfig:
    """Все числа агента в одном месте; времена — верхние границы, реально режет Deadline."""

    # итерации и стоп-критерии
    max_iters: int = 12
    hypotheses_per_plan: int = 6
    max_empty_plans: int = 2            # подряд планов без новых гипотез → стоп
    max_consecutive_failures: int = 3
    max_no_improve: int = 5             # плато
    max_tokens: int = 120_000           # доля агента в токенах на задачу (вход + выход)
    min_start_seconds: float = 300.0    # если до конца окна research меньше — не начинаем
    llm_reserve_seconds: float = 60.0   # оценка времени на вызовы LLM за одну итерацию
    # песочница и оценка
    local_evaluator: bool = True        # нет evaluator.py (P1) → свой LocalEvaluator (agent_eval.py); False → []
    features_timeout: float = 300.0
    evaluate_timeout: float = 900.0
    max_feature_cols: int = 2000
    max_feature_gb: float = 4.0         # размер признаков train в памяти
    max_nan_share: float = 0.5          # «NaN-лавина»: доля пропусков выше max(этого, входной + 0.25)
    max_repairs: int = 1                # попыток ремонта по traceback / ошибке проверки
    max_format_fixes: int = 1           # попыток «пришли один блок ```python»
    # правило принятия
    accept_k: float = 1.0               # gain > accept_k · SE шума CV
    min_rel_gain: float = 5e-4          # и gain > min_rel_gain · |cv лучшего|
    default_n_folds: int = 5            # если evaluator не отдал fold_scores
    suspicious_k: float = 8.0           # «чудесный» скачок: gain > suspicious_k · SE ...
    suspicious_rel: float = 0.1         # ... и > 10% от |cv| → перепроверка guards
    only_better_than_best: bool = True  # вернуть только кандидатов не хуже входного best
    # LLM. qwen3.8-27b — рассуждающая модель: скрытое рассуждение тратит max_tokens, поэтому потолки с запасом
    plan_max_tokens: int = 3000
    plan_temperature: float = 0.7
    code_max_tokens: int = 4000
    code_temperature: float = 0.2
    critic_max_tokens: int = 1200
    use_critic: bool = True
    no_think: bool = True               # дописывать /no_think (Qwen3: без длинного <think>)
    # подсказка шлюзу «без рассуждения» (формат OpenRouter); идёт в **kw клиента, после ответа 400 не передаём
    llm_extra: dict = field(default_factory=lambda: {"extra_body": {"reasoning": {"enabled": False}}})
    empty_retry_factor: float = 2.5     # пустой ответ (лимит ушёл на рассуждение) → 1 повтор с max_tokens × это
    max_answer_tokens: int = 10_000
    # размеры кусков промпта, символы
    profile_chars: int = 5000
    task_text_chars: int = 1200
    memory_chars: int = 1500
    coder_memory_chars: int = 900       # кодеру: принятые шаги с диффом и 4 последних эксперимента
    errors_chars: int = 800             # сводка «где ошибается модель» для планировщика (0 — не считать)
    # подбор гиперпараметров (вид гипотезы tune; нужен tuning.tune у P1)
    tune_seconds: float = 900.0         # потолок на один подбор ...
    tune_share: float = 0.3             # ... и не больше этой доли остатка окна research
    tune_min_seconds: float = 30.0
    tune_trials: int = 30
    tune_seed: int = 42
    auto_tune: bool = True              # плато/пустой план, а подбора не было → один раз подобрать перед остановкой
    # выбор модели (вид гипотезы model_search): все MODELS на признаках лучшего кода тем же evaluator
    model_search: bool = True           # предлагать планировщику
    model_search_seconds: float = 1200.0  # потолок на один выбор модели ...
    model_search_share: float = 0.35    # ... и не больше этой доли остатка окна research
    model_search_min_seconds: float = 30.0
    auto_model_search: bool = True      # плато/пустой план, а выбора не было → один раз перед остановкой (до tune)
    # конец исследования: множители вероятностей классов под метрику по меткам (macro F1, accuracy, ...)
    postprocess: bool = True
    postprocess_folds: int = 5          # части OOF для честной оценки весов
    error_chars: int = 1500
    # файлы
    workdir_name: str = ".agent_work"
    keep_artifacts: bool = False


def _optional(module: str, attr: str) -> Any:
    """Функция модуля команды, если он уже смержен в main; иначе None. Заглушки (STUB = True) не берём."""
    try:
        found = importlib.import_module(module)
    except ImportError:
        return None
    return None if getattr(found, "STUB", False) else getattr(found, attr, None)


# ============================================================ парсинг ответов LLM

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_TAG = re.compile(r"</?think>", re.I)
_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)


def strip_think(text: str | None) -> str:
    """Убрать рассуждения Qwen. Незакрытый <think> (ответ обрезан) — убрать только тег."""
    return _THINK_TAG.sub("", _THINK_BLOCK.sub("", text or "")).strip()


def extract_code(text: str | None) -> str | None:
    """Код из ответа кодера: первый блок ```python с build_features, иначе первый python-блок.

    Без ограждений — текст от первого import/def, если в нём есть build_features.
    Незакрытый блок (ответ обрезан по max_tokens) берётся до конца: дальше решит compile().
    """
    text = strip_think(text)
    blocks = [body for lang, body in _FENCE.findall(text) if lang.lower() in ("python", "py", "python3", "")]
    for body in blocks:
        if "def build_features" in body:
            return body.strip() + "\n"
    if blocks and blocks[0].strip():
        return blocks[0].strip() + "\n"
    if "def build_features" in text:
        start = re.search(r"^(import |from |def )", text, re.M)
        return text[start.start():].strip() + "\n" if start else None
    return None


def _clean_json(text: str) -> str:
    """Убрать //, /* */ и # комментарии вне строк и висячие запятые."""
    out, i, in_str = [], 0, False
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif text.startswith("//", i) or ch == "#":
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
            continue
        else:
            out.append(ch)
        i += 1
    return re.sub(r",\s*([}\]])", r"\1", "".join(out))


def _balanced(text: str, start: int) -> str | None:
    """Подстрока от открывающей скобки до парной (с учётом строк в обеих кавычках)."""
    depth, quote, i = 0, "", start
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return None


def _is_payload(value: Any) -> bool:
    """Ответ — объект или список объектов (а не случайный «[1]» из текста вокруг)."""
    return isinstance(value, dict) or (isinstance(value, list) and any(isinstance(v, dict) for v in value))


def _first_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch in "{["][:50]
    for i in starts:
        for variant in (text[i:], _clean_json(text[i:])):
            try:
                value = decoder.raw_decode(variant)[0]
                if _is_payload(value):
                    return value
            except json.JSONDecodeError:
                pass
        chunk = _balanced(text, i)
        if chunk:  # одинарные кавычки, True/False/None — как питоновский литерал
            pythonish = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", chunk)))
            try:
                value = ast.literal_eval(_clean_json(pythonish))
                if _is_payload(value):
                    return value
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                pass
    return None


def extract_json(text: str | None) -> Any:
    """Первый JSON-объект/массив в ответе: сначала из ```json блоков, потом из всего текста."""
    text = strip_think(text)
    fenced = [body for lang, body in _FENCE.findall(text) if lang.lower() in ("json", "")]
    for chunk in [*fenced, text]:
        value = _first_json(chunk)
        if value is not None:
            return value
    raise ValueError("no JSON object found in the answer")


# ============================================================ статическая проверка кода

ALLOWED_IMPORTS = {"numpy", "pandas", "scipy", "sklearn", "math", "re", "itertools", "functools", "collections",
                   "datetime", "statistics", "string", "typing", "warnings", "operator", "calendar",
                   "unicodedata", "heapq", "bisect", "__future__"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "breakpoint", "input", "exit", "quit", "help", "memoryview"}
FORBIDDEN_ATTRS = {"to_csv", "to_parquet", "to_pickle", "to_json", "to_excel", "to_hdf", "to_sql", "to_feather",
                   "to_stata", "to_orc", "to_clipboard", "tofile", "fromfile", "save", "savez", "savez_compressed",
                   "savetxt", "load", "loadtxt", "genfromtxt", "fromregex", "memmap", "system", "popen", "remove",
                   "unlink", "rmtree", "chdir"}
_TOP_LEVEL_OK = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign)


def validate_code(code: str) -> list[str]:
    """Проблемы кода кодера без его исполнения; пустой список — можно запускать в песочнице."""
    try:
        tree = ast.parse(code)
        compile(tree, "<features>", "exec")
    except SyntaxError as error:
        return [f"SyntaxError: {error.msg} (line {error.lineno})"]
    except (ValueError, MemoryError, RecursionError) as error:
        return [f"code does not compile: {error}"]

    problems: list[str] = []
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_features"), None)
    if func is None:
        problems.append("no top-level function `build_features(train, test, spec)`")
    elif len(func.args.posonlyargs) + len(func.args.args) < 3 and func.args.vararg is None:
        problems.append("`build_features` must accept three arguments: (train, test, spec)")

    for node in tree.body:
        docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not isinstance(node, _TOP_LEVEL_OK) and not docstring:
            problems.append(f"top-level `{type(node).__name__}` statement (line {node.lineno}) is not allowed: "
                            "only imports, functions and constants at module level")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                problems.append("relative imports are not allowed")
                continue
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of `{name}` is not allowed (allowed: numpy, pandas, scipy, sklearn, "
                                    "math, re, collections, itertools, ...)")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            problems.append(f"use of `{node.id}` is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                problems.append(f"access to dunder attribute `{node.attr}` is not allowed")
            elif node.attr in FORBIDDEN_ATTRS or node.attr.startswith("read_"):
                problems.append(f"`.{node.attr}` is not allowed: no file, network or OS access")
    return list(dict.fromkeys(problems))


# белый список гиперпараметров: (min, max) — число в границах; set — одно из значений
MODELS = ("lightgbm", "catboost", "xgboost")
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "lightgbm": {"n_estimators": (10, 5000), "learning_rate": (1e-3, 0.5), "num_leaves": (2, 1024),
                 "max_depth": (-1, 16), "min_child_samples": (1, 10_000), "min_child_weight": (0.0, 1e3),
                 "subsample": (0.1, 1.0), "subsample_freq": (0, 10), "colsample_bytree": (0.1, 1.0),
                 "reg_alpha": (0.0, 1e3), "reg_lambda": (0.0, 1e3), "min_split_gain": (0.0, 10.0),
                 "max_bin": (15, 1023), "cat_smooth": (0.0, 1e3), "cat_l2": (0.0, 1e3),
                 "max_cat_threshold": (1, 1024), "min_data_per_group": (1, 10_000), "path_smooth": (0.0, 1e3),
                 "class_weight": {"balanced", None}, "extra_trees": {True, False}},
    "catboost": {"iterations": (10, 5000), "learning_rate": (1e-3, 0.5), "depth": (1, 10),
                 "l2_leaf_reg": (0.0, 1e3), "random_strength": (0.0, 100.0), "bagging_temperature": (0.0, 100.0),
                 "border_count": (16, 1024), "one_hot_max_size": (1, 255), "min_data_in_leaf": (1, 10_000),
                 "auto_class_weights": {"Balanced", "SqrtBalanced", None}},
    "xgboost": {"n_estimators": (10, 5000), "learning_rate": (1e-3, 0.5), "max_depth": (1, 16),
                "min_child_weight": (0.0, 1e3), "subsample": (0.1, 1.0), "colsample_bytree": (0.1, 1.0),
                "reg_alpha": (0.0, 1e3), "reg_lambda": (0.0, 1e3), "gamma": (0.0, 100.0), "max_bin": (16, 1024),
                "max_cat_to_onehot": (1, 255)},
}
CONSTANTS = ("MODEL", "PARAMS", "TARGET_TRANSFORM")
# время обучения относительно lightgbm (грубо): model_search пропускает модель, которая не влезет в бюджет
MODEL_COST = {"lightgbm": 1.0, "xgboost": 1.5, "catboost": 3.0}
# одно понятие у разных моделей; None — аналога нет, параметр при смене модели отбрасывается
_SHARED_PARAMS = (
    {"lightgbm": "n_estimators", "catboost": "iterations", "xgboost": "n_estimators"},
    {"lightgbm": "learning_rate", "catboost": "learning_rate", "xgboost": "learning_rate"},
    {"lightgbm": "max_depth", "catboost": "depth", "xgboost": "max_depth"},
    {"lightgbm": "reg_lambda", "catboost": "l2_leaf_reg", "xgboost": "reg_lambda"},
    {"lightgbm": "min_child_samples", "catboost": "min_data_in_leaf", "xgboost": None},
    {"lightgbm": "class_weight", "catboost": "auto_class_weights", "xgboost": None},
    {"lightgbm": "subsample", "catboost": None, "xgboost": "subsample"},
    {"lightgbm": "colsample_bytree", "catboost": None, "xgboost": "colsample_bytree"},
)
_CLASS_WEIGHTS = {"balanced": "Balanced", "Balanced": "balanced", "SqrtBalanced": "balanced"}


def translate_params(params: dict, src: str, dst: str) -> tuple[dict, list[str]]:
    """Гиперпараметры модели src → dst: общие понятия переносятся, остальное отбрасывается (второй элемент)."""
    if src == dst:
        return dict(params), []
    out, dropped = {}, []
    for key, value in params.items():
        row = next((r for r in _SHARED_PARAMS if r.get(src) == key), None)
        target = row.get(dst) if row else None
        if target is None or (target in ("max_depth", "depth") and isinstance(value, (int, float)) and value <= 0):
            dropped.append(key)  # max_depth=-1 у lightgbm — «без ограничения», у catboost/xgboost аналога нет
            continue
        out[target] = _CLASS_WEIGHTS.get(value, value) if isinstance(value, str) else value
    return out, dropped


def extract_constants(code: str) -> tuple[dict, list[str]]:
    """MODEL / PARAMS / TARGET_TRANSFORM из кода статически (ast.literal_eval), без исполнения."""
    found: dict = {}
    problems: list[str] = []
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in CONSTANTS:
                try:
                    found[target.id] = ast.literal_eval(value)
                except (ValueError, SyntaxError, TypeError):
                    problems.append(f"{target.id} must be a literal value (no expressions or calls)")
    return found, problems


def set_constant(code: str, name: str, value: Any) -> str:
    """Код с заменённым (или дописанным) литералом верхнего уровня `name = value`."""
    lines = code.rstrip("\n").split("\n")
    for node in ast.parse(code).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            lines[node.lineno - 1:node.end_lineno] = [f"{name} = {value!r}"]
            return "\n".join(lines) + "\n"
    return "\n".join(lines) + f"\n{name} = {value!r}\n"


def normalize_model_config(consts: dict, task_type: str) -> tuple[str, dict, str | None, list[str]]:
    """Привести MODEL/PARAMS/TARGET_TRANSFORM к белому списку; что выкинуто — в notes."""
    notes: list[str] = []
    model = str(consts.get("MODEL") or "lightgbm").lower().strip()
    if model not in MODELS:
        notes.append(f"MODEL={model!r} is not supported, using lightgbm")
        model = "lightgbm"

    raw = consts.get("PARAMS") or {}
    if not isinstance(raw, dict):
        notes.append("PARAMS must be a dict, ignored")
        raw = {}
    allowed, params = MODEL_PARAMS[model], {}
    for key, value in raw.items():
        rule = allowed.get(key)
        if rule is None:
            notes.append(f"PARAMS[{key!r}] is not allowed for {model}, ignored")
        elif isinstance(rule, set):
            if value in rule:
                params[key] = value
            else:
                notes.append(f"PARAMS[{key!r}]={value!r} must be one of {sorted(rule, key=str)}, ignored")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            notes.append(f"PARAMS[{key!r}] must be a number, ignored")
        else:
            low, high = rule
            clipped = min(max(value, low), high)
            if clipped != value:
                notes.append(f"PARAMS[{key!r}]={value} clipped to {clipped}")
            params[key] = int(round(clipped)) if isinstance(low, int) and isinstance(high, int) else float(clipped)

    transform = consts.get("TARGET_TRANSFORM")
    if transform is not None and (transform != "log1p" or task_type != "regression"):
        notes.append(f"TARGET_TRANSFORM={transform!r} is not supported for {task_type}, using None")
        transform = None
    return model, params, transform, notes


# ============================================================ данные задачи

@dataclass
class TaskData:
    train_x: pd.DataFrame  # признаки train без таргета, id и drop_cols — только это попадает в песочницу
    test_x: pd.DataFrame
    y: Any                 # таргет остаётся в процессе харнесса (нужен только для профиля)
    spec_dict: dict        # что видит код LLM: тип, метрика, имя таргета, группы колонок, заметки
    nan_share: float = 0.0


def _targets(spec: Any) -> list[str]:
    target = getattr(spec, "target", None)
    return [target] if isinstance(target, str) else list(target or [])


def spec_to_dict(spec: Any, columns: list[str]) -> dict:
    cols = set(columns)
    targets = _targets(spec)

    def present(name: str) -> list[str]:
        return [c for c in (getattr(spec, name, None) or []) if c in cols]

    return {"task_type": str(getattr(spec, "task_type", "")), "metric": str(getattr(spec, "metric", "")),
            "greater_is_better": bool(getattr(spec, "greater_is_better", False)),
            "target": targets[0] if len(targets) == 1 else targets, "id_col": getattr(spec, "id_col", None),
            "cat_cols": present("cat_cols"), "date_cols": present("date_cols"), "text_cols": present("text_cols"),
            "feature_notes": {k: str(v)[:200] for k, v in (getattr(spec, "feature_notes", None) or {}).items()
                              if k in cols}}


_find_file = find_file  # перенесено в agent_eval.py: файлы задачи ищет и свой evaluator


def detect_text_cols(frame: pd.DataFrame, spec_dict: dict, sample: int = 5000) -> list[str]:
    """Строковые колонки со свободным текстом (длинные, многословные, почти уникальные), не отмеченные в spec.

    Эвристика до task_understanding (P2): как категория такая колонка бесполезна — почти все значения разные.
    """
    known = set(spec_dict["text_cols"]) | set(spec_dict["date_cols"])
    found = []
    for col in frame.columns:
        series = frame[col]
        if col in known or pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        values = series.dropna().astype(str)
        if values.empty:
            continue
        values = values.sample(min(sample, len(values)), random_state=0)
        if (values.str.len().mean() >= 30 and values.str.split().str.len().mean() >= 3
                and values.nunique() / len(values) >= 0.3):
            found.append(col)
    return found


def load_task_data(spec: Any) -> TaskData:
    train = pd.read_csv(_find_file(spec, "train.csv"), low_memory=False)
    test = pd.read_csv(_find_file(spec, "test.csv"), low_memory=False)
    targets = _targets(spec)
    drop = {getattr(spec, "id_col", None), *targets, *(getattr(spec, "drop_cols", None) or [])}
    features = [c for c in test.columns if c in train.columns and c not in drop]
    y = train[targets[0]] if len(targets) == 1 else train[targets]
    train_x, test_x = train[features].reset_index(drop=True), test[features].reset_index(drop=True)
    nan_share = float(train_x.isna().sum().sum() / train_x.size) if train_x.size else 0.0
    spec_dict = spec_to_dict(spec, features)
    guessed = detect_text_cols(train_x, spec_dict)
    if guessed:  # свободный текст, которого нет в spec.text_cols: обрабатывается как текст, а не категория
        spec_dict["text_cols"] += guessed
        spec_dict["cat_cols"] = [c for c in spec_dict["cat_cols"] if c not in guessed]
        log.info("research: text columns detected: %s", guessed)
    return TaskData(train_x, test_x, y.reset_index(drop=True), spec_dict, nan_share)


def _mini_profile(data: TaskData) -> dict:
    """Компактный профиль данных, пока нет eda.profile (P1): типы, пропуски, кардинальность, таргет."""
    spec = data.spec_dict
    columns = []
    for name in data.train_x.columns:
        series = data.train_x[name]
        info: dict[str, Any] = {"name": name, "missing_pct": round(float(series.isna().mean()) * 100, 1),
                                "nunique": int(series.nunique(dropna=True))}
        if name in spec["date_cols"]:
            info["dtype"] = "date"
        elif name in spec["text_cols"]:
            info["dtype"] = "text"
        elif pd.api.types.is_bool_dtype(series):
            info["dtype"] = "bool"
        elif pd.api.types.is_numeric_dtype(series):
            info.update(dtype="num", min=float(series.min()), max=float(series.max()))
        else:
            info.update(dtype="cat", top=series.value_counts().index[:3].astype(str).tolist())
        columns.append(info)

    target: dict[str, Any] = {"name": spec["target"]}
    y = data.y
    if isinstance(y, pd.Series) and spec["task_type"] == "regression" and pd.api.types.is_numeric_dtype(y):
        target.update(mean=float(y.mean()), std=float(y.std()), min=float(y.min()), max=float(y.max()),
                      skew=float(y.skew()), zeros_pct=round(float((y == 0).mean()) * 100, 1))
    elif isinstance(y, pd.Series):
        target["classes"] = {str(k): int(v) for k, v in y.astype(str).value_counts().head(15).items()}
    return {"n_train": len(data.train_x), "n_test": len(data.test_x), "target": target, "columns": columns}


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    for col in frame.columns:
        if frame[col].dtype == object:  # смешанные типы в object pyarrow не пишет
            frame[col] = frame[col].where(frame[col].isna(), frame[col].astype(str))
    return frame


def write_inputs(data: TaskData, workdir: Path) -> tuple[Path, Path]:
    """Вход песочницы: признаки без таргета, id и drop_cols. Таргет в песочницу не попадает никогда."""
    folder = Path(workdir) / "input"
    folder.mkdir(parents=True, exist_ok=True)
    paths = folder / "train_X.parquet", folder / "test_X.parquet"
    _parquet_safe(data.train_x).to_parquet(paths[0], index=False)
    _parquet_safe(data.test_x).to_parquet(paths[1], index=False)
    return paths


# базовый модуль («эксперимент 0»): исходные признаки + детерминированная подготовка дат и текста
BASE_CODE = '''import numpy as np
import pandas as pd


def build_features(train, test, spec):
    X_train, X_test = train.copy(), test.copy()
    for df in (X_train, X_test):
        for col in spec.get("date_cols", []):
            if col in df.columns:
                dates = pd.to_datetime(df[col], errors="coerce")
                df[col + "_year"] = dates.dt.year
                df[col + "_month"] = dates.dt.month
                df[col + "_dayofweek"] = dates.dt.dayofweek
                df.drop(columns=col, inplace=True)
        for col in spec.get("text_cols", []):
            if col in df.columns:
                text = df[col].fillna("").astype(str)
                df[col + "_len"] = text.str.len()
                df[col + "_words"] = text.str.split().str.len()
                df.drop(columns=col, inplace=True)
    return X_train, X_test


MODEL = "lightgbm"
PARAMS = {}
TARGET_TRANSFORM = None
'''

# то же для задач с текстовыми колонками: + TF-IDF слов и биграмм на train+test → SVD (без таргета).
# Предобученные модели и эмбеддинги запрещены правилами: смысл текста — только из данных задачи.
BASE_CODE_TEXT = '''import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def text_svd(train_text, test_text, prefix, n_components=20):
    """TF-IDF слов и биграмм по train+test → SVD: плотные признаки смысла текста (таргет не нужен)."""
    both = pd.concat([train_text, test_text], ignore_index=True).fillna("").astype(str)
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                                 sublinear_tf=True).fit_transform(both)
    except ValueError:  # пустой словарь: ни одно слово не встречается дважды
        return None
    k = min(n_components, matrix.shape[1] - 1)
    if k < 1:
        return None
    dense = TruncatedSVD(n_components=k, random_state=0).fit_transform(matrix)
    columns = [f"{prefix}_svd_{i}" for i in range(k)]
    n = len(train_text)
    return (pd.DataFrame(dense[:n], columns=columns, index=train_text.index),
            pd.DataFrame(dense[n:], columns=columns, index=test_text.index))


def build_features(train, test, spec):
    X_train, X_test = train.copy(), test.copy()
    for col in spec.get("text_cols", []):
        if col in X_train.columns:
            parts = text_svd(X_train[col], X_test[col], col)
            if parts is not None:
                X_train, X_test = X_train.join(parts[0]), X_test.join(parts[1])
    for df in (X_train, X_test):
        for col in spec.get("date_cols", []):
            if col in df.columns:
                dates = pd.to_datetime(df[col], errors="coerce")
                df[col + "_year"] = dates.dt.year
                df[col + "_month"] = dates.dt.month
                df[col + "_dayofweek"] = dates.dt.dayofweek
                df.drop(columns=col, inplace=True)
        for col in spec.get("text_cols", []):
            if col in df.columns:
                text = df[col].fillna("").astype(str)
                df[col + "_len"] = text.str.len()
                df[col + "_words"] = text.str.split().str.len()
                df.drop(columns=col, inplace=True)
    return X_train, X_test


MODEL = "lightgbm"
PARAMS = {}
TARGET_TRANSFORM = None
'''


def base_code_for(spec_dict: dict) -> str:
    """Стартовый модуль: с TF-IDF только при текстовых колонках (иначе — короче промпт кодера)."""
    return BASE_CODE_TEXT if spec_dict.get("text_cols") else BASE_CODE


# ============================================================ песочница

@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    artifacts: list[str] = field(default_factory=list)
    timed_out: bool = False


def _tail(path: Path, limit: int = 20_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(max(0, path.stat().st_size - limit))
        return fh.read().decode("utf-8", errors="replace")


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # процесс в своей сессии: убиваем всю группу
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        proc.kill()


def _local_run_code(code: str, cwd: Path, timeout: float) -> ExecResult:
    """Запасная песочница, пока нет sandbox.py (P5): отдельный процесс, пустой env, таймаут."""
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "_runner.py").write_text(code, encoding="utf-8")
    env = {"PATH": os.defpath, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}  # без OPENAI_*
    with open(cwd / "stdout.txt", "wb") as out, open(cwd / "stderr.txt", "wb") as err:
        proc = subprocess.Popen([sys.executable, "-I", "-u", "_runner.py"], cwd=cwd, env=env, stdout=out,
                                stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.wait()
            timed_out = True
    return ExecResult(_tail(cwd / "stdout.txt"), _tail(cwd / "stderr.txt"), proc.returncode,
                      sorted(p.name for p in cwd.iterdir()), timed_out)


RUNNER = '''"""Раннер эксперимента агента: строит признаки кодом LLM, результат — только через файлы."""
import importlib.util
import json
import sys

import pandas as pd


def fail(message):
    print("HARNESS_ERROR: " + message, file=sys.stderr)
    sys.exit(3)


def frame(name, value, n):
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if not isinstance(value, pd.DataFrame):
        fail(f"{name} is {type(value).__name__}, expected a pandas DataFrame")
    if len(value) != n:
        fail(f"{name} has {len(value)} rows, expected {n}: rows must not be added or dropped")
    if not value.index.equals(pd.RangeIndex(n)):
        try:
            same_rows = value.index.is_unique and set(value.index) == set(range(n))
        except TypeError:
            same_rows = False
        if same_rows:
            fail(f"rows of {name} were reordered: keep the original row order")
        value = value.reset_index(drop=True)
    value.columns = [str(c) for c in value.columns]
    duplicated = value.columns[value.columns.duplicated()].tolist()
    if duplicated:
        fail(f"{name} has duplicated column names: {duplicated[:10]}")
    for col in value.columns:
        if value[col].dtype == object:
            value[col] = value[col].where(value[col].isna(), value[col].astype(str))
    return value


def main():
    with open("spec.json", encoding="utf-8") as fh:
        spec = json.load(fh)
    train = pd.read_parquet(__TRAIN__)
    test = pd.read_parquet(__TEST__)
    n_train, n_test = len(train), len(test)
    loader = importlib.util.spec_from_file_location("features", "features.py")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    result = module.build_features(train, test, spec)
    if not (isinstance(result, tuple) and len(result) == 2):
        fail("build_features must return a tuple (X_train, X_test)")
    frame("X_train", result[0], n_train).to_parquet("feat_train.parquet")
    frame("X_test", result[1], n_test).to_parquet("feat_test.parquet")


main()
'''


@dataclass
class FeatureOutcome:
    x_train: pd.DataFrame | None = None
    x_test: pd.DataFrame | None = None
    error: str = ""
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0
    timed_out: bool = False


def run_features(code: str, exp_dir: Path, inputs: tuple[Path, Path], data: TaskData, cfg: AgentConfig,
                 run_code: RunCodeFn, timeout: float) -> FeatureOutcome:
    """Построить признаки кодом LLM в песочнице и проверить результат харнессом."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    outputs = exp_dir / "feat_train.parquet", exp_dir / "feat_test.parquet"
    for path in outputs:
        path.unlink(missing_ok=True)
    (exp_dir / "features.py").write_text(code, encoding="utf-8")
    (exp_dir / "spec.json").write_text(json.dumps(data.spec_dict, ensure_ascii=False, default=str),
                                       encoding="utf-8")
    runner = RUNNER.replace("__TRAIN__", repr(str(inputs[0]))).replace("__TEST__", repr(str(inputs[1])))

    start = time.monotonic()
    try:
        result = run_code(runner, exp_dir, timeout)
    except Exception as error:  # сбой самой песочницы — не повод ронять агента
        return FeatureOutcome(error=f"sandbox error: {type(error).__name__}: {error}",
                              seconds=time.monotonic() - start)
    seconds = time.monotonic() - start
    if getattr(result, "timed_out", False):
        return FeatureOutcome(error=f"timeout: building features took longer than {timeout:.0f} s; "
                                    "the code is too slow or too heavy", seconds=seconds, timed_out=True)
    stderr = str(getattr(result, "stderr", "") or "")
    if getattr(result, "returncode", 1) != 0 or not all(p.exists() for p in outputs):
        harness = [line for line in stderr.splitlines() if line.startswith("HARNESS_ERROR: ")]
        error = (harness[-1].removeprefix("HARNESS_ERROR: ") if harness
                 else stderr.strip()[-4000:] or f"process exited with code {getattr(result, 'returncode', '?')}")
        return FeatureOutcome(error=error, seconds=seconds)
    try:
        x_train, x_test = pd.read_parquet(outputs[0]), pd.read_parquet(outputs[1])
    except Exception as error:
        return FeatureOutcome(error=f"cannot read features: {type(error).__name__}: {error}", seconds=seconds)
    finally:
        if not cfg.keep_artifacts:
            for path in outputs:
                path.unlink(missing_ok=True)
    x_train, x_test, problems, notes = validate_features(x_train, x_test, data, cfg)
    if problems:
        return FeatureOutcome(error="; ".join(problems), notes=notes, seconds=seconds)
    return FeatureOutcome(x_train, x_test, notes=notes, seconds=seconds)


def _to_float(series: pd.Series) -> pd.Series:
    try:
        return series.astype("float64")
    except (TypeError, ValueError):
        return pd.to_numeric(series, errors="coerce").astype("float64")


def _to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    epoch = pd.Timestamp("1970-01-01", tz=getattr(series.dt, "tz", None))
    return (series - epoch).dt.total_seconds()


def _as_category(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Строки/категории → category с общим словарём для train и test."""
    a, b = a.astype(object), b.astype(object)
    a, b = a.where(a.isna(), a.astype(str)), b.where(b.isna(), b.astype(str))
    dtype = pd.CategoricalDtype(sorted(pd.unique(pd.concat([a, b]).dropna())))
    return a.astype(dtype), b.astype(dtype)


def _looks_like_row_index(values: np.ndarray) -> bool:
    """Арифметическая прогрессия по строкам (arange, номер строки, cumcount по всем) — утечка через порядок."""
    if len(values) < 100 or np.isnan(values).any():
        return False
    step = np.diff(values)
    return bool(step[0] != 0 and np.all(step == step[0]))


def validate_features(x_train: pd.DataFrame, x_test: pd.DataFrame, data: TaskData,
                      cfg: AgentConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Проверка выхода build_features и приведение типов для бустинга. problems непусто → эксперимент failed."""
    problems: list[str] = []
    notes: list[str] = []
    if len(x_train) != len(data.train_x) or len(x_test) != len(data.test_x):
        problems.append(f"row count changed: got {len(x_train)}/{len(x_test)}, "
                        f"expected {len(data.train_x)}/{len(data.test_x)}")
    if set(x_train.columns) != set(x_test.columns):
        only_train = [c for c in x_train.columns if c not in x_test.columns][:10]
        only_test = [c for c in x_test.columns if c not in x_train.columns][:10]
        problems.append(f"train/test columns differ: only in train {only_train}, only in test {only_test}")
    if not 0 < x_train.shape[1] <= cfg.max_feature_cols:
        problems.append(f"{x_train.shape[1]} feature columns, allowed 1..{cfg.max_feature_cols}")
    forbidden = [c for c in x_train.columns if c in {*_as_list(data.spec_dict["target"]), data.spec_dict["id_col"]}]
    if forbidden:
        problems.append(f"target/id columns must not be features: {forbidden}")
    if problems:
        return x_train, x_test, problems, notes
    if not x_train.index.equals(pd.RangeIndex(len(x_train))):
        x_train = x_train.reset_index(drop=True)
    if not x_test.index.equals(pd.RangeIndex(len(x_test))):
        x_test = x_test.reset_index(drop=True)

    nan_share = sum(int(x_train[c].isna().sum()) for c in x_train.columns) / max(1, x_train.size)
    if nan_share > max(cfg.max_nan_share, data.nan_share + 0.25):
        problems.append(f"NaN avalanche: {nan_share:.0%} of feature values are missing "
                        f"(input had {data.nan_share:.0%})")

    out_train, out_test = {}, {}
    inf_cols, index_cols, constant, bad = [], [], [], []
    for col in x_train.columns:
        a, b = x_train[col], x_test[col]
        if pd.api.types.is_datetime64_any_dtype(a) or pd.api.types.is_timedelta64_dtype(a):
            a = _to_seconds(a)
            if pd.api.types.is_datetime64_any_dtype(b) or pd.api.types.is_timedelta64_dtype(b):
                b = _to_seconds(b)
            notes.append(f"{col}: datetime converted to seconds")
        if pd.api.types.is_bool_dtype(a) or (pd.api.types.is_extension_array_dtype(a)
                                             and pd.api.types.is_numeric_dtype(a)):
            a = _to_float(a)
        if pd.api.types.is_bool_dtype(b) or (pd.api.types.is_extension_array_dtype(b)
                                             and pd.api.types.is_numeric_dtype(b)):
            b = _to_float(b)

        numeric = pd.api.types.is_numeric_dtype(a)
        if numeric != pd.api.types.is_numeric_dtype(b) or pd.api.types.is_complex_dtype(a):
            bad.append(f"{col} ({a.dtype} in train, {b.dtype} in test)")
            continue
        if numeric:
            values = a.to_numpy(dtype="float64", na_value=np.nan)
            if np.isinf(values).any() or np.isinf(b.to_numpy(dtype="float64", na_value=np.nan)).any():
                inf_cols.append(col)
            elif col not in data.train_x.columns and _looks_like_row_index(values):
                index_cols.append(col)
        elif isinstance(a.dtype, pd.CategoricalDtype) or a.dtype == object or pd.api.types.is_string_dtype(a):
            a, b = _as_category(a, b)
        else:
            bad.append(f"{col} ({a.dtype})")
            continue
        if a.nunique(dropna=False) <= 1:
            constant.append(col)
            continue
        out_train[col], out_test[col] = a, b

    if bad:
        problems.append(f"unsupported or mismatched dtypes: {bad[:10]}")
    if inf_cols:
        problems.append(f"inf values in columns {inf_cols[:10]}: replace them with NaN")
    if index_cols:
        problems.append(f"columns {index_cols[:10]} look like a row index/ID: row position must not be a feature")
    if constant:
        notes.append(f"dropped constant columns: {constant[:10]}")
    if not out_train and not problems:
        problems.append("all feature columns are constant")
    x_train = pd.DataFrame(out_train, index=pd.RangeIndex(len(x_train)))
    x_test = pd.DataFrame(out_test, index=pd.RangeIndex(len(x_test)))
    if x_train.memory_usage(deep=False).sum() > cfg.max_feature_gb * 1e9:
        problems.append(f"features use more than {cfg.max_feature_gb} GB")
    return x_train, x_test, problems, notes


def _as_list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value]


# ============================================================ адаптер к evaluator.evaluate

def _rows_like(features: pd.DataFrame, frame: Any) -> pd.DataFrame:
    """Признаки под строки, которые evaluator передал в build_features (весь набор или его часть)."""
    if not isinstance(frame, pd.DataFrame) or len(frame) == len(features):
        return features.copy()
    try:  # часть train (например, без hold-out P1) с исходными номерами строк в индексе
        return features.loc[frame.index].copy()
    except KeyError as error:
        raise ValueError(f"evaluator passed {len(frame)} rows that cannot be matched to {len(features)} "
                         "precomputed feature rows") from error


def make_build_features(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Callable:
    """build_features для evaluator.evaluate: код LLM уже исполнен в песочнице, отдаём готовые признаки."""
    def build_features(train: Any, test: Any, spec: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _rows_like(x_train, train), _rows_like(x_test, test)

    return build_features


# ============================================================ решение «принять / отклонить»

@dataclass
class Decision:
    accept: bool
    gain: float           # > 0 — лучше с учётом направления метрики
    threshold: float
    noise: float          # стандартная ошибка разницы CV
    suspicious: bool = False
    reason: str = ""


def _fold_scores(cand: Candidate) -> np.ndarray | None:
    scores = (getattr(cand, "meta", None) or {}).get("fold_scores")
    try:
        values = np.asarray(scores, dtype=float)
    except (TypeError, ValueError):
        return None
    return values if values.ndim == 1 and len(values) and np.isfinite(values).all() else None


def decide(new: Candidate, current: Candidate, greater_is_better: bool, cfg: AgentConfig) -> Decision:
    """Принять, если выигрыш больше шума CV (accept_k · SE) и больше min_rel_gain · |cv|.

    Сплиты у кандидатов одинаковые, поэтому при наличии fold_scores шум — SE парных разностей по фолдам;
    иначе — cv_std / sqrt(n_folds) текущего лучшего.
    """
    new_score, cur_score = float(new.cv_score), float(current.cv_score)
    if not math.isfinite(new_score):
        return Decision(False, math.nan, math.inf, math.nan, reason="CV score is not finite")
    if not math.isfinite(cur_score):
        return Decision(True, math.inf, 0.0, 0.0, reason="current best has no finite CV score")
    sign = 1.0 if greater_is_better else -1.0
    gain = sign * (new_score - cur_score)

    ours, theirs = _fold_scores(new), _fold_scores(current)
    if ours is not None and theirs is not None and len(ours) == len(theirs) >= 2:
        diffs = sign * (ours - theirs)
        noise, source = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))), "paired folds"
    else:
        folds = len(theirs) if theirs is not None else cfg.default_n_folds
        std = float(current.cv_std) if math.isfinite(float(current.cv_std)) else 0.0
        noise, source = std / math.sqrt(max(1, folds)), "fold std"

    threshold = max(cfg.accept_k * noise, cfg.min_rel_gain * abs(cur_score))
    accept = gain > threshold
    suspicious = accept and gain > cfg.suspicious_k * noise and gain > cfg.suspicious_rel * abs(cur_score)
    reason = f"gain {gain:+.5g} {'>' if accept else '<='} threshold {threshold:.5g} (noise {noise:.3g}, {source})"
    return Decision(accept, gain, threshold, noise, suspicious, reason)


# ============================================================ время, токены, вызовы LLM

def _time_left(deadline: Any) -> float:
    """Секунды до конца окна research: deadline.stage("research"), иначе deadline.left()."""
    if deadline is None:
        return math.inf
    stage = getattr(deadline, "stage", None)
    if callable(stage):
        try:
            value = stage("research")
            return float(value.left() if hasattr(value, "left") else value)
        except Exception:
            pass
    left = getattr(deadline, "left", None)
    try:
        return float(left()) if callable(left) else math.inf
    except Exception:
        return math.inf


class _SubDeadline:
    """Дедлайн одного вызова evaluate: min(лимит на кандидата, остаток окна research)."""

    def __init__(self, parent: Any, seconds: float) -> None:
        self._parent, self._end = parent, time.monotonic() + seconds

    def left(self) -> float:
        return max(0.0, min(self._end - time.monotonic(), _time_left(self._parent)))

    def stage(self, name: str) -> float:
        return self.left()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._parent, name)


def _read_usage(llm: Any) -> tuple[int, int] | None:
    """Счётчик токенов клиента (P2), если он его отдаёт: словарь usage, объект usage или атрибуты."""
    usage = getattr(llm, "usage", None)
    if callable(usage):
        try:
            usage = usage()
        except Exception:
            usage = None
    if isinstance(usage, dict):
        tokens_in = next((usage[k] for k in ("prompt_tokens", "tokens_in", "input_tokens") if k in usage), None)
        tokens_out = next((usage[k] for k in ("completion_tokens", "tokens_out", "output_tokens") if k in usage), 0)
        if tokens_in is not None:
            return int(tokens_in), int(tokens_out or 0)
    for source in (usage, llm):
        for a, b in (("prompt_tokens", "completion_tokens"), ("tokens_in", "tokens_out")):
            if source is not None and isinstance(getattr(source, a, None), (int, float)):
                return int(getattr(source, a)), int(getattr(source, b, 0) or 0)
    return None


def _call(fn: Callable, *args: Any, **kw: Any) -> Any:
    """Вызов метода клиента только с теми именованными аргументами, которые он принимает."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kw)
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kw = {k: v for k, v in kw.items() if k in params}
    return fn(*args, **kw)


def _plain_error(error: BaseException, limit: int = 200) -> str:
    """Текст ошибки без HTML-страниц шлюза («504 Gateway Time-out» вместо всего nginx-ответа)."""
    text = str(error)
    if "<" in text and ">" in text:
        title = re.search(r"<title>(.*?)</title>", text, re.S | re.I)
        text = title.group(1) if title else re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())[:limit]


def _is_budget_error(error: Exception) -> bool:
    text = str(error).lower()
    return "budget" in text and "exceed" in text


def _is_bad_request(error: Exception) -> bool:
    """Шлюз отверг параметры запроса (HTTP 400/422), а не упал сам."""
    for source in (error, getattr(error, "response", None)):
        if getattr(source, "status_code", None) in (400, 422):
            return True
    return False


def _no_think(messages: list[dict]) -> list[dict]:
    """Qwen3: «/no_think» в последнем сообщении пользователя отключает длинный <think>."""
    messages = [dict(m) for m in messages]
    for message in reversed(messages):
        if message["role"] == "user":
            message["content"] = message["content"].rstrip() + "\n/no_think"
            break
    return messages


# ============================================================ гипотезы

@dataclass
class Hypothesis:
    title: str
    kind: str = "features"
    rationale: str = ""
    expected_gain: str = "mid"
    cost: str = "mid"
    id: str = ""


_KIND_ALIASES = {"feature": "features", "feature_engineering": "features", "interaction": "features",
                 "aggregation": "features", "categorical": "encoding", "encode": "encoding",
                 "imputation": "missing", "missing_values": "missing", "outlier": "outliers",
                 "clipping": "outliers", "target": "target_transform", "transform": "target_transform",
                 "log_target": "target_transform", "hyperparameters": "params", "hyperparams": "params",
                 "param": "params", "models": "model", "algorithm": "model", "tuning": "tune",
                 "optuna": "tune", "hpo": "tune", "hyperparameter_search": "tune", "hyperparameter_tuning": "tune",
                 "autotune": "tune", "model_selection": "model_search", "compare_models": "model_search",
                 "model_comparison": "model_search", "algorithm_search": "model_search", "models_search": "model_search",
                 "selection": "feature_selection", "drop": "feature_selection"}
_LEVELS = {"low": 1, "mid": 2, "high": 3}
_LEVEL_ALIASES = {"medium": "mid", "moderate": "mid", "med": "mid"}


def normalize_hypotheses(data: Any, task_type: str, tried: list[str], max_n: int,
                         is_duplicate: Callable[[str, str], bool] | None = None,
                         can_tune: bool = False, can_search_models: bool = False) -> tuple[list[Hypothesis], str]:
    """Ответ планировщика → гипотезы без повторов (с памятью и между собой), по убыванию выигрыш/стоимость.

    tried — точные повторы названий; is_duplicate(title, kind) — повторы по смыслу (перефразированные).
    Без тюнера (can_tune=False) гипотеза tune становится params: параметры подберёт кодер;
    без выбора модели (can_search_models=False) model_search становится model: модель сменит кодер.
    """
    analysis, items = "", data
    if isinstance(data, dict):
        analysis = str(data.get("analysis") or "")
        items = next((data[k] for k in ("hypotheses", "experiments", "ideas", "items")
                      if isinstance(data.get(k), list)), [])
    seen = {normalize_title(t) for t in tried}
    result = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str):
            item = {"title": item}
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title") or item.get("name") or item.get("hypothesis")
                             or item.get("description") or "").split())
        key = normalize_title(title)
        if not key or key in seen:
            continue
        kind = re.sub(r"[\s-]+", "_", str(item.get("kind") or item.get("type") or "features").lower().strip())
        strategies = (P.TUNE_KIND, P.MODEL_SEARCH_KIND)
        kind = kind if kind in P.KINDS or kind in strategies else _KIND_ALIASES.get(kind, "features")
        if kind == P.TUNE_KIND and not can_tune:
            kind = "params"
        if kind == P.MODEL_SEARCH_KIND and not can_search_models:
            kind = "model"
        if kind == "target_transform" and task_type != "regression":
            continue
        levels = []
        for field_name in ("expected_gain", "cost"):
            level = str(item.get(field_name) or "mid").lower().strip()
            level = _LEVEL_ALIASES.get(level, level)
            levels.append(level if level in _LEVELS else "mid")
        if is_duplicate is not None and is_duplicate(title, kind):
            continue
        seen.add(key)
        result.append(Hypothesis(title=title[:200], kind=kind, rationale=str(item.get("rationale") or "")[:400],
                                 expected_gain=levels[0], cost=levels[1], id=str(item.get("id") or "")))
    result.sort(key=lambda h: -_LEVELS[h.expected_gain] / _LEVELS[h.cost])
    return result[:max_n], analysis


# ============================================================ цикл исследования

class _Research:
    def __init__(self, spec: Any, best: Candidate | None, llm: Any, deadline: Any, cfg: AgentConfig, *,
                 evaluate: EvaluateFn, run_code: RunCodeFn, profile: Callable | None, guards: GuardFn | None,
                 data: TaskData | None, tuner: TuneFn | None = None) -> None:
        self.spec, self.best, self.llm, self.deadline, self.cfg = spec, best, llm, deadline, cfg
        self.evaluate, self.run_code, self.profile_fn, self.guards, self.data = evaluate, run_code, profile, guards, data
        self.tuner = tuner
        self.greater = bool(getattr(spec, "greater_is_better", False))
        self.memory = ExperimentMemory()
        self.accepted: list[Candidate] = []
        self.queue: deque[Hypothesis] = deque()
        self.current: Candidate | None = None
        self.base_code = BASE_CODE
        self.insights = ""
        self.errors_text = ""
        self.llm_extra = dict(cfg.llm_extra)
        self.tokens_in = self.tokens_out = self.llm_calls = 0
        self.iteration = self.consecutive_failures = self.no_improve = self.empty_plans = self.since_critic = 0
        self.durations: list[float] = []
        self.reference_seconds = 60.0
        self.stop_reason = ""
        self.workdir = Path(spec.task_dir) / cfg.workdir_name

    # ---------------------------------------------------------------- главный цикл
    def run(self) -> list[Candidate]:
        try:
            self._prepare()
            if not self._init_reference():
                return []
            while True:
                self.stop_reason = self._stop_reason()
                if self.stop_reason.startswith("plateau") and self._queue_auto_strategy():
                    self.stop_reason = ""
                if self.stop_reason:
                    break
                if not self.queue:
                    self._maybe_critic()
                    self._plan()
                    if not self.queue:
                        self.empty_plans += 1
                        if self.empty_plans >= self.cfg.max_empty_plans and not self._queue_auto_strategy():
                            self.stop_reason = "planner gives no new hypotheses"
                            break
                        continue
                    self.empty_plans = 0
                self._iterate(self.queue.popleft())
            self.postprocess()
            return self.result()
        except LLMUnavailable:
            self.stop_reason = "LLM unavailable"
            raise
        finally:
            self._finish()

    def _queue_auto_strategy(self) -> bool:
        """Перед остановкой (плато, пустой план): один раз выбор модели, затем один раз подбор её параметров."""
        cfg = self.cfg
        done = {e.kind for e in self.memory.experiments} | {h.kind for h in self.queue}
        if cfg.model_search and cfg.auto_model_search and P.MODEL_SEARCH_KIND not in done:
            self.queue.appendleft(Hypothesis(title="Compare lightgbm, catboost and xgboost (auto, before stopping)",
                                             kind=P.MODEL_SEARCH_KIND, rationale="feature ideas stopped improving CV"))
            log.info("research: no more gains from features — queued automatic model search")
            return True
        if not cfg.auto_tune or self.tuner is None or P.TUNE_KIND in done:
            return False
        model = normalize_model_config(extract_constants(self.base_code)[0], self.task_type)[0]
        self.queue.appendleft(Hypothesis(title=f"Tune {model} hyperparameters (auto, before stopping)",
                                         kind=P.TUNE_KIND, rationale="feature ideas stopped improving CV"))
        log.info("research: no more gains from features — queued automatic hyperparameter tuning")
        return True

    def postprocess(self) -> None:
        """Конец исследования: множители вероятностей классов под метрику по меткам — к текущему лучшему.

        Веса подбираются на частях OOF и проверяются на оставшейся (postprocess.py); принимается обычным
        правилом decide по парным частям. Кандидат — тот же код, у oof/test_pred перевзвешенные вероятности.
        """
        cfg = self.cfg
        current = self.current
        meta = (getattr(current, "meta", None) or {}) if current is not None else {}
        if (not cfg.postprocess or current is None or self.task_type == "regression" or meta.get("postprocess")
                or self._left() < 5):
            return
        start = time.monotonic()
        try:
            found = fit_class_weights(self.data.y, current.oof, self.data.spec_dict["metric"],
                                      classes=meta.get("classes"), n_splits=cfg.postprocess_folds,
                                      parts=meta.get("fold_ids"))
            if found is None:
                return
            before, after = (types.SimpleNamespace(cv_score=float(np.mean(s)), cv_std=float(np.std(s)),
                                                   meta={"fold_scores": s})
                             for s in (found.folds_before, found.folds_after))
            decision = decide(after, before, True, cfg)  # все метрики по меткам — чем больше, тем лучше
            weights = {name: round(float(w), 4) for name, w in zip(found.classes, found.weights)}
            status = "accepted" if decision.accept else "rejected"
            if decision.accept:
                by_folds = meta.get("fold_ids") is not None and len(found.folds_after) == len(meta.get("fold_scores") or [])
                # по фолдам evaluator «после» сравнимо с другими кандидатами парно; по случайным частям — нет
                post_meta = {k: v for k, v in meta.items() if k != "fold_scores"}
                if by_folds:
                    post_meta["fold_scores"] = list(found.folds_after)
                post_meta.update(postprocess={"method": "class_weights", "weights": weights, "gain": decision.gain,
                                              "folds_before": found.folds_before, "folds_after": found.folds_after})
                post_cv = (float(np.mean(found.folds_after)) if by_folds
                           else float(current.cv_score) + decision.gain)
                post = Candidate(name="agent_post", cv_score=post_cv,
                                 cv_std=float(current.cv_std), oof=apply_weights(current.oof, found.weights),
                                 test_pred=apply_weights(current.test_pred, found.weights), meta=post_meta)
                self.accepted.append(post)
                self.current = post
            top = ", ".join(f"{k}×{v:g}" for k, v in sorted(weights.items(), key=lambda kv: -kv[1])[:4])
            self.memory.add(Experiment(
                id=self.memory.next_id(), hypothesis=f"Class probability weights for {self.data.spec_dict['metric']}",
                kind="postprocess", status=status, cv_score=float(current.cv_score) + decision.gain,
                gain=decision.gain, error=f"{decision.reason} | weights {top}"[:1000],
                duration_s=round(time.monotonic() - start, 1), base_hash=code_hash(self.base_code)))
            log.info("research: postprocess [%s] %s | weights %s | %.1fs", status, decision.reason, top,
                     time.monotonic() - start)
            self.memory.dump_jsonl(self.workdir / "experiments.jsonl")
        except Exception as error:  # постобработка — бонус, а не обязательный шаг
            log.warning("research: postprocess failed: %s: %s", type(error).__name__, P.clip(str(error), 300))

    def result(self) -> list[Candidate]:
        """Принятые кандидаты по убыванию качества; при only_better_than_best — не хуже входного best."""
        cands = [c for c in self.accepted if math.isfinite(float(c.cv_score))]
        if self.cfg.only_better_than_best and self.best is not None and math.isfinite(float(self.best.cv_score)):
            sign = 1.0 if self.greater else -1.0
            cands = [c for c in cands if sign * (float(c.cv_score) - float(self.best.cv_score)) >= 0]
        return sorted(cands, key=lambda c: float(c.cv_score), reverse=self.greater)

    def _prepare(self) -> None:
        cfg = self.cfg
        self.data = self.data or load_task_data(self.spec)
        self.task_type = self.data.spec_dict["task_type"]
        self.memory.columns = [str(c) for c in self.data.train_x.columns]
        self.inputs = write_inputs(self.data, self.workdir)

        mini = _mini_profile(self.data)
        profile: Any = None
        if self.profile_fn is not None:
            try:
                target = self.data.spec_dict["target"]
                train = (self.data.train_x.assign(**{target: self.data.y})
                         if isinstance(self.data.y, pd.Series) and isinstance(target, str) else self.data.train_x)
                profile = self.profile_fn(train, self.data.test_x, self.spec)
            except Exception as error:
                log.warning("research: eda.profile failed (%s), using the mini profile", error)
        if isinstance(profile, str) and profile.strip():
            self.profile_text = P.clip(profile, cfg.profile_chars)
        else:
            self.profile_text = P.render_profile(profile or mini, self.data.spec_dict["feature_notes"],
                                                 cfg.profile_chars)
        self.columns_text = P.render_columns_brief(mini)
        direction = "higher is better" if self.greater else "lower is better"
        self.task_line = (f"type: {self.task_type} | metric: {self.data.spec_dict['metric']} ({direction}) | "
                          f"target: {self.data.spec_dict['target']} | rows: train {len(self.data.train_x)}, "
                          f"test {len(self.data.test_x)}")
        self.task_text = str(getattr(self.spec, "raw_task_text", "") or "")

    def _init_reference(self) -> bool:
        """Точка отсчёта: код best (если это кандидат агента) или стартовый модуль (BASE_CODE, с текстом —
        BASE_CODE_TEXT), оценённый тем же evaluator."""
        meta = (getattr(self.best, "meta", None) or {}) if self.best is not None else {}
        if meta.get("code") and math.isfinite(float(self.best.cv_score)):
            self.base_code, self.current = meta["code"], self.best
            self.reference_seconds = float(meta.get("seconds", self.reference_seconds))
            self._update_errors()
            return True
        start_code = base_code_for(self.data.spec_dict)
        start = time.monotonic()
        cand, error = self._evaluate_code(start_code, "exp_00_base")
        if cand is None and start_code != BASE_CODE:  # TF-IDF не справился (время, память) — без него
            log.warning("research: text base features failed (%s), retrying without TF-IDF", P.clip(error, 300))
            start_code, start = BASE_CODE, time.monotonic()
            cand, error = self._evaluate_code(start_code, "exp_00_base")
        self.reference_seconds = time.monotonic() - start
        if cand is None:
            log.warning("research: base features failed, stopping: %s", P.clip(error, 500))
            return False
        self.base_code, self.current = start_code, cand
        best_cv = f"{float(self.best.cv_score):.5g}" if self.best is not None else "n/a"
        log.info("research: reference %s=%.5g (input best %s), %.0fs", self.data.spec_dict["metric"],
                 cand.cv_score, best_cv, self.reference_seconds)
        self._update_errors()
        return True

    def _update_errors(self) -> None:
        """Где ошибается текущий лучший: OOF против таргета в процессе харнесса, в промпт — только агрегаты."""
        if self.cfg.errors_chars <= 0 or self.current is None:
            return
        start = time.monotonic()
        meta = getattr(self.current, "meta", None) or {}
        self.errors_text = analyze_errors(
            self.data.train_x, self.data.y, getattr(self.current, "oof", None), self.task_type,
            classes=meta.get("classes"), cat_cols=self.data.spec_dict["cat_cols"], max_chars=self.cfg.errors_chars)
        log.info("research: error analysis %s in %.1fs", "updated" if self.errors_text else "unavailable",
                 time.monotonic() - start)

    def _finish(self) -> None:
        log.info("research: stop (%s); experiments %d, accepted %d, LLM calls %d, tokens in/out %d/%d",
                 self.stop_reason or "error", len(self.memory.experiments), len(self.accepted), self.llm_calls,
                 self.tokens_in, self.tokens_out)
        try:
            self.memory.dump_jsonl(self.workdir / "experiments.jsonl")
            if not self.cfg.keep_artifacts:
                for path in (self.workdir / "input").glob("*.parquet"):
                    path.unlink(missing_ok=True)
        except OSError as error:
            log.warning("research: cannot write artifacts: %s", error)

    # ---------------------------------------------------------------- бюджеты
    def _left(self) -> float:
        return _time_left(self.deadline)

    def tokens_used(self) -> int:
        return self.tokens_in + self.tokens_out

    def _expected_seconds(self) -> float:
        """Сколько займёт ещё одна итерация: типичная длительность (не меньше эталонной оценки) + LLM."""
        recent = self.durations[-5:]
        typical = max(float(np.median(recent)) if recent else 0.0, self.reference_seconds)
        return typical * 1.25 + self.cfg.llm_reserve_seconds

    def _stop_reason(self) -> str:
        cfg = self.cfg
        if self.iteration >= cfg.max_iters:
            return f"max_iters={cfg.max_iters}"
        left, need = self._left(), self._expected_seconds()
        if left < need:
            return f"time: {left:.0f}s left, an iteration needs ~{need:.0f}s"
        if self.tokens_used() >= cfg.max_tokens:
            return f"tokens: {self.tokens_used()} >= {cfg.max_tokens}"
        if self.consecutive_failures >= cfg.max_consecutive_failures:
            return f"{self.consecutive_failures} failures in a row"
        if self.no_improve >= cfg.max_no_improve:
            return f"plateau: {self.no_improve} experiments without improvement"
        return ""

    # ---------------------------------------------------------------- LLM
    def _llm(self, method: str, messages: list[dict], *args: Any, **kw: Any) -> Any:
        """Вызов LLM: подсказка «без рассуждения», откат без неё при 400, повтор при пустом ответе."""
        cfg = self.cfg
        if cfg.no_think:
            messages = _no_think(messages)
        extra = dict(self.llm_extra)
        try:
            result = self._llm_once(method, messages, *args, **{**extra, **kw})
        except LLMUnavailable:
            raise
        except Exception as error:
            if not (extra and _is_bad_request(error)):
                raise
            log.info("research: gateway rejected %s (%s), calling without it", sorted(extra),
                     P.clip(str(error), 200))
            self.llm_extra = {}
            result = self._llm_once(method, messages, *args, **kw)
        if method == "chat" and not strip_think(result) and kw.get("max_tokens"):
            kw["max_tokens"] = min(cfg.max_answer_tokens, int(kw["max_tokens"] * cfg.empty_retry_factor))
            log.info("research: empty answer (reasoning used the token limit?), retry with max_tokens=%d",
                     kw["max_tokens"])
            result = self._llm_once(method, messages, *args, **{**self.llm_extra, **kw})
        return result

    def _llm_once(self, method: str, messages: list[dict], *args: Any, **kw: Any) -> Any:
        before, result = _read_usage(self.llm), None
        try:
            result = _call(getattr(self.llm, method), messages, *args, **kw)
            return result
        except LLMUnavailable:
            raise
        except Exception as error:
            if _is_budget_error(error):
                raise LLMUnavailable(str(error)) from error
            raise
        finally:
            after = _read_usage(self.llm)
            if before is not None and after is not None:
                spent_in, spent_out = after[0] - before[0], after[1] - before[1]
            else:  # клиент не отдаёт счётчик — грубая оценка по длине
                answer = result if isinstance(result, str) else json.dumps(result, default=str) if result else ""
                spent_in, spent_out = P.messages_tokens(messages), P.estimate_tokens(answer)
            self.tokens_in += spent_in
            self.tokens_out += spent_out
            self.llm_calls += 1

    def _cv_text(self) -> str:
        return f"CV {self.data.spec_dict['metric']} = {float(self.current.cv_score):.5g}"

    def _plan(self) -> None:
        cfg = self.cfg
        messages = P.build_plan_messages(
            task_line=self.task_line, task_text=self.task_text, profile=self.profile_text, base_code=self.base_code,
            cv_text=self._cv_text(), n=cfg.hypotheses_per_plan, errors=self.errors_text,
            can_tune=self.tuner is not None, can_search_models=cfg.model_search,
            memory=self.memory.planner_brief(cfg.memory_chars, kinds=self._kinds(), task_type=self.task_type),
            task_chars=cfg.task_text_chars, profile_chars=cfg.profile_chars, memory_chars=cfg.memory_chars)
        kw = dict(max_tokens=cfg.plan_max_tokens, temperature=cfg.plan_temperature)
        data: Any = None
        try:
            if callable(getattr(self.llm, "chat_json", None)):
                try:
                    data = self._llm("chat_json", messages, P.PLAN_SCHEMA, **kw)
                except LLMUnavailable:
                    raise
                except Exception as error:  # клиент не справился с JSON — разбираем сами
                    log.info("research: planner chat_json failed (%s), parsing plain chat", error)
            if not _is_payload(data):
                data = extract_json(self._llm("chat", messages, **kw))
        except LLMUnavailable:
            raise
        except Exception as error:
            log.info("research: no usable plan: %s", P.clip(str(error), 300))
            data = None
        tried = self.memory.tried_titles() + [h.title for h in self.queue]
        hypotheses, analysis = normalize_hypotheses(data, self.task_type, tried, cfg.hypotheses_per_plan,
                                                    is_duplicate=self._duplicate_check(),
                                                    can_tune=self.tuner is not None,
                                                    can_search_models=cfg.model_search)
        if analysis:
            self.insights = P.clip(analysis, 500)
        self.queue.extend(hypotheses)
        log.info("research: plan → %d new hypotheses: %s", len(hypotheses),
                 "; ".join(f"[{h.kind}] {h.title}" for h in hypotheses))

    def _kinds(self) -> tuple[str, ...]:
        return (P.KINDS + ((P.MODEL_SEARCH_KIND,) if self.cfg.model_search else ())
                + ((P.TUNE_KIND,) if self.tuner is not None else ()))

    def _duplicate_check(self) -> Callable[[str, str], bool]:
        """Повтор по смыслу: уже проверенная гипотеза, гипотеза в очереди или выше в этом же плане.
        Выбор модели — повтор, если он уже в плане/очереди или уже делался от текущего лучшего кода."""
        pending = [(h.title, h.kind) for h in self.queue]
        base = code_hash(self.base_code)

        def is_duplicate(title: str, kind: str) -> bool:
            if kind == P.MODEL_SEARCH_KIND:  # название не важно: важно, от какого кода сравнивали модели
                if any(k == kind for _, k in pending) or any(
                        e.kind == kind and e.base_hash == base for e in self.memory.experiments):
                    log.info("research: skip duplicate model search: %s", title)
                    return True
                pending.append((title, kind))
                return False
            same = self.memory.find_similar(title, kind)
            if same is not None or any(self.memory.same_idea(title, kind, t, k) for t, k in pending):
                log.info("research: skip duplicate hypothesis [%s] %s%s", kind, title,
                         f" (same idea as #{same.id})" if same is not None else "")
                return True
            pending.append((title, kind))
            return False

        return is_duplicate

    def _maybe_critic(self) -> None:
        """Перед re-plan: короткий разбор неудач, если их накопилось и токены позволяют."""
        cfg = self.cfg
        if not cfg.use_critic or self.since_critic < 2 or self.tokens_used() > 0.7 * cfg.max_tokens:
            return
        self.since_critic = 0
        messages = P.build_critic_messages(task_line=self.task_line, memory=self.memory.summary(cfg.memory_chars))
        try:
            note = strip_think(self._llm("chat", messages, max_tokens=cfg.critic_max_tokens, temperature=0.3))
        except LLMUnavailable:
            raise
        except Exception as error:
            log.info("research: critic failed: %s", error)
            return
        if note:
            self.memory.add_note(P.clip(note, 800))

    def _write_code(self, hypothesis: Hypothesis) -> str | None:
        cfg = self.cfg
        messages = P.build_code_messages(
            task_line=self.task_line, columns=self.columns_text, insights=self.insights, base_code=self.base_code,
            cv_text=self._cv_text(), hypothesis=asdict(hypothesis),
            memory=self.memory.summary(cfg.coder_memory_chars, recent=4), memory_chars=cfg.coder_memory_chars,
            model_params=MODEL_PARAMS, timeout=cfg.features_timeout)
        kw = dict(max_tokens=cfg.code_max_tokens, temperature=cfg.code_temperature)
        answer = self._llm("chat", messages, **kw)
        code = extract_code(answer)
        for _ in range(cfg.max_format_fixes):
            if code is not None:
                break
            messages = P.build_format_fix_messages(messages, str(answer or ""))
            answer = self._llm("chat", messages, **kw)
            code = extract_code(answer)
        return code

    def _repair(self, hypothesis: Hypothesis, code: str, error: str) -> str | None:
        cfg = self.cfg
        messages = P.build_repair_messages(
            task_line=self.task_line, columns=self.columns_text, hypothesis=asdict(hypothesis), code=code,
            error=error, model_params=MODEL_PARAMS, timeout=cfg.features_timeout, error_chars=cfg.error_chars)
        return extract_code(self._llm("chat", messages, max_tokens=cfg.code_max_tokens,
                                      temperature=cfg.code_temperature))

    # ---------------------------------------------------------------- эксперимент
    def _evaluate_code(self, code: str, name: str) -> tuple[Candidate | None, str]:
        """Проверка кода → песочница → проверка признаков → evaluator. (кандидат, "") или (None, ошибка)."""
        cfg = self.cfg
        problems = validate_code(code)
        consts: dict = {}
        if not problems:
            consts, problems = extract_constants(code)
        if problems:
            return None, "static check failed: " + "; ".join(problems)
        model, params, transform, notes = normalize_model_config(consts, self.task_type)

        features = run_features(code, self.workdir / name, self.inputs, self.data, cfg, self.run_code,
                                timeout=min(cfg.features_timeout, max(1.0, self._left())))
        if features.error:
            return None, features.error
        if self._left() < max(10.0, 0.5 * self.reference_seconds):
            return None, "no time left to evaluate"
        try:
            budget = _SubDeadline(self.deadline, min(cfg.evaluate_timeout, self._left()))
            cand = self.evaluate(self.spec, make_build_features(features.x_train, features.x_test),
                                 {"model": model, "params": params, "target_transform": transform}, budget)
        except Exception as error:
            return None, f"evaluation failed: {type(error).__name__}: {P.clip(str(error), 1000)}"
        if cand is None or not math.isfinite(float(getattr(cand, "cv_score", math.nan))):
            return None, "evaluation returned no finite CV score"
        if not isinstance(getattr(cand, "meta", None), dict):
            cand.meta = {}
        cand.meta.update(code=code, model=model, params=params, target_transform=transform,
                         n_features=int(features.x_train.shape[1]), harness_notes=notes + features.notes)
        return cand, ""

    def _iterate(self, hypothesis: Hypothesis) -> None:
        cfg = self.cfg
        self.iteration += 1
        n = self.iteration
        start, tokens_in, tokens_out = time.monotonic(), self.tokens_in, self.tokens_out
        status, error, digest = "failed", "", ""
        base_code, last_code = self.base_code, None  # база до эксперимента: _judge её заменит при принятии
        cand: Candidate | None = None
        decision: Decision | None = None
        note = ""  # итог стратегии (какие модели сравнили) — в память рядом с решением
        try:
            if hypothesis.kind == P.TUNE_KIND:
                code, error = self._tuned_code(n)
                repairs = cfg.max_repairs  # чинить нечего: код писал не LLM
            elif hypothesis.kind == P.MODEL_SEARCH_KIND:
                code, cand, note = self._model_search(n)  # кандидат уже оценён: второй раз не оцениваем
                error = "" if cand is not None else note
                if cand is not None:
                    last_code, digest = code, code_hash(code)
                repairs = cfg.max_repairs
            else:
                code = self._write_code(hypothesis)
                error = "" if code else "answer has no ```python code block"
                repairs = 0
            while code is not None and cand is None:
                last_code = code
                digest = code_hash(code)
                same = self.memory.find_code(digest)
                if digest == code_hash(self.base_code) or same is not None:
                    status = "rejected"
                    error = (f"same code as experiment #{same.id}" if same is not None
                             else "no change vs the current best code")
                    break
                cand, error = self._evaluate_code(code, f"exp_{n:02d}")
                if cand is not None or repairs >= cfg.max_repairs or self._left() < self._expected_seconds():
                    break
                repairs += 1
                fixed = self._repair(hypothesis, code, error)
                if fixed is None:
                    error += " | repair answer has no code block"
                code = fixed
            if cand is not None:
                decision = decide(cand, self.current, self.greater, cfg)
                status, error = self._judge(cand, decision, hypothesis, n)
                ignored = [note for note in cand.meta.get("harness_notes", []) if note.startswith(CONSTANTS)]
                if ignored:  # LLM должна узнать, что её MODEL/PARAMS не применились
                    error = f"harness: {'; '.join(ignored)} | {error}"
                if note:
                    error = f"{note} | {error}"
        except LLMUnavailable:
            raise
        except Exception as exc:  # ошибка харнесса на одном эксперименте не должна останавливать цикл
            log.exception("research: experiment %d crashed", n)
            status, error = "failed", f"internal error: {type(exc).__name__}: {_plain_error(exc, 300)}"

        duration = time.monotonic() - start
        changed = code_diff(base_code, last_code, max_lines=0) if last_code else ""
        self.memory.add(Experiment(
            id=self.memory.next_id(), hypothesis=hypothesis.title, kind=hypothesis.kind, status=status,
            cv_score=float(cand.cv_score) if cand is not None else None,
            cv_std=float(cand.cv_std) if cand is not None else None,
            gain=decision.gain if decision is not None else None, code_hash=digest, error=error[:1000],
            duration_s=round(duration, 1), tokens_in=self.tokens_in - tokens_in,
            tokens_out=self.tokens_out - tokens_out, columns=self._touched_columns(changed),
            diff=code_diff(base_code, last_code) if last_code else "", base_hash=code_hash(base_code)))
        if hypothesis.kind not in (P.TUNE_KIND, P.MODEL_SEARCH_KIND):  # у стратегий свой бюджет времени
            self.durations.append(duration)
        self.consecutive_failures = self.consecutive_failures + 1 if status == "failed" else 0
        self.no_improve = 0 if status == "accepted" else self.no_improve + 1
        self.since_critic += status != "accepted"
        log.info("research #%d [%s] [%s] %s | %s | %.0fs | tokens %d", n, status, hypothesis.kind,
                 hypothesis.title, P.clip(error, 300), duration, self.tokens_used())
        try:
            self.memory.dump_jsonl(self.workdir / "experiments.jsonl")
        except OSError:
            pass

    def _tuned_code(self, n: int) -> tuple[str | None, str]:
        """Подбор гиперпараметров на признаках текущего лучшего кода; результат — тот же код с новым PARAMS."""
        cfg = self.cfg
        consts, problems = extract_constants(self.base_code)
        if problems:
            return None, "tuning: " + "; ".join(problems)
        model, params, transform, _ = normalize_model_config(consts, self.task_type)
        features = run_features(self.base_code, self.workdir / f"exp_{n:02d}_tune", self.inputs, self.data, cfg,
                                self.run_code, timeout=min(cfg.features_timeout, max(1.0, self._left())))
        if features.error:
            return None, f"tuning: features of the current best failed: {P.clip(features.error, 500)}"
        budget = min(cfg.tune_seconds, cfg.tune_share * self._left())
        if budget < cfg.tune_min_seconds:
            return None, f"tuning: only {budget:.0f}s available"
        settings = {"model": model, "params": dict(params), "target_transform": transform,
                    "n_trials": cfg.tune_trials, "seed": cfg.tune_seed,
                    "evaluate": self.evaluate}  # подбор на тех же фолдах и тем же обучением, что оценка агента
        start = time.monotonic()
        try:
            found = self.tuner(self.spec, make_build_features(features.x_train, features.x_test), settings,
                               _SubDeadline(self.deadline, budget))
        except Exception as error:
            return None, f"tuning failed: {type(error).__name__}: {P.clip(str(error), 500)}"
        if not isinstance(found, dict):
            return None, f"tuning returned {type(found).__name__}, expected a dict of hyperparameters"
        _, tuned, _, notes = normalize_model_config({"MODEL": model, "PARAMS": {**params, **found}}, self.task_type)
        log.info("research: tuning %s in %.0fs → %s%s", model, time.monotonic() - start, tuned,
                 f" ({'; '.join(notes)})" if notes else "")
        return set_constant(self.base_code, "PARAMS", tuned), ""

    def _model_search(self, n: int) -> tuple[str | None, Candidate | None, str]:
        """Выбор модели: признаки текущего лучшего кода строятся один раз, каждая другая модель из MODELS
        оценивается тем же evaluator. Результат — (код лучшей, её уже оценённый кандидат, итог сравнения);
        принимать ли её вместо текущей, решает обычное правило decide."""
        cfg = self.cfg
        consts, problems = extract_constants(self.base_code)
        if problems:
            return None, None, "model search: " + "; ".join(problems)
        current, params, transform, _ = normalize_model_config(consts, self.task_type)
        budget = min(cfg.model_search_seconds, cfg.model_search_share * self._left())
        if budget < cfg.model_search_min_seconds:
            return None, None, f"model search: only {budget:.0f}s available"
        end = time.monotonic() + budget
        features = run_features(self.base_code, self.workdir / f"exp_{n:02d}_models", self.inputs, self.data, cfg,
                                self.run_code, timeout=min(cfg.features_timeout, max(1.0, self._left())))
        if features.error:
            return None, None, f"model search: features of the current best failed: {P.clip(features.error, 500)}"
        build = make_build_features(features.x_train, features.x_test)
        sign = 1.0 if self.greater else -1.0
        report = [f"{current} (current) cv={float(self.current.cv_score):.5g}"]
        best_code, best = None, None
        for model in sorted((m for m in MODELS if m != current), key=MODEL_COST.get):
            left = end - time.monotonic()
            need = self.reference_seconds * MODEL_COST[model] / MODEL_COST[current]
            if need > left:
                report.append(f"{model} skipped (needs ~{need:.0f}s, {left:.0f}s left)")
                continue
            moved, dropped = translate_params(params, current, model)
            _, moved, _, _ = normalize_model_config({"MODEL": model, "PARAMS": moved}, self.task_type)
            try:
                cand = self.evaluate(self.spec, build, {"model": model, "params": moved, "target_transform": transform},
                                     _SubDeadline(self.deadline, left))
            except Exception as error:
                report.append(f"{model} failed ({type(error).__name__}: {P.clip(str(error), 120)})")
                continue
            if cand is None or not math.isfinite(float(getattr(cand, "cv_score", math.nan))):
                report.append(f"{model} failed (no finite CV score)")
                continue
            report.append(f"{model} cv={float(cand.cv_score):.5g}")
            if best is None or sign * (float(cand.cv_score) - float(best.cv_score)) > 0:
                if not isinstance(getattr(cand, "meta", None), dict):
                    cand.meta = {}
                best_code = set_constant(set_constant(self.base_code, "MODEL", model), "PARAMS", moved)
                cand.meta.update(code=best_code, model=model, params=moved, target_transform=transform,
                                 n_features=int(features.x_train.shape[1]), harness_notes=list(features.notes),
                                 dropped_params=dropped)
                best = cand
        summary = "models: " + "; ".join(report)
        log.info("research: %s", summary)
        if best is None:
            return None, None, f"model search: no other model evaluated ({summary})"
        return best_code, best, summary

    def _touched_columns(self, changed: str) -> list[str]:
        """Колонки, которых касается дифф: названные литералами и целые группы spec (spec.get("cat_cols"))."""
        columns = columns_in(changed, self.memory.columns)
        for group in ("cat_cols", "date_cols", "text_cols"):
            if re.search(rf"""['"]{group}['"]""", changed):
                columns += self.data.spec_dict.get(group, [])
        return list(dict.fromkeys(columns))

    def _judge(self, cand: Candidate, decision: Decision, hypothesis: Hypothesis, n: int) -> tuple[str, str]:
        if not decision.accept:
            return "rejected", decision.reason
        if decision.suspicious:
            issues = self._check_guards(cand)
            if issues:
                return "rejected", f"suspicious jump, guards: {P.clip('; '.join(map(str, issues)), 300)}"
            if issues is None:
                cand.meta["suspicious"] = True
                log.warning("research: suspicious jump (%s) accepted without guards", decision.reason)
        cand.name = f"agent_{n:02d}_{hypothesis.kind}"
        cand.meta.update(hypothesis=hypothesis.title, kind=hypothesis.kind, iter=n, gain=decision.gain,
                         seconds=round(self.reference_seconds, 1))
        self.accepted.append(cand)
        self.current, self.base_code = cand, cand.meta["code"]
        self._update_errors()
        return "accepted", decision.reason

    def _check_guards(self, cand: Candidate) -> list | None:
        """guards.check_candidate (P5): список проблем; None — проверить нечем."""
        if self.guards is None:
            return None
        try:
            return list(self.guards(self.spec, cand, self.current) or [])
        except Exception as error:
            log.warning("research: guards failed: %s", error)
            return None


def research(spec: Any, best: Candidate | None, llm: LLMClient | None, deadline: Any, *,
             config: AgentConfig | None = None,
             evaluate: EvaluateFn | None = None,
             run_code: RunCodeFn | None = None,
             profile: Callable | None = None,
             guards: GuardFn | None = None,
             data: Any = None,
             tuner: TuneFn | None = None) -> list[Candidate]:
    """Принятые кандидаты по убыванию качества CV (первый — лучший) или [].

    Необязательные аргументы — для тестов и для P1: подменить evaluator/песочницу/профиль
    и передать уже загруженные данные. По умолчанию берутся модули команды, если они есть.
    """
    cfg = config or AgentConfig()
    if llm is None:
        log.info("research: LLM нет — исследование пропущено")
        return []
    if os.environ.get("LLM_DISABLED") == "1" and not getattr(llm, "offline", False):
        log.info("research: LLM_DISABLED=1 — исследование пропущено")
        return []
    left = _time_left(deadline)
    if left < cfg.min_start_seconds:
        log.info("research: до конца окна %.0f с < %.0f с — исследование пропущено", left, cfg.min_start_seconds)
        return []
    evaluate = evaluate or _optional("evaluator", "evaluate")
    if evaluate is None and cfg.local_evaluator:
        log.info("research: evaluator.evaluate (P1) недоступен — оценка своим LocalEvaluator (agent_eval.py)")
        evaluate = LocalEvaluator(early_stopping=100)  # с fallback.py (P3) — его обучение моделей
    if evaluate is None:
        log.warning("research: evaluator.evaluate недоступен — исследование пропущено")
        return []

    loop = None
    try:
        loop = _Research(spec, best, llm, deadline, cfg, evaluate=evaluate,
                         run_code=run_code or _optional("sandbox", "run_code") or _local_run_code,
                         profile=profile or _optional("eda", "profile"),
                         guards=guards or _optional("guards", "check_candidate"), data=data,
                         tuner=tuner or _optional("tuning", "tune"))
        return loop.run()
    except LLMUnavailable as error:
        log.warning("research: LLM недоступна (%s) — возвращаем накопленное", P.clip(str(error), 300))
        if loop is not None:
            loop.postprocess()  # без LLM: множители классов по уже посчитанным OOF
    except Exception:
        log.exception("research: непредвиденная ошибка — возвращаем накопленное")
    try:
        return loop.result() if loop is not None else []
    except Exception:
        return []
