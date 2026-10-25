"""Короткий прогон агента на public-задаче (шаг 8 плана ветки). Вне ZIP.

ВНИМАНИЕ: без --dry-run тратит общий бюджет команды на LLM — запускать редко, с --iters 3, по согласованию.

    .venv/bin/python tools/try_agent.py "Регрессия/task.md" --dry-run     # без LLM: проверка трубы и размера промптов
    .venv/bin/python tools/try_agent.py "Регрессия/task.md" --iters 3     # живой Qwen
    .venv/bin/python tools/try_agent.py "Регрессия/task.md" --dry-run --tune   # + подбор (StubTuner, пока нет P1)

Ключ и адрес — из переменных окружения или .env (python-dotenv). Оценка — evaluator.evaluate (P1), если он
есть, иначе LocalEvaluator агента (agent_eval.py) — как в боевом запуске.
Спецификация задачи — эвристики ниже (до task_understanding P2).
Отчёт: <каталог задачи>/.agent_work/try_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import agent  # noqa: E402
import prompts_agent as P  # noqa: E402
from mocks import (LIGHTGBM_GRID, MockLLM, SimpleDeadline, SpecStub, StubTuner, fenced, make_best,  # noqa: E402
                   plan)

METRICS = [(r"macro[\s_-]*f1|f1[\s_-]*macro", "macro_f1", True), (r"weighted[\s_-]*f1", "weighted_f1", True),
           (r"\brmse\b", "rmse", False), (r"\bmae\b", "mae", False), (r"roc[\s_-]*auc|\bauc\b", "roc_auc", True),
           (r"log[\s_-]*loss", "logloss", False), (r"\baccuracy\b", "accuracy", True), (r"\br2\b", "r2", True)]

# универсальный код для --dry-run: частотное кодирование категориальных колонок
COUNT_CODE = '''import numpy as np
import pandas as pd


def build_features(train, test, spec):
    X_train, X_test = train.copy(), test.copy()
    both = pd.concat([train, test], ignore_index=True)
    for col in spec.get("cat_cols", []):
        counts = both[col].value_counts()
        X_train[col + "_count"] = X_train[col].map(counts).astype(float)
        X_test[col + "_count"] = X_test[col].map(counts).astype(float)
    return X_train, X_test


MODEL = "lightgbm"
PARAMS = {}
TARGET_TRANSFORM = None
'''


def heuristic_spec(task_md: Path) -> SpecStub:
    """Минимальная спецификация по файлам задачи — до task_understanding (P2)."""
    task_dir = task_md.parent
    probe = SpecStub(task_dir=task_dir, id_col="", target="", task_type="", metric="", greater_is_better=False)
    sample = pd.read_csv(agent._find_file(probe, "sample_submission.csv"), nrows=5)
    id_col, target = sample.columns[0], sample.columns[1]
    train = pd.read_csv(agent._find_file(probe, "train.csv"), low_memory=False)
    y = train[target]
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        task_type, default = "regression", ("rmse", False)
    else:
        task_type = "binary" if y.nunique() == 2 else "multiclass"
        default = ("roc_auc", True) if task_type == "binary" else ("macro_f1", True)

    text = task_md.read_text(encoding="utf-8")
    found = [(m.start(), name, greater) for pattern, name, greater in METRICS
             for m in [re.search(pattern, text, re.I)] if m]
    metric, greater = (min(found)[1], min(found)[2]) if found else default

    notes: dict[str, str] = {}
    try:
        dictionary = pd.read_csv(agent._find_file(probe, "data_dictionary.csv"))
        name_col = next((c for c in dictionary.columns if c.lower() in ("feature", "name", "column")),
                        dictionary.columns[0])
        desc_col = next((c for c in dictionary.columns if "desc" in c.lower()), dictionary.columns[-1])
        unit_col = next((c for c in dictionary.columns if c.lower() == "unit"), None)
        for _, row in dictionary.iterrows():
            note = str(row[desc_col])
            if unit_col and pd.notna(row[unit_col]):
                note += f" [{row[unit_col]}]"
            notes[str(row[name_col])] = note
    except (FileNotFoundError, pd.errors.ParserError, IndexError):
        pass

    features = [c for c in train.columns if c not in (id_col, target)]
    cat_cols = [c for c in features if train[c].dtype == object]
    return SpecStub(task_dir=task_dir, id_col=id_col, target=target, task_type=task_type, metric=metric,
                    greater_is_better=greater, feature_notes=notes, cat_cols=cat_cols, raw_task_text=text)


class QwenClient:
    """Минимальный клиент для живого прогона (полноценный, с кэшем и ретраями, — llm_client.py у P2).

    chat_json намеренно нет: агент сам разберёт JSON из chat(), без второго платного вызова.
    """

    def __init__(self, raw_dir: Path | None = None, timeout: float = 180.0) -> None:
        from openai import OpenAI
        self.client = OpenAI(timeout=timeout, max_retries=2)  # ключ и адрес — только из env
        self.model = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.log: list[dict] = []
        self.raw_dir = raw_dir  # сырые ответы — для разбора сбоев формата

    def chat(self, messages: list[dict], max_tokens: int = 1500, temperature: float = 0.2,
             extra_body: dict | None = None) -> str:
        from openai import APIConnectionError, APIStatusError
        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           max_tokens=max_tokens, temperature=temperature,
                                                           extra_body=extra_body)
        except APIStatusError as error:
            if "budget" in str(error).lower():
                raise agent.LLMUnavailable(str(error)) from error
            raise
        except APIConnectionError as error:
            raise agent.LLMUnavailable(f"{type(error).__name__}: {error}") from error
        usage = response.usage
        if usage:
            self.usage["prompt_tokens"] += usage.prompt_tokens
            self.usage["completion_tokens"] += usage.completion_tokens
        content = response.choices[0].message.content or ""
        details = getattr(usage, "completion_tokens_details", None)
        role = MockLLM.role(messages)
        self.log.append({"role": role, "seconds": round(time.monotonic() - start, 1),
                         "prompt_tokens": usage.prompt_tokens if usage else None,
                         "completion_tokens": usage.completion_tokens if usage else None,
                         "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                         "max_tokens": max_tokens, "extra_body": extra_body,
                         "finish_reason": response.choices[0].finish_reason, "has_think": "<think>" in content})
        if self.raw_dir is not None:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            (self.raw_dir / f"{len(self.log):02d}_{role}.txt").write_text(content, encoding="utf-8")
        return content


def dry_run_llm() -> MockLLM:
    return MockLLM({"planner": [plan("Count encoding of categorical columns", kind="encoding")],
                    "coder": [fenced(COUNT_CODE)],
                    "critic": ["Nothing to add."]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_md", type=Path)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--minutes", type=float, default=30.0, help="окно research")
    parser.add_argument("--max-tokens", type=int, default=40_000, help="потолок токенов агента на прогон")
    parser.add_argument("--trees", type=int, default=300, help="деревьев в LocalEvaluator")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="MockLLM вместо Qwen: бюджет не тратится")
    parser.add_argument("--tune", action="store_true",
                        help="вид гипотезы tune: tuning.tune (P1) или, без него, перебор сетки StubTuner")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    spec = heuristic_spec(args.task_md.resolve())
    if args.dry_run:
        llm = dry_run_llm()
    else:
        if os.environ.get("LLM_DISABLED") == "1":
            print("LLM_DISABLED=1 — живой прогон отключён (используйте --dry-run)")
            return 1
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY не задан")
            return 1
        raw_dir = spec.task_dir / ".agent_work" / "llm_raw"
        for old in raw_dir.glob("*.txt"):
            old.unlink()
        llm = QwenClient(raw_dir=raw_dir)

    print(f"spec: {spec.task_type}, metric={spec.metric} (greater={spec.greater_is_better}), "
          f"target={spec.target}, id={spec.id_col}, cat_cols={spec.cat_cols}, notes={len(spec.feature_notes)}")

    evaluate = agent._optional("evaluator", "evaluate") or agent.LocalEvaluator(n_splits=args.folds,
                                                                                n_estimators=args.trees)
    start = time.monotonic()
    best = make_best(spec, evaluate) if isinstance(evaluate, agent.LocalEvaluator) else None
    print(f"fallback-кандидат: cv={best.cv_score:.5g} ± {best.cv_std:.3g} за {time.monotonic() - start:.0f} с"
          if best else "fallback-кандидат: нет (реальный evaluator)")

    config = agent.AgentConfig(max_iters=args.iters, max_tokens=args.max_tokens, min_start_seconds=60)
    start = time.monotonic()
    tuner = None
    if args.tune:
        tuner = agent._optional("tuning", "tune") or StubTuner(evaluator=evaluate, grid=LIGHTGBM_GRID,
                                                                greater_is_better=spec.greater_is_better)
    result = agent.research(spec, best, llm, SimpleDeadline(args.minutes * 60), config=config, evaluate=evaluate,
                            tuner=tuner)
    seconds = time.monotonic() - start

    workdir = spec.task_dir / config.workdir_name
    rows = [json.loads(line) for line in (workdir / "experiments.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = getattr(llm, "log", None) or [
        {"role": role, "prompt_tokens_est": P.messages_tokens(messages)} for role, messages, _ in llm.calls]
    report = {"task": str(args.task_md), "dry_run": args.dry_run, "seconds": round(seconds), "spec_metric": spec.metric,
              "best_cv": best.cv_score if best else None,
              "result": [{"name": c.name, "cv": c.cv_score, "hypothesis": c.meta.get("hypothesis")} for c in result],
              "experiments": rows, "llm_calls": calls,
              "usage": None if args.dry_run else llm.usage,
              "failed_share": round(sum(r["status"] == "failed" for r in rows) / max(1, len(rows)), 2)}
    (workdir / "try_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")

    print(f"\nисследование: {seconds:.0f} с, экспериментов {len(rows)}, принято в ответ {len(result)}")
    for row in rows:
        print(f"  #{row['id']} [{row['status']}] [{row['kind']}] {row['hypothesis']} | cv={row['cv_score']} "
              f"gain={row['gain']} | {row['duration_s']} с | tokens {row['tokens_in']}/{row['tokens_out']} | "
              f"{(row['error'] or '')[:120]}")
    for call in calls:
        print("  LLM:", call)
    if not args.dry_run:
        n = max(1, len(rows))
        print(f"токены: всего {llm.usage}, на итерацию ≈ "
              f"{(llm.usage['prompt_tokens'] + llm.usage['completion_tokens']) // n}")
    print("отчёт:", workdir / "try_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
