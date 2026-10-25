"""Прогон настоящего solution.solve на задаче — весь конвейер, как на платформе (вне ZIP).

    .venv/bin/python tools/try_solution.py "Регрессия/task.md" --dry-run          # MockLLM: бюджет не тратится
    .venv/bin/python tools/try_solution.py "Регрессия/task.md" --iters 3          # живой Qwen (тратит общий бюджет)
    .venv/bin/python tools/try_solution.py "Регрессия/task.md" --automl 300       # + AutoGluon на 5 минут

В отличие от tools/try_agent.py (агент отдельно, свой клиент) здесь работает ровно то, что в ZIP: понимание задачи,
fallback, [AutoGluon], агент через RecipeEvaluator от лучшего кандидата fallback, бленд, запись и проверка файла;
LLM — тот же клиент llm_client (P2), что на платформе. Ключ — из окружения или из .env в корне репозитория
(загружается в окружение процесса, нигде не печатается).
Задача копируется в .agent_work/try/<имя>/ — папки с данными не засоряются. Отчёт — там же, report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import agent  # noqa: E402
import solution  # noqa: E402

# код признаков для --dry-run: в режиме «поверх fallback» добавляются только новые колонки
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
'''

# что ищем в логе: (ключ отчёта, начало сообщения)
MARKERS = (("environment", "окружение:"), ("understanding", "понимание задачи"), ("fallback", "fallback готов"),
           ("automl", "automl: "), ("reference", "research: reference"), ("research_stop", "research: stop"),
           ("written", "submission.csv ←"), ("done", "готово за"))


class Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def copy_task(task_md: Path, out: Path) -> Path:
    """task.md и все CSV задачи (с вложенными путями) в чистую папку."""
    src = task_md.parent
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    shutil.copy(task_md, out / "task.md")
    for path in src.rglob("*.csv"):
        rel = path.relative_to(src)
        if ".agent_work" in rel.parts or path.name == "submission.csv":
            continue
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, out / rel)
    return out / "task.md"


def mock_llm() -> object:
    from mocks import MockLLM, fenced, plan
    return MockLLM({"task_analyst": ['{"date_cols": [], "text_cols": [], "cat_cols": [], "drop_cols": []}'],
                    "planner": [plan("Count encoding of categorical columns", kind="encoding")],
                    "coder": [fenced(COUNT_CODE)], "critic": ["Nothing to add."]})


def live_llm(task_dir: Path) -> object:
    from llm_client import _load_dotenv
    if not os.environ.get("OPENAI_API_KEY"):
        _load_dotenv(str(ROOT / ".env"))  # ключ — только в окружение процесса
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("нет OPENAI_API_KEY ни в окружении, ни в .env — для проверки без LLM есть --dry-run")
    os.environ.pop("LLM_DISABLED", None)
    client = solution.make_llm(task_dir)
    if client is None:
        raise SystemExit("solution.make_llm не создал клиент — см. лог выше")
    return client


def usage(llm: object) -> dict:
    inner = getattr(llm, "client", None)
    if inner is not None and callable(getattr(inner, "usage_summary", None)):
        return inner.usage_summary()
    calls = getattr(llm, "calls", [])
    return {"calls": len(calls), "roles": [role for role, _, _ in calls]}


def report(records: list[logging.LogRecord], run_dir: Path, llm: object, seconds: float, problems: list) -> dict:
    messages = [r.getMessage() for r in records]
    found = {key: [m for m in messages if m.startswith(prefix)] for key, prefix in MARKERS}
    experiments = []
    path = run_dir / ".agent_work" / "experiments.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            experiments.append({k: row.get(k) for k in ("id", "kind", "status", "hypothesis", "cv_score", "gain",
                                                          "error", "duration_s")})
    return {"seconds": round(seconds), "submission_problems": problems, "llm_usage": usage(llm),
            "recipe_mode": bool(found["reference"]) and "fallback recipe" in found["reference"][0],
            **{key: values for key, values in found.items()}, "experiments": experiments,
            "warnings": [f"{r.name}: {r.getMessage()[:300]}" for r in records if r.levelno >= logging.WARNING]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task_md", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="MockLLM вместо Qwen: бюджет не тратится")
    parser.add_argument("--iters", type=int, default=3, help="итераций агента")
    parser.add_argument("--minutes", type=float, default=30.0, help="конец окна исследования от старта, мин")
    parser.add_argument("--automl", type=float, default=0.0, help="секунд на AutoGluon (0 — без него)")
    parser.add_argument("--out", type=Path, default=ROOT / ".agent_work" / "try")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    collector = Collector()
    root = logging.getLogger()
    root.setLevel(logging.INFO)  # basicConfig не трогает уже настроенный логгер (например, под pytest)
    root.addHandler(collector)
    task_md = copy_task(args.task_md.resolve(), args.out / args.task_md.resolve().parent.name)
    os.environ["SOLUTION_AUTOML_SECONDS"] = str(args.automl)
    llm = mock_llm() if args.dry_run else live_llm(task_md.parent)
    start = time.monotonic()
    deadline = solution.Deadline(start, solution.TOTAL_SECONDS, args.minutes * 60)
    problems = solution.solve(task_md, llm=llm, deadline=deadline,
                              agent_config=agent.AgentConfig(max_iters=args.iters))
    root.removeHandler(collector)
    result = report(collector.records, task_md.parent, llm, time.monotonic() - start, problems)
    (task_md.parent / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                                encoding="utf-8")
    print("\n===== отчёт:", task_md.parent / "report.json")
    for key in ("environment", "understanding", "fallback", "reference", "research_stop", "written", "done"):
        for line in result[key][-2:]:
            print(f"[{key}] {line[:260]}")
    print("[recipe_mode]", result["recipe_mode"], "| [llm]", result["llm_usage"])
    for row in result["experiments"]:
        print(f"  #{row['id']} [{row['status']}] [{row['kind']}] {str(row['hypothesis'])[:70]} | cv={row['cv_score']} "
              f"gain={row['gain']} | {str(row['error'] or '')[:120]}")
    print("[файл]", "ok" if not problems else problems)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
