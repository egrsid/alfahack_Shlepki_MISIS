"""Прогон ZIP-сабмита на тестовых задачах (роль P5, написано P4 — владелец может переписать). Вне ZIP.

    .venv/bin/python tools/make_datasets.py                       # один раз: задачи в .agent_work/datasets/
    .venv/bin/python tools/run_all.py                             # собрать ZIP из рабочей копии и прогнать все
    .venv/bin/python tools/run_all.py dist/solution_abc.zip --only bin_auc,reg_no_id --timeout 1800

На каждую задачу: чистая папка = файлы задачи + распакованный ZIP (как на платформе), запуск
`python3 solution.py task.md` без LLM (LLM_DISABLED=1, если не передан --llm), затем проверка `submission.csv`
(`submission.check_file`) и метрика по спрятанному таргету test против константы. Итог — таблица; код возврата 1,
если хоть одна задача без корректного файла.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fallback  # noqa: E402
import metrics  # noqa: E402
import submission  # noqa: E402

DATASETS = ROOT / ".agent_work" / "datasets"
RUNS = ROOT / ".agent_work" / "runs"


def _aligned(pred: pd.DataFrame, answers: pd.DataFrame, id_col: str | None) -> pd.DataFrame:
    if id_col is None:
        return pred
    order = pred.set_index(pred[id_col].astype(str))
    return order.loc[answers[id_col].astype(str)].reset_index(drop=True)


def score(task_root: Path, frame: pd.DataFrame) -> tuple[float, float]:
    """(метрика сабмита, метрика константы из train) по спрятанному таргету."""
    meta = json.loads((task_root / "meta.json").read_text(encoding="utf-8"))
    answers = pd.read_csv(task_root / "answers.csv")
    train = pd.read_csv(next((task_root / "task").rglob("train.csv")))
    targets = meta["target"] if isinstance(meta["target"], list) else [meta["target"]]
    pred = _aligned(frame, answers, meta["id_col"])
    got, const = [], []
    for col in targets:
        y = answers[col].to_numpy()
        if meta["proba"]:
            classes = metrics.classes_of(train[col])
            prior = float(np.mean(train[col].astype(str) == classes[-1]))
            base = np.full(len(y), prior)
            y = metrics.encode(y, classes)
        elif metrics.get(meta["metric"]).task == "regression":
            base = np.full(len(y), float(train[col].median() if meta["metric"] in ("mae", "mape") else train[col].mean()))
        else:
            base = np.full(len(y), train[col].mode().iloc[0])
        got.append(metrics.score(meta["metric"], y, pred[col].to_numpy()))
        const.append(metrics.score(meta["metric"], y, base))
    return float(np.mean(got)), float(np.mean(const))


def run_one(name: str, zip_path: Path, timeout: float, llm: bool) -> dict:
    task_root = DATASETS / name
    run_dir = RUNS / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(task_root / "task", run_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(run_dir)
    env = {k: v for k, v in os.environ.items() if llm or not k.startswith("OPENAI_")}
    if not llm:
        env["LLM_DISABLED"] = "1"
    start = time.monotonic()
    row = {"task": name, "seconds": 0.0, "ok": False, "score": None, "const": None, "note": ""}
    with open(run_dir / "run.log", "w", encoding="utf-8") as log_file:
        try:
            proc = subprocess.run([sys.executable, "solution.py", "task.md"], cwd=run_dir, env=env,
                                  stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout)
            row["note"] = f"код {proc.returncode}" if proc.returncode else ""
        except subprocess.TimeoutExpired:
            row["note"] = f"таймаут {timeout:.0f} с"
    row["seconds"] = round(time.monotonic() - start)
    try:
        spec = fallback.guess_spec(run_dir / "task.md")
        problems = submission.check_file(spec)
        row["ok"] = not problems
        if problems:
            row["note"] = (row["note"] + "; " if row["note"] else "") + "; ".join(problems)[:200]
        elif (task_root / "meta.json").is_file():  # у public-задач ответов нет — только проверка файла
            row["score"], row["const"] = score(task_root, pd.read_csv(run_dir / "submission.csv"))
    except Exception as error:  # noqa: BLE001 — отчёт, а не падение прогона
        row["note"] = (row["note"] + "; " if row["note"] else "") + f"{type(error).__name__}: {error}"[:200]
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip", nargs="?", type=Path, help="ZIP; по умолчанию собирается из рабочей копии")
    parser.add_argument("--only", default="")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--llm", action="store_true", help="не выключать LLM (тратит общий бюджет)")
    args = parser.parse_args(argv)
    zip_path = args.zip
    if zip_path is None:
        sys.path.insert(0, str(ROOT / "tools"))
        import make_zip
        zip_path = make_zip.build(RUNS / "solution_local.zip")
    names = [n for n in args.only.split(",") if n] or sorted(p.name for p in DATASETS.iterdir() if p.is_dir())
    rows = []
    for name in names:
        row = run_one(name, zip_path, args.timeout, args.llm)
        rows.append(row)
        fmt = lambda v: "—" if v is None else f"{v:.4g}"  # noqa: E731
        print(f"{row['task']:<18} {row['seconds']:>5} с  {'ok ' if row['ok'] else 'FAIL'}  "
              f"score {fmt(row['score']):>8}  const {fmt(row['const']):>8}  {row['note']}", flush=True)
    failed = [r["task"] for r in rows if not r["ok"]]
    print(f"\nзадач {len(rows)}, без корректного файла: {len(failed)} {failed if failed else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
