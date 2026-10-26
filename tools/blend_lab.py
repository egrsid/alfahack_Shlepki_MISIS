"""Лаборатория бленда (роль P1; вне ZIP): кандидаты на отложенных копиях считаются один раз, стратегии бленда и
правила записи сравниваются за секунды.

    .venv/bin/python tools/blend_lab.py copy reg_year4 --last-train 3     # копия «годы 1–3 → год 4» из public
    .venv/bin/python tools/blend_lab.py build reg_year5                   # fallback → зоопарк A → сиды → зоопарк B
    .venv/bin/python tools/blend_lab.py eval reg_year5 reg_year4 cls_groups25

build повторяет порядок solution.solve без LLM и без записи файлов (fallback.run(write=False) → final.train →
бленд → final.bag → final.train(tag="b")) и сохраняет всё в .agent_work/holdout/<имя>/lab.pkl. eval собирает
бленды стратегиями из STRATEGIES (параметры ensemble.blend) и печатает CV (честное; по фолдам поздних периодов, как
сравнивает solution.pick), запишет ли его pick вместо fallback и метрику на отложенной части (все строки / строки
без новых уровней категорий). Копии и разметка — как в tools/holdout.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import ensemble  # noqa: E402
import fallback  # noqa: E402
import final  # noqa: E402
import solution  # noqa: E402
from holdout import HOLDOUT, scorer, write_meta  # noqa: E402

log = logging.getLogger("blend_lab")


def make_period_copy(name: str, source: Path, column: str, last_train: int) -> Path:
    """Копия public-задачи: train — строки с column ≤ last_train, test — column == last_train + 1 (без таргета),
    ответы — в answers.csv."""
    root = HOLDOUT / name
    task = root / "task"
    task.mkdir(parents=True, exist_ok=True)
    spec = fallback.guess_spec(source / "task.md")
    target, id_col = spec.target, spec.id_col
    train = pd.read_csv(source / "train.csv", low_memory=False)
    part, hold = train[train[column] <= last_train], train[train[column] == last_train + 1]
    part.to_csv(task / "train.csv", index=False)
    hold.drop(columns=[target]).to_csv(task / "test.csv", index=False)
    hold[[id_col, target]].to_csv(root / "answers.csv", index=False)
    pd.DataFrame({id_col: hold[id_col], target: float(part[target].mean())}).to_csv(
        task / "sample_submission.csv", index=False)
    for extra in ("task.md", "data_dictionary.csv"):
        if (source / extra).exists():
            shutil.copy(source / extra, task / extra)
    print(f"{name}: train {len(part)} строк, test {len(hold)} строк")
    return root


def build(name: str, zoo_a: float, zoo_b: float) -> Path:
    root = HOLDOUT / name
    spec = fallback.guess_spec(root / "task" / "task.md")
    write_meta(spec, root)
    deadline = solution.Deadline(time.monotonic(), solution.TOTAL_SECONDS, solution.RESEARCH_END)
    times = {}
    start = time.monotonic()
    best = fallback.run(spec, deadline, fallback.FallbackConfig(write=False))
    times["fallback"] = time.monotonic() - start
    base = [fallback.honest_view(best), *fallback.pool(best)]

    start = time.monotonic()
    zoo = final.train(spec, best, until=start + zoo_a)
    times["zoo_a"] = time.monotonic() - start
    mixed = ensemble.blend(spec, [*base, *zoo])
    weights = dict((mixed.meta or {}).get("blend") or {}) if mixed is not None else {}

    start = time.monotonic()
    until = start + zoo_b
    bagged = final.bag(spec, best, zoo, weights, until=until) if zoo and weights else []
    times["bag"] = time.monotonic() - start
    more = final.train(spec, best, until=until, n_random=2 * final.N_RANDOM, listed=False, tag="b", seed_offset=1)
    times["zoo_b"] = time.monotonic() - start - times["bag"]

    path = root / "lab.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"spec": spec, "best": best, "zoo": zoo, "bagged": bagged, "zoo_b": more, "times": times}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{name}: сохранено {path} ({path.stat().st_size / 1e6:.0f} МБ); время {json.dumps(times)}; "
          f"зоопарк A {len(zoo)}, сиды {len(bagged)}, зоопарк B {len(more)}")
    return path


def load(name: str) -> dict:
    with open(HOLDOUT / name / "lab.pkl", "rb") as fh:
        return pickle.load(fh)


def final_pool(lab: dict) -> tuple[list, list]:
    """(кандидаты fallback: честный лучший + модели, зоопарк в том виде, как он идёт в финальный бленд solution:
    досчитанные сиды вместо одиночных + зоопарк B)."""
    best = lab["best"]
    base = [fallback.honest_view(best), *fallback.pool(best)]
    replaced = {b.meta["bagged_from"] for b in lab["bagged"]}
    zoo = [*lab["bagged"], *(c for c in lab["zoo"] if c.name not in replaced), *lab["zoo_b"]]
    return base, zoo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_copy = sub.add_parser("copy")
    p_copy.add_argument("name")
    p_copy.add_argument("--source", default=str(ROOT / "Регрессия"))
    p_copy.add_argument("--column", default="Year_Factor")
    p_copy.add_argument("--last-train", type=int, required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("names", nargs="+")
    p_build.add_argument("--zoo-a", type=float, default=1800.0)
    p_build.add_argument("--zoo-b", type=float, default=1200.0)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("names", nargs="+")
    p_eval.add_argument("--only", default="", help="стратегии через запятую")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "copy":
        make_period_copy(args.name, Path(args.source), args.column, args.last_train)
    elif args.cmd == "build":
        for name in args.names:
            build(name, args.zoo_a, args.zoo_b)
    else:
        for name in args.names:
            evaluate(name, [s for s in args.only.split(",") if s])
    return 0


STRATEGIES: dict[str, dict] = {
    "prod": {},
    "per_fold": {"select": "per_fold"},
    "pool24_pf": {"select": "per_fold", "max_pool": 24},
    "bag10": {"bags": 10},
    "seg": {"segments": True},
    "seg_pf": {"segments": True, "select": "per_fold"},
    "seg_bag10": {"segments": True, "bags": 10},
    "last": {"fit": "last"},
    "last2": {"fit": "last2"},
    "seg_last2": {"segments": True, "fit": "last2"},
    "seg_last": {"segments": True, "fit": "last"},
    "last_bag10": {"fit": "last", "bags": 10},
    "last_bag20_pf": {"fit": "last", "bags": 20, "select": "per_fold"},
    "seg_last_bag10": {"segments": True, "fit": "last", "bags": 10},
    "last_pool24": {"fit": "last", "max_pool": 24, "select": "per_fold"},
    "fb_prod": {"_pool": "fallback"},
    "fb_seg": {"_pool": "fallback", "segments": True},
}


def paired(spec, cand, ref, folds) -> tuple[float, float]:
    """(средний выигрыш cand над ref по сравниваемым фолдам в сторону «лучше», его стандартная ошибка)."""
    a = np.asarray(solution.on_folds(fallback.honest_view(cand), folds).meta["fold_scores_eff"], dtype=float)
    b = np.asarray(solution.on_folds(fallback.honest_view(ref), folds).meta["fold_scores_eff"], dtype=float)
    diff = (a - b) * (1.0 if solution.metrics.greater_is_better(spec.metric) else -1.0)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return float(diff.mean()), se


def evaluate(name: str, only: list[str]) -> pd.DataFrame:
    lab = load(name)
    spec, best = lab["spec"], lab["best"]
    root = HOLDOUT / name
    score, seen_share = scorer(spec, root, root / "task")
    base, zoo = final_pool(lab)
    folds = ensemble.recent_folds(best.meta, len(np.asarray(best.oof)))
    rows = []

    def add(label: str, cand, seconds: float = 0.0, opts: dict | None = None) -> None:
        if cand is None:
            rows.append({"strategy": label, "cv": None})
            return
        view = fallback.honest_view(cand)
        hold, hold_seen = score(cand)
        gain, se = paired(spec, cand, best, folds) if cand is not best else (0.0, 0.0)
        writes = cand is not best and solution.pick(spec, best, [cand], best, "lab") is not None
        rows.append({"strategy": label, "cv": round(fallback.effective_cv(view), 5),
                     "cv_cmp": round(fallback.effective_cv(solution.on_folds(view, folds)), 5),
                     "gain": round(gain, 5), "se": round(se, 5), "pick": "write" if writes else "",
                     "holdout": round(hold, 5), "holdout_seen": round(hold_seen, 5), "sec": round(seconds, 1),
                     "members": len((cand.meta or {}).get("blend") or {})})

    add("fallback", best)
    for label, opts in STRATEGIES.items():
        if only and label not in only:
            continue
        opts = dict(opts)
        pool = base if opts.pop("_pool", "all") == "fallback" else [*base, *zoo]
        start = time.monotonic()
        try:
            cand = ensemble.blend(spec, pool, **opts)
        except Exception as error:  # noqa: BLE001 — стратегия упала: строка с ошибкой, остальные считаются
            log.exception("%s: %s", label, error)
            cand = None
        add(label, cand, time.monotonic() - start)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "lab_table.csv", index=False)
    print(f"\n## {name}: metric {spec.metric}; кандидатов fallback {len(base)}, зоопарка {len(zoo)}; "
          f"сравниваемые фолды {folds}; строк без новых уровней {seen_share:.1%}")
    print(frame.to_string(index=False))
    return frame


if __name__ == "__main__":
    sys.exit(main())
