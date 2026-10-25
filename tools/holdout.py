"""Стенд замеров на отложенных копиях public-задач (роль P1; вне ZIP): что дают зоопарк (final.py) и бленды.

    .venv/bin/python tools/holdout.py                               # reg_year5 и cls_groups25
    .venv/bin/python tools/holdout.py reg_year5 --zoo 900 --random 0

Копия задачи — .agent_work/holdout/<имя>/{task/, answers.csv}: train — часть public-train, test — отложенная часть
с ответами (reg_year5: годы 1–4 → год 5; cls_groups25: 25% групп дублей). Без LLM и без записи файлов задачи:
fallback.run(write=False) → final.train (этап A) → бленды → final.bag (этап B) → бленд. Таблица на каждого кандидата:
CV, которое видит решение (честное, с порогом/весами классов; cv_recent — по фолдам поздних периодов, как сравнивает
solution.pick), «write» — pick записал бы его вместо fallback, и метрика на отложенной части — по всем строкам и
по строкам, где все категории встречались в train («без новых уровней»: в reg_year5 штат State_11 есть только
в test, на нём любая модель ошибается в разы — этот штат заслоняет остальное). Таблица — в
.agent_work/holdout/<имя>/final_table.csv; заодно пишется meta.json для tools/run_all.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ensemble  # noqa: E402
import fallback  # noqa: E402
import final  # noqa: E402
import metrics  # noqa: E402
import solution  # noqa: E402
import submission  # noqa: E402

HOLDOUT = ROOT / ".agent_work" / "holdout"


def seen_rows(task_dir: Path) -> np.ndarray:
    """Строки test, у которых значения всех строковых колонок встречались в train."""
    train = pd.read_csv(task_dir / "train.csv", low_memory=False)
    test = pd.read_csv(task_dir / "test.csv", low_memory=False)
    mask = np.ones(len(test), dtype=bool)
    for col in test.columns:
        if test[col].dtype == object and col in train.columns:
            mask &= test[col].isna().to_numpy() | test[col].isin(set(train[col].dropna())).to_numpy()
    return mask


def scorer(spec, root: Path, task_dir: Path):
    """Функция cand → (метрика по всем строкам отложенной части, по строкам без новых уровней)."""
    target = submission.targets_of(spec)[0]
    answers = pd.read_csv(root / "answers.csv")
    ids = pd.read_csv(task_dir / "test.csv", usecols=[spec.id_col])[spec.id_col]
    truth = answers.set_index(answers[spec.id_col].astype(str)).loc[ids.astype(str), target].to_numpy()
    seen = seen_rows(task_dir)

    def score(cand) -> tuple[float, float]:
        meta = cand.meta or {}
        classes = meta.get("classes")
        y = truth if classes is None else metrics.encode(truth, classes)
        pred, decision = np.asarray(cand.test_pred), meta.get("decision")
        return (metrics.score(spec.metric, y, pred, decision),
                metrics.score(spec.metric, y[seen], pred[seen], decision))
    return score, float(seen.mean())


def write_meta(spec, root: Path) -> None:
    meta = root / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"metric": spec.metric, "target": submission.targets_of(spec)[0],
                                    "id_col": spec.id_col, "proba": False}), encoding="utf-8")


def measure(name: str, zoo_seconds: float, n_random: int, bag: bool) -> pd.DataFrame:
    root = HOLDOUT / name
    task_dir = root / "task"
    spec = fallback.guess_spec(task_dir / "task.md")
    write_meta(spec, root)
    score, seen_share = scorer(spec, root, task_dir)
    deadline = solution.Deadline(time.monotonic(), solution.TOTAL_SECONDS, solution.RESEARCH_END)
    start = time.monotonic()
    best = fallback.run(spec, deadline, fallback.FallbackConfig(write=False))
    fallback_seconds = time.monotonic() - start
    base = [fallback.honest_view(best), *fallback.pool(best)]

    start = time.monotonic()
    zoo = final.train(spec, best, until=start + zoo_seconds, n_random=n_random)
    zoo_seconds_used = time.monotonic() - start
    rows: list[tuple[str, object]] = [(c.name, c) for c in base] + [(c.name, c) for c in zoo]
    plain = {"recent": False, "new_levels": False}
    blend_zoo = ensemble.blend(spec, [*base, *zoo])
    rows += [("ensemble: fallback (old)", ensemble.blend(spec, base, **plain)),
             ("ensemble: fallback", ensemble.blend(spec, base)),
             ("ensemble: fallback + zoo (old)", ensemble.blend(spec, [*base, *zoo], **plain)),
             ("ensemble: fallback + zoo, recent only", ensemble.blend(spec, [*base, *zoo], new_levels=False)),
             ("ensemble: fallback + zoo", blend_zoo)]
    bag_seconds = 0.0
    if bag and blend_zoo is not None:
        start = time.monotonic()
        bagged = final.bag(spec, best, zoo, blend_zoo.meta.get("blend") or {}, until=start + zoo_seconds)
        bag_seconds = time.monotonic() - start
        if bagged:
            replaced = {b.meta["bagged_from"] for b in bagged}
            rows += [(b.name, b) for b in bagged]
            rows.append(("ensemble: fallback + zoo + seeds",
                         ensemble.blend(spec, [*base, *bagged, *(c for c in zoo if c.name not in replaced)])))

    folds = ensemble.recent_folds(best.meta, len(np.asarray(best.oof)))
    table = []
    for label, cand in rows:
        if cand is None:
            continue
        view = fallback.honest_view(cand)
        hold, hold_seen = score(cand)
        recent = solution.on_folds(view, folds)
        writes = cand is not best and solution.pick(spec, best, [cand], best, "holdout") is not None
        weights = (cand.meta or {}).get("blend")
        extra = (cand.meta or {}).get("new_levels")
        table.append({"candidate": label, "cv": round(fallback.effective_cv(view), 5),
                      "cv_recent": round(fallback.effective_cv(recent), 5), "pick": "write" if writes else "",
                      "holdout": round(hold, 5), "holdout_seen": round(hold_seen, 5),
                      "blend": json.dumps(weights, ensure_ascii=False) if weights else "",
                      "new_levels": json.dumps(extra, ensure_ascii=False) if extra else ""})
    frame = pd.DataFrame(table)
    frame.to_csv(root / "final_table.csv", index=False)
    print(f"\n## {name}: metric {spec.metric}; fallback {fallback_seconds:.0f} s, zoo {zoo_seconds_used:.0f} s "
          f"({len(zoo)} models), seeds {bag_seconds:.0f} s; rows without new levels {seen_share:.1%}")
    print(frame.to_string(index=False))
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=["reg_year5", "cls_groups25"])
    parser.add_argument("--zoo", type=float, default=1800.0, help="окно этапа A (и отдельно этапа B), с")
    parser.add_argument("--random", type=int, default=final.N_RANDOM, help="случайных вариантов после списка")
    parser.add_argument("--no-bag", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for name in args.names:
        measure(name, args.zoo, args.random, not args.no_bag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
