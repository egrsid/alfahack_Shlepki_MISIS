"""Финальный бленд кандидатов fallback и агента (роль P1; написано P4 — владелец может переписать).

    blend(spec, cands) -> Candidate | None

cands[0] — опорный кандидат (лучший у fallback): его фолды (`meta["folds"]`), строки train (`meta["train_rows"]`),
классы и метрика задают оценку. Остальные (модели fallback — fallback.pool, AutoML, агент) берутся, только если их OOF покрывает те же строки,
а классы совпадают — тогда столбцы вероятностей переставляются в порядок опорного. Веса — жадный отбор с
возвращением (Caruana) по метрике задачи на OOF, как в fallback.blend. Оценка честная: для каждого фолда веса
подбираются на остальных фолдах (fold_scores/fold_scores_eff — по этим «чужим» весам), итоговые веса для test — на
всех OOF; порог/веса классов — тоже по чужим фолдам (metrics.crossfit_decision). Решение «писать ли бленд» —
у вызывающего по парным фолдам с порогом шума (fallback.clearly_better). None — бленд не нужен или невозможен.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import metrics
import submission

log = logging.getLogger("ensemble")

MAX_STEPS = 30
MAX_POOL = 16      # кандидатов в отборе (сильнейшие по своему OOF)


def _truth(spec: Any, ref: Any) -> np.ndarray:
    """Таргет train в строках и кодировке опорного кандидата."""
    meta = ref.meta
    target = submission.targets_of(spec)[0]
    y = pd.read_csv(submission.task_file(spec, "train.csv"), usecols=[target])[target]
    rows = meta.get("train_rows")
    if rows is not None:
        y = y.iloc[np.asarray(rows, dtype=int)]
    else:
        y = y[y.notna()]
    classes = meta.get("classes")
    return y.to_numpy(dtype=float) if classes is None else metrics.encode(y, classes)


def _aligned(cand: Any, ref: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """(oof, test_pred) кандидата в форме опорного или None (другие строки, классы или форма)."""
    oof, test_pred = np.asarray(cand.oof, dtype=float), np.asarray(cand.test_pred, dtype=float)
    ref_oof, ref_test = np.asarray(ref.oof), np.asarray(ref.test_pred)
    if len(oof) != len(ref_oof) or len(test_pred) != len(ref_test):
        return None
    ref_classes = ref.meta.get("classes")
    if ref_classes is None:  # регрессия
        return (oof, test_pred) if oof.shape == ref_oof.shape and test_pred.shape == ref_test.shape else None
    own = [metrics.label_str(c) for c in ((cand.meta or {}).get("classes") or ref_classes)]
    order = [metrics.label_str(c) for c in ref_classes]
    if sorted(own) != sorted(order) or oof.ndim != 2 or oof.shape[1] != len(own):
        return None
    index = [own.index(c) for c in order]
    return oof[:, index], test_pred[:, index]


def caruana(y: np.ndarray, oofs: list[np.ndarray], metric: str, max_steps: int = MAX_STEPS) -> np.ndarray:
    """Веса жадного отбора с возвращением (Caruana) по метрике задачи на переданных строках.

    Все max_steps шагов выполняются всегда, итог — лучший набор за все шаги: так достигаются веса вроде 0.8/0.2
    со слабой, но непохожей моделью (смесь 50/50 с ней хуже — остановка на первом «нет улучшения» её теряла).
    """
    counts = np.zeros(len(oofs))
    current, best, best_counts = None, None, None
    for step in range(max_steps):
        choice, choice_score = None, None
        for i, oof in enumerate(oofs):
            trial = oof if current is None else (current * step + oof) / (step + 1)
            value = metrics.score(metric, y, trial)
            if choice_score is None or metrics.is_better(metric, value, choice_score):
                choice, choice_score = i, value
        counts[choice] += 1
        current = oofs[choice] if current is None else (current * step + oofs[choice]) / (step + 1)
        if best is None or metrics.is_better(metric, choice_score, best):
            best, best_counts = choice_score, counts.copy()
    return best_counts / best_counts.sum()


def _mix(weights: np.ndarray, arrays: list[np.ndarray], rows: Any = slice(None)) -> np.ndarray:
    return sum(w * a[rows] for w, a in zip(weights, arrays) if w > 0)


def blend(spec: Any, cands: list[Any], *, decision_seconds: float = 30.0) -> Any | None:
    if len(cands) < 2:
        return None
    ref = cands[0]
    meta = getattr(ref, "meta", None) or {}
    folds = meta.get("folds")
    if folds is None:
        return None
    folds = np.asarray(folds, dtype=int)
    metric = meta.get("metric") or spec.metric
    y = _truth(spec, ref)
    if len(y) != len(folds):
        log.warning("ensemble: таргет %d строк, фолдов %d — без бленда", len(y), len(folds))
        return None

    entries: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    seen: set[str] = set()
    for cand in cands:
        pair = _aligned(cand, ref)
        if pair is None or cand.name in seen:
            if pair is None:
                log.info("ensemble: %s не совпадает с %s по строкам/классам — пропущен", cand.name, ref.name)
            continue
        seen.add(cand.name)
        entries.append((cand.name, pair[0], pair[1], float(metrics.score(metric, y, pair[0]))))
    if len(entries) < 2:
        return None
    sign = 1.0 if metrics.greater_is_better(metric) else -1.0
    entries = sorted(entries, key=lambda e: -sign * e[3])[:MAX_POOL]  # сильнейшие по своему OOF
    names = [e[0] for e in entries]
    oofs, tests = [e[1] for e in entries], [e[2] for e in entries]

    weights = caruana(y, oofs, metric)
    if (weights > 0).sum() < 2:
        log.info("ensemble: бленд не нужен — лучше всех одна модель %s", names[int(np.argmax(weights))])
        return None
    # честная оценка: веса для фолда f подбираются на остальных фолдах, оценка — на f
    fold_ids = np.unique(folds)
    honest = np.zeros_like(oofs[0])
    for f in fold_ids:
        inside = folds == f
        w_f = caruana(y[~inside], [o[~inside] for o in oofs], metric)
        honest[inside] = _mix(w_f, oofs, inside)
    fold_scores = [float(metrics.score(metric, y[folds == f], honest[folds == f])) for f in fold_ids]
    oof, test_pred = _mix(weights, oofs), _mix(weights, tests)
    used = {n: round(float(w), 4) for n, w in zip(names, weights) if w > 0}
    out_meta = {k: v for k, v in meta.items() if k not in ("decision", "fold_scores_post", "label_shift")}
    out_meta.update(blend=used, fold_scores=fold_scores, fold_scores_eff=list(fold_scores), decision=None,
                    cv_score_post=float(np.mean(fold_scores)), metric=metric, honest_oof=True)
    if meta.get("classes") is not None and metrics.get(metric).tunable and decision_seconds > 1:
        try:
            post, decision = metrics.crossfit_decision(metric, y, honest, folds, time_limit=decision_seconds)
            if decision is not None and metrics.is_better(metric, float(np.mean(post)), float(np.mean(fold_scores))):
                out_meta.update(decision=decision, cv_score_post=float(np.mean(post)),
                                fold_scores_post=[float(s) for s in post], fold_scores_eff=[float(s) for s in post])
        except Exception as error:  # noqa: BLE001 — бленд без порога лучше, чем никакого
            log.warning("ensemble: порог/веса классов не подобраны: %s", error)
    log.info("ensemble: бленд %s: честная cv=%.6g (с порогом/весами %.6g)", used, float(np.mean(fold_scores)),
             out_meta["cv_score_post"])
    return type(ref)(name="ensemble", cv_score=float(np.mean(fold_scores)), cv_std=float(np.std(fold_scores)),
                     oof=oof, test_pred=test_pred, meta=out_meta)
