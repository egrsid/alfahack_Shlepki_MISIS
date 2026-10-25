"""Финальный бленд кандидатов fallback и агента (роль P1; написано P4 — владелец может переписать).

    blend(spec, cands) -> Candidate | None

cands[0] — опорный кандидат (лучший у fallback): его фолды (`meta["folds"]`), строки train (`meta["train_rows"]`),
классы и метрика задают оценку. Остальные (модели fallback — fallback.pool, AutoML, агент) берутся, только если их OOF покрывает те же строки,
а классы совпадают — тогда столбцы вероятностей переставляются в порядок опорного. Веса — жадный отбор с
возвращением (Caruana) по метрике задачи на OOF, как в fallback.blend. Оценка честная: для каждого фолда веса
подбираются на остальных фолдах (fold_scores/fold_scores_eff — по этим «чужим» весам), итоговые веса для test — на
всех OOF; порог/веса классов — тоже по чужим фолдам (metrics.crossfit_decision). Решение «писать ли бленд» —
у вызывающего по парным фолдам с порогом шума (fallback.clearly_better). None — бленд не нужен или невозможен.

Test похож не на все строки train одинаково (recent, new_levels — можно выключить):
- схема CV «по периодам» (test — новый период): веса подбираются по OOF поздней половины периодов (честно: для
  фолда f — по поздним строкам остальных фолдов). Копия public-регрессии «годы 1–4 → год 5»: бленд с зоопарком
  30.05 по всем строкам → 29.83 по двум последним годам (RMSE без нового штата);
- строки test с уровнем категории, которого нет во всём train, получают отдельный бленд, подобранный на
  «стресс-строках» OOF — тех, чей уровень не встречался в train их фолда (модель фолда видела его так же впервые).
  На новых уровнях модели расходятся в разы: та же копия, новый в test штат — XGBoost 53 RMSE против ~40 у LightGBM
  с лог-таргетом и RandomForest, а в бленде у XGBoost вес 0.7. Оценка CV бленда от этого не меняется (на OOF таких
  строк нет), меняется только test_pred этих строк; meta["new_levels"] — сколько строк и с какими весами.
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
MIN_FIT_ROWS = 50  # меньше поздних строк вне фолда — веса фолда по всем его остальным строкам
STRESS_MIN_ROWS = 100  # меньше стресс-строк OOF — отдельного бленда для новых уровней нет


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


def _data(meta: dict, n_rows: int) -> Any:
    """Данные fallback (TaskData) опорного кандидата, если они про те же строки OOF; иначе None."""
    data = getattr(meta.get("context"), "data", None)
    return data if data is not None and len(data.y) == n_rows else None


def recent_rows(meta: dict, n_rows: int) -> np.ndarray | None:
    """Строки OOF поздней половины периодов (схема CV «по периодам», уровней ≥ 3); None — все строки."""
    data = _data(meta, n_rows)
    if data is None or (getattr(data, "cv_scheme", None) or {}).get("kind") != "period" or data.period_order is None:
        return None
    order = np.asarray(data.period_order, dtype=float)
    levels = np.unique(order[np.isfinite(order)])
    if len(levels) < 3:
        return None
    return order >= levels[len(levels) // 2]


def recent_folds(meta: dict, n_rows: int) -> list[int] | None:
    """Номера фолдов (в порядке fold_scores), где большинство строк — поздняя половина периодов; меньше двух — None
    (парное сравнение по одному фолду — шум)."""
    fit = recent_rows(meta, n_rows)
    folds = meta.get("folds")
    if fit is None or folds is None:
        return None
    folds = np.asarray(folds, dtype=int)
    picked = [i for i, f in enumerate(np.unique(folds)) if fit[folds == f].mean() >= 0.5]
    return picked if len(picked) >= 2 else None


def new_level_rows(meta: dict, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(стресс-строки OOF: уровень категории встречается только в своём фолде — модель фолда его не видела;
    строки test с уровнем, которого нет во всём train) по категориальным колонкам fallback или None."""
    data = _data(meta, len(folds))
    prep = getattr(data, "prep", None)
    if prep is None or len(prep.X) != len(folds):
        return None
    stress, risky = np.zeros(len(folds), dtype=bool), np.zeros(len(prep.X_test), dtype=bool)
    for col in prep.cat_cols:
        if col not in prep.X.columns or col not in prep.X_test.columns:
            continue
        train, test = prep.X[col].astype(object), prep.X_test[col].astype(object)
        risky |= test.notna().to_numpy() & ~test.isin(set(train.dropna())).to_numpy()
        codes = pd.Series(pd.factorize(train)[0])          # −1 — пропуск
        table = pd.crosstab(codes, folds)                  # уровень × фолд
        single = table.index[(table > 0).sum(axis=1) == 1]
        single = single[single >= 0]
        stress |= codes.isin(set(single)).to_numpy()
    return stress, risky


def blend(spec: Any, cands: list[Any], *, decision_seconds: float = 30.0, recent: bool = True,
          new_levels: bool = True) -> Any | None:
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

    fit = recent_rows(meta, len(y)) if recent else None
    fit_all = np.ones(len(y), dtype=bool) if fit is None else fit
    entries: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    seen: set[str] = set()
    for cand in cands:
        pair = _aligned(cand, ref)
        if pair is None or cand.name in seen:
            if pair is None:
                log.info("ensemble: %s не совпадает с %s по строкам/классам — пропущен", cand.name, ref.name)
            continue
        seen.add(cand.name)
        entries.append((cand.name, pair[0], pair[1], float(metrics.score(metric, y[fit_all], pair[0][fit_all]))))
    if len(entries) < 2:
        return None
    sign = 1.0 if metrics.greater_is_better(metric) else -1.0
    entries = sorted(entries, key=lambda e: -sign * e[3])[:MAX_POOL]  # сильнейшие по своему OOF (на строках fit)
    names = [e[0] for e in entries]
    oofs, tests = [e[1] for e in entries], [e[2] for e in entries]

    weights = caruana(y[fit_all], [o[fit_all] for o in oofs], metric)
    if (weights > 0).sum() < 2:
        log.info("ensemble: бленд не нужен — лучше всех одна модель %s", names[int(np.argmax(weights))])
        return None
    # честная оценка: веса для фолда f подбираются на остальных фолдах (на их поздних строках, если fit), оценка — на f
    fold_ids = np.unique(folds)
    honest = np.zeros_like(oofs[0])
    for f in fold_ids:
        inside = folds == f
        rows = ~inside & fit_all
        if rows.sum() < MIN_FIT_ROWS:
            rows = ~inside
        w_f = caruana(y[rows], [o[rows] for o in oofs], metric)
        honest[inside] = _mix(w_f, oofs, inside)
    fold_scores = [float(metrics.score(metric, y[folds == f], honest[folds == f])) for f in fold_ids]
    oof, test_pred = _mix(weights, oofs), _mix(weights, tests)
    used = {n: round(float(w), 4) for n, w in zip(names, weights) if w > 0}
    out_meta = {k: v for k, v in meta.items() if k not in ("decision", "fold_scores_post", "label_shift")}
    out_meta.update(blend=used, fold_scores=fold_scores, fold_scores_eff=list(fold_scores), decision=None,
                    cv_score_post=float(np.mean(fold_scores)), metric=metric, honest_oof=True,
                    blend_rows="recent periods" if fit is not None else "all")
    levels = new_level_rows(meta, folds) if new_levels else None
    if levels is not None and levels[1].any() and levels[0].sum() >= STRESS_MIN_ROWS:
        stress, risky = levels
        w_new = caruana(y[stress], [o[stress] for o in oofs], metric)
        test_pred = np.array(test_pred, dtype=float, copy=True)
        test_pred[risky] = _mix(w_new, tests, risky)
        out_meta["new_levels"] = {"test_rows": int(risky.sum()), "stress_rows": int(stress.sum()),
                                  "test_index": [int(i) for i in np.flatnonzero(risky)],
                                  "blend": {n: round(float(w), 4) for n, w in zip(names, w_new) if w > 0}}
        log.info("ensemble: %d строк test с новыми уровнями категорий — бленд по %d стресс-строкам OOF: %s",
                 int(risky.sum()), int(stress.sum()), out_meta["new_levels"]["blend"])
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
