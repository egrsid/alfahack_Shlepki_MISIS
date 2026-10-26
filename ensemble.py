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
- регрессия с повторяющейся сущностью (find_entity у fallback): свои веса для строк, где у сущности есть история, и
  для новых сущностей (segments). У новых ошибка в разы больше (копия «годы 1–4 → год 5»: 55 против 22 RMSE), и общие
  веса подбирались под них — модели на истории сущности (lag) получали ~0, хотя на знакомых зданиях лучшие; в
  public-test знакомых зданий 79% строк. Веса — среднее Caruana по бутстреп-выборкам строк (bags), сильнейшие для
  честной оценки фолда — по остальным фолдам (select="per_fold").
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
SEGMENT_MIN_ROWS = 300  # меньше строк в сегменте «сущность знакома / новая» — веса без сегментов
LAST_MIN_SHARE = 0.1    # fit="last"/"last2": последний(е) период(ы) меньше этой доли строк — поздняя половина периодов
# по умолчанию (замеры на копиях public-регрессии, RMSE по знакомым штатам, tools/blend_lab.py): сегменты сущности +
# среднее весов по 10 бутстреп-выборкам — год 4: 33.48 → 32.26, год 5: 29.73 → 29.64 (fallback 34.49 / 30.08);
# веса по последнему периоду (fit="last") — год 5: 29.22, но год 4: 33.75 (хуже) — не по умолчанию
REGRESSION_BAGS = 10    # bags=None: столько бутстреп-выборок для метрик регрессии, 0 — для классификации (не мерили)


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


def _top(scores: list[float], metric: str, k: int) -> np.ndarray:
    """Номера k сильнейших по оценке (порядок — от лучшего)."""
    sign = 1.0 if metrics.greater_is_better(metric) else -1.0
    return np.argsort(-sign * np.asarray(scores, dtype=float), kind="stable")[:k]


def solve(y: np.ndarray, oofs: list[np.ndarray], rows: np.ndarray, metric: str, *, pool: np.ndarray | None = None,
          max_pool: int = MAX_POOL, bags: int = 0, seed: int = 0) -> np.ndarray:
    """Веса всех oofs по строкам rows: отбор max_pool сильнейших по своей оценке на этих строках (pool — готовый
    отбор), затем Caruana; bags > 1 — среднее весов Caruana по bags бутстреп-выборкам строк (веса меньше
    подгоняются под шум конкретных строк)."""
    if pool is None:
        pool = _top([metrics.score(metric, y[rows], o[rows]) for o in oofs], metric, max_pool)
    sub_y, sub = y[rows], [oofs[i][rows] for i in pool]
    if bags > 1:
        rng = np.random.default_rng(seed)
        found = np.zeros(len(pool))
        for _ in range(bags):
            idx = rng.integers(0, len(sub_y), len(sub_y))
            found += caruana(sub_y[idx], [o[idx] for o in sub], metric)
        found /= bags
    else:
        found = caruana(sub_y, sub, metric)
    out = np.zeros(len(oofs))
    out[pool] = found
    return out


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


def entity_rows(meta: dict, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(строки OOF, у чьей сущности была история, когда строку предсказывали; строки test, чья сущность есть в
    train) по грубому ключу сущности fallback (find_entity) или None (сущности нет).

    Схема «по периодам»: история — та же сущность в более раннем периоде (как у признаков сущности: строка периода g
    видит только периоды < g). Иначе — та же сущность в других фолдах (вложенный OOF)."""
    data = _data(meta, len(folds))
    entity = getattr(data, "entity", None)
    if not entity:
        return None
    level = (entity.get("levels") or [entity])[0]
    key, key_test = np.asarray(level["key_train"]), np.asarray(level["key_test"])
    if len(key) != len(folds):
        return None
    frame = pd.DataFrame({"k": key, "f": folds})
    if data.period_order is not None:
        order = np.asarray(data.period_order, dtype=float)
        known = order > pd.Series(order).groupby(key).transform("min").to_numpy()
    else:
        total = frame.groupby("k")["f"].transform("size").to_numpy()
        own = frame.groupby(["k", "f"])["f"].transform("size").to_numpy()
        known = total > own
    return known, np.isin(key_test, key)


def blend(spec: Any, cands: list[Any], *, decision_seconds: float = 30.0, recent: bool = True,
          new_levels: bool = True, fit: str = "recent", select: str = "per_fold", max_pool: int = MAX_POOL,
          segments: bool = True, bags: int | None = None) -> Any | None:
    """Бленд кандидатов (см. docstring модуля). Параметры отбора весов:
    fit — строки для весов при схеме «по периодам»: "recent" (поздняя половина периодов), "last" / "last2"
    (последний / два последних периода — история сущностей там как у test), "all";
    select — отбор max_pool сильнейших: "global" — один раз по всем фолдам, "per_fold" — для честной оценки фолда f
    по остальным фолдам (отбор по всем фолдам, включая f, завышал честную оценку тем сильнее, чем больше зоопарк);
    segments — регрессия с сущностью: свои веса для строк, где у сущности есть история, и для новых сущностей
    (у новых ошибка в разы больше, общие веса подбираются в основном под них);
    bags — среднее весов Caruana по бутстреп-выборкам строк (0 — один отбор; None — REGRESSION_BAGS для регрессии)."""
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
    if bags is None:
        bags = REGRESSION_BAGS if metrics.get(metric).task == "regression" else 0

    late = recent_rows(meta, len(y)) if recent else None
    data = _data(meta, len(y))
    order = None
    if late is not None and fit in ("last", "last2"):
        order = np.asarray(data.period_order, dtype=float)

    def fit_rows(inside: np.ndarray | None) -> np.ndarray:
        """Строки для весов без строк inside (None — все): поздние периоды / последние периоды вне inside."""
        outside = np.ones(len(y), dtype=bool) if inside is None else ~inside
        if late is None or fit == "all":
            return outside
        rows = outside & late
        if order is not None:
            levels = np.unique(order[outside & np.isfinite(order)])
            last = outside & np.isin(order, levels[-(1 if fit == "last" else 2):])
            if last.sum() >= max(MIN_FIT_ROWS, LAST_MIN_SHARE * outside.sum()):
                rows = last
        return rows if rows.sum() >= MIN_FIT_ROWS else outside

    fit_all = fit_rows(None)
    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen: set[str] = set()
    for cand in cands:
        pair = _aligned(cand, ref)
        if pair is None or cand.name in seen:
            if pair is None:
                log.info("ensemble: %s не совпадает с %s по строкам/классам — пропущен", cand.name, ref.name)
            continue
        seen.add(cand.name)
        entries.append((cand.name, pair[0], pair[1]))
    if len(entries) < 2:
        return None
    names = [e[0] for e in entries]
    oofs, tests = [e[1] for e in entries], [e[2] for e in entries]
    # сильнейшие по своему OOF на строках fit — для итоговых весов (и для честной оценки при select="global")
    pool = _top([metrics.score(metric, y[fit_all], o[fit_all]) for o in oofs], metric, max_pool)

    # сегменты строк: (маска OOF, маска test); без сегментов — один «сегмент» из всех строк
    parts = [(np.ones(len(y), dtype=bool), np.ones(len(tests[0]), dtype=bool))]
    seg = entity_rows(meta, folds) if segments and metrics.get(metric).task == "regression" else None
    if seg is not None:
        known, known_test = seg
        if min((fit_all & known).sum(), (fit_all & ~known).sum()) >= SEGMENT_MIN_ROWS:
            parts = [(known, known_test), (~known, ~known_test)]
        else:
            seg = None

    def weights_for(rows: np.ndarray, honest_fold: bool, salt: int) -> np.ndarray:
        chosen = None if (honest_fold and select == "per_fold") else pool
        return solve(y, oofs, rows, metric, pool=chosen, max_pool=max_pool, bags=bags, seed=salt)

    seg_weights = [weights_for(fit_all & part, False, 1000 + i) for i, (part, _) in enumerate(parts)]
    test_share = [float(np.mean(part_test)) for _, part_test in parts]
    weights = sum(s * w for s, w in zip(test_share, seg_weights)) / max(sum(test_share), 1e-12)
    if (weights > 0).sum() < 2:
        log.info("ensemble: бленд не нужен — лучше всех одна модель %s", names[int(np.argmax(weights))])
        return None
    # честная оценка: веса для фолда f подбираются на остальных фолдах (на их поздних строках, если fit), оценка — на f
    fold_ids = np.unique(folds)
    honest = np.zeros_like(oofs[0])
    for f in fold_ids:
        inside = folds == f
        base = fit_rows(inside)
        for i, (part, _) in enumerate(parts):
            if not (inside & part).any():
                continue
            rows = base & part
            if rows.sum() < MIN_FIT_ROWS:
                rows = ~inside & part
            if rows.sum() < MIN_FIT_ROWS:
                rows = base
            honest[inside & part] = _mix(weights_for(rows, True, 10 * int(f) + i), oofs, inside & part)
    fold_scores = [float(metrics.score(metric, y[folds == f], honest[folds == f])) for f in fold_ids]
    oof = np.zeros_like(oofs[0])
    test_pred = np.zeros_like(np.asarray(tests[0], dtype=float))
    for w, (part, part_test) in zip(seg_weights, parts):
        oof[part] = _mix(w, oofs, part)
        if part_test.any():
            test_pred[part_test] = _mix(w, tests, part_test)
    used = {n: round(float(w), 4) for n, w in zip(names, weights) if w > 0}
    out_meta = {k: v for k, v in meta.items() if k not in ("decision", "fold_scores_post", "label_shift")}
    out_meta.update(blend=used, fold_scores=fold_scores, fold_scores_eff=list(fold_scores), decision=None,
                    cv_score_post=float(np.mean(fold_scores)), metric=metric, honest_oof=True,
                    blend_rows=("all" if late is None or fit == "all" else
                                "recent periods" if fit == "recent" else f"{fit} periods"),
                    blend_options={"fit": fit, "select": select, "max_pool": max_pool, "segments": seg is not None,
                                   "bags": bags})
    if seg is not None:
        out_meta["segments"] = {
            name: {"oof_rows": int(part.sum()), "test_rows": int(part_test.sum()),
                   "blend": {n: round(float(v), 4) for n, v in zip(names, w) if v > 0}}
            for name, w, (part, part_test) in zip(("known", "new"), seg_weights, parts)}
        log.info("ensemble: сегменты сущности: знакомые %d/%d строк OOF/test, новые %d/%d",
                 int(parts[0][0].sum()), int(parts[0][1].sum()), int(parts[1][0].sum()), int(parts[1][1].sum()))
    levels = new_level_rows(meta, folds) if new_levels else None
    if levels is not None and levels[1].any() and levels[0].sum() >= STRESS_MIN_ROWS:
        stress, risky = levels
        w_new = solve(y, oofs, stress, metric, pool=pool)
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
