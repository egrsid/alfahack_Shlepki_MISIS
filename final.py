"""Финальный этап (роль P1): «зоопарк» моделей на рецепте лучшего кандидата fallback — члены финального бленда.

    train(spec, best, until, threads=None) -> list[Candidate]                    # этап A: новые модели
    bag(spec, best, cands, weights, until, threads=None) -> list[Candidate]      # этап B: сиды весомых членов
    train(spec, best, until, listed=False, tag="b", seed_offset=1)               # этап B: ещё случайные варианты

Зачем. Fallback (P3) за 5–25 минут обучает 4–8 моделей, агент чаще всего ничего не принимает, и до конца 3 часов
остаётся 1.5–2 часа (живой прогон public-регрессии закончился за 36 минут). Здесь в это время учатся ДРУГИЕ модели
на ТЕХ ЖЕ матрицах, фолдах, признаках сущности и исключениях, что у рецепта (fallback.cv_model): лог-таргет,
медленный шаг, другая глубина и регуляризация, веса классов. Каждая — кандидат с честной OOF-оценкой (одиночная
модель на фолдах fallback); решает финальный бленд (ensemble.blend → solution.pick): член со слабым собственным CV
получает вес, если ошибается иначе, чем остальные.

Замер (копия public-регрессии «годы 1–4 → год 5», RMSE без нового в test штата): fallback 30.31, Caruana по моделям
fallback 30.08, с членами xgb_log / lgbm_slow — 30.05. Модели: xgb_log 30.28, lgbm_log_slow 30.42, lgbm_slow 30.50,
cat_log 31.51 против 30.66 у LightGBM fallback; 11–28 с на модель.

lag-признаки (test — новый период и есть сущность fallback): последний известный таргет сущности в более раннем
периоде и давность (lag_columns). Та же копия: XGBoost 30.66 → 30.24, LightGBM с сущностью 30.66 → 30.52; по CV всех
фолдов хуже (в году 1 истории нет), на последнем фолде — лучше (31.6 против 32.3). Строка периода g видит только
периоды < g, test — весь train; таргеты валидационного периода бывают признаком у ПОЗДНИХ строк обучения, но не у
самих валидационных строк (их таргет модель не видит).

Ничего не пишет на диск: кандидатов записывает solution.py, если бленд с ними честно лучше уже записанного.
Сбой одной модели или нехватка времени пропускает её, а не останавливает этап.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

import fallback
import metrics

log = logging.getLogger("final")

SLOW_LR = 0.4           # «медленный» член: шаг × это, деревьев ÷ это (ранняя остановка выберет сама)
BAG_SEEDS = 3           # этап B: сид fallback + 2 новых, как bag3 у fallback
BAG_MIN_WEIGHT = 0.05   # этап B: сиды только для членов с таким весом в бленде и больше
N_RANDOM = 8            # после списка: столько случайных вариантов LightGBM / XGBoost, пока есть время
MIN_SECONDS = 20.0      # меньше осталось — новую модель не начинаем
CAP_FACTOR = 3.0        # потолок времени модели: CAP_FACTOR × оценка (оценки по прогонам fallback бывают занижены)
CLASS_WEIGHTED = 3.0    # дисбаланс max/min размера класса, с которого члены классификации — с весами классов


@dataclass(frozen=True)
class Member:
    """Новая модель на рецепте: модель fallback, переопределения параметров (поверх fallback._params), таргет,
    веса строк (agent_eval.row_weights) и множитель оценки времени (медленный шаг — больше деревьев)."""
    name: str
    model: str
    extra: dict = field(default_factory=dict)
    transform: str | None = None
    weights: dict | None = None
    cost: float = 1.0
    lag: bool = False      # матрицы рецепта + lag-признаки сущности (lag_data)
    exclude: tuple[str, ...] | None = None  # None — исключения опорного прогона; модели fallback — свои
    entity: bool | None = None              # None — признаки сущности как у опорного прогона


def context(best: Any) -> Any:
    """Рецепт лучшего кандидата fallback (FallbackContext) или None."""
    return (getattr(best, "meta", None) or {}).get("context")


def reference(ctx: Any) -> Any:
    """Опорный прогон рецепта: самый весомый член лучшего кандидата (его признаки сущности, исключения, таргет);
    константа — лучший по CV прогон."""
    runs = {r.name: r for r in ctx.runs}
    weighted = [(w, runs[n]) for n, w in (ctx.members or {}).items() if n in runs and w > 0]
    if weighted:
        return max(weighted, key=lambda wr: wr[0])[1]
    sign = 1.0 if metrics.greater_is_better(ctx.data.metric) else -1.0
    return max(ctx.runs, key=lambda r: sign * r.cv)


def lag_columns(data: Any) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """lag-признаки сущности для train и test: по каждому уровню ключа (data.entity) — таргет сущности в последнем
    более раннем периоде (среднее по строкам того периода) и давность в периодах. None — нет периодов или сущности."""
    if data.entity is None or data.period_order is None:
        return None
    order = np.asarray(data.period_order, dtype=float)
    test_period = float(np.nanmax(order)) + 1.0
    train_cols, test_cols = {}, {}
    for i, level in enumerate(data.entity.get("levels") or [data.entity]):
        key, key_test = np.asarray(level["key_train"]), np.asarray(level["key_test"])
        per_key = (pd.DataFrame({"k": key, "y": np.asarray(data.y, dtype=float), "p": order})
                   .groupby(["k", "p"], sort=False)["y"].mean().reset_index().sort_values("p"))
        last, gap = np.full(len(key), np.nan), np.full(len(key), np.nan)
        for g in np.unique(order[np.isfinite(order)]):
            rows = np.flatnonzero(order == g)
            src = per_key[per_key["p"] < g]
            if src.empty:
                continue
            latest = src.groupby("k", sort=False).last()
            keys = pd.Series(key[rows])
            last[rows], gap[rows] = keys.map(latest["y"]).to_numpy(), g - keys.map(latest["p"]).to_numpy()
        latest = per_key.groupby("k", sort=False).last()
        keys_test = pd.Series(key_test)
        train_cols[f"lag{i + 1}__last"], train_cols[f"lag{i + 1}__gap"] = last, gap
        test_cols[f"lag{i + 1}__last"] = keys_test.map(latest["y"]).to_numpy()
        test_cols[f"lag{i + 1}__gap"] = test_period - keys_test.map(latest["p"]).to_numpy()
    return pd.DataFrame(train_cols).astype("float32"), pd.DataFrame(test_cols).astype("float32")


def lag_data(ctx: Any) -> Any | None:
    """Копия данных рецепта, где к матрицам LightGBM/XGBoost и CatBoost добавлены lag-признаки (кэш в ctx)."""
    cached = getattr(ctx, "_final_lag_data", "missing")
    if cached != "missing":
        return cached
    data, out = ctx.data, None
    columns = lag_columns(data)
    if columns is not None:
        extra_tr, extra_te = columns
        out = copy.copy(data)
        out.frames = {}
        for model in ("lightgbm", "catboost"):
            x, x_test, cats = data.frame(model)
            out.frames[model] = (pd.concat([x.reset_index(drop=True), extra_tr], axis=1),
                                 pd.concat([x_test.reset_index(drop=True), extra_te], axis=1), cats)
    ctx._final_lag_data = out
    return out


def _imbalance(y: np.ndarray) -> float:
    counts = np.bincount(np.asarray(y, dtype=int))
    counts = counts[counts > 0]
    return float(counts.max() / counts.min()) if len(counts) else 1.0


def members(ctx: Any, ref: Any) -> list[Member]:
    """Список новых моделей по типу задачи — по убыванию «польза / цена» (замеры — в docstring модуля)."""
    data = ctx.data
    slow = {"learning_rate": fallback._lr(len(data.y)) * SLOW_LR,
            "n_estimators": int(ctx.config.max_rounds / SLOW_LR)}
    if data.task == "regression":
        same = ref.transform
        # «другой» таргет: лог, если рецепт без него и таргет неотрицательный; если рецепт уже с логом — без лога
        alt = None if same == "log1p" else ("log1p" if float(np.min(data.y)) >= 0 else "none")
        alt_tf = None if alt in (None, "none") else alt
        tag = "_log" if alt_tf == "log1p" else "_raw"
        out = []
        if alt != "none":
            out.append(Member(f"xgb{tag}", "xgboost", {}, alt_tf, cost=1.5))
        out.append(Member("lgbm_slow", "lightgbm", slow, same, cost=1.0 / SLOW_LR))
        if alt != "none":
            out += [Member(f"lgbm{tag}_slow", "lightgbm", slow, alt_tf, cost=1.0 / SLOW_LR),
                    Member(f"cat{tag}", "catboost", {}, alt_tf, cost=2.5)]
        out += [Member("lgbm_et", "lightgbm", {"extra_trees": True}, same),
                Member("xgb_d5", "xgboost", {"max_depth": 5}, same, cost=1.5)]
        if data.entity is not None and data.period_order is not None:  # test — новый период: lag-признаки
            out[1:1] = [Member("xgb_lag", "xgboost", {}, same, lag=True, cost=1.5),
                        Member("lgbm_lag", "lightgbm", {}, same, lag=True)]
            if alt != "none":
                out.append(Member(f"xgb{tag}_lag", "xgboost", {}, alt_tf, lag=True, cost=1.5))
        return out
    regular = {"min_child_samples": 100, "colsample_bytree": 0.5, "reg_lambda": 5.0}
    if metrics.get(data.metric).tunable and _imbalance(data.y) >= CLASS_WEIGHTED:
        # метрика по меткам при дисбалансе (macro F1 и т. п.): веса классов у всех членов, порог — у бленда
        return [Member("cat_sqrtbal", "catboost", {"auto_class_weights": "SqrtBalanced"}, cost=2.5),
                Member("xgb_classw", "xgboost", {}, weights={"scheme": "class", "power": 0.5}, cost=1.5),
                Member("lgbm_bal_reg", "lightgbm", {"class_weight": "balanced", **regular}),
                Member("lgbm_bal_slow", "lightgbm", {"class_weight": "balanced", **slow}, cost=1.0 / SLOW_LR)]
    return [Member("lgbm_slow", "lightgbm", slow, cost=1.0 / SLOW_LR),
            Member("lgbm_reg", "lightgbm", regular),
            Member("xgb_d5", "xgboost", {"max_depth": 5}, cost=1.5),
            Member("cat_d8", "catboost", {"depth": 8}, cost=4.0)]


def random_members(ctx: Any, ref: Any, n: int = N_RANDOM, seed: int = 0, tag: str = "r") -> list[Member]:
    """Случайные варианты LightGBM / XGBoost (листья, мин. размер листа, доли строк/колонок, L2) — для разнообразия
    бленда, когда список уже обучен. Сид фиксирован: один и тот же запуск даёт те же варианты."""
    data = ctx.data
    rng = np.random.default_rng(seed)
    weighted = data.task != "regression" and metrics.get(data.metric).tunable and _imbalance(data.y) >= CLASS_WEIGHTED
    transforms = [ref.transform]
    if data.task == "regression" and ref.transform != "log1p" and float(np.min(data.y)) >= 0:
        transforms.append("log1p")
    out = []
    for i in range(n):
        transform = transforms[int(rng.integers(len(transforms)))]
        common = {"subsample": float(rng.choice([0.6, 0.8, 1.0])), "colsample_bytree": float(rng.choice([0.4, 0.6, 0.8])),
                  "reg_lambda": float(rng.choice([0.0, 1.0, 5.0, 20.0]))}
        if i % 2 == 0:
            extra = {**common, "num_leaves": int(rng.choice([15, 31, 63, 127])),
                     "min_child_samples": int(rng.choice([5, 20, 50, 100]))}
            if weighted:
                extra["class_weight"] = "balanced"
            out.append(Member(f"lgbm_{tag}{i}", "lightgbm", extra, transform))
        else:
            extra = {**common, "max_depth": int(rng.choice([4, 6, 8, 10])),
                     "min_child_weight": float(rng.choice([1.0, 5.0, 20.0]))}
            out.append(Member(f"xgb_{tag}{i}", "xgboost", extra, transform,
                              {"scheme": "class", "power": 0.5} if weighted else None, cost=1.5))
    return out


def _unit(ctx: Any) -> float:
    """Секунды первого LightGBM fallback (масштаб времени задачи); нет LightGBM — медиана прогонов."""
    lgbm = [r.seconds for r in ctx.runs if r.model == "lightgbm"]
    return float(lgbm[0] if lgbm else np.median([r.seconds for r in ctx.runs]))


def estimate(ctx: Any, member: Member, unit: float) -> float:
    """Ожидаемые секунды модели. member.cost — цена в единицах первого LightGBM (unit): если fallback учил ту же
    модель — её медианное время × (cost / обычная цена модели), иначе unit × cost."""
    same = [r.seconds for r in ctx.runs if r.model == member.model]
    if same:
        return float(np.median(same)) * member.cost / fallback._COST.get(member.model, 1.0)
    return unit * member.cost


def fit(ctx: Any, ref: Any, member: Member, stop_at: float, threads: int) -> Any | None:
    """fallback.Run модели на рецепте (те же фолды, признаки сущности и исключения, что у опорного прогона) или
    None (упала, нет времени, NaN/inf в прогнозах)."""
    data = lag_data(ctx) if member.lag else ctx.data
    if data is None:
        log.info("final: %s пропущен: нет lag-признаков (нет периодов или сущности)", member.name)
        return None
    try:
        weights = None
        if member.weights:
            import agent_eval
            weights = agent_eval.row_weights(data, member.weights)
        exclude = ref.exclude if member.exclude is None else member.exclude
        entity = ref.entity if member.entity is None else member.entity
        run = fallback.cv_model(data, member.name, member.model, ctx.config, stop_at, transform=member.transform,
                                extra=dict(member.extra), threads=threads, exclude=tuple(exclude),
                                entity=bool(entity), sample_weight=weights)
    except Exception as error:  # noqa: BLE001 — сбой одной модели не роняет этап
        log.warning("final: %s упал: %s: %s", member.name, type(error).__name__, str(error)[:300])
        return None
    if not (np.isfinite(run.oof).all() and np.isfinite(run.test_pred).all()):
        log.warning("final: %s — NaN/inf в прогнозах, пропущен", member.name)
        return None
    log.info("final: %s: cv=%.6g ± %.3g (%s, %.0f с, итераций ~%d)", member.name, run.cv,
             float(np.std(run.fold_scores)), data.metric, run.seconds, int(np.mean(run.iterations or [0])))
    return run


def candidate(data: Any, name: str, oof: np.ndarray, test_pred: np.ndarray, member: Member, seconds: float,
              seeds: int = 1) -> Any:
    """Кандидат одной модели в форме fallback.pool: OOF на фолдах fallback, оценки фолдов по сырым прогнозам."""
    scores = [float(s) for s in fallback._fold_scores(data, oof)]
    return fallback.Candidate(name, float(np.mean(scores)), float(np.std(scores)), oof, test_pred,
                              {"fold_scores": scores, "fold_scores_eff": list(scores), "folds": data.folds,
                               "train_rows": data.rows, "classes": data.classes, "metric": data.metric,
                               "model": member.model, "cv_score_post": float(np.mean(scores)),
                               "member": member, "seconds": float(seconds), "seeds": seeds})


def train(spec: Any, best: Any, until: float, threads: int | None = None, n_random: int = N_RANDOM,
          listed: bool = True, tag: str = "r", seed_offset: int = 0) -> list:
    """Модели members() (listed=False — без них), затем random_members() с сидом рецепта + seed_offset и меткой tag
    в именах, пока до until (time.monotonic()) хватает времени. Модель, которая по оценке не успевает, пропускается
    (следующая может быть дешевле)."""
    ctx = context(best)
    if ctx is None or not ctx.runs:
        log.info("final: у fallback нет рецепта — зоопарк пропущен")
        return []
    ref = reference(ctx)
    threads = threads or fallback.n_threads()
    unit = _unit(ctx)
    plan = [*(members(ctx, ref) if listed else []),
            *random_members(ctx, ref, n_random, seed=int(ctx.config.seed) + seed_offset, tag=tag)]
    log.info("final: опорный прогон %s (сущность %s, исключено %d, таргет %s); моделей в плане %d, окно %.0f с",
             ref.name, ref.entity, len(ref.exclude), ref.transform or "как есть", len(plan), until - time.monotonic())
    found = []
    for member in plan:
        left = until - time.monotonic()
        need = estimate(ctx, member, unit)
        if left < MIN_SECONDS or need > left:
            log.info("final: %s пропущен: оценка %.0f с, осталось %.0f с", member.name, need, left)
            continue
        run = fit(ctx, ref, member, time.monotonic() + min(left, max(MIN_SECONDS, CAP_FACTOR * need)), threads)
        if run is not None:
            found.append(candidate(ctx.data, f"final_{member.name}", run.oof, run.test_pred, member, run.seconds))
    return found


def pool_members(best: Any) -> list:
    """Кандидаты моделей fallback (fallback.pool) с рецептом их прогона в meta["member"] — чтобы bag досчитывал сиды
    и им: в бленде у них бывает большой вес (public-мультикласс: случайный лес 0.4, LightGBM balanced 0.24), а обучены
    они одним сидом (сидами fallback усредняет только свой лучший кандидат)."""
    ctx = context(best)
    if ctx is None:
        return []
    runs = {f"fallback_{r.name}": r for r in ctx.runs}
    out = []
    for cand in fallback.pool(best):
        run = runs.get(cand.name)
        if run is None or run.model not in fallback._COST:
            continue
        member = Member(run.name, run.model, dict(run.extra), run.transform, exclude=tuple(run.exclude),
                        entity=bool(run.entity))
        cand.meta.update(member=member, seconds=float(run.seconds), seeds=1)
        out.append(cand)
    return out


def bag(spec: Any, best: Any, cands: list, weights: dict, until: float, threads: int | None = None,
        seeds: int = BAG_SEEDS, min_weight: float = BAG_MIN_WEIGHT) -> list:
    """Этап B: члены зоопарка с весом ≥ min_weight в бленде (weights: имя → вес) — ещё seeds − 1 сидов, среднее
    OOF и test. Возвращает только полностью досчитанные (имя `<член>_bag<seeds>`, meta["bagged_from"])."""
    ctx = context(best)
    if ctx is None or seeds <= 1:
        return []
    ref = reference(ctx)
    threads = threads or fallback.n_threads()
    out = []
    for cand in sorted(cands, key=lambda c: -float(weights.get(c.name, 0.0))):
        member = (cand.meta or {}).get("member")
        if member is None or float(weights.get(cand.name, 0.0)) < min_weight or cand.meta.get("seeds", 1) > 1:
            continue
        need = float(cand.meta.get("seconds", 0.0)) * (seeds - 1)
        if need > until - time.monotonic():
            log.info("final: сиды %s пропущены: нужно ~%.0f с, осталось %.0f с", cand.name, need,
                     until - time.monotonic())
            continue
        oofs, tests = [cand.oof], [cand.test_pred]
        for k in range(1, seeds):
            again = replace(member, name=f"{member.name}_seed{k}",
                            extra={**member.extra, **fallback.seed_params(member.model, ctx.config.seed + k)})
            left = until - time.monotonic()
            run = fit(ctx, ref, again, time.monotonic() + min(left, CAP_FACTOR * float(cand.meta["seconds"])), threads)
            if run is None:
                break
            oofs.append(run.oof)
            tests.append(run.test_pred)
        if len(oofs) < seeds:
            continue
        bagged = candidate(ctx.data, f"{cand.name}_bag{seeds}", sum(oofs) / seeds, sum(tests) / seeds, member,
                           float(cand.meta["seconds"]), seeds)
        bagged.meta["bagged_from"] = cand.name
        log.info("final: %s: cv=%.6g (один сид %.6g)", bagged.name, bagged.cv_score, cand.cv_score)
        out.append(bagged)
    return out
