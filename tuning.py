"""Подбор гиперпараметров бустингов через optuna (роль P1; написано P4 — владелец может переписать).

Контракт (вызывает agent.py, вид гипотезы tune):

    tune(spec, build_features, cfg, deadline) -> dict гиперпараметров

    build_features — готовые признаки текущего лучшего кандидата агента (код LLM уже исполнен);
    cfg = {"model": "lightgbm|catboost|xgboost", "params": {...стартовые...}, "target_transform": None|"log1p",
           "n_trials": int, "seed": int, "evaluate": evaluator агента (необязательно)};
    deadline — объект с .left(): подбор укладывается в него и возвращает лучшее найденное, а не бросает исключение.

Оценка — тем же evaluator, что у агента (те же фолды и обучение, что у fallback), но только на первых
`max_folds` фолдах ради скорости. Нет evaluator в cfg — LocalEvaluator на честных фолдах fallback.cv_folds.
Сначала оцениваются стартовые параметры; найденное возвращается, только если на тех же фолдах оно лучше,
иначе — стартовые (агент такой «подбор» отклонит без лишней оценки). Итог на всех фолдах агент считает сам.
optuna, sklearn и бустинги импортируются внутри функций: модуль импортируется и без них.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any, Callable

log = logging.getLogger("tuning")

MAX_FOLDS = 3           # фолдов на одну пробу (все фолды — в итоговой оценке агента)
TIME_MARGIN = 1.3       # запас: следующая проба начинается, только если влезает с этим множителем


def _space(trial: Any, model: str) -> dict:
    """Пространство поиска: структура деревьев и регуляризация; число деревьев режет ранняя остановка."""
    if model == "catboost":
        return {"depth": trial.suggest_int("depth", 4, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0)}
    if model == "xgboost":
        return {"max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0)}
    params = {"num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
              "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
              "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
              "subsample": trial.suggest_float("subsample", 0.5, 1.0),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
              "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
              "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)}
    params["subsample_freq"] = 1 if params["subsample"] < 1.0 else 0  # без freq LightGBM subsample не применяет
    return params


def _greater_is_better(spec: Any) -> bool:
    try:
        import metrics
        return bool(metrics.greater_is_better(spec.metric))
    except Exception:  # noqa: BLE001 — нет metrics.py или метрика вне реестра
        return bool(getattr(spec, "greater_is_better", False))


def _left(deadline: Any) -> float:
    try:
        return float(deadline.left()) if deadline is not None else math.inf
    except Exception:  # noqa: BLE001
        return math.inf


def _evaluator(spec: Any, evaluate: Callable | None, max_folds: int) -> Callable:
    """Evaluator агента на первых max_folds фолдах; нет его — LocalEvaluator на фолдах fallback."""
    from agent_eval import LocalEvaluator

    if isinstance(evaluate, LocalEvaluator):
        quick = copy.copy(evaluate)  # тот же кэш данных, фолды и обучение
        quick.max_folds = max_folds
        return quick
    if evaluate is not None:  # evaluator P1: фолды не урезаем
        return evaluate
    try:
        import fallback
        folds, rows, _ = fallback.cv_folds(spec)
        return LocalEvaluator(folds=folds, rows=rows, early_stopping=100, max_folds=max_folds)
    except Exception as error:  # noqa: BLE001
        log.warning("tuning: фолды fallback недоступны (%s) — KFold", error)
        return LocalEvaluator(early_stopping=100, max_folds=max_folds)


def tune(spec: Any, build_features: Callable, cfg: dict, deadline: Any = None) -> dict:
    import optuna

    model = str(cfg.get("model") or "lightgbm")
    start_params = dict(cfg.get("params") or {})
    n_trials = max(1, int(cfg.get("n_trials") or 30))
    evaluate = _evaluator(spec, cfg.get("evaluate"), int(cfg.get("max_folds") or MAX_FOLDS))
    greater = _greater_is_better(spec)
    durations: list[float] = []

    def run(params: dict) -> float:
        began = time.monotonic()
        cand = evaluate(spec, build_features, {"model": model, "params": params,
                                               "target_transform": cfg.get("target_transform")}, deadline)
        durations.append(time.monotonic() - began)
        return float(cand.cv_score)

    try:
        base = run(start_params)
    except Exception as error:  # noqa: BLE001 — нечего сравнивать: вернуть стартовые
        log.warning("tuning: стартовые параметры не оценены: %s", error)
        return start_params

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize" if greater else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=int(cfg.get("seed", 42))))

    def objective(trial: Any) -> float:
        if _left(deadline) < TIME_MARGIN * max(durations):
            trial.study.stop()
            raise optuna.TrialPruned("no time for another trial")
        params = {**start_params, **_space(trial, model)}
        try:
            return run(params)
        except TimeoutError:
            trial.study.stop()
            raise optuna.TrialPruned("deadline")
        except Exception as error:  # noqa: BLE001 — неудачная проба не роняет подбор
            log.info("tuning: проба %d упала: %s", trial.number, error)
            raise optuna.TrialPruned(str(error)[:200])

    timeout = _left(deadline)
    study.optimize(objective, n_trials=n_trials, timeout=None if math.isinf(timeout) else timeout,
                   catch=(), gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        return start_params
    best = study.best_trial
    better = best.value > base if greater else best.value < base
    log.info("tuning %s: %d проб, старт %.6g → лучшее %.6g%s", model, len(done), base, best.value,
             "" if better else " — не лучше, остаются стартовые")
    if not better:
        return start_params
    params = {**start_params, **best.params}
    if model == "lightgbm":
        params["subsample_freq"] = 1 if params.get("subsample", 1.0) < 1.0 else 0
    return params
