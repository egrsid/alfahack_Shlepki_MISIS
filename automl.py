"""AutoGluon на фолдах fallback (роль P1/P3; написано P4 — владелец может переписать).

    run(spec, best, deadline, seconds=...) -> list[Candidate]

best — кандидат fallback: его фолды (`meta["folds"]`), строки train (`meta["train_rows"]`), классы и метрика.
AutoGluon получает сырые признаки этих строк и колонку `__fold__` = номер фолда (`groups` → LeaveOneGroupOut):
делит ровно по нашим фолдам, поэтому OOF каждой его модели сравнимы с fallback по парным фолдам и смешиваются
с ним в ensemble.blend. Стэкинг выключен (чистые OOF), предобученные модели (TabPFN) исключены правилами.
Кандидат — на каждую модель лидерборда (включая его WeightedEnsemble). AutoGluon работает в отдельном процессе
(`python3 automl.py job.pkl`): изоляция OpenMP/torch от уже обученных бустингов, жёсткий таймаут, падение или
нехватка памяти не роняют решение. Каталог с данными и моделями удаляется после. Любой сбой → [] (fallback
остаётся). autogluon импортируется только в процессе-воркере.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import metrics
import submission

log = logging.getLogger("automl")

FOLD_COL = "__fold__"
RESERVE = 60.0          # на предсказания и сборку кандидатов после fit
PREDICT_MARGIN = 120.0  # процессу-воркеру сверх бюджета fit: предсказания OOF/test всех моделей
MIN_SECONDS = 60.0
# предобученные (запрещены правилами) и тяжёлые/нетабличные модели
EXCLUDED = ["TABPFN", "TABPFNMIX", "FT_TRANSFORMER", "FASTTEXT", "AG_TEXT_NN", "AG_IMAGE_NN", "AG_AUTOMM"]
# имя метрики metrics.py → eval_metric AutoGluon (нет в словаре — метрика AutoGluon по умолчанию)
AG_METRICS = {"rmse": "root_mean_squared_error", "mse": "mean_squared_error", "mae": "mean_absolute_error",
              "medae": "median_absolute_error", "r2": "r2", "mape": "mean_absolute_percentage_error",
              "smape": "symmetric_mean_absolute_percentage_error", "rmsle": "root_mean_squared_error",
              "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro",
              "weighted_f1": "f1_weighted", "micro_f1": "f1_micro", "f1": "f1", "precision": "precision",
              "recall": "recall", "mcc": "mcc", "roc_auc": "roc_auc", "pr_auc": "average_precision",
              "logloss": "log_loss", "qwk": "quadratic_kappa"}


def _left(deadline: Any) -> float:
    try:
        return float(deadline.left()) if deadline is not None else math.inf
    except Exception:  # noqa: BLE001
        return math.inf


def _hyperparameters() -> dict:
    """Набор моделей AutoGluon «default» без FASTAI (в контейнере нет fastai) и предобученных."""
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
    config = dict(get_hyperparameter_config("default"))
    for key in ("FASTAI", *EXCLUDED):
        config.pop(key, None)
    return config


def _problem_type(meta: dict) -> str:
    classes = meta.get("classes")
    if classes is None:
        return "regression"
    return "binary" if len(classes) == 2 else "multiclass"


def _proba(frame: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Вероятности AutoGluon (колонки — исходные метки) в порядке классов fallback."""
    by_name = {metrics.label_str(c): c for c in frame.columns}
    out = np.zeros((len(frame), len(classes)))
    for j, name in enumerate(classes):
        col = by_name.get(metrics.label_str(name))
        if col is not None:
            out[:, j] = frame[col].to_numpy(dtype=float)
    return out


def _fit_predict(job: dict) -> dict:
    """В отдельном процессе: обучение AutoGluon и предсказания всех моделей лидерборда (OOF и test)."""
    from autogluon.tabular import TabularPredictor

    data, test = pd.read_pickle(job["train"]), pd.read_pickle(job["test"])
    classes, problem, label = job["classes"], job["problem"], job["label"]
    extra = {"positive_class": classes[-1]} if problem == "binary" else {}  # как у fallback: последний класс
    predictor = TabularPredictor(label=label, problem_type=problem, eval_metric=job["eval_metric"],
                                 path=job["path"], verbosity=1, groups=FOLD_COL, **extra)
    fit_kw = dict(time_limit=job["budget"], num_bag_folds=job["n_folds"], num_bag_sets=1, num_stack_levels=0,
                  dynamic_stacking=False, hyperparameters=job["models"] or _hyperparameters(),
                  excluded_model_types=EXCLUDED, calibrate_decision_threshold=False)
    if job["threads"]:
        fit_kw["num_cpus"] = int(job["threads"])
    predictor.fit(data, **fit_kw)
    out: dict = {"names": []}
    for name in predictor.leaderboard(silent=True)["model"]:
        try:
            if problem == "regression":
                oof = predictor.predict_oof(model=name).to_numpy(dtype=float)
                test_pred = predictor.predict(test, model=name).to_numpy(dtype=float)
            else:
                oof = _proba(predictor.predict_proba_oof(model=name), classes)
                test_pred = _proba(predictor.predict_proba(test, model=name), classes)
        except Exception as error:  # noqa: BLE001 — модель без OOF (например, не бэггинг) пропускаем
            print(f"automl: {name} без OOF/предсказаний: {str(error)[:200]}", flush=True)
            continue
        out["names"].append(name)
        out[f"oof::{name}"], out[f"test::{name}"] = oof, test_pred
    return out


def available() -> bool:
    """Установлен ли AutoGluon (без импорта: сам импорт — только в процессе-воркере)."""
    import importlib.util
    try:
        return importlib.util.find_spec("autogluon.tabular") is not None
    except (ImportError, ValueError):
        return False


def _command(job_path: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), str(job_path)]


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.kill()
    proc.wait()


def run(spec: Any, best: Any, deadline: Any = None, *, seconds: float = 2400.0, threads: int | None = None,
        workdir: Path | None = None, models: dict | None = None) -> list[Any]:
    """Кандидаты AutoGluon по убыванию CV. Обучение — в отдельном процессе: torch в одном процессе с уже
    обученными LightGBM/XGBoost (OpenMP) может зависнуть; там же жёсткий таймаут и защита от падения/памяти."""
    start = time.monotonic()
    meta = getattr(best, "meta", None) or {}
    folds = meta.get("folds")
    if folds is None or isinstance(spec.target, list):
        return []
    budget = min(float(seconds), _left(deadline) - RESERVE)
    if budget < MIN_SECONDS:
        log.info("automl: нет времени (%.0f с) — пропуск", budget)
        return []
    if not available():
        log.warning("automl: AutoGluon (autogluon.tabular) не установлен — пропуск")
        return []
    root = Path(workdir) if workdir is not None else Path(spec.task_dir) / ".agent_work" / "automl"
    try:
        folds = np.asarray(folds, dtype=int)
        classes = meta.get("classes")
        metric = meta.get("metric") or spec.metric
        target = submission.targets_of(spec)[0]
        train = pd.read_csv(submission.task_file(spec, "train.csv"), low_memory=False)
        test = pd.read_csv(submission.task_file(spec, "test.csv"), low_memory=False)
        rows = meta.get("train_rows")
        if rows is not None:
            train = train.iloc[np.asarray(rows, dtype=int)]
        else:
            train = train[train[target].notna()]
        if len(train) != len(folds):
            log.warning("automl: строк train %d, фолдов %d — пропуск", len(train), len(folds))
            return []
        drop = {spec.id_col, target, *(getattr(spec, "drop_cols", None) or [])}
        features = [c for c in test.columns if c in train.columns and c not in drop]
        data = train[features].reset_index(drop=True)
        data[target] = train[target].to_numpy()
        data[FOLD_COL] = folds
        problem = _problem_type(meta)
        if problem != "regression":
            data[target] = data[target].map(metrics.label_str)  # метки как у fallback: 1, 1.0, "1" → "1"

        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        job = {"train": str(root / "train.pkl"), "test": str(root / "test.pkl"), "path": str(root / "models"),
               "out": str(root / "out.pkl"), "label": target, "problem": problem, "classes": classes,
               "eval_metric": AG_METRICS.get(metric), "budget": budget, "n_folds": len(np.unique(folds)),
               "threads": threads, "models": models}
        data.to_pickle(job["train"])
        test[features].reset_index(drop=True).to_pickle(job["test"])
        with open(root / "job.pkl", "wb") as fh:
            pickle.dump(job, fh)
        log.info("automl: AutoGluon %s, метрика %s, %d строк × %d признаков, %d фолдов, бюджет %.0f с", problem,
                 AG_METRICS.get(metric, "по умолчанию"), len(data), len(features), job["n_folds"], budget)
        timeout = budget + max(PREDICT_MARGIN, 0.25 * budget)  # fit укладывается в budget, дальше — предсказания
        with open(root / "automl.log", "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(_command(root / "job.pkl"),
                                    stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill(proc)
                log.warning("automl: AutoGluon не уложился в %.0f с — убит", timeout)
                return []
        if code != 0 or not Path(job["out"]).is_file():
            tail = (root / "automl.log").read_text(encoding="utf-8", errors="replace")[-1500:]
            log.warning("automl: процесс AutoGluon завершился с кодом %s:\n%s", code, tail)
            return []
        out = pd.read_pickle(job["out"])

        y = data[target].to_numpy(dtype=float) if problem == "regression" else metrics.encode(data[target], classes)
        cands = []
        for name in out["names"]:
            oof, test_pred = np.asarray(out[f"oof::{name}"], dtype=float), np.asarray(out[f"test::{name}"], dtype=float)
            if not (np.isfinite(oof).all() and np.isfinite(test_pred).all()) or len(oof) != len(y):
                continue
            scores = [float(metrics.score(metric, y[folds == f], oof[folds == f])) for f in np.unique(folds)]
            cands.append(type(best)(name=f"automl_{name}", cv_score=float(np.mean(scores)),
                                    cv_std=float(np.std(scores)), oof=oof, test_pred=test_pred,
                                    meta={"folds": folds, "train_rows": rows, "classes": classes, "metric": metric,
                                          "fold_scores": scores, "fold_scores_eff": list(scores),
                                          "cv_score_post": float(np.mean(scores)), "model": f"autogluon:{name}"}))
        sign = 1.0 if metrics.greater_is_better(metric) else -1.0
        cands.sort(key=lambda c: -sign * c.cv_score)
        log.info("automl: %d моделей за %.0f с: %s", len(cands), time.monotonic() - start,
                 ", ".join(f"{c.name[7:]}={c.cv_score:.5g}" for c in cands[:8]))
        return cands
    except Exception as error:  # noqa: BLE001 — AutoML — бонус: сбой не трогает fallback
        log.warning("automl упал: %s: %s", type(error).__name__, str(error)[:500])
        return []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str]) -> int:
    """Процесс-воркер: python3 automl.py job.pkl → out.pkl с OOF и test каждой модели."""
    with open(argv[1], "rb") as fh:
        job = pickle.load(fh)
    out = _fit_predict(job)
    pd.to_pickle(out, job["out"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
