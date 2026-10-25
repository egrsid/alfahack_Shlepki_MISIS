"""Решение: python3 solution.py task.md → submission.csv рядом с task.md.

0. понимание задачи (task_understanding): эвристики по файлам + проверенные подсказки LLM (метрика, даты, текст).
1. fallback (P3): spec по файлам задачи, честная схема CV (периоды / группы дублей), бустинги, бленд, веса классов.
   submission.csv пишется сразу (константа), затем после каждой лучшей модели.
2. AutoML (automl.py): по умолчанию выключен (SOLUTION_AUTOML_SECONDS > 0 — включить): на public AutoGluon
   проигрывал fallback и на CV, и на отложенной части, а 40 минут нужнее агенту. Включённый — AutoGluon на ТЕХ ЖЕ
   фолдах; бленд «все модели fallback + AutoGluon».
2½. зоопарк A (final.py, P1): другие модели на ТЕХ ЖЕ матрицах, фолдах и признаках сущности, что у рецепта fallback
   (лог-таргет, медленный шаг, глубина, веса классов, случайные варианты); бленд «fallback + зоопарк».
3. агент (P4): LLM-цикл гипотез на ТЕХ ЖЕ фолдах; затем бленд «fallback + AutoGluon + зоопарк + агент» (ensemble).
3½. зоопарк B: сиды для весомых в бленде членов зоопарка и ещё случайные варианты; финальный бленд всего.
   Раскладка 3 часов: fallback → зоопарк A до T+1:10 → агент до T+2:05 (доводка до сидов — до T+2:25) →
   зоопарк B до T+2:40 → последние 20 минут на бленд, запись и проверку. Агент остановился раньше — дальше сразу;
   без LLM зоопарк A идёт до T+2:05. Раньше после ухода агента (public-регрессия: на T+36 мин) время простаивало.
   После шагов 2, 2½, 3 и 3½ пишется лучший из кандидатов и бленда, только если по парным фолдам он лучше уже записанного
   больше шума и прошёл guards (сверка test-прогноза с fallback). Сравнение — по честным оценкам с обеих сторон:
   у бленда fallback веса фолда подобраны на остальных фолдах (fallback.honest_view), как у ensemble и агента.
4. проверка записанного файла; если он испорчен — переписывается кандидатом fallback.

Любой сбой агента или LLM оставляет сабмит fallback. Ключ и адрес LLM — только из переменных окружения
(OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL). Переменные SOLUTION_* — только для локальных проверок.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import fallback
import metrics
import submission

log = logging.getLogger("solution")

TOTAL_SECONDS = 3 * 3600          # лимит платформы на задачу (включая LLM)
RESEARCH_END = TOTAL_SECONDS - 3300  # агент не начинает ничего после T+2:05; SOLUTION_RESEARCH_END
AUTOML_SECONDS = 0.0              # AutoGluon после fallback: выключен (проигрывал fallback), SOLUTION_AUTOML_SECONDS
AGENT_MIN_SECONDS = 900.0         # AutoML и зоопарк оставляют агенту хотя бы столько окна research
AGENT_WINDOW = 3300.0             # окно агента до RESEARCH_END (зоопарк A заканчивается за столько до него: T+1:10)
AGENT_FINISH = 1200.0             # доводка агента до сидов — не позже RESEARCH_END + это (T+2:25)
FINAL_RESERVE = 1200.0            # последние 20 минут: финальный бленд, запись и проверка файла (зоопарк B — до них)
FINAL_SECONDS = float(TOTAL_SECONDS)  # потолок одного этапа зоопарка, SOLUTION_FINAL_SECONDS (0 — зоопарк выкл.)
# test — новый период: финальный бленд (веса по поздним периодам, новые уровни) пишется, если он не хуже fallback
# больше шума, а не только «лучше больше шума». Копия public-регрессии «годы 1–4 → год 5»: бленд на годе 5 лучше
# (29.91 против 30.30, по всем строкам 42.7 против 49.7), а честное CV по поздним фолдам — хуже (37.68 против 36.84):
# при 4–5 периодах веса фолда подбираются по 1–2 годам, и CV тут шумнее, чем выигрыш. Осознанный риск.
PERIOD_BLEND_NONINFERIOR = True


class Deadline:
    """Общий дедлайн задачи: left() — до конца 3 часов, stage("research") — до конца окна агента."""

    def __init__(self, start: float, total: float, research_end: float) -> None:
        self.start, self.total, self.research_end = start, total, research_end

    def left(self) -> float:
        return max(0.0, self.start + self.total - time.monotonic())

    def stage(self, name: str) -> float:
        if name == "research":
            return max(0.0, min(self.start + self.research_end - time.monotonic(), self.left()))
        return self.left()


class AgentLLM:
    """llm_client.LLMClient (P2) для агента.

    chat_json не отдаём: план агент разбирает сам (клиент на невалидном JSON бросает LLMUnavailable, а для агента
    это «LLM недоступна — остановиться»). HTTP 400/422 пробрасываем исходной ошибкой: агент сам повторит запрос
    без подсказки «без рассуждения», если шлюз её не принял.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    @property
    def prompt_tokens(self) -> int:
        return int(self.client.prompt_tokens)

    @property
    def completion_tokens(self) -> int:
        return int(self.client.completion_tokens)

    def chat(self, messages: list[dict], **kw: Any) -> str:
        try:
            return self.client.chat(messages, **kw)
        except Exception as error:
            cause = error.__cause__
            if getattr(cause, "status_code", None) in (400, 422):
                raise cause from None
            raise


def make_llm(task_dir: Path) -> Any:
    """Клиент LLM или None (нет ключа, LLM_DISABLED=1, нет llm_client.py)."""
    if os.environ.get("LLM_DISABLED") == "1" or not os.environ.get("OPENAI_API_KEY"):
        log.info("LLM выключена (LLM_DISABLED=1 или нет OPENAI_API_KEY) — только fallback")
        return None
    try:
        from llm_client import LLMClient
    except ImportError as error:
        log.warning("llm_client недоступен (%s) — только fallback", error)
        return None
    return AgentLLM(LLMClient(cache_dir=str(task_dir / ".agent_work" / "llm_cache")))


def agent_spec(spec: Any) -> Any:
    """Копия spec для агента: заметки по колонкам (data_dictionary.csv) и все строковые колонки как категории."""
    import task_understanding

    spec = copy.copy(spec)
    if not getattr(spec, "feature_notes", None):
        spec.feature_notes = task_understanding.read_dictionary(spec)
    try:
        head = pd.read_csv(submission.task_file(spec, "train.csv"), nrows=2000)
        skip = {*submission.targets_of(spec), spec.id_col, *(getattr(spec, "date_cols", None) or []),
                *(getattr(spec, "text_cols", None) or [])}
        strings = [c for c in head.columns if head[c].dtype == object and c not in skip]
        spec.cat_cols = list(dict.fromkeys([*(getattr(spec, "cat_cols", None) or []), *strings]))
    except Exception as error:
        log.info("строковые колонки не определены: %s", error)
    return spec


def run_agent(spec: Any, best: Any, llm: Any, deadline: Deadline, config: Any = None) -> list:
    """Кандидаты агента по убыванию CV. Есть рецепт fallback (meta["context"]) — агент стартует с лучшего
    кандидата fallback и добавляет к нему новые колонки (RecipeEvaluator); иначе — свой LocalEvaluator на тех же
    фолдах и строках train."""
    import agent

    meta = getattr(best, "meta", None) or {}
    evaluate = agent._optional("evaluator", "evaluate")
    if evaluate is None and meta.get("context") is not None:
        try:  # агент улучшает сам лучший кандидат fallback: тот же рецепт + новые колонки
            evaluate = agent.RecipeEvaluator(best, threads=fallback.n_threads())
        except ValueError as error:
            log.info("агент: рецепт fallback недоступен (%s) — свой LocalEvaluator", error)
    if evaluate is None:
        evaluate = agent.LocalEvaluator(folds=meta.get("folds"), rows=meta.get("train_rows"),
                                        early_stopping=100)  # фолды и модели как у fallback
    return agent.research(agent_spec(spec), best, llm, deadline, config=config, evaluate=evaluate)


def suspicious(spec: Any, cand: Any, best: Any) -> list[str]:
    """Проблемы кандидата по guards.check_candidate против fallback (пусто — можно писать)."""
    try:
        import guards
        problems = guards.check_candidate(spec, cand, best)
    except Exception:
        log.exception("guards упали — кандидат %s не проверен", cand.name)
        return []
    if problems:
        log.warning("guards: %s отклонён: %s", cand.name, "; ".join(problems))
    return problems


def on_folds(cand: Any, folds: list[int] | None) -> Any:
    """Кандидат с оценками только по фолдам folds (None — как есть): для сравнения по поздним периодам."""
    meta = getattr(cand, "meta", None) or {}
    scores = meta.get("fold_scores_eff", meta.get("fold_scores"))
    if folds is None or scores is None or len(scores) <= max(folds):
        return cand
    chosen = [float(scores[i]) for i in folds]
    view = copy.copy(cand)
    view.meta = {**meta, "fold_scores": chosen, "fold_scores_eff": chosen, "cv_score_post": float(np.mean(chosen))}
    return view


def pick(spec: Any, current: Any, options: list, trusted: Any, stage: str) -> Any | None:
    """Лучший по CV из options, если он честно лучше записанного current (парные фолды, выигрыш больше шума)
    и прошёл guards против trusted (кандидат fallback); иначе None — файл не трогаем.

    Оценки — честные с обеих сторон (fallback.honest_view): веса бленда fallback подобраны на тех же OOF, по которым
    он оценивается, и без этого претендент с честной оценкой (ensemble, агент) проигрывал бы ему по построению.
    Test — новый период (схема CV fallback «по периодам»): сравниваются только фолды поздней половины периодов
    (ensemble.recent_folds) — ранние периоды на test похожи меньше (копия public-регрессии «годы 1–4 → год 5»:
    бленд с весами по поздним годам на годе 5 лучше fallback, а по всем фолдам поровну — хуже).
    """
    folds = None
    try:
        import ensemble
        meta = getattr(trusted, "meta", None) or {}
        folds = ensemble.recent_folds(meta, len(np.asarray(trusted.oof)))
    except Exception:
        log.exception("%s: поздние фолды не определены — сравнение по всем", stage)
    ref = on_folds(fallback.honest_view(current), folds)
    sign = 1.0 if metrics.greater_is_better(spec.metric) else -1.0
    views = sorted(((c, on_folds(fallback.honest_view(c), folds)) for c in options),
                   key=lambda cv: -sign * fallback.effective_cv(cv[1]))
    if folds and PERIOD_BLEND_NONINFERIOR:
        base = on_folds(fallback.honest_view(trusted), folds)
        for cand, view in views:
            meta = getattr(cand, "meta", None) or {}
            if (meta.get("blend") and meta.get("honest_oof") and not fallback.clearly_better(spec.metric, base, view)
                    and not suspicious(spec, cand, trusted)):
                log.info("%s: бленд %s cv=%.6g не хуже fallback (%.6g) больше шума на фолдах поздних периодов %s — "
                         "пишется (test — новый период)", stage, cand.name, fallback.effective_cv(view),
                         fallback.effective_cv(base), folds)
                return cand
    good = [(c, v) for c, v in views
            if fallback.clearly_better(spec.metric, v, ref) and not suspicious(spec, c, trusted)]
    if not good:
        if views:
            top = views[0][1]
            log.info("%s: лучший кандидат %s cv=%.6g не лучше записанного %s (%.6g, честная оценка) больше шума",
                     stage, top.name, fallback.effective_cv(top), current.name, fallback.effective_cv(ref))
        return None
    winner, view = good[0]
    log.info("%s: %s cv=%.6g лучше записанного %s (%.6g, честная оценка%s) больше шума", stage, winner.name,
             fallback.effective_cv(view), current.name, fallback.effective_cv(ref),
             f", фолды поздних периодов {folds}" if folds else "")
    return winner


def with_new_levels(cand: Any, mixed: Any) -> Any | None:
    """Кандидат cand, у которого прогнозы строк test с новыми уровнями категорий взяты из бленда mixed (там они —
    по стресс-строкам OOF, ensemble.new_level_rows); None — нечего менять (нет таких строк, cand — сам бленд или
    уже с ними, другие классы). Защита и тогда, когда сам бленд по CV не пишется: на копии public-регрессии с новым
    штатом в test — 42.7 RMSE по всем строкам против 49.7 у fallback."""
    info = ((getattr(mixed, "meta", None) or {}).get("new_levels")) if mixed is not None else None
    meta = getattr(cand, "meta", None) or {}
    if not info or cand is mixed or meta.get("new_levels") or meta.get("classes") != mixed.meta.get("classes"):
        return None
    index = np.asarray(info.get("test_index", []), dtype=int)
    test_pred = np.array(cand.test_pred, dtype=float, copy=True)
    if not len(index) or test_pred.shape != np.shape(mixed.test_pred):
        return None
    test_pred[index] = np.asarray(mixed.test_pred, dtype=float)[index]
    out = copy.copy(cand)
    out.name, out.test_pred = f"{cand.name}+new_levels", test_pred
    out.meta = {**meta, "new_levels": {k: v for k, v in info.items() if k != "test_index"} | {"from": mixed.name}}
    return out


def improve(spec: Any, best: Any, current: Any, extra: list, stage: str, members: list = (),
            report: dict | None = None) -> Any:
    """Бленд «все модели fallback + extra + members» и сами extra; что честно лучше записанного — пишется.
    members — только члены бленда (пул агента: как отдельный сабмит не прошли его правило принятия). report — сюда
    кладётся собранный бленд (report["blend"], веса — в его meta["blend"]). Возвращает то, что теперь в файле."""
    options, mixed = list(extra), None
    try:
        import ensemble
        # fallback — с честным OOF: его веса бленда подобраны на тех же строках, а ensemble оценивает честно
        mixed = ensemble.blend(spec, [fallback.honest_view(best), *fallback.pool(best), *extra, *members])
        if mixed is not None:
            options.insert(0, mixed)
        if report is not None:
            report["blend"] = mixed
    except Exception:
        log.exception("%s: бленд не собран", stage)
    winner = pick(spec, current, options, best, stage)
    patched = with_new_levels(winner if winner is not None else current, mixed)
    if patched is not None and not suspicious(spec, patched, best):
        log.info("%s: %d строк test с новыми уровнями категорий — прогнозы из бленда %s", stage,
                 patched.meta["new_levels"]["test_rows"], mixed.name)
        winner = patched
    if winner is None:
        return current
    submission.write_candidate(spec, winner)
    log.info("submission.csv ← %s (%s)", stage, winner.name)
    return winner


def automl_seconds(deadline: Deadline, with_agent: bool) -> float:
    """Бюджет AutoGluon: AUTOML_SECONDS (SOLUTION_AUTOML_SECONDS — для локальных проверок, 0 — выкл.), не больше
    окна research за вычетом минимального окна агента."""
    limit = float(os.environ.get("SOLUTION_AUTOML_SECONDS", AUTOML_SECONDS))
    return max(0.0, min(limit, deadline.stage("research") - (AGENT_MIN_SECONDS if with_agent else 0.0)))


def zoo_a_end(deadline: Deadline, with_agent: bool) -> float:
    """Конец зоопарка A (секунды от старта): за AGENT_WINDOW до конца окна research (короткое окно — за
    AGENT_MIN_SECONDS), без агента — конец окна research."""
    end = deadline.research_end
    if with_agent:
        end -= AGENT_WINDOW if end > 2 * AGENT_WINDOW else AGENT_MIN_SECONDS
    return end


def zoo_seconds(deadline: Deadline, end: float) -> float:
    """Окно этапа зоопарка до end (секунды от старта): не больше SOLUTION_FINAL_SECONDS (FINAL_SECONDS) и не позже
    чем за FINAL_RESERVE до конца 3 часов."""
    limit = float(os.environ.get("SOLUTION_FINAL_SECONDS", FINAL_SECONDS))
    end = min(end, deadline.total - FINAL_RESERVE)
    return max(0.0, min(limit, deadline.start + end - time.monotonic()))


def agent_deadline(deadline: Deadline) -> Deadline:
    """Дедлайн агента: окно research то же, но «конец» (по нему агент считает время доводки до сидов) — через
    AGENT_FINISH после окна research: доводка не съедает зоопарк B и финальный бленд."""
    return Deadline(deadline.start, min(deadline.total, deadline.research_end + AGENT_FINISH), deadline.research_end)


def blend_weights(report: dict) -> dict:
    mixed = report.get("blend")
    return dict((mixed.meta or {}).get("blend") or {}) if mixed is not None else {}


def run_zoo_a(spec: Any, best: Any, current: Any, deadline: Deadline, with_agent: bool) -> tuple[Any, list, dict]:
    """Зоопарк A (final.train) на рецепте fallback → бленд и запись, если честно лучше записанного. Возвращает (что
    теперь в файле, кандидаты зоопарка, веса бленда — для сидов этапа B)."""
    seconds = zoo_seconds(deadline, zoo_a_end(deadline, with_agent))
    if seconds < 60:
        log.info("зоопарк A пропущен: окно %.0f с", seconds)
        return current, [], {}
    import final

    start = time.monotonic()
    zoo = final.train(spec, best, until=start + seconds, threads=fallback.n_threads())
    report: dict = {}
    current = improve(spec, best, current, zoo, "зоопарк A", report=report)
    log.info("зоопарк A: %d моделей за %.0f с (окно %.0f с)", len(zoo), time.monotonic() - start, seconds)
    return current, zoo, blend_weights(report)


def run_zoo_b(spec: Any, best: Any, current: Any, deadline: Deadline, zoo: list, weights: dict,
              extra: list = (), members: list = ()) -> tuple[Any, list]:
    """Зоопарк B — после агента до FINAL_RESERVE: сиды для членов зоопарка с весом в последнем бленде (final.bag),
    затем ещё случайные варианты; финальный бленд всего (extra — AutoML и принятые агентом, members — пул агента).
    Возвращает (что теперь в файле, кандидаты зоопарка — с сидами вместо одиночных, где досчитаны)."""
    seconds = zoo_seconds(deadline, deadline.total - FINAL_RESERVE)
    if seconds < 60:
        log.info("зоопарк B пропущен: окно %.0f с", seconds)
        return current, zoo
    import final

    start = time.monotonic()
    until, threads = start + seconds, fallback.n_threads()
    bagged = final.bag(spec, best, zoo, weights, until=until, threads=threads) if zoo and weights else []
    if bagged:
        replaced = {b.meta["bagged_from"] for b in bagged}
        zoo = [*bagged, *(c for c in zoo if c.name not in replaced)]
    more = final.train(spec, best, until=until, threads=threads, n_random=2 * final.N_RANDOM, listed=False,
                       tag="b", seed_offset=1)
    zoo = [*zoo, *more]
    if bagged or more:
        current = improve(spec, best, current, [*extra, *zoo], "зоопарк B", members=members)
    log.info("зоопарк B: сиды у %d, новых моделей %d за %.0f с (окно %.0f с)", len(bagged), len(more),
             time.monotonic() - start, seconds)
    return current, zoo


def understand(task_path: Path, client: Any) -> Any:
    """spec задачи: task_understanding (эвристики + проверенный ответ LLM), при сбое — эвристики fallback."""
    try:
        import task_understanding
        return task_understanding.understand_task(task_path, client)
    except Exception:
        log.exception("понимание задачи упало — эвристики fallback")
        return fallback.guess_spec(task_path)


OPTIONAL_PACKAGES = ("lightgbm", "xgboost", "catboost", "sklearn", "optuna", "autogluon.tabular", "torch", "openai",
                     "psutil", "pyarrow", "fastparquet")


def log_environment() -> dict[str, bool]:
    """Какие пакеты есть в окружении (без импорта): по логу платформы видно, чего не хватило."""
    import importlib.util

    found = {}
    for name in OPTIONAL_PACKAGES:
        try:
            found[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found[name] = False
    log.info("окружение: Python %s; есть: %s; нет: %s", sys.version.split()[0],
             ", ".join(n for n, ok in found.items() if ok) or "—",
             ", ".join(n for n, ok in found.items() if not ok) or "—")
    return found


def solve(task_path: str | Path, *, llm: Any = "auto", deadline: Deadline | None = None,
          agent_config: Any = None) -> list[str]:
    """Весь запуск; возвращает проблемы записанного submission.csv (пусто — файл в порядке)."""
    start = time.monotonic()
    task_path = Path(task_path).resolve()
    log_environment()
    if deadline is None:
        research_end = float(os.environ.get("SOLUTION_RESEARCH_END", RESEARCH_END))
        deadline = Deadline(start, TOTAL_SECONDS, research_end)
    client = None
    try:
        client = make_llm(task_path.parent) if llm == "auto" else llm
    except Exception:
        log.exception("клиент LLM не создан — только fallback")
    spec = understand(task_path, client)
    multi = isinstance(spec.target, list)

    best = None
    try:
        best = (fallback.run_multi if multi else fallback.run)(spec, deadline)
    except Exception:
        log.exception("fallback упал — остаётся константный сабмит")
        submission.write_dummy(spec)

    current, found, zoo, weights, accepted, pool = best, [], [], {}, [], []
    if best is not None and not multi:
        seconds = automl_seconds(deadline, client is not None)
        if seconds > 0:
            try:
                import automl
                found = automl.run(spec, best, deadline, seconds=seconds, threads=fallback.n_threads())
                current = improve(spec, best, current, found, "AutoML")
            except Exception:
                log.exception("AutoML упал — остаётся сабмит fallback")
        try:
            current, zoo, weights = run_zoo_a(spec, best, current, deadline, client is not None)
        except Exception:
            log.exception("зоопарк A упал — остаётся записанный сабмит")

    if best is not None and not multi and deadline.stage("research") > 0 and client is not None:
        try:
            if agent_config is None and os.environ.get("SOLUTION_AGENT_ITERS"):  # только для локальных проверок
                import agent
                agent_config = agent.AgentConfig(max_iters=int(os.environ["SOLUTION_AGENT_ITERS"]))
            agent_cands = run_agent(spec, best, client, agent_deadline(deadline), agent_config)
            pool = [c for c in agent_cands if (getattr(c, "meta", None) or {}).get("pool_only")]
            accepted = [c for c in agent_cands if not (getattr(c, "meta", None) or {}).get("pool_only")]
            log.info("агент: принято %d, в пуле бленда %d", len(accepted), len(pool))
            report: dict = {}
            current = improve(spec, best, current, [*found, *zoo, *accepted], "агент", members=pool, report=report)
            weights = blend_weights(report) or weights
        except Exception:
            log.exception("агент упал — остаётся записанный сабмит")

    if best is not None and not multi:
        try:
            current, zoo = run_zoo_b(spec, best, current, deadline, zoo, weights, [*found, *accepted], pool)
        except Exception:
            log.exception("зоопарк B упал — остаётся записанный сабмит")

    problems = submission.check_file(spec)
    if problems and best is not None:
        log.error("сабмит не прошёл проверку (%s) — переписываю кандидатом fallback", problems)
        try:
            submission.write_candidate(spec, best)
            problems = submission.check_file(spec)
        except Exception:
            log.exception("кандидат fallback не записан")
    if problems:
        submission.write_dummy(spec, overwrite=True)
        problems = submission.check_file(spec)
    log.info("готово за %.0f с; submission.csv: %s", time.monotonic() - start, problems or "ok")
    return problems


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return 0 if not solve(argv[1] if len(argv) > 1 else "task.md") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
