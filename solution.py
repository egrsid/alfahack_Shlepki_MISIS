"""Решение: python3 solution.py task.md → submission.csv рядом с task.md.

0. понимание задачи (task_understanding): эвристики по файлам + проверенные подсказки LLM (метрика, даты, текст).
1. fallback (P3): spec по файлам задачи, честная схема CV (периоды / группы дублей), бустинги, бленд, веса классов.
   submission.csv пишется сразу (константа), затем после каждой лучшей модели.
2. AutoML (automl.py): AutoGluon на ТЕХ ЖЕ фолдах, до 40 мин; бленд «все модели fallback + AutoGluon».
3. агент (P4): LLM-цикл гипотез на ТЕХ ЖЕ фолдах; затем бленд «fallback + AutoGluon + агент» (ensemble).
   После шагов 2 и 3 пишется лучший из кандидатов и бленда, только если по парным фолдам он лучше уже записанного
   больше шума и прошёл guards (сверка test-прогноза с fallback).
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

import pandas as pd

import fallback
import metrics
import submission

log = logging.getLogger("solution")

TOTAL_SECONDS = 3 * 3600          # лимит платформы на задачу (включая LLM)
RESEARCH_END = TOTAL_SECONDS - 1800  # агент не начинает ничего после T+2ч30: запас на запись и проверку
AUTOML_SECONDS = 2400.0           # AutoGluon сразу после fallback (не зависит от LLM)
AGENT_MIN_SECONDS = 900.0         # AutoML оставляет агенту хотя бы столько окна research


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


def pick(spec: Any, current: Any, options: list, trusted: Any, stage: str) -> Any | None:
    """Лучший по CV из options, если он честно лучше записанного current (парные фолды, выигрыш больше шума)
    и прошёл guards против trusted (кандидат fallback); иначе None — файл не трогаем."""
    good = [c for c in options if fallback.clearly_better(spec.metric, c, current) and not suspicious(spec, c, trusted)]
    if not good:
        if options:
            top = options[0]
            log.info("%s: лучший кандидат %s cv=%.6g не лучше записанного %s (%.6g) больше шума", stage, top.name,
                     fallback.effective_cv(top), current.name, fallback.effective_cv(current))
        return None
    sign = 1.0 if metrics.greater_is_better(spec.metric) else -1.0
    winner = max(good, key=lambda c: sign * fallback.effective_cv(c))
    log.info("%s: %s cv=%.6g лучше записанного %s (%.6g) больше шума", stage, winner.name,
             fallback.effective_cv(winner), current.name, fallback.effective_cv(current))
    return winner


def improve(spec: Any, best: Any, current: Any, extra: list, stage: str) -> Any:
    """Бленд «все модели fallback + extra» и сами extra; что честно лучше записанного — пишется. Возвращает
    то, что теперь в файле."""
    options = list(extra)
    if extra:
        try:
            import ensemble
            mixed = ensemble.blend(spec, [best, *fallback.pool(best), *extra])
            if mixed is not None:
                options.insert(0, mixed)
        except Exception:
            log.exception("%s: бленд не собран", stage)
    winner = pick(spec, current, options, best, stage)
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


def understand(task_path: Path, client: Any) -> Any:
    """spec задачи: task_understanding (эвристики + проверенный ответ LLM), при сбое — эвристики fallback."""
    try:
        import task_understanding
        return task_understanding.understand_task(task_path, client)
    except Exception:
        log.exception("понимание задачи упало — эвристики fallback")
        return fallback.guess_spec(task_path)


def solve(task_path: str | Path, *, llm: Any = "auto", deadline: Deadline | None = None,
          agent_config: Any = None) -> list[str]:
    """Весь запуск; возвращает проблемы записанного submission.csv (пусто — файл в порядке)."""
    start = time.monotonic()
    task_path = Path(task_path).resolve()
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

    current, found = best, []
    if best is not None and not multi:
        seconds = automl_seconds(deadline, client is not None)
        if seconds > 0:
            try:
                import automl
                found = automl.run(spec, best, deadline, seconds=seconds, threads=fallback.n_threads())
                current = improve(spec, best, current, found, "AutoML")
            except Exception:
                log.exception("AutoML упал — остаётся сабмит fallback")

    if best is not None and not multi and deadline.stage("research") > 0 and client is not None:
        try:
            if agent_config is None and os.environ.get("SOLUTION_AGENT_ITERS"):  # только для локальных проверок
                import agent
                agent_config = agent.AgentConfig(max_iters=int(os.environ["SOLUTION_AGENT_ITERS"]))
            agent_cands = run_agent(spec, best, client, deadline, agent_config)
            current = improve(spec, best, current, [*found, *agent_cands], "агент")
        except Exception:
            log.exception("агент упал — остаётся записанный сабмит")

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
