"""Решение: python3 solution.py task.md → submission.csv рядом с task.md.

1. fallback (P3): spec по файлам задачи, честная схема CV (периоды / группы дублей), бустинги, бленд, веса классов.
   submission.csv пишется сразу (константа), затем после каждой лучшей модели.
2. агент (P4): LLM-цикл гипотез (признаки, выбор модели, постобработка) на ТЕХ ЖЕ фолдах, что у fallback.
   Его кандидат пишется, только если по парным фолдам он лучше fallback больше шума.
3. проверка записанного файла; если он испорчен — переписывается кандидатом fallback.

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
import submission

log = logging.getLogger("solution")

TOTAL_SECONDS = 3 * 3600          # лимит платформы на задачу (включая LLM)
RESEARCH_END = TOTAL_SECONDS - 1800  # агент не начинает ничего после T+2ч30: запас на запись и проверку


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
    """Копия spec для агента: заметки по колонкам из data_dictionary.csv и строковые колонки как категории."""
    spec = copy.copy(spec)
    try:
        notes: dict[str, str] = {}
        dictionary = pd.read_csv(submission.task_file(spec, "data_dictionary.csv"))
        name_col = next((c for c in dictionary.columns if str(c).lower() in ("feature", "name", "column", "field")),
                        dictionary.columns[0])
        desc_col = next((c for c in dictionary.columns if "desc" in str(c).lower()), dictionary.columns[-1])
        unit_col = next((c for c in dictionary.columns if str(c).lower() == "unit"), None)
        for _, row in dictionary.iterrows():
            note = str(row[desc_col])
            if unit_col is not None and pd.notna(row[unit_col]):
                note += f" [{row[unit_col]}]"
            notes[str(row[name_col])] = note
        spec.feature_notes = {**notes, **(getattr(spec, "feature_notes", None) or {})}
    except Exception as error:  # словаря может не быть или формат другой
        log.info("data_dictionary.csv не прочитан: %s", error)
    if not getattr(spec, "cat_cols", None):
        try:
            head = pd.read_csv(submission.task_file(spec, "train.csv"), nrows=2000)
            targets = submission.targets_of(spec)
            spec.cat_cols = [c for c in head.columns
                             if head[c].dtype == object and c not in targets and c != spec.id_col]
        except Exception as error:
            log.info("строковые колонки не определены: %s", error)
    return spec


def run_agent(spec: Any, best: Any, llm: Any, deadline: Deadline, config: Any = None) -> Any | None:
    """Лучший кандидат агента, если он честно лучше fallback (те же фолды, выигрыш больше шума), иначе None."""
    import agent

    meta = getattr(best, "meta", None) or {}
    evaluate = agent._optional("evaluator", "evaluate") or agent.LocalEvaluator(
        folds=meta.get("folds"), rows=meta.get("train_rows"), early_stopping=100)  # фолды и модели как у fallback
    found = agent.research(agent_spec(spec), best, llm, deadline, config=config, evaluate=evaluate)
    for cand in found:  # по убыванию CV
        if fallback.clearly_better(spec.metric, cand, best):
            log.info("агент: %s cv=%.6g лучше fallback (%.6g) больше шума", cand.name, cand.cv_score,
                     fallback.effective_cv(best))
            return cand
    if found:
        log.info("агент: лучший кандидат %s cv=%.6g не лучше fallback (%.6g) больше шума — остаётся fallback",
                 found[0].name, found[0].cv_score, fallback.effective_cv(best))
    return None


def solve(task_path: str | Path, *, llm: Any = "auto", deadline: Deadline | None = None,
          agent_config: Any = None) -> list[str]:
    """Весь запуск; возвращает проблемы записанного submission.csv (пусто — файл в порядке)."""
    start = time.monotonic()
    task_path = Path(task_path).resolve()
    if deadline is None:
        research_end = float(os.environ.get("SOLUTION_RESEARCH_END", RESEARCH_END))
        deadline = Deadline(start, TOTAL_SECONDS, research_end)
    spec = fallback.guess_spec(task_path)
    multi = isinstance(spec.target, list)

    best = None
    try:
        best = (fallback.run_multi if multi else fallback.run)(spec, deadline)
    except Exception:
        log.exception("fallback упал — остаётся константный сабмит")
        submission.write_dummy(spec)

    if best is not None and not multi and deadline.stage("research") > 0:
        try:
            client = make_llm(Path(spec.task_dir)) if llm == "auto" else llm
            if agent_config is None and os.environ.get("SOLUTION_AGENT_ITERS"):  # только для локальных проверок
                import agent
                agent_config = agent.AgentConfig(max_iters=int(os.environ["SOLUTION_AGENT_ITERS"]))
            if client is not None:
                winner = run_agent(spec, best, client, deadline, agent_config)
                if winner is not None:
                    submission.write_candidate(spec, winner)
                    log.info("submission.csv ← агент (%s)", winner.name)
        except Exception:
            log.exception("агент упал — остаётся сабмит fallback")

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
