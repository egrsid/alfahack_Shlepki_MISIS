"""Клиент для LLM. Ключ и адрес берём только из переменных окружения."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import jsonschema
import openai
from openai import OpenAI

log = logging.getLogger(__name__)

# модель по умолчанию, на платформе её можно переопределить через OPENAI_MODEL
DEFAULT_MODEL = "qwen/qwen3.8-27b"

MAX_RETRIES = 3    # сколько всего попыток делаем, если ошибка временная
BASE_DELAY = 2.0   # пауза перед повтором растёт: 2, потом 4 секунды


class LLMUnavailable(Exception):
    # кидаем её, если с LLM что-то не так: нет ключа, кончился бюджет, упала сеть.
    # тот, кто вызывает клиента, должен её поймать и уйти на запасной вариант без LLM
    pass


def _load_dotenv(path: str = ".env") -> None:
    # читаем .env руками, чтобы локально не exportить переменные каждый раз.
    # на платформе файла нет, там всё уже лежит в окружении
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # пропускаем пустые строки, комментарии и мусор без "="
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # setdefault, чтобы не затереть переменную, которая уже задана снаружи
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _is_budget_error(e: Exception) -> bool:
    # прокси пишет "Budget has been exceeded", поэтому смотрим по тексту ошибки,
    # а не по коду: код у такой ошибки может быть разным
    return "budget" in str(e).lower()


def _is_retryable(e: Exception) -> bool:
    # временные проблемы, которые имеет смысл повторить:
    # нет связи или таймаут (таймаут входит в APIConnectionError), 429, 5xx
    if isinstance(e, openai.APIConnectionError):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code == 429 or e.status_code >= 500
    return False


def _extract_json(text: str) -> dict:
    # модель может вывести блок рассуждений <think>...</think>, его выкидываем
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # собираем варианты, где может лежать JSON, и пробуем по очереди
    candidates = [text]
    # вариант 1: JSON обёрнут в ```json ... ```
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    # вариант 2: JSON окружён обычным текстом, берём от первой { до последней }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        # нам нужен именно объект, а не список или число
        if isinstance(data, dict):
            return data
    raise ValueError("в ответе нет корректного JSON-объекта")


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        timeout: float = 120.0,
        cache_dir: str | None = ".llm_cache",
    ):
        _load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        # папка для кэша; если передать None, кэш полностью выключается
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # сам клиент создаём не сразу, а при первом запросе,
        # чтобы без ключа ничего не падало при простом создании объекта
        self._client: OpenAI | None = None

        # счётчики расхода, чтобы понимать, сколько тратим
        self.calls = 0              # сколько раз реально сходили в API
        self.cache_hits = 0         # сколько раз взяли ответ из кэша
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise LLMUnavailable("OPENAI_API_KEY не задан")
            # OpenAI() сам читает OPENAI_API_KEY и OPENAI_BASE_URL из окружения.
            # max_retries=0, потому что повторы делаем сами в chat()
            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        return self._client

    # ---------- кэш ----------

    def _cache_path(self, messages: list[dict], kwargs: dict) -> Path | None:
        if self.cache_dir is None:
            return None
        # ключ кэша зависит от модели, сообщений и параметров запроса:
        # если хоть что-то поменялось, это уже другой запрос
        raw = json.dumps(
            {"model": self.model, "messages": messages, "kwargs": kwargs},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _cache_read(self, path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        except Exception:
            # битый файл кэша не должен ломать работу, просто сходим в API
            return None

    def _cache_write(self, path: Path | None, content: str) -> None:
        if path is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # сначала пишем во временный файл, потом переименовываем,
            # чтобы при обрыве не остался наполовину записанный кэш
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"content": content}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except Exception:
            pass  # кэш нужен для экономии, из-за него падать нельзя

    def _cache_delete(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    # ---------- учёт токенов ----------

    def _register_usage(self, resp) -> None:
        self.calls += 1
        usage = getattr(resp, "usage", None)
        # usage иногда может не прийти, тогда просто не считаем токены
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0
        log.info(
            "LLM: запрос #%d, токенов всего: %d",
            self.calls,
            self.prompt_tokens + self.completion_tokens,
        )

    def usage_summary(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    # ---------- сам запрос ----------

    def chat(self, messages: list[dict], use_cache: bool = True, **kwargs) -> str:
        """Один запрос к модели, возвращает текст ответа.

        use_cache=False нужен, когда мы нарочно хотим получить новый ответ
        (например, при повторной попытке починить плохой JSON).
        """
        cache_path = self._cache_path(messages, kwargs) if use_cache else None
        cached = self._cache_read(cache_path)
        if cached is not None:
            self.cache_hits += 1
            log.info("LLM: ответ взят из кэша")
            return cached

        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs
                )
                self._register_usage(resp)
                # content может быть None, поэтому подстраховываемся пустой строкой
                content = resp.choices[0].message.content or ""
                # пустой ответ в кэш не кладём, он скорее всего неудачный
                if content:
                    self._cache_write(cache_path, content)
                return content
            except Exception as e:
                last_error = e
                # бюджет кончился: повторять бессмысленно, сразу сдаёмся
                if _is_budget_error(e):
                    raise LLMUnavailable(f"Бюджет LLM исчерпан: {e}") from e
                # ошибка не временная (например, неверный ключ): тоже сдаёмся сразу
                if not _is_retryable(e):
                    raise LLMUnavailable(f"Ошибка запроса к LLM: {e}") from e
                # это была последняя попытка, ждать уже незачем
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(BASE_DELAY * 2 ** attempt)

        raise LLMUnavailable(
            f"LLM не ответила после {MAX_RETRIES} попыток: {last_error}"
        ) from last_error

    def chat_json(
        self,
        messages: list[dict],
        schema: dict | None = None,
        fix_attempts: int = 2,
        **kwargs,
    ) -> dict:
        """Запрос, в котором ждём от модели JSON-объект.

        Если JSON не нашёлся или не подошёл под schema, просим модель
        исправиться, до fix_attempts раз. В самом промпте нужно просить JSON.
        """
        # путь кэша для исходного запроса: туда в конце положим проверенный ответ
        orig_path = self._cache_path(messages, kwargs)
        current = list(messages)

        for attempt in range(fix_attempts + 1):
            # на первой попытке можно взять кэш, на повторных нужен свежий ответ
            text = self.chat(current, use_cache=(attempt == 0), **kwargs)
            try:
                data = _extract_json(text)
                if schema is not None:
                    jsonschema.validate(data, schema)
            except (ValueError, jsonschema.ValidationError) as e:
                # у ValidationError есть .message, у ValueError только str(e)
                problem = getattr(e, "message", str(e))
                log.info("LLM: плохой JSON (попытка %d): %s", attempt + 1, problem)
                # показываем модели её ответ и говорим, что не так
                if text:
                    current = current + [{"role": "assistant", "content": text}]
                current = current + [{
                    "role": "user",
                    "content": (
                        f"Ответ не подходит: {problem}. "
                        "Верни только исправленный JSON-объект, без пояснений и без markdown."
                    ),
                }]
                continue

            # если пришлось исправлять, кладём хороший ответ в кэш под исходный запрос,
            # чтобы в следующий раз не мучиться заново
            if attempt > 0:
                self._cache_write(orig_path, json.dumps(data, ensure_ascii=False))
            return data

        # так и не получилось: плохой ответ из кэша убираем, чтобы он не мешал потом
        self._cache_delete(orig_path)
        raise LLMUnavailable(
            f"Модель не вернула корректный JSON за {fix_attempts + 1} попытки"
        )