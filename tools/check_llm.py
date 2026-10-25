"""Smoke-test доступа к LLM: один короткий запрос.

Запуск: .venv/bin/python tools/check_llm.py
Ключ и адрес берутся из .env или из переменных окружения (на платформе их задаёт она сама).
"""

import os
import sys

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

FALLBACK_URL = "https://api.alfahack-gateway26.ru/v1"


def ask(base_url: str | None) -> None:
    client = OpenAI(base_url=base_url, timeout=60, max_retries=1)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b"),
        messages=[{"role": "user", "content": "Ответь одним словом: ок"}],
        max_tokens=300,
    )
    usage = response.usage
    message = response.choices[0].message
    print("Ответ:", (message.content or "").strip())
    print("finish_reason:", response.choices[0].finish_reason)
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if reasoning:
        print("Есть поле reasoning, длина:", len(reasoning))
    if usage:
        print(f"Токены: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}")


def main() -> int:
    load_dotenv()  # переменные окружения платформы имеют приоритет над .env только если уже заданы
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or "ПОДСТАВЬТЕ" in key:
        print("OPENAI_API_KEY не задан: впишите ключ в .env (см. .env.example)")
        return 1
    print("Адрес:", os.getenv("OPENAI_BASE_URL", "<не задан>"))
    try:
        ask(None)
        return 0
    except APIConnectionError as error:
        print("Основной адрес недоступен:", type(error).__name__, "— пробую запасной")
    except APIStatusError as error:
        print(f"Ошибка API {error.status_code}: {error.message}")
        return 2
    try:
        ask(FALLBACK_URL)
        print("Работает запасной адрес: поменяйте OPENAI_BASE_URL в .env")
        return 0
    except Exception as error:
        print("Запасной адрес тоже не сработал:", type(error).__name__, error)
        return 3


if __name__ == "__main__":
    sys.exit(main())
