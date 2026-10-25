"""Ручной smoke-тест: запросы к LLM, кэш и JSON.
Тратит немного бюджета, поэтому в автотесты не входит."""
import sys
from pathlib import Path

# добавляем корень проекта в путь, иначе из папки tests не найдётся llm_client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import LLMClient, LLMUnavailable


def main() -> int:
    client = LLMClient()
    question = [{"role": "user", "content": "Ответь одним словом: привет"}]
    json_question = [{
        "role": "user",
        "content": 'Верни JSON вида {"word": "..."} с одним словом-приветствием. Только JSON.',
    }]
    schema = {
        "type": "object",
        "properties": {"word": {"type": "string"}},
        "required": ["word"],
    }
    try:
        # первый запрос идёт в API, второй такой же должен прийти из кэша
        first = client.chat(question, max_tokens=500)
        second = client.chat(question, max_tokens=500)
        data = client.chat_json(json_question, schema=schema, max_tokens=1000)
    except LLMUnavailable as e:
        print("LLM недоступна:", e)
        return 1
    print("Ответ LLM:", repr(first))
    print("Второй ответ такой же:", first == second)
    print("JSON:", data)
    print("Расход:", client.usage_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())