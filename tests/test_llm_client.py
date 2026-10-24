"""Тесты LLMClient на заглушках: настоящая LLM не вызывается, бюджет не тратится."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import openai

# добавляем корень проекта в путь, иначе из папки tests не найдётся llm_client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import LLMClient, LLMUnavailable

MESSAGES = [{"role": "user", "content": "привет"}]


class FakeCompletions:
    """Подделка под client.chat.completions: отвечает заранее заготовленными ответами."""

    def __init__(self, replies):
        # replies: список ответов по порядку. Строка вернётся как ответ модели,
        # а исключение будет брошено, как будто API сломалось
        self.replies = list(replies)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def connection_error():
    # такую ошибку openai кидает, когда пропала связь или вышел таймаут
    return openai.APIConnectionError(request=httpx.Request("POST", "http://test"))


class LLMClientTests(unittest.TestCase):
    def setUp(self):
        # для каждого теста своя пустая папка кэша, чтобы тесты не влияли друг на друга
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache_dir = tmp.name

    def make_client(self, replies, cache=True):
        client = LLMClient(cache_dir=self.cache_dir if cache else None)
        fake = FakeCompletions(replies)
        # подсовываем подделку вместо настоящего клиента OpenAI
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        return client, fake

    # ---------- chat ----------

    def test_chat_returns_text_and_counts_tokens(self):
        client, fake = self.make_client(["Привет"])
        self.assertEqual(client.chat(MESSAGES), "Привет")
        self.assertEqual(client.usage_summary()["total_tokens"], 15)

    def test_second_identical_request_comes_from_cache(self):
        client, fake = self.make_client(["Привет"])
        client.chat(MESSAGES)
        client.chat(MESSAGES)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(client.cache_hits, 1)

    def test_cache_can_be_disabled(self):
        client, fake = self.make_client(["раз", "два"], cache=False)
        client.chat(MESSAGES)
        client.chat(MESSAGES)
        self.assertEqual(fake.calls, 2)

    def test_budget_error_is_not_retried(self):
        client, fake = self.make_client([Exception("Budget has been exceeded")])
        with self.assertRaises(LLMUnavailable):
            client.chat(MESSAGES)
        # повторять смысла нет, значит запрос был ровно один
        self.assertEqual(fake.calls, 1)

    def test_temporary_error_is_retried(self):
        client, fake = self.make_client([connection_error(), "Привет"])
        # подменяем sleep, чтобы тест не ждал настоящие секунды
        with mock.patch("llm_client.time.sleep"):
            self.assertEqual(client.chat(MESSAGES), "Привет")
        self.assertEqual(fake.calls, 2)

    def test_gives_up_after_max_retries(self):
        client, fake = self.make_client([connection_error()] * 3)
        with mock.patch("llm_client.time.sleep"):
            with self.assertRaises(LLMUnavailable):
                client.chat(MESSAGES)
        self.assertEqual(fake.calls, 3)

    def test_non_temporary_error_is_not_retried(self):
        client, fake = self.make_client([ValueError("что-то непонятное")])
        with self.assertRaises(LLMUnavailable):
            client.chat(MESSAGES)
        self.assertEqual(fake.calls, 1)

    def test_no_api_key(self):
        client = LLMClient(cache_dir=None)
        # на время теста вычищаем окружение, чтобы ключа точно не было
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMUnavailable):
                client.chat(MESSAGES)

    # ---------- chat_json ----------

    def test_json_inside_markdown_and_think_block(self):
        reply = '<think>подумаю</think>\n```json\n{"word": "привет"}\n```'
        client, fake = self.make_client([reply])
        self.assertEqual(client.chat_json(MESSAGES), {"word": "привет"})

    def test_json_with_text_around(self):
        client, fake = self.make_client(['Вот ответ: {"a": 1} надеюсь, подходит'])
        self.assertEqual(client.chat_json(MESSAGES), {"a": 1})

    def test_bad_json_is_fixed_on_second_try(self):
        client, fake = self.make_client(["это не json", '{"a": 1}'])
        self.assertEqual(client.chat_json(MESSAGES), {"a": 1})
        self.assertEqual(fake.calls, 2)

    def test_schema_violation_is_fixed(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        }
        # в первом ответе нет обязательного поля "a"
        client, fake = self.make_client(['{"b": 2}', '{"a": 1}'])
        self.assertEqual(client.chat_json(MESSAGES, schema=schema), {"a": 1})
        self.assertEqual(fake.calls, 2)

    def test_json_never_valid_raises(self):
        client, fake = self.make_client(["мусор", "мусор", "мусор"])
        with self.assertRaises(LLMUnavailable):
            client.chat_json(MESSAGES)
        # первая попытка и две попытки исправить
        self.assertEqual(fake.calls, 3)

    def test_fixed_json_is_cached_for_next_time(self):
        client, fake = self.make_client(["не json", '{"a": 1}'])
        client.chat_json(MESSAGES)
        calls_before = fake.calls
        # второй раз ответ должен прийти из кэша, без новых запросов
        self.assertEqual(client.chat_json(MESSAGES), {"a": 1})
        self.assertEqual(fake.calls, calls_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)