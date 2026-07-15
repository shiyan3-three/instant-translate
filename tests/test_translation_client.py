"""Tests for OpenAI-compatible request payloads."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from app.translation.client import ClientConfig, OpenAICompatibleClient, TranslationError


class OpenAICompatibleClientTests(unittest.TestCase):
    """Verify generic completion and translation payloads."""

    def test_complete_sends_system_and_user_messages(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "optimized prompt"}}]
        }

        with patch("app.translation.client.httpx.post", return_value=response) as post:
            client = OpenAICompatibleClient(
                ClientConfig(
                    base_url="https://api.example.test/v1",
                    api_key="key",
                    model="model",
                )
            )

            result = client.complete("system", "user")

        self.assertEqual(result, "optimized prompt")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "user"})
        self.assertNotIn("thinking", payload)
        self.assertGreaterEqual(post.call_args.kwargs["timeout"], 30.0)

    def test_chat_can_send_explicit_disabled_thinking(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "translated"}}]
        }

        with patch("app.translation.client.httpx.post", return_value=response) as post:
            client = OpenAICompatibleClient(
                ClientConfig(
                    base_url="https://api.deepseek.test/v1",
                    api_key="key",
                    model="deepseek-v4-flash",
                )
            )

            result = client.chat(
                [{"role": "user", "content": "hello"}],
                thinking="disabled",
            )

        self.assertEqual(result, "translated")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_chat_can_send_explicit_enabled_thinking(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "confirmed"}}]
        }

        with patch("app.translation.client.httpx.post", return_value=response) as post:
            client = OpenAICompatibleClient(
                ClientConfig(
                    base_url="https://api.deepseek.test/v1",
                    api_key="key",
                    model="deepseek-v4-pro",
                )
            )

            result = client.chat(
                [{"role": "user", "content": "请确认规则"}],
                thinking="enabled",
            )

        self.assertEqual(result, "confirmed")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["thinking"], {"type": "enabled"})

    def test_chat_rejects_truncated_translation(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "partial translation"},
                }
            ]
        }

        with patch("app.translation.client.httpx.post", return_value=response), patch(
            "app.logger.get_debug_logger"
        ) as debug_logger:
            client = OpenAICompatibleClient(
                ClientConfig(
                    base_url="https://api.deepseek.test/v1",
                    api_key="key",
                    model="deepseek-v4-flash",
                )
            )
            with self.assertRaisesRegex(TranslationError, "truncated"):
                client.chat([{"role": "user", "content": "hello"}])

        debug_logger.return_value.debug.assert_any_call(
            "API response structure: message_keys=%s content_len=%d reasoning_len=%d finish_reason=%r",
            ["content"],
            len("partial translation"),
            0,
            "length",
        )

    def test_chat_normal_stop_response_is_returned(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": "translated"}}]
        }
        with patch("app.translation.client.httpx.post", return_value=response):
            self.assertEqual(
                self._client().chat([{"role": "user", "content": "hello"}]),
                "translated",
            )

    def test_completion_response_shape_errors_are_translation_errors(self) -> None:
        cases = {
            "non-json": ValueError("bad JSON"),
            "json-array": [],
            "missing-choices": {},
            "empty-choices": {"choices": []},
            "missing-message": {"choices": [{}]},
            "missing-content": {"choices": [{"message": {}}]},
            "empty-content": {"choices": [{"message": {"content": "  "}}]},
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                response = Mock()
                response.raise_for_status.return_value = None
                response.json.side_effect = payload if isinstance(payload, Exception) else None
                if not isinstance(payload, Exception):
                    response.json.return_value = payload
                with patch("app.translation.client.httpx.post", return_value=response):
                    with self.assertRaises(TranslationError):
                        self._client().complete("system", "user")

    @staticmethod
    def _client() -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            ClientConfig(
                base_url="https://api.example.test/v1",
                api_key="key",
                model="model",
            )
        )

    def test_complete_rejects_reasoning_content_without_final_answer(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "Thinking about the translation...",
                    }
                }
            ]
        }

        with patch("app.translation.client.httpx.post", return_value=response):
            client = OpenAICompatibleClient(
                ClientConfig(
                    base_url="https://api.deepseek.test/v1",
                    api_key="key",
                    model="deepseek-v4-reasoner",
                )
            )

            with self.assertRaisesRegex(TranslationError, "empty translation"):
                client.complete("system", "user")

    def test_list_models_parses_openai_style_payload(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [
                {"id": "deepseek-v4-pro"},
                {"id": "deepseek-v4-flash"},
                {"id": "deepseek-v4-flash"},
                {"foo": "bar"},
            ]
        }

        with patch("app.translation.client.httpx.get", return_value=response) as get:
            client = OpenAICompatibleClient(
                ClientConfig(
                    base_url="https://api.example.test/v1",
                    api_key="key",
                    model="unused",
                )
            )
            models = client.list_models()

        self.assertEqual(models, ["deepseek-v4-flash", "deepseek-v4-pro"])
        self.assertEqual(
            get.call_args.kwargs["url"],
            "https://api.example.test/v1/models",
        )
        self.assertIn("Authorization", get.call_args.kwargs["headers"])

    def test_list_models_raises_on_http_error(self) -> None:
        import httpx

        response = Mock()
        response.status_code = 401
        response.text = "unauthorized"
        error = httpx.HTTPStatusError(
            "boom",
            request=Mock(),
            response=response,
        )
        response.raise_for_status.side_effect = error

        with patch("app.translation.client.httpx.get", return_value=response):
            client = OpenAICompatibleClient(
                ClientConfig(
                    base_url="https://api.example.test/v1",
                    api_key="bad",
                    model="unused",
                )
            )
            with self.assertRaisesRegex(TranslationError, "API error 401"):
                client.list_models()


if __name__ == "__main__":
    unittest.main()
