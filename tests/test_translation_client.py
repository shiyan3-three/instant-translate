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


if __name__ == "__main__":
    unittest.main()
