"""Tests for AI-assisted prompt optimization."""

from __future__ import annotations

import unittest

from app.prompt.models import PromptConstraints
from app.prompt.optimizer import PromptOptimizationError, PromptOptimizer
from app.settings import AppSettings


class FakeClient:
    """Client test double for prompt optimization."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.configs = []

    def complete(self, system_prompt: str, user_prompt: str, thinking=None) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "Optimized rules"


class JsonFakeClient(FakeClient):
    def complete(self, system_prompt: str, user_prompt: str, thinking=None) -> str:
        self.calls.append((system_prompt, user_prompt))
        return """{
          "supplemental_rules": "Keep the translation concise.",
          "constraint_policy": {
            "version": 1,
            "rules": [
              {
                "type": "max_length",
                "params": {"characters": 80},
                "enforcement": "both",
                "scope": {},
                "source_text": "80 characters maximum"
              }
            ]
          }
        }"""


class PromptOptimizerTests(unittest.TestCase):
    """Verify optimizer uses AI settings and compiler messages."""

    def test_optimize_returns_ai_rules(self) -> None:
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.fast_model = "fast-model"
        settings.ai.thinking_model = "thinking-model"
        fake_client = FakeClient()

        def factory(config):
            fake_client.configs.append(config)
            return fake_client

        optimizer = PromptOptimizer(client_factory=factory)

        result = optimizer.optimize(settings, PromptConstraints(text="short style"))

        self.assertEqual(result, "Optimized rules")
        self.assertEqual(len(fake_client.calls), 1)
        self.assertEqual(fake_client.configs[0].model, "thinking-model")
        self.assertIn("short style", fake_client.calls[0][1])
        self.assertIn("Do not create glossary tables", fake_client.calls[0][0])
        self.assertIn("Knowledge Reference layer", fake_client.calls[0][0])

    def test_optimize_parses_safe_policy_json(self) -> None:
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.thinking_model = "thinking-model"
        client = JsonFakeClient()
        optimizer = PromptOptimizer(client_factory=lambda config: client)

        result = optimizer.optimize(settings, PromptConstraints(text="Stay concise"))

        self.assertEqual(result, "Keep the translation concise.")
        self.assertEqual(result.policy.rules[0].type, "max_length")

    def test_optimize_rejects_missing_ai_settings(self) -> None:
        optimizer = PromptOptimizer()

        with self.assertRaises(PromptOptimizationError):
            optimizer.optimize(AppSettings(), PromptConstraints(text="anything"))


if __name__ == "__main__":
    unittest.main()
