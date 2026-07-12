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
          "user_summary": "翻译会保持简洁，同时不改变原意。",
          "change_items": [{"title": "控制长度", "description": "译文最多八十个字符。"}],
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
        self.assertIn("保持简洁", result.user_summary)
        self.assertEqual(result.change_items[0]["title"], "控制长度")

    def test_invalid_review_fields_degrade_without_changing_machine_rules(self) -> None:
        result = PromptOptimizer._parse_result(
            '{"supplemental_rules":"Machine rule", "constraint_policy":{"version":1,"rules":[]},'
            '"user_summary":["wrong"], "change_items":{"title":"wrong"}}'
        )
        self.assertEqual(result, "Machine rule")
        self.assertEqual(result.user_summary, "")
        self.assertEqual(result.change_items, ())

    def test_review_metadata_never_becomes_machine_text_when_rules_are_missing(self) -> None:
        result = PromptOptimizer._parse_result(
            '{"user_summary":"仅供用户看的说明",'
            '"change_items":[{"title":"说明","description":"不能进入机器规则"}]}'
        )
        self.assertEqual(str(result), "")
        self.assertEqual(result.user_summary, "仅供用户看的说明")

    def test_non_chinese_review_text_degrades_to_empty(self) -> None:
        result = PromptOptimizer._parse_result(
            '{"supplemental_rules":"Machine rule", "user_summary":"English only",'
            '"change_items":[{"title":"English","description":"English"}]}'
        )
        self.assertEqual(str(result), "Machine rule")
        self.assertEqual(result.user_summary, "")
        self.assertEqual(result.change_items, ())

    def test_review_text_has_local_length_limits(self) -> None:
        payload = {
            "supplemental_rules": "Machine rule",
            "user_summary": "中" * 900,
            "change_items": [{"title": "题" * 60, "description": "说" * 300}],
        }
        import json
        result = PromptOptimizer._parse_result(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(len(result.user_summary), 800)
        self.assertEqual(len(result.change_items[0]["title"]), 40)
        self.assertEqual(len(result.change_items[0]["description"]), 240)

    def test_optimize_rejects_missing_ai_settings(self) -> None:
        optimizer = PromptOptimizer()

        with self.assertRaises(PromptOptimizationError):
            optimizer.optimize(AppSettings(), PromptConstraints(text="anything"))


if __name__ == "__main__":
    unittest.main()
