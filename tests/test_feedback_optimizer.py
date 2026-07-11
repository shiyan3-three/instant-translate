from __future__ import annotations

import unittest

from app.feedback.optimizer import FeedbackOptimizer
from app.feedback.store import FeedbackRecord
from app.settings import AppSettings


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete(self, system_prompt: str, user_prompt: str, thinking=None) -> str:
        self.calls.append((system_prompt, user_prompt, thinking))
        return self.response


class FeedbackOptimizerTests(unittest.TestCase):
    def test_accepts_code_fenced_json_and_keeps_three_keywords(self) -> None:
        client = FakeClient(
            """```json
            {
              "trigger": "小偷",
              "trigger_options": ["小偷", "电车", "钱包", "第四项"],
              "rule": "小偷按扒手语境自然翻译，不要逐字拼接。",
              "improved_translation": "さいふ  が  でんしゃ  で  すり  に  ぬすまれた"
            }
            ```"""
        )
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.thinking_model = "deepseek-v4-pro"
        record = FeedbackRecord.create(
            group_id=1,
            source_language="中文",
            target_language="日本語",
            ocr_text="钱包在电车被小偷偷走了",
            translation_text="wrong",
        )
        optimizer = FeedbackOptimizer(client_factory=lambda config: client)

        result = optimizer.optimize(settings, record)

        self.assertEqual(result.trigger, "小偷")
        self.assertEqual(result.trigger_options, ["小偷", "电车", "钱包"])
        self.assertIn("扒手", result.rule)
        self.assertEqual(client.calls[0][2], "enabled")

    def test_accepts_improved_translation_without_keyword_memory(self) -> None:
        client = FakeClient(
            """{
              "trigger": "",
              "trigger_options": [],
              "rule": "当前句是被动受害表达，按自然被动句翻译。",
              "improved_translation": "りさん は となりの ひと に おそく まで さわがれました"
            }"""
        )
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.thinking_model = "deepseek-v4-pro"
        record = FeedbackRecord.create(
            group_id=1,
            source_language="中文",
            target_language="日本語",
            ocr_text="邻居闹到很晚，小李被吵到了。",
            translation_text="wrong",
        )
        optimizer = FeedbackOptimizer(client_factory=lambda config: client)

        result = optimizer.optimize(settings, record)

        self.assertEqual(result.trigger, "")
        self.assertEqual(result.trigger_options, [])
        self.assertIn("被动受害", result.rule)
        self.assertEqual(
            result.improved_translation,
            "りさん は となりの ひと に おそく まで さわがれました",
        )


if __name__ == "__main__":
    unittest.main()
