"""Tests for the translation agent."""

from __future__ import annotations

import unittest

from app.agent.agent import TranslationAgent
from app.translation.client import ClientConfig


class FakeClient:
    """Test double that records calls and returns canned responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["OK"]
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], thinking: bool = False) -> str:
        self.calls.append({"messages": list(messages), "thinking": thinking})
        idx = len(self.calls) - 1
        if idx < len(self.responses):
            return self.responses[idx]
        return self.responses[-1]


class TranslationAgentTests(unittest.TestCase):
    """Verify agent session lifecycle and message construction."""

    def setUp(self) -> None:
        self.fake = FakeClient(["已理解规则", "HELLO", "WORLD"])
        self.agent = TranslationAgent.__new__(TranslationAgent)
        self.agent._client = self.fake
        self.agent.messages = []

    def test_digest_rules_builds_correct_messages(self) -> None:
        self.agent.digest_rules("翻译规则：全大写。")

        self.assertEqual(len(self.agent.messages), 3)  # system + user + assistant
        self.assertEqual(self.agent.messages[0]["role"], "system")
        self.assertIn("全大写", self.agent.messages[0]["content"])
        self.assertEqual(self.agent.messages[1]["role"], "user")
        self.assertEqual(self.agent.messages[2]["role"], "assistant")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertTrue(self.fake.calls[0]["thinking"])

    def test_translate_appends_and_returns(self) -> None:
        self.agent.digest_rules("规则。")
        result = self.agent.translate("hello")

        self.assertEqual(result, "HELLO")
        self.assertEqual(len(self.agent.messages), 5)
        self.assertEqual(self.agent.messages[3]["role"], "user")
        self.assertEqual(self.agent.messages[3]["content"], "hello")
        self.assertEqual(self.agent.messages[4]["role"], "assistant")

    def test_second_translate_uses_accumulated_context(self) -> None:
        self.agent.digest_rules("规则。")
        self.agent.translate("hello")
        result = self.agent.translate("world")

        self.assertEqual(result, "WORLD")
        self.assertEqual(len(self.agent.messages), 7)
        # Second call sends all previous messages
        self.assertEqual(len(self.fake.calls[2]["messages"]), 6)

    def test_digest_then_translate_all_with_thinking(self) -> None:
        self.agent.digest_rules("规则。")
        self.agent.translate("hello")

        # All calls should have thinking=True
        for call in self.fake.calls:
            self.assertTrue(call["thinking"], "All calls use thinking=enabled")
