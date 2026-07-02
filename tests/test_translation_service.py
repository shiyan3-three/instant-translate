"""Tests for translation-service prompt selection."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent.agent import TranslationAgent
from app.agent.session_store import AgentSessionMeta
from app.feedback.store import FeedbackStore
from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.settings import AppSettings
from app.translation.client import TranslationError
from app.translation.service import TranslationRequest, TranslationService


class FailingClient:
    """Client test double that always fails."""

    def translate(self, system_prompt: str, source_text: str) -> str:
        raise TranslationError("old request timed out")


class FakeSessionStore:
    """In-memory session store test double."""

    def __init__(self, loaded_messages=None) -> None:
        self.loaded_messages = loaded_messages
        self.load_calls = []
        self.save_calls = []

    def load(self, meta: AgentSessionMeta):
        self.load_calls.append(meta)
        return self.loaded_messages

    def save(self, meta: AgentSessionMeta, messages: list[dict]) -> None:
        self.save_calls.append((meta, list(messages)))

    def delete(self, group_id: int) -> None:
        pass


class FakeMemoryAgent:
    """Agent test double that records runtime memory hints."""

    def __init__(self) -> None:
        self.messages = []
        self.text = ""
        self.memory_hints = None

    def translate(self, text: str, memory_hints=None) -> str:
        self.text = text
        self.memory_hints = memory_hints
        return "the national college entrance exam starts soon"


class TranslationServicePromptTests(unittest.TestCase):
    """Verify runtime translation prompts prefer confirmed compiled prompts."""

    def test_current_prompt_uses_compiled_prompt_file_when_available(self) -> None:
        compiled_content = f"{DEFAULT_BASE_PROMPT}\n\nCONFIRMED USER CONSTRAINT"
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "compiled.md"
            prompt_path.write_text(compiled_content, encoding="utf-8")
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = str(prompt_path)
            service = TranslationService(settings)

            prompt = service._current_prompt("English", "中文")
            service.shutdown()

        self.assertIn("CONFIRMED USER CONSTRAINT", prompt)
        self.assertIn("Translate from English to 中文.", prompt)
        # Compact joins lines with spaces; DEFAULT_BASE_PROMPT content is present
        self.assertIn("instant translation assistant", prompt)

    def test_current_prompt_falls_back_to_fixed_template_without_compiled_file(self) -> None:
        settings = AppSettings()
        settings.prompt.compiled_prompt_path = ""
        service = TranslationService(settings)

        prompt = service._current_prompt("日本語", "English")
        service.shutdown()

        self.assertIn(DEFAULT_BASE_PROMPT, prompt)
        self.assertIn("Translate from 日本語 to English.", prompt)

    def test_current_prompt_clarifies_hiragana_as_semantic_translation(self) -> None:
        compiled_content = """
# Instant Translate Compiled Prompt

## User Constraint Layer
如果中文翻译为日语时，翻译的日语句子只能由平假名构成

## AI Optimization Layer
All kanji must be converted to their corresponding hiragana readings without exception.
- Convert all Chinese characters to hiragana equivalents.
- Maintain original meaning while adhering to the hiragana-only constraint.
**Examples**:
- 中国語 (ちゅうごくご) → ちゅうごくご

## Runtime Direction
For every request, follow the source and target languages appended by the app.
""".strip()
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "compiled.md"
            prompt_path.write_text(compiled_content, encoding="utf-8")
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = str(prompt_path)
            service = TranslationService(settings)

            prompt = service._current_prompt("中文", "日本語")
            service.shutdown()

        self.assertIn("先把源文本按语义翻译成自然日语", prompt)
        self.assertIn("不能仅因字符是“高”“考”就输出“こうこう”", prompt)
        self.assertIn("自然日语质量规则", prompt)
        self.assertIn("不要逐词硬译中文量词", prompt)
        self.assertIn("不要译成“いちにんのおんな”", prompt)
        self.assertIn("单元测试用例", prompt)
        self.assertIn("たんたいてすとけーす", prompt)
        self.assertIn("不要机械翻译成“いえ”", prompt)
        self.assertNotIn("Convert all Chinese characters to hiragana equivalents", prompt)
        self.assertNotIn("corresponding hiragana readings", prompt)
        self.assertNotIn("中国語 (ちゅうごくご)", prompt)

    def test_stale_translation_error_does_not_notify_result_callback(self) -> None:
        service = TranslationService(AppSettings())
        ctx = service._ensure_group(1)
        ctx.current_request_id = 2
        service._build_client = lambda: FailingClient()
        results = []

        service._execute(
            TranslationRequest(
                group_id=1,
                request_id=1,
                ocr_text="old text",
                source_language="English",
                target_language="中文",
            ),
            results.append,
        )
        service.shutdown()

        self.assertEqual(results, [])

    def test_invalidate_group_requests_marks_existing_request_stale(self) -> None:
        service = TranslationService(AppSettings())
        request = TranslationRequest(
            group_id=1,
            request_id=1,
            ocr_text="old text",
            source_language="English",
            target_language="中文",
        )
        ctx = service._ensure_group(1)
        ctx.current_request_id = 1

        service.invalidate_group_requests(1, reason="ocr changed")

        self.assertTrue(service._is_stale_request(request, ctx))
        service.shutdown()

    def test_execute_injects_confirmed_feedback_memory_when_trigger_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feedback_store = FeedbackStore(tmp)
            record = feedback_store.add_feedback(
                group_id=1,
                source_language="English",
                target_language="中文",
                ocr_text="The gaokao starts soon",
                translation_text="高中考试很快开始",
            )
            feedback_store.approve_feedback(
                record.id,
                trigger="gaokao",
                rule="gaokao 应译为中国高考/大学入学考试，不要译成普通高中考试。",
                preferred_translation="高考很快开始",
            )
            service = TranslationService(AppSettings(), feedback_store=feedback_store)
            fake_agent = FakeMemoryAgent()
            service._ensure_agent = lambda group_id, source, target: fake_agent
            request = TranslationRequest(
                group_id=1,
                request_id=1,
                ocr_text="The gaokao starts soon",
                source_language="English",
                target_language="中文",
            )
            service._ensure_group(1).current_request_id = 1
            results = []

            service._execute(request, results.append)
            service.shutdown()

            self.assertEqual(results[0].text, "the national college entrance exam starts soon")
            self.assertEqual(fake_agent.text, "The gaokao starts soon")
            self.assertIsNotNone(fake_agent.memory_hints)
            self.assertIn("gaokao 应译为", fake_agent.memory_hints[0])


class TranslationServiceAgentPerGroupTests(unittest.TestCase):
    """Verify each group gets its own independent Agent instance."""

    def test_different_groups_create_different_agents(self) -> None:
        """Two groups should have separate Agent instances."""
        service = TranslationService(AppSettings())
        
        with patch.object(TranslationAgent, 'digest_rules'), \
             patch.object(TranslationAgent, '__init__', return_value=None):
            agent1 = service._ensure_agent(1, "English", "中文")
            agent2 = service._ensure_agent(2, "日本語", "English")
        
        service.shutdown()
        
        self.assertIsNot(agent1, agent2, "Different groups should have different agents")
        self.assertEqual(len(service._agents), 2)
        self.assertIn(1, service._agents)
        self.assertIn(2, service._agents)

    def test_same_group_reuses_same_agent(self) -> None:
        """Multiple calls for same group should reuse the same Agent."""
        service = TranslationService(AppSettings())
        
        with patch.object(TranslationAgent, 'digest_rules'), \
             patch.object(TranslationAgent, '__init__', return_value=None):
            agent1 = service._ensure_agent(1, "English", "中文")
            agent2 = service._ensure_agent(1, "English", "中文")
        
        service.shutdown()
        
        self.assertIs(agent1, agent2, "Same group should reuse same agent")
        self.assertEqual(len(service._agents), 1)

    def test_reset_agent_only_clears_specified_group(self) -> None:
        """reset_agent(group_id) should only clear that group's agent."""
        service = TranslationService(AppSettings())
        
        with patch.object(TranslationAgent, 'digest_rules'), \
             patch.object(TranslationAgent, '__init__', return_value=None):
            agent1 = service._ensure_agent(1, "English", "中文")
            agent2 = service._ensure_agent(2, "日本語", "English")
        
        service.reset_agent(1)
        
        self.assertNotIn(1, service._agents, "Group 1 agent should be cleared")
        self.assertIn(2, service._agents, "Group 2 agent should remain")
        self.assertIs(service._agents[2], agent2, "Group 2 agent should be unchanged")
        
        service.shutdown()

    def test_ensure_agent_passes_fast_and_thinking_model_configs(self) -> None:
        """Agent should receive separate fast and thinking model configs."""
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.fast_model = "deepseek-v4-flash"
        settings.ai.thinking_model = "deepseek-v4-pro"
        service = TranslationService(settings)

        with patch.object(TranslationAgent, 'digest_rules'), \
             patch.object(TranslationAgent, '__init__', return_value=None) as init_mock:
            service._ensure_agent(1, "中文", "日本語")

        service.shutdown()

        fast_config, thinking_config = init_mock.call_args.args
        self.assertEqual(fast_config.model, "deepseek-v4-flash")
        self.assertEqual(thinking_config.model, "deepseek-v4-pro")
        self.assertEqual(fast_config.base_url, "https://api.example.test/v1")
        self.assertEqual(thinking_config.api_key, "key")

    def test_ensure_agent_restores_saved_session_without_digest(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "confirmed"},
        ]
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.fast_model = "deepseek-v4-flash"
        settings.ai.thinking_model = "deepseek-v4-pro"
        store = FakeSessionStore(loaded_messages=messages)
        service = TranslationService(settings, session_store=store)

        with patch.object(TranslationAgent, 'digest_rules') as digest_mock, \
             patch.object(TranslationAgent, 'restore_messages') as restore_mock:
            service._ensure_agent(1, "中文", "日本語")

        service.shutdown()

        digest_mock.assert_not_called()
        restore_mock.assert_called_once_with(messages)
        self.assertEqual(store.load_calls[0].fast_model, "deepseek-v4-flash")
        self.assertEqual(store.load_calls[0].thinking_model, "deepseek-v4-pro")

    def test_ensure_agent_saves_session_after_new_digest(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "confirmed"},
        ]
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.fast_model = "deepseek-v4-flash"
        settings.ai.thinking_model = "deepseek-v4-pro"
        store = FakeSessionStore()
        service = TranslationService(settings, session_store=store)

        def fake_digest(agent, prompt):
            agent.messages = messages

        with patch.object(TranslationAgent, 'digest_rules', autospec=True, side_effect=fake_digest):
            service._ensure_agent(1, "中文", "日本語")

        service.shutdown()

        self.assertEqual(len(store.save_calls), 1)
        self.assertEqual(store.save_calls[0][1], messages)

    def test_different_groups_translate_concurrently_with_api_limit(self) -> None:
        service = TranslationService(
            AppSettings(),
            max_workers=3,
            max_concurrent_api_calls=2,
        )
        gate = threading.Event()
        two_active = threading.Event()
        all_done = threading.Event()
        counter_lock = threading.Lock()
        state = {"active": 0, "max_active": 0}
        results = []

        class BlockingAgent:
            messages = []

            def translate(self, text: str, memory_hints=None) -> str:
                with counter_lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                    if state["active"] >= 2:
                        two_active.set()
                try:
                    if not gate.wait(timeout=2.0):
                        raise TranslationError("test gate timed out")
                    return f"translated: {text}"
                finally:
                    with counter_lock:
                        state["active"] -= 1

        fake_agent = BlockingAgent()
        service._ensure_agent = lambda group_id, source, target: fake_agent

        def on_result(result) -> None:
            with counter_lock:
                results.append(result)
                if len(results) == 3:
                    all_done.set()

        try:
            for group_id in (1, 2, 3):
                service.request_translation(
                    group_id,
                    f"text {group_id}",
                    on_result,
                )

            self.assertTrue(two_active.wait(timeout=1.0), "different groups did not overlap")
            self.assertEqual(state["max_active"], 2)
            gate.set()
            self.assertTrue(all_done.wait(timeout=2.0))
            self.assertEqual(len(results), 3)
            self.assertTrue(all(result.error is None for result in results))
        finally:
            gate.set()
            service.shutdown()


if __name__ == "__main__":
    unittest.main()
