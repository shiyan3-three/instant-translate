"""Tests for translation-service prompt selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agent.agent import TranslationAgent
from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.settings import AppSettings
from app.translation.client import TranslationError
from app.translation.service import TranslationRequest, TranslationService


class FailingClient:
    """Client test double that always fails."""

    def translate(self, system_prompt: str, source_text: str) -> str:
        raise TranslationError("old request timed out")


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


if __name__ == "__main__":
    unittest.main()
