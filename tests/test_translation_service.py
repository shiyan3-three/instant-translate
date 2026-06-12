"""Tests for translation-service prompt selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
