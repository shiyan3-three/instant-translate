"""Tests for top-level application bootstrap objects."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.app_context import ApplicationContext
from app.main import build_application_context
from app.settings import AppSettings


class BuildApplicationContextTests(unittest.TestCase):
    """Verify the default desktop runtime state."""

    def test_build_application_context_uses_expected_defaults(self) -> None:
        context = build_application_context()

        self.assertIsInstance(context, ApplicationContext)
        self.assertEqual(context.hotkeys.create_selection, "Ctrl+Shift+Z")
        self.assertEqual(context.hotkeys.toggle_edit_mode, "Ctrl+Shift+X")
        self.assertEqual(context.active_group_count, 0)
        self.assertTrue(
            context.settings.prompt.compiled_prompt_path.endswith("compiled-prompt.md"),
            f"compiled_prompt_path should end with 'compiled-prompt.md', got {context.settings.prompt.compiled_prompt_path!r}",
        )

    def test_settings_load_normalizes_prompt_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "ai": {},
                        "prompt": {
                            "compiled_prompt_path": "prompts/prompts/compiled-prompt.md"
                        },
                    }
                ),
                encoding="utf-8",
            )

            settings = AppSettings.load(settings_path)

        self.assertEqual(settings.prompt.compiled_prompt_path, "prompts/compiled-prompt.md")
