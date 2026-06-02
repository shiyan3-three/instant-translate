"""Tests for top-level application bootstrap objects."""

from __future__ import annotations

import unittest

from app.app_context import ApplicationContext
from app.main import build_application_context


class BuildApplicationContextTests(unittest.TestCase):
    """Verify the default desktop runtime state."""

    def test_build_application_context_uses_expected_defaults(self) -> None:
        context = build_application_context()

        self.assertIsInstance(context, ApplicationContext)
        self.assertEqual(context.hotkeys.create_selection, "Ctrl+Shift+Z")
        self.assertEqual(context.hotkeys.toggle_edit_mode, "Ctrl+Shift+X")
        self.assertEqual(context.active_group_count, 0)
        self.assertEqual(context.settings.prompt.compiled_prompt_path, "compiled-prompt.md")
