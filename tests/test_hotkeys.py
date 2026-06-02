"""Tests for Windows global hotkey parsing."""

from __future__ import annotations

import unittest

from app.hotkeys import GlobalHotkeyService


class GlobalHotkeyServiceTests(unittest.TestCase):
    """Cover shortcut parsing without depending on a real desktop event loop."""

    def test_parse_shortcut_supports_default_create_selection_binding(self) -> None:
        modifiers, virtual_key = GlobalHotkeyService.parse_shortcut("Ctrl+Shift+Z")

        self.assertEqual(modifiers, 0x0006)
        self.assertEqual(virtual_key, ord("Z"))

    def test_parse_shortcut_rejects_unknown_tokens(self) -> None:
        with self.assertRaises(ValueError):
            GlobalHotkeyService.parse_shortcut("Ctrl+Magic+Z")
