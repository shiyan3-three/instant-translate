"""Tests for the first-stage desktop shell widgets."""

from __future__ import annotations

import unittest

from app.app_context import ApplicationContext
from app.gui.main_window import MainWindow
from app.gui.settings_window import SettingsWindow
from app.gui.tray_icon import TrayIconController
from tests.test_support import ensure_qapplication


class MainWindowTests(unittest.TestCase):
    """Verify visible shell defaults."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_main_window_shows_title_and_hotkey_summary(self) -> None:
        window = MainWindow(ApplicationContext())

        self.assertEqual(window.windowTitle(), "Instant Translate")
        self.assertIn("Ctrl+Shift+Z", window.hotkey_summary_label.text())
        self.assertEqual(window.open_settings_button.text(), "打开设置")


class SettingsWindowTests(unittest.TestCase):
    """Verify editable settings fields are seeded from application settings."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_settings_window_binds_ai_and_prompt_fields(self) -> None:
        context = ApplicationContext()
        context.settings.ai.base_url = "https://example.test/v1"
        context.settings.ai.model = "gpt-test"
        context.settings.prompt.constraints_text = "保持简洁"
        context.settings.prompt.knowledge_reference_paths = ["C:/docs/glossary.md"]
        window = SettingsWindow(context.settings)

        self.assertEqual(window.base_url_input.text(), "https://example.test/v1")
        self.assertEqual(window.model_input.text(), "gpt-test")
        self.assertEqual(window.constraints_input.toPlainText(), "保持简洁")
        self.assertEqual(window.knowledge_reference_list.count(), 1)


class TrayIconControllerTests(unittest.TestCase):
    """Verify tray actions stay aligned with the current MVP shell."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_tray_menu_exposes_primary_actions(self) -> None:
        context = ApplicationContext()
        main_window = MainWindow(context)
        settings_window = SettingsWindow(context.settings)
        controller = TrayIconController(self.app, main_window, settings_window)

        action_texts = [action.text() for action in controller.menu.actions() if action.text()]

        self.assertIn("显示主窗口", action_texts)
        self.assertIn("打开设置", action_texts)
        self.assertIn("退出", action_texts)
