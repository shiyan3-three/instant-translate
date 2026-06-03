"""Tests for the first-stage desktop shell widgets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.app_context import ApplicationContext
from app.gui.main_window import MainWindow
from app.gui.main_window import PromptPage
from app.gui.settings_window import SettingsWindow
from app.gui.tray_icon import TrayIconController
from app.prompt.storage import PromptStorage
from app.settings import AppSettings
from tests.test_support import ensure_qapplication


class MainWindowTests(unittest.TestCase):
    """Verify visible shell defaults."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_main_window_has_sidebar_and_pages(self) -> None:
        window = MainWindow(ApplicationContext())

        self.assertEqual(window.windowTitle(), "Instant Translate")
        self.assertIsNotNone(window._nav_language)
        self.assertIsNotNone(window._nav_model)
        self.assertIsNotNone(window._nav_prompt)
        self.assertIsNotNone(window._nav_settings)
        self.assertEqual(window._stack.count(), 4)


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


class PromptPageTests(unittest.TestCase):
    """Verify the main-window prompt page enables compiled prompts."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_save_and_enable_writes_compiled_prompt_to_runtime_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compiled_path = Path(tmp) / "prompts" / "compiled-prompt.md"
            settings = AppSettings()
            settings.ai.base_url = "https://api.example.test/v1"
            settings.ai.api_key = "key"
            settings.ai.model = "model"
            settings.prompt.compiled_prompt_path = "prompts/compiled-prompt.md"
            optimizer = FakePromptOptimizer("Optimized glossary rules")
            page = PromptPage(
                constraints_text="Keep database terms literal.",
                knowledge_paths=[],
                compiled_prompt_path=settings.prompt.compiled_prompt_path,
                settings=settings,
                optimizer=optimizer,
                prompt_storage=PromptStorage(config_dir=Path(tmp)),
                confirm_compiled_prompt=lambda content: True,
                save_settings=lambda: None,
            )

            page._on_save()

            compiled = compiled_path.read_text(encoding="utf-8")
            self.assertIn("Fixed Template Layer", compiled)
            self.assertIn("Keep database terms literal.", compiled)
            self.assertIn("Optimized glossary rules", compiled)
            self.assertEqual(settings.prompt.constraints_text, "Keep database terms literal.")
            self.assertEqual(settings.prompt.compiled_prompt_path, "prompts/compiled-prompt.md")
            self.assertEqual(page._compiled_path.text(), "prompts/compiled-prompt.md")


class FakePromptOptimizer:
    """Prompt optimizer test double."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls = []

    def optimize(self, settings, constraints, references=None) -> str:
        self.calls.append((settings, constraints, references or []))
        return self.result


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
