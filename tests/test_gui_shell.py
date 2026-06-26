"""Tests for the first-stage desktop shell widgets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy

from app.app_context import ApplicationContext
from app.feedback.optimizer import FeedbackOptimization
from app.feedback.store import FeedbackStore
from app.gui.main_window import FeedbackPage
from app.gui.main_window import MainWindow
from app.gui.main_window import TemplatePage
from app.prompt.optimizer import PromptOptimizationError
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
        self.assertIsNotNone(window._nav_template)
        self.assertIsNotNone(window._nav_model)
        self.assertIsNotNone(window._nav_feedback)
        self.assertIsNotNone(window._nav_settings)
        self.assertEqual(window._stack.count(), 4)

    def test_template_page_has_language_and_prompt_sections(self) -> None:
        window = MainWindow(ApplicationContext())
        template_page = window._template_page

        # Has language selection
        self.assertIsNotNone(template_page._source_combo)
        self.assertIsNotNone(template_page._target_combo)
        
        # Has constraints section
        self.assertIsNotNone(template_page._constraints_btn)
        
        # Has knowledge references section
        self.assertIsNotNone(template_page._knowledge_list)


class SettingsWindowTests(unittest.TestCase):
    """Verify editable settings fields are seeded from application settings."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_settings_window_binds_ai_and_prompt_fields(self) -> None:
        context = ApplicationContext()
        context.settings.ai.base_url = "https://example.test/v1"
        context.settings.ai.fast_model = "deepseek-v4-flash"
        context.settings.ai.thinking_model = "deepseek-v4-pro"
        context.settings.prompt.constraints_text = "保持简洁"
        context.settings.prompt.knowledge_reference_paths = ["C:/docs/glossary.md"]
        window = SettingsWindow(context.settings)

        self.assertEqual(window.base_url_input.text(), "https://example.test/v1")
        self.assertEqual(window.model_input.text(), "deepseek-v4-flash")
        self.assertEqual(window.thinking_model_input.text(), "deepseek-v4-pro")
        self.assertEqual(window.constraints_input.toPlainText(), "保持简洁")
        self.assertEqual(window.knowledge_reference_list.count(), 1)


    def test_knowledge_reference_dir_uses_prompt_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            window = SettingsWindow(AppSettings())
            window._prompt_storage = PromptStorage(config_dir=Path(tmp))

            reference_dir = window._knowledge_reference_dir()

            self.assertEqual(reference_dir, Path(tmp) / "prompts" / "references")
            self.assertTrue(reference_dir.exists())


class TemplatePageTests(unittest.TestCase):
    """Verify the template page enables compiled prompts."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_save_and_enable_writes_compiled_prompt_to_runtime_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compiled_path = Path(tmp) / "prompts" / "compiled-prompt.md"
            settings = AppSettings()
            settings.ai.base_url = "https://api.example.test/v1"
            settings.ai.api_key = "key"
            settings.ai.fast_model = "model"
            settings.ai.thinking_model = "model"
            settings.prompt.compiled_prompt_path = "prompts/compiled-prompt.md"
            optimizer = FakePromptOptimizer("Optimized glossary rules")
            page = TemplatePage(
                source="English",
                target="中文",
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


    def test_knowledge_reference_dir_uses_prompt_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = TemplatePage(
                prompt_storage=PromptStorage(config_dir=Path(tmp)),
                confirm_compiled_prompt=lambda content: True,
                save_settings=lambda: None,
            )

            reference_dir = page._knowledge_reference_dir()

            self.assertEqual(reference_dir, Path(tmp) / "prompts" / "references")
            self.assertTrue(reference_dir.exists())

    def test_long_template_content_does_not_force_horizontal_growth(self) -> None:
        long_constraints = "术语约束：" + "非常长的提示词内容" * 80
        page = TemplatePage(
            constraints_text=long_constraints,
            confirm_compiled_prompt=lambda content: True,
            save_settings=lambda: None,
        )

        self.assertEqual(
            page._constraints_btn.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        self.assertEqual(
            page._knowledge_list.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

    def test_save_falls_back_to_raw_prompt_when_ai_optimization_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compiled_path = Path(tmp) / "prompts" / "compiled-prompt.md"
            settings = AppSettings()
            settings.ai.base_url = "https://api.example.test/v1"
            settings.ai.api_key = "key"
            settings.ai.thinking_model = "model"
            settings.prompt.compiled_prompt_path = "prompts/compiled-prompt.md"
            page = TemplatePage(
                constraints_text="Keep API terms literal.",
                compiled_prompt_path=settings.prompt.compiled_prompt_path,
                settings=settings,
                optimizer=FailingPromptOptimizer("network timeout"),
                prompt_storage=PromptStorage(config_dir=Path(tmp)),
                confirm_compiled_prompt=lambda content: True,
                save_settings=lambda: None,
            )

            page._on_save()

            compiled = compiled_path.read_text(encoding="utf-8")
            self.assertIn("Keep API terms literal.", compiled)
            self.assertIn("已保存并启用", page._save_status.text())
            self.assertIn("AI 优化失败", page._save_status.text())


class FakePromptOptimizer:
    """Prompt optimizer test double."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls = []

    def optimize(self, settings, constraints, references=None) -> str:
        self.calls.append((settings, constraints, references or []))
        return self.result


class FailingPromptOptimizer:
    """Prompt optimizer test double that fails like the remote AI call."""

    def __init__(self, message: str) -> None:
        self.message = message

    def optimize(self, settings, constraints, references=None) -> str:
        raise PromptOptimizationError(self.message)


class FeedbackPageTests(unittest.TestCase):
    """Verify the optimize-translation page writes confirmed memory."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_ai_suggestion_can_be_confirmed_into_local_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="高考马上开始",
                translation_text="高校の試験がまもなく始まる",
            )
            page = FeedbackPage(
                store,
                AppSettings(),
                optimizer=FakeFeedbackOptimizer(
                    FeedbackOptimization(
                        trigger="高考",
                        rule="出现“高考”时，按中国大学入学考试语境翻译。",
                        improved_translation="大学入学共通テストがまもなく始まる",
                    )
                ),
            )

            page._note_text.setPlainText("原译文误解了考试类型。")
            page._on_ai_optimize()
            page._on_confirm()

            self.assertEqual(page._feedback_combo.count(), 0)
            rules = store.match_memory_rules(
                "明天高考",
                source_language="中文",
                target_language="日本語",
            )
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0].trigger, "高考")
            self.assertIn("大学入学考试", rules[0].rule)
            self.assertIn("大学入学共通テスト", rules[0].preferred_translation)

    def test_feedback_page_keeps_detail_fields_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="明天,跑完所有的单元测试用例。",
                translation_text="あした、すべてのゆにっとてすとようれいをうごかしおわる。",
            )
            page = FeedbackPage(store, AppSettings(), optimizer=FakeFeedbackOptimizer(
                FeedbackOptimization(trigger="单元测试", rule="保持技术术语自然。")
            ))

            self.assertIsNotNone(page._detail_scroll)
            self.assertGreaterEqual(page._source_text.minimumHeight(), 80)
            self.assertGreaterEqual(page._rule_text.minimumHeight(), 110)
            self.assertEqual(
                page._detail_scroll.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            self.assertEqual(
                page._current_translation_text.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )

    def test_ai_trigger_candidates_can_be_selected_or_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="明天跑完所有的单元测试用例。",
                translation_text="あした、すべてのゆにっとてすとようれいをうごかしおわる。",
            )
            page = FeedbackPage(store, AppSettings(), optimizer=FakeFeedbackOptimizer(
                FeedbackOptimization(
                    trigger="单元测试用例",
                    trigger_options=["单元测试用例", "跑完", "测试用例"],
                    rule="出现“单元测试用例”时，按软件测试语境自然翻译。",
                )
            ))

            page._on_ai_optimize()
            self.assertEqual(
                page._keyword_editor.keywords(),
                ["单元测试用例", "跑完", "测试用例"],
            )

            page._keyword_editor.set_keywords(["软件测试"])
            page._on_confirm()

            rules = store.match_memory_rules(
                "软件测试需要跑完",
                source_language="中文",
                target_language="日本語",
            )
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0].trigger, "软件测试")

    def test_confirmed_memory_matches_any_keyword_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="明天跑完所有的单元测试用例。",
                translation_text="あした、すべてのテストを実行し終える。",
            )
            page = FeedbackPage(store, AppSettings())

            page._set_keyword_options(["单元测试用例", "测试用例"], "跑完")
            page._rule_text.setPlainText("这些词出现时，按软件测试语境翻译。")
            page._on_confirm()

            rules = store.match_memory_rules(
                "测试用例已经更新",
                source_language="中文",
                target_language="日本語",
            )
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0].trigger, "跑完 / 单元测试用例 / 测试用例")

    def test_confirm_allows_empty_accepted_translation_when_rule_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="昨天把系统环境配置好了",
                translation_text="きのうシステム環境を設定した",
            )
            page = FeedbackPage(store, AppSettings())

            page._set_keyword_options(["系统环境"], "系统环境")
            page._rule_text.setPlainText("出现“系统环境”时，按软件部署语境翻译。")
            page._corrected_translation_text.setPlainText("")
            page._on_confirm()

            rules = store.match_memory_rules(
                "系统环境配置",
                source_language="中文",
                target_language="日本語",
            )
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0].preferred_translation, "")

    def test_main_window_refresh_runtime_state_updates_feedback_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            window = MainWindow(ApplicationContext(), feedback_store=store)
            self.assertEqual(window._feedback_page._feedback_combo.count(), 0)

            store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="高考马上开始",
                translation_text="高校の試験がまもなく始まる",
            )
            window.refresh_runtime_state()

            self.assertEqual(window._feedback_page._feedback_combo.count(), 1)


class FakeFeedbackOptimizer:
    """Feedback optimizer test double."""

    def __init__(self, result: FeedbackOptimization) -> None:
        self.result = result
        self.calls = []

    def optimize(self, settings, record):
        self.calls.append((settings, record))
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
