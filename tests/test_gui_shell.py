"""Tests for the first-stage desktop shell widgets."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QMessageBox, QPlainTextEdit, QPushButton,
    QSizePolicy, QTabWidget, QScrollArea,
)

from app.app_context import ApplicationContext
from app.feedback.optimizer import FeedbackOptimization
from app.feedback.store import FeedbackStore
from app.gui.main_window import FeedbackPage, FeedbackOptimizationDialog, TitleBar
from app.gui.main_window import MainWindow
from app.gui.main_window import ModelPage
from app.gui.main_window import TemplatePage
from app.prompt.optimizer import PromptOptimizationError
from app.prompt.policy import ConstraintPolicyCompiler, OptimizedPrompt
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

    def test_settings_window_reference_preview_and_candidate_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            storage.save_compiled_prompt(
                "# Instant Translate Compiled Prompt\n\n"
                "## AI Optimization Layer\n"
                "数据库 -> [でえたべえす]\n\n"
                "## Knowledge Reference Layer\n"
                "No knowledge references.\n",
                "prompts/compiled-prompt.md",
            )
            reference_path = Path(tmp) / "terms.md"
            reference_path.write_text("## Glossary\nAPI -> [えーぴーあい]\n", encoding="utf-8")
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = "prompts/compiled-prompt.md"
            settings.prompt.knowledge_reference_paths = [str(reference_path)]
            window = SettingsWindow(settings)
            window._prompt_storage = storage

            preview = window._build_reference_preview()
            with patch(
                "app.gui.settings_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ):
                window._on_export_ai_reference_candidates()

            candidate_path = Path(tmp) / "prompts" / "references" / "ai-optimization-candidates.md"

        self.assertIn("API -> [えーぴーあい]", preview)
        listed = [
            window.knowledge_reference_list.item(i).text()
            for i in range(window.knowledge_reference_list.count())
        ]
        self.assertIn(str(candidate_path), listed)


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

            # _on_save now runs in a background thread; process events until done
            import time
            deadline = time.monotonic() + 5.0
            while not compiled_path.exists() and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.01)

            compiled = compiled_path.read_text(encoding="utf-8")
            policy_path = compiled_path.with_suffix(".policy.json")
            self.assertIn("Fixed Template Layer", compiled)
            self.assertIn("Keep database terms literal.", compiled)
            self.assertIn("Optimized glossary rules", compiled)
            self.assertTrue(policy_path.exists())
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

    def test_user_preview_is_chinese_and_uses_real_policy_and_reference_counts(self) -> None:
        page = TemplatePage(
            constraints_text="输出只能使用平假名。",
            knowledge_paths=["terms.md"],
            confirm_compiled_prompt=lambda content: True,
        )
        optimized = OptimizedPrompt(
            "Machine rule only.",
            user_summary="AI 会补充检查语义完整性。",
            change_items=[{"title": "避免漏译", "description": "检查否定和完成状态。"}],
        )
        policy = ConstraintPolicyCompiler.compile("输出只能使用平假名。").to_dict()
        preview = page._build_user_preview(
            optimized,
            policy=policy,
            machine_content="EXACT MACHINE PROMPT",
            reference_package={
                "version": 1, "entries": [{"source": "API"}],
                "style_guidance": ["natural"], "risk_notes": ["ambiguous"],
            },
        )
        readable = str(preview)
        self.assertIn("一、基本目标", readable)
        self.assertIn("输出只能使用平假名。", readable)
        self.assertIn("AI 会补充", readable)
        self.assertIn("限制输出字符范围", readable)
        self.assertIn("四、程序会强制检查", readable)
        self.assertIn("五、模型需要遵守", readable)
        self.assertIn("术语 1 条", readable)
        self.assertNotIn('"rules"', readable)
        self.assertIn("EXACT MACHINE PROMPT", preview.advanced_content)
        self.assertIn('"rules"', preview.advanced_content)

    def test_preview_without_ai_summary_or_references_is_explicit(self) -> None:
        page = TemplatePage(confirm_compiled_prompt=lambda content: True)
        preview = page._build_user_preview("Machine-only legacy response")
        self.assertIn("AI 未提供中文变更说明", str(preview))
        self.assertIn("具体术语读法未由用户引用层固定", str(preview))

    def test_default_confirmation_dialog_has_easy_and_advanced_tabs(self) -> None:
        page = TemplatePage()
        preview = page._build_user_preview(
            OptimizedPrompt("machine"),
            policy={"version": 1, "rules": []},
            machine_content="EXACT MACHINE",
        )
        captured = {}
        def fake_exec(dialog):
            tabs = dialog.findChild(QTabWidget)
            captured["tabs"] = [tabs.tabText(i) for i in range(tabs.count())]
            captured["texts"] = [
                editor.toPlainText() for editor in dialog.findChildren(QPlainTextEdit)
            ]
            captured["buttons"] = [button.text() for button in dialog.findChildren(QPushButton)]
            return 0
        with patch.object(QDialog, "exec", fake_exec):
            self.assertFalse(page._confirm_compiled_prompt_with_dialog(preview))
        self.assertEqual(captured["tabs"], ["易懂说明", "高级内容"])
        self.assertTrue(any("EXACT MACHINE" in text for text in captured["texts"]))
        self.assertIn("返回修改", captured["buttons"])
        self.assertIn("确认并启用", captured["buttons"])

    def test_policy_preview_separates_local_and_model_rules_and_explains_scopes(self) -> None:
        page = TemplatePage(confirm_compiled_prompt=lambda content: True)
        policy = {
            "version": 1,
            "rules": [
                {"type": "separator", "enforcement": "both", "params": {"scope": "global", "min_spaces": 2}},
                {"type": "term_wrapper", "enforcement": "both", "params": {
                    "selection_mode": "references_and_ascii", "left": "[", "right": "]"
                }},
                {"type": "model_instruction", "enforcement": "model", "params": {"text": "keep meaning"}},
            ],
        }
        readable = str(page._build_user_preview(policy=policy))
        local_section, model_section = readable.split("五、模型需要遵守", 1)
        self.assertIn("所有分词空格至少使用 2 个空格", local_section)
        self.assertIn("只强制用户引用术语和 ASCII 技术标识", local_section)
        self.assertNotIn("程序无法机械验证", local_section)
        self.assertIn("程序无法机械验证", model_section)

    def test_reference_preview_shows_parsed_terms_style_and_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "terms.md"
            reference_path.write_text(
                "## Glossary\n"
                "API -> [えーぴーあい]\n"
                "## Style\n"
                "- keep natural word order\n"
                "## Risk\n"
                "risk: triggers: API; only wrap API as a technical term\n",
                encoding="utf-8",
            )
            page = TemplatePage(
                knowledge_paths=[str(reference_path)],
                prompt_storage=PromptStorage(config_dir=Path(tmp)),
                confirm_compiled_prompt=lambda content: True,
                save_settings=lambda: None,
            )

            preview = page._build_reference_preview()

        self.assertIn("术语 1 条", preview)
        self.assertIn("API -> [えーぴーあい]", preview)
        self.assertIn("keep natural word order", preview)
        self.assertIn("only wrap API", preview)

    def test_export_ai_candidates_can_be_confirmed_into_knowledge_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            storage.save_compiled_prompt(
                "# Instant Translate Compiled Prompt\n\n"
                "## AI Optimization Layer\n"
                "| Source | Target |\n"
                "|--------|--------|\n"
                "| 软件 | [そふとうぇあ] |\n\n"
                "## Knowledge Reference Layer\n"
                "No knowledge references.\n",
                "prompts/compiled-prompt.md",
            )
            page = TemplatePage(
                compiled_prompt_path="prompts/compiled-prompt.md",
                prompt_storage=storage,
                confirm_compiled_prompt=lambda content: True,
                save_settings=lambda: None,
            )

            with patch(
                "app.gui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ):
                page._on_export_ai_reference_candidates()

            candidate_path = Path(tmp) / "prompts" / "references" / "ai-optimization-candidates.md"

        self.assertIn(str(candidate_path), page._knowledge_paths)
        self.assertIn("ai-optimization-candidates.md", page._knowledge_list.toPlainText())

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

            # _on_save now runs in a background thread; process events until done
            import time
            deadline = time.monotonic() + 5.0
            while not compiled_path.exists() and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.01)

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


class ModelPageTests(unittest.TestCase):
    """Verify model combos can be filled from /models results."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_fetch_models_populates_editable_combos(self) -> None:
        page = ModelPage(
            base_url="https://api.example.test/v1",
            api_key="key",
            fast_model="kept-fast",
            thinking_model="kept-thinking",
        )
        page._show_models_result(
            ["deepseek-v4-flash", "deepseek-v4-pro", "kept-fast"],
            "已拉取 3 个模型",
        )

        self.assertFalse(page._model.isEditable())
        self.assertGreaterEqual(page._model.count(), 3)
        self.assertEqual(page._model.currentText(), "kept-fast")
        self.assertEqual(page._thinking_model.currentText(), "kept-thinking")
        self.assertIn("deepseek-v4-pro", [page._model.itemText(i) for i in range(page._model.count())])
        self.assertIn("已拉取", page._test_status.text())
        self.assertTrue(page._models_loaded)

    def test_fetch_models_button_disabled_without_credentials(self) -> None:
        page = ModelPage()
        self.assertFalse(page._fetch_models_btn.isEnabled())
        self.assertFalse(page._test_btn.isEnabled())
        page._base_url.setText("https://api.example.test/v1")
        page._api_key.setText("key")
        self.assertTrue(page._fetch_models_btn.isEnabled())
        self.assertTrue(page._test_btn.isEnabled())


class FeedbackPageTests(unittest.TestCase):
    """Verify the optimize-translation page writes confirmed memory."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()
        def accept_ai(dialog):
            for _ in range(5):
                if dialog._stack.currentIndex() == 0 and not dialog.suggestion.improved_translation.strip():
                    dialog._skip_current()
                else:
                    dialog._use_ai()
            dialog._finish_review()
            return 1
        self._dialog_patch = patch.object(FeedbackOptimizationDialog, "exec", accept_ai)
        self._dialog_patch.start()
        self.addCleanup(self._dialog_patch.stop)

    def _wait_for_ai(self, page: FeedbackPage, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while page._ai_optimizing and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertFalse(page._ai_optimizing, "feedback AI optimization did not finish")

    def test_feedback_uses_compact_actions_and_separate_memory_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = FeedbackPage(FeedbackStore(tmp), AppSettings())

            self.assertIsInstance(page._action_layout, QHBoxLayout)
            self.assertIsInstance(page._feedback_tabs, QTabWidget)
            self.assertEqual(page._feedback_tabs.count(), 2)
            self.assertEqual(page._feedback_tabs.tabText(0), "译文修正")
            self.assertEqual(page._feedback_tabs.tabText(1), "长期记忆（可选）")
            self.assertEqual(page._accept_translation_button.text(), "仅采用译文")
            self.assertEqual(page._accept_memory_button.text(), "保存为长期记忆")
            self.assertLessEqual(page._accept_translation_button.maximumWidth(), 100)
            self.assertLessEqual(page._accept_memory_button.maximumWidth(), 132)
            self.assertLessEqual(page.sizeHint().width(), 440)
            self.assertEqual(
                page._detail_scroll.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            self.assertFalse(hasattr(page, "_problem_summary_text"))
            self.assertFalse(hasattr(page, "_memory_recommendation_label"))
            self.assertIs(page._optimizer._feedback_store, page._feedback_store)

    def test_empty_ai_candidates_never_clear_existing_user_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="原文", translation_text="wrong",
            )
            page = FeedbackPage(
                store, AppSettings(),
                optimizer=FakeFeedbackOptimizer(FeedbackOptimization(
                    improved_translation="", trigger_options=[], rule="",
                    problem_summary="仅有问题说明", memory_recommended=False,
                )),
            )
            page._corrected_translation_text.setPlainText("用户已有译文")
            page._keyword_editor.set_keywords(["用户已有关键词"])
            page._rule_text.setPlainText("用户已有规则")

            page._on_ai_optimize()
            self._wait_for_ai(page)

            self.assertEqual(page._corrected_translation_text.toPlainText(), "用户已有译文")
            self.assertEqual(page._keyword_editor.keywords(), ["用户已有关键词"])
            self.assertEqual(page._rule_text.toPlainText(), "用户已有规则")

    def test_review_dialog_reuses_shell_components_without_scroll_area(self) -> None:
        parent = FeedbackPage(FeedbackStore(tempfile.mkdtemp()), AppSettings())
        parent.resize(640, 480)
        dialog = FeedbackOptimizationDialog(
            FeedbackOptimization(
                problem_summary="主客体错误", improved_translation="correct",
                trigger_options=["系统", "队列"], rule="按技术语境处理。",
            ),
            parent,
        )

        self.assertIsInstance(dialog._title_bar, TitleBar)
        self.assertEqual(dialog._stack.count(), 5)
        self.assertEqual(len(dialog._nav_buttons), 5)
        self.assertEqual(dialog.findChildren(QScrollArea), [])
        self.assertLessEqual(dialog.width(), parent.width())
        self.assertLessEqual(dialog.height(), parent.height())
        dialog._nav_buttons[3].click()
        self.assertEqual(dialog._stack.currentIndex(), 3)
        dialog._ai_keyword_editor.remove_keyword("系统")
        self.assertEqual(dialog.ai_keywords(), ["队列"])

    def test_rejecting_review_dialog_leaves_user_fields_and_store_unchanged(self) -> None:
        class RejectDialog:
            decision = ""
            def __init__(self, suggestion, parent): pass
            def exec(self): return 0

        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="原文", translation_text="wrong",
            )
            page = FeedbackPage(
                store, AppSettings(),
                optimizer=FakeFeedbackOptimizer(FeedbackOptimization(
                    improved_translation="AI correct", trigger="系统",
                    trigger_options=["系统"], rule="AI rule",
                )),
                dialog_factory=RejectDialog,
            )
            page._corrected_translation_text.setPlainText("用户原草稿")
            page._keyword_editor.set_keywords(["用户词"])
            page._rule_text.setPlainText("用户规则")

            page._on_ai_optimize()
            self._wait_for_ai(page)

            self.assertEqual(page._corrected_translation_text.toPlainText(), "用户原草稿")
            self.assertEqual(page._keyword_editor.keywords(), ["用户词"])
            self.assertEqual(page._rule_text.toPlainText(), "用户规则")
            self.assertEqual(store.get_feedback(record.id).status, "pending")
            self.assertEqual(store.list_memory_rules(), [])

    def test_user_decision_requires_translation_and_preserves_existing_note(self) -> None:
        dialog = FeedbackOptimizationDialog(FeedbackOptimization(improved_translation="AI"))
        dialog._use_user()
        self.assertEqual(dialog.decision, "")
        self.assertIn("完整", dialog._error_label.text())
        dialog._user_editors[0].setPlainText("用户译文")
        dialog._user_editors[1].setPlainText("用户问题判断")
        dialog._user_editors[2].setPlainText("用户规则")
        dialog._user_editors[3].setPlainText("总体意见")
        dialog._user_keyword_editor.set_keywords(["用户词"])
        dialog._use_user()
        self.assertEqual(dialog._review_states[0], FeedbackOptimizationDialog.USED_USER)
        self.assertEqual(dialog.decision, "")
        self.assertEqual(dialog._stack.currentIndex(), 1)
        self.assertEqual(dialog._nav_buttons[0].property("reviewState"), "done")
        self.assertIn("优化译文", dialog._nav_buttons[0].text())
        self.assertEqual(dialog._skip_button.text(), "跳过")
        self.assertEqual(dialog._use_ai_button.text(), "采用优化")
        self.assertEqual(dialog._use_user_button.text(), "确认修改")
        self.assertEqual(dialog._finish_button.text(), "完成审查")
        for _ in range(4):
            dialog._skip_current()
        self.assertEqual(dialog._nav_buttons[1].property("reviewState"), "skipped")
        self.assertTrue(dialog._finish_button.isEnabled())
        dialog._finish_review()
        self.assertEqual(dialog.decision, FeedbackOptimizationDialog.COMPLETED)

    def test_dialog_rejects_empty_ai_translation_and_skips_empty_memory_candidates(self) -> None:
        dialog = FeedbackOptimizationDialog(FeedbackOptimization())
        dialog._use_ai()
        self.assertEqual(dialog._review_states[0], FeedbackOptimizationDialog.UNREVIEWED)
        self.assertIn("不能采用空译文", dialog._error_label.text())
        dialog._skip_current()
        dialog._select_page(2)
        dialog._use_ai()
        self.assertEqual(dialog._review_states[2], FeedbackOptimizationDialog.SKIPPED)
        dialog._select_page(3)
        dialog._use_ai()
        self.assertEqual(dialog._review_states[3], FeedbackOptimizationDialog.SKIPPED)

    def test_dialog_minimize_also_minimizes_owning_window(self) -> None:
        parent = FeedbackPage(FeedbackStore(tempfile.mkdtemp()), AppSettings())
        dialog = FeedbackOptimizationDialog(FeedbackOptimization(), parent)
        with patch.object(parent, "showMinimized") as owner_minimize, \
             patch.object(dialog, "showMinimized") as dialog_minimize:
            dialog._minimize_with_owner()
        owner_minimize.assert_called_once_with()
        dialog_minimize.assert_not_called()

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
            self._wait_for_ai(page)
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
            self.assertGreaterEqual(page._source_text.minimumHeight(), 60)
            self.assertLessEqual(page._source_text.maximumHeight(), 110)
            self.assertGreaterEqual(page._rule_text.minimumHeight(), 70)
            self.assertLessEqual(page._rule_text.maximumHeight(), 110)
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
            self._wait_for_ai(page)
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

    def test_ai_improved_translation_without_keyword_can_be_accepted_without_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            source = "邻居闹到很晚，小李被吵到了。"
            expected = "りさん は となりの ひと に おそく まで さわがれました"
            store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text=source,
                translation_text="りさん は となりの ひと を うるさく しました",
            )
            page = FeedbackPage(store, AppSettings(), optimizer=FakeFeedbackOptimizer(
                FeedbackOptimization(
                    trigger="",
                    trigger_options=[],
                    rule="",
                    improved_translation=expected,
                    problem_summary="原译文把被动受害关系翻错了。",
                    memory_recommended=False,
                )
            ))

            page._on_ai_optimize()
            self._wait_for_ai(page)

            self.assertEqual(page._keyword_editor.keywords(), [])
            self.assertEqual(page._corrected_translation_text.toPlainText(), expected)

            page._on_accept_translation()

            self.assertEqual(store.list_memory_rules(), [])
            accepted = store.list_feedback(status="accepted")
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0].corrected_translation, expected)

    def test_save_memory_without_keyword_is_rejected_even_with_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="一次性句子", translation_text="wrong",
            )
            page = FeedbackPage(store, AppSettings())
            page._corrected_translation_text.setPlainText("correct")
            page._rule_text.setPlainText("可复用规则")

            page._on_accept_with_memory()

            self.assertEqual(store.list_memory_rules(), [])
            self.assertEqual(store.get_feedback(record.id).status, "pending")
            self.assertEqual(
                page._status_label.text(),
                "保存长期记忆需要确认关键词和规则。",
            )

    def test_ai_memory_recommendation_never_saves_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="高考开始", translation_text="wrong",
            )
            page = FeedbackPage(
                store,
                AppSettings(),
                optimizer=FakeFeedbackOptimizer(
                    FeedbackOptimization(
                        trigger="高考",
                        trigger_options=["高考"],
                        rule="按大学入学考试语境翻译。",
                        improved_translation="correct",
                        problem_summary="考试类型错误。",
                        memory_recommended=True,
                    )
                ),
            )
            page._on_ai_optimize()
            self._wait_for_ai(page)

            self.assertEqual(store.list_memory_rules(), [])
            self.assertEqual(store.list_feedback(status="pending")[0].status, "pending")
            self.assertEqual(page._feedback_tabs.currentIndex(), 0)

    def test_accept_translation_reports_stale_overlay_without_losing_feedback_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=7, source_language="中文", target_language="日本語",
                ocr_text="旧句子", translation_text="wrong",
            )
            calls = []
            page = FeedbackPage(
                store,
                AppSettings(),
                translation_applier=lambda group_id, source, translation: (
                    calls.append((group_id, source, translation)) or False
                ),
            )
            page._corrected_translation_text.setPlainText("认可译文")

            page._on_accept_translation()

            self.assertEqual(calls, [(7, "旧句子", "认可译文")])
            self.assertEqual(store.get_feedback(record.id).status, "accepted")
            self.assertIn("原选区内容已变化", page._status_label.text())
            self.assertEqual(store.list_memory_rules(), [])

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

    def test_ai_optimization_runs_without_blocking_gui_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="高考马上开始",
                translation_text="高校の試験がまもなく始まる",
            )
            optimizer = BlockingFeedbackOptimizer(
                FeedbackOptimization(
                    trigger="高考",
                    trigger_options=["高考"],
                    rule="按中国大学入学考试语境翻译。",
                )
            )
            page = FeedbackPage(store, AppSettings(), optimizer=optimizer)

            started_at = time.perf_counter()
            page._on_ai_optimize()
            elapsed = time.perf_counter() - started_at

            self.assertLess(elapsed, 0.2)
            self.assertTrue(page._ai_optimizing)
            self.assertFalse(page._ai_optimize_button.isEnabled())

            optimizer.release.set()
            self._wait_for_ai(page)

            self.assertEqual(page._keyword_editor.keywords(), ["高考"])
            self.assertTrue(page._ai_optimize_button.isEnabled())

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


class BlockingFeedbackOptimizer(FakeFeedbackOptimizer):
    """Feedback optimizer that stays busy until the test releases it."""

    def __init__(self, result: FeedbackOptimization) -> None:
        super().__init__(result)
        self.release = threading.Event()

    def optimize(self, settings, record):
        self.calls.append((settings, record))
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("test optimizer release timed out")
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
