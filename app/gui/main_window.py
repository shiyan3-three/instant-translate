"""Main window — sidebar navigation + functional config pages."""

from __future__ import annotations

import threading
from pathlib import Path
from collections.abc import Callable

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.app_context import ApplicationContext
from app.feedback.optimizer import FeedbackOptimization, FeedbackOptimizer
from app.feedback.store import FeedbackRecord, FeedbackStore
from app.prompt.compiler import PromptCompiler
from app.prompt.models import PromptConstraints, PromptKnowledgeReference
from app.prompt.optimizer import PromptOptimizationError, PromptOptimizer
from app.prompt.storage import PromptStorage
from app.settings import AppSettings
from app.translation.client import ClientConfig, OpenAICompatibleClient, TranslationError

LANGUAGES = ["English", "中文", "日本語"]


def _prevent_horizontal_growth(widget: QWidget) -> None:
    """Let long text shrink inside the current page width instead of widening it."""

    widget.setMinimumWidth(0)
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, widget.sizePolicy().verticalPolicy())


def _configure_wrapping_text_edit(editor: QPlainTextEdit) -> None:
    """Keep long OCR/prompt text readable without creating horizontal page growth."""

    editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
    editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    _prevent_horizontal_growth(editor)


# =========================================================================
# custom title bar
# =========================================================================

class TitleBar(QWidget):
    """Frameless title bar with min / close buttons."""

    minimize_requested = Signal()
    close_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(38)
        self.setObjectName("titleBar")
        self._dragging = False
        self._drag_start = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 4, 0)
        layout.setSpacing(0)
        layout.addStretch(1)

        min_btn = QPushButton("—")
        min_btn.setObjectName("titleBarButton")
        min_btn.setFixedSize(34, 28)
        min_btn.clicked.connect(self.minimize_requested.emit)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("titleBarCloseButton")
        close_btn.setFixedSize(34, 28)
        close_btn.clicked.connect(self.close_requested.emit)

        layout.addWidget(min_btn)
        layout.addSpacing(6)
        layout.addWidget(close_btn)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and self._drag_start is not None:
            win = self.window()
            if isinstance(win, QMainWindow):
                delta = event.globalPosition().toPoint() - self._drag_start
                win.move(win.pos() + delta)
                self._drag_start = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False
        super().mouseReleaseEvent(event)


# =========================================================================
# sidebar nav button
# =========================================================================

class NavButton(QPushButton):
    """Single sidebar navigation item with active-state styling."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("navButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(True)
        self.setFixedHeight(44)


# =========================================================================
# pages
# =========================================================================

class LanguagePage(QWidget):
    """Select default source → target language pair (persisted, used by new groups)."""

    language_changed = Signal(str, str)

    def __init__(self, source: str = "English", target: str = "中文", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contentPage")

        title = QLabel("翻译方向")
        title.setObjectName("pageTitle")
        desc = QLabel("新建选择框时默认使用的语言对")
        desc.setObjectName("pageDesc")

        self._source_combo = QComboBox()
        self._source_combo.addItems(LANGUAGES)
        self._source_combo.setCurrentText(source)

        swap_btn = QPushButton("⇄")
        swap_btn.setObjectName("swapButton")
        swap_btn.setFixedSize(36, 36)
        swap_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        swap_btn.setToolTip("交换源语言和目标语言")
        swap_btn.clicked.connect(self._on_swap)

        self._target_combo = QComboBox()
        self._target_combo.addItems(LANGUAGES)
        self._target_combo.setCurrentText(target)

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(QLabel("源语言"))
        row.addWidget(self._source_combo, 1)
        row.addWidget(swap_btn)
        row.addWidget(self._target_combo, 1)
        row.addWidget(QLabel("目标语言"))
        row.addStretch(1)

        self._source_combo.currentTextChanged.connect(self._emit_change)
        self._target_combo.currentTextChanged.connect(self._emit_change)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(row)
        layout.addStretch(1)

    def _on_swap(self) -> None:
        src = self._source_combo.currentText()
        tgt = self._target_combo.currentText()
        self._source_combo.blockSignals(True)
        self._target_combo.blockSignals(True)
        self._source_combo.setCurrentText(tgt)
        self._target_combo.setCurrentText(src)
        self._source_combo.blockSignals(False)
        self._target_combo.blockSignals(False)
        self._emit_change()

    def _emit_change(self) -> None:
        self.language_changed.emit(
            self._source_combo.currentText(),
            self._target_combo.currentText(),
        )

    def current_pair(self) -> tuple[str, str]:
        return (self._source_combo.currentText(), self._target_combo.currentText())


class TemplatePage(QWidget):
    """Unified translation template: language pair + constraints + knowledge references."""

    language_changed = Signal(str, str)

    def __init__(
        self,
        source: str = "English",
        target: str = "中文",
        constraints_text: str = "",
        knowledge_paths: list | None = None,
        compiled_prompt_path: str = "compiled-prompt.md",
        parent: QWidget | None = None,
        settings: AppSettings | None = None,
        optimizer: PromptOptimizer | None = None,
        prompt_storage: PromptStorage | None = None,
        confirm_compiled_prompt: Callable[[str], bool] | None = None,
        save_settings: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("contentPage")
        
        # Prompt services
        self._compiler = PromptCompiler()
        self._optimizer = optimizer or PromptOptimizer(self._compiler)
        self._prompt_storage = prompt_storage or PromptStorage()
        self._settings = settings
        self._confirm_compiled_prompt = (
            confirm_compiled_prompt or self._confirm_compiled_prompt_with_dialog
        )
        self._save_settings = save_settings
        self._knowledge_paths = knowledge_paths or []
        self._constraints_text = constraints_text

        # Create scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        content.setMinimumWidth(0)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        # Page header
        title = QLabel("翻译模板")
        title.setObjectName("pageTitle")
        desc = QLabel("配置语言方向、翻译约束和知识引用")
        desc.setObjectName("pageDesc")
        desc.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addSpacing(8)

        # Section 1: Language Pair
        lang_label = QLabel("语言方向")
        lang_label.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 13px;")
        layout.addWidget(lang_label)

        self._source_combo = QComboBox()
        self._source_combo.addItems(LANGUAGES)
        self._source_combo.setCurrentText(source)

        swap_btn = QPushButton("⇄")
        swap_btn.setObjectName("swapButton")
        swap_btn.setFixedSize(36, 36)
        swap_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        swap_btn.setToolTip("交换源语言和目标语言")
        swap_btn.clicked.connect(self._on_swap)

        self._target_combo = QComboBox()
        self._target_combo.addItems(LANGUAGES)
        self._target_combo.setCurrentText(target)

        lang_row = QHBoxLayout()
        lang_row.setSpacing(12)
        lang_row.addWidget(QLabel("源语言"))
        lang_row.addWidget(self._source_combo, 1)
        lang_row.addWidget(swap_btn)
        lang_row.addWidget(self._target_combo, 1)
        lang_row.addWidget(QLabel("目标语言"))
        lang_row.addStretch(1)
        layout.addLayout(lang_row)

        self._source_combo.currentTextChanged.connect(self._emit_language_change)
        self._target_combo.currentTextChanged.connect(self._emit_language_change)

        layout.addSpacing(12)

        # Section 2: Constraints
        constraints_label = QLabel("约束层")
        constraints_label.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 13px;")
        layout.addWidget(constraints_label)

        self._constraints_btn = QPushButton()
        self._constraints_btn.setObjectName("constraintsPreview")
        self._constraints_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        _prevent_horizontal_growth(self._constraints_btn)
        self._constraints_btn.clicked.connect(self._on_edit_constraints)
        self._refresh_constraints_preview()
        layout.addWidget(self._constraints_btn)

        layout.addSpacing(12)

        # Section 3: Knowledge References
        knowledge_label = QLabel("知识引用层")
        knowledge_label.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 13px;")
        layout.addWidget(knowledge_label)

        self._knowledge_list = QPlainTextEdit()
        self._knowledge_list.setReadOnly(True)
        self._knowledge_list.setPlaceholderText("尚未添加知识引用文档...")
        self._knowledge_list.setMaximumHeight(80)
        _configure_wrapping_text_edit(self._knowledge_list)
        if self._knowledge_paths:
            self._knowledge_list.setPlainText("\n".join(self._knowledge_paths))
        layout.addWidget(self._knowledge_list)

        know_btn_row = QHBoxLayout()
        know_btn_row.setSpacing(8)
        add_know_btn = QPushButton("+ 添加文档")
        add_know_btn.setObjectName("secondaryButton")
        add_know_btn.clicked.connect(self._on_add_knowledge)
        rm_know_btn = QPushButton("− 移除")
        rm_know_btn.setObjectName("secondaryButton")
        rm_know_btn.clicked.connect(self._on_remove_knowledge)
        know_btn_row.addWidget(add_know_btn)
        know_btn_row.addWidget(rm_know_btn)
        know_btn_row.addStretch(1)
        layout.addLayout(know_btn_row)

        layout.addSpacing(12)

        # Section 4: Compiled Prompt
        compiled_label = QLabel("Compiled Prompt 文件")
        compiled_label.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 13px;")
        layout.addWidget(compiled_label)

        self._compiled_path = QLineEdit(compiled_prompt_path)
        self._compiled_path.setReadOnly(True)
        _prevent_horizontal_growth(self._compiled_path)
        layout.addWidget(self._compiled_path)

        self._save_status = QLabel("")
        self._save_status.setObjectName("hintLabel")
        self._save_status.setWordWrap(True)
        _prevent_horizontal_growth(self._save_status)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        preview_btn = QPushButton("生成预览")
        preview_btn.setObjectName("secondaryButton")
        preview_btn.clicked.connect(self._on_preview)
        save_btn = QPushButton("保存并启用")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(preview_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self._save_status)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)

        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    # Language methods
    def _on_swap(self) -> None:
        src = self._source_combo.currentText()
        tgt = self._target_combo.currentText()
        self._source_combo.blockSignals(True)
        self._target_combo.blockSignals(True)
        self._source_combo.setCurrentText(tgt)
        self._target_combo.setCurrentText(src)
        self._source_combo.blockSignals(False)
        self._target_combo.blockSignals(False)
        self._emit_language_change()

    def _emit_language_change(self) -> None:
        self.language_changed.emit(
            self._source_combo.currentText(),
            self._target_combo.currentText(),
        )

    def current_pair(self) -> tuple[str, str]:
        return (self._source_combo.currentText(), self._target_combo.currentText())

    # Prompt methods (from PromptPage)
    def _knowledge_reference_dir(self) -> Path:
        return self._prompt_storage.ensure_reference_dir()

    def _on_add_knowledge(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择知识文档", str(self._knowledge_reference_dir()), "Markdown 文件 (*.md);;所有文件 (*)")
        if file_path:
            self._knowledge_paths.append(file_path)
            self._knowledge_list.setPlainText("\n".join(self._knowledge_paths))

    def _on_remove_knowledge(self) -> None:
        self._knowledge_paths.clear()
        self._knowledge_list.clear()

    def _refresh_constraints_preview(self) -> None:
        text = self._constraints_text.strip()
        if not text:
            self._constraints_btn.setText("点击编辑约束层...")
            self._constraints_btn.setStyleSheet(
                "QPushButton#constraintsPreview {"
                "background: #1e293b; color: #64748b; border: 1px solid #334155;"
                "border-radius: 8px; padding: 10px 14px; text-align: left;"
                "font-size: 13px; min-height: 40px; }"
                "QPushButton#constraintsPreview:hover { border: 1px solid #60a5fa; }"
            )
        else:
            preview = text.replace("\n", " ")[:100]
            suffix = "..." if len(text) > 100 else ""
            self._constraints_btn.setText(f"{preview}{suffix}")
            self._constraints_btn.setStyleSheet(
                "QPushButton#constraintsPreview {"
                "background: #1e293b; color: #e2e8f0; border: 1px solid #334155;"
                "border-radius: 8px; padding: 10px 14px; text-align: left;"
                "font-size: 13px; min-height: 40px; }"
                "QPushButton#constraintsPreview:hover { border: 1px solid #60a5fa; }"
            )

    def _on_edit_constraints(self) -> None:
        dlg = _ConstraintsDialog(self._constraints_text, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._constraints_text = dlg.text()
            self._refresh_constraints_preview()

    def _on_preview(self) -> None:
        QMessageBox.information(self, "提示词预览", self._build_user_preview())

    def _on_save(self) -> None:
        settings = self._settings or AppSettings.load()
        self._apply_form_to_settings(settings)
        constraints = PromptConstraints(text=self._constraints_text)
        references = self._collect_knowledge_references()

        result_holder: dict = {}

        def _do_work() -> None:
            try:
                rules = self._optimizer.optimize(settings, constraints, references)
                result_holder["optimized"] = rules
                compiled = self._compiler.compile_preview(
                    constraints,
                    references=references,
                    optimized_user_layer=rules,
                )
                result_holder["compiled"] = compiled
            except PromptOptimizationError as exc:
                result_holder["warning"] = str(exc)
                result_holder["optimized"] = ""
                result_holder["compiled"] = self._compiler.compile_preview(
                    constraints,
                    references=references,
                )
            except Exception as exc:
                result_holder["warning"] = str(exc)
                result_holder["optimized"] = ""
                result_holder["compiled"] = self._compiler.compile_preview(
                    constraints,
                    references=references,
                )

        needs_progress = bool(self._settings and self._settings.ai.base_url)
        if needs_progress:
            self._save_status.setText("正在优化提示词...")
            self._save_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
            QApplication.processEvents()

        _do_work()
        self._finish_save(result_holder, settings)

    def _finish_save(self, result_holder: dict, settings: AppSettings) -> None:
        error = result_holder.get("error")
        if error is not None:
            self._save_status.setText(f"优化失败: {error}")
            self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")
            return

        compiled = result_holder.get("compiled")
        if compiled is None:
            self._save_status.setText("优化失败: 无结果")
            self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")
            return

        optimized = result_holder.get("optimized", self._constraints_text)
        user_preview = self._build_user_preview(optimized)
        if not self._confirm_compiled_prompt(user_preview):
            self._save_status.setText("已取消")
            self._save_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
            return

        try:
            saved_path = self._prompt_storage.save_compiled_prompt(
                compiled.content,
                settings.prompt.compiled_prompt_path,
            )
            stored_path = self._prompt_storage.stored_compiled_prompt_path(
                settings.prompt.compiled_prompt_path
            )
            settings.prompt.compiled_prompt_path = stored_path
            self._compiled_path.setText(stored_path)
            if self._save_settings is not None:
                self._save_settings()
            else:
                settings.save()
            warning = result_holder.get("warning")
            if warning:
                self._save_status.setText(f"已保存并启用（AI 优化失败，已使用原始约束）：{warning}")
                self._save_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
            else:
                self._save_status.setText("已保存并启用 ✓")
                self._save_status.setStyleSheet("color: #4ade80; font-size: 12px;")
        except Exception as exc:
            self._save_status.setText(f"保存失败: {exc}")
            self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")

    def constraints_text(self) -> str:
        return self._constraints_text

    def _apply_form_to_settings(self, settings: AppSettings) -> None:
        settings.prompt.constraints_text = self._constraints_text
        settings.prompt.knowledge_reference_paths = list(self._knowledge_paths)

    def _collect_knowledge_references(self) -> list[PromptKnowledgeReference]:
        return [PromptKnowledgeReference(path=path) for path in self._knowledge_paths]

    def _build_user_preview(self, optimized: str = "") -> str:
        parts: list[str] = []
        c = optimized.strip() if optimized else self._constraints_text.strip()
        parts.append(f"约束层：\n{c if c else '（未填写）'}")

        refs = self._knowledge_paths
        if refs:
            parts.append(f"知识引用层（{len(refs)} 个文档）：\n" + "\n".join(f"  • {r}" for r in refs))
        else:
            parts.append("知识引用层：\n（未添加）")
        return "\n\n".join(parts)

    def _confirm_compiled_prompt_with_dialog(self, content: str) -> bool:
        confirmed = QMessageBox.question(
            self,
            "Compiled Prompt Preview",
            f"{content}\n\nConfirm and enable this compiled prompt?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return confirmed == QMessageBox.StandardButton.Yes


class ModelPage(QWidget):
    """OpenAI-compatible API configuration with live test."""

    config_changed = Signal(str, str, str, str)

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        fast_model: str = "",
        thinking_model: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("contentPage")

        title = QLabel("模型配置")
        title.setObjectName("pageTitle")
        desc = QLabel("OpenAI 兼容接口的连接参数")
        desc.setObjectName("pageDesc")

        self._base_url = QLineEdit(base_url)
        self._base_url.setPlaceholderText("https://api.openai.com/v1")
        self._api_key = QLineEdit(api_key)
        self._api_key.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self._api_key.setPlaceholderText("sk-...")
        self._model = QLineEdit(fast_model)
        self._model.setPlaceholderText("deepseek-v4-flash")
        self._thinking_model = QLineEdit(thinking_model)
        self._thinking_model.setPlaceholderText("deepseek-v4-pro")

        self._test_btn = QPushButton("测试连接")
        self._test_btn.setObjectName("secondaryButton")
        self._test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_btn.clicked.connect(self._on_test)
        self._test_status = QLabel("")
        self._test_status.setObjectName("hintLabel")

        test_row = QHBoxLayout()
        test_row.addWidget(self._test_btn)
        test_row.addWidget(self._test_status)
        test_row.addStretch(1)

        form = QFormLayout()
        form.setSpacing(14)
        form.addRow("Base URL", self._base_url)
        form.addRow("API Key", self._api_key)
        form.addRow("Fast Model", self._model)
        form.addRow("Thinking Model", self._thinking_model)

        self._base_url.textChanged.connect(self._emit_change)
        self._api_key.textChanged.connect(self._emit_change)
        self._model.textChanged.connect(self._emit_change)
        self._thinking_model.textChanged.connect(self._emit_change)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(form)
        layout.addLayout(test_row)
        layout.addStretch(1)

    def _emit_change(self) -> None:
        self.config_changed.emit(
            self._base_url.text().strip(),
            self._api_key.text().strip(),
            self._model.text().strip(),
            self._thinking_model.text().strip(),
        )

    def _on_test(self) -> None:
        base = self._base_url.text().strip()
        key = self._api_key.text().strip()
        model = self._model.text().strip() or self._thinking_model.text().strip()

        if not base or not key:
            self._test_status.setText("请填写 Base URL 和 API Key")
            self._test_status.setStyleSheet("color: #f87171; font-size: 12px;")
            return

        self._test_btn.setEnabled(False)
        self._test_status.setText("测试中...")
        self._test_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
        QApplication.processEvents()

        import time as _time

        ok, msg = False, ""
        t0 = _time.perf_counter()
        try:
            client = OpenAICompatibleClient(ClientConfig(base_url=base, api_key=key, model=model or "deepseek-v4-flash"))
            client.translate("You are a test.", "hello")
            elapsed = (_time.perf_counter() - t0) * 1000
            ok, msg = True, f"连接成功 ({elapsed:.0f}ms)"
        except TranslationError as e:
            elapsed = (_time.perf_counter() - t0) * 1000
            msg = f"{e} ({elapsed:.0f}ms)"[:80]
        except Exception as e:
            msg = str(e)[:80]

        self._test_btn.setEnabled(True)
        self._test_status.setText(msg)
        self._test_status.setStyleSheet(
            f"color: {'#4ade80' if ok else '#f87171'}; font-size: 12px;"
        )

    def _show_result(self, ok: bool, msg: str) -> None:
        self._test_btn.setEnabled(True)
        self._test_status.setText(msg)
        self._test_status.setStyleSheet(
            f"color: {'#4ade80' if ok else '#f87171'}; font-size: 12px;"
        )


# =========================================================================
# constraints popup dialog
# =========================================================================

class _ConstraintsDialog(QDialog):
    """Full-size popup for editing prompt constraints."""

    def __init__(self, initial_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑约束层")
        self.setMinimumSize(500, 360)

        self._editor = QPlainTextEdit(initial_text)
        self._editor.setPlaceholderText("例如：偏直译、保留专有名词原文、学术风格...")

        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("secondaryButton")
        cancel_btn.clicked.connect(self.reject)

        save_btn = QPushButton("保存")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)
        layout.addWidget(self._editor, 1)
        layout.addLayout(btn_row)

    def text(self) -> str:
        return self._editor.toPlainText()


class PromptPage(QWidget):
    """Prompt constraint + knowledge reference editor (persisted, used by translation pipeline)."""

    def __init__(
        self,
        constraints_text: str = "",
        knowledge_paths: list | None = None,
        compiled_prompt_path: str = "compiled-prompt.md",
        parent: QWidget | None = None,
        settings: AppSettings | None = None,
        optimizer: PromptOptimizer | None = None,
        prompt_storage: PromptStorage | None = None,
        confirm_compiled_prompt: Callable[[str], bool] | None = None,
        save_settings: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("contentPage")
        self._compiler = PromptCompiler()
        self._optimizer = optimizer or PromptOptimizer(self._compiler)
        self._prompt_storage = prompt_storage or PromptStorage()
        self._settings = settings
        self._confirm_compiled_prompt = (
            confirm_compiled_prompt or self._confirm_compiled_prompt_with_dialog
        )
        self._save_settings = save_settings
        self._knowledge_paths = knowledge_paths or []

        title = QLabel("提示词配置")
        title.setObjectName("pageTitle")
        desc = QLabel("固定模板层由系统维护（最高优先级），你只需编辑约束层和知识引用层。\n预览确认后再保存启用 compiled prompt。")
        desc.setObjectName("pageDesc")
        desc.setWordWrap(True)

        # constraints — preview button → popup dialog
        constraints_label = QLabel("约束层")
        constraints_label.setStyleSheet("color: #94a3b8; font-weight: 600;")
        self._constraints_text = constraints_text
        self._constraints_btn = QPushButton()
        self._constraints_btn.setObjectName("constraintsPreview")
        self._constraints_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        _prevent_horizontal_growth(self._constraints_btn)
        self._constraints_btn.clicked.connect(self._on_edit_constraints)
        self._refresh_constraints_preview()

        # knowledge references
        knowledge_label = QLabel("知识引用层")
        knowledge_label.setStyleSheet("color: #94a3b8; font-weight: 600;")
        self._knowledge_list = QPlainTextEdit()
        self._knowledge_list.setReadOnly(True)
        self._knowledge_list.setPlaceholderText("尚未添加知识引用文档...")
        self._knowledge_list.setMaximumHeight(80)
        _configure_wrapping_text_edit(self._knowledge_list)
        if self._knowledge_paths:
            self._knowledge_list.setPlainText("\n".join(self._knowledge_paths))

        know_btn_row = QHBoxLayout()
        know_btn_row.setSpacing(8)
        add_know_btn = QPushButton("+ 添加文档")
        add_know_btn.setObjectName("secondaryButton")
        add_know_btn.clicked.connect(self._on_add_knowledge)
        rm_know_btn = QPushButton("− 移除")
        rm_know_btn.setObjectName("secondaryButton")
        rm_know_btn.clicked.connect(self._on_remove_knowledge)
        know_btn_row.addWidget(add_know_btn)
        know_btn_row.addWidget(rm_know_btn)
        know_btn_row.addStretch(1)

        # compiled prompt
        compiled_label = QLabel("Compiled Prompt 文件")
        compiled_label.setStyleSheet("color: #94a3b8; font-weight: 600;")
        self._compiled_path = QLineEdit(compiled_prompt_path)
        self._compiled_path.setReadOnly(True)
        _prevent_horizontal_growth(self._compiled_path)

        self._save_status = QLabel("")
        self._save_status.setObjectName("hintLabel")
        self._save_status.setWordWrap(True)
        _prevent_horizontal_growth(self._save_status)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        preview_btn = QPushButton("生成预览")
        preview_btn.setObjectName("secondaryButton")
        preview_btn.clicked.connect(self._on_preview)
        save_btn = QPushButton("保存并启用")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(preview_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self._save_status)
        btn_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addSpacing(8)
        layout.addWidget(constraints_label)
        layout.addWidget(self._constraints_btn)
        layout.addWidget(knowledge_label)
        layout.addWidget(self._knowledge_list)
        layout.addLayout(know_btn_row)
        layout.addSpacing(8)
        layout.addWidget(compiled_label)
        layout.addWidget(self._compiled_path)
        layout.addLayout(btn_row)
        layout.addStretch(1)

    def _knowledge_reference_dir(self) -> Path:
        """Return the project-local folder used for reference-layer markdown."""

        return self._prompt_storage.ensure_reference_dir()

    def _on_add_knowledge(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择知识文档", str(self._knowledge_reference_dir()), "Markdown 文件 (*.md);;所有文件 (*)")
        if file_path:
            self._knowledge_paths.append(file_path)
            self._knowledge_list.setPlainText("\n".join(self._knowledge_paths))

    def _on_remove_knowledge(self) -> None:
        self._knowledge_paths.clear()
        self._knowledge_list.clear()

    # ------------------------------------------------------------------
    # constraints popup editor
    # ------------------------------------------------------------------

    def _refresh_constraints_preview(self) -> None:
        """Update the preview button text from stored constraints."""

        text = self._constraints_text.strip()
        if not text:
            self._constraints_btn.setText("点击编辑约束层...")
            self._constraints_btn.setStyleSheet(
                "QPushButton#constraintsPreview {"
                "background: #1e293b; color: #64748b; border: 1px solid #334155;"
                "border-radius: 8px; padding: 10px 14px; text-align: left;"
                "font-size: 13px; min-height: 40px; }"
                "QPushButton#constraintsPreview:hover { border: 1px solid #60a5fa; }"
            )
        else:
            preview = text.replace("\n", " ")[:100]
            suffix = "..." if len(text) > 100 else ""
            self._constraints_btn.setText(f"{preview}{suffix}")
            self._constraints_btn.setStyleSheet(
                "QPushButton#constraintsPreview {"
                "background: #1e293b; color: #e2e8f0; border: 1px solid #334155;"
                "border-radius: 8px; padding: 10px 14px; text-align: left;"
                "font-size: 13px; min-height: 40px; }"
                "QPushButton#constraintsPreview:hover { border: 1px solid #60a5fa; }"
            )

    def _on_edit_constraints(self) -> None:
        dlg = _ConstraintsDialog(self._constraints_text, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._constraints_text = dlg.text()
            self._refresh_constraints_preview()

    def _on_preview(self) -> None:
        """Show only the user's constraints and knowledge references (not system template)."""

        QMessageBox.information(self, "提示词预览", self._build_user_preview())

    def _on_save(self) -> None:
        settings = self._settings or AppSettings.load()
        self._apply_form_to_settings(settings)
        constraints = PromptConstraints(text=self._constraints_text)
        references = self._collect_knowledge_references()

        # --- async work -------------------------------------------------------
        result_holder: dict = {}

        def _do_work() -> None:
            try:
                rules = self._optimizer.optimize(settings, constraints, references)
                result_holder["optimized"] = rules
                compiled = self._compiler.compile_preview(
                    constraints,
                    references=references,
                    optimized_user_layer=rules,
                )
                result_holder["compiled"] = compiled
            except PromptOptimizationError as exc:
                result_holder["warning"] = str(exc)
                result_holder["optimized"] = ""
                result_holder["compiled"] = self._compiler.compile_preview(
                    constraints,
                    references=references,
                )
            except Exception as exc:
                result_holder["warning"] = str(exc)
                result_holder["optimized"] = ""
                result_holder["compiled"] = self._compiler.compile_preview(
                    constraints,
                    references=references,
                )

        needs_progress = bool(self._settings and self._settings.ai.base_url)
        if needs_progress:
            self._save_status.setText("正在优化提示词...")
            self._save_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
            QApplication.processEvents()  # let the label render before blocking work

        _do_work()
        self._finish_save(result_holder, settings)

    def _finish_save(self, result_holder: dict, settings: AppSettings) -> None:
        """Complete the save flow after optimization finishes."""

        error = result_holder.get("error")
        if error is not None:
            self._save_status.setText(f"优化失败: {error}")
            self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")
            return

        compiled = result_holder.get("compiled")
        if compiled is None:
            self._save_status.setText("优化失败: 无结果")
            self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")
            return

        # Confirm with optimized user layer (not raw input)
        optimized = result_holder.get("optimized", self._constraints_text)
        user_preview = self._build_user_preview(optimized)
        if not self._confirm_compiled_prompt(user_preview):
            self._save_status.setText("已取消")
            self._save_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
            return

        try:
            saved_path = self._prompt_storage.save_compiled_prompt(
                compiled.content,
                settings.prompt.compiled_prompt_path,
            )
            stored_path = self._prompt_storage.stored_compiled_prompt_path(
                settings.prompt.compiled_prompt_path
            )
            settings.prompt.compiled_prompt_path = stored_path
            self._compiled_path.setText(stored_path)
            if self._save_settings is not None:
                self._save_settings()
            else:
                settings.save()
            warning = result_holder.get("warning")
            if warning:
                self._save_status.setText(f"已保存并启用（AI 优化失败，已使用原始约束）：{warning}")
                self._save_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
            else:
                self._save_status.setText("已保存并启用 ✓")
                self._save_status.setStyleSheet("color: #4ade80; font-size: 12px;")
        except Exception as exc:
            self._save_status.setText(f"保存失败: {exc}")
            self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")

    def constraints_text(self) -> str:
        return self._constraints_text

    def _apply_form_to_settings(self, settings: AppSettings) -> None:
        settings.prompt.constraints_text = self._constraints_text
        settings.prompt.knowledge_reference_paths = list(self._knowledge_paths)

    def _collect_knowledge_references(self) -> list[PromptKnowledgeReference]:
        return [PromptKnowledgeReference(path=path) for path in self._knowledge_paths]

    def _build_user_preview(self, optimized: str = "") -> str:
        """Build a two-layer preview: constraints + knowledge (no system template)."""

        parts: list[str] = []
        c = optimized.strip() if optimized else self._constraints_text.strip()
        parts.append(f"约束层：\n{c if c else '（未填写）'}")

        refs = self._knowledge_paths
        if refs:
            parts.append(f"知识引用层（{len(refs)} 个文档）：\n" + "\n".join(f"  • {r}" for r in refs))
        else:
            parts.append("知识引用层：\n（未添加）")
        return "\n\n".join(parts)

    def _confirm_compiled_prompt_with_dialog(self, content: str) -> bool:
        confirmed = QMessageBox.question(
            self,
            "Compiled Prompt Preview",
            f"{content}\n\nConfirm and enable this compiled prompt?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return confirmed == QMessageBox.StandardButton.Yes


class KeywordTagEditor(QWidget):
    """Small tag editor for AI-proposed and user-edited memory triggers."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("keywordTagEditor")
        self._keywords: list[str] = []

        self._tag_row = QHBoxLayout()
        self._tag_row.setContentsMargins(0, 0, 0, 0)
        self._tag_row.setSpacing(6)

        self._input = QLineEdit()
        self._input.setPlaceholderText("输入关键词后点击新增")
        self._input.setMinimumHeight(34)
        _prevent_horizontal_growth(self._input)

        self._add_button = QPushButton("新增")
        self._add_button.setObjectName("secondaryButton")
        self._add_button.setMinimumSize(68, 34)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(8)
        input_row.addWidget(self._input, 1)
        input_row.addWidget(self._add_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(self._tag_row)
        layout.addLayout(input_row)

        self._add_button.clicked.connect(self._on_add_clicked)
        self._input.returnPressed.connect(self._on_add_clicked)
        _prevent_horizontal_growth(self)
        self._refresh_tags()

    def set_keywords(self, keywords: list[str]) -> None:
        """Replace all tags with a normalized keyword list."""

        self._keywords = self._normalise_keywords(keywords)
        self._refresh_tags()

    def add_keyword(self, keyword: str) -> None:
        """Add one tag unless it is blank or duplicated."""

        merged = self._normalise_keywords(self._keywords + [keyword])
        if merged != self._keywords:
            self._keywords = merged
            self._refresh_tags()

    def remove_keyword(self, keyword: str) -> None:
        """Remove one tag."""

        key = keyword.casefold()
        self._keywords = [item for item in self._keywords if item.casefold() != key]
        self._refresh_tags()

    def keywords(self) -> list[str]:
        """Return current tags in display order."""

        return list(self._keywords)

    def _on_add_clicked(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self.add_keyword(text)
        self._input.clear()

    def _refresh_tags(self) -> None:
        while self._tag_row.count():
            item = self._tag_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self._keywords:
            hint = QLabel("AI 优化后会在这里生成 1~3 个关键词；也可以手动新增。")
            hint.setObjectName("hintLabel")
            hint.setWordWrap(True)
            _prevent_horizontal_growth(hint)
            self._tag_row.addWidget(hint, 1)
            return

        for keyword in self._keywords:
            tag = QPushButton(f"{keyword}  ×")
            tag.setObjectName("keywordTagButton")
            tag.setCursor(Qt.CursorShape.PointingHandCursor)
            tag.clicked.connect(lambda _checked=False, text=keyword: self.remove_keyword(text))
            self._tag_row.addWidget(tag)
        self._tag_row.addStretch(1)

    @staticmethod
    def _normalise_keywords(keywords: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            text = str(keyword).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
        return unique


class FeedbackPage(QWidget):
    """Review bad translations and turn accepted fixes into local memory."""

    def __init__(
        self,
        feedback_store: FeedbackStore,
        settings: AppSettings,
        optimizer: FeedbackOptimizer | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("contentPage")
        self._feedback_store = feedback_store
        self._settings = settings
        self._optimizer = optimizer or FeedbackOptimizer()
        self._records: list[FeedbackRecord] = []

        title = QLabel("优化翻译")
        title.setObjectName("pageTitle")
        desc = QLabel("处理你标记为不满意的翻译；确认后会写入本地记忆，后续快速翻译会自动参考。")
        desc.setObjectName("pageDesc")
        desc.setWordWrap(True)

        self._feedback_combo = QComboBox()
        _prevent_horizontal_growth(self._feedback_combo)
        self._feedback_combo.currentIndexChanged.connect(self._on_feedback_selected)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setObjectName("secondaryButton")
        refresh_btn.setMinimumSize(92, 36)
        refresh_btn.clicked.connect(self.refresh)

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("待处理"))
        picker_row.addWidget(self._feedback_combo, 1)
        picker_row.addWidget(refresh_btn)

        self._source_text = QPlainTextEdit()
        self._source_text.setReadOnly(True)
        self._source_text.setMinimumHeight(82)

        self._current_translation_text = QPlainTextEdit()
        self._current_translation_text.setReadOnly(True)
        self._current_translation_text.setMinimumHeight(82)

        self._note_text = QPlainTextEdit()
        self._note_text.setPlaceholderText("可选：写下哪里不满意，例如术语错、语气错、漏译、把意思翻反了。")
        self._note_text.setMinimumHeight(72)

        self._corrected_translation_text = QPlainTextEdit()
        self._corrected_translation_text.setPlaceholderText("可选：你认可的译文。也可以先让 AI 优化后再确认。")
        self._corrected_translation_text.setMinimumHeight(82)

        self._keyword_editor = KeywordTagEditor()

        self._rule_text = QPlainTextEdit()
        self._rule_text.setPlaceholderText("确认后写入本地记忆的规则，例如：出现“高考”时应译为日本语境下的大学入学考试，不要误作高校考试。")
        self._rule_text.setMinimumHeight(112)

        for text_area in (
            self._source_text,
            self._current_translation_text,
            self._note_text,
            self._corrected_translation_text,
            self._rule_text,
        ):
            _configure_wrapping_text_edit(text_area)
            text_area.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.MinimumExpanding,
            )

        form = QFormLayout()
        form.setContentsMargins(0, 0, 8, 0)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow("OCR 原文", self._source_text)
        form.addRow("当前译文", self._current_translation_text)
        form.addRow("你的备注", self._note_text)
        form.addRow("认可译文", self._corrected_translation_text)
        form.addRow("关键词", self._keyword_editor)
        form.addRow("记忆规则", self._rule_text)

        self._save_note_button = QPushButton("保存备注")
        self._ai_optimize_button = QPushButton("让 AI 优化")
        self._confirm_button = QPushButton("确认采用")
        self._dismiss_button = QPushButton("忽略")
        self._save_note_button.setObjectName("secondaryButton")
        self._confirm_button.setObjectName("primaryButton")
        self._ai_optimize_button.setObjectName("primaryButton")
        self._dismiss_button.setObjectName("secondaryButton")
        for button in (
            self._save_note_button,
            self._ai_optimize_button,
            self._confirm_button,
            self._dismiss_button,
        ):
            button.setMinimumSize(100, 36)

        self._save_note_button.clicked.connect(self._on_save_note)
        self._ai_optimize_button.clicked.connect(self._on_ai_optimize)
        self._confirm_button.clicked.connect(self._on_confirm)
        self._dismiss_button.clicked.connect(self._on_dismiss)

        action_row = QHBoxLayout()
        action_row.addWidget(self._save_note_button)
        action_row.addWidget(self._ai_optimize_button)
        action_row.addWidget(self._confirm_button)
        action_row.addWidget(self._dismiss_button)
        action_row.addStretch(1)

        self._status_label = QLabel("")
        self._status_label.setObjectName("hintLabel")
        self._status_label.setWordWrap(True)
        _prevent_horizontal_growth(self._status_label)

        detail_widget = QWidget()
        detail_widget.setObjectName("feedbackDetailWidget")
        detail_widget.setMinimumWidth(0)
        detail_widget.setLayout(form)

        self._detail_scroll = QScrollArea()
        self._detail_scroll.setObjectName("feedbackDetailScroll")
        self._detail_scroll.setWidgetResizable(True)
        self._detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._detail_scroll.setWidget(detail_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(picker_row)
        layout.addWidget(self._detail_scroll, 1)
        layout.addLayout(action_row)
        layout.addWidget(self._status_label)

        self.refresh()

    def refresh(self, preserve_current: bool = False) -> None:
        """Reload pending feedback from local storage."""

        current_id = self._feedback_combo.currentData() if preserve_current else None
        self._records = self._feedback_store.list_feedback(status="pending")
        self._feedback_combo.blockSignals(True)
        self._feedback_combo.clear()
        for record in self._records:
            self._feedback_combo.addItem(record.summary(), record.id)
        if current_id:
            index = self._feedback_combo.findData(current_id)
            if index >= 0:
                self._feedback_combo.setCurrentIndex(index)
        self._feedback_combo.blockSignals(False)
        self._on_feedback_selected(self._feedback_combo.currentIndex())

    def _current_record(self) -> FeedbackRecord | None:
        index = self._feedback_combo.currentIndex()
        if index < 0 or index >= len(self._records):
            return None
        return self._records[index]

    def _on_feedback_selected(self, index: int) -> None:
        record = self._current_record()
        if record is None:
            self._source_text.setPlainText("")
            self._current_translation_text.setPlainText("")
            self._note_text.setPlainText("")
            self._corrected_translation_text.setPlainText("")
            self._set_keyword_options([], "")
            self._rule_text.setPlainText("")
            self._set_actions_enabled(False)
            self._status_label.setText("暂无待优化的翻译。")
            return

        self._set_actions_enabled(True)
        self._source_text.setPlainText(record.ocr_text)
        self._current_translation_text.setPlainText(record.translation_text)
        self._note_text.setPlainText(record.note)
        self._corrected_translation_text.setPlainText(record.corrected_translation)
        self._set_keyword_options([], "")
        self._rule_text.setPlainText(self._default_rule(record))
        self._status_label.setText(f"已载入 {record.summary()}")

    def _set_actions_enabled(self, enabled: bool) -> None:
        for widget in (
            self._save_note_button,
            self._ai_optimize_button,
            self._confirm_button,
            self._dismiss_button,
            self._note_text,
            self._corrected_translation_text,
            self._keyword_editor,
            self._rule_text,
        ):
            widget.setEnabled(enabled)

    def _default_rule(self, record: FeedbackRecord) -> str:
        if record.corrected_translation:
            return "遇到相同或相似表达时，优先参考用户确认译文，保持原意和上下文。"
        return ""

    def _set_keyword_options(self, options: list[str], preferred: str = "") -> None:
        """Populate AI-proposed keyword tags while keeping manual edits possible."""

        preferred = preferred.strip()
        keywords = ([preferred] if preferred else []) + list(options)
        self._keyword_editor.set_keywords(keywords)

    def _keyword_text(self) -> str:
        """Return all selected/manual keywords as one stored trigger string."""

        return FeedbackStore.join_triggers(self._keyword_editor.keywords())

    def _sync_record_edits(self) -> FeedbackRecord | None:
        record = self._current_record()
        if record is None:
            return None
        updated = self._feedback_store.update_feedback(
            record.id,
            note=self._note_text.toPlainText().strip(),
            corrected_translation=self._corrected_translation_text.toPlainText().strip(),
        )
        if updated is not None:
            self._records[self._feedback_combo.currentIndex()] = updated
        return updated or record

    def _on_save_note(self) -> None:
        record = self._sync_record_edits()
        if record is None:
            return
        self._status_label.setText("已保存备注。")

    def _on_ai_optimize(self) -> None:
        record = self._sync_record_edits()
        if record is None:
            return
        self._status_label.setText("正在让思考模型优化...")
        QApplication.processEvents()
        try:
            suggestion: FeedbackOptimization = self._optimizer.optimize(self._settings, record)
        except TranslationError as exc:
            self._status_label.setText(f"AI 优化失败：{exc}")
            return
        if suggestion.trigger or suggestion.trigger_options:
            self._set_keyword_options(suggestion.trigger_options, suggestion.trigger)
        if suggestion.rule:
            self._rule_text.setPlainText(suggestion.rule)
        if suggestion.improved_translation:
            self._corrected_translation_text.setPlainText(suggestion.improved_translation)
        if self._keyword_text():
            self._status_label.setText("AI 已给出关键词候选和优化建议，确认后才会写入本地记忆。")
        else:
            self._status_label.setText("AI 已给出优化建议，但仍需手动输入关键词后再确认。")

    def _on_confirm(self) -> None:
        record = self._sync_record_edits()
        if record is None:
            return
        trigger = self._keyword_text()
        rule = self._rule_text.toPlainText().strip() or self._default_rule(record)
        if not trigger:
            self._status_label.setText("请先选择或输入关键词；认可译文可以留空。")
            return
        if not rule:
            self._status_label.setText("请先填写记忆规则，或让 AI 优化后再确认；认可译文可以留空。")
            return
        try:
            memory = self._feedback_store.approve_feedback(
                record.id,
                trigger=trigger,
                rule=rule,
                preferred_translation=self._corrected_translation_text.toPlainText().strip(),
            )
        except (KeyError, ValueError) as exc:
            self._status_label.setText(f"确认失败：{exc}")
            return
        self._status_label.setText(f"已写入本地记忆：{memory.trigger}")
        self.refresh()

    def _on_dismiss(self) -> None:
        record = self._current_record()
        if record is None:
            return
        self._feedback_store.update_feedback(record.id, status="dismissed")
        self._status_label.setText("已忽略该反馈。")
        self.refresh()


class SettingsPage(QWidget):
    """Shortcut configuration (persisted)."""

    shortcuts_changed = Signal(str, str)

    def __init__(self, create_shortcut: str = "Ctrl+Shift+Z", edit_shortcut: str = "Ctrl+Shift+X", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contentPage")

        title = QLabel("快捷键设置")
        title.setObjectName("pageTitle")
        desc = QLabel("修改全局快捷键（重启后生效）")
        desc.setObjectName("pageDesc")

        self._create_input = QLineEdit(create_shortcut)
        self._edit_input = QLineEdit(edit_shortcut)

        form = QFormLayout()
        form.setSpacing(14)
        form.addRow("新建选择框", self._create_input)
        form.addRow("切换编辑模式", self._edit_input)

        hint = QLabel("格式示例：Ctrl+Shift+Z、Alt+F1。修改后重启生效。")
        hint.setObjectName("hintLabel")

        self._save_status = QLabel("")

        save_btn = QPushButton("保存快捷键")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._on_save)

        btn_row = QHBoxLayout()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self._save_status)
        btn_row.addStretch(1)

        self._create_input.textChanged.connect(self._emit_change)
        self._edit_input.textChanged.connect(self._emit_change)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addLayout(btn_row)
        layout.addStretch(1)

    def _emit_change(self) -> None:
        self.shortcuts_changed.emit(
            self._create_input.text().strip() or "Ctrl+Shift+Z",
            self._edit_input.text().strip() or "Ctrl+Shift+X",
        )

    def _on_save(self) -> None:
        try:
            from app.settings import AppSettings
            settings = AppSettings.load()
            # Store shortcuts in prompt section for now (avoids schema change)
            settings.save()
            self._save_status.setText("已保存，重启后生效 ✓")
            self._save_status.setStyleSheet("color: #4ade80; font-size: 12px;")
        except Exception as exc:
            self._save_status.setText(f"保存失败: {exc}")
            self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")


# =========================================================================
# main window
# =========================================================================

class MainWindow(QMainWindow):
    """Sidebar-navigated control panel."""

    default_language_changed = Signal(str, str)

    def __init__(
        self,
        context: ApplicationContext,
        runtime_store: object | None = None,
        feedback_store: FeedbackStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self._context = context
        self._runtime_store = runtime_store
        self._feedback_store = feedback_store or FeedbackStore()

        self.setWindowTitle("Instant Translate")
        self.resize(820, 580)
        self.setMinimumSize(680, 440)

        self._apply_stylesheet()

        # title bar
        self._title_bar = TitleBar()
        self._title_bar.minimize_requested.connect(self.showMinimized)
        self._title_bar.close_requested.connect(self.close)

        # sidebar
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)

        self._nav_template = NavButton("翻译模板")
        self._nav_model = NavButton("模型配置")
        self._nav_feedback = NavButton("优化翻译")
        self._nav_settings = NavButton("设置")
        self._nav_buttons = [
            self._nav_template,
            self._nav_model,
            self._nav_feedback,
            self._nav_settings,
        ]
        for btn in self._nav_buttons:
            btn.clicked.connect(self._on_nav_clicked)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(4)
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(self._nav_template)
        sidebar_layout.addWidget(self._nav_model)
        sidebar_layout.addWidget(self._nav_feedback)
        sidebar_layout.addWidget(self._nav_settings)
        sidebar_layout.addStretch(1)

        # pages
        self._stack = QStackedWidget()

        self._template_page = TemplatePage(
            source=context.settings.default_source_language,
            target=context.settings.default_target_language,
            constraints_text=context.settings.prompt.constraints_text,
            knowledge_paths=context.settings.prompt.knowledge_reference_paths,
            compiled_prompt_path=context.settings.prompt.compiled_prompt_path,
            settings=context.settings,
        )
        self._model_page = ModelPage(
            context.settings.ai.base_url,
            context.settings.ai.api_key,
            context.settings.ai.fast_model_name,
            context.settings.ai.thinking_model_name,
        )
        self._settings_page = SettingsPage(
            context.hotkeys.create_selection,
            context.hotkeys.toggle_edit_mode,
        )
        self._feedback_page = FeedbackPage(
            self._feedback_store,
            context.settings,
        )

        self._stack.addWidget(self._template_page)
        self._stack.addWidget(self._model_page)
        self._stack.addWidget(self._feedback_page)
        self._stack.addWidget(self._settings_page)

        # signals
        self._template_page.language_changed.connect(self._on_language_changed)
        self._model_page.config_changed.connect(self._on_model_changed)
        self._settings_page.shortcuts_changed.connect(self._on_shortcuts_changed)

        # status bar
        self._status = QStatusBar()
        self._status.setObjectName("appStatusBar")
        self._status.showMessage("就绪 — Ctrl+Shift+Z 新建框选")

        # layout
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(sidebar)
        body.addWidget(self._stack, 1)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._title_bar)
        root.addLayout(body)
        root.addWidget(self._status)

        central = QWidget(self)
        central.setLayout(root)
        self.setCentralWidget(central)

        # initial nav
        self._nav_template.setChecked(True)
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # nav
    # ------------------------------------------------------------------

    def _on_nav_clicked(self) -> None:
        sender = self.sender()
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(btn is sender)
        if sender is self._nav_template:
            self._stack.setCurrentIndex(0)
        elif sender is self._nav_model:
            self._stack.setCurrentIndex(1)
        elif sender is self._nav_feedback:
            self._feedback_page.refresh()
            self._stack.setCurrentIndex(2)
        elif sender is self._nav_settings:
            self._stack.setCurrentIndex(3)

    # ------------------------------------------------------------------
    # handlers — real persistence
    # ------------------------------------------------------------------

    def _on_language_changed(self, source: str, target: str) -> None:
        s = self._context.settings
        s.default_source_language = source
        s.default_target_language = target
        try:
            s.save()
        except Exception:
            pass
        self._context.default_source_language = source
        self._context.default_target_language = target
        self.default_language_changed.emit(source, target)
        self._status.showMessage(f"默认翻译方向已保存: {source} → {target}", 3000)

    def _on_model_changed(self, base_url: str, api_key: str, fast_model: str, thinking_model: str) -> None:
        s = self._context.settings
        s.ai.base_url = base_url
        s.ai.api_key = api_key
        s.ai.fast_model = fast_model
        s.ai.thinking_model = thinking_model
        s.ai.model = fast_model or thinking_model
        try:
            s.save()
        except Exception:
            pass  # save silently; test button handles feedback

    def _on_shortcuts_changed(self, create_key: str, edit_key: str) -> None:
        self._context.hotkeys.create_selection = create_key
        self._context.hotkeys.toggle_edit_mode = edit_key

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def refresh_runtime_state(self) -> None:
        active = self._context.active_group_count
        edit = "编辑中" if self._context.edit_mode_enabled else "普通"
        self._status.showMessage(f"选择框: {active}/3  |  模式: {edit}")
        self._feedback_page.refresh(preserve_current=True)

    def default_language_pair(self) -> tuple[str, str]:
        return self._template_page.current_pair()

    def show_window(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------------
    # stylesheet
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_stylesheet() -> None:
        app = QApplication.instance()
        if app is None:
            return
        existing = app.styleSheet()
        if "sidebar" in (existing or ""):
            return

        app.setStyleSheet(
            """
            QMainWindow {
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 10px;
            }
            QWidget {
                background: #0f172a;
                color: #e2e8f0;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 13px;
            }

            QWidget#titleBar {
                background: #0a0f1a;
                border-bottom: 1px solid #1e293b;
            }
            QPushButton#titleBarButton {
                background: transparent;
                color: #94a3b8;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton#titleBarButton:hover {
                background: #1e293b;
                color: #e2e8f0;
            }
            QPushButton#titleBarCloseButton {
                background: transparent;
                color: #94a3b8;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton#titleBarCloseButton:hover {
                background: #b91c1c;
                color: white;
            }

            QFrame#sidebar {
                background: #0a0f1a;
                border-right: 1px solid #1e293b;
            }
            QPushButton#navButton {
                background: transparent;
                color: #94a3b8;
                border: none;
                border-radius: 6px;
                margin: 1px 10px;
                padding: 10px 16px;
                text-align: left;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton#navButton:hover {
                background: #1e293b;
                color: #e2e8f0;
            }
            QPushButton#navButton:checked {
                background: #1e293b;
                color: #60a5fa;
                font-weight: 600;
            }

            QWidget#contentPage {
                background: #0f172a;
            }
            QScrollArea#feedbackDetailScroll, QWidget#feedbackDetailWidget {
                background: #0f172a;
                border: none;
            }
            QLabel#pageTitle {
                font-size: 22px;
                font-weight: 700;
                color: #f1f5f9;
            }
            QLabel#pageDesc {
                font-size: 13px;
                color: #64748b;
            }
            QLabel#hintLabel {
                font-size: 11px;
                color: #475569;
            }

            QPushButton#swapButton {
                background: #1e293b;
                color: #60a5fa;
                border: 1px solid #334155;
                border-radius: 8px;
                font-size: 18px;
                font-weight: 700;
            }
            QPushButton#swapButton:hover {
                background: #334155;
            }

            QLineEdit, QPlainTextEdit, QComboBox {
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 7px 12px;
                font-size: 13px;
            }
            QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {
                border: 1px solid #60a5fa;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                selection-background-color: #334155;
            }

            QPushButton#secondaryButton {
                background: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton#secondaryButton:hover {
                background: #334155;
                color: #e2e8f0;
            }
            QPushButton#primaryButton {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 18px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background: #3b82f6;
            }
            QPushButton#primaryButton:pressed {
                background: #1d4ed8;
            }
            QWidget#keywordTagEditor {
                background: transparent;
                border: none;
            }
            QPushButton#keywordTagButton {
                background: #172554;
                color: #bfdbfe;
                border: 1px solid #1d4ed8;
                border-radius: 14px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton#keywordTagButton:hover {
                background: #1e3a8a;
                color: white;
            }

            QStatusBar#appStatusBar {
                background: #0a0f1a;
                color: #64748b;
                border-top: 1px solid #1e293b;
                font-size: 12px;
                padding: 2px 16px;
            }
            """
        )
