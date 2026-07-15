"""Main window — sidebar navigation + functional config pages."""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable

from PySide6.QtCore import QSize, Qt, Signal, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.app_context import ApplicationContext
from app.feedback.optimizer import FeedbackOptimization, FeedbackOptimizer
from app.feedback.learning import FeedbackLearningService
from app.feedback.store import (
    FeedbackRecord,
    FeedbackStorageUnavailable,
    FeedbackStore,
    MemoryRule,
)
from app.logger import get_debug_logger, get_logger
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


def _make_content_card(title: str = "") -> tuple[QFrame, QVBoxLayout]:
    """Build one mock-aligned content card and return its inner layout."""

    card = QFrame()
    card.setObjectName("contentCard")
    card.setMinimumWidth(0)
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(16, 16, 16, 16)
    card_layout.setSpacing(12)
    if title:
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        card_layout.addWidget(title_label)
    return card, card_layout


def _make_field_label(text: str) -> QLabel:
    """Return the muted compact field label used by the HTML mock."""

    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


# =========================================================================
# custom title bar
# =========================================================================

class TitleBar(QWidget):
    """Frameless title bar with brand, page name, min / close buttons."""

    minimize_requested = Signal()
    close_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(38)
        self.setObjectName("titleBar")
        self._dragging = False
        self._drag_start = None
        self._page_name = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 8, 0)
        layout.setSpacing(0)

        self._brand_dot = QLabel()
        self._brand_dot.setObjectName("titleBarDot")
        self._brand_dot.setFixedSize(8, 8)
        self._brand_label = QLabel("Instant Translate")
        self._brand_label.setObjectName("titleBarBrand")
        self._separator_label = QLabel("·")
        self._separator_label.setObjectName("titleBarSeparator")
        self._page_label = QLabel("")
        self._page_label.setObjectName("titleBarPage")
        layout.addWidget(self._brand_dot)
        layout.addSpacing(10)
        layout.addWidget(self._brand_label)
        layout.addSpacing(8)
        layout.addWidget(self._separator_label)
        layout.addSpacing(8)
        layout.addWidget(self._page_label)
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

    def set_page_name(self, page_name: str) -> None:
        """Update title bar brand line: Instant Translate · {page}."""

        self._page_name = (page_name or "").strip()
        self._page_label.setText(self._page_name)
        self._separator_label.setVisible(bool(self._page_name))
        self._page_label.setVisible(bool(self._page_name))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and self._drag_start is not None:
            win = self.window()
            if isinstance(win, (QMainWindow, QDialog)):
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

    def __init__(self, icon: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(f"{icon}    {text}", parent)
        self.setObjectName("navButton")
        self.label_text = text
        self.icon_text = icon
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(True)
        self.setFixedHeight(44)


class CompiledPromptReview(str):
    """String-compatible Chinese preview carrying exact machine artifacts."""

    def __new__(cls, user_content: str, machine_content: str = "", policy: dict | None = None):
        obj = str.__new__(cls, user_content)
        obj.machine_content = machine_content
        obj.policy = policy or {"version": 1, "rules": []}
        return obj

    @property
    def advanced_content(self) -> str:
        return (
            "【将保存的机器 Prompt】\n"
            f"{self.machine_content or '（尚未生成）'}\n\n"
            "【实际 Policy JSON】\n"
            + json.dumps(self.policy, ensure_ascii=False, indent=2, sort_keys=True)
        )


def _policy_rule_explanation(rule: dict) -> str:
    rule_type = str(rule.get("type", ""))
    params = rule.get("params", {}) if isinstance(rule.get("params", {}), dict) else {}
    separator_scope = str(params.get("scope", "global"))
    separator_text = (
        f"所有分词空格至少使用 {params.get('min_spaces', 1)} 个空格。"
        if separator_scope == "global"
        else f"只在相邻方括号术语之间使用至少 {params.get('min_spaces', 1)} 个空格。"
    )
    selection_mode = str(params.get("selection_mode", "domain_inference"))
    wrapper_scope = {
        "references_only": "只强制用户知识引用层中明确列出的术语",
        "references_and_ascii": "只强制用户引用术语和 ASCII 技术标识",
        "domain_inference": "允许模型根据领域语境识别其他术语",
    }.get(selection_mode, "按高级术语选择规则")
    descriptions = {
        "allowed_characters": f"限制输出字符范围：{', '.join(map(str, params.get('scripts', []))) or '按已配置字符集'}。",
        "punctuation": f"只允许指定标点：{'、'.join(map(str, params.get('allowed', []))) or '不使用额外标点'}。",
        "term_wrapper": f"{wrapper_scope}，并用 {params.get('left', '[')}…{params.get('right', ']')} 标记。",
        "separator": separator_text,
        "preserve": "保留用户指定的文字或表达，不得改写。",
        "forbidden_literals": "禁止输出用户指定的文字或表达。",
        "max_length": f"输出长度最多 {params.get('characters', '?')} 个字符。",
        "line_breaks": f"换行处理方式：{params.get('mode', 'preserve')}。",
        "case": f"英文字母大小写处理方式：{params.get('mode', 'preserve')}。",
        "literal_replace": "对用户明确指定的固定文字执行替换。",
        "model_instruction": "存在需要翻译模型遵守、但程序无法机械验证的语义或风格规则。",
    }
    return descriptions.get(rule_type, "")


# =========================================================================
# pages
# =========================================================================

class TemplatePage(QWidget):
    """Unified translation template: language pair + constraints + knowledge references."""

    language_changed = Signal(str, str)
    compiled_prompt_saved = Signal()
    ai_work_started = Signal()
    ai_work_finished = Signal()
    _ai_work_done = Signal(dict, object)

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
        self._closing = False
        self._ai_generation = 0
        self._ai_state_lock = threading.Lock()

        # Create scrollable content
        scroll = QScrollArea()
        scroll.setObjectName("contentScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        content.setObjectName("contentScrollBody")
        content.setMinimumWidth(0)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(14)

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
        language_card, language_card_layout = _make_content_card("语言方向")

        self._source_combo = QComboBox()
        self._source_combo.addItems(LANGUAGES)
        self._source_combo.setCurrentText(source)

        self._swap_btn = QPushButton("⇄")
        self._swap_btn.setObjectName("swapButton")
        self._swap_btn.setFixedSize(36, 36)
        self._swap_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._swap_btn.setToolTip("交换源语言和目标语言")
        self._swap_btn.clicked.connect(self._on_swap)

        self._target_combo = QComboBox()
        self._target_combo.addItems(LANGUAGES)
        self._target_combo.setCurrentText(target)

        # Field columns: label above combo so ⇄ can AlignBottom with combos.
        source_field = QVBoxLayout()
        source_field.setContentsMargins(0, 0, 0, 0)
        source_field.setSpacing(6)
        source_label = QLabel("源语言")
        source_label.setObjectName("feedbackSectionLabel")
        source_field.addWidget(source_label)
        source_field.addWidget(self._source_combo)

        target_field = QVBoxLayout()
        target_field.setContentsMargins(0, 0, 0, 0)
        target_field.setSpacing(6)
        target_label = QLabel("目标语言")
        target_label.setObjectName("feedbackSectionLabel")
        target_field.addWidget(target_label)
        target_field.addWidget(self._target_combo)

        lang_row = QHBoxLayout()
        lang_row.setSpacing(12)
        lang_row.addLayout(source_field, 1)
        lang_row.addWidget(self._swap_btn, 0, Qt.AlignmentFlag.AlignBottom)
        lang_row.addLayout(target_field, 1)
        language_card_layout.addLayout(lang_row)
        layout.addWidget(language_card)

        self._source_combo.currentTextChanged.connect(self._emit_language_change)
        self._target_combo.currentTextChanged.connect(self._emit_language_change)

        # Section 2: Constraints + Knowledge References
        reference_card, reference_card_layout = _make_content_card("约束与知识引用")
        constraints_label = QLabel("约束层")
        constraints_label.setObjectName("feedbackSectionLabel")
        reference_card_layout.addWidget(constraints_label)

        self._constraints_btn = QPushButton()
        self._constraints_btn.setObjectName("constraintsPreview")
        self._constraints_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        _prevent_horizontal_growth(self._constraints_btn)
        self._constraints_btn.clicked.connect(self._on_edit_constraints)
        self._refresh_constraints_preview()
        reference_card_layout.addWidget(self._constraints_btn)

        reference_card_layout.addSpacing(2)

        # Section 3: Knowledge References
        knowledge_label = QLabel("知识引用层")
        knowledge_label.setObjectName("feedbackSectionLabel")
        reference_card_layout.addWidget(knowledge_label)

        self._knowledge_list = QPlainTextEdit()
        self._knowledge_list.setReadOnly(True)
        self._knowledge_list.setPlaceholderText("尚未添加知识引用文档...")
        self._knowledge_list.setMaximumHeight(80)
        _configure_wrapping_text_edit(self._knowledge_list)
        if self._knowledge_paths:
            self._knowledge_list.setPlainText("\n".join(self._knowledge_paths))
        reference_card_layout.addWidget(self._knowledge_list)

        know_btn_row = QHBoxLayout()
        know_btn_row.setSpacing(8)
        add_know_btn = QPushButton("+ 添加文档")
        add_know_btn.setObjectName("secondaryButton")
        add_know_btn.clicked.connect(self._on_add_knowledge)
        rm_know_btn = QPushButton("− 移除")
        rm_know_btn.setObjectName("secondaryButton")
        rm_know_btn.clicked.connect(self._on_remove_knowledge)
        self._preview_refs_btn = QPushButton("预览解析")
        self._preview_refs_btn.setObjectName("secondaryButton")
        self._preview_refs_btn.clicked.connect(self._on_preview_references)
        self._export_ai_refs_btn = QPushButton("导出AI候选")
        self._export_ai_refs_btn.setObjectName("secondaryButton")
        self._export_ai_refs_btn.clicked.connect(self._on_export_ai_reference_candidates)
        know_btn_row.addWidget(add_know_btn)
        know_btn_row.addWidget(rm_know_btn)
        know_btn_row.addWidget(self._preview_refs_btn)
        know_btn_row.addWidget(self._export_ai_refs_btn)
        know_btn_row.addStretch(1)
        reference_card_layout.addLayout(know_btn_row)

        # Section 4: Compiled Prompt
        compiled_label = QLabel("Compiled Prompt 文件")
        compiled_label.setObjectName("feedbackSectionLabel")
        reference_card_layout.addWidget(compiled_label)

        self._compiled_path = QLineEdit(compiled_prompt_path)
        self._compiled_path.setReadOnly(True)
        _prevent_horizontal_growth(self._compiled_path)
        reference_card_layout.addWidget(self._compiled_path)
        layout.addWidget(reference_card)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._preview_btn = QPushButton("生成预览")
        self._preview_btn.setObjectName("secondaryButton")
        self._preview_btn.clicked.connect(self._on_preview)
        self._save_btn = QPushButton("保存并启用")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._save_status = QLabel("")
        self._save_status.setObjectName("hintLabel")
        self._save_status.setWordWrap(True)
        _prevent_horizontal_growth(self._save_status)
        self._save_status.setMaximumHeight(30)
        layout.addWidget(self._save_status)

        layout.addStretch(1)

        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        self._ai_work_done.connect(self._on_ai_work_done)

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

    # Knowledge reference and prompt editing helpers
    def _knowledge_reference_dir(self) -> Path:
        return self._prompt_storage.ensure_reference_dir()

    def _on_add_knowledge(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择知识文档", str(self._knowledge_reference_dir()), "Markdown 文件 (*.md);;所有文件 (*)")
        if file_path:
            self._add_knowledge_path(file_path)

    def _on_remove_knowledge(self) -> None:
        self._knowledge_paths.clear()
        self._knowledge_list.clear()

    def _add_knowledge_path(self, file_path: str) -> bool:
        path = str(file_path).strip()
        if not path or path in self._knowledge_paths:
            return False
        self._knowledge_paths.append(path)
        self._refresh_knowledge_list()
        return True

    def _refresh_knowledge_list(self) -> None:
        self._knowledge_list.setPlainText("\n".join(self._knowledge_paths))

    def _on_preview_references(self) -> None:
        QMessageBox.information(self, "知识引用解析预览", self._build_reference_preview())

    def _build_reference_preview(self) -> str:
        if not self._knowledge_paths:
            return "尚未添加知识引用文档。"
        parts: list[str] = []
        total_entries = 0
        total_style = 0
        total_risk = 0
        for path in self._knowledge_paths:
            package = self._prompt_storage.preview_reference_file(path)
            total_entries += len(package.entries)
            total_style += len(package.style_guidance)
            total_risk += len(package.risk_notes)
            lines = [f"文件：{path}"]
            if package.entries:
                lines.append("术语：")
                for entry in package.entries[:12]:
                    lines.append(f"  • {entry.source} -> {entry.target}")
                if len(package.entries) > 12:
                    lines.append(f"  • ... 另 {len(package.entries) - 12} 条")
            if package.style_guidance:
                lines.append("风格：")
                lines.extend(f"  • {item}" for item in package.style_guidance[:6])
            if package.risk_notes:
                lines.append("风险/注意：")
                lines.extend(f"  • {item}" for item in package.risk_notes[:6])
            if package.is_empty:
                lines.append("  （未解析出术语、风格或风险提示）")
            parts.append("\n".join(lines))
        summary = f"总计：术语 {total_entries} 条，风格 {total_style} 条，风险/注意 {total_risk} 条。"
        return summary + "\n\n" + "\n\n".join(parts)

    def _on_export_ai_reference_candidates(self) -> None:
        candidate_path = self._prompt_storage.export_ai_optimization_reference_candidates(
            self._compiled_path.text().strip() or "prompts/compiled-prompt.md"
        )
        if candidate_path is None:
            QMessageBox.information(
                self,
                "没有候选术语",
                "当前 compiled prompt 的 AI Optimization Layer 中没有可导出的术语候选。",
            )
            return
        confirmed = QMessageBox.question(
            self,
            "导出AI候选术语",
            f"已导出候选文件：\n{candidate_path}\n\n"
            "这些术语来自 AI 生成内容，尚未被信任。是否现在加入知识引用列表？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed == QMessageBox.StandardButton.Yes:
            added = self._add_knowledge_path(str(candidate_path))
            if added:
                self._save_status.setText("已加入候选知识引用，请预览确认后再保存启用。")
                self._save_status.setStyleSheet("color: #fbbf24; font-size: 12px;")

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
        with self._ai_state_lock:
            if self._closing:
                return
            self._ai_generation += 1
            generation = self._ai_generation
        settings = copy.deepcopy(self._settings or AppSettings.load())
        self._apply_form_to_settings(settings)
        constraints = PromptConstraints(text=self._constraints_text)
        references = self._collect_knowledge_references()

        self._save_status.setText("正在优化提示词...")
        self._save_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
        self._preview_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self.ai_work_started.emit()

        def _run() -> None:
            result_holder: dict = {"generation": generation}
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
                # This daemon thread is a Qt boundary: keep the UI alive and
                # fall back to the unoptimized preview, but make the failure
                # observable without recording prompt contents.
                get_debug_logger().exception(
                    "Template prompt optimization worker failed; using unoptimized preview"
                )
                result_holder["warning"] = str(exc)
                result_holder["optimized"] = ""
                result_holder["compiled"] = self._compiler.compile_preview(
                    constraints,
                    references=references,
                )
            with self._ai_state_lock:
                should_emit = not self._closing and generation == self._ai_generation
            if should_emit:
                try:
                    self._ai_work_done.emit(result_holder, settings)
                except RuntimeError:
                    return

        threading.Thread(target=_run, daemon=True).start()

    def _on_ai_work_done(self, result_holder: dict, settings: object) -> None:
        with self._ai_state_lock:
            if self._closing or result_holder.get("generation") != self._ai_generation:
                return
        self.ai_work_finished.emit()
        self._preview_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
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
        user_preview = self._build_user_preview(
            optimized,
            policy=compiled.policy,
            machine_content=compiled.content,
            reference_package=compiled.reference_package,
        )
        if not self._confirm_compiled_prompt(user_preview):
            self._save_status.setText("已取消")
            self._save_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
            return

        try:
            configured_path = settings.prompt.compiled_prompt_path
            bundle_snapshot = self._prompt_storage.snapshot_compiled_prompt_bundle(
                configured_path
            )
            saved_path = self._prompt_storage.save_compiled_prompt(
                compiled.content,
                configured_path,
                policy=compiled.policy,
                reference_package=compiled.reference_package,
            )
            stored_path = self._prompt_storage.stored_compiled_prompt_path(
                settings.prompt.compiled_prompt_path
            )
            settings.prompt.compiled_prompt_path = stored_path
            self._compiled_path.setText(stored_path)
            original_prompt_settings = (
                copy.deepcopy(self._settings.prompt)
                if self._settings is not None
                else None
            )
            if self._save_settings is not None:
                if self._settings is not None:
                    self._settings.prompt = copy.deepcopy(settings.prompt)
                try:
                    self._save_settings()
                except Exception:
                    if self._settings is not None and original_prompt_settings is not None:
                        self._settings.prompt = original_prompt_settings
                    self._prompt_storage.restore_compiled_prompt_bundle(bundle_snapshot)
                    raise
            else:
                try:
                    settings.save()
                except Exception:
                    self._prompt_storage.restore_compiled_prompt_bundle(bundle_snapshot)
                    raise
                if self._settings is not None:
                    self._settings.prompt = copy.deepcopy(settings.prompt)
            warning = result_holder.get("warning")
            if warning:
                self._save_status.setText(f"已保存并启用（AI 优化失败，已使用原始约束）：{warning}")
                self._save_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
            else:
                self._save_status.setText("已保存并启用 ✓")
                self._save_status.setStyleSheet("color: #4ade80; font-size: 12px;")
            self.compiled_prompt_saved.emit()
        except Exception as exc:
            # Saving is a Qt slot boundary.  Rollback has already happened
            # in the inner save branches; surface a safe message and retain
            # the traceback in debug logs instead of leaking it to Qt.
            get_debug_logger().exception("Template prompt save failed")
            self._save_status.setText(f"保存失败: {exc}")
            self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")

    def shutdown(self) -> None:
        """Invalidate queued worker callbacks before this QWidget is deleted."""

        with self._ai_state_lock:
            self._closing = True
            self._ai_generation += 1

    def constraints_text(self) -> str:
        return self._constraints_text

    def _apply_form_to_settings(self, settings: AppSettings) -> None:
        settings.prompt.constraints_text = self._constraints_text
        settings.prompt.knowledge_reference_paths = list(self._knowledge_paths)

    def _collect_knowledge_references(self) -> list[PromptKnowledgeReference]:
        return [PromptKnowledgeReference(path=path) for path in self._knowledge_paths]

    def _build_user_preview(
        self,
        optimized: str = "",
        policy: dict | None = None,
        machine_content: str = "",
        reference_package: dict | None = None,
    ) -> CompiledPromptReview:
        parts = [
            "一、基本目标\n"
            "准确保留主客体、动作关系、否定、条件、例外、使役、被动、请求、命令和完成状态。"
            "自然表达、礼貌和敬语不能改变原意；用户明确要求的输出格式仍会遵守。",
            "二、你填写的要求\n" + (self._constraints_text.strip() or "（未填写额外要求）"),
        ]
        summary = getattr(optimized, "user_summary", "")
        change_items = getattr(optimized, "change_items", ())
        ai_lines: list[str] = []
        if summary:
            ai_lines.append(str(summary).strip())
        for item in change_items:
            title = str(item.get("title", "")).strip()
            description = str(item.get("description", "")).strip()
            if title and description:
                ai_lines.append(f"• {title}：{description}")
        parts.append(
            "三、AI 帮你补充的说明\n"
            + ("\n".join(ai_lines) if ai_lines else "AI 未提供中文变更说明，可在高级内容中查看机器规则。")
        )

        rules = policy.get("rules", []) if isinstance(policy, dict) else []
        locally_enforced: list[str] = []
        model_enforced: list[str] = []
        unknown_local = 0
        unknown_model = 0
        for rule in rules if isinstance(rules, list) else []:
            if not isinstance(rule, dict):
                continue
            text = _policy_rule_explanation(rule)
            enforcement = str(rule.get("enforcement", "both")).casefold()
            target = model_enforced if enforcement == "model" else locally_enforced
            if text:
                target.append(f"• {text}")
            elif enforcement == "model":
                unknown_model += 1
            else:
                unknown_local += 1
        if unknown_local:
            locally_enforced.append(f"• 存在 {unknown_local} 条高级本地规则，请在高级内容中查看。")
        if unknown_model:
            model_enforced.append(f"• 存在 {unknown_model} 条高级模型规则，请在高级内容中查看。")
        parts.append(
            "四、程序会强制检查\n"
            + ("\n".join(locally_enforced) if locally_enforced else "（当前没有可由程序机械检查的规则）")
        )
        parts.append(
            "五、模型需要遵守\n"
            + ("\n".join(model_enforced) if model_enforced else "（当前没有额外的模型执行规则）")
        )

        package = reference_package if isinstance(reference_package, dict) else {}
        entries = package.get("entries", []) if isinstance(package.get("entries", []), list) else []
        styles = package.get("style_guidance", []) if isinstance(package.get("style_guidance", []), list) else []
        risks = package.get("risk_notes", []) if isinstance(package.get("risk_notes", []), list) else []
        if self._knowledge_paths:
            reference_text = (
                f"已配置 {len(self._knowledge_paths)} 个引用文档；解析出术语 {len(entries)} 条、"
                f"风格说明 {len(styles)} 条、风险提示 {len(risks)} 条。"
            )
        else:
            reference_text = "未配置知识引用文档；具体术语读法未由用户引用层固定。"
        parts.append("六、知识引用层\n" + reference_text)
        return CompiledPromptReview("\n\n".join(parts), machine_content, policy)

    def _confirm_compiled_prompt_with_dialog(self, content: str) -> bool:
        dlg = QDialog(self)
        dlg.setObjectName("compiledPromptDialog")
        dlg.setWindowTitle("确认翻译模板")
        dlg.resize(680, 500)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        tabs = QTabWidget()
        tabs.setObjectName("compiledPromptTabs")
        easy_viewer = QPlainTextEdit()
        easy_viewer.setObjectName("compiledPromptViewer")
        easy_viewer.setReadOnly(True)
        easy_viewer.setPlainText(str(content))
        advanced_viewer = QPlainTextEdit()
        advanced_viewer.setObjectName("compiledPromptViewer")
        advanced_viewer.setReadOnly(True)
        advanced_viewer.setPlainText(getattr(content, "advanced_content", "（无高级内容）"))
        tabs.addTab(easy_viewer, "易懂说明")
        tabs.addTab(advanced_viewer, "高级内容")
        layout.addWidget(tabs, 1)

        hint = QLabel("请确认易懂说明；需要核对机器实际内容时可打开“高级内容”。")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        no_btn = QPushButton("返回修改")
        no_btn.setObjectName("secondaryButton")
        yes_btn = QPushButton("确认并启用")
        yes_btn.setObjectName("primaryButton")
        no_btn.setDefault(True)
        btn_row.addWidget(no_btn)
        btn_row.addWidget(yes_btn)
        layout.addLayout(btn_row)

        yes_btn.clicked.connect(lambda: dlg.done(1))
        no_btn.clicked.connect(lambda: dlg.done(0))
        return dlg.exec() == 1


class ModelPage(QWidget):
    """OpenAI-compatible API configuration with model list + live test."""

    config_changed = Signal(str, str, str, str)
    _test_done = Signal(bool, str)
    _models_done = Signal(object, str)  # models list or None, status message

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
        self._fetching_models = False
        self._testing = False
        self._models_loaded = False
        self._closing = False
        self._models_generation = 0
        self._test_generation = 0

        title = QLabel("模型配置")
        title.setObjectName("pageTitle")
        desc = QLabel("连接 OpenAI 兼容接口，Fast / Thinking 模型只从 /models 返回的列表中选择。")
        desc.setObjectName("pageDesc")
        desc.setWordWrap(True)

        self._base_url = QLineEdit(base_url)
        self._base_url.setPlaceholderText("https://api.openai.com/v1")
        self._api_key = QLineEdit(api_key)
        self._api_key.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self._api_key.setPlaceholderText("sk-...")

        # Select-only dropdowns. List content comes from GET /models, never a hardcoded catalog.
        self._model = self._make_model_combo()
        self._thinking_model = self._make_model_combo()
        self._seed_model_value(self._model, fast_model, placeholder="等待拉取模型…")
        self._seed_model_value(self._thinking_model, thinking_model, placeholder="等待拉取模型…")

        self._fetch_models_btn = QPushButton("拉取模型")
        self._fetch_models_btn.setObjectName("secondaryButton")
        self._fetch_models_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fetch_models_btn.setToolTip("使用当前 Base URL + API Key 请求 /models（非本地硬编码列表）")
        self._fetch_models_btn.clicked.connect(self._on_fetch_models)

        self._test_btn = QPushButton("测试连接")
        self._test_btn.setObjectName("primaryButton")
        self._test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_btn.setToolTip("用当前 Fast 模型发一条最短请求验证连通性")
        self._test_btn.clicked.connect(self._on_test)

        self._save_config_btn = QPushButton("保存配置")
        self._save_config_btn.setObjectName("secondaryButton")
        self._save_config_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_config_btn.clicked.connect(self._on_save_configuration)

        self._test_status = QLabel("")
        self._test_status.setObjectName("hintLabel")
        self._test_status.setWordWrap(True)
        _prevent_horizontal_growth(self._test_status)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self._fetch_models_btn)
        action_row.addWidget(self._test_btn)
        action_row.addWidget(self._save_config_btn)
        action_row.addWidget(self._test_status, 1)

        self._credential_status = QLabel()
        self._credential_status.setObjectName("connectionStatusPill")
        self._connection_summary = QLabel()
        self._connection_summary.setObjectName("hintLabel")
        self._connection_summary.setWordWrap(True)

        summary_card, summary_layout = _make_content_card()
        summary_head = QHBoxLayout()
        summary_title = QLabel("连接摘要")
        summary_title.setObjectName("cardTitle")
        summary_head.addWidget(summary_title)
        summary_head.addStretch(1)
        summary_head.addWidget(self._credential_status)
        summary_layout.addLayout(summary_head)
        summary_layout.addWidget(self._connection_summary)

        interface_card, interface_layout = _make_content_card("接口")
        interface_form = QFormLayout()
        interface_form.setContentsMargins(0, 0, 0, 0)
        interface_form.setSpacing(12)
        interface_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        interface_form.addRow(_make_field_label("Base URL"), self._base_url)
        interface_form.addRow(_make_field_label("API Key"), self._api_key)
        interface_layout.addLayout(interface_form)

        models_card, models_layout = _make_content_card("模型（来自 /models，非硬编码）")
        models_form = QFormLayout()
        models_form.setContentsMargins(0, 0, 0, 0)
        models_form.setSpacing(12)
        models_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        models_form.addRow(_make_field_label("Fast Model"), self._model)
        models_form.addRow(_make_field_label("Thinking Model"), self._thinking_model)
        models_layout.addLayout(models_form)

        self._base_url.textChanged.connect(self._on_credentials_changed)
        self._api_key.textChanged.connect(self._on_credentials_changed)
        self._model.currentTextChanged.connect(self._on_model_selection_changed)
        self._thinking_model.currentTextChanged.connect(self._on_model_selection_changed)
        self._test_done.connect(self._show_test_result)
        self._models_done.connect(self._show_models_result)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(summary_card)
        layout.addWidget(interface_card)
        layout.addWidget(models_card)
        layout.addLayout(action_row)
        layout.addStretch(1)

        self._refresh_connection_summary()
        self._refresh_action_enabled()
        # Existing saved credentials: auto-fill combos without requiring a manual click.
        if base_url.strip() and api_key.strip():
            self._set_status("检测到已保存的 URL/Key，正在自动拉取模型…", "busy")
            QTimer.singleShot(0, self.ensure_models_loaded)

    @staticmethod
    def _make_model_combo() -> QComboBox:
        """Build a compact select-style combo aligned with input fields."""

        combo = QComboBox()
        combo.setObjectName("modelSelectCombo")
        combo.setEditable(False)
        combo.setMinimumHeight(36)
        combo.setMaximumHeight(36)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        combo.setMinimumWidth(0)
        combo.setMaxVisibleItems(14)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(16)
        combo.setCursor(Qt.CursorShape.PointingHandCursor)
        # Keep default app style; QSS alone draws a light chevron (no Fusion override).
        return combo

    @staticmethod
    def _seed_model_value(combo: QComboBox, value: str, placeholder: str) -> None:
        """Show saved model or a placeholder until /models returns."""

        value = (value or "").strip()
        combo.blockSignals(True)
        combo.clear()
        if value:
            combo.addItem(value)
            combo.setCurrentIndex(0)
        else:
            combo.addItem(placeholder)
            combo.setItemData(0, False, Qt.ItemDataRole.UserRole - 1)
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _selected_model_text(self, combo: QComboBox) -> str:
        text = combo.currentText().strip()
        if text in {
            "等待拉取模型…",
            "先拉取模型列表…",
            "（无可选模型，请检查 /models）",
        }:
            return ""
        return text

    def _emit_change(self) -> None:
        self.config_changed.emit(
            self._base_url.text().strip(),
            self._api_key.text().strip(),
            self._selected_model_text(self._model),
            self._selected_model_text(self._thinking_model),
        )

    def _on_credentials_changed(self) -> None:
        # URL/Key changed: list is stale until next successful fetch.
        self._models_loaded = False
        self._refresh_connection_summary()
        self._refresh_action_enabled()

    def _on_model_selection_changed(self) -> None:
        """Persist intentional model selections, never individual keypresses."""

        if not self._closing:
            self._emit_change()

    def _on_save_configuration(self) -> None:
        if self._closing:
            return
        self._emit_change()

    def _refresh_connection_summary(self) -> None:
        """Mirror the mock's compact credential summary without exposing secrets."""

        base_url = self._base_url.text().strip()
        api_key = self._api_key.text().strip()
        configured = bool(base_url and api_key)
        self._credential_status.setText("●  已配置凭据" if configured else "●  等待配置")
        self._credential_status.setProperty("configured", configured)
        self._credential_status.style().unpolish(self._credential_status)
        self._credential_status.style().polish(self._credential_status)
        if configured:
            tail = api_key[-4:] if len(api_key) >= 4 else "••••"
            self._connection_summary.setText(f"{base_url}  ·  key ••••{tail}")
        else:
            self._connection_summary.setText("填写 Base URL 与 API Key 后即可拉取模型列表。")

    def ensure_models_loaded(self) -> None:
        """Auto-fetch model list once when credentials are available."""

        if self._models_loaded or self._fetching_models:
            return
        if not (self._base_url.text().strip() and self._api_key.text().strip()):
            return
        self._on_fetch_models()

    def _refresh_action_enabled(self) -> None:
        has_creds = bool(self._base_url.text().strip() and self._api_key.text().strip())
        self._fetch_models_btn.setEnabled(has_creds and not self._fetching_models and not self._testing)
        self._test_btn.setEnabled(has_creds and not self._testing and not self._fetching_models)
        self._save_config_btn.setEnabled(not self._closing and not self._fetching_models and not self._testing)

    def _set_status(self, message: str, tone: str = "hint") -> None:
        colors = {
            "hint": "#475569",
            "busy": "#fbbf24",
            "ok": "#4ade80",
            "error": "#f87171",
        }
        self._test_status.setText(message)
        self._test_status.setStyleSheet(
            f"color: {colors.get(tone, colors['hint'])}; font-size: 12px;"
        )

    def _populate_model_combo(self, combo: QComboBox, models: list[str], preferred: str) -> None:
        """Replace combo items with a real selectable model list."""

        preferred = preferred.strip()
        current = self._selected_model_text(combo) or preferred
        combo.blockSignals(True)
        combo.clear()
        if not models:
            combo.addItem("（无可选模型，请检查 /models）")
            combo.setItemData(0, False, Qt.ItemDataRole.UserRole - 1)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
            return

        for model_id in models:
            combo.addItem(model_id)
        if current:
            index = combo.findText(current)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                # Keep previously saved model even if provider list omitted it.
                combo.insertItem(0, current)
                combo.setCurrentIndex(0)
        else:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _on_fetch_models(self) -> None:
        base = self._base_url.text().strip()
        key = self._api_key.text().strip()
        if not base or not key:
            self._set_status("请先填写 Base URL 和 API Key", "error")
            return
        if self._fetching_models:
            return

        self._fetching_models = True
        self._models_generation += 1
        generation = self._models_generation
        self._refresh_action_enabled()
        self._set_status("正在拉取模型列表...", "busy")

        def _run() -> None:
            try:
                client = OpenAICompatibleClient(
                    ClientConfig(base_url=base, api_key=key, model="unused")
                )
                models = client.list_models()
                result = ([], "接口返回空列表，请检查 Base URL 是否支持 /models") if not models else (
                    models, f"已拉取 {len(models)} 个模型，请在下拉框中选择"
                )
            except TranslationError as exc:
                result = (None, str(exc))
            except Exception as exc:
                # Model discovery runs off the Qt thread.  Preserve the
                # user-facing failure result rather than letting a backend
                # implementation error escape the worker.
                get_debug_logger().exception("Model discovery worker failed")
                result = (None, str(exc))
            if not self._closing and generation == self._models_generation:
                try:
                    self._models_done.emit(*result)
                except RuntimeError:
                    return

        threading.Thread(target=_run, daemon=True).start()

    def _show_models_result(self, models: object, message: str) -> None:
        if self._closing:
            return
        self._fetching_models = False
        self._refresh_action_enabled()
        if models is None:
            self._models_loaded = False
            self._set_status(f"拉取失败：{message}"[:120], "error")
            return

        model_ids = [str(item) for item in models] if isinstance(models, list) else []
        preferred_fast = self._selected_model_text(self._model)
        preferred_thinking = self._selected_model_text(self._thinking_model)
        self._populate_model_combo(self._model, model_ids, preferred_fast)
        self._populate_model_combo(self._thinking_model, model_ids, preferred_thinking)
        self._models_loaded = bool(model_ids)
        self._emit_change()
        tone = "ok" if model_ids else "busy"
        self._set_status(message, tone)

    def _on_test(self) -> None:
        base = self._base_url.text().strip()
        key = self._api_key.text().strip()
        model = (
            self._selected_model_text(self._model)
            or self._selected_model_text(self._thinking_model)
        )

        if not base or not key:
            self._set_status("请填写 Base URL 和 API Key", "error")
            return
        if not model:
            self._set_status("请先从下拉框选择要测试的模型", "error")
            return
        if self._testing:
            return

        self._testing = True
        self._test_generation += 1
        generation = self._test_generation
        self._refresh_action_enabled()
        self._set_status(f"测试中（{model}）...", "busy")
        threading.Thread(
            target=self._run_connection_test,
            args=(base, key, model, generation),
            daemon=True,
        ).start()

    def _run_connection_test(self, base: str, key: str, model: str, generation: int) -> None:
        """Run the network probe away from the Qt event loop."""

        import time as _time

        ok, msg = False, ""
        t0 = _time.perf_counter()
        try:
            client = OpenAICompatibleClient(
                ClientConfig(base_url=base, api_key=key, model=model, timeout_seconds=30)
            )
            client.translate("You are a test.", "hello")
            elapsed = (_time.perf_counter() - t0) * 1000
            ok, msg = True, f"连接成功 · {model} · {elapsed:.0f}ms"
        except TranslationError as exc:
            elapsed = (_time.perf_counter() - t0) * 1000
            msg = f"{exc} ({elapsed:.0f}ms)"[:120]
        except Exception as exc:
            # This is the final backend/Qt worker boundary; retain a short
            # user message while keeping the full exception in debug logs.
            get_debug_logger().exception("Connection test worker failed")
            msg = str(exc)[:120]
        if not self._closing and generation == self._test_generation:
            try:
                self._test_done.emit(ok, msg)
            except RuntimeError:
                return

    def _show_test_result(self, ok: bool, msg: str) -> None:
        if self._closing:
            return
        self._testing = False
        self._refresh_action_enabled()
        self._set_status(msg, "ok" if ok else "error")

    def show_save_result(self, success: bool, message: str) -> None:
        """Display persistence feedback without exposing the API key."""

        self._set_status(message, "ok" if success else "error")

    def shutdown(self) -> None:
        """Invalidate late daemon-thread completions during window teardown."""

        self._closing = True
        self._models_generation += 1
        self._test_generation += 1


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


class FeedbackOptimizationDialog(QDialog):
    """Modal review surface that keeps AI candidates separate from user fields."""

    COMPLETED = "completed"
    UNREVIEWED = "未处理"
    USED_AI = "已采用 AI"
    USED_USER = "已采用修改"
    SKIPPED = "已跳过"

    def __init__(
        self,
        suggestion: FeedbackOptimization,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setObjectName("feedbackOptimizationDialog")
        self.suggestion = suggestion
        self.decision = ""
        self._title_bar = TitleBar(self)
        self._title_bar.minimize_requested.connect(self._minimize_with_owner)
        self._title_bar.close_requested.connect(self.reject)

        self._section_names = (
            "优化译文", "问题判断", "关键词",
        )
        self._review_states = [self.UNREVIEWED] * len(self._section_names)
        self._review_choices: list[dict[str, object] | None] = [None] * len(self._section_names)
        self._nav_buttons: list[QPushButton] = []
        for _ in self._section_names:
            btn = QPushButton("")
            btn.setObjectName("reviewNavButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(40)
            self._nav_buttons.append(btn)
        self._stack = QStackedWidget()
        self._user_editors: list[QPlainTextEdit] = []
        self._ai_keyword_editor = KeywordTagEditor()
        self._ai_keyword_editor.set_keywords(suggestion.trigger_options or ([suggestion.trigger] if suggestion.trigger else []))
        self._user_keyword_editor = KeywordTagEditor()

        pages = (
            self._text_page(suggestion.improved_translation, "AI 未提供优化译文。"),
            self._text_page(suggestion.problem_summary, "AI 未提供问题判断。"),
            self._keyword_page(),
        )
        for page in pages:
            self._stack.addWidget(page)
        for index, button in enumerate(self._nav_buttons):
            button.clicked.connect(lambda checked=False, i=index: self._select_page(i))
        self._select_page(0)

        sidebar = QWidget()
        sidebar.setObjectName("reviewSidebar")
        sidebar.setFixedWidth(148)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(4)
        for button in self._nav_buttons:
            side_layout.addWidget(button)
        side_layout.addStretch(1)

        self._error_label = QLabel("")
        self._error_label.setObjectName("hintLabel")
        self._error_label.setWordWrap(True)
        self._use_ai_button = QPushButton("采用优化")
        self._use_ai_button.setObjectName("primaryButton")
        self._use_ai_button.clicked.connect(self._use_ai)
        self._use_user_button = QPushButton("确认修改")
        self._use_user_button.setObjectName("primaryButton")
        self._use_user_button.clicked.connect(self._use_user)
        self._skip_button = QPushButton("跳过")
        self._skip_button.setObjectName("secondaryButton")
        self._skip_button.clicked.connect(self._skip_current)
        self._finish_button = QPushButton("完成审查")
        self._finish_button.setObjectName("primaryButton")
        self._finish_button.setEnabled(False)
        self._finish_button.clicked.connect(self._finish_review)
        for button in (
            self._skip_button,
            self._use_ai_button,
            self._use_user_button,
            self._finish_button,
        ):
            button.setFixedHeight(34)
            button.setMinimumWidth(72)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self._skip_button)
        buttons.addStretch(1)
        buttons.addWidget(self._use_ai_button)
        buttons.addWidget(self._use_user_button)
        buttons.addWidget(self._finish_button)

        right = QVBoxLayout()
        right.addWidget(self._stack, 1)
        right.addWidget(self._error_label)
        right.addLayout(buttons)
        body = QHBoxLayout()
        body.setContentsMargins(12, 8, 16, 16)
        body.addWidget(sidebar)
        body.addLayout(right, 1)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._title_bar)
        root.addLayout(body, 1)

        parent_size = parent.size() if parent is not None else QSize(760, 560)
        width = max(520, int(parent_size.width() * 0.88))
        height = max(420, int(parent_size.height() * 0.88))
        self.resize(min(width, parent_size.width()), min(height, parent_size.height()))
        self.setMinimumSize(min(520, parent_size.width()), min(420, parent_size.height()))
        self._refresh_nav_labels()

    def _text_page(self, ai_text: str, empty_text: str = "AI 未提供候选。") -> QWidget:
        page = QWidget()
        ai = QPlainTextEdit(ai_text or empty_text)
        ai.setReadOnly(True)
        user = QPlainTextEdit()
        self._user_editors.append(user)
        for editor in (ai, user):
            _configure_wrapping_text_edit(editor)
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("AI 候选"))
        layout.addWidget(ai, 1)
        layout.addWidget(QLabel("我的意见 / 修改"))
        layout.addWidget(user, 1)
        return page

    def _keyword_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("AI 候选（可编辑）"))
        self._keyword_hint = QLabel(
            "" if self.suggestion.trigger_options or self.suggestion.trigger
            else "本次被判断为单句修正，未生成长期记忆关键词。"
        )
        self._keyword_hint.setObjectName("hintLabel")
        self._keyword_hint.setWordWrap(True)
        layout.addWidget(self._keyword_hint)
        layout.addWidget(self._ai_keyword_editor, 1)
        layout.addWidget(QLabel("我的意见 / 修改"))
        layout.addWidget(self._user_keyword_editor, 1)
        return page

    def _select_page(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        if hasattr(self, "_error_label"):
            self._error_label.clear()
        for i, button in enumerate(self._nav_buttons):
            button.setChecked(i == index)

    def _use_ai(self) -> None:
        index = self._stack.currentIndex()
        value: object
        if index == 0:
            value = self.suggestion.improved_translation
            if not str(value).strip():
                self._error_label.setText("AI 没有提供优化译文，不能采用空译文。")
                return
        elif index == 1:
            value = self.suggestion.problem_summary
        elif index == 2:
            value = self.ai_keywords()
            if not value:
                self._record_current(self.SKIPPED, None)
                return
        self._record_current(self.USED_AI, value)

    def _use_user(self) -> None:
        index = self._stack.currentIndex()
        value: object = self.user_keywords() if index == 2 else self._user_text(index)
        if not value:
            message = "请先填写完整的修改译文。" if index == 0 else "请先填写当前项目的修改内容。"
            self._error_label.setText(message)
            return
        self._record_current(self.USED_USER, value)

    def _skip_current(self) -> None:
        self._record_current(self.SKIPPED, None)

    def _record_current(self, state: str, value: object) -> None:
        index = self._stack.currentIndex()
        self._review_states[index] = state
        self._review_choices[index] = {"state": state, "value": value}
        self._refresh_nav_labels()
        self._finish_button.setEnabled(all(item != self.UNREVIEWED for item in self._review_states))
        for offset in range(1, len(self._review_states) + 1):
            candidate = (index + offset) % len(self._review_states)
            if self._review_states[candidate] == self.UNREVIEWED:
                self._select_page(candidate)
                break

    def _finish_review(self) -> None:
        if not all(item != self.UNREVIEWED for item in self._review_states):
            self._error_label.setText("请逐项采用、修改或跳过后再完成审查。")
            return
        self.decision = self.COMPLETED
        self.accept()

    def _refresh_nav_labels(self) -> None:
        """Show short section titles with a status dot (gray / red / green)."""

        for name, state, button in zip(
            self._section_names, self._review_states, self._nav_buttons
        ):
            if state == self.UNREVIEWED:
                review_state = "pending"
            elif state == self.SKIPPED:
                review_state = "skipped"
            else:
                review_state = "done"
            button.setText(f"●  {name}")
            button.setProperty("reviewState", review_state)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _user_text(self, section_index: int) -> str:
        editor_index = {0: 0, 1: 1}.get(section_index)
        return self._user_editors[editor_index].toPlainText().strip() if editor_index is not None else ""

    def review_choice(self, section_index: int) -> dict[str, object] | None:
        return self._review_choices[section_index]

    def _minimize_with_owner(self) -> None:
        owner = self.parentWidget().window() if self.parentWidget() is not None else None
        if owner is not None and owner is not self:
            owner.showMinimized()
        else:
            self.showMinimized()

    def user_problem_note(self) -> str:
        return self._user_text(1)

    def user_translation(self) -> str:
        return self._user_text(0)

    def ai_keywords(self) -> list[str]:
        return self._ai_keyword_editor.keywords()

    def user_keywords(self) -> list[str]:
        return self._user_keyword_editor.keywords()

class FeedbackPage(QWidget):
    """Review bad translations and turn accepted fixes into local memory."""

    ai_work_started = Signal()
    ai_work_finished = Signal()
    _ai_work_done = Signal(dict)
    _consolidation_done = Signal(dict)

    def sizeHint(self) -> QSize:
        """Keep the page compatible with the main window's 680px minimum width."""

        hint = super().sizeHint()
        return QSize(min(hint.width(), 440), hint.height())

    def __init__(
        self,
        feedback_store: FeedbackStore,
        settings: AppSettings,
        optimizer: FeedbackOptimizer | None = None,
        learning_service: FeedbackLearningService | None = None,
        dialog_factory: Callable[[FeedbackOptimization, QWidget | None], FeedbackOptimizationDialog] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("contentPage")
        self._feedback_store = feedback_store
        self._settings = settings
        self._optimizer = optimizer or FeedbackOptimizer(feedback_store=feedback_store)
        self._learning_service = learning_service or FeedbackLearningService(
            feedback_store,
            settings,
            self._optimizer,
        )
        self._dialog_factory = dialog_factory or FeedbackOptimizationDialog
        self._records: list[FeedbackRecord] = []
        self._ai_optimizing = False
        self._ai_job_id = 0
        self._ai_state_lock = threading.Lock()
        self._ai_worker_thread: threading.Thread | None = None
        self._closing = False
        self._ai_problem_summary = ""
        self._active_memory_rule_id = ""
        self._visible_memory_rules: list[MemoryRule] = []
        self._ai_work_done.connect(self._on_ai_work_done)
        self._consolidation_done.connect(self._on_consolidation_done)

        title = QLabel("优化翻译")
        title.setObjectName("pageTitle")
        desc = QLabel(
            "提交正确译文后，纠错会立即参与相似句翻译；长期规则由后台根据纠错库自动归纳。"
        )
        desc.setObjectName("pageDesc")
        desc.setWordWrap(True)

        self._feedback_combo = QComboBox()
        _prevent_horizontal_growth(self._feedback_combo)
        self._feedback_combo.currentIndexChanged.connect(self._on_feedback_selected)

        self._refresh_button = QPushButton("刷新")
        self._refresh_button.setObjectName("secondaryButton")
        self._refresh_button.setMinimumSize(92, 36)
        self._refresh_button.clicked.connect(
            lambda _checked=False: self.refresh(
                preserve_current=True,
                preserve_drafts=True,
            )
        )

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("反馈记录"))
        picker_row.addWidget(self._feedback_combo, 1)
        picker_row.addWidget(self._refresh_button)

        compare_height = 90
        self._source_text = QPlainTextEdit()
        self._source_text.setReadOnly(True)
        self._source_text.setFixedHeight(compare_height)

        self._current_translation_text = QPlainTextEdit()
        self._current_translation_text.setReadOnly(True)
        self._current_translation_text.setFixedHeight(compare_height)

        self._note_text = QPlainTextEdit()
        self._note_text.setPlaceholderText("可选：写下哪里不满意，例如术语错、语气错、漏译、把意思翻反了。")
        self._note_text.setFixedHeight(compare_height)

        self._corrected_translation_text = QPlainTextEdit()
        self._corrected_translation_text.setPlaceholderText(
            "可以直接输入你自己的译文；也可以让 AI 生成建议后再修改。"
        )
        self._corrected_translation_text.setFixedHeight(compare_height)

        self._keyword_editor = KeywordTagEditor()

        self._rule_text = QPlainTextEdit()
        self._rule_text.setPlaceholderText("确认后写入本地记忆的规则，例如：出现“高考”时应译为日本语境下的大学入学考试，不要误作高校考试。")
        self._rule_text.setFixedHeight(compare_height)

        for text_area in (
            self._source_text,
            self._current_translation_text,
            self._note_text,
            self._corrected_translation_text,
            self._rule_text,
        ):
            _configure_wrapping_text_edit(text_area)
            text_area.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )

        # Stack OCR / current translation / note / accepted translation vertically.
        # Side-by-side OCR|TX is forbidden by implementer.html §6.
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(comparison_widget)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(8)
        for label_text, editor in (
            ("OCR 原文", self._source_text),
            ("当前译文", self._current_translation_text),
            ("你的备注（问题说明）", self._note_text),
            ("你认可的译文（可直接填写）", self._corrected_translation_text),
        ):
            section_label = QLabel(label_text)
            section_label.setObjectName("feedbackSectionLabel")
            comparison_layout.addWidget(section_label)
            comparison_layout.addWidget(editor)
        self._provenance_label = QLabel("")
        self._provenance_label.setObjectName("hintLabel")
        self._provenance_label.setWordWrap(True)
        comparison_layout.addWidget(self._provenance_label)
        keyword_label = QLabel("关键词（用于查找相似纠错）")
        keyword_label.setObjectName("feedbackSectionLabel")
        comparison_layout.addWidget(keyword_label)
        comparison_layout.addWidget(self._keyword_editor)

        self._memory_panel = QWidget()
        self._memory_panel.setObjectName("feedbackMemoryPanel")
        memory_form = QFormLayout(self._memory_panel)
        memory_form.setContentsMargins(16, 16, 16, 12)
        memory_form.setSpacing(10)
        memory_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        memory_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        memory_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._memory_status_label = QLabel("")
        self._memory_status_label.setObjectName("feedbackMemoryStatus")
        self._memory_status_label.setWordWrap(True)
        self._memory_status_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._memory_status_label.setMinimumHeight(42)
        self._memory_status_label.setMinimumWidth(0)
        memory_status_policy = self._memory_status_label.sizePolicy()
        memory_status_policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        memory_status_policy.setVerticalPolicy(QSizePolicy.Policy.Preferred)
        memory_status_policy.setHeightForWidth(True)
        self._memory_status_label.setSizePolicy(memory_status_policy)
        self._memory_rule_combo = QComboBox()
        self._memory_rule_combo.setObjectName("feedbackMemoryRuleCombo")
        self._memory_rule_combo.setMinimumWidth(0)
        self._memory_rule_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._memory_rule_combo.currentIndexChanged.connect(
            self._on_memory_rule_selected
        )
        memory_form.addRow(_make_field_label("相关规则"), self._memory_rule_combo)
        memory_form.addRow(_make_field_label("当前状态"), self._memory_status_label)
        memory_form.addRow(_make_field_label("记忆规则"), self._rule_text)
        form = QVBoxLayout()
        form.setContentsMargins(16, 16, 16, 12)
        form.setSpacing(10)
        form.addWidget(comparison_widget)

        self._ai_optimize_button = QPushButton("让 AI 优化")
        self._accept_translation_button = QPushButton("提交修改")
        self._save_keywords_button = QPushButton("保存关键词")
        self._save_rule_button = QPushButton("保存规则")
        self._toggle_rule_button = QPushButton("停用规则")
        self._toggle_correction_button = QPushButton("停用纠错")
        self._dismiss_button = QPushButton("忽略")
        # Match mock btn-row: only「仅采用译文」is primary on the action bar;
        # 「让 AI 优化」is secondary and opens the review dialog.
        self._ai_optimize_button.setObjectName("secondaryButton")
        self._accept_translation_button.setObjectName("primaryButton")
        self._save_keywords_button.setObjectName("secondaryButton")
        self._save_rule_button.setObjectName("primaryButton")
        self._toggle_rule_button.setObjectName("secondaryButton")
        self._toggle_correction_button.setObjectName("secondaryButton")
        self._dismiss_button.setObjectName("secondaryButton")
        compact_widths = {
            self._ai_optimize_button: 100,
            self._accept_translation_button: 100,
            self._save_keywords_button: 100,
            self._save_rule_button: 100,
            self._toggle_rule_button: 90,
            self._toggle_correction_button: 90,
            self._dismiss_button: 64,
        }
        for button, width in compact_widths.items():
            button.setFixedHeight(34)
            button.setMaximumWidth(width)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)


        self._ai_optimize_button.clicked.connect(self._on_ai_optimize)
        self._accept_translation_button.clicked.connect(self._on_accept_translation)
        self._save_keywords_button.clicked.connect(self._on_save_keywords)
        self._save_rule_button.clicked.connect(self._on_save_rule)
        self._toggle_rule_button.clicked.connect(self._on_toggle_rule)
        self._toggle_correction_button.clicked.connect(self._on_toggle_correction)
        self._dismiss_button.clicked.connect(self._on_dismiss)
        memory_actions = QHBoxLayout()
        memory_actions.setContentsMargins(0, 0, 0, 0)
        memory_actions.setSpacing(8)
        memory_actions.addWidget(self._save_rule_button)
        memory_actions.addWidget(self._toggle_rule_button)
        memory_actions.addStretch(1)
        memory_form.addRow("", memory_actions)

        self._action_layout = QHBoxLayout()
        self._action_layout.setContentsMargins(0, 0, 0, 0)
        self._action_layout.setSpacing(8)
        self._action_layout.addWidget(self._ai_optimize_button)
        self._action_layout.addWidget(self._save_keywords_button)
        self._action_layout.addWidget(self._accept_translation_button)
        self._action_layout.addWidget(self._toggle_correction_button)
        self._action_layout.addStretch(1)
        self._action_layout.addWidget(self._dismiss_button)

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

        self._memory_scroll = QScrollArea()
        self._memory_scroll.setObjectName("feedbackMemoryScroll")
        self._memory_scroll.setWidgetResizable(True)
        self._memory_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._memory_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._memory_scroll.setWidget(self._memory_panel)

        self._feedback_tabs = QTabWidget()
        self._feedback_tabs.setObjectName("feedbackTabs")
        self._feedback_tabs.addTab(self._detail_scroll, "译文修正")
        self._feedback_tabs.addTab(self._memory_scroll, "长期记忆")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(picker_row)
        layout.addWidget(self._feedback_tabs, 1)
        layout.addLayout(self._action_layout)
        layout.addWidget(self._status_label)

        self.refresh()

    def refresh(
        self,
        preserve_current: bool = False,
        preserve_drafts: bool = False,
    ) -> None:
        """Reload pending feedback from local storage."""

        draft = self._capture_draft_state() if preserve_drafts else None
        current_id = (
            self._feedback_combo.currentData()
            if preserve_current or self._ai_optimizing
            else None
        )
        try:
            loaded_records = self._feedback_store.list_feedback()
        except FeedbackStorageUnavailable as exc:
            if draft is None or not draft["record_id"]:
                self._records = []
                self._feedback_combo.blockSignals(True)
                self._feedback_combo.clear()
                self._feedback_combo.blockSignals(False)
                self._on_feedback_selected(-1)
            else:
                self._set_actions_enabled(False)
            self._feedback_combo.setEnabled(False)
            self._status_label.setText(
                "反馈存储正在等待安全恢复；实时翻译仍可使用，"
                "但暂时不能读取或提交纠错。修复存储后请重启应用。"
            )
            get_logger().warning("Feedback page storage unavailable: %s", exc)
            return
        self._feedback_combo.setEnabled(True)
        self._records = [
            record for record in loaded_records if record.status != "dismissed"
        ]
        self._records.sort(key=lambda item: item.updated_at, reverse=True)
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
        if draft is not None and draft["record_id"] == self._feedback_combo.currentData():
            self._restore_draft_state(draft)

    def _capture_draft_state(self) -> dict[str, object]:
        """Capture unsaved editor state before an explicit data refresh."""

        return {
            "record_id": self._feedback_combo.currentData(),
            "note": self._note_text.toPlainText(),
            "translation": self._corrected_translation_text.toPlainText(),
            "keywords": self._keyword_editor.keywords(),
            "rule_id": self._active_memory_rule_id,
            "rule": self._rule_text.toPlainText(),
            "rule_modified": self._rule_text.document().isModified(),
            "tab": self._feedback_tabs.currentIndex(),
        }

    def _restore_draft_state(self, draft: dict[str, object]) -> None:
        """Restore drafts only when the refreshed record/rule identity is unchanged."""

        self._note_text.setPlainText(str(draft["note"]))
        self._corrected_translation_text.setPlainText(str(draft["translation"]))
        self._keyword_editor.set_keywords(list(draft["keywords"]))
        if (
            draft["rule_modified"]
            and draft["rule_id"]
            and draft["rule_id"] == self._active_memory_rule_id
        ):
            self._rule_text.setPlainText(str(draft["rule"]))
            self._rule_text.document().setModified(True)
        self._feedback_tabs.setCurrentIndex(int(draft["tab"]))

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
            self._provenance_label.setText("")
            self._memory_status_label.setText("暂无相关长期规则。")
            self._active_memory_rule_id = ""
            self._visible_memory_rules = []
            self._memory_rule_combo.clear()
            self._feedback_tabs.setCurrentIndex(0)
            self._feedback_tabs.setTabText(1, "长期记忆")
            self._set_actions_enabled(False)
            self._status_label.setText("暂无待优化的翻译。")
            return

        self._source_text.setPlainText(record.ocr_text)
        self._current_translation_text.setPlainText(record.translation_text)
        self._note_text.setPlainText(record.note)
        self._corrected_translation_text.setPlainText(record.corrected_translation)
        self._provenance_label.setText(
            "本次错误译文实际使用："
            f"纠错案例 {len(record.matched_correction_ids or [])} 条，"
            f"长期规则 {len(record.matched_memory_rule_ids or [])} 条。"
        )
        self._set_keyword_options(list(record.keywords or []), "")
        self._ai_problem_summary = record.ai_problem_summary
        storage_available = True
        try:
            self._load_memory_for_record(record)
        except FeedbackStorageUnavailable as exc:
            storage_available = False
            self._active_memory_rule_id = ""
            self._visible_memory_rules = []
            self._memory_rule_combo.clear()
            self._rule_text.clear()
            self._memory_status_label.setText(
                "反馈存储暂不可用，当前无法读取或修改长期规则。"
            )
            get_logger().warning("Feedback memory storage unavailable: %s", exc)
        self._set_actions_enabled(storage_available)
        self._feedback_tabs.setCurrentIndex(0)
        self._feedback_tabs.setTabText(1, "长期记忆")
        self._status_label.setText(f"已载入 {record.summary()}")

    def _set_actions_enabled(self, enabled: bool) -> None:
        enabled = enabled and not self._ai_optimizing
        for widget in (
            self._ai_optimize_button,
            self._accept_translation_button,
            self._save_keywords_button,
            self._dismiss_button,
            self._note_text,
            self._corrected_translation_text,
            self._keyword_editor,
            self._rule_text,
            self._memory_rule_combo,
        ):
            widget.setEnabled(enabled)
        self._save_rule_button.setEnabled(enabled and bool(self._active_memory_rule_id))
        self._toggle_rule_button.setEnabled(enabled and bool(self._active_memory_rule_id))
        record = self._current_record()
        submitted = record is not None and record.status in {"accepted", "confirmed"}
        self._toggle_correction_button.setEnabled(enabled and submitted)
        self._toggle_correction_button.setText(
            "停用纠错" if record is None or record.enabled else "启用纠错"
        )
        self._dismiss_button.setEnabled(
            enabled and record is not None and record.status == "pending"
        )
        self._dismiss_button.setText(
            "忽略" if record is None or record.status == "pending" else "已提交"
        )

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

    def _load_memory_for_record(self, record: FeedbackRecord) -> None:
        selected_rule_id = self._active_memory_rule_id
        ids = [
            record.memory_rule_id,
            *(record.matched_memory_rule_ids or []),
        ]
        rules: list[MemoryRule] = []
        seen: set[str] = set()
        for rule_id in ids:
            if not rule_id or rule_id in seen:
                continue
            seen.add(rule_id)
            memory = self._feedback_store.get_memory_rule(rule_id)
            if memory is not None:
                rules.append(memory)
        # Secondary evidence records do not own memory_rule_id.  Include rules
        # that cite this record so users can inspect or correct the rule that
        # was actually derived from their feedback.
        for memory in self._feedback_store.list_memory_rules(enabled_only=False):
            if memory.id in seen or record.id not in (memory.source_feedback_ids or []):
                continue
            seen.add(memory.id)
            rules.append(memory)
        if not rules:
            rules = self._feedback_store.match_memory_rules(
                record.ocr_text,
                source_language=record.source_language,
                target_language=record.target_language,
                limit=3,
            )
        self._visible_memory_rules = rules
        self._memory_rule_combo.blockSignals(True)
        self._memory_rule_combo.clear()
        for memory in rules:
            versioned = bool(
                memory.source_feedback_ids
                and set(memory.source_feedback_digests or {})
                == set(memory.source_feedback_ids or [])
            )
            origin = (
                "用户"
                if memory.user_locked
                else ("自动" if versioned else "旧自动规则·待确认")
            )
            state = "启用" if memory.enabled else "停用"
            trigger = memory.trigger.strip() or "无触发词"
            self._memory_rule_combo.addItem(
                f"{origin} · {state} · {trigger}",
                memory.id,
            )
        self._memory_rule_combo.blockSignals(False)
        if not rules:
            self._active_memory_rule_id = ""
            self._toggle_rule_button.setText("停用规则")
            self._rule_text.setPlainText("")
            if record.consolidation_status in {"queued", "running"}:
                self._memory_status_label.setText("后台正在根据纠错库归纳长期规则。")
            elif record.consolidation_status == "failed":
                self._memory_status_label.setText(
                    f"纠错案例已生效，但长期规则归纳失败：{record.consolidation_error}"
                )
            else:
                self._memory_status_label.setText("本次翻译未使用长期规则。")
            return
        selected_index = self._memory_rule_combo.findData(selected_rule_id)
        if selected_index < 0:
            selected_index = 0
        self._memory_rule_combo.setCurrentIndex(selected_index)
        self._show_memory_rule(rules[selected_index], record)

    def _reload_memory_after_durable_action(
        self,
        record: FeedbackRecord,
        *,
        draft: dict[str, object] | None = None,
    ) -> bool:
        """Refresh memory UI without misreporting an already-durable action."""

        try:
            self._load_memory_for_record(record)
        except FeedbackStorageUnavailable as exc:
            self._set_actions_enabled(False)
            self._status_label.setText(
                "操作已经保存，但反馈存储暂时无法刷新；请稍后点击刷新。"
            )
            get_logger().warning(
                "Feedback memory refresh unavailable after durable action: %s",
                exc,
            )
            return False
        except Exception as exc:
            self._status_label.setText(
                f"操作已经保存，但长期规则界面刷新失败：{exc}"
            )
            get_logger().exception(
                "Feedback memory refresh failed after durable action"
            )
            return False
        if draft is not None and draft["record_id"] == record.id:
            self._restore_draft_state(draft)
        return True

    def _on_memory_rule_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._visible_memory_rules):
            return
        record = self._current_record()
        if record is None:
            return
        self._show_memory_rule(self._visible_memory_rules[index], record)
        self._set_actions_enabled(True)

    def _show_memory_rule(self, memory: MemoryRule, record: FeedbackRecord) -> None:
        self._active_memory_rule_id = memory.id
        self._rule_text.setPlainText(memory.rule)
        self._toggle_rule_button.setText("停用规则" if memory.enabled else "启用规则")
        versioned = bool(
            memory.source_feedback_ids
            and set(memory.source_feedback_digests or {})
            == set(memory.source_feedback_ids or [])
        )
        if memory.user_locked:
            origin = "用户已修改"
        elif versioned:
            origin = "后台自动归纳"
        else:
            origin = "旧自动规则未注入运行时；请修改并保存后再启用"
        used = "；本次错误翻译实际注入过" if memory.id in (record.matched_memory_rule_ids or []) else ""
        self._memory_status_label.setText(
            f"{origin}；来源纠错 {len(memory.source_feedback_ids or []) or 1} 条{used}。"
        )

    def _report_feedback_action_failure(self, action: str, exc: Exception) -> None:
        get_logger().exception("Feedback UI action failed | action=%s", action)
        self._status_label.setText(f"{action}失败：{exc}")

    def _on_save_keywords(self) -> None:
        record = self._current_record()
        if record is None:
            return
        try:
            keywords = self._keyword_editor.keywords()
            if record.status in {"accepted", "confirmed"} and record.corrected_translation.strip():
                updated = self._learning_service.submit_correction(
                    record.id,
                    corrected_translation=record.corrected_translation,
                    note=self._note_text.toPlainText().strip(),
                    keywords=keywords,
                    ai_problem_summary=self._ai_problem_summary,
                    on_complete=self._consolidation_done.emit,
                )
            else:
                updated = self._feedback_store.save_keywords(record.id, keywords)
        except Exception as exc:
            self._report_feedback_action_failure("保存关键词", exc)
            return
        self._records[self._feedback_combo.currentIndex()] = updated
        if updated.consolidation_status == "failed":
            self._status_label.setText(
                "关键词已保存并会参与相似纠错检索；"
                f"长期规则归纳失败：{updated.consolidation_error}"
            )
        else:
            self._status_label.setText("关键词已保存；后续相似纠错检索会使用这些关键词。")

    def _on_save_rule(self) -> None:
        if not self._active_memory_rule_id:
            self._status_label.setText("当前没有可修改的长期规则。")
            return
        try:
            memory = self._feedback_store.update_memory_rule(
                self._active_memory_rule_id,
                rule_text=self._rule_text.toPlainText(),
                user_locked=True,
            )
        except Exception as exc:
            self._report_feedback_action_failure("保存规则", exc)
            return
        self._memory_status_label.setText(
            f"用户已修改并锁定；来源纠错 {len(memory.source_feedback_ids or []) or 1} 条。"
        )
        record = self._current_record()
        if record is not None and not self._reload_memory_after_durable_action(record):
            return
        self._status_label.setText("长期规则已保存；后台 Pro 不会自动覆盖用户修改。")

    def _on_toggle_rule(self) -> None:
        if not self._active_memory_rule_id:
            self._status_label.setText("当前没有可停用的长期规则。")
            return
        draft = self._capture_draft_state()
        try:
            memory = self._feedback_store.get_memory_rule(self._active_memory_rule_id)
            if memory is None:
                self._status_label.setText("长期规则已不存在，请刷新。")
                return
            memory = self._feedback_store.update_memory_rule(
                memory.id,
                enabled=not memory.enabled,
            )
        except Exception as exc:
            self._report_feedback_action_failure("修改规则状态", exc)
            return
        self._toggle_rule_button.setText("停用规则" if memory.enabled else "启用规则")
        state = "启用" if memory.enabled else "停用"
        self._memory_status_label.setText(f"规则已由用户{state}并锁定。")
        record = self._current_record()
        if record is not None and not self._reload_memory_after_durable_action(
            record,
            draft=draft,
        ):
            return
        self._memory_status_label.setText(
            f"规则已由用户{state}；自动规则仍会按证据有效性和触发范围校验。"
        )
        self._status_label.setText(f"长期规则已{state}。")

    def _record_with_current_drafts(self) -> FeedbackRecord | None:
        record = self._current_record()
        if record is None:
            return None
        return replace(
            record,
            note=self._note_text.toPlainText().strip(),
            corrected_translation=self._corrected_translation_text.toPlainText().strip(),
            keywords=self._keyword_editor.keywords(),
        )

    def _on_ai_optimize(self) -> None:
        record = self._record_with_current_drafts()
        with self._ai_state_lock:
            if record is None or self._ai_optimizing or self._closing:
                return
            self._ai_optimizing = True
            self._ai_job_id += 1
            job_id = self._ai_job_id
        record_id = record.id
        self._status_label.setText("正在让思考模型优化...")
        self._feedback_combo.setEnabled(False)
        self._refresh_button.setEnabled(False)
        self._set_actions_enabled(False)
        self.ai_work_started.emit()

        def _run() -> None:
            payload: dict = {"job_id": job_id, "record_id": record_id}
            try:
                payload["suggestion"] = self._optimizer.optimize(self._settings, record)
            except TranslationError as exc:
                payload["error"] = str(exc)
            except Exception as exc:
                payload["error"] = str(exc)
            # The page may have been closed while the daemon was waiting on a
            # remote request.  Never enqueue UI work for a closed/stale page.
            with self._ai_state_lock:
                should_emit = not self._closing and job_id == self._ai_job_id
            if should_emit:
                # Do not invoke Qt/external code while holding the state lock.
                # A concurrent shutdown may win after this check; the slot
                # repeats the closing/token check before touching widgets.
                try:
                    self._ai_work_done.emit(payload)
                except RuntimeError:
                    # PySide raises when the QObject was deleted in the tiny
                    # interval after the closing/token check.
                    return

        worker = threading.Thread(
            target=_run,
            name=f"feedback-ai-review-{job_id}",
            daemon=True,
        )
        with self._ai_state_lock:
            if self._closing or job_id != self._ai_job_id:
                self._ai_optimizing = False
                return
            self._ai_worker_thread = worker
        try:
            worker.start()
        except Exception as exc:
            with self._ai_state_lock:
                if job_id == self._ai_job_id:
                    self._ai_worker_thread = None
                    self._ai_optimizing = False
                closed = self._closing
            if not closed:
                self._feedback_combo.setEnabled(True)
                self._refresh_button.setEnabled(True)
                self._set_actions_enabled(self._current_record() is not None)
                self._status_label.setText(f"AI 优化启动失败：{exc}")
                self.ai_work_finished.emit()

    def _on_ai_work_done(self, payload: dict) -> None:
        with self._ai_state_lock:
            if self._closing or payload.get("job_id") != self._ai_job_id:
                return
            self._ai_worker_thread = None
            self._ai_optimizing = False
        self._feedback_combo.setEnabled(True)
        self._refresh_button.setEnabled(True)
        self._set_actions_enabled(self._current_record() is not None)
        self.ai_work_finished.emit()

        error = payload.get("error")
        if error:
            self._status_label.setText(f"AI 优化失败：{error}")
            return

        record = self._current_record()
        if record is None or record.id != payload.get("record_id"):
            self._status_label.setText("AI 优化已完成，但原反馈条目已发生变化，请重新选择。")
            return

        suggestion = payload.get("suggestion")
        if not isinstance(suggestion, FeedbackOptimization):
            self._status_label.setText("AI 优化失败：返回结果无效。")
            return
        dialog = self._dialog_factory(suggestion, self)
        dialog.exec()
        with self._ai_state_lock:
            if self._closing or payload.get("job_id") != self._ai_job_id:
                return
        if dialog.decision != FeedbackOptimizationDialog.COMPLETED:
            self._status_label.setText("未采用 AI 建议，主页面内容保持不变。")
            return

        translation_choice = dialog.review_choice(0)
        problem_choice = dialog.review_choice(1)
        keyword_choice = dialog.review_choice(2)
        if translation_choice and translation_choice["state"] != FeedbackOptimizationDialog.SKIPPED:
            translation_value = str(translation_choice["value"] or "").strip()
            if translation_value:
                self._corrected_translation_text.setPlainText(translation_value)
        if keyword_choice and keyword_choice["state"] != FeedbackOptimizationDialog.SKIPPED:
            keyword_value = list(keyword_choice["value"] or [])
            if keyword_value:
                self._keyword_editor.set_keywords(keyword_value)
        if problem_choice and problem_choice["state"] != FeedbackOptimizationDialog.SKIPPED:
            self._ai_problem_summary = str(problem_choice["value"] or "").strip()
        self._on_accept_translation()

    def _on_accept_translation(self) -> None:
        record = self._current_record()
        if record is None:
            return
        translation = self._corrected_translation_text.toPlainText().strip()
        if not translation:
            self._status_label.setText("请先填写非空的优化后/认可译文。")
            return
        try:
            accepted = self._learning_service.submit_correction(
                record.id,
                corrected_translation=translation,
                note=self._note_text.toPlainText().strip(),
                keywords=self._keyword_editor.keywords(),
                ai_problem_summary=self._ai_problem_summary,
                on_complete=self._consolidation_done.emit,
            )
        except Exception as exc:
            self._report_feedback_action_failure("提交修改", exc)
            return
        self.refresh(preserve_current=True)
        if accepted.consolidation_status == "failed":
            self._status_label.setText(
                f"修改已提交并开始生效；长期规则尚未归纳：{accepted.consolidation_error}"
            )
        elif accepted.consolidation_status == "completed":
            self._status_label.setText("修改已提交并开始生效；长期规则无需重复归纳。")
        else:
            self._status_label.setText("修改已提交并开始生效；后台正在自动归纳长期规则。")

    def _on_confirm(self) -> None:
        """Backward-compatible internal alias for submitting a correction."""

        self._on_accept_translation()

    def _on_consolidation_done(self, payload: dict) -> None:
        if self._closing:
            return
        current = self._current_record()
        if current is None or current.id != payload.get("feedback_id"):
            feedback_id = payload.get("feedback_id")
            if isinstance(feedback_id, str):
                try:
                    updated = self._feedback_store.get_feedback(feedback_id)
                except FeedbackStorageUnavailable:
                    updated = None
                if updated is not None:
                    for index, cached in enumerate(self._records):
                        if cached.id != feedback_id:
                            continue
                        self._records[index] = updated
                        self._feedback_combo.blockSignals(True)
                        self._feedback_combo.setItemText(index, updated.summary())
                        self._feedback_combo.blockSignals(False)
                        break
            # A background completion belongs to another feedback item.  Its
            # durable state will be visible on refresh/selection; it must not
            # overwrite the status or drafts of the item currently reviewed.
            return
        index = self._feedback_combo.currentIndex()
        active_rule_id = self._active_memory_rule_id
        rule_draft = self._rule_text.toPlainText()
        rule_draft_modified = self._rule_text.document().isModified()
        current_tab = self._feedback_tabs.currentIndex()
        try:
            updated = self._feedback_store.get_feedback(current.id)
            if updated is None:
                self.refresh(preserve_current=True)
                self._status_label.setText("当前纠错记录已被删除，请重新选择。")
                return
            self._records[index] = updated
            self._feedback_combo.blockSignals(True)
            self._feedback_combo.setItemText(index, updated.summary())
            self._feedback_combo.blockSignals(False)
            # Refresh only durable status/provenance.  A full page refresh
            # would erase note/translation/keyword drafts typed while Pro ran.
            self._load_memory_for_record(updated)
            if (
                rule_draft_modified
                and active_rule_id
                and self._active_memory_rule_id == active_rule_id
            ):
                self._rule_text.setPlainText(rule_draft)
                self._rule_text.document().setModified(True)
            self._feedback_tabs.setCurrentIndex(current_tab)
            self._set_actions_enabled(True)
        except FeedbackStorageUnavailable as exc:
            self._status_label.setText(f"纠错已处理，但反馈存储暂时无法刷新：{exc}")
            return
        except Exception as exc:
            self._report_feedback_action_failure("刷新长期规则", exc)
            return
        error = payload.get("error")
        if error:
            self._status_label.setText(
                f"纠错案例已经生效；后台长期规则归纳失败：{error}"
            )
        elif payload.get("skipped") == "stale":
            self._status_label.setText(
                payload.get("message")
                or "纠错案例已生效；证据发生变化，本次长期规则未保存。"
            )
        elif payload.get("skipped") == "unchanged":
            self._status_label.setText("纠错内容未变化，无需重复归纳长期规则。")
        else:
            self._status_label.setText("纠错案例和后台长期规则均已更新。")

    def _on_toggle_correction(self) -> None:
        record = self._current_record()
        if record is None or record.status not in {"accepted", "confirmed"}:
            self._status_label.setText("当前没有可启用或停用的已提交纠错。")
            return
        try:
            updated = self._learning_service.set_correction_enabled(
                record.id,
                not record.enabled,
            )
        except Exception as exc:
            self._report_feedback_action_failure("修改纠错状态", exc)
            return
        self.refresh(preserve_current=True, preserve_drafts=True)
        state = "启用" if updated.enabled else "停用"
        suffix = (
            "；由它生成且未被用户锁定的自动规则也已停用。"
            if not updated.enabled
            else "；该纠错会重新参与后续检索。"
        )
        self._status_label.setText(f"纠错案例已{state}{suffix}")

    def _on_dismiss(self) -> None:
        record = self._current_record()
        if record is None:
            return
        if record.status != "pending":
            self._status_label.setText("已生效的纠错不能用“忽略”撤销；请单独停用长期规则。")
            return
        try:
            self._feedback_store.update_feedback(record.id, status="dismissed")
        except Exception as exc:
            self._report_feedback_action_failure("忽略反馈", exc)
            return
        self._status_label.setText("已忽略该反馈。")
        self.refresh()

    def shutdown(self) -> None:
        with self._ai_state_lock:
            self._closing = True
            self._ai_job_id += 1
            self._ai_optimizing = False
        self._learning_service.shutdown(wait=False)


class SettingsPage(QWidget):
    """Shortcut configuration (persisted)."""

    save_requested = Signal(str, str)

    def __init__(self, create_shortcut: str = "Ctrl+Shift+Z", edit_shortcut: str = "Ctrl+Shift+X", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contentPage")

        title = QLabel("快捷键设置")
        title.setObjectName("pageTitle")
        desc = QLabel("修改全局快捷键（保存后即时生效）")
        desc.setObjectName("pageDesc")

        self._create_input = QLineEdit(create_shortcut)
        self._edit_input = QLineEdit(edit_shortcut)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(14)
        form.addRow(_make_field_label("新建选择框"), self._create_input)
        form.addRow(_make_field_label("切换编辑模式"), self._edit_input)

        shortcuts_card, shortcuts_layout = _make_content_card()
        shortcuts_layout.addLayout(form)

        hint = QLabel("格式示例：Ctrl+Shift+Z、Alt+F1。保存时自动检测是否被占用。")
        hint.setObjectName("hintLabel")

        self._save_status = QLabel("")

        save_btn = QPushButton("保存快捷键")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self._on_save)

        btn_row = QHBoxLayout()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self._save_status)
        btn_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(shortcuts_card)
        layout.addLayout(btn_row)
        layout.addWidget(hint)
        layout.addStretch(1)

    def _on_save(self) -> None:
        create_key = self._create_input.text().strip()
        edit_key = self._edit_input.text().strip()
        if not create_key or not edit_key:
            self._show_error("快捷键不能为空")
            return
        try:
            from app.hotkeys import GlobalHotkeyService
            GlobalHotkeyService.parse_shortcut(create_key)
            GlobalHotkeyService.parse_shortcut(edit_key)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        if create_key.lower() == edit_key.lower():
            self._show_error("两个快捷键不能相同")
            return
        self.save_requested.emit(create_key, edit_key)

    def show_save_result(self, success: bool, message: str) -> None:
        if success:
            self._save_status.setText(f"{message} ✓")
            self._save_status.setStyleSheet("color: #4ade80; font-size: 12px;")
        else:
            self._save_status.setText(message)
            self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")

    def _show_error(self, message: str) -> None:
        self._save_status.setText(message)
        self._save_status.setStyleSheet("color: #f87171; font-size: 12px;")


# =========================================================================
# main window
# =========================================================================

class MainWindow(QMainWindow):
    """Sidebar-navigated control panel."""

    default_language_changed = Signal(str, str)
    compiled_prompt_saved = Signal()
    model_config_changed = Signal()
    hotkeys_save_requested = Signal(str, str)

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
        # The reference shell is 1060x620.  Starting at the same density keeps
        # cards from looking oversized while the minimum still supports small
        # laptop screens.
        self.resize(1060, 620)
        self.setMinimumSize(760, 480)

        self._apply_stylesheet()

        # title bar
        self._title_bar = TitleBar()
        self._title_bar.minimize_requested.connect(self.showMinimized)
        self._title_bar.close_requested.connect(self.close)

        # sidebar
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(200)

        self._nav_template = NavButton("☰", "翻译模板")
        self._nav_model = NavButton("◈", "模型配置")
        self._nav_feedback = NavButton("✎", "优化翻译")
        self._nav_settings = NavButton("⚙", "设置")
        self._nav_buttons = [
            self._nav_template,
            self._nav_model,
            self._nav_feedback,
            self._nav_settings,
        ]
        for btn in self._nav_buttons:
            btn.clicked.connect(self._on_nav_clicked)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 0, 10, 12)
        sidebar_layout.setSpacing(4)
        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(self._nav_template)
        sidebar_layout.addWidget(self._nav_model)
        sidebar_layout.addWidget(self._nav_feedback)
        sidebar_layout.addWidget(self._nav_settings)
        sidebar_layout.addStretch(1)

        # pages
        self._stack = QStackedWidget()
        self._stack.setObjectName("pageStack")

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
        self._template_page.compiled_prompt_saved.connect(self.compiled_prompt_saved)
        self._template_page.ai_work_started.connect(self._show_ai_progress)
        self._template_page.ai_work_finished.connect(self._hide_ai_progress)
        self._feedback_page.ai_work_started.connect(self._show_ai_progress)
        self._feedback_page.ai_work_finished.connect(self._hide_ai_progress)
        self._model_page.config_changed.connect(self._on_model_changed)
        self._settings_page.save_requested.connect(self.hotkeys_save_requested)

        # status bar
        self._status = QStatusBar()
        self._status.setObjectName("appStatusBar")
        self._status.showMessage(f"就绪 — {context.hotkeys.create_selection} 新建框选")

        self._ai_progress = QProgressBar()
        self._ai_progress.setObjectName("aiProgressBar")
        self._ai_progress.setFixedWidth(140)
        self._ai_progress.setFixedHeight(8)
        self._ai_progress.setTextVisible(False)
        self._ai_progress.setRange(0, 100)
        self._ai_progress.setValue(0)
        self._ai_progress.setVisible(False)
        self._status.addPermanentWidget(self._ai_progress)

        self._ai_progress_timer = QTimer()
        self._ai_progress_timer.setInterval(300)
        self._ai_progress_timer.timeout.connect(self._tick_ai_progress)
        self._ai_progress_value = 0

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
        central.setObjectName("appRoot")
        central.setLayout(root)
        self.setCentralWidget(central)

        # initial nav
        self._nav_template.setChecked(True)
        self._stack.setCurrentIndex(0)
        self._title_bar.set_page_name("翻译模板")

    # ------------------------------------------------------------------
    # nav
    # ------------------------------------------------------------------

    def _on_nav_clicked(self) -> None:
        sender = self.sender()
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(btn is sender)
        if sender is self._nav_template:
            self._stack.setCurrentIndex(0)
            self._title_bar.set_page_name("翻译模板")
        elif sender is self._nav_model:
            self._model_page.ensure_models_loaded()
            self._stack.setCurrentIndex(1)
            self._title_bar.set_page_name("模型配置")
        elif sender is self._nav_feedback:
            self._feedback_page.refresh(preserve_current=True, preserve_drafts=True)
            self._stack.setCurrentIndex(2)
            self._title_bar.set_page_name("优化翻译")
        elif sender is self._nav_settings:
            self._stack.setCurrentIndex(3)
            self._title_bar.set_page_name("设置")

    # ------------------------------------------------------------------
    # handlers — real persistence
    # ------------------------------------------------------------------

    def _on_language_changed(self, source: str, target: str) -> None:
        s = self._context.settings
        previous = (s.default_source_language, s.default_target_language)
        s.default_source_language = source
        s.default_target_language = target
        try:
            s.save()
        except OSError as exc:
            s.default_source_language, s.default_target_language = previous
            get_logger().error("Default language settings save failed: %s", exc)
            self._status.showMessage(f"默认翻译方向保存失败：{exc}", 5000)
            return
        self._context.default_source_language = source
        self._context.default_target_language = target
        self.default_language_changed.emit(source, target)
        self._status.showMessage(f"默认翻译方向已保存: {source} → {target}", 3000)

    def _on_model_changed(self, base_url: str, api_key: str, fast_model: str, thinking_model: str) -> None:
        s = self._context.settings
        previous = (
            s.ai.base_url,
            s.ai.api_key,
            s.ai.fast_model,
            s.ai.thinking_model,
            s.ai.model,
        )
        s.ai.base_url = base_url
        s.ai.api_key = api_key
        s.ai.fast_model = fast_model
        s.ai.thinking_model = thinking_model
        s.ai.model = fast_model or thinking_model
        try:
            s.save()
        except OSError as exc:
            (
                s.ai.base_url,
                s.ai.api_key,
                s.ai.fast_model,
                s.ai.thinking_model,
                s.ai.model,
            ) = previous
            get_logger().error("Model settings save failed: %s", exc)
            self._model_page.show_save_result(False, "配置保存失败；未写入磁盘。")
            return
        self._model_page.show_save_result(True, "模型配置已保存")
        current = (base_url, api_key, fast_model, thinking_model)
        if current != previous[:4]:
            self.model_config_changed.emit()

    def show_hotkey_result(self, success: bool, message: str) -> None:
        self._settings_page.show_save_result(success, message)
        if success:
            self._status.showMessage(f"快捷键已更新 — {self._context.hotkeys.create_selection} 新建框选", 3000)

    # ------------------------------------------------------------------
    # AI progress bar
    # ------------------------------------------------------------------

    def _show_ai_progress(self) -> None:
        self._ai_progress_value = 0
        self._ai_progress.setValue(0)
        self._ai_progress.setVisible(True)
        self._ai_progress_timer.start()

    def _hide_ai_progress(self) -> None:
        self._ai_progress_timer.stop()
        self._ai_progress.setValue(100)
        QTimer.singleShot(500, lambda: self._ai_progress.setVisible(False))

    def _tick_ai_progress(self) -> None:
        if self._ai_progress_value < 95:
            self._ai_progress_value += 2
            if self._ai_progress_value > 95:
                self._ai_progress_value = 95
            self._ai_progress.setValue(self._ai_progress_value)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def refresh_runtime_state(self) -> None:
        active = self._context.active_group_count
        edit = "编辑中" if self._context.edit_mode_enabled else "普通"
        self._status.showMessage(f"选择框: {active}/3  |  模式: {edit}")

    def refresh_feedback_records(self) -> None:
        """Reload feedback data without discarding unsaved editor drafts."""

        self._feedback_page.refresh(preserve_current=True, preserve_drafts=True)

    def default_language_pair(self) -> tuple[str, str]:
        return self._template_page.current_pair()

    def show_window(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def shutdown_background_tasks(self) -> None:
        self._template_page.shutdown()
        self._model_page.shutdown()
        self._feedback_page.shutdown()
        self._ai_progress_timer.stop()

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
                background: transparent;
                color: #e2e8f0;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 13px;
            }
            QDialog,
            QWidget#appRoot,
            QStackedWidget#pageStack,
            QWidget#contentPage {
                background: #0f172a;
            }
            QScrollArea#contentScroll,
            QScrollArea#contentScroll > QWidget > QWidget,
            QWidget#contentScrollBody {
                background: #0f172a;
                border: none;
            }
            QLabel {
                background: transparent;
                border: none;
            }

            QWidget#titleBar {
                background: #0a0f1a;
                border-bottom: 1px solid #1e293b;
            }
            QLabel#titleBarBrand {
                background: transparent;
                color: #f1f5f9;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }
            QLabel#titleBarDot {
                background: #60a5fa;
                border: none;
                border-radius: 4px;
            }
            QLabel#titleBarSeparator {
                background: transparent;
                color: #64748b;
                font-size: 13px;
            }
            QLabel#titleBarPage {
                background: transparent;
                color: #94a3b8;
                font-size: 12px;
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
                border-left: 3px solid transparent;
                border-radius: 6px;
                margin: 0;
                padding: 11px 14px 11px 12px;
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
                border-left: 3px solid #60a5fa;
            }

            QWidget#reviewSidebar {
                background: #0a0f1a;
                border-right: 1px solid #1e293b;
            }
            QPushButton#reviewNavButton {
                background: transparent;
                color: #e2e8f0;
                border: none;
                border-radius: 6px;
                padding: 8px 10px;
                text-align: left;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton#reviewNavButton:hover {
                background: #1e293b;
            }
            QPushButton#reviewNavButton:checked {
                background: #1e293b;
                color: #f1f5f9;
                font-weight: 600;
            }
            QPushButton#reviewNavButton[reviewState="pending"] {
                color: #94a3b8;
            }
            QPushButton#reviewNavButton[reviewState="pending"]:checked {
                color: #e2e8f0;
            }
            QPushButton#reviewNavButton[reviewState="skipped"] {
                color: #f87171;
            }
            QPushButton#reviewNavButton[reviewState="skipped"]:checked {
                color: #fca5a5;
            }
            QPushButton#reviewNavButton[reviewState="done"] {
                color: #4ade80;
            }
            QPushButton#reviewNavButton[reviewState="done"]:checked {
                color: #86efac;
            }

            QWidget#contentPage {
                background: #0f172a;
            }
            QFrame#contentCard {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
            }
            QLabel#cardTitle {
                background: transparent;
                color: #94a3b8;
                font-size: 13px;
                font-weight: 600;
            }
            QScrollArea#feedbackDetailScroll,
            QScrollArea#feedbackMemoryScroll,
            QWidget#feedbackDetailWidget,
            QWidget#feedbackMemoryPanel {
                background: #0f172a;
                border: none;
            }
            QDialog#compiledPromptDialog {
                background: #0f172a;
                color: #e2e8f0;
            }
            QPlainTextEdit#compiledPromptViewer {
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QTabWidget#feedbackTabs::pane,
            QTabWidget#compiledPromptTabs::pane {
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 10px;
                top: -1px;
                padding: 4px;
            }
            QTabWidget#feedbackTabs QTabBar::tab,
            QTabWidget#compiledPromptTabs QTabBar::tab {
                background: transparent;
                color: #94a3b8;
                border: none;
                border-bottom: 2px solid transparent;
                padding: 8px 14px;
                margin-right: 4px;
                font-size: 13px;
                font-weight: 500;
            }
            QTabWidget#feedbackTabs QTabBar::tab:selected,
            QTabWidget#compiledPromptTabs QTabBar::tab:selected {
                color: #60a5fa;
                font-weight: 600;
                border-bottom: 2px solid #60a5fa;
            }
            QTabWidget#feedbackTabs QTabBar::tab:hover,
            QTabWidget#compiledPromptTabs QTabBar::tab:hover {
                color: #e2e8f0;
            }
            QLabel#feedbackSectionLabel {
                color: #94a3b8;
                font-weight: 600;
                font-size: 12px;
            }
            QLabel#fieldLabel {
                color: #94a3b8;
                font-weight: 600;
                font-size: 12px;
            }
            QLabel#feedbackMemoryRecommendation {
                color: #94a3b8;
                font-size: 13px;
                background: transparent;
                border: none;
                padding: 0;
            }
            QLabel#feedbackMemoryStatus {
                color: #94a3b8;
                font-size: 12px;
                background: transparent;
                border: none;
                padding: 4px 0;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 2px 2px 2px 0;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 4px;
                min-height: 28px;
            }
            QScrollBar::handle:vertical:hover {
                background: #475569;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
                border: none;
                background: none;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 10px;
                margin: 0 2px 2px 2px;
            }
            QScrollBar::handle:horizontal {
                background: #334155;
                border-radius: 4px;
                min-width: 28px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #475569;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0;
                border: none;
                background: none;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: transparent;
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
            QLabel#connectionStatusPill {
                background: rgba(251, 191, 36, 0.12);
                color: #fbbf24;
                border: none;
                border-radius: 11px;
                padding: 4px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#connectionStatusPill[configured="true"] {
                background: rgba(74, 222, 128, 0.12);
                color: #4ade80;
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

            QComboBox {
                background: #0f172a;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px 28px 6px 12px;
                font-size: 13px;
                min-height: 20px;
            }
            QComboBox:hover {
                border: 1px solid #475569;
            }
            QComboBox:focus {
                border: 1px solid #60a5fa;
            }
            QComboBox:disabled {
                color: #64748b;
                background: #0f172a;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 26px;
                border: none;
                background: transparent;
            }
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #94a3b8;
                margin-right: 8px;
            }
            QComboBox::down-arrow:on {
                border-top-color: #60a5fa;
            }
            QComboBox QAbstractItemView {
                background: #0f172a;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                selection-background-color: #1e293b;
                selection-color: #60a5fa;
                outline: 0;
                padding: 4px;
            }
            QComboBox#modelSelectCombo {
                min-height: 22px;
            }
            QLineEdit, QPlainTextEdit {
                background: #0f172a;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 7px 12px;
                font-size: 13px;
            }
            QLineEdit:focus, QPlainTextEdit:focus {
                border: 1px solid #60a5fa;
            }

            QPushButton#secondaryButton {
                background: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 8px;
                min-height: 34px;
                padding: 0 14px;
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
                min-height: 34px;
                padding: 0 14px;
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

            QProgressBar#aiProgressBar {
                background: #1e293b;
                border: none;
                border-radius: 4px;
                max-height: 8px;
                min-height: 8px;
            }
            QProgressBar#aiProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2563eb,
                    stop:1 #60a5fa
                );
                border-radius: 4px;
            }

            QStatusBar#appStatusBar {
                background: #0a0f1a;
                color: #64748b;
                border-top: 1px solid #1e293b;
                font-size: 12px;
                padding: 2px 16px;
            }
            QStatusBar#appStatusBar::item {
                border: none;
            }
            """
        )
