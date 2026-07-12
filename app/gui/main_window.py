"""Main window — sidebar navigation + functional config pages."""

from __future__ import annotations

import json
import threading
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

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("navButton")
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

    # Prompt methods (from PromptPage)
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
        settings = self._settings or AppSettings.load()
        self._apply_form_to_settings(settings)
        constraints = PromptConstraints(text=self._constraints_text)
        references = self._collect_knowledge_references()

        self._save_status.setText("正在优化提示词...")
        self._save_status.setStyleSheet("color: #fbbf24; font-size: 12px;")
        self._preview_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self.ai_work_started.emit()

        def _run() -> None:
            result_holder: dict = {}
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
            self._ai_work_done.emit(result_holder, settings)

        threading.Thread(target=_run, daemon=True).start()

    def _on_ai_work_done(self, result_holder: dict, settings: object) -> None:
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
            saved_path = self._prompt_storage.save_compiled_prompt(
                compiled.content,
                settings.prompt.compiled_prompt_path,
                policy=compiled.policy,
                reference_package=compiled.reference_package,
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
            self.compiled_prompt_saved.emit()
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
        dlg.setWindowTitle("确认翻译模板")
        dlg.resize(680, 500)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 16)

        tabs = QTabWidget()
        easy_viewer = QPlainTextEdit()
        easy_viewer.setReadOnly(True)
        easy_viewer.setPlainText(str(content))
        advanced_viewer = QPlainTextEdit()
        advanced_viewer.setReadOnly(True)
        advanced_viewer.setPlainText(getattr(content, "advanced_content", "（无高级内容）"))
        tabs.addTab(easy_viewer, "易懂说明")
        tabs.addTab(advanced_viewer, "高级内容")
        layout.addWidget(tabs, 1)

        hint = QLabel("请确认易懂说明；需要核对机器实际内容时可打开“高级内容”。")
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        no_btn = QPushButton("返回修改")
        yes_btn = QPushButton("确认并启用")
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

        title = QLabel("模型配置")
        title.setObjectName("pageTitle")
        desc = QLabel("已保存 URL/Key 时会自动拉取模型；也可点「拉取模型」刷新。Fast / Thinking 从列表中选择，也可手输。")
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
        self._test_btn.setObjectName("secondaryButton")
        self._test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_btn.setToolTip("用当前 Fast 模型发一条最短请求验证连通性")
        self._test_btn.clicked.connect(self._on_test)

        self._test_status = QLabel("")
        self._test_status.setObjectName("hintLabel")
        self._test_status.setWordWrap(True)
        _prevent_horizontal_growth(self._test_status)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self._fetch_models_btn)
        action_row.addWidget(self._test_btn)
        action_row.addWidget(self._test_status, 1)

        form = QFormLayout()
        form.setSpacing(14)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow("Base URL", self._base_url)
        form.addRow("API Key", self._api_key)
        form.addRow("Fast Model", self._model)
        form.addRow("Thinking Model", self._thinking_model)

        self._base_url.textChanged.connect(self._on_credentials_changed)
        self._api_key.textChanged.connect(self._on_credentials_changed)
        self._model.currentTextChanged.connect(self._emit_change)
        self._thinking_model.currentTextChanged.connect(self._emit_change)
        self._test_done.connect(self._show_test_result)
        self._models_done.connect(self._show_models_result)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(form)
        layout.addLayout(action_row)
        layout.addStretch(1)

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
        self._refresh_action_enabled()
        self._emit_change()

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
        self._refresh_action_enabled()
        self._set_status("正在拉取模型列表...", "busy")

        def _run() -> None:
            try:
                client = OpenAICompatibleClient(
                    ClientConfig(base_url=base, api_key=key, model="unused")
                )
                models = client.list_models()
                if not models:
                    self._models_done.emit([], "接口返回空列表，请检查 Base URL 是否支持 /models")
                else:
                    self._models_done.emit(models, f"已拉取 {len(models)} 个模型，请在下拉框中选择")
            except TranslationError as exc:
                self._models_done.emit(None, str(exc))
            except Exception as exc:
                self._models_done.emit(None, str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _show_models_result(self, models: object, message: str) -> None:
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
        self._refresh_action_enabled()
        self._set_status(f"测试中（{model}）...", "busy")
        threading.Thread(
            target=self._run_connection_test,
            args=(base, key, model),
            daemon=True,
        ).start()

    def _run_connection_test(self, base: str, key: str, model: str) -> None:
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
            msg = str(exc)[:120]
        self._test_done.emit(ok, msg)

    def _show_test_result(self, ok: bool, msg: str) -> None:
        self._testing = False
        self._refresh_action_enabled()
        self._set_status(msg, "ok" if ok else "error")


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
        self._preview_btn = QPushButton("生成预览")
        self._preview_btn.setObjectName("secondaryButton")
        self._preview_btn.clicked.connect(self._on_preview)
        self._save_btn = QPushButton("保存并启用")
        self._save_btn.setObjectName("primaryButton")
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._save_btn)
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
        user_preview = self._build_user_preview(optimized, policy=compiled.policy)
        if not self._confirm_compiled_prompt(user_preview):
            self._save_status.setText("已取消")
            self._save_status.setStyleSheet("color: #94a3b8; font-size: 12px;")
            return

        try:
            saved_path = self._prompt_storage.save_compiled_prompt(
                compiled.content,
                settings.prompt.compiled_prompt_path,
                policy=compiled.policy,
                reference_package=compiled.reference_package,
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

    def _build_user_preview(self, optimized: str = "", policy: dict | None = None) -> str:
        """Build a two-layer preview: constraints + knowledge (no system template)."""

        parts: list[str] = []
        c = optimized.strip() if optimized else self._constraints_text.strip()
        parts.append(f"约束层：\n{c if c else '（未填写）'}")

        refs = self._knowledge_paths
        if refs:
            parts.append(f"知识引用层（{len(refs)} 个文档）：\n" + "\n".join(f"  • {r}" for r in refs))
        else:
            parts.append("知识引用层：\n（未添加）")
        if policy:
            parts.append(
                "本地声明式约束（仅执行这些白名单规则）：\n"
                + json.dumps(policy, ensure_ascii=False, indent=2)
            )
        return "\n\n".join(parts)

    def _confirm_compiled_prompt_with_dialog(self, content: str) -> bool:
        dlg = QDialog(self)
        dlg.setWindowTitle("Compiled Prompt Preview")
        dlg.setFixedSize(600, 420)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 16)

        viewer = QPlainTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText(content)
        layout.addWidget(viewer)

        hint = QLabel("Confirm and enable this compiled prompt?")
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        yes_btn = QPushButton("Yes")
        no_btn = QPushButton("No")
        no_btn.setDefault(True)
        btn_row.addWidget(yes_btn)
        btn_row.addWidget(no_btn)
        layout.addLayout(btn_row)

        yes_btn.clicked.connect(lambda: dlg.done(1))
        no_btn.clicked.connect(lambda: dlg.done(0))
        return dlg.exec() == 1


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
            "AI 优化译文", "AI 问题判断", "AI 关键词候选", "AI 记忆规则候选", "AI 总体建议",
        )
        self._review_states = [self.UNREVIEWED] * len(self._section_names)
        self._review_choices: list[dict[str, object] | None] = [None] * len(self._section_names)
        self._nav_buttons = [NavButton("") for _ in self._section_names]
        self._stack = QStackedWidget()
        self._user_editors: list[QPlainTextEdit] = []
        self._ai_keyword_editor = KeywordTagEditor()
        self._ai_keyword_editor.set_keywords(suggestion.trigger_options or ([suggestion.trigger] if suggestion.trigger else []))
        self._user_keyword_editor = KeywordTagEditor()

        pages = (
            self._text_page(suggestion.improved_translation),
            self._text_page(suggestion.problem_summary),
            self._keyword_page(),
            self._text_page(suggestion.rule),
            self._text_page(
                "AI 建议保存长期记忆。" if suggestion.memory_recommended
                else "AI 不建议保存长期记忆。"
            ),
        )
        for page in pages:
            self._stack.addWidget(page)
        for index, button in enumerate(self._nav_buttons):
            button.clicked.connect(lambda checked=False, i=index: self._select_page(i))
        self._select_page(0)

        sidebar = QWidget()
        sidebar.setFixedWidth(170)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(8, 8, 8, 8)
        for button in self._nav_buttons:
            side_layout.addWidget(button)
        side_layout.addStretch(1)

        self._error_label = QLabel("")
        self._error_label.setObjectName("hintLabel")
        self._error_label.setWordWrap(True)
        self._use_ai_button = QPushButton("采用 AI 当前项")
        self._use_ai_button.setObjectName("primaryButton")
        self._use_ai_button.clicked.connect(self._use_ai)
        self._use_user_button = QPushButton("保存我的修改")
        self._use_user_button.setObjectName("primaryButton")
        self._use_user_button.clicked.connect(self._use_user)
        self._skip_button = QPushButton("跳过此项")
        self._skip_button.setObjectName("secondaryButton")
        self._skip_button.clicked.connect(self._skip_current)
        self._finish_button = QPushButton("完成审查并返回")
        self._finish_button.setObjectName("primaryButton")
        self._finish_button.setEnabled(False)
        self._finish_button.clicked.connect(self._finish_review)
        buttons = QHBoxLayout()
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

    def _text_page(self, ai_text: str) -> QWidget:
        page = QWidget()
        ai = QPlainTextEdit(ai_text)
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
        elif index == 3:
            value = self.suggestion.rule
            if not str(value).strip():
                self._record_current(self.SKIPPED, None)
                return
        else:
            value = (
                "AI 建议保存长期记忆。" if self.suggestion.memory_recommended
                else "AI 不建议保存长期记忆。"
            )
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
        for name, state, button in zip(self._section_names, self._review_states, self._nav_buttons):
            button.setText(f"{name} · {state}")

    def _user_text(self, section_index: int) -> str:
        editor_index = {0: 0, 1: 1, 3: 2, 4: 3}.get(section_index)
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

    def user_rule(self) -> str:
        return self._user_text(3)

    def user_overall_note(self) -> str:
        return self._user_text(4)


class FeedbackPage(QWidget):
    """Review bad translations and turn accepted fixes into local memory."""

    ai_work_started = Signal()
    ai_work_finished = Signal()
    _ai_work_done = Signal(dict)

    def sizeHint(self) -> QSize:
        """Keep the page compatible with the main window's 680px minimum width."""

        hint = super().sizeHint()
        return QSize(min(hint.width(), 440), hint.height())

    def __init__(
        self,
        feedback_store: FeedbackStore,
        settings: AppSettings,
        optimizer: FeedbackOptimizer | None = None,
        translation_applier: Callable[[int, str, str], bool] | None = None,
        dialog_factory: Callable[[FeedbackOptimization, QWidget | None], FeedbackOptimizationDialog] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("contentPage")
        self._feedback_store = feedback_store
        self._settings = settings
        self._optimizer = optimizer or FeedbackOptimizer(feedback_store=feedback_store)
        self._translation_applier = translation_applier
        self._dialog_factory = dialog_factory or FeedbackOptimizationDialog
        self._records: list[FeedbackRecord] = []
        self._ai_optimizing = False
        self._ai_job_id = 0
        self._ai_work_done.connect(self._on_ai_work_done)

        title = QLabel("优化翻译")
        title.setObjectName("pageTitle")
        desc = QLabel(
            "先核对原文和当前译文；可以只采用本次结果，长期记忆需要单独展开并确认。"
        )
        desc.setObjectName("pageDesc")
        desc.setWordWrap(True)

        self._feedback_combo = QComboBox()
        _prevent_horizontal_growth(self._feedback_combo)
        self._feedback_combo.currentIndexChanged.connect(self._on_feedback_selected)

        self._refresh_button = QPushButton("刷新")
        self._refresh_button.setObjectName("secondaryButton")
        self._refresh_button.setMinimumSize(92, 36)
        self._refresh_button.clicked.connect(self.refresh)

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("待处理"))
        picker_row.addWidget(self._feedback_combo, 1)
        picker_row.addWidget(self._refresh_button)

        self._source_text = QPlainTextEdit()
        self._source_text.setReadOnly(True)
        self._source_text.setMinimumHeight(68)
        self._source_text.setMaximumHeight(104)

        self._current_translation_text = QPlainTextEdit()
        self._current_translation_text.setReadOnly(True)
        self._current_translation_text.setMinimumHeight(68)
        self._current_translation_text.setMaximumHeight(104)

        self._note_text = QPlainTextEdit()
        self._note_text.setPlaceholderText("可选：写下哪里不满意，例如术语错、语气错、漏译、把意思翻反了。")
        self._note_text.setMinimumHeight(56)
        self._note_text.setMaximumHeight(82)

        self._corrected_translation_text = QPlainTextEdit()
        self._corrected_translation_text.setPlaceholderText(
            "可以直接输入你自己的译文；也可以让 AI 生成建议后再修改。"
        )
        self._corrected_translation_text.setMinimumHeight(72)
        self._corrected_translation_text.setMaximumHeight(112)

        self._keyword_editor = KeywordTagEditor()

        self._rule_text = QPlainTextEdit()
        self._rule_text.setPlaceholderText("确认后写入本地记忆的规则，例如：出现“高考”时应译为日本语境下的大学入学考试，不要误作高校考试。")
        self._rule_text.setMinimumHeight(76)
        self._rule_text.setMaximumHeight(104)

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
                QSizePolicy.Policy.MinimumExpanding,
            )

        comparison_widget = QWidget()
        comparison_layout = QGridLayout(comparison_widget)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setHorizontalSpacing(12)
        comparison_layout.setVerticalSpacing(6)
        source_label = QLabel("OCR 原文")
        current_label = QLabel("当前译文")
        source_label.setObjectName("feedbackSectionLabel")
        current_label.setObjectName("feedbackSectionLabel")
        comparison_layout.addWidget(source_label, 0, 0)
        comparison_layout.addWidget(current_label, 0, 1)
        comparison_layout.addWidget(self._source_text, 1, 0)
        comparison_layout.addWidget(self._current_translation_text, 1, 1)
        comparison_layout.setColumnStretch(0, 1)
        comparison_layout.setColumnStretch(1, 1)

        self._memory_panel = QWidget()
        self._memory_panel.setObjectName("feedbackMemoryPanel")
        memory_form = QFormLayout(self._memory_panel)
        memory_form.setContentsMargins(16, 16, 16, 12)
        memory_form.setSpacing(10)
        memory_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        memory_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        memory_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        memory_form.addRow("关键词", self._keyword_editor)
        memory_form.addRow("记忆规则", self._rule_text)
        form = QFormLayout()
        form.setContentsMargins(16, 16, 16, 12)
        form.setSpacing(9)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow(comparison_widget)
        form.addRow("你的备注（问题说明）", self._note_text)
        form.addRow("你认可的译文（可直接填写）", self._corrected_translation_text)

        self._save_note_button = QPushButton("保存备注")
        self._ai_optimize_button = QPushButton("让 AI 优化")
        self._accept_translation_button = QPushButton("仅采用译文")
        self._accept_memory_button = QPushButton("保存为长期记忆")
        self._dismiss_button = QPushButton("忽略")
        self._save_note_button.setObjectName("secondaryButton")
        self._accept_translation_button.setObjectName("primaryButton")
        self._accept_memory_button.setObjectName("primaryButton")
        self._ai_optimize_button.setObjectName("primaryButton")
        self._dismiss_button.setObjectName("secondaryButton")
        compact_widths = {
            self._save_note_button: 108,
            self._ai_optimize_button: 100,
            self._accept_translation_button: 100,
            self._accept_memory_button: 132,
            self._dismiss_button: 64,
        }
        for button, width in compact_widths.items():
            button.setFixedHeight(34)
            button.setMaximumWidth(width)
            button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)


        self._save_note_button.clicked.connect(self._on_save_note)
        self._ai_optimize_button.clicked.connect(self._on_ai_optimize)
        self._accept_translation_button.clicked.connect(self._on_accept_translation)
        self._accept_memory_button.clicked.connect(self._on_accept_with_memory)
        self._dismiss_button.clicked.connect(self._on_dismiss)
        memory_form.addRow("", self._accept_memory_button)

        self._action_layout = QHBoxLayout()
        self._action_layout.setContentsMargins(0, 0, 0, 0)
        self._action_layout.setSpacing(8)
        self._action_layout.addWidget(self._save_note_button)
        self._action_layout.addWidget(self._ai_optimize_button)
        self._action_layout.addWidget(self._accept_translation_button)
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
        self._feedback_tabs.addTab(self._memory_scroll, "长期记忆（可选）")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(picker_row)
        layout.addWidget(self._feedback_tabs, 1)
        layout.addLayout(self._action_layout)
        layout.addWidget(self._status_label)

        self.refresh()

    def refresh(self, preserve_current: bool = False) -> None:
        """Reload pending feedback from local storage."""

        current_id = (
            self._feedback_combo.currentData()
            if preserve_current or self._ai_optimizing
            else None
        )
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
            self._feedback_tabs.setCurrentIndex(0)
            self._feedback_tabs.setTabText(1, "长期记忆（可选）")
            self._set_actions_enabled(False)
            self._status_label.setText("暂无待优化的翻译。")
            return

        self._set_actions_enabled(True)
        self._source_text.setPlainText(record.ocr_text)
        self._current_translation_text.setPlainText(record.translation_text)
        self._note_text.setPlainText(record.note)
        self._corrected_translation_text.setPlainText(record.corrected_translation)
        self._set_keyword_options([], "")
        self._rule_text.setPlainText("")
        self._feedback_tabs.setCurrentIndex(0)
        self._feedback_tabs.setTabText(1, "长期记忆（可选）")
        self._status_label.setText(f"已载入 {record.summary()}")

    def _set_actions_enabled(self, enabled: bool) -> None:
        enabled = enabled and not self._ai_optimizing
        for widget in (
            self._save_note_button,
            self._ai_optimize_button,
            self._accept_translation_button,
            self._accept_memory_button,
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

    def set_translation_applier(
        self,
        callback: Callable[[int, str, str], bool] | None,
    ) -> None:
        """Connect the page to the existing selection workflow controller."""

        self._translation_applier = callback

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
        if record is None or self._ai_optimizing:
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
            self._ai_work_done.emit(payload)

        threading.Thread(target=_run, daemon=True).start()

    def _on_ai_work_done(self, payload: dict) -> None:
        if payload.get("job_id") != self._ai_job_id:
            return

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
        if dialog.decision != FeedbackOptimizationDialog.COMPLETED:
            self._status_label.setText("未采用 AI 建议，主页面内容保持不变。")
            return

        translation_choice = dialog.review_choice(0)
        problem_choice = dialog.review_choice(1)
        keyword_choice = dialog.review_choice(2)
        rule_choice = dialog.review_choice(3)
        overall_choice = dialog.review_choice(4)
        if translation_choice and translation_choice["state"] != FeedbackOptimizationDialog.SKIPPED:
            translation_value = str(translation_choice["value"] or "").strip()
            if translation_value:
                self._corrected_translation_text.setPlainText(translation_value)
        if keyword_choice and keyword_choice["state"] != FeedbackOptimizationDialog.SKIPPED:
            keyword_value = list(keyword_choice["value"] or [])
            if keyword_value:
                self._keyword_editor.set_keywords(keyword_value)
        if rule_choice and rule_choice["state"] != FeedbackOptimizationDialog.SKIPPED:
            rule_value = str(rule_choice["value"] or "").strip()
            if rule_value:
                self._rule_text.setPlainText(rule_value)

        additions: list[str] = []
        for choice in (problem_choice, overall_choice):
            if choice and choice["state"] != FeedbackOptimizationDialog.SKIPPED:
                text = str(choice["value"] or "").strip()
                if text:
                    additions.append(text)
        old_note = self._note_text.toPlainText().strip()
        merged = [old_note] if old_note else []
        merged.extend(text for text in additions if text not in merged)
        self._note_text.setPlainText("\n".join(merged))
        self._status_label.setText("审查结果已复制为草稿；请在主页面再次确认保存。")

    def _apply_translation_to_overlay(
        self,
        record: FeedbackRecord,
        translation: str,
    ) -> bool | None:
        if self._translation_applier is None:
            return None
        return self._translation_applier(record.group_id, record.ocr_text, translation)

    def _on_accept_translation(self) -> None:
        record = self._sync_record_edits()
        if record is None:
            return
        translation = self._corrected_translation_text.toPlainText().strip()
        if not translation:
            self._status_label.setText("请先填写非空的优化后/认可译文。")
            return
        try:
            accepted = self._feedback_store.accept_translation(record.id, translation)
        except (KeyError, ValueError) as exc:
            self._status_label.setText(f"采用失败：{exc}")
            return
        overlay_updated = self._apply_translation_to_overlay(accepted, translation)
        self.refresh()
        if overlay_updated is False:
            self._status_label.setText("译文已保存；原选区内容已变化，未覆盖当前翻译框。")
        else:
            self._status_label.setText("已仅采用本次译文，未创建长期记忆。")

    def _on_accept_with_memory(self) -> None:
        record = self._sync_record_edits()
        if record is None:
            return
        trigger = self._keyword_text()
        preferred_translation = self._corrected_translation_text.toPlainText().strip()
        rule = self._rule_text.toPlainText().strip()
        if not trigger or not rule:
            self._feedback_tabs.setCurrentIndex(1)
            self._status_label.setText("保存长期记忆需要确认关键词和规则。")
            return
        try:
            memory = self._feedback_store.approve_feedback(
                record.id,
                trigger=trigger,
                rule=rule,
                preferred_translation=preferred_translation,
            )
        except (KeyError, ValueError) as exc:
            self._status_label.setText(f"确认失败：{exc}")
            return
        overlay_updated = (
            self._apply_translation_to_overlay(record, preferred_translation)
            if preferred_translation else None
        )
        self.refresh()
        if overlay_updated is False:
            self._status_label.setText(
                f"已写入本地记忆：{memory.trigger}；原选区内容已变化，未覆盖当前翻译框。"
            )
        else:
            self._status_label.setText(f"已采用并写入本地记忆：{memory.trigger}")

    def _on_confirm(self) -> None:
        """Backward-compatible internal alias for the explicit memory path."""

        self._on_accept_with_memory()

    def _on_dismiss(self) -> None:
        record = self._current_record()
        if record is None:
            return
        self._feedback_store.update_feedback(record.id, status="dismissed")
        self._status_label.setText("已忽略该反馈。")
        self.refresh()


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
        form.setSpacing(14)
        form.addRow("新建选择框", self._create_input)
        form.addRow("切换编辑模式", self._edit_input)

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
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addLayout(btn_row)
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
            self._model_page.ensure_models_loaded()
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
        previous = (
            s.ai.base_url,
            s.ai.api_key,
            s.ai.fast_model_name,
            s.ai.thinking_model_name,
        )
        s.ai.base_url = base_url
        s.ai.api_key = api_key
        s.ai.fast_model = fast_model
        s.ai.thinking_model = thinking_model
        s.ai.model = fast_model or thinking_model
        try:
            s.save()
        except Exception:
            pass  # save silently; test button handles feedback
        current = (base_url, api_key, fast_model, thinking_model)
        if current != previous:
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
        self._feedback_page.refresh(preserve_current=True)

    def set_feedback_translation_applier(
        self,
        callback: Callable[[int, str, str], bool] | None,
    ) -> None:
        """Connect accepted feedback to the existing overlay workflow."""

        self._feedback_page.set_translation_applier(callback)

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
            QScrollArea#feedbackDetailScroll,
            QScrollArea#feedbackMemoryScroll,
            QWidget#feedbackDetailWidget,
            QWidget#feedbackMemoryPanel {
                background: #0f172a;
                border: none;
            }
            QTabWidget#feedbackTabs::pane {
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 10px;
                top: -1px;
                padding: 4px;
            }
            QTabWidget#feedbackTabs QTabBar::tab {
                background: transparent;
                color: #94a3b8;
                border: none;
                border-bottom: 2px solid transparent;
                padding: 8px 14px;
                margin-right: 4px;
                font-size: 13px;
                font-weight: 500;
            }
            QTabWidget#feedbackTabs QTabBar::tab:selected {
                color: #60a5fa;
                font-weight: 600;
                border-bottom: 2px solid #60a5fa;
            }
            QTabWidget#feedbackTabs QTabBar::tab:hover {
                color: #e2e8f0;
            }
            QLabel#feedbackSectionLabel {
                color: #94a3b8;
                font-weight: 600;
                font-size: 13px;
            }
            QLabel#feedbackMemoryRecommendation {
                color: #94a3b8;
                font-size: 13px;
                background: transparent;
                border: none;
                padding: 0;
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
                background: #1e293b;
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
                background: #1e293b;
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
