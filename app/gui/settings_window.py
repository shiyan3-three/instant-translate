"""Settings window for AI and prompt configuration."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.prompt.compiler import PromptCompiler
from app.prompt.models import PromptConstraints, PromptKnowledgeReference
from app.prompt.optimizer import PromptOptimizationError, PromptOptimizer
from app.prompt.storage import PromptStorage
from app.settings import AppSettings


class SettingsWindow(QDialog):
    """Host AI configuration and prompt editing controls."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._compiler = PromptCompiler()
        self._optimizer = PromptOptimizer(self._compiler)
        self._prompt_storage = PromptStorage()
        self.setWindowTitle("设置")
        self.resize(760, 560)

        self.base_url_input = QLineEdit()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.model_input = QLineEdit()
        self.thinking_model_input = QLineEdit()
        self.constraints_input = QPlainTextEdit()
        self.knowledge_reference_list = QListWidget()

        self.add_knowledge_button = QPushButton("\u6dfb\u52a0")
        self.remove_knowledge_button = QPushButton("\u5220\u9664")
        self.preview_knowledge_button = QPushButton("预览解析")
        self.export_ai_candidates_button = QPushButton("导出AI候选")
        knowledge_button_row = QHBoxLayout()
        knowledge_button_row.addWidget(self.add_knowledge_button)
        knowledge_button_row.addWidget(self.remove_knowledge_button)
        knowledge_button_row.addWidget(self.preview_knowledge_button)
        knowledge_button_row.addWidget(self.export_ai_candidates_button)
        knowledge_button_row.addStretch(1)

        self.compiled_prompt_path_input = QLineEdit()
        self.compiled_prompt_path_input.setReadOnly(True)
        self.preview_compiled_prompt_button = QPushButton("\u9884\u89c8 Compiled Prompt")
        self.save_settings_button = QPushButton("\u4fdd\u5b58\u8bbe\u7f6e")

        ai_group = QGroupBox("AI \u63a5\u53e3\u914d\u7f6e")
        ai_layout = QFormLayout(ai_group)
        ai_layout.addRow("Base URL", self.base_url_input)
        ai_layout.addRow("API Key", self.api_key_input)
        ai_layout.addRow("Fast Model", self.model_input)
        ai_layout.addRow("Thinking Model", self.thinking_model_input)

        prompt_group = QGroupBox("Prompt \u914d\u7f6e")
        prompt_layout = QFormLayout(prompt_group)
        prompt_layout.addRow("\u7ea6\u675f\u5c42", self.constraints_input)
        prompt_layout.addRow("\u77e5\u8bc6\u5f15\u7528", self.knowledge_reference_list)
        prompt_layout.addRow("", knowledge_button_row)
        prompt_layout.addRow("Compiled Prompt \u6587\u4ef6", self.compiled_prompt_path_input)

        hint_label = QLabel(
            "\u63d0\u793a\uff1a\u56fa\u5b9a\u6a21\u677f\u5c42\u7531\u7cfb\u7edf\u7ef4\u62a4\uff1b\u7528\u6237\u4e3b\u8981\u7f16\u8f91\u7ea6\u675f\u5c42\u548c\u77e5\u8bc6\u5f15\u7528\u5c42\uff0c"
            "\u786e\u8ba4\u9884\u89c8\u540e\u518d\u751f\u6210\u6700\u7ec8\u7684 compiled prompt\u3002"
        )
        hint_label.setWordWrap(True)

        button_row = QHBoxLayout()
        button_row.addWidget(self.preview_compiled_prompt_button)
        button_row.addStretch(1)
        button_row.addWidget(self.save_settings_button)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(ai_group)
        root_layout.addWidget(prompt_group)
        root_layout.addWidget(hint_label)
        root_layout.addLayout(button_row)

        self._connect_signals()
        self.reload_from_settings()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def reload_from_settings(self) -> None:
        """Reload widget values from the current settings object."""

        self.base_url_input.setText(self._settings.ai.base_url)
        self.api_key_input.setText(self._settings.ai.api_key)
        self.model_input.setText(self._settings.ai.fast_model_name)
        self.thinking_model_input.setText(self._settings.ai.thinking_model_name)
        self.constraints_input.setPlainText(self._settings.prompt.constraints_text)
        self.knowledge_reference_list.clear()
        self.knowledge_reference_list.addItems(self._settings.prompt.knowledge_reference_paths)
        self.compiled_prompt_path_input.setText(self._settings.prompt.compiled_prompt_path)

    def show_window(self) -> None:
        """Bring the settings dialog to the foreground."""

        self.reload_from_settings()
        self.show()
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.save_settings_button.clicked.connect(self._on_save)
        self.preview_compiled_prompt_button.clicked.connect(self._on_preview)
        self.add_knowledge_button.clicked.connect(self._on_add_knowledge)
        self.remove_knowledge_button.clicked.connect(self._on_remove_knowledge)
        self.preview_knowledge_button.clicked.connect(self._on_preview_references)
        self.export_ai_candidates_button.clicked.connect(self._on_export_ai_reference_candidates)

    def _on_save(self) -> None:
        self._apply_form_to_settings()

        try:
            self._settings.save()
        except OSError as exc:
            QMessageBox.warning(self, "\u4fdd\u5b58\u5931\u8d25", f"\u65e0\u6cd5\u5199\u5165\u914d\u7f6e\u6587\u4ef6\uff1a{exc}")
            return

        QMessageBox.information(self, "\u5df2\u4fdd\u5b58", "\u8bbe\u7f6e\u5df2\u4fdd\u5b58\u3002")

    def _apply_form_to_settings(self) -> None:
        """Persist current widget values into the settings object."""

        self._settings.ai.base_url = self.base_url_input.text().strip()
        self._settings.ai.api_key = self.api_key_input.text().strip()
        self._settings.ai.fast_model = self.model_input.text().strip()
        self._settings.ai.thinking_model = self.thinking_model_input.text().strip()
        self._settings.ai.model = self._settings.ai.fast_model or self._settings.ai.thinking_model
        self._settings.prompt.constraints_text = self.constraints_input.toPlainText()

        paths: list[str] = []
        for i in range(self.knowledge_reference_list.count()):
            item = self.knowledge_reference_list.item(i)
            if item:
                paths.append(item.text())
        self._settings.prompt.knowledge_reference_paths = paths

    def _on_preview(self) -> None:
        self._apply_form_to_settings()
        constraints = PromptConstraints(text=self.constraints_input.toPlainText())
        references = self._collect_knowledge_references()

        try:
            optimized_rules = self._optimizer.optimize(
                self._settings,
                constraints,
                references,
            )
        except PromptOptimizationError as exc:
            QMessageBox.warning(
                self,
                "Prompt \u751f\u6210\u5931\u8d25",
                str(exc),
            )
            return

        compiled = self._compiler.compile_preview(
            constraints,
            references=references,
            optimized_user_layer=optimized_rules,
        )
        confirmed = QMessageBox.question(
            self,
            "Compiled Prompt \u9884\u89c8",
            f"{compiled.content}\n\nLocal Policy:\n"
            f"{json.dumps(compiled.policy or {}, ensure_ascii=False, indent=2)}\n\n"
            "\u786e\u8ba4\u751f\u6210\u5e76\u5199\u5165 compiled prompt \u6587\u4ef6\u5417\uff1f",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        try:
            saved_path = self._prompt_storage.save_compiled_prompt(
                compiled.content,
                self._settings.prompt.compiled_prompt_path,
                policy=compiled.policy,
                reference_package=compiled.reference_package,
            )
            stored_path = self._prompt_storage.stored_compiled_prompt_path(
                self._settings.prompt.compiled_prompt_path
            )
            self._settings.prompt.compiled_prompt_path = stored_path
            self.compiled_prompt_path_input.setText(stored_path)
            self._settings.save()
        except OSError as exc:
            QMessageBox.warning(self, "\u4fdd\u5b58\u5931\u8d25", f"\u65e0\u6cd5\u5199\u5165 compiled prompt\uff1a{exc}")
            return

        QMessageBox.information(
            self,
            "Compiled Prompt \u5df2\u751f\u6210",
            f"\u5df2\u5199\u5165\uff1a{saved_path}",
        )

    def _collect_knowledge_references(self) -> list[PromptKnowledgeReference]:
        """Collect enabled knowledge references from the list widget."""

        references: list[PromptKnowledgeReference] = []
        for i in range(self.knowledge_reference_list.count()):
            item = self.knowledge_reference_list.item(i)
            if item:
                references.append(PromptKnowledgeReference(path=item.text()))
        return references

    def _knowledge_reference_dir(self) -> Path:
        """Return the project-local folder used for reference-layer markdown."""

        return self._prompt_storage.ensure_reference_dir()

    def _on_add_knowledge(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "\u9009\u62e9\u77e5\u8bc6\u6587\u6863",
            str(self._knowledge_reference_dir()),
            "Markdown \u6587\u4ef6 (*.md);;\u6240\u6709\u6587\u4ef6 (*)",
        )
        if file_path:
            self._add_knowledge_path(file_path)

    def _on_remove_knowledge(self) -> None:
        for item in self.knowledge_reference_list.selectedItems():
            row = self.knowledge_reference_list.row(item)
            self.knowledge_reference_list.takeItem(row)

    def _add_knowledge_path(self, file_path: str) -> bool:
        path = str(file_path).strip()
        if not path:
            return False
        for i in range(self.knowledge_reference_list.count()):
            item = self.knowledge_reference_list.item(i)
            if item and item.text() == path:
                return False
        self.knowledge_reference_list.addItem(path)
        return True

    def _on_preview_references(self) -> None:
        QMessageBox.information(self, "知识引用解析预览", self._build_reference_preview())

    def _build_reference_preview(self) -> str:
        paths = [
            self.knowledge_reference_list.item(i).text()
            for i in range(self.knowledge_reference_list.count())
            if self.knowledge_reference_list.item(i)
        ]
        if not paths:
            return "尚未添加知识引用文档。"
        parts: list[str] = []
        total_entries = 0
        total_style = 0
        total_risk = 0
        for path in paths:
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
            self.compiled_prompt_path_input.text().strip() or "prompts/compiled-prompt.md"
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
            self._add_knowledge_path(str(candidate_path))
