"""Settings window for AI and prompt configuration."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.settings import AppSettings


class SettingsWindow(QDialog):
    """Host AI configuration and prompt editing controls."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("设置")
        self.resize(760, 560)

        self.base_url_input = QLineEdit()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.model_input = QLineEdit()
        self.constraints_input = QPlainTextEdit()
        self.knowledge_reference_list = QListWidget()
        self.compiled_prompt_path_input = QLineEdit()
        self.compiled_prompt_path_input.setReadOnly(True)
        self.preview_compiled_prompt_button = QPushButton("预览 Compiled Prompt")
        self.save_settings_button = QPushButton("保存设置")

        ai_group = QGroupBox("AI 接口配置")
        ai_layout = QFormLayout(ai_group)
        ai_layout.addRow("Base URL", self.base_url_input)
        ai_layout.addRow("API Key", self.api_key_input)
        ai_layout.addRow("Model", self.model_input)

        prompt_group = QGroupBox("Prompt 配置")
        prompt_layout = QFormLayout(prompt_group)
        prompt_layout.addRow("约束层", self.constraints_input)
        prompt_layout.addRow("知识引用", self.knowledge_reference_list)
        prompt_layout.addRow("Compiled Prompt 文件", self.compiled_prompt_path_input)

        hint_label = QLabel(
            "提示：固定模板层由系统维护；用户主要编辑约束层和知识引用层，"
            "确认预览后再生成最终的 compiled prompt。"
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

        self.reload_from_settings()

    def reload_from_settings(self) -> None:
        """Reload widget values from the current settings object."""

        self.base_url_input.setText(self._settings.ai.base_url)
        self.api_key_input.setText(self._settings.ai.api_key)
        self.model_input.setText(self._settings.ai.model)
        self.constraints_input.setPlainText(self._settings.prompt.constraints_text)
        self.knowledge_reference_list.clear()
        self.knowledge_reference_list.addItems(self._settings.prompt.knowledge_reference_paths)
        self.compiled_prompt_path_input.setText(self._settings.prompt.compiled_prompt_path)

    def show_window(self) -> None:
        """Bring the settings dialog to the foreground."""

        self.show()
        self.raise_()
        self.activateWindow()
