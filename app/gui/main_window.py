"""Main window for the first desktop shell milestone."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from app.app_context import ApplicationContext


class MainWindow(QMainWindow):
    """Represent the first interactive desktop shell."""

    settings_requested = Signal()

    def __init__(self, context: ApplicationContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self.setWindowTitle("Instant Translate")
        self.resize(960, 640)

        self.open_settings_button = QPushButton("打开设置")
        self.open_settings_button.clicked.connect(self.settings_requested.emit)

        self.hotkey_summary_label = QLabel()
        self.hotkey_summary_label.setWordWrap(True)

        self.state_summary_label = QLabel()
        self.state_summary_label.setWordWrap(True)

        title_label = QLabel("Instant Translate")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title_label.setStyleSheet("font-size: 22px; font-weight: 600;")

        description_label = QLabel(
            "第一阶段桌面壳已接通主窗口、托盘和全局快捷键服务。"
            " 下一阶段会把框选层、OCR 和翻译管线逐步挂上来。"
        )
        description_label.setWordWrap(True)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(16)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(self.hotkey_summary_label)
        layout.addWidget(self.state_summary_label)
        layout.addWidget(self.open_settings_button)
        layout.addStretch(1)
        self.setCentralWidget(central_widget)

        self.refresh_runtime_state()

    def refresh_runtime_state(self) -> None:
        """Refresh user-facing runtime summaries."""

        hotkeys = self._context.hotkeys
        self.hotkey_summary_label.setText(
            "快捷键："
            f"新建选择框 {hotkeys.create_selection}，"
            f"切换编辑模式 {hotkeys.toggle_edit_mode}"
        )
        edit_mode = "开启" if self._context.edit_mode_enabled else "关闭"
        self.state_summary_label.setText(
            f"当前选择框数量：{self._context.active_group_count} / 3\n"
            f"编辑模式：{edit_mode}\n"
            f"状态：{self._context.status_message}"
        )
        self.statusBar().showMessage(self._context.status_message)

    def show_window(self) -> None:
        """Bring the main window to the foreground."""

        self.show()
        self.raise_()
        self.activateWindow()
