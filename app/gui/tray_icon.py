"""System tray integration for the desktop shell."""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon, QWidget

from app.gui.main_window import MainWindow
from app.gui.settings_window import SettingsWindow


class TrayIconController:
    """Manage tray actions such as open settings and exit."""

    def __init__(
        self,
        app: QApplication,
        main_window: MainWindow,
        settings_window: SettingsWindow,
        parent: QWidget | None = None,
    ) -> None:
        self._app = app
        self._main_window = main_window
        self._settings_window = settings_window

        icon = main_window.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.tray_icon = QSystemTrayIcon(icon, parent or main_window)
        self.tray_icon.setToolTip("Instant Translate")

        self.menu = QMenu(parent or main_window)
        self.toggle_main_window_action = QAction("显示主窗口", self.menu)
        self.open_settings_action = QAction("打开设置", self.menu)
        self.exit_action = QAction("退出", self.menu)

        self.menu.addAction(self.toggle_main_window_action)
        self.menu.addAction(self.open_settings_action)
        self.menu.addSeparator()
        self.menu.addAction(self.exit_action)
        self.tray_icon.setContextMenu(self.menu)

        self.toggle_main_window_action.triggered.connect(self.toggle_main_window)
        self.open_settings_action.triggered.connect(self._settings_window.show_window)
        self.exit_action.triggered.connect(self._app.quit)
        self.tray_icon.activated.connect(self._handle_activation)

    def show(self) -> None:
        """Show the tray icon and refresh its action labels."""

        self.refresh_main_window_action_text()
        self.tray_icon.show()

    def toggle_main_window(self) -> None:
        """Toggle the main window visibility."""

        if self._main_window.isVisible():
            self._main_window.hide()
        else:
            self._main_window.show_window()
        self.refresh_main_window_action_text()

    def refresh_main_window_action_text(self) -> None:
        """Update the menu label to match the current visibility state."""

        if self._main_window.isVisible():
            self.toggle_main_window_action.setText("隐藏主窗口")
        else:
            self.toggle_main_window_action.setText("显示主窗口")

    def _handle_activation(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._main_window.show_window()
            self.refresh_main_window_action_text()
