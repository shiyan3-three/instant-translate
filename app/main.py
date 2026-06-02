"""Application entry point for instant-translate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PySide6.QtWidgets import QApplication

from app.app_context import ApplicationContext
from app.gui.main_window import MainWindow
from app.gui.settings_window import SettingsWindow
from app.gui.tray_icon import TrayIconController
from app.hotkeys import GlobalHotkeyService
from app.overlay.edit_mode_controller import EditModeController
from app.overlay.selection_manager import SelectionWorkflowController
from app.state.runtime_store import RuntimeStore


@dataclass
class DesktopShell:
    """Bundle the first desktop shell components for the running app."""

    app: QApplication
    context: ApplicationContext
    main_window: MainWindow
    settings_window: SettingsWindow
    tray_icon: TrayIconController
    hotkeys: GlobalHotkeyService
    runtime_store: RuntimeStore
    selection_workflow: SelectionWorkflowController


def build_application_context() -> ApplicationContext:
    """Create the top-level application context."""

    return ApplicationContext()


def build_desktop_shell(argv: Sequence[str] | None = None) -> DesktopShell:
    """Construct the first interactive desktop shell without entering the event loop."""

    app = QApplication.instance() or QApplication(list(argv or []))
    app.setApplicationName("Instant Translate")
    app.setQuitOnLastWindowClosed(False)

    context = build_application_context()
    runtime_store = RuntimeStore()
    settings_window = SettingsWindow(context.settings)
    main_window = MainWindow(context)
    tray_icon = TrayIconController(app, main_window, settings_window, main_window)
    hotkeys = GlobalHotkeyService(app, context.hotkeys, main_window)
    selection_workflow = SelectionWorkflowController(
        app=app,
        context=context,
        runtime_store=runtime_store,
        edit_mode_controller=EditModeController(),
        on_state_changed=main_window.refresh_runtime_state,
    )

    main_window.settings_requested.connect(settings_window.show_window)
    hotkeys.create_selection_triggered.connect(selection_workflow.request_new_selection)
    hotkeys.toggle_edit_mode_triggered.connect(selection_workflow.toggle_edit_mode)

    return DesktopShell(
        app=app,
        context=context,
        main_window=main_window,
        settings_window=settings_window,
        tray_icon=tray_icon,
        hotkeys=hotkeys,
        runtime_store=runtime_store,
        selection_workflow=selection_workflow,
    )


def main() -> int:
    """Bootstrap the desktop application and enter the event loop."""

    shell = build_desktop_shell()
    shell.main_window.show()
    shell.tray_icon.show()
    try:
        shell.hotkeys.start()
    except RuntimeError as exc:
        shell.context.status_message = f"快捷键服务启动失败：{exc}"
        shell.main_window.refresh_runtime_state()

    try:
        return shell.app.exec()
    finally:
        shell.hotkeys.stop()
        shell.selection_workflow.close()


if __name__ == "__main__":
    raise SystemExit(main())
