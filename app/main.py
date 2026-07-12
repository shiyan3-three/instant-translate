"""Application entry point for instant-translate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PySide6.QtWidgets import QApplication

from app.app_context import ApplicationContext
from app.feedback.store import FeedbackStore
from app.gui.main_window import MainWindow
from app.gui.tray_icon import TrayIconController
from app.hotkeys import GlobalHotkeyService, HotkeyMap
from app.overlay.edit_mode_controller import EditModeController
from app.overlay.selection_manager import SelectionWorkflowController
from app.settings import AppSettings
from app.state.runtime_store import RuntimeStore


@dataclass
class DesktopShell:
    """Bundle the first desktop shell components for the running app."""

    app: QApplication
    context: ApplicationContext
    main_window: MainWindow
    tray_icon: TrayIconController
    hotkeys: GlobalHotkeyService
    runtime_store: RuntimeStore
    feedback_store: FeedbackStore
    selection_workflow: SelectionWorkflowController


def build_application_context() -> ApplicationContext:
    """Create the top-level application context with persisted settings."""

    settings = AppSettings.load()
    return ApplicationContext(
        settings=settings,
        hotkeys=HotkeyMap(
            create_selection=settings.hotkey_create_selection,
            toggle_edit_mode=settings.hotkey_toggle_edit_mode,
        ),
        default_source_language=settings.default_source_language,
        default_target_language=settings.default_target_language,
    )


def build_desktop_shell(argv: Sequence[str] | None = None) -> DesktopShell:
    """Construct the first interactive desktop shell without entering the event loop."""

    app = QApplication.instance() or QApplication(list(argv or []))
    app.setApplicationName("Instant Translate")
    app.setQuitOnLastWindowClosed(False)

    context = build_application_context()
    runtime_store = RuntimeStore()
    feedback_store = FeedbackStore()
    main_window = MainWindow(
        context,
        runtime_store=runtime_store,
        feedback_store=feedback_store,
    )
    tray_icon = TrayIconController(app, main_window, parent=main_window)
    hotkeys = GlobalHotkeyService(app, context.hotkeys, main_window)
    selection_workflow = SelectionWorkflowController(
        app=app,
        context=context,
        runtime_store=runtime_store,
        feedback_store=feedback_store,
        edit_mode_controller=EditModeController(),
        on_state_changed=main_window.refresh_runtime_state,
    )
    main_window.set_feedback_translation_applier(
        selection_workflow.apply_accepted_translation
    )

    def _apply_default_language(src: str, tgt: str) -> None:
        context.default_source_language = src
        context.default_target_language = tgt
        for gid in runtime_store.active_group_ids():
            config = runtime_store.configs.get(gid)
            if config is not None:
                config.source_language = src
                config.target_language = tgt
            region = runtime_store.regions.get(gid)
            if region is not None:
                selection_workflow._upsert_selection_box(gid, region)
        selection_workflow.reload_translation_agents()

    main_window.default_language_changed.connect(_apply_default_language)
    main_window.compiled_prompt_saved.connect(selection_workflow.reload_translation_agents)
    main_window.model_config_changed.connect(selection_workflow.reload_translation_agents)
    hotkeys.create_selection_triggered.connect(selection_workflow.request_new_selection)
    hotkeys.toggle_edit_mode_triggered.connect(selection_workflow.toggle_edit_mode)

    def _on_hotkeys_save(create_key: str, edit_key: str) -> None:
        new_map = HotkeyMap(create_selection=create_key, toggle_edit_mode=edit_key)
        success, msg = hotkeys.re_register(new_map)
        if success:
            context.settings.hotkey_create_selection = create_key
            context.settings.hotkey_toggle_edit_mode = edit_key
            context.settings.save()
            context.hotkeys = new_map
            main_window.show_hotkey_result(True, "已保存并生效")
        else:
            main_window.show_hotkey_result(False, msg)

    main_window.hotkeys_save_requested.connect(_on_hotkeys_save)

    return DesktopShell(
        app=app,
        context=context,
        main_window=main_window,
        tray_icon=tray_icon,
        hotkeys=hotkeys,
        runtime_store=runtime_store,
        feedback_store=feedback_store,
        selection_workflow=selection_workflow,
    )


def main() -> int:
    """Bootstrap the desktop application and enter the event loop."""

    from app.logger import get_logger

    log = get_logger()
    shell = build_desktop_shell()
    shell.main_window.show()
    shell.tray_icon.show()
    try:
        shell.hotkeys.start()
        log.info("快捷键服务已启动")
    except RuntimeError as exc:
        log.error("快捷键服务启动失败：%s", exc)
        shell.context.status_message = f"快捷键服务启动失败：{exc}"
        shell.main_window.refresh_runtime_state()

    log.info("应用就绪，进入事件循环")
    try:
        return shell.app.exec()
    finally:
        log.info("应用退出，正在清理资源")
        shell.hotkeys.stop()
        shell.selection_workflow.close()


if __name__ == "__main__":
    raise SystemExit(main())
