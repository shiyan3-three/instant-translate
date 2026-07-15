"""Application entry point for instant-translate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Sequence

from app.runtime import ensure_supported_python

if TYPE_CHECKING:
    from PySide6.QtCore import QLockFile
    from PySide6.QtWidgets import QApplication

    from app.app_context import ApplicationContext
    from app.feedback.store import FeedbackStore
    from app.gui.main_window import MainWindow
    from app.gui.tray_icon import TrayIconController
    from app.hotkeys import GlobalHotkeyService
    from app.overlay.selection_manager import SelectionWorkflowController
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

    from app.app_context import ApplicationContext
    from app.hotkeys import HotkeyMap
    from app.settings import AppSettings

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


def _acquire_instance_lock(path: str | Path | None = None) -> QLockFile | None:
    """Return a held process lock, or ``None`` when another instance owns it."""

    from PySide6.QtCore import QLockFile
    from app.settings import AppSettings

    lock_path = Path(path) if path is not None else AppSettings.config_dir() / "instant-translate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(lock_path))
    if not lock.tryLock(0):
        return None
    return lock


def build_desktop_shell(argv: Sequence[str] | None = None) -> DesktopShell:
    """Construct the first interactive desktop shell without entering the event loop."""

    ensure_supported_python()
    from PySide6.QtWidgets import QApplication
    from app.feedback.store import FeedbackStore
    from app.gui.main_window import MainWindow
    from app.gui.tray_icon import TrayIconController
    from app.hotkeys import GlobalHotkeyService
    from app.overlay.edit_mode_controller import EditModeController
    from app.overlay.selection_manager import SelectionWorkflowController
    from app.state.runtime_store import RuntimeStore

    app = QApplication.instance() or QApplication(list(argv or []))
    app.setApplicationName("Instant Translate")
    app.setQuitOnLastWindowClosed(False)

    context = build_application_context()
    runtime_store = RuntimeStore()
    # An unresolved rollback journal must block feedback reads/writes, not the
    # entire OCR translation application.  Consumers surface the unavailable
    # state while the store continues to retry safe recovery on every access.
    feedback_store = FeedbackStore(allow_unavailable=True)
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
        on_feedback_changed=main_window.refresh_feedback_records,
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
        _apply_hotkey_change(context, hotkeys, main_window, new_map)

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


def _run_ocr_smoke() -> int:
    """Exercise one real Paddle inference without starting Qt or the API client."""

    from app.logger import get_logger
    from app.ocr.engine import OcrEngine

    log = get_logger()
    engine = OcrEngine()
    engine.warm_up("中文")
    if engine._backend != "paddle" or "ch" not in engine._warmed_languages:
        log.error(
            "OCR smoke failed | backend=%s warmed=%s",
            engine._backend,
            sorted(engine._warmed_languages),
        )
        return 2
    log.info("OCR smoke passed | backend=paddle lang=ch")
    return 0


def _apply_hotkey_change(context, hotkeys, main_window, new_map) -> None:
    """Apply and persist hotkeys, restoring the runtime mapping on save failure."""

    from app.logger import get_logger

    old_map = context.hotkeys
    old_values = (
        context.settings.hotkey_create_selection,
        context.settings.hotkey_toggle_edit_mode,
    )
    success, message = hotkeys.re_register(new_map)
    if not success:
        main_window.show_hotkey_result(False, message)
        return

    context.settings.hotkey_create_selection = new_map.create_selection
    context.settings.hotkey_toggle_edit_mode = new_map.toggle_edit_mode
    try:
        context.settings.save()
    except Exception as exc:
        (
            context.settings.hotkey_create_selection,
            context.settings.hotkey_toggle_edit_mode,
        ) = old_values
        rollback_ok, rollback_message = hotkeys.re_register(old_map)
        context.hotkeys = old_map if rollback_ok else new_map
        detail = "快捷键保存失败，运行时已恢复原配置。"
        if not rollback_ok:
            detail = f"快捷键保存失败，且运行时回滚失败：{rollback_message}"
        get_logger().error("Hotkey settings save failed: %s", exc)
        main_window.show_hotkey_result(False, detail)
        return

    context.hotkeys = new_map
    main_window.show_hotkey_result(True, "已保存并生效")


def main() -> int:
    """Bootstrap the desktop application and enter the event loop."""

    ensure_supported_python()
    if "--smoke-ocr" in sys.argv[1:]:
        return _run_ocr_smoke()
    from app.logger import get_logger

    log = get_logger()
    log.info(
        "应用启动基线 | python=%s executable=%s source_root=%s",
        sys.version.split()[0],
        sys.executable,
        Path(__file__).resolve().parent.parent,
    )
    instance_lock = _acquire_instance_lock()
    if instance_lock is None:
        log.warning("应用已在运行，本次启动已退出")
        return 0
    shell: DesktopShell | None = None
    try:
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
        return shell.app.exec()
    finally:
        if shell is not None:
            log.info("应用退出，正在清理资源")
            shell.main_window.shutdown_background_tasks()
            shell.hotkeys.stop()
            shell.selection_workflow.close()
        instance_lock.unlock()


if __name__ == "__main__":
    raise SystemExit(main())
