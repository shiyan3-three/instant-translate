"""Selection workflow controller for creating and displaying fixed screen regions."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from app.app_context import ApplicationContext
from app.capture.change_detector import RegionChangeDetector
from app.capture.screen_capture import ScreenCaptureService
from app.logger import get_logger, get_debug_logger
from app.ocr.engine import OcrEngine
from app.ocr.postprocess import is_duplicate_ocr_text, normalize_ocr_text
from app.overlay.edit_mode_controller import EditModeController
from app.overlay.selection_box import SelectionBoxModel, SelectionBoxWidget
from app.overlay.selection_overlay import RegionSelectionOverlay
from app.overlay.translation_window import TranslationWindowModel, TranslationWindowWidget
from app.state.group_state import ScreenRegion
from app.state.runtime_store import RuntimeStore
from app.translation.service import TranslationService

GROUP_COLORS = {
    1: "#2F80ED",
    2: "#27AE60",
    3: "#F2994A",
}


class SelectionWorkflowController(QObject):
    """Coordinate selection boxes with their linked translation windows."""

    _translation_ready = Signal(int, str)

    def __init__(
        self,
        app: QApplication,
        context: ApplicationContext,
        runtime_store: RuntimeStore,
        edit_mode_controller: EditModeController,
        on_state_changed,
        capture_service: ScreenCaptureService | None = None,
        change_detector: RegionChangeDetector | None = None,
    ) -> None:
        super().__init__()
        self._app = app
        self._context = context
        self._runtime_store = runtime_store
        self._edit_mode_controller = edit_mode_controller
        self._on_state_changed = on_state_changed
        self._capture_service = capture_service or ScreenCaptureService()
        self._change_detector = change_detector or RegionChangeDetector()
        self._pending_group_id: int | None = None
        self._selection_boxes: dict[int, SelectionBoxWidget] = {}
        self._translation_windows: dict[int, TranslationWindowWidget] = {}

        # --- polling pipeline ------------------------------------------------
        self._ocr_engine: OcrEngine | None = None
        self._translation_service: TranslationService | None = None
        self._poll_executor = ThreadPoolExecutor(max_workers=2)
        self._last_logged_ocr: dict[int, str] = {}
        self._processing_groups: set[int] = set()
        self._processing_lock = Lock()
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._tick)
        self._translation_ready.connect(self._on_translation_ready)

        self.selection_overlay = RegionSelectionOverlay(app)
        self.selection_overlay.selection_completed.connect(self.finalize_selection)
        self.selection_overlay.selection_cancelled.connect(self.cancel_selection)

    @property
    def selection_box_count(self) -> int:
        """Expose the number of live selection-box widgets for tests and diagnostics."""

        return len(self._selection_boxes)

    @property
    def translation_window_count(self) -> int:
        """Expose the number of live translation windows for tests and diagnostics."""

        return len(self._translation_windows)

    def request_new_selection(self) -> bool:
        """Open the full-screen selector for the next available group."""

        if self._pending_group_id is not None:
            self._context.status_message = "\u5df2\u6709\u9009\u62e9\u6846\u6b63\u5728\u521b\u5efa\u3002"
            self._notify_state_changed()
            return False

        next_group_id = self._runtime_store.next_available_group_id()
        if next_group_id is None:
            self._context.status_message = "\u6700\u591a\u652f\u6301 3 \u4e2a\u9009\u62e9\u6846\u3002"
            self._notify_state_changed()
            return False

        self._pending_group_id = next_group_id
        self._context.status_message = f"\u6b63\u5728\u6846\u9009\u7b2c {next_group_id} \u7ec4\u3002"
        self._notify_state_changed()
        self.selection_overlay.begin(next_group_id)
        return True

    def request_reselect(self, group_id: int) -> bool:
        """Reopen the full-screen selector for one existing group."""

        if self._pending_group_id is not None:
            self._context.status_message = "\u8bf7\u5148\u5b8c\u6210\u5f53\u524d\u6846\u9009\u3002"
            self._notify_state_changed()
            return False
        if group_id not in self._runtime_store.regions:
            return False

        self._pending_group_id = group_id
        self._context.status_message = f"\u6b63\u5728\u91cd\u9009\u7b2c {group_id} \u7ec4\u3002"
        self._notify_state_changed()
        self.selection_overlay.begin(group_id)
        return True

    def finalize_selection(self, region: ScreenRegion) -> bool:
        """Complete the current selection and show linked overlays."""

        if self._pending_group_id is None:
            return False
        if not region.is_valid():
            self._context.status_message = "\u6846\u9009\u533a\u57df\u592a\u5c0f\uff0c\u8bf7\u91cd\u65b0\u9009\u62e9\u3002"
            self._pending_group_id = None
            self._notify_state_changed()
            return False

        group_id = self._pending_group_id
        self._pending_group_id = None
        self._runtime_store.save_region(group_id, region)
        self._runtime_store.clear_translation_window_position(group_id)
        self._change_detector.reset_group(group_id)

        # Apply default language pair
        config = self._runtime_store.configs[group_id]
        config.source_language = self._context.default_source_language
        config.target_language = self._context.default_target_language

        self._upsert_selection_box(group_id, region)
        self._upsert_translation_window(group_id, region)

        if not self._poll_timer.isActive():
            self._poll_timer.start()

        self._context.status_message = f"\u5df2\u521b\u5efa\u7b2c {group_id} \u7ec4\u9009\u62e9\u6846\u3002"
        self._notify_state_changed()
        return True

    def cancel_selection(self) -> None:
        """Cancel the in-progress selection session."""

        self._pending_group_id = None
        self._context.status_message = "\u5df2\u53d6\u6d88\u6846\u9009\u3002"
        self._notify_state_changed()

    def resize_group(self, group_id: int, x: int, y: int, w: int, h: int) -> bool:
        """Persist a resized selection box and refresh linked overlays."""

        region = self._runtime_store.regions.get(group_id)
        if region is None:
            return False

        updated_region = ScreenRegion(x=x, y=y, width=w, height=h)
        self._runtime_store.save_region(group_id, updated_region)
        self._change_detector.reset_group(group_id)

        self._upsert_selection_box(group_id, updated_region)
        if group_id not in self._runtime_store.translation_window_positions:
            self._upsert_translation_window(group_id, updated_region)

        self._context.status_message = f"\u5df2\u8c03\u6574\u7b2c {group_id} \u7ec4\u5927\u5c0f\u3002"
        self._notify_state_changed()
        return True

    def move_group(self, group_id: int, x: int, y: int) -> bool:
        """Persist a moved selection box and refresh linked overlays."""

        region = self._runtime_store.regions.get(group_id)
        if region is None:
            return False

        updated_region = ScreenRegion(x=x, y=y, width=region.width, height=region.height)
        self._runtime_store.save_region(group_id, updated_region)
        self._upsert_selection_box(group_id, updated_region)

        if group_id not in self._runtime_store.translation_window_positions:
            self._upsert_translation_window(group_id, updated_region)

        self._context.status_message = f"\u5df2\u79fb\u52a8\u7b2c {group_id} \u7ec4\u3002"
        self._notify_state_changed()
        return True

    def move_translation_window(self, group_id: int, x: int, y: int) -> bool:
        """Persist a manually dragged translation-window position."""

        if group_id not in self._runtime_store.regions:
            return False

        self._runtime_store.set_translation_window_position(group_id, x, y)
        self._upsert_translation_window(group_id, self._runtime_store.regions[group_id])
        self._context.status_message = f"\u5df2\u79fb\u52a8\u7b2c {group_id} \u7ec4\u7ffb\u8bd1\u7a97\u3002"
        self._notify_state_changed()
        return True

    def set_translation_dock(self, group_id: int, dock: str) -> bool:
        """Change the dock direction and clear any manual translation-window override."""

        if group_id not in self._runtime_store.configs or dock not in {"top", "bottom", "left", "right"}:
            return False

        self._runtime_store.configs[group_id].translation_dock = dock
        self._runtime_store.clear_translation_window_position(group_id)
        self._upsert_translation_window(group_id, self._runtime_store.regions[group_id])
        dock_labels = {
            "top": "\u4e0a\u4fa7",
            "bottom": "\u4e0b\u4fa7",
            "left": "\u5de6\u4fa7",
            "right": "\u53f3\u4fa7",
        }
        self._context.status_message = (
            f"\u7b2c {group_id} \u7ec4\u7ffb\u8bd1\u7a97\u5df2\u5207\u6362\u5230{dock_labels[dock]}\u3002"
        )
        self._notify_state_changed()
        return True

    def toggle_group_pause(self, group_id: int) -> bool:
        """Toggle paused state for one group and refresh linked overlays."""

        if group_id not in self._runtime_store.configs:
            return False

        config = self._runtime_store.configs[group_id]
        config.paused = not config.paused
        region = self._runtime_store.regions[group_id]
        self._upsert_selection_box(group_id, region)
        self._upsert_translation_window(group_id, region)

        label = "\u6682\u505c" if config.paused else "\u7ee7\u7eed"
        self._context.status_message = (
            f"\u7b2c {group_id} \u7ec4\u5df2{label}\u3002"
        )
        self._notify_state_changed()
        return config.paused

    def delete_group(self, group_id: int) -> bool:
        """Delete one existing selection group and close all linked overlays."""

        if group_id not in self._runtime_store.regions:
            return False

        self._runtime_store.remove_group(group_id)
        self._change_detector.reset_group(group_id)
        self._clear_processing(group_id)
        if self._translation_service is not None:
            self._translation_service.reset_group(group_id)
        self._last_logged_ocr.pop(group_id, None)

        box = self._selection_boxes.pop(group_id, None)
        if box is not None:
            box.close()

        translation_window = self._translation_windows.pop(group_id, None)
        if translation_window is not None:
            translation_window.close()

        self._context.status_message = f"\u5df2\u5220\u9664\u7b2c {group_id} \u7ec4\u3002"
        self._notify_state_changed()
        return True

    def toggle_edit_mode(self) -> bool:
        """Toggle edit mode and refresh all visible overlays."""

        enabled = self._edit_mode_controller.toggle()
        self._context.edit_mode_enabled = enabled
        self._context.status_message = (
            "\u7f16\u8f91\u6a21\u5f0f\u5df2\u5f00\u542f\u3002"
            if enabled
            else "\u7f16\u8f91\u6a21\u5f0f\u5df2\u5173\u95ed\u3002"
        )

        for box in self._selection_boxes.values():
            box.apply_edit_mode(enabled)
        for translation_window in self._translation_windows.values():
            translation_window.apply_edit_mode(enabled)
        for box in self._selection_boxes.values():
            if box.toolbar_visible:
                box.toolbar_panel.raise_()

        self._notify_state_changed()
        return enabled

    def get_selection_box(self, group_id: int) -> SelectionBoxWidget | None:
        """Return one selection-box widget by group id."""

        return self._selection_boxes.get(group_id)

    def get_translation_window(self, group_id: int) -> TranslationWindowWidget | None:
        """Return one linked translation window by group id."""

        return self._translation_windows.get(group_id)

    # ------------------------------------------------------------------
    # polling pipeline
    # ------------------------------------------------------------------

    def _is_processing(self, group_id: int) -> bool:
        """Return whether this group already has an OCR task in flight."""

        with self._processing_lock:
            return group_id in self._processing_groups

    def _try_mark_processing(self, group_id: int) -> bool:
        """Mark a group as processing, unless it is already busy."""

        with self._processing_lock:
            if group_id in self._processing_groups:
                return False
            self._processing_groups.add(group_id)
            return True

    def _clear_processing(self, group_id: int) -> None:
        """Release the in-flight OCR marker for one group."""

        with self._processing_lock:
            self._processing_groups.discard(group_id)

    def _tick(self) -> None:
        """Called by QTimer every 500 ms — scan active groups for changes."""

        t0 = time.perf_counter()

        for group_id in self._runtime_store.active_group_ids():
            config = self._runtime_store.configs.get(group_id)
            if config is None or config.paused:
                continue
            if self._is_processing(group_id):
                continue

            region = self._runtime_store.regions[group_id]
            screen = self._app.primaryScreen()

            t_cap = time.perf_counter()
            try:
                box = self._selection_boxes.get(group_id)
                if box is None:
                    frame = self._capture_service.capture(screen, region)
                else:
                    frame = box.capture_with_chrome_hidden(
                        lambda: self._capture_service.capture(screen, region)
                    )
            except ValueError:
                continue
            cap_ms = (time.perf_counter() - t_cap) * 1000

            t_chg = time.perf_counter()
            changed = self._change_detector.should_process(group_id, frame)
            chg_ms = (time.perf_counter() - t_chg) * 1000

            if not changed:
                continue

            tick_ms = (time.perf_counter() - t0) * 1000
            get_debug_logger().debug(
                "[G%d] tick: capture=%.1fms change=%.1fms total=%.1fms",
                group_id, cap_ms, chg_ms, tick_ms,
            )
            if not self._try_mark_processing(group_id):
                continue
            try:
                self._poll_executor.submit(self._process_frame, group_id, frame)
            except Exception:
                self._clear_processing(group_id)
                raise

    def _process_frame(self, group_id: int, frame) -> None:
        """Run OCR and translation, then release this group's in-flight marker."""

        try:
            self._process_frame_inner(group_id, frame)
        finally:
            self._clear_processing(group_id)

    def _process_frame_inner(self, group_id: int, frame) -> None:
        """Run OCR and request translation — executes on a background thread."""

        log = get_logger()
        t0 = time.perf_counter()

        engine = self._ensure_ocr()
        config = self._runtime_store.configs.get(group_id)
        source_lang = config.source_language if config else "English"

        try:
            result = engine.recognise(frame, source_lang)
        except RuntimeError:
            log.warning("[G%d] OCR engine init failed", group_id)
            return

        ocr_ms = (time.perf_counter() - t0) * 1000

        if result.is_empty:
            get_debug_logger().debug("[G%d] OCR empty (%.0fms)", group_id, ocr_ms)
            self._translation_ready.emit(group_id, "\u672a\u8bc6\u522b\u5230\u6587\u672c")
            return

        pair = f"{source_lang}→{config.target_language}"

        clean_text = normalize_ocr_text(result.raw_text)
        if not clean_text:
            return

        last = self._last_logged_ocr.get(group_id, "")
        if clean_text == last or is_duplicate_ocr_text(clean_text, last):
            get_debug_logger().debug("[G%d] OCR dup skipped: %r", group_id, clean_text[:60])
            return
        self._last_logged_ocr[group_id] = clean_text

        log.info("[G%d] %s | OCR: %r (%.0fms)", group_id, pair, clean_text[:80], ocr_ms)
        get_debug_logger().debug("[G%d] OCR: %.0fms  text=%r", group_id, ocr_ms, clean_text[:80])

        state = self._runtime_store.runtime_states[group_id]
        state.latest_ocr_text = clean_text

        service = self._ensure_translation()
        t_req = time.perf_counter()
        service.request_translation(
            group_id=group_id,
            ocr_text=clean_text,
            source_language=source_lang,
            target_language=config.target_language,
            on_result=lambda r, p=pair: self._on_translation_result(group_id, r, t_req, p),
        )

    def _on_translation_result(self, group_id: int, result, t_start: float, pair: str = "") -> None:
        api_ms = (time.perf_counter() - t_start) * 1000
        log = get_logger()

        if result.text:
            log.info("[G%d] %s | API: %r (%.0fms)", group_id, pair, result.text[:80], api_ms)
        else:
            log.warning("[G%d] %s | API 失败: %s (%.0fms)", group_id, pair, result.error, api_ms)

        self._translation_ready.emit(
            group_id,
            (result.text or f"\u7ffb\u8bd1\u5931\u8d25\uff1a{result.error}"),
        )

    def _on_translation_ready(self, group_id: int, text: str) -> None:
        """Update the translation window — always runs on the Qt main thread."""

        region = self._runtime_store.regions.get(group_id)
        if region is None:
            return

        state = self._runtime_store.runtime_states.get(group_id)
        if state is not None:
            state.latest_translation_text = text

        self._upsert_translation_window(group_id, region)

    def _ensure_ocr(self) -> OcrEngine:
        if self._ocr_engine is None:
            self._ocr_engine = OcrEngine()
        return self._ocr_engine

    def _ensure_translation(self) -> TranslationService:
        if self._translation_service is None:
            self._translation_service = TranslationService(self._context.settings)
        return self._translation_service

    def set_source_language(self, group_id: int, language: str) -> None:
        """Handle source-language change from the toolbar combo."""

        config = self._runtime_store.configs.get(group_id)
        if config is None:
            return

        config.source_language = language
        region = self._runtime_store.regions[group_id]
        self._upsert_translation_window(group_id, region)
        self._context.status_message = f"\u7b2c {group_id} \u7ec4\u6e90\u8bed\u8a00\u5df2\u5207\u6362\u5230 {language}\u3002"
        self._notify_state_changed()

    def set_target_language(self, group_id: int, language: str) -> None:
        """Handle target-language change from the toolbar combo."""

        config = self._runtime_store.configs.get(group_id)
        if config is None:
            return

        config.target_language = language
        region = self._runtime_store.regions[group_id]
        self._upsert_translation_window(group_id, region)
        self._context.status_message = f"\u7b2c {group_id} \u7ec4\u76ee\u6807\u8bed\u8a00\u5df2\u5207\u6362\u5230 {language}\u3002"
        self._notify_state_changed()

    def close(self) -> None:
        """Close all overlay widgets managed by this controller."""

        self._poll_timer.stop()
        self._poll_executor.shutdown(wait=False)
        if self._translation_service is not None:
            self._translation_service.shutdown()

        self.selection_overlay.close()
        for box in self._selection_boxes.values():
            box.close()
        for translation_window in self._translation_windows.values():
            translation_window.close()

    def _upsert_selection_box(self, group_id: int, region: ScreenRegion) -> None:
        config = self._runtime_store.configs[group_id]
        model = SelectionBoxModel(
            group_id=group_id,
            x=region.x,
            y=region.y,
            width=region.width,
            height=region.height,
            accent_color=GROUP_COLORS[group_id],
            paused=config.paused,
            source_language=config.source_language,
            target_language=config.target_language,
        )

        box = self._selection_boxes.get(group_id)
        if box is None:
            box = SelectionBoxWidget(model)
            box.reselect_requested.connect(self.request_reselect)
            box.delete_requested.connect(self.delete_group)
            box.pause_toggled.connect(self.toggle_group_pause)
            box.source_language_changed.connect(self.set_source_language)
            box.target_language_changed.connect(self.set_target_language)
            box.moved.connect(self.move_group)
            box.resized.connect(self.resize_group)
            self._selection_boxes[group_id] = box
        else:
            box.apply_model(model)

        box.apply_edit_mode(self._edit_mode_controller.enabled)
        box.show()

    def _upsert_translation_window(self, group_id: int, region: ScreenRegion) -> None:
        config = self._runtime_store.configs[group_id]
        runtime_state = self._runtime_store.runtime_states[group_id]
        screen = self._app.primaryScreen().virtualGeometry()
        computed_geometry = TranslationWindowWidget.compute_geometry(region, screen, config.translation_dock)
        position_override = self._runtime_store.translation_window_positions.get(group_id)
        x, y = position_override if position_override is not None else (computed_geometry.x(), computed_geometry.y())
        text = runtime_state.latest_translation_text or (
            "\u5df2\u6682\u505c\uff0c\u7b49\u5f85\u7ee7\u7eed\u3002"
            if config.paused
            else "\u7b49\u5f85\u76d1\u6d4b\u753b\u9762\u53d8\u5316..."
        )
        model = TranslationWindowModel(
            group_id=group_id,
            x=x,
            y=y,
            width=computed_geometry.width(),
            height=computed_geometry.height(),
            source_language=config.source_language,
            target_language=config.target_language,
            text=text,
            preferred_dock=config.translation_dock,
            accent_color=GROUP_COLORS[group_id],
        )

        translation_window = self._translation_windows.get(group_id)
        if translation_window is None:
            translation_window = TranslationWindowWidget(model)
            translation_window.moved.connect(self.move_translation_window)
            translation_window.dock_changed.connect(self.set_translation_dock)
            self._translation_windows[group_id] = translation_window
        else:
            translation_window.apply_model(model)

        translation_window.apply_edit_mode(self._edit_mode_controller.enabled)
        translation_window.show()

    def _notify_state_changed(self) -> None:
        self._context.active_group_count = len(self._runtime_store.active_group_ids())
        self._on_state_changed()
