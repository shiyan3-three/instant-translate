"""Selection workflow controller for creating and displaying fixed screen regions."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from app.app_context import ApplicationContext
from app.capture.change_detector import RegionChangeDetector
from app.capture.screen_capture import ScreenCaptureService
from app.feedback.store import FeedbackStore
from app.logger import get_logger, get_debug_logger
from app.ocr.engine import OcrEngine
from app.ocr.postprocess import (
    is_duplicate_ocr_text,
    is_stable_ocr_text,
    is_suspicious_ocr_text,
    normalize_ocr_text,
)
from app.overlay.edit_mode_controller import EditModeController
from app.overlay.ocr_text_window import OcrTextWindowModel, OcrTextWindowWidget
from app.overlay.selection_box import SelectionBoxModel, SelectionBoxWidget
from app.overlay.selection_overlay import RegionSelectionOverlay
from app.overlay.translation_window import TranslationWindowModel, TranslationWindowWidget
from app.state.group_state import GroupRuntimeState, ScreenRegion
from app.state.runtime_store import RuntimeStore
from app.translation.service import TranslationService

GROUP_COLORS = {
    1: "#2F80ED",
    2: "#27AE60",
    3: "#F2994A",
}


@dataclass
class OcrCandidate:
    """Per-group OCR text waiting until it remains stable long enough."""

    text: str
    first_seen_at: float
    seen_count: int = 1


class SelectionWorkflowController(QObject):
    """Coordinate selection boxes with their linked translation windows."""

    _ocr_text_ready = Signal(int, str)
    _translation_ready = Signal(int, str)
    OCR_STABLE_SECONDS = 2.0
    OCR_STABLE_MIN_READS = 2

    def __init__(
        self,
        app: QApplication,
        context: ApplicationContext,
        runtime_store: RuntimeStore,
        edit_mode_controller: EditModeController,
        on_state_changed,
        capture_service: ScreenCaptureService | None = None,
        change_detector: RegionChangeDetector | None = None,
        feedback_store: FeedbackStore | None = None,
    ) -> None:
        super().__init__()
        self._app = app
        self._context = context
        self._runtime_store = runtime_store
        self._edit_mode_controller = edit_mode_controller
        self._on_state_changed = on_state_changed
        self._capture_service = capture_service or ScreenCaptureService()
        self._change_detector = change_detector or RegionChangeDetector()
        self._feedback_store = feedback_store or FeedbackStore()
        self._pending_group_id: int | None = None
        self._selection_boxes: dict[int, SelectionBoxWidget] = {}
        self._translation_windows: dict[int, TranslationWindowWidget] = {}
        self._ocr_windows: dict[int, OcrTextWindowWidget] = {}

        # --- polling pipeline ------------------------------------------------
        # _ocr_engine is created eagerly below, after the executor is ready.
        self._translation_service: TranslationService | None = None
        self._poll_executor = ThreadPoolExecutor(max_workers=2)
        self._last_logged_ocr: dict[int, str] = {}
        self._pending_ocr_candidates: dict[int, OcrCandidate] = {}
        self._ocr_stable_seconds = self.OCR_STABLE_SECONDS
        self._stable_clock = time.monotonic
        self._processing_groups: dict[int, int] = {}
        self._group_versions: dict[int, int] = {}
        self._processing_lock = Lock()
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._tick)
        self._ocr_text_ready.connect(self._on_ocr_text_ready)
        self._translation_ready.connect(self._on_translation_ready)

        self.selection_overlay = RegionSelectionOverlay(app)
        self.selection_overlay.selection_completed.connect(self.finalize_selection)
        self.selection_overlay.selection_cancelled.connect(self.cancel_selection)

        # Eagerly create the OCR engine and warm it up in the background
        # so the first real recognition avoids a multi-second cold start.
        self._ocr_engine = OcrEngine()
        default_src = getattr(context, 'default_source_language', 'English')
        self._poll_executor.submit(self._ocr_engine.warm_up, default_src)

    @property
    def selection_box_count(self) -> int:
        """Expose the number of live selection-box widgets for tests and diagnostics."""

        return len(self._selection_boxes)

    @property
    def translation_window_count(self) -> int:
        """Expose the number of live translation windows for tests and diagnostics."""

        return len(self._translation_windows)

    @property
    def ocr_window_count(self) -> int:
        """Expose the number of live OCR viewer windows for tests and diagnostics."""

        return len(self._ocr_windows)

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
        group_version = self._bump_group_version(group_id)
        self._runtime_store.save_region(group_id, region)
        self._runtime_store.runtime_states[group_id] = GroupRuntimeState()
        self._runtime_store.clear_translation_window_position(group_id)
        self._change_detector.reset_group(group_id)
        self._reset_ocr_tracking(group_id)
        self._clear_processing(group_id)
        self._reset_translation_request_state(group_id, "selection finalized")

        # Apply default language pair
        config = self._runtime_store.configs[group_id]
        config.source_language = self._context.default_source_language
        config.target_language = self._context.default_target_language

        self._upsert_selection_box(group_id, region)
        self._upsert_translation_window(group_id, region)

        if not self._poll_timer.isActive():
            self._poll_timer.start()

        get_debug_logger().debug("[G%d] OCR pipeline reset for new selection version=%d", group_id, group_version)
        get_logger().info("[G%d] \u5df2\u521b\u5efa\u9009\u62e9\u6846 (%dx%d @ %d,%d)", group_id, region.width, region.height, region.x, region.y)
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
        self._bump_group_version(group_id)
        self._runtime_store.save_region(group_id, updated_region)
        self._change_detector.reset_group(group_id)
        self._reset_ocr_tracking(group_id)
        self._clear_processing(group_id)
        self._reset_translation_request_state(group_id, "selection resized")

        self._upsert_selection_box(group_id, updated_region)
        if group_id not in self._runtime_store.translation_window_positions:
            self._upsert_translation_window(group_id, updated_region)
        if group_id in self._ocr_windows:
            self._upsert_ocr_window(group_id, updated_region)

        self._context.status_message = f"\u5df2\u8c03\u6574\u7b2c {group_id} \u7ec4\u5927\u5c0f\u3002"
        self._notify_state_changed()
        return True

    def move_group(self, group_id: int, x: int, y: int) -> bool:
        """Persist a moved selection box and refresh linked overlays."""

        region = self._runtime_store.regions.get(group_id)
        if region is None:
            return False

        updated_region = ScreenRegion(x=x, y=y, width=region.width, height=region.height)
        self._bump_group_version(group_id)
        self._runtime_store.save_region(group_id, updated_region)
        self._change_detector.reset_group(group_id)
        self._reset_ocr_tracking(group_id)
        self._clear_processing(group_id)
        self._reset_translation_request_state(group_id, "selection moved")
        self._upsert_selection_box(group_id, updated_region)

        if group_id not in self._runtime_store.translation_window_positions:
            self._upsert_translation_window(group_id, updated_region)
        if group_id in self._ocr_windows:
            self._upsert_ocr_window(group_id, updated_region)

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
        self._upsert_selection_box(group_id, self._runtime_store.regions[group_id])
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
        get_logger().info("[G%d] \u5df2%s", group_id, label)
        self._context.status_message = (
            f"\u7b2c {group_id} \u7ec4\u5df2{label}\u3002"
        )
        self._notify_state_changed()
        return config.paused

    def delete_group(self, group_id: int) -> bool:
        """Delete one existing selection group and close all linked overlays."""

        if group_id not in self._runtime_store.regions:
            return False

        self._bump_group_version(group_id)
        self._clear_processing(group_id)
        self._runtime_store.remove_group(group_id)
        self._change_detector.reset_group(group_id)
        if self._translation_service is not None:
            self._translation_service.invalidate_group_requests(group_id, reason="selection deleted")
            self._translation_service.reset_group(group_id)
            self._translation_service.reset_agent(group_id)
        self._reset_ocr_tracking(group_id)

        box = self._selection_boxes.pop(group_id, None)
        if box is not None:
            box.close()

        translation_window = self._translation_windows.pop(group_id, None)
        if translation_window is not None:
            translation_window.close()

        ocr_window = self._ocr_windows.pop(group_id, None)
        if ocr_window is not None:
            ocr_window.close()

        get_logger().info("[G%d] \u5df2\u5220\u9664\u9009\u62e9\u6846", group_id)
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
        if enabled:
            for group_id in self._runtime_store.active_group_ids():
                self._change_detector.reset_group(group_id)
                self._reset_ocr_tracking(group_id)
                self._invalidate_translation_requests(group_id, "edit mode enabled")

        self._notify_state_changed()
        return enabled

    def get_selection_box(self, group_id: int) -> SelectionBoxWidget | None:
        """Return one selection-box widget by group id."""

        return self._selection_boxes.get(group_id)

    def get_translation_window(self, group_id: int) -> TranslationWindowWidget | None:
        """Return one linked translation window by group id."""

        return self._translation_windows.get(group_id)

    def get_ocr_window(self, group_id: int) -> OcrTextWindowWidget | None:
        """Return one linked OCR viewer window by group id."""

        return self._ocr_windows.get(group_id)

    def toggle_ocr_window(self, group_id: int) -> bool:
        """Open or close the OCR text viewer for one group."""

        region = self._runtime_store.regions.get(group_id)
        if region is None:
            return False

        existing = self._ocr_windows.pop(group_id, None)
        if existing is not None:
            existing.close()
            self._context.status_message = f"\u5df2\u5173\u95ed\u7b2c {group_id} \u7ec4 OCR \u6587\u672c\u7a97\u3002"
            self._notify_state_changed()
            return False

        self._upsert_ocr_window(group_id, region)
        self._context.status_message = f"\u5df2\u6253\u5f00\u7b2c {group_id} \u7ec4 OCR \u6587\u672c\u7a97\u3002"
        self._notify_state_changed()
        return True

    def mark_translation_feedback(self, group_id: int) -> bool:
        """Save the current OCR/translation pair as pending user feedback."""

        config = self._runtime_store.configs.get(group_id)
        state = self._runtime_store.runtime_states.get(group_id)
        if config is None or state is None:
            return False

        ocr_text = state.latest_ocr_text.strip()
        translation_text = state.latest_translation_text.strip()
        if not ocr_text or not translation_text or translation_text.startswith("翻译失败"):
            self._context.status_message = f"第 {group_id} 组暂无可标记的翻译。"
            self._notify_state_changed()
            return False

        record = self._feedback_store.add_feedback(
            group_id=group_id,
            source_language=config.source_language,
            target_language=config.target_language,
            ocr_text=ocr_text,
            translation_text=translation_text,
        )
        get_logger().info("[G%d] Translation feedback marked | id=%s", group_id, record.id[:8])
        self._context.status_message = f"已标记第 {group_id} 组翻译，稍后可在主界面“优化翻译”中处理。"
        self._notify_state_changed()
        return True

    # ------------------------------------------------------------------
    # polling pipeline
    # ------------------------------------------------------------------

    def _is_processing(self, group_id: int) -> bool:
        """Return whether this group already has an OCR task in flight."""

        with self._processing_lock:
            return group_id in self._processing_groups

    def _try_mark_processing(self, group_id: int, group_version: int) -> bool:
        """Mark a group as processing, unless it is already busy."""

        with self._processing_lock:
            if group_id in self._processing_groups:
                return False
            self._processing_groups[group_id] = group_version
            return True

    def _clear_processing(self, group_id: int, group_version: int | None = None) -> None:
        """Release the in-flight OCR marker for one group."""

        with self._processing_lock:
            if group_version is None or self._processing_groups.get(group_id) == group_version:
                self._processing_groups.pop(group_id, None)

    def _bump_group_version(self, group_id: int) -> int:
        """Invalidate older async OCR work for one group id."""

        self._group_versions[group_id] = self._group_versions.get(group_id, 0) + 1
        return self._group_versions[group_id]

    def _group_version(self, group_id: int) -> int:
        return self._group_versions.get(group_id, 0)

    def _is_stale_group_work(self, group_id: int, group_version: int) -> bool:
        return (
            group_version != self._group_version(group_id)
            or group_id not in self._runtime_store.regions
            or group_id not in self._runtime_store.runtime_states
        )

    def _reset_ocr_tracking(self, group_id: int) -> None:
        """Clear accepted and pending OCR text for one group."""

        self._last_logged_ocr.pop(group_id, None)
        self._pending_ocr_candidates.pop(group_id, None)

    def _invalidate_translation_requests(self, group_id: int, reason: str) -> None:
        """Mark in-flight translations stale when OCR/region state changes."""

        if self._translation_service is None:
            return
        invalidate = getattr(self._translation_service, "invalidate_group_requests", None)
        if invalidate is not None:
            invalidate(group_id, reason=reason)

    def _reset_translation_request_state(self, group_id: int, reason: str) -> None:
        """Invalidate in-flight work and clear duplicate-text dedupe for one group."""

        if self._translation_service is None:
            return
        self._translation_service.invalidate_group_requests(group_id, reason=reason)
        self._translation_service.reset_group(group_id)

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
                frame = self._capture_service.capture(screen, region)
            except ValueError:
                continue
            cap_ms = (time.perf_counter() - t_cap) * 1000

            t_chg = time.perf_counter()
            changed = self._change_detector.should_process(group_id, frame)
            chg_ms = (time.perf_counter() - t_chg) * 1000

            has_pending_ocr = group_id in self._pending_ocr_candidates
            if not changed and not has_pending_ocr:
                continue

            tick_ms = (time.perf_counter() - t0) * 1000
            get_debug_logger().debug(
                "[G%d] tick: capture=%.1fms change=%.1fms total=%.1fms",
                group_id, cap_ms, chg_ms, tick_ms,
            )
            group_version = self._group_version(group_id)
            if not self._try_mark_processing(group_id, group_version):
                continue
            try:
                self._poll_executor.submit(self._process_frame, group_id, frame, group_version)
            except Exception:
                self._clear_processing(group_id, group_version)
                raise

    def _process_frame(self, group_id: int, frame, group_version: int | None = None) -> None:
        """Run OCR and translation, then release this group's in-flight marker."""

        if group_version is None:
            group_version = self._group_version(group_id)
        try:
            if self._is_stale_group_work(group_id, group_version):
                get_debug_logger().debug(
                    "[G%d] OCR worker ignored before start: stale version=%d current=%d",
                    group_id,
                    group_version,
                    self._group_version(group_id),
                )
                return
            self._process_frame_inner(group_id, frame, group_version)
        finally:
            self._clear_processing(group_id, group_version)

    def _process_frame_inner(self, group_id: int, frame, group_version: int) -> None:
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
        if self._is_stale_group_work(group_id, group_version):
            get_debug_logger().debug(
                "[G%d] OCR worker ignored after recognise: stale version=%d current=%d",
                group_id,
                group_version,
                self._group_version(group_id),
            )
            return

        if result.is_empty:
            get_debug_logger().debug("[G%d] OCR empty (%.0fms)", group_id, ocr_ms)
            self._pending_ocr_candidates.pop(group_id, None)
            self._invalidate_translation_requests(group_id, "ocr empty")
            self._translation_ready.emit(group_id, "\u672a\u8bc6\u522b\u5230\u6587\u672c")
            return

        pair = f"{source_lang}→{config.target_language}"

        clean_text = normalize_ocr_text(result.raw_text)
        if not clean_text:
            self._pending_ocr_candidates.pop(group_id, None)
            self._invalidate_translation_requests(group_id, "ocr normalized empty")
            return
        if is_suspicious_ocr_text(clean_text, source_lang):
            self._pending_ocr_candidates.pop(group_id, None)
            self._invalidate_translation_requests(group_id, "ocr suspicious")
            get_debug_logger().debug("[G%d] OCR suspicious skipped: %r", group_id, clean_text[:60])
            return

        last = self._last_logged_ocr.get(group_id, "")
        if clean_text == last or is_duplicate_ocr_text(clean_text, last):
            get_debug_logger().debug("[G%d] OCR dup skipped: %r", group_id, clean_text[:60])
            return

        now = self._stable_clock()
        pending = self._pending_ocr_candidates.get(group_id)
        if pending is None:
            self._pending_ocr_candidates[group_id] = OcrCandidate(clean_text, first_seen_at=now)
            self._invalidate_translation_requests(group_id, "ocr candidate started")
            get_debug_logger().debug("[G%d] OCR waiting for stable window: %r", group_id, clean_text[:60])
            return

        same_candidate = clean_text == pending.text or is_duplicate_ocr_text(clean_text, pending.text)
        if not same_candidate:
            fuzzy_stable = is_stable_ocr_text(clean_text, pending.text)
            self._pending_ocr_candidates[group_id] = OcrCandidate(clean_text, first_seen_at=now)
            self._invalidate_translation_requests(group_id, "ocr candidate changed")
            get_debug_logger().debug(
                "[G%d] OCR candidate changed; restarting stable window fuzzy=%s text=%r",
                group_id,
                fuzzy_stable,
                clean_text[:60],
            )
            return

        pending.seen_count += 1
        stable_for = now - pending.first_seen_at
        if pending.seen_count < self.OCR_STABLE_MIN_READS or stable_for < self._ocr_stable_seconds:
            get_debug_logger().debug(
                "[G%d] OCR stable candidate waiting %.1fs/%.1fs reads=%d text=%r",
                group_id,
                stable_for,
                self._ocr_stable_seconds,
                pending.seen_count,
                clean_text[:60],
            )
            return
        if self._is_stale_group_work(group_id, group_version):
            get_debug_logger().debug(
                "[G%d] OCR worker ignored before accept: stale version=%d current=%d",
                group_id,
                group_version,
                self._group_version(group_id),
            )
            return
        self._pending_ocr_candidates.pop(group_id, None)
        self._last_logged_ocr[group_id] = clean_text

        log.info("[G%d] %s | OCR: %r (%.0fms)", group_id, pair, clean_text[:80], ocr_ms)
        get_debug_logger().debug("[G%d] OCR: %.0fms  text=%r", group_id, ocr_ms, clean_text[:80])

        state = self._runtime_store.runtime_states[group_id]
        state.latest_ocr_text = clean_text
        self._ocr_text_ready.emit(group_id, clean_text)

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

    def _on_ocr_text_ready(self, group_id: int, text: str) -> None:
        """Update the OCR viewer window on the Qt main thread."""

        region = self._runtime_store.regions.get(group_id)
        if region is None:
            return

        state = self._runtime_store.runtime_states.get(group_id)
        if state is not None:
            state.latest_ocr_text = text

        if group_id in self._ocr_windows:
            self._upsert_ocr_window(group_id, region)

    def _ensure_ocr(self) -> OcrEngine:
        if self._ocr_engine is None:
            self._ocr_engine = OcrEngine()
        return self._ocr_engine

    def _ensure_translation(self) -> TranslationService:
        if self._translation_service is None:
            self._translation_service = TranslationService(
                self._context.settings,
                feedback_store=self._feedback_store,
            )
        return self._translation_service

    def set_source_language(self, group_id: int, language: str) -> None:
        """Handle source-language change from the toolbar combo."""

        config = self._runtime_store.configs.get(group_id)
        if config is None:
            return

        config.source_language = language
        self._reset_ocr_tracking(group_id)
        if self._translation_service is not None:
            self._translation_service.reset_group(group_id)
            self._translation_service.reset_agent(group_id)
        region = self._runtime_store.regions[group_id]
        self._upsert_translation_window(group_id, region)
        get_logger().info("[G%d] \u6e90\u8bed\u8a00\u5207\u6362\u4e3a %s", group_id, language)
        self._context.status_message = f"\u7b2c {group_id} \u7ec4\u6e90\u8bed\u8a00\u5df2\u5207\u6362\u5230 {language}\u3002"
        self._notify_state_changed()

    def set_target_language(self, group_id: int, language: str) -> None:
        """Handle target-language change from the toolbar combo."""

        config = self._runtime_store.configs.get(group_id)
        if config is None:
            return

        config.target_language = language
        self._reset_ocr_tracking(group_id)
        if self._translation_service is not None:
            self._translation_service.reset_group(group_id)
            self._translation_service.reset_agent(group_id)
        region = self._runtime_store.regions[group_id]
        self._upsert_translation_window(group_id, region)
        get_logger().info("[G%d] \u76ee\u6807\u8bed\u8a00\u5207\u6362\u4e3a %s", group_id, language)
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
        for ocr_window in self._ocr_windows.values():
            ocr_window.close()

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
            translation_dock=config.translation_dock,
        )

        box = self._selection_boxes.get(group_id)
        if box is None:
            box = SelectionBoxWidget(model)
            box.reselect_requested.connect(self.request_reselect)
            box.ocr_view_requested.connect(self.toggle_ocr_window)
            box.delete_requested.connect(self.delete_group)
            box.pause_toggled.connect(self.toggle_group_pause)
            box.source_language_changed.connect(self.set_source_language)
            box.target_language_changed.connect(self.set_target_language)
            box.translation_dock_changed.connect(self.set_translation_dock)
            box.feedback_requested.connect(self.mark_translation_feedback)
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
            self._translation_windows[group_id] = translation_window
        else:
            translation_window.apply_model(model)

        translation_window.apply_edit_mode(self._edit_mode_controller.enabled)
        translation_window.show()

    def _upsert_ocr_window(self, group_id: int, region: ScreenRegion) -> None:
        runtime_state = self._runtime_store.runtime_states[group_id]
        screen = self._app.primaryScreen().virtualGeometry()
        computed_geometry = OcrTextWindowWidget.compute_geometry(region, screen)
        ocr_window = self._ocr_windows.get(group_id)
        x, y = (
            (ocr_window.x(), ocr_window.y())
            if ocr_window is not None
            else (computed_geometry.x(), computed_geometry.y())
        )
        text = runtime_state.latest_ocr_text or "\u7b49\u5f85 OCR \u6587\u672c..."
        model = OcrTextWindowModel(
            group_id=group_id,
            x=x,
            y=y,
            width=computed_geometry.width(),
            height=computed_geometry.height(),
            text=text,
            accent_color=GROUP_COLORS[group_id],
        )

        if ocr_window is None:
            ocr_window = OcrTextWindowWidget(model)
            self._ocr_windows[group_id] = ocr_window
        else:
            ocr_window.apply_model(model)

        ocr_window.show()

    def _notify_state_changed(self) -> None:
        self._context.active_group_count = len(self._runtime_store.active_group_ids())
        self._on_state_changed()
