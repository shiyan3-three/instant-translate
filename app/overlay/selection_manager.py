"""Selection workflow controller for creating and displaying fixed screen regions."""

from __future__ import annotations

import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

from PySide6.QtCore import QObject, QRect, QTimer, Signal
from PySide6.QtWidgets import QApplication

from app.app_context import ApplicationContext
from app.capture.change_detector import RegionChangeDetector
from app.capture.screen_capture import ScreenCaptureService
from app.feedback.store import FeedbackStorageUnavailable, FeedbackStore
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
    _translation_ready = Signal(int, str, int)
    _translation_retry_requested = Signal(int)
    OCR_STABLE_SECONDS = 2.0
    OCR_STABLE_MIN_READS = 2
    OCR_REGION_SETTLE_SECONDS = 0.8

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
        on_feedback_changed=None,
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
        self._on_feedback_changed = on_feedback_changed or (lambda: None)
        self._pending_group_id: int | None = None
        self._selection_boxes: dict[int, SelectionBoxWidget] = {}
        self._translation_windows: dict[int, TranslationWindowWidget] = {}
        self._ocr_windows: dict[int, OcrTextWindowWidget] = {}
        self._closing = False

        # --- polling pipeline ------------------------------------------------
        # _ocr_engine is created eagerly below, after the executor is ready.
        self._translation_service: TranslationService | None = None
        self._poll_executor = ThreadPoolExecutor(max_workers=2)
        self._ocr_lock = Lock()
        self._last_logged_ocr: dict[int, str] = {}
        self._pending_ocr_candidates: dict[int, OcrCandidate] = {}
        self._empty_ocr_candidates: dict[int, OcrCandidate] = {}
        self._region_settle_until: dict[int, float] = {}
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
        self._translation_retry_requested.connect(
            self._on_translation_retry_requested
        )

        self.selection_overlay = RegionSelectionOverlay(app)
        self.selection_overlay.selection_completed.connect(self.finalize_selection)
        self.selection_overlay.selection_cancelled.connect(self.cancel_selection)

        # Eagerly create the OCR engine and warm it up in the background
        # so the first real recognition avoids a multi-second cold start.
        self._ocr_engine = OcrEngine()
        default_src = getattr(context, 'default_source_language', 'English')
        self._poll_executor.submit(self._warm_up_ocr, default_src)
        self._prepare_agent_profile(
            default_src,
            getattr(context, "default_target_language", "中文"),
        )

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

        if self._closing:
            return False
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
        self._region_settle_until.pop(group_id, None)
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
        self._mark_region_settling(group_id)
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
        self._mark_region_settling(group_id)
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
        """Persist a manually dragged translation-window base position."""

        if group_id not in self._runtime_store.regions:
            return False

        # x/y are base coordinates (without edit clearance) from the widget.
        self._runtime_store.set_translation_window_position(group_id, x, y)
        window = self._translation_windows.get(group_id)
        if window is not None:
            window.set_base_position(x, y)
        else:
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
        # The OCR corner button is the control the user just clicked.  Update
        # it synchronously instead of waiting for a later OCR-window rebuild.
        ocr_window = self._ocr_windows.get(group_id)
        if ocr_window is not None:
            ocr_window.model.paused = config.paused
            ocr_window.set_paused(config.paused)

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
            # A selection box is transient UI state.  Its persisted Agent
            # session belongs to the language/prompt profile and must survive
            # deletion so recreating G1 does not silently lose its digest.
            self._translation_service.reset_agent(group_id)
        self._reset_ocr_tracking(group_id)
        self._region_settle_until.pop(group_id, None)

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
        # Show main toolbars first so measured height is accurate for OCR shift.
        for box in self._selection_boxes.values():
            if enabled and box.toolbar_visible:
                box.toolbar_panel.adjustSize()
                box.toolbar_panel.raise_()

        for group_id, translation_window in self._translation_windows.items():
            shift = self._translation_edit_shift_y(group_id) if enabled else 0
            translation_window.apply_edit_mode(enabled, shift_y=shift)
            if not enabled:
                translation_window.set_body_immersive(False)
                translation_window.set_group_immersive(False)
        for group_id, ocr_window in self._ocr_windows.items():
            shift = self._ocr_edit_shift_y(group_id) if enabled else 0
            ocr_window.apply_edit_mode(enabled, shift_y=shift)
            if not enabled:
                ocr_window.set_body_immersive(False)
                ocr_window.set_group_immersive(False)
        for box in self._selection_boxes.values():
            if not enabled:
                box.set_group_immersive(False)
            if box.toolbar_visible:
                box.toolbar_panel.raise_()
        if enabled:
            for group_id in self._runtime_store.active_group_ids():
                self._change_detector.reset_group(group_id)
                self._reset_ocr_tracking(group_id)
                self._invalidate_translation_requests(group_id, "edit mode enabled")

        self._notify_state_changed()
        return enabled

    def _main_toolbar_height(self, group_id: int) -> int:
        box = self._selection_boxes.get(group_id)
        if box is None:
            return 36
        panel = box.toolbar_panel
        panel.adjustSize()
        return max(1, panel.height())

    def _translation_edit_shift_y(self, group_id: int) -> int:
        """Move translation card down by 1.5× its own corner toolbar height."""

        window = self._translation_windows.get(group_id)
        if window is None:
            return 0
        return int(round(window.corner_toolbar_height() * 1.5))

    def _ocr_edit_shift_y(self, group_id: int) -> int:
        """Move OCR card up by 1.5× main selection toolbar height."""

        return -int(round(self._main_toolbar_height(group_id) * 1.5))

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

        correction_ids: list[str] = []
        memory_rule_ids: list[str] = []
        correction_hint_snapshots: dict[str, str] = {}
        memory_hint_snapshots: dict[str, str] = {}
        if self._translation_service is not None:
            snapshot_method = getattr(
                self._translation_service,
                "memory_provenance_snapshot",
                None,
            )
            if callable(snapshot_method):
                snapshot = snapshot_method(group_id)
                if (
                    snapshot.ocr_text.strip() != ocr_text
                    or snapshot.translation_text.strip() != translation_text
                    or getattr(snapshot, "source_language", config.source_language)
                    != config.source_language
                    or getattr(snapshot, "target_language", config.target_language)
                    != config.target_language
                ):
                    self._context.status_message = (
                        f"第 {group_id} 组的新 OCR 尚未完成翻译，请等待当前译文更新后再标记。"
                    )
                    get_logger().warning(
                        "[G%d] Feedback mark rejected: OCR/translation snapshot mismatch",
                        group_id,
                    )
                    self._notify_state_changed()
                    return False
                correction_ids = list(snapshot.correction_ids)
                memory_rule_ids = list(snapshot.memory_rule_ids)
                correction_hint_snapshots = dict(snapshot.correction_hint_snapshots)
                memory_hint_snapshots = dict(snapshot.memory_hint_snapshots)
            else:
                correction_ids, memory_rule_ids = self._translation_service.memory_provenance(group_id)
        try:
            record = self._feedback_store.add_feedback(
                group_id=group_id,
                source_language=config.source_language,
                target_language=config.target_language,
                ocr_text=ocr_text,
                translation_text=translation_text,
                matched_memory_rule_ids=memory_rule_ids,
                matched_correction_ids=correction_ids,
                matched_memory_hint_snapshots=memory_hint_snapshots,
                matched_correction_hint_snapshots=correction_hint_snapshots,
            )
        except FeedbackStorageUnavailable as exc:
            self._context.status_message = (
                "反馈存储正在等待安全恢复，本次未写入纠错；实时翻译不受影响。"
            )
            get_logger().warning(
                "[G%d] Feedback mark blocked by unavailable storage: %s",
                group_id,
                exc,
            )
            self._notify_state_changed()
            return False
        except Exception as exc:
            self._context.status_message = (
                "反馈保存失败，本次未写入纠错；实时翻译不受影响。"
            )
            get_logger().exception(
                "[G%d] Feedback mark failed: %s",
                group_id,
                exc,
            )
            self._notify_state_changed()
            return False
        get_logger().info("[G%d] Translation feedback marked | id=%s", group_id, record.id[:8])
        self._context.status_message = f"已标记第 {group_id} 组翻译，稍后可在主界面“优化翻译”中处理。"
        try:
            self._on_feedback_changed()
        except Exception:
            # The feedback is already durable.  A view-refresh failure must not
            # report the save itself as failed.
            get_logger().exception("[G%d] Feedback view refresh failed", group_id)
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
        self._empty_ocr_candidates.pop(group_id, None)

    def _mark_region_settling(self, group_id: int) -> None:
        """Pause OCR briefly after manual geometry changes finish."""

        self._region_settle_until[group_id] = (
            self._stable_clock() + self.OCR_REGION_SETTLE_SECONDS
        )

    def _is_region_settling(self, group_id: int) -> bool:
        """Return True while recent move/resize repaint noise may affect OCR."""

        settle_until = self._region_settle_until.get(group_id)
        if settle_until is None:
            return False
        if self._stable_clock() < settle_until:
            return True
        self._region_settle_until.pop(group_id, None)
        return False

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

        if self._closing:
            return
        t0 = time.perf_counter()

        for group_id in self._runtime_store.active_group_ids():
            config = self._runtime_store.configs.get(group_id)
            if config is None or config.paused:
                continue
            if self._is_processing(group_id):
                continue
            if self._is_region_settling(group_id):
                continue

            region = self._runtime_store.regions[group_id]
            screens = self._app.screens()
            if not screens:
                continue

            t_cap = time.perf_counter()
            try:
                frame = self._capture_region(screens, region)
            except ValueError:
                continue
            cap_ms = (time.perf_counter() - t_cap) * 1000

            t_chg = time.perf_counter()
            changed = self._change_detector.should_process(group_id, frame)
            chg_ms = (time.perf_counter() - t_chg) * 1000

            has_pending_ocr = (
                group_id in self._pending_ocr_candidates
                or group_id in self._empty_ocr_candidates
            )
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

    def _capture_region(self, screens, region: ScreenRegion):
        """Capture a virtual-desktop region, retaining old test-double compatibility."""

        capture_region = getattr(self._capture_service, "capture_region", None)
        if callable(capture_region):
            return capture_region(screens, region)
        # Older integrations supplied a one-screen ``capture(screen, region)``
        # service.  Use the screen containing the region centre as a safe
        # compatibility fallback; production uses the cross-screen compositor.
        if not screens:
            raise ValueError("no screens available")
        center_x = region.x + region.width // 2
        center_y = region.y + region.height // 2
        selected = next(
            (
                screen
                for screen in screens
                if screen.geometry().contains(center_x, center_y)
            ),
            screens[0],
        )
        return self._capture_service.capture(selected, region)

    def _process_frame(self, group_id: int, frame, group_version: int | None = None) -> None:
        """Run OCR and translation, then release this group's in-flight marker."""

        if group_version is None:
            group_version = self._group_version(group_id)
        try:
            if self._closing or self._is_stale_group_work(group_id, group_version):
                get_debug_logger().debug(
                    "[G%d] OCR worker ignored before start: stale version=%d current=%d",
                    group_id,
                    group_version,
                    self._group_version(group_id),
                )
                return
            self._process_frame_inner(group_id, frame, group_version)
        except Exception:
            get_debug_logger().exception(
                "[G%d] Unexpected OCR worker failure", group_id
            )
        finally:
            self._clear_processing(group_id, group_version)

    def _process_frame_inner(self, group_id: int, frame, group_version: int) -> None:
        """Run OCR and request translation — executes on a background thread."""

        log = get_logger()
        t0 = time.perf_counter()

        config = self._runtime_store.configs.get(group_id)
        source_lang = config.source_language if config else "English"

        try:
            # PaddleOCR mutates predictor state; serialise both lazy
            # initialisation and recognition across selection groups.
            with self._ocr_lock:
                engine = self._ensure_ocr()
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
            now = self._stable_clock()
            empty = self._empty_ocr_candidates.get(group_id)
            if empty is None:
                self._empty_ocr_candidates[group_id] = OcrCandidate("", first_seen_at=now)
                self._invalidate_translation_requests(group_id, "ocr empty candidate started")
                return
            empty.seen_count += 1
            stable_for = now - empty.first_seen_at
            if (
                empty.seen_count < self.OCR_STABLE_MIN_READS
                or stable_for < self._ocr_stable_seconds
            ):
                return
            self._empty_ocr_candidates.pop(group_id, None)
            if not self._closing:
                self._translation_ready.emit(
                    group_id,
                    "\u672a\u8bc6\u522b\u5230\u6587\u672c",
                    -1,
                )
            return

        self._empty_ocr_candidates.pop(group_id, None)

        pair = f"{source_lang}→{config.target_language}"

        clean_text = normalize_ocr_text(result.raw_text)
        if not clean_text:
            self._pending_ocr_candidates.pop(group_id, None)
            self._invalidate_translation_requests(group_id, "ocr normalized empty")
            return
        if is_suspicious_ocr_text(clean_text, source_lang):
            self._pending_ocr_candidates.pop(group_id, None)
            self._invalidate_translation_requests(group_id, "ocr suspicious")
            get_debug_logger().debug(
                "[G%d] OCR suspicious skipped: len=%d sha256=%s",
                group_id, len(clean_text), self._text_digest(clean_text),
            )
            return

        last = self._last_logged_ocr.get(group_id, "")
        if clean_text == last or is_duplicate_ocr_text(clean_text, last):
            get_debug_logger().debug(
                "[G%d] OCR duplicate skipped: len=%d sha256=%s",
                group_id, len(clean_text), self._text_digest(clean_text),
            )
            return

        now = self._stable_clock()
        pending = self._pending_ocr_candidates.get(group_id)
        if pending is None:
            self._pending_ocr_candidates[group_id] = OcrCandidate(clean_text, first_seen_at=now)
            self._invalidate_translation_requests(group_id, "ocr candidate started")
            get_debug_logger().debug(
                "[G%d] OCR waiting for stable window: len=%d sha256=%s",
                group_id, len(clean_text), self._text_digest(clean_text),
            )
            return

        same_candidate = clean_text == pending.text or is_duplicate_ocr_text(clean_text, pending.text)
        if not same_candidate:
            fuzzy_stable = is_stable_ocr_text(clean_text, pending.text)
            if fuzzy_stable:
                pending.text = clean_text
                pending.seen_count += 1
            else:
                self._pending_ocr_candidates[group_id] = OcrCandidate(clean_text, first_seen_at=now)
                self._invalidate_translation_requests(group_id, "ocr candidate changed")
                get_debug_logger().debug(
                    "[G%d] OCR candidate changed; restarting stable window len=%d sha256=%s",
                    group_id,
                    len(clean_text),
                    self._text_digest(clean_text),
                )
                return
        else:
            pending.seen_count += 1
        stable_for = now - pending.first_seen_at
        if pending.seen_count < self.OCR_STABLE_MIN_READS or stable_for < self._ocr_stable_seconds:
            get_debug_logger().debug(
                "[G%d] OCR stable candidate waiting %.1fs/%.1fs reads=%d len=%d sha256=%s",
                group_id,
                stable_for,
                self._ocr_stable_seconds,
                pending.seen_count,
                len(clean_text),
                self._text_digest(clean_text),
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

        log.info(
            "[G%d] %s | OCR accepted len=%d sha256=%s (%.0fms)",
            group_id,
            pair,
            len(clean_text),
            self._text_digest(clean_text),
            ocr_ms,
        )
        get_debug_logger().debug(
            "[G%d] OCR: %.0fms len=%d sha256=%s",
            group_id, ocr_ms, len(clean_text), self._text_digest(clean_text),
        )

        state = self._runtime_store.runtime_states[group_id]
        state.latest_ocr_text = clean_text
        if self._closing:
            return
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
        if self._closing:
            return
        api_ms = (time.perf_counter() - t_start) * 1000
        log = get_logger()
        source = getattr(result, "source", "api")
        result_revision = getattr(result, "feedback_revision", -1)

        if source == "feedback_revision_churn":
            get_debug_logger().warning(
                "[G%d] Translation revision retry exhausted; scheduling fresh OCR capture",
                group_id,
            )
            self._translation_retry_requested.emit(group_id)
            return
        if (
            result_revision >= 0
            and result_revision != self._current_feedback_revision()
        ):
            get_debug_logger().warning(
                "[G%d] Translation result became stale before UI dispatch; scheduling fresh OCR capture",
                group_id,
            )
            self._translation_retry_requested.emit(group_id)
            return

        if result.text:
            if source == "cache" or getattr(
                result, "from_cache", False
            ):
                log.info(
                    "[G%d] %s | translation cache hit len=%d sha256=%s (%.0fms)",
                    group_id, pair, len(result.text), self._text_digest(result.text), api_ms,
                )
            else:
                log.info(
                    "[G%d] %s | API success len=%d sha256=%s (%.0fms)",
                    group_id, pair, len(result.text), self._text_digest(result.text), api_ms,
                )
        else:
            log.warning("[G%d] %s | API 失败: %s (%.0fms)", group_id, pair, result.error, api_ms)

        self._translation_ready.emit(
            group_id,
            (result.text or f"\u7ffb\u8bd1\u5931\u8d25\uff1a{result.error}"),
            result_revision,
        )

    def _on_translation_ready(
        self,
        group_id: int,
        text: str,
        feedback_revision: int = -1,
    ) -> None:
        """Update the translation window — always runs on the Qt main thread."""

        if self._closing:
            return
        if (
            feedback_revision >= 0
            and feedback_revision != self._current_feedback_revision()
        ):
            get_debug_logger().warning(
                "[G%d] Translation result became stale in UI queue; scheduling fresh OCR capture",
                group_id,
            )
            self._translation_retry_requested.emit(group_id)
            return
        region = self._runtime_store.regions.get(group_id)
        if region is None:
            return

        state = self._runtime_store.runtime_states.get(group_id)
        if state is not None:
            state.latest_translation_text = text

        self._upsert_translation_window(group_id, region)

    def _on_translation_retry_requested(self, group_id: int) -> None:
        """Reopen OCR duplicate detection after bounded revision retries end."""

        if self._closing or group_id not in self._runtime_store.regions:
            return
        self.request_ocr_refresh(group_id)

    def _on_ocr_text_ready(self, group_id: int, text: str) -> None:
        """Update the OCR viewer window on the Qt main thread."""

        if self._closing:
            return
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

    def _warm_up_ocr(self, source_language: str) -> None:
        """Warm the shared OCR engine under the recognition lock."""

        if self._closing:
            return
        with self._ocr_lock:
            self._ensure_ocr().warm_up(source_language)

    def _screen_for_region(self, region: ScreenRegion):
        """Return the screen containing the centre of a desktop region."""

        center_x = region.x + region.width // 2
        center_y = region.y + region.height // 2
        for screen in self._app.screens():
            geometry = screen.geometry()
            if (
                geometry.left() <= center_x <= geometry.right()
                and geometry.top() <= center_y <= geometry.bottom()
            ):
                return screen
        return self._app.primaryScreen()

    @staticmethod
    def _text_digest(text: str) -> str:
        """Return a short content fingerprint suitable for operational logs."""

        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def _ensure_translation(self) -> TranslationService:
        if self._translation_service is None:
            self._translation_service = TranslationService(
                self._context.settings,
                feedback_store=self._feedback_store,
            )
        return self._translation_service

    def _prepare_agent_profile(self, source_language: str, target_language: str) -> None:
        """Schedule Pro/session preparation without waiting for a selection box."""

        ai = self._context.settings.ai
        if not (
            ai.base_url.strip()
            and ai.api_key.strip()
            and ai.fast_model_name.strip()
            and ai.thinking_model_name.strip()
        ):
            return
        self._ensure_translation().prepare_agent_profile(source_language, target_language)

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
        self._prepare_agent_profile(language, config.target_language)
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
        self._prepare_agent_profile(config.source_language, language)
        region = self._runtime_store.regions[group_id]
        self._upsert_translation_window(group_id, region)
        get_logger().info("[G%d] \u76ee\u6807\u8bed\u8a00\u5207\u6362\u4e3a %s", group_id, language)
        self._context.status_message = f"\u7b2c {group_id} \u7ec4\u76ee\u6807\u8bed\u8a00\u5df2\u5207\u6362\u5230 {language}\u3002"
        self._notify_state_changed()

    def reload_translation_agents(self) -> None:
        """Reset all active group agents so they pick up the latest compiled prompt."""

        for group_id in self._runtime_store.active_group_ids():
            self._reset_ocr_tracking(group_id)
            self._invalidate_translation_requests(group_id, "compiled prompt saved")
            if self._translation_service is not None:
                self._translation_service.reset_group(group_id)
                self._translation_service.reset_agent(group_id)
        self._prepare_active_agent_profiles()
        get_logger().info("All active translation agents reset for compiled prompt reload")

    def _prepare_active_agent_profiles(self) -> None:
        """Schedule profile preparation for default and active language pairs."""

        pairs: list[tuple[str, str]] = [
            (
                self._context.default_source_language,
                self._context.default_target_language,
            )
        ]
        for group_id in self._runtime_store.active_group_ids():
            config = self._runtime_store.configs.get(group_id)
            if config is None:
                continue
            pairs.append((config.source_language, config.target_language))

        seen: set[tuple[str, str]] = set()
        for source_language, target_language in pairs:
            pair = (source_language, target_language)
            if pair in seen:
                continue
            seen.add(pair)
            self._prepare_agent_profile(source_language, target_language)

    def request_ocr_refresh(self, group_id: int) -> None:
        """Force re-capture and re-recognise OCR for one group, bypassing change detection."""

        self._change_detector.reset_group(group_id)
        self._reset_ocr_tracking(group_id)
        self._clear_processing(group_id)
        self._tick()

    def _current_feedback_revision(self) -> int:
        revision = getattr(self._feedback_store, "knowledge_revision", -1)
        if callable(revision):
            revision = revision()
        if not isinstance(revision, int) or isinstance(revision, bool):
            return -1
        return revision

    def close(self) -> None:
        """Close all overlay widgets managed by this controller."""

        if self._closing:
            return
        self._closing = True
        self._poll_timer.stop()
        for group_id in list(self._group_versions):
            self._bump_group_version(group_id)
        with self._processing_lock:
            self._processing_groups.clear()
        self._poll_executor.shutdown(wait=False, cancel_futures=True)
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
            box.toolbar_panel.group_immersive_toggled.connect(
                lambda on, gid=group_id: self.set_group_immersive(gid, on)
            )
            self._selection_boxes[group_id] = box
        else:
            box.apply_model(model)

        box.apply_edit_mode(self._edit_mode_controller.enabled)
        box.show()

    def _upsert_translation_window(self, group_id: int, region: ScreenRegion) -> None:
        config = self._runtime_store.configs[group_id]
        runtime_state = self._runtime_store.runtime_states[group_id]
        host_screen = self._screen_for_region(region)
        # Use the monitor that owns the selection, not always the primary desktop.
        screen = host_screen.availableGeometry() if host_screen is not None else self._app.primaryScreen().availableGeometry()
        dpr = float(host_screen.devicePixelRatio()) if host_screen is not None else 1.0
        text = runtime_state.latest_translation_text or (
            "\u5df2\u6682\u505c\uff0c\u7b49\u5f85\u7ee7\u7eed\u3002"
            if config.paused
            else "\u7b49\u5f85\u76d1\u6d4b\u753b\u9762\u53d8\u5316..."
        )
        computed_geometry = TranslationWindowWidget.compute_geometry(
            region,
            screen,
            config.translation_dock,
            text,
            device_pixel_ratio=dpr,
        )
        position_override = self._runtime_store.translation_window_positions.get(group_id)
        x, y = position_override if position_override is not None else (computed_geometry.x(), computed_geometry.y())
        capture_rect = QRect(region.x, region.y, region.width, region.height)
        candidate_rect = QRect(x, y, computed_geometry.width(), computed_geometry.height())
        if candidate_rect.intersects(capture_rect):
            x, y = computed_geometry.x(), computed_geometry.y()
            self._runtime_store.translation_window_positions.pop(group_id, None)
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
            translation_window.feedback_requested.connect(self.mark_translation_feedback)
            translation_window.body_immersive_toggled.connect(
                lambda gid, on: self._on_translation_body_immersive(gid, on)
            )
            self._translation_windows[group_id] = translation_window
        else:
            translation_window.apply_model(model)

        enabled = self._edit_mode_controller.enabled
        shift = self._translation_edit_shift_y(group_id) if enabled else 0
        translation_window.apply_edit_mode(enabled, shift_y=shift)
        # Preserve group immersive chrome-off across model refresh.
        box = self._selection_boxes.get(group_id)
        if box is not None and box.group_immersive:
            translation_window.set_group_immersive(True)
        else:
            translation_window.set_group_immersive(False)
        translation_window.show()

    def _upsert_ocr_window(self, group_id: int, region: ScreenRegion) -> None:
        runtime_state = self._runtime_store.runtime_states[group_id]
        host_screen = self._screen_for_region(region)
        screen = host_screen.availableGeometry() if host_screen is not None else self._app.primaryScreen().availableGeometry()
        dpr = float(host_screen.devicePixelRatio()) if host_screen is not None else 1.0
        text = runtime_state.latest_ocr_text or "\u7b49\u5f85 OCR \u6587\u672c..."
        computed_geometry = OcrTextWindowWidget.compute_geometry(
            region,
            screen,
            text,
            device_pixel_ratio=dpr,
        )
        ocr_window = self._ocr_windows.get(group_id)
        if ocr_window is not None:
            # Keep user-dragged base; never treat shifted screen pos as base.
            x, y = ocr_window.base_position()
        else:
            x, y = computed_geometry.x(), computed_geometry.y()
        capture_rect = QRect(region.x, region.y, region.width, region.height)
        candidate_rect = QRect(x, y, computed_geometry.width(), computed_geometry.height())
        if candidate_rect.intersects(capture_rect):
            x, y = computed_geometry.x(), computed_geometry.y()
        model = OcrTextWindowModel(
            group_id=group_id,
            x=x,
            y=y,
            width=computed_geometry.width(),
            height=computed_geometry.height(),
            text=text,
            accent_color=GROUP_COLORS[group_id],
            paused=self._runtime_store.configs[group_id].paused,
        )

        if ocr_window is None:
            ocr_window = OcrTextWindowWidget(model)
            ocr_window.refresh_clicked.connect(self.request_ocr_refresh)
            ocr_window.pause_toggled.connect(self.toggle_group_pause)
            ocr_window.body_immersive_toggled.connect(
                lambda gid, on: self._on_ocr_body_immersive(gid, on)
            )
            self._ocr_windows[group_id] = ocr_window
        else:
            ocr_window.apply_model(model)

        enabled = self._edit_mode_controller.enabled
        shift = self._ocr_edit_shift_y(group_id) if enabled else 0
        ocr_window.apply_edit_mode(enabled, shift_y=shift)
        box = self._selection_boxes.get(group_id)
        if box is not None and box.group_immersive:
            ocr_window.set_group_immersive(True)
        else:
            ocr_window.set_group_immersive(False)
        ocr_window.show()

    def set_group_immersive(self, group_id: int, on: bool) -> bool:
        """Group ◎: hide selection outline + card chrome; keep body text (mock)."""

        box = self._selection_boxes.get(group_id)
        if box is None:
            return False
        box.set_group_immersive(on)
        translation_window = self._translation_windows.get(group_id)
        if translation_window is not None:
            translation_window.set_group_immersive(on)
        ocr_window = self._ocr_windows.get(group_id)
        if ocr_window is not None:
            ocr_window.set_group_immersive(on)
        return True

    def _on_translation_body_immersive(self, group_id: int, on: bool) -> None:
        """Per-card ◎ only affects the translation window (not whole group)."""

        translation_window = self._translation_windows.get(group_id)
        if translation_window is not None:
            translation_window.set_body_immersive(on)

    def _on_ocr_body_immersive(self, group_id: int, on: bool) -> None:
        """Per-card ◎ only affects the OCR window (not whole group)."""

        ocr_window = self._ocr_windows.get(group_id)
        if ocr_window is not None:
            ocr_window.set_body_immersive(on)

    def _notify_state_changed(self) -> None:
        self._context.active_group_count = len(self._runtime_store.active_group_ids())
        self._on_state_changed()
