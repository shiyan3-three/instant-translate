"""Tests for the fixed-region selection workflow."""

from __future__ import annotations

import unittest

from app.app_context import ApplicationContext
from app.capture.screen_capture import CapturedRegionFrame
from app.ocr.engine import OcrResult
from app.overlay.edit_mode_controller import EditModeController
from app.overlay.selection_manager import SelectionWorkflowController
from app.state.group_state import ScreenRegion
from app.state.runtime_store import RuntimeStore
from tests.test_support import ensure_qapplication


class FakeCaptureService:
    """Capture test double that can inspect overlay state during capture."""

    def __init__(self) -> None:
        self.capture_calls = 0
        self.on_capture = None

    def capture(self, screen, region: ScreenRegion) -> CapturedRegionFrame:
        self.capture_calls += 1
        if self.on_capture is not None:
            self.on_capture()
        return CapturedRegionFrame(width=region.width, height=region.height, pixel_bytes=b"\x00" * 16)


class FakeChangeDetector:
    """Change detector test double that keeps OCR from running."""

    def __init__(self, changed: bool = False) -> None:
        self.changed = changed
        self.reset_calls: list[int] = []

    def should_process(self, group_id: int, frame: CapturedRegionFrame) -> bool:
        return self.changed

    def reset_group(self, group_id: int) -> None:
        self.reset_calls.append(group_id)


class FakeOcrEngine:
    """OCR test double returning fixed text."""

    def __init__(self, text: str) -> None:
        self.text = text

    def recognise(self, frame: CapturedRegionFrame, source_language: str = "English") -> OcrResult:
        return OcrResult(raw_text=self.text)


class SequenceOcrEngine:
    """OCR test double returning one text per recognise call."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls = 0

    def recognise(self, frame: CapturedRegionFrame, source_language: str = "English") -> OcrResult:
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return OcrResult(raw_text=text)


class FakeTranslationService:
    """Translation test double recording requests without network calls."""

    def __init__(self) -> None:
        self.requests: list[dict[str, str | int]] = []
        self.invalidations: list[dict[str, str | int]] = []
        self.reset_calls: list[int] = []

    def request_translation(
        self,
        group_id: int,
        ocr_text: str,
        source_language: str,
        target_language: str,
        on_result,
    ) -> None:
        self.requests.append(
            {
                "group_id": group_id,
                "ocr_text": ocr_text,
                "source_language": source_language,
                "target_language": target_language,
            }
        )

    def reset_group(self, group_id: int) -> None:
        self.reset_calls.append(group_id)

    def invalidate_group_requests(self, group_id: int, reason: str = "") -> None:
        self.invalidations.append({"group_id": group_id, "reason": reason})

    def shutdown(self) -> None:
        pass


class FakeFeedbackStore:
    """Feedback store test double recording saved feedback."""

    def __init__(self) -> None:
        self.calls = []

    def add_feedback(self, **kwargs):
        self.calls.append(kwargs)

        class Record:
            id = "feedback-test-id"

        return Record()


class FakeDeferredExecutor:
    """Executor test double that stores submitted work until tests run it."""

    def __init__(self) -> None:
        self.submitted = []
        self.shutdown_called = False

    def submit(self, fn, *args):
        self.submitted.append((fn, args))
        return None

    def run_next(self) -> None:
        fn, args = self.submitted.pop(0)
        fn(*args)

    def shutdown(self, wait: bool = False) -> None:
        self.shutdown_called = True


class SelectionWorkflowControllerTests(unittest.TestCase):
    """Verify selection creation flow, persistent edit tools, and linked overlays."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()
        self.context = ApplicationContext()
        self.runtime_store = RuntimeStore()
        self.capture_service = FakeCaptureService()
        self.change_detector = FakeChangeDetector()
        self.controller = SelectionWorkflowController(
            app=self.app,
            context=self.context,
            runtime_store=self.runtime_store,
            edit_mode_controller=EditModeController(),
            on_state_changed=lambda: None,
            capture_service=self.capture_service,
            change_detector=self.change_detector,
        )

    def tearDown(self) -> None:
        self.controller.close()

    def test_request_and_finalize_selection_creates_first_group(self) -> None:
        requested = self.controller.request_new_selection()
        completed = self.controller.finalize_selection(ScreenRegion(100, 120, 360, 100))

        self.assertTrue(requested)
        self.assertTrue(completed)
        self.assertEqual(self.runtime_store.active_group_ids(), [1])
        self.assertEqual(self.context.active_group_count, 1)
        self.assertEqual(self.controller.selection_box_count, 1)
        self.assertEqual(self.controller.translation_window_count, 1)
        self.assertEqual(self.context.status_message, "已创建第 1 组选择框。")

    def test_normal_mode_keeps_only_border_inside_capture_region(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))

        box = self.controller.get_selection_box(1)
        translation_window = self.controller.get_translation_window(1)

        self.assertIsNotNone(box)
        self.assertIsNotNone(translation_window)
        assert box is not None
        assert translation_window is not None
        self.assertFalse(box.toolbar_visible)
        self.assertFalse(box.size_badge_visible)
        self.assertTrue(box.group_badge.isHidden())
        self.assertEqual(box.outline_width, 2)
        self.assertEqual(box.group_badge.text(), "1")
        self.assertEqual(translation_window.width(), 260)
        self.assertTrue(translation_window.language_pair_label.isHidden())
        self.assertFalse(translation_window.translation_label.isHidden())

    def test_selection_toolbar_exposes_ocr_view_button(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))

        box = self.controller.get_selection_box(1)

        self.assertIsNotNone(box)
        assert box is not None
        self.assertEqual(box.toolbar_panel.ocr_button.text(), "OCR")
        self.assertEqual(box.toolbar_panel.dock_combo.currentData(), "bottom")
        self.assertEqual(box.toolbar_panel.feedback_button.text(), "翻译有误")

    def test_edit_mode_hides_inner_badges_and_keeps_toolbar_above_box(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))
        self.controller.toggle_edit_mode()

        box = self.controller.get_selection_box(1)
        translation_window = self.controller.get_translation_window(1)

        self.assertIsNotNone(box)
        self.assertIsNotNone(translation_window)
        assert box is not None
        assert translation_window is not None
        self.assertTrue(box.toolbar_visible)
        self.assertTrue(box.group_badge.isHidden())
        self.assertFalse(box.size_badge_visible)
        self.assertEqual(box.size_badge.text(), "260 x 90")
        self.assertEqual(box.outline_width, 4)
        self.assertFalse(translation_window.language_pair_label.isHidden())
        self.assertFalse(translation_window.dock_controls_visible)
        self.assertLessEqual(box.toolbar_panel.frameGeometry().bottom(), box.frameGeometry().top())
        self.assertLess(box.toolbar_panel.frameGeometry().bottom(), translation_window.frameGeometry().top())

    def test_small_box_still_shows_floating_toolbar(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 70, 36))
        self.controller.toggle_edit_mode()

        box = self.controller.get_selection_box(1)

        self.assertIsNotNone(box)
        assert box is not None
        self.assertTrue(box.toolbar_visible)
        self.assertGreater(box.toolbar_panel.width(), box.width())

    def test_pause_toggle_updates_group_state(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))

        paused = self.controller.toggle_group_pause(1)
        box = self.controller.get_selection_box(1)

        self.assertTrue(paused)
        self.assertTrue(self.runtime_store.configs[1].paused)
        assert box is not None
        self.assertEqual(box.toolbar_panel.pause_button.text(), "继续")

    def test_translation_dock_is_controlled_from_selection_toolbar(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))

        changed = self.controller.set_translation_dock(1, "top")
        box = self.controller.get_selection_box(1)

        self.assertTrue(changed)
        self.assertEqual(self.runtime_store.configs[1].translation_dock, "top")
        self.assertIsNotNone(box)
        assert box is not None
        self.assertEqual(box.toolbar_panel.dock_combo.currentData(), "top")

    def test_mark_translation_feedback_saves_latest_ocr_and_translation(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))
        fake_store = FakeFeedbackStore()
        self.controller._feedback_store = fake_store
        self.runtime_store.runtime_states[1].latest_ocr_text = "The gaokao starts soon"
        self.runtime_store.runtime_states[1].latest_translation_text = "高中考试很快开始"
        self.runtime_store.configs[1].source_language = "English"
        self.runtime_store.configs[1].target_language = "中文"

        saved = self.controller.mark_translation_feedback(1)

        self.assertTrue(saved)
        self.assertEqual(len(fake_store.calls), 1)
        self.assertEqual(fake_store.calls[0]["ocr_text"], "The gaokao starts soon")
        self.assertEqual(fake_store.calls[0]["translation_text"], "高中考试很快开始")
        self.assertEqual(fake_store.calls[0]["source_language"], "English")
        self.assertEqual(fake_store.calls[0]["target_language"], "中文")

    def test_move_group_updates_runtime_region_and_translation_window(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))

        moved = self.controller.move_group(1, 180, 220)
        region = self.runtime_store.regions[1]
        translation_window = self.controller.get_translation_window(1)

        self.assertTrue(moved)
        self.assertEqual((region.x, region.y), (180, 220))
        self.assertIsNotNone(translation_window)
        assert translation_window is not None
        self.assertEqual(self.context.status_message, "已移动第 1 组。")
        self.assertEqual(translation_window.windowTitle(), "Translation 1")

    def test_resize_group_resets_translation_dedupe_cache(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))
        fake_translation = FakeTranslationService()
        self.controller._translation_service = fake_translation

        resized = self.controller.resize_group(1, 40, 50, 360, 110)

        self.assertTrue(resized)
        self.assertEqual(fake_translation.reset_calls, [1])
        self.assertEqual(
            fake_translation.invalidations[-1],
            {"group_id": 1, "reason": "selection resized"},
        )

    def test_move_group_resets_translation_dedupe_cache(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))
        fake_translation = FakeTranslationService()
        self.controller._translation_service = fake_translation

        moved = self.controller.move_group(1, 180, 220)

        self.assertTrue(moved)
        self.assertEqual(fake_translation.reset_calls, [1])
        self.assertEqual(
            fake_translation.invalidations[-1],
            {"group_id": 1, "reason": "selection moved"},
        )

    def test_delete_group_removes_box_and_linked_translation_window(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))
        self.controller.toggle_ocr_window(1)

        deleted = self.controller.delete_group(1)

        self.assertTrue(deleted)
        self.assertEqual(self.runtime_store.active_group_ids(), [])
        self.assertEqual(self.controller.selection_box_count, 0)
        self.assertEqual(self.controller.translation_window_count, 0)
        self.assertEqual(self.controller.ocr_window_count, 0)
        self.assertEqual(self.context.status_message, "已删除第 1 组。")

    def test_stale_ocr_worker_after_delete_cannot_update_new_group(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))
        old_version = self.controller._group_version(1)
        self.controller.delete_group(1)
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(80, 100, 260, 90))
        self.controller._ocr_engine = FakeOcrEngine("old deleted text")
        self.controller._ocr_stable_seconds = 0.0
        fake_translation = FakeTranslationService()
        self.controller._translation_service = fake_translation

        self.controller._process_frame(
            1,
            CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x00" * 16),
            old_version,
        )

        self.assertEqual(self.runtime_store.runtime_states[1].latest_ocr_text, "")
        self.assertEqual(fake_translation.requests, [])

    def test_request_new_selection_rejects_fourth_group(self) -> None:
        for group_id in (1, 2, 3):
            self.runtime_store.save_region(group_id, ScreenRegion(0, group_id * 100, 240, 80))

        requested = self.controller.request_new_selection()

        self.assertFalse(requested)
        self.assertEqual(self.context.status_message, "最多支持 3 个选择框。")

    def test_poll_continues_capture_in_edit_mode_without_inner_badges(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))
        self.controller.toggle_edit_mode()

        box = self.controller.get_selection_box(1)
        assert box is not None
        self.assertTrue(box.group_badge.isHidden())
        self.assertTrue(box.size_badge.isHidden())
        self.assertTrue(box.toolbar_visible)

        self.controller._tick()

        self.assertEqual(self.capture_service.capture_calls, 1)
        self.assertTrue(box.group_badge.isHidden())
        self.assertTrue(box.size_badge.isHidden())
        self.assertTrue(box.toolbar_visible)

    def test_edit_mode_forces_next_ocr_even_when_region_is_static(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))
        self.change_detector.reset_calls.clear()

        self.controller.toggle_edit_mode()

        self.assertEqual(self.change_detector.reset_calls, [1])

    def test_process_frame_sends_clean_ocr_text_to_translation(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))
        self.controller.toggle_ocr_window(1)
        self.controller._ocr_stable_seconds = 0.0
        self.controller._ocr_engine = FakeOcrEngine("260 x 90\n重选 暂停 删除\nHello   world")
        fake_translation = FakeTranslationService()
        self.controller._translation_service = fake_translation

        self.controller._process_frame(
            1,
            CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x00" * 16),
        )
        self.controller._process_frame(
            1,
            CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x00" * 16),
        )

        self.assertEqual(self.runtime_store.runtime_states[1].latest_ocr_text, "Hello world")
        self.assertEqual(fake_translation.requests[0]["ocr_text"], "Hello world")
        ocr_window = self.controller.get_ocr_window(1)
        self.assertIsNotNone(ocr_window)
        assert ocr_window is not None
        self.assertEqual(ocr_window.ocr_label.text(), "Hello world")

    def test_toggle_ocr_window_shows_latest_text(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))
        self.runtime_store.runtime_states[1].latest_ocr_text = "Hello OCR"

        opened = self.controller.toggle_ocr_window(1)

        self.assertTrue(opened)
        self.assertEqual(self.controller.ocr_window_count, 1)
        ocr_window = self.controller.get_ocr_window(1)
        self.assertIsNotNone(ocr_window)
        assert ocr_window is not None
        self.assertEqual(ocr_window.ocr_label.text(), "Hello OCR")

    def test_process_frame_waits_for_stable_ocr_before_translation(self) -> None:
        self.context.default_source_language = "中文"
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))
        clock = [0.0]
        self.controller._stable_clock = lambda: clock[0]
        self.controller._ocr_stable_seconds = 2.0
        self.controller._ocr_engine = FakeOcrEngine(
            "如果要优化数据库的性能的话,\n我们来探讨一下添加索引的方案吧"
        )
        fake_translation = FakeTranslationService()
        self.controller._translation_service = fake_translation

        frame = CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x00" * 16)
        self.controller._process_frame(1, frame)
        self.assertEqual(fake_translation.requests, [])

        clock[0] = 1.9
        self.controller._process_frame(1, frame)
        self.assertEqual(fake_translation.requests, [])

        clock[0] = 2.1
        self.controller._process_frame(1, frame)

        self.assertEqual(len(fake_translation.requests), 1)
        self.assertEqual(
            fake_translation.requests[0]["ocr_text"],
            "如果要优化数据库的性能的话,\n我们来探讨一下添加索引的方案吧",
        )

    def test_process_frame_restarts_stable_window_when_ocr_changes(self) -> None:
        self.context.default_source_language = "中文"
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))
        clock = [0.0]
        self.controller._stable_clock = lambda: clock[0]
        self.controller._ocr_stable_seconds = 2.0
        self.controller._ocr_engine = SequenceOcrEngine(
            [
                "第一段文本",
                "第二段文本",
                "第二段文本",
                "第二段文本",
            ]
        )
        fake_translation = FakeTranslationService()
        self.controller._translation_service = fake_translation
        frame = CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x00" * 16)

        self.controller._process_frame(1, frame)
        clock[0] = 2.0
        self.controller._process_frame(1, frame)
        clock[0] = 3.9
        self.controller._process_frame(1, frame)
        self.assertEqual(fake_translation.requests, [])

        clock[0] = 4.1
        self.controller._process_frame(1, frame)

        self.assertEqual(len(fake_translation.requests), 1)
        self.assertEqual(fake_translation.requests[0]["ocr_text"], "第二段文本")
        self.assertTrue(fake_translation.invalidations)

    def test_process_frame_skips_duplicate_after_ocr_normalization(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))
        self.controller._last_logged_ocr[1] = "Hello world"
        self.controller._ocr_stable_seconds = 0.0
        self.controller._ocr_engine = FakeOcrEngine("260 x 90\nHello   world")
        fake_translation = FakeTranslationService()
        self.controller._translation_service = fake_translation

        self.controller._process_frame(
            1,
            CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x00" * 16),
        )

        self.assertEqual(fake_translation.requests, [])

    def test_tick_does_not_queue_same_group_while_ocr_is_processing(self) -> None:
        self.change_detector.changed = True
        self.controller._poll_executor.shutdown(wait=False)
        executor = FakeDeferredExecutor()
        self.controller._poll_executor = executor
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))

        self.controller._tick()
        self.controller._tick()

        self.assertEqual(len(executor.submitted), 1)
        self.assertEqual(self.capture_service.capture_calls, 1)

    def test_tick_confirms_pending_ocr_even_when_region_is_static(self) -> None:
        self.change_detector.changed = True
        self.controller._poll_executor.shutdown(wait=False)
        executor = FakeDeferredExecutor()
        self.controller._poll_executor = executor
        self.controller._ocr_engine = FakeOcrEngine("Hello world")
        self.controller._ocr_stable_seconds = 0.0
        fake_translation = FakeTranslationService()
        self.controller._translation_service = fake_translation
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))

        self.controller._tick()
        executor.run_next()

        self.assertEqual(fake_translation.requests, [])

        self.change_detector.changed = False
        self.controller._tick()

        self.assertEqual(len(executor.submitted), 1)
        executor.run_next()
        self.assertEqual(fake_translation.requests[0]["ocr_text"], "Hello world")

    def test_group_can_process_again_after_ocr_worker_finishes(self) -> None:
        self.change_detector.changed = True
        self.controller._poll_executor.shutdown(wait=False)
        executor = FakeDeferredExecutor()
        self.controller._poll_executor = executor
        self.controller._ocr_engine = FakeOcrEngine("Hello world")
        self.controller._translation_service = FakeTranslationService()
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 120, 260, 90))

        self.controller._tick()
        executor.run_next()
        self.controller._tick()

        self.assertEqual(len(executor.submitted), 1)
        self.assertEqual(self.capture_service.capture_calls, 2)
