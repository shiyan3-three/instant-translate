"""Tests for the fixed-region selection workflow."""

from __future__ import annotations

import unittest

from app.app_context import ApplicationContext
from app.overlay.edit_mode_controller import EditModeController
from app.overlay.selection_manager import SelectionWorkflowController
from app.state.group_state import ScreenRegion
from app.state.runtime_store import RuntimeStore
from tests.test_support import ensure_qapplication


class SelectionWorkflowControllerTests(unittest.TestCase):
    """Verify selection creation flow, persistent edit tools, and linked overlays."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()
        self.context = ApplicationContext()
        self.runtime_store = RuntimeStore()
        self.controller = SelectionWorkflowController(
            app=self.app,
            context=self.context,
            runtime_store=self.runtime_store,
            edit_mode_controller=EditModeController(),
            on_state_changed=lambda: None,
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

    def test_normal_mode_keeps_only_border_and_number(self) -> None:
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
        self.assertEqual(box.outline_width, 2)
        self.assertEqual(box.group_badge.text(), "1")
        self.assertEqual(translation_window.width(), 260)
        self.assertTrue(translation_window.language_pair_label.isHidden())
        self.assertFalse(translation_window.translation_label.isHidden())

    def test_edit_mode_keeps_toolbar_and_size_visible_and_toolbar_stays_above_box(self) -> None:
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
        self.assertTrue(box.size_badge_visible)
        self.assertEqual(box.size_badge.text(), "260 x 90")
        self.assertEqual(box.outline_width, 4)
        self.assertFalse(translation_window.language_pair_label.isHidden())
        self.assertTrue(translation_window.dock_controls_visible)
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

    def test_delete_group_removes_box_and_linked_translation_window(self) -> None:
        self.controller.request_new_selection()
        self.controller.finalize_selection(ScreenRegion(40, 50, 260, 90))

        deleted = self.controller.delete_group(1)

        self.assertTrue(deleted)
        self.assertEqual(self.runtime_store.active_group_ids(), [])
        self.assertEqual(self.controller.selection_box_count, 0)
        self.assertEqual(self.controller.translation_window_count, 0)
        self.assertEqual(self.context.status_message, "已删除第 1 组。")

    def test_request_new_selection_rejects_fourth_group(self) -> None:
        for group_id in (1, 2, 3):
            self.runtime_store.save_region(group_id, ScreenRegion(0, group_id * 100, 240, 80))

        requested = self.controller.request_new_selection()

        self.assertFalse(requested)
        self.assertEqual(self.context.status_message, "最多支持 3 个选择框。")
