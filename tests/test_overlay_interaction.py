"""Tests for overlay input pass-through behavior."""

from __future__ import annotations

import unittest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent

from app.overlay.selection_box import SelectionBoxModel, SelectionBoxWidget
from app.overlay.translation_window import TranslationWindowModel, TranslationWindowWidget
from app.overlay.window_interaction import is_window_click_through
from tests.test_support import ensure_qapplication


class OverlayInteractionTests(unittest.TestCase):
    """Verify edit mode receives mouse input on overlay bodies."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_selection_box_toggles_input_passthrough_with_edit_mode(self) -> None:
        widget = SelectionBoxWidget(
            SelectionBoxModel(group_id=1, x=100, y=100, width=280, height=90)
        )
        widget.show()
        self.app.processEvents()

        widget.apply_edit_mode(False)
        self.app.processEvents()
        self.assertTrue(widget.input_passthrough_enabled)
        self.assertTrue(is_window_click_through(widget))

        widget.apply_edit_mode(True)
        self.app.processEvents()
        self.assertFalse(widget.input_passthrough_enabled)
        self.assertFalse(is_window_click_through(widget))

        widget.close()

    def test_selection_box_body_drag_uses_interior_hit_area_in_edit_mode(self) -> None:
        widget = SelectionBoxWidget(
            SelectionBoxModel(group_id=1, x=100, y=100, width=280, height=90)
        )
        widget.show()
        widget.apply_edit_mode(True)
        self.app.processEvents()

        self.assertTrue(widget.body_hit_area_enabled)
        self.assertFalse(widget.mask().isEmpty())
        self.assertEqual(widget.mask().boundingRect(), widget.rect())

        press_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPoint(120, 45),
            QPoint(220, 145),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        move_event = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPoint(140, 65),
            QPoint(240, 165),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPoint(140, 65),
            QPoint(240, 165),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )

        widget.mousePressEvent(press_event)
        widget.mouseMoveEvent(move_event)
        widget.mouseReleaseEvent(release_event)

        self.assertEqual((widget.x(), widget.y()), (120, 120))
        widget.close()

    def test_translation_window_toggles_input_passthrough_with_edit_mode(self) -> None:
        widget = TranslationWindowWidget(
            TranslationWindowModel(
                group_id=1,
                x=100,
                y=220,
                width=280,
                height=132,
                source_language="English",
                target_language="\u4e2d\u6587",
                text="\u7b49\u5f85\u76d1\u6d4b\u753b\u9762\u53d8\u5316...",
            )
        )
        widget.show()
        self.app.processEvents()

        widget.apply_edit_mode(False)
        self.app.processEvents()
        self.assertTrue(widget.input_passthrough_enabled)
        self.assertTrue(is_window_click_through(widget))

        widget.apply_edit_mode(True)
        self.app.processEvents()
        self.assertFalse(widget.input_passthrough_enabled)
        self.assertFalse(is_window_click_through(widget))

        widget.close()
