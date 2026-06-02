"""Tests for translation-window skeleton behavior."""

from __future__ import annotations

import unittest

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QMouseEvent

from app.overlay.translation_window import TranslationWindowModel, TranslationWindowWidget
from app.state.group_state import ScreenRegion
from tests.test_support import ensure_qapplication


class TranslationWindowWidgetTests(unittest.TestCase):
    """Verify translation window placement and display rules."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()

    def test_compute_geometry_prefers_bottom_when_space_is_available(self) -> None:
        region = ScreenRegion(120, 160, 320, 96)
        screen = QRect(0, 0, 1920, 1080)

        geometry = TranslationWindowWidget.compute_geometry(region, screen, "bottom")

        self.assertEqual(geometry.width(), 320)
        self.assertEqual(geometry.height(), 132)
        self.assertEqual(geometry.y(), 268)
        self.assertGreaterEqual(geometry.x(), 0)

    def test_compute_geometry_falls_back_above_when_bottom_overflows(self) -> None:
        region = ScreenRegion(120, 980, 320, 96)
        screen = QRect(0, 0, 1920, 1080)

        geometry = TranslationWindowWidget.compute_geometry(region, screen, "bottom")

        self.assertLess(geometry.y(), region.y)

    def test_widget_shows_group_and_translation_text_in_normal_mode(self) -> None:
        model = TranslationWindowModel(
            group_id=2,
            x=100,
            y=200,
            width=320,
            height=132,
            source_language="English",
            target_language="\u4e2d\u6587",
            text="\u7b49\u5f85\u76d1\u6d4b\u753b\u9762\u53d8\u5316...",
        )

        widget = TranslationWindowWidget(model)

        self.assertEqual(widget.windowTitle(), "Translation 2")
        self.assertEqual(widget.group_badge.text(), "2")
        self.assertEqual(widget.language_pair_label.text(), "English => \u4e2d\u6587")
        self.assertIn("\u7b49\u5f85\u76d1\u6d4b", widget.translation_label.text())
        self.assertTrue(widget.language_pair_label.isHidden())
        self.assertFalse(widget.dock_controls_visible)
        self.assertFalse(widget.translation_label.isHidden())

    def test_edit_mode_reveals_language_pair_and_dock_buttons(self) -> None:
        model = TranslationWindowModel(
            group_id=1,
            x=100,
            y=200,
            width=320,
            height=132,
            source_language="English",
            target_language="\u4e2d\u6587",
            text="\u7b49\u5f85\u76d1\u6d4b\u753b\u9762\u53d8\u5316...",
        )
        widget = TranslationWindowWidget(model)

        widget.apply_edit_mode(True)

        self.assertTrue(widget.dock_controls_visible)
        self.assertFalse(widget.language_pair_label.isHidden())
        self.assertEqual(widget.dock_up_button.text(), "\u4e0a")
        self.assertEqual(widget.dock_right_button.text(), "\u53f3")

    def test_mouse_drag_updates_window_position(self) -> None:
        model = TranslationWindowModel(
            group_id=1,
            x=100,
            y=200,
            width=320,
            height=132,
            source_language="English",
            target_language="\u4e2d\u6587",
            text="\u7b49\u5f85\u76d1\u6d4b\u753b\u9762\u53d8\u5316...",
        )
        widget = TranslationWindowWidget(model)
        widget.apply_edit_mode(True)

        press_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPoint(30, 30),
            QPoint(130, 230),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        move_event = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPoint(50, 40),
            QPoint(150, 240),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPoint(50, 40),
            QPoint(150, 240),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )

        widget.mousePressEvent(press_event)
        widget.mouseMoveEvent(move_event)
        widget.mouseReleaseEvent(release_event)

        self.assertEqual((widget.x(), widget.y()), (120, 210))
