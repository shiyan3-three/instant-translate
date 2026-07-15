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

        # Empty text uses content floor, not selection width.
        self.assertEqual(geometry.width(), 160)
        self.assertEqual(geometry.height(), 72)
        self.assertEqual(geometry.y(), 160 + 96 + 12)
        self.assertGreaterEqual(geometry.x(), 0)

    def test_compute_geometry_falls_back_above_when_bottom_overflows(self) -> None:
        region = ScreenRegion(120, 980, 320, 96)
        screen = QRect(0, 0, 1920, 1080)

        geometry = TranslationWindowWidget.compute_geometry(region, screen, "bottom")

        self.assertLess(geometry.y(), region.y)

    def test_compute_geometry_keeps_toolbar_usable_for_tiny_selection(self) -> None:
        region = ScreenRegion(120, 160, 70, 36)
        screen = QRect(0, 0, 1920, 1080)

        geometry = TranslationWindowWidget.compute_geometry(region, screen, "bottom")

        self.assertEqual(geometry.width(), 160)

    def test_compute_geometry_expands_width_for_long_translation_text(self) -> None:
        region = ScreenRegion(120, 160, 320, 96)
        screen = QRect(0, 0, 1920, 1080)
        long_text = (
            "この  [いんたあふぇえす]  は  すでに  せいこう  を  "
            "へんきゃく  している  が  ぺえじ  は  まだ  ろおどちゅう  "
            "を  ひょうじ  している"
        )

        geometry = TranslationWindowWidget.compute_geometry(region, screen, "bottom", long_text)

        from app.overlay.translation_window import max_window_width, max_window_height

        self.assertGreater(geometry.width(), region.width)
        self.assertLessEqual(geometry.width(), max_window_width())

    def test_compute_geometry_never_exceeds_hard_caps(self) -> None:
        from app.overlay.translation_window import max_window_width, max_window_height

        region = ScreenRegion(10, 10, 100, 40)
        screen = QRect(0, 0, 3840, 2160)
        huge = "漢字" * 2000

        geometry = TranslationWindowWidget.compute_geometry(region, screen, "bottom", huge)

        self.assertLessEqual(geometry.width(), max_window_width())
        self.assertLessEqual(geometry.height(), max_window_height())

    def test_compute_geometry_expands_height_for_very_long_translation_text(self) -> None:
        from app.overlay.translation_window import max_window_height

        region = ScreenRegion(120, 160, 320, 96)
        screen = QRect(0, 0, 1920, 1080)
        very_long_text = "ながい  ほんやく  " * 80

        geometry = TranslationWindowWidget.compute_geometry(region, screen, "bottom", very_long_text)

        self.assertGreater(geometry.height(), 72)
        self.assertLessEqual(geometry.height(), max_window_height())

    def test_compute_geometry_sizes_short_text_to_content_not_region(self) -> None:
        from app.overlay.translation_window import max_window_width, max_window_height

        region = ScreenRegion(120, 160, 900, 96)
        screen = QRect(0, 0, 1920, 1080)

        geometry = TranslationWindowWidget.compute_geometry(region, screen, "bottom", "ありがとう")

        self.assertLess(geometry.width(), region.width)
        self.assertGreaterEqual(geometry.width(), 160)
        self.assertLessEqual(geometry.width(), max_window_width())
        self.assertLessEqual(geometry.height(), max_window_height())

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
        self.assertEqual(widget.language_pair_label.text(), "English → \u4e2d\u6587")
        self.assertIn("\u7b49\u5f85\u76d1\u6d4b", widget.translation_label.text())
        self.assertTrue(widget.language_pair_label.isHidden())
        self.assertFalse(widget.dock_controls_visible)
        self.assertFalse(widget.translation_label.isHidden())
        self.assertIn("QLabel#translationBody", widget.styleSheet())
        self.assertIn("font-size: 14px", widget.styleSheet())

    def test_edit_mode_reveals_language_pair_and_mock_corner_toolbar(self) -> None:
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

        self.assertFalse(widget.language_pair_label.isHidden())
        self.assertTrue(widget.dock_controls_visible)
        self.assertEqual(widget.collapse_button.text(), "›")

        widget.toggle_corner_collapsed()

        self.assertTrue(widget.corner_collapsed)
        self.assertTrue(widget.corner_tools_widget.isHidden())
        self.assertEqual(widget.collapse_button.text(), "‹")

    def test_body_immersive_keeps_corner_bar_available_and_text_position_stable(self) -> None:
        widget = TranslationWindowWidget(
            TranslationWindowModel(
                group_id=1,
                x=100,
                y=200,
                width=320,
                height=132,
                source_language="中文",
                target_language="日本語",
                text="ほんやく",
            )
        )
        self.addCleanup(widget.close)
        widget.show()
        widget.apply_edit_mode(True)
        self.app.processEvents()
        before = widget.body_text_global_top_left()

        widget.set_body_immersive(True)
        self.app.processEvents()

        self.assertEqual(widget.body_text_global_top_left(), before)
        self.assertTrue(widget.dock_controls_visible)
        self.assertTrue(widget.body_immersive_button.isChecked())
        self.assertEqual(widget.body_immersive_button.text(), "◉")
        self.assertEqual(widget.dock_up_button.size().width(), 28)
        self.assertEqual(widget.collapse_button.size().width(), 28)

        widget.set_body_immersive(False)
        self.app.processEvents()
        self.assertEqual(widget.body_immersive_button.text(), "⊘")
        self.assertEqual(widget.group_badge.styleSheet(), "")
        self.assertEqual(widget.language_pair_label.styleSheet(), "")

        widget.set_group_immersive(True)
        self.app.processEvents()
        self.assertFalse(widget.dock_controls_visible)

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
