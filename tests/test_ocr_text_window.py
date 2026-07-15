"""Tests for OCR text viewer overlay."""

from __future__ import annotations

import unittest

from PySide6.QtCore import QRect

from app.overlay.ocr_text_window import OcrTextWindowModel, OcrTextWindowWidget
from app.state.group_state import ScreenRegion
from tests.test_support import ensure_qapplication


class OcrTextWindowWidgetTests(unittest.TestCase):
    """Verify the OCR viewer shows and copies the text sent to translation."""

    def setUp(self) -> None:
        self.app = ensure_qapplication()
        self.app.clipboard().clear()

    def test_compute_geometry_prefers_above_when_space_is_available(self) -> None:
        region = ScreenRegion(120, 200, 320, 96)
        screen = QRect(0, 0, 1920, 1080)

        geometry = OcrTextWindowWidget.compute_geometry(region, screen)

        from app.overlay.ocr_text_window import max_window_height

        self.assertEqual(geometry.width(), 160)
        self.assertLess(geometry.y(), region.y)
        self.assertLessEqual(geometry.height(), max_window_height())

    def test_compute_geometry_sizes_to_text_not_region_width(self) -> None:
        from app.overlay.ocr_text_window import max_window_width, max_window_height

        region = ScreenRegion(120, 200, 900, 96)
        screen = QRect(0, 0, 1920, 1080)

        geometry = OcrTextWindowWidget.compute_geometry(region, screen, "短句OCR")

        self.assertLess(geometry.width(), region.width)
        self.assertGreaterEqual(geometry.width(), 160)
        self.assertLessEqual(geometry.width(), max_window_width())
        self.assertLessEqual(geometry.height(), max_window_height())

    def test_compute_geometry_caps_long_ocr_text(self) -> None:
        from app.overlay.ocr_text_window import max_window_width, max_window_height

        region = ScreenRegion(120, 200, 200, 40)
        screen = QRect(0, 0, 1920, 1080)
        long_text = "这是一段很长的识别文本，用于验证宽度高度上限。" * 20

        geometry = OcrTextWindowWidget.compute_geometry(region, screen, long_text)

        self.assertLessEqual(geometry.width(), max_window_width())
        self.assertLessEqual(geometry.height(), max_window_height())
        self.assertGreater(geometry.width(), 200)

    def test_widget_shows_group_text_and_copies_to_clipboard(self) -> None:
        model = OcrTextWindowModel(
            group_id=1,
            x=100,
            y=200,
            width=320,
            height=116,
            text="Hello OCR",
            accent_color="#2F80ED",
        )
        widget = OcrTextWindowWidget(model)

        self.assertEqual(widget.windowTitle(), "OCR 1")
        self.assertEqual(widget.group_badge.text(), "1")
        self.assertEqual(widget.ocr_label.text(), "Hello OCR")

        widget.copy_button.click()

        self.assertEqual(self.app.clipboard().text(), "Hello OCR")

    def test_apply_model_updates_text_and_geometry(self) -> None:
        widget = OcrTextWindowWidget(
            OcrTextWindowModel(
                group_id=1,
                x=100,
                y=200,
                width=320,
                height=116,
                text="Waiting",
            )
        )

        widget.apply_model(
            OcrTextWindowModel(
                group_id=1,
                x=120,
                y=220,
                width=360,
                height=116,
                text="Updated OCR",
            )
        )

        self.assertEqual(widget.geometry().getRect(), (120, 220, 360, 116))
        self.assertEqual(widget.ocr_label.text(), "Updated OCR")

    def test_edit_mode_shows_full_mock_corner_toolbar_and_collapse(self) -> None:
        widget = OcrTextWindowWidget(
            OcrTextWindowModel(
                group_id=1,
                x=100,
                y=200,
                width=320,
                height=116,
                text="OCR",
            )
        )
        self.addCleanup(widget.close)

        widget.apply_edit_mode(True)

        self.assertTrue(widget.corner_controls_visible)
        self.assertEqual(widget.pause_button.text(), "")
        self.assertFalse(widget.pause_button.icon().isNull())
        self.assertEqual(widget.refresh_button.text(), "↻")
        self.assertEqual(widget.copy_button.text(), "⧉")
        self.assertEqual(widget.body_immersive_button.text(), "⊘")
        self.assertEqual(widget.pause_button.size().width(), 28)
        self.assertEqual(widget.pause_button.iconSize().width(), 18)
        self.assertEqual(widget.collapse_button.size().width(), 28)

        widget.toggle_corner_collapsed()

        self.assertTrue(widget.corner_collapsed)
        self.assertTrue(widget.corner_tools_widget.isHidden())
        self.assertEqual(widget.collapse_button.text(), "‹")

    def test_ocr_immersive_keeps_text_position_and_can_be_undone_from_corner(self) -> None:
        widget = OcrTextWindowWidget(
            OcrTextWindowModel(
                group_id=1,
                x=100,
                y=200,
                width=320,
                height=116,
                text="OCR body",
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
        self.assertTrue(widget.corner_controls_visible)
        self.assertEqual(widget.body_immersive_button.text(), "◉")

        widget.set_body_immersive(False)
        self.app.processEvents()
        self.assertEqual(widget.body_immersive_button.text(), "⊘")
        self.assertEqual(widget.group_badge.styleSheet(), "")
        self.assertEqual(widget.title_label.styleSheet(), "")

        widget.set_group_immersive(True)
        self.app.processEvents()
        self.assertFalse(widget.corner_controls_visible)


if __name__ == "__main__":
    unittest.main()
