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

        self.assertEqual(geometry.width(), 320)
        self.assertLess(geometry.y(), region.y)

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


if __name__ == "__main__":
    unittest.main()
