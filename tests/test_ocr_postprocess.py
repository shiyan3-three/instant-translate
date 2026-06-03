"""Tests for OCR text cleanup before translation."""

from __future__ import annotations

import unittest

from app.ocr.postprocess import is_duplicate_ocr_text, normalize_ocr_text


class OcrPostprocessTests(unittest.TestCase):
    """Verify OCR noise cleanup and duplicate detection."""

    def test_normalize_removes_overlay_size_and_toolbar_noise(self) -> None:
        raw = "\n260 x 90\n重选 暂停 删除\n  Hello   world  \n"

        clean = normalize_ocr_text(raw)

        self.assertEqual(clean, "Hello world")

    def test_normalize_removes_unicode_multiplication_size_badge(self) -> None:
        raw = "1023 × 86\nこんにちは 世界"

        clean = normalize_ocr_text(raw)

        self.assertEqual(clean, "こんにちは 世界")

    def test_duplicate_detection_ignores_trivial_ocr_noise(self) -> None:
        previous = "260 x 90\nHello   world"
        current = "Hello world"

        self.assertTrue(is_duplicate_ocr_text(current, previous))

    def test_duplicate_detection_keeps_real_text_changes(self) -> None:
        previous = "Hello world"
        current = "Hello new world"

        self.assertFalse(is_duplicate_ocr_text(current, previous))


if __name__ == "__main__":
    unittest.main()
