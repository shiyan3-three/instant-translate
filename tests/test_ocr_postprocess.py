"""Tests for OCR text cleanup before translation."""

from __future__ import annotations

import unittest

from app.ocr.postprocess import (
    is_duplicate_ocr_text,
    is_suspicious_ocr_text,
    is_stable_ocr_text,
    normalize_ocr_text,
)


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

    def test_normalize_removes_embedded_overlay_size_badge(self) -> None:
        raw = "如果要优化数据库的性能的话,\n我们来探讨一下添加索引1的595x37"

        clean = normalize_ocr_text(raw)

        self.assertNotIn("595", clean)
        self.assertNotIn("37", clean)

    def test_suspicious_detection_rejects_dimension_only_noise(self) -> None:
        self.assertTrue(is_suspicious_ocr_text("595 x 3", "中文"))

    def test_suspicious_detection_rejects_source_language_script_mismatch(self) -> None:
        self.assertTrue(is_suspicious_ocr_text("の、\n初加索引的", "中文"))

    def test_stability_detection_accepts_same_clean_text(self) -> None:
        previous = "如果要优化数据库的性能的话,\n我们来探讨一下添加索引的方案吧"
        current = "如果要优化数据库的性能的话,\n我们来探讨一下添加索引的方案吧"

        self.assertTrue(is_stable_ocr_text(current, previous))


if __name__ == "__main__":
    unittest.main()
