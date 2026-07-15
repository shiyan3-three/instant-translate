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

    def test_normalize_removes_app_toolbar_language_and_feedback_noise(self) -> None:
        raw = (
            "中文 日本語 下 重选 OCR 翻译有误 暂停 删除\n"
            "今天下午两点半，我准备把这份合同检查完。"
        )

        clean = normalize_ocr_text(raw)

        self.assertEqual(clean, "今天下午两点半,我准备把这份合同检查完。")

    def test_normalize_removes_search_box_and_language_pair_noise(self) -> None:
        raw = "Q搜索\n中文 => 日本語\n明明没有提交申请，系统却显示已经审核通过。"

        clean = normalize_ocr_text(raw)

        self.assertEqual(clean, "明明没有提交申请,系统却显示已经审核通过。")

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

    def test_normalize_removes_social_metric_noise_lines(self) -> None:
        raw = "关注6.7W\n点赞45.6W\n古力娜扎丁字牛仔裤, 这穿搭太酷了!"

        clean = normalize_ocr_text(raw)

        self.assertEqual(clean, "古力娜扎丁字牛仔裤, 这穿搭太酷了!")

    def test_normalize_removes_only_standalone_structural_ui_noise(self) -> None:
        raw = (
            "(cache)\n"
            "UV-7 (resource)\n"
            "(timing)\n"
            "Please check cache timing settings.\n"
            "Fix Bug 123."
        )

        clean = normalize_ocr_text(raw)

        self.assertEqual(clean, "Please check cache timing settings.\nFix Bug 123.")

    def test_structural_noise_filter_does_not_remove_semantic_negation_or_numbers(self) -> None:
        raw = (
            "Do not enable cache.\nRetry count is 2.\nThe timing changed.\n"
            "初始化()\n翻译()\nテスト()\nreset()"
        )

        self.assertEqual(normalize_ocr_text(raw), raw)

    def test_suspicious_detection_rejects_dimension_only_noise(self) -> None:
        self.assertTrue(is_suspicious_ocr_text("595 x 3", "中文"))

    def test_suspicious_detection_rejects_single_group_number_noise(self) -> None:
        self.assertTrue(is_suspicious_ocr_text("1", "中文"))

    def test_suspicious_detection_rejects_source_language_script_mismatch(self) -> None:
        self.assertTrue(is_suspicious_ocr_text("の、\n初加索引的", "中文"))

    def test_stability_detection_accepts_same_clean_text(self) -> None:
        previous = "如果要优化数据库的性能的话,\n我们来探讨一下添加索引的方案吧"
        current = "如果要优化数据库的性能的话,\n我们来探讨一下添加索引的方案吧"

        self.assertTrue(is_stable_ocr_text(current, previous))


if __name__ == "__main__":
    unittest.main()
