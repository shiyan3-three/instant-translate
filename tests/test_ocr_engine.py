from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app.ocr.engine import OcrEngine, OcrResult
from app.ocr.preprocess import ImagePreprocessor


class OcrWarmupTests(unittest.TestCase):
    def test_paddle_273_constructor_uses_only_2x_arguments(self) -> None:
        engine = OcrEngine()
        paddle = type("Paddle", (), {"PaddleOCR": Mock(return_value=object())})

        with patch.dict("sys.modules", {"paddleocr": paddle}):
            engine._ensure_engine("English")

        kwargs = paddle.PaddleOCR.call_args.kwargs
        self.assertEqual(kwargs["lang"], "en")
        self.assertTrue(kwargs["use_angle_cls"])
        self.assertFalse(kwargs["use_gpu"])
        self.assertFalse(kwargs["show_log"])
        self.assertNotIn("use_doc_orientation_classify", kwargs)
    def test_paddle_warmup_skips_already_warmed_language_without_ensure(self) -> None:
        engine = OcrEngine()
        engine._backend = "paddle"
        engine._engines["en"] = object()
        engine._warmed_languages.add("en")

        with patch.object(engine, "_ensure_engine") as ensure, patch.object(
            engine,
            "_run_paddle",
            return_value=OcrResult(),
        ) as run_paddle:
            engine.warm_up("English")

        ensure.assert_not_called()
        run_paddle.assert_not_called()

    def test_paddle_warmup_runs_one_real_inference_per_language(self) -> None:
        engine = OcrEngine()
        engine._backend = "paddle"
        engine._engines["ch"] = object()

        with patch.object(engine, "_ensure_engine"), patch.object(
            engine,
            "_run_paddle",
            return_value=OcrResult(),
        ) as run_paddle:
            engine.warm_up("中文")
            engine.warm_up("中文")

        run_paddle.assert_called_once()


class OcrPreprocessTests(unittest.TestCase):
    def test_small_selection_region_gets_padding_and_upscale_for_ocr(self) -> None:
        image = Image.new("RGBA", (120, 45), (255, 255, 255, 255))

        processed = ImagePreprocessor().process(image)

        self.assertGreaterEqual(processed.height, 180)
        self.assertGreater(processed.width, image.width)

    def test_normal_height_region_keeps_original_size_by_default(self) -> None:
        image = Image.new("RGBA", (220, 120), (255, 255, 255, 255))

        processed = ImagePreprocessor().process(image)

        self.assertEqual(processed.size, image.size)


if __name__ == "__main__":
    unittest.main()
