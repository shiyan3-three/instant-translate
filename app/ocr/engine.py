"""OCR engine — dual-backend text extraction for fixed screen regions.

Tries PaddleOCR first (best accuracy for CJK), falls back to Tesseract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.capture.screen_capture import CapturedRegionFrame

from app.ocr.preprocess import ImagePreprocessor, PreprocessConfig


@dataclass
class OcrLine:
    """One recognized text line with position and confidence."""

    text: str
    confidence: float
    box: tuple[int, int, int, int] | None = None


@dataclass
class OcrResult:
    """Aggregate OCR output for one captured region frame."""

    lines: list[OcrLine] = field(default_factory=list)
    raw_text: str = ""

    @property
    def is_empty(self) -> bool:
        """Return whether the OCR produced usable text."""

        return not self.raw_text.strip()


class OcrEngine:
    """Run OCR against a preprocessed image region.

    Backend priority
    ----------------
    1. **PaddleOCR** — best accuracy for Chinese / Japanese / English,
       especially on complex backgrounds.  Requires ``paddlepaddle``
       and ``paddleocr`` to be installed.
    2. **Tesseract** — zero-Python-dependency fallback via
       ``pytesseract``.  Requires the Tesseract OCR engine to be
       installed separately (``scoop install tesseract`` /
       ``choco install tesseract`` / https://github.com/UB-Mannheim/tesseract/wiki).

    The engine is lazily initialised so the application can start even
    when neither backend is available — OCR will simply return empty
    results until one is configured.
    """

    _TESSERACT_LANG_MAP = {
        "en": "eng",
        "ch": "chi_sim+eng",
        "japan": "jpn+eng",
    }

    _PADDLE_LANG_MAP = {
        "English": "en",
        "中文": "ch",
        "日本語": "japan",
    }

    def __init__(
        self,
        preprocessor: ImagePreprocessor | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        self._preprocessor = preprocessor or ImagePreprocessor()
        self._preprocess_config = preprocess_config
        self._backend = None  # "paddle" | "tesseract" | None
        self._engines: dict[str, object] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def recognise(
        self,
        frame: CapturedRegionFrame,
        source_language: str = "English",
    ) -> OcrResult:
        """Extract text from one captured region frame."""

        import time as _time
        from app.logger import get_debug_logger

        t0 = _time.perf_counter()

        image = self._preprocessor.from_rgba_bytes(
            frame.pixel_bytes,
            frame.width,
            frame.height,
        )
        processed = self._preprocessor.process(image, self._preprocess_config)
        pp_ms = (_time.perf_counter() - t0) * 1000

        result = self._run_ocr(processed, source_language)
        total_ms = (_time.perf_counter() - t0) * 1000

        if not result.is_empty:
            get_debug_logger().debug(
                "OCR: prep=%.0fms total=%.0fms text=%r",
                pp_ms, total_ms, result.raw_text[:80],
            )

        return result

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _run_ocr(self, image, source_language: str) -> OcrResult:
        self._ensure_engine(source_language)

        if self._backend == "paddle":
            return self._run_paddle(image, source_language)
        if self._backend == "tesseract":
            return self._run_tesseract(image, source_language)
        return OcrResult()

    def _run_paddle(self, image, source_language: str) -> OcrResult:
        lang_code = self._PADDLE_LANG_MAP.get(source_language, "en")
        ocr = self._engines[lang_code]

        import numpy as np

        array = np.array(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)

        try:
            raw_results = ocr.ocr(array)
        except NotImplementedError:
            # PaddlePaddle oneDNN / PIR attribute bug on some platforms
            # → fall through to Tesseract
            self._backend = "tesseract"
            self._engines.pop(lang_code, None)
            return self._run_tesseract(image, source_language)

        if raw_results is None or len(raw_results) == 0:
            return OcrResult()

        lines: list[OcrLine] = []
        for group in raw_results:
            if group is None:
                continue
            for item in group:
                box_points, (text, confidence) = item
                x1 = int(min(p[0] for p in box_points))
                y1 = int(min(p[1] for p in box_points))
                x2 = int(max(p[0] for p in box_points))
                y2 = int(max(p[1] for p in box_points))
                lines.append(
                    OcrLine(
                        text=text,
                        confidence=float(confidence),
                        box=(x1, y1, x2, y2),
                    )
                )

        combined = "\n".join(line.text for line in lines)
        return OcrResult(lines=lines, raw_text=combined)

    def _run_tesseract(self, image, source_language: str) -> OcrResult:
        lang_code = self._PADDLE_LANG_MAP.get(source_language, "en")
        tesseract_lang = self._TESSERACT_LANG_MAP.get(lang_code, "eng")

        try:
            import pytesseract
        except ImportError:
            return OcrResult()

        try:
            text = pytesseract.image_to_string(image, lang=tesseract_lang)
        except Exception:
            return OcrResult()

        clean = text.strip()
        if not clean:
            return OcrResult()
        return OcrResult(raw_text=clean)

    def _ensure_engine(self, source_language: str) -> None:
        lang_code = self._PADDLE_LANG_MAP.get(source_language, "en")

        # Backend already active and this language's engine is cached → done
        if self._backend == "tesseract":
            return
        if self._backend == "paddle" and lang_code in self._engines:
            return

        # 1) try PaddleOCR -------------------------------------------------
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            pass
        else:
            try:
                engine = PaddleOCR(
                    lang=lang_code,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                self._engines[lang_code] = engine
                self._backend = "paddle"
                return
            except Exception:
                pass

        # 2) try Tesseract -------------------------------------------------
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            pass
        else:
            self._backend = "tesseract"
            return

        raise RuntimeError(
            "No OCR backend is available.  Install one of:\n"
            "  pip install paddlepaddle paddleocr    (recommended)\n"
            "  pip install pytesseract               (lighter, requires system Tesseract)"
        )
