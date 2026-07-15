"""OCR engine — dual-backend text extraction for fixed screen regions.

Tries PaddleOCR first (best accuracy for CJK), falls back to Tesseract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.capture.screen_capture import CapturedRegionFrame

from app.ocr.preprocess import ImagePreprocessor, PreprocessConfig


def _prepare_frozen_paddleocr_imports() -> None:
    """Expose PaddleOCR 2.x's bundled ``ppocr``/``tools`` packages.

    PaddleOCR 2.7 imports those directories as top-level packages.  A normal
    installation makes that work by adding its package directory to
    ``sys.path``; a PyInstaller bundle does not, even when the source tree is
    present as collected data.
    """

    if not getattr(sys, "frozen", False):
        return
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    paddleocr_root = bundle_root / "paddleocr"
    if paddleocr_root.is_dir():
        paddleocr_path = str(paddleocr_root)
        if paddleocr_path not in sys.path:
            sys.path.insert(0, paddleocr_path)


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
        self._warmed_languages: set[str] = set()
        self._engine_lock = threading.RLock()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def warm_up(self, source_language: str = "English") -> None:
        """Pre-initialise the OCR backend so the first real call is fast.

        Safe to call from a background thread.  If the backend is
        already initialised for this language, this is a no-op.
        """
        lang_code = self._PADDLE_LANG_MAP.get(source_language, "en")
        try:
            with self._engine_lock:
                if self._backend == "paddle" and lang_code in self._warmed_languages:
                    from app.logger import get_debug_logger
                    get_debug_logger().debug("OCR inference warmup skipped: cached lang=%s", lang_code)
                    return

                self._ensure_engine(source_language)
                if self._backend != "paddle" or lang_code in self._warmed_languages:
                    return

                # Constructing PaddleOCR does not initialise all inference
                # kernels.  A tiny real pass moves the multi-second cold cost
                # into the existing background warm-up thread.
                from PIL import Image, ImageDraw

                image = Image.new("RGB", (96, 32), "white")
                ImageDraw.Draw(image).text((4, 7), "ABC", fill="black")
                self._run_paddle(image, source_language)
                self._warmed_languages.add(lang_code)
                from app.logger import get_debug_logger
                get_debug_logger().debug("OCR inference warmup complete: lang=%s", lang_code)
        except RuntimeError:
            pass  # no backend available — logged inside _ensure_engine
        except Exception as exc:
            # Warm-up is an optimisation; a failure must not disable normal OCR.
            from app.logger import get_debug_logger
            get_debug_logger().warning("OCR inference warmup failed: %s", exc)

    def recognise(
        self,
        frame: CapturedRegionFrame,
        source_language: str = "English",
    ) -> OcrResult:
        """Extract text from one captured region frame."""

        import time as _time
        from app.logger import get_debug_logger

        t0 = _time.perf_counter()

        # --- OCR-IN marker (for diagnosing stuck/crash) ---
        get_debug_logger().debug(
            "[OCR-IN] lang=%s size=%dx%d",
            source_language, frame.width, frame.height,
        )

        image = self._preprocessor.from_rgba_bytes(
            frame.pixel_bytes,
            frame.width,
            frame.height,
        )
        processed = self._preprocessor.process(image, self._preprocess_config)
        pp_ms = (_time.perf_counter() - t0) * 1000

        result = self._run_ocr(processed, source_language)
        total_ms = (_time.perf_counter() - t0) * 1000

        # --- OCR-OUT marker ---
        out_len = len(result.raw_text)
        get_debug_logger().debug(
            "[OCR-OUT] lang=%s prep=%.0fms total=%.0fms len=%d sha256=%s",
            source_language,
            pp_ms,
            total_ms,
            out_len,
            hashlib.sha256(result.raw_text.encode("utf-8")).hexdigest()[:12],
        )

        return result

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _run_ocr(self, image, source_language: str) -> OcrResult:
        with self._engine_lock:
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
            from app.logger import get_logger
            get_logger().warning("PaddleOCR 运行时异常 (NotImplementedError)，降级到 Tesseract")
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
            from app.logger import get_debug_logger
            get_debug_logger().debug("pytesseract 未安装，Tesseract OCR 不可用")
            return OcrResult()

        try:
            text = pytesseract.image_to_string(image, lang=tesseract_lang)
        except Exception as exc:
            from app.logger import get_debug_logger
            get_debug_logger().warning("Tesseract OCR 识别失败: %s", exc)
            return OcrResult()

        clean = text.strip()
        if not clean:
            return OcrResult()
        return OcrResult(raw_text=clean)

    def _ensure_engine(self, source_language: str) -> None:
        lang_code = self._PADDLE_LANG_MAP.get(source_language, "en")

        # Diagnostic logging
        from app.logger import get_debug_logger
        get_debug_logger().debug(
            "OCR engine ensure: backend=%s lang=%s cached_langs=%s instance_id=%s",
            self._backend, lang_code, list(self._engines.keys()), id(self)
        )

        # Backend already active and this language's engine is cached → done
        if self._backend == "tesseract":
            return
        if self._backend == "paddle" and lang_code in self._engines:
            return

        # 1) try PaddleOCR -------------------------------------------------
        _prepare_frozen_paddleocr_imports()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            get_debug_logger().warning("PaddleOCR import failed: %s", exc)
        else:
            try:
                engine = PaddleOCR(
                    lang=lang_code,
                    # PaddleOCR 2.7.x API.  Do not use 3.x document-pipeline
                    # flags here: our declared dependency is paddleocr==2.7.3.
                    use_angle_cls=True,
                    use_gpu=False,
                    show_log=False,
                )
                self._engines[lang_code] = engine
                self._backend = "paddle"
                from app.logger import get_logger
                get_logger().info("OCR 后端已初始化: PaddleOCR (lang=%s)", lang_code)
                return
            except Exception as exc:
                from app.logger import get_debug_logger
                get_debug_logger().warning("PaddleOCR 初始化失败 (lang=%s): %s", lang_code, exc)
                pass

        # 2) try Tesseract -------------------------------------------------
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            pass
        else:
            self._backend = "tesseract"
            from app.logger import get_logger
            get_logger().info("OCR 后端已初始化: Tesseract")
            return

        raise RuntimeError(
            "No OCR backend is available.  Install one of:\n"
            "  pip install paddlepaddle paddleocr    (recommended)\n"
            "  pip install pytesseract               (lighter, requires system Tesseract)"
        )
