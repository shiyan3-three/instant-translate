"""Lightweight OCR-oriented image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class PreprocessConfig:
    """Tune preprocessing parameters for different capture scenarios."""

    grayscale: bool = True
    auto_contrast: bool = True
    contrast_factor: float = 1.5
    sharpen: bool = True
    binarize: bool = False
    binarize_threshold: int = 128
    upscale_factor: float = 1.0
    invert: bool = False
    small_region_min_height: int = 96
    small_region_horizontal_padding: int = 12
    small_region_upscale_below_height: int = 72
    small_region_upscale_factor: float = 2.0


class ImagePreprocessor:
    """Apply lightweight OCR-oriented image cleanup.

    Designed to improve OCR accuracy when source text appears on noisy,
    low-contrast, or busy backgrounds — common in live streams, games,
    and video players.
    """

    _default_config = PreprocessConfig()

    def process(
        self,
        image: Image.Image,
        config: PreprocessConfig | None = None,
    ) -> Image.Image:
        """Run the full preprocessing pipeline on one captured region image."""

        cfg = config or self._default_config
        _, original_height = image.size

        if original_height < cfg.small_region_min_height:
            vertical_padding = max(0, cfg.small_region_min_height - original_height)
            top = vertical_padding // 2
            bottom = vertical_padding - top
            left = max(0, cfg.small_region_horizontal_padding)
            right = left
            fill = (255, 255, 255, 255) if image.mode == "RGBA" else "white"
            image = ImageOps.expand(
                image,
                border=(left, top, right, bottom),
                fill=fill,
            )

        if cfg.grayscale:
            image = image.convert("L")

        if cfg.auto_contrast:
            image = ImageOps.autocontrast(image, cutoff=2)

        if cfg.contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(cfg.contrast_factor)

        if cfg.sharpen:
            image = image.filter(ImageFilter.SHARPEN)

        if cfg.binarize:
            image = image.point(lambda p: 255 if p > cfg.binarize_threshold else 0, mode="1")
            image = image.convert("L")

        if cfg.invert:
            image = ImageOps.invert(image)

        upscale_factor = cfg.upscale_factor
        if original_height < cfg.small_region_upscale_below_height:
            upscale_factor = max(upscale_factor, cfg.small_region_upscale_factor)

        if upscale_factor > 1.0:
            w, h = image.size
            image = image.resize(
                (int(w * upscale_factor), int(h * upscale_factor)),
                Image.LANCZOS,
            )

        return image

    @classmethod
    def from_rgba_bytes(cls, pixel_bytes: bytes, width: int, height: int) -> Image.Image:
        """Convert raw RGBA8888 bytes (as returned by QImage) into a PIL Image."""

        image = Image.frombytes("RGBA", (width, height), pixel_bytes, "raw", "RGBA")
        return image

    @classmethod
    def to_bytes(cls, image: Image.Image) -> bytes:
        """Convert a PIL Image back to raw pixel bytes for downstream consumers."""

        if image.mode == "RGBA":
            return image.tobytes("raw", "RGBA")
        if image.mode == "L":
            rgb = image.convert("RGB")
            return rgb.tobytes("raw", "RGB")
        return image.tobytes()
