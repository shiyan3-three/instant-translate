"""Screen capture helpers for OCR preparation."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QImage, QScreen

from app.state.group_state import ScreenRegion


@dataclass(frozen=True)
class CapturedRegionFrame:
    """Raw pixel snapshot for one fixed screen region."""

    width: int
    height: int
    pixel_bytes: bytes

    def is_empty(self) -> bool:
        """Return whether the captured frame contains any pixel payload."""

        return self.width <= 0 or self.height <= 0 or not self.pixel_bytes


class ScreenCaptureService:
    """Capture image data from a fixed desktop region."""

    def capture(self, screen: QScreen, region: ScreenRegion) -> CapturedRegionFrame:
        """Capture one region from the provided screen."""

        if not region.is_valid():
            raise ValueError("screen region is too small to capture")

        pixmap = screen.grabWindow(0, region.x, region.y, region.width, region.height)
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        return self.frame_from_image(image)

    @staticmethod
    def frame_from_image(image: QImage) -> CapturedRegionFrame:
        """Convert one Qt image into a compact pixel snapshot."""

        converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
        size = converted.sizeInBytes()
        bits = converted.constBits()
        return CapturedRegionFrame(
            width=converted.width(),
            height=converted.height(),
            pixel_bytes=bytes(bits[:size]),
        )
