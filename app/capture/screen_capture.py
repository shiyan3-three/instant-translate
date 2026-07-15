"""Screen capture helpers for OCR preparation."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect
from PySide6.QtGui import QImage, QPainter, QScreen

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

        if not region.is_valid(minimum_size=1):
            raise ValueError("screen region is too small to capture")

        # ScreenRegion uses virtual-desktop coordinates, while QScreen.grabWindow
        # expects coordinates relative to the selected screen.
        geometry = screen.geometry()
        local_x = region.x - geometry.x()
        local_y = region.y - geometry.y()
        pixmap = screen.grabWindow(0, local_x, local_y, region.width, region.height)
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        return self.frame_from_image(image)

    def capture_region(
        self,
        screens: list[QScreen],
        region: ScreenRegion,
    ) -> CapturedRegionFrame:
        """Capture a virtual-desktop region, compositing every intersecting screen.

        ``QScreen.grabWindow`` uses each screen's local coordinates.  A
        region that straddles screens therefore cannot be read from only the
        screen containing its centre.  Each intersecting tile is captured in
        local coordinates and painted at its virtual-desktop offset.
        """

        if not region.is_valid(minimum_size=1):
            raise ValueError("screen region is too small to capture")
        request = QRect(region.x, region.y, region.width, region.height)
        intersections: list[tuple[QScreen, QRect, QRect]] = []
        for screen in screens:
            geometry = screen.geometry()
            intersection = geometry.intersected(request)
            if intersection.isEmpty():
                continue
            local = QRect(
                intersection.x() - geometry.x(),
                intersection.y() - geometry.y(),
                intersection.width(),
                intersection.height(),
            )
            intersections.append((screen, intersection, local))
        if not intersections:
            raise ValueError("screen region does not intersect any available screen")
        if len(intersections) == 1 and intersections[0][1] == request:
            return self.capture(intersections[0][0], region)

        composed = QImage(
            region.width,
            region.height,
            QImage.Format.Format_RGBA8888,
        )
        composed.fill(0)
        painter = QPainter(composed)
        try:
            for screen, intersection, local in intersections:
                pixmap = screen.grabWindow(
                    0,
                    local.x(),
                    local.y(),
                    local.width(),
                    local.height(),
                )
                image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                painter.drawImage(
                    intersection.x() - region.x,
                    intersection.y() - region.y,
                    image,
                )
        finally:
            painter.end()
        return self.frame_from_image(composed)

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
