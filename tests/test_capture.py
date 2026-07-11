"""Tests for screen-capture preparation utilities."""

from __future__ import annotations

import unittest

from PySide6.QtGui import QImage

from app.capture.change_detector import RegionChangeDetector
from app.capture.screen_capture import CapturedRegionFrame, ScreenCaptureService
from app.state.group_state import ScreenRegion


class RegionChangeDetectorTests(unittest.TestCase):
    """Verify capture change detection before OCR integration."""

    def test_first_frame_for_group_is_treated_as_changed(self) -> None:
        detector = RegionChangeDetector()
        frame = CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x10" * 64)

        changed = detector.should_process(1, frame)

        self.assertTrue(changed)

    def test_identical_frames_do_not_retrigger_processing(self) -> None:
        detector = RegionChangeDetector()
        frame = CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x10" * 64)

        detector.should_process(1, frame)
        changed = detector.should_process(1, frame)

        self.assertFalse(changed)

    def test_different_frames_trigger_processing(self) -> None:
        detector = RegionChangeDetector()
        first = CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x10" * 64)
        second = CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x20" * 64)

        detector.should_process(1, first)
        changed = detector.should_process(1, second)

        self.assertTrue(changed)

    def test_tiny_pixel_noise_does_not_retrigger_processing(self) -> None:
        detector = RegionChangeDetector(sample_stride=1, min_changed_ratio=0.003)
        first = self._solid_frame(20, 20, 0)
        noisy = bytearray(first.pixel_bytes)
        noisy[0:4] = bytes((255, 255, 255, 255))
        second = CapturedRegionFrame(width=20, height=20, pixel_bytes=bytes(noisy))

        detector.should_process(1, first)
        changed = detector.should_process(1, second)

        self.assertFalse(changed)

    def test_text_sized_pixel_change_retriggers_processing(self) -> None:
        detector = RegionChangeDetector(sample_stride=1, min_changed_ratio=0.003)
        first = self._solid_frame(20, 20, 0)
        changed_pixels = bytearray(first.pixel_bytes)
        for pixel_index in range(8):
            offset = pixel_index * 4
            changed_pixels[offset : offset + 4] = bytes((255, 255, 255, 255))
        second = CapturedRegionFrame(width=20, height=20, pixel_bytes=bytes(changed_pixels))

        detector.should_process(1, first)
        changed = detector.should_process(1, second)

        self.assertTrue(changed)

    def test_reset_group_forgets_previous_signature(self) -> None:
        detector = RegionChangeDetector()
        frame = CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x10" * 64)

        detector.should_process(1, frame)
        detector.reset_group(1)
        changed = detector.should_process(1, frame)

        self.assertTrue(changed)

    @staticmethod
    def _solid_frame(width: int, height: int, value: int) -> CapturedRegionFrame:
        pixel = bytes((value, value, value, 255))
        return CapturedRegionFrame(width=width, height=height, pixel_bytes=pixel * width * height)


class ScreenCaptureServiceTests(unittest.TestCase):
    def test_virtual_desktop_region_is_converted_to_screen_local_coordinates(self) -> None:
        class Geometry:
            @staticmethod
            def x():
                return -1920

            @staticmethod
            def y():
                return 100

        class Pixmap:
            @staticmethod
            def toImage():
                image = QImage(30, 30, QImage.Format.Format_RGBA8888)
                image.fill(0)
                return image

        class Screen:
            def __init__(self):
                self.args = None

            @staticmethod
            def geometry():
                return Geometry()

            def grabWindow(self, *args):
                self.args = args
                return Pixmap()

        screen = Screen()
        service = ScreenCaptureService()

        service.capture(screen, ScreenRegion(-1900, 130, 30, 30))

        self.assertEqual(screen.args, (0, 20, 30, 30, 30))
