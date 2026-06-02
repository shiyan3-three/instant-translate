"""Tests for screen-capture preparation utilities."""

from __future__ import annotations

import unittest

from app.capture.change_detector import RegionChangeDetector
from app.capture.screen_capture import CapturedRegionFrame


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

    def test_reset_group_forgets_previous_signature(self) -> None:
        detector = RegionChangeDetector()
        frame = CapturedRegionFrame(width=4, height=4, pixel_bytes=b"\x10" * 64)

        detector.should_process(1, frame)
        detector.reset_group(1)
        changed = detector.should_process(1, frame)

        self.assertTrue(changed)
