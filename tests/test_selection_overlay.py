"""Tests for drag-selection helpers."""

from __future__ import annotations

import unittest

from PySide6.QtCore import QPoint

from app.overlay.selection_overlay import RegionSelectionOverlay
from app.state.group_state import ScreenRegion


class RegionSelectionOverlayTests(unittest.TestCase):
    """Cover pure helper behavior without interactive mouse simulation."""

    def test_build_region_normalizes_drag_direction(self) -> None:
        region = RegionSelectionOverlay.build_region(QPoint(220, 140), QPoint(70, 25))

        self.assertEqual(region.x, 70)
        self.assertEqual(region.y, 25)
        self.assertEqual(region.width, 150)
        self.assertEqual(region.height, 115)

    def test_format_region_label_matches_selection_box_size_style(self) -> None:
        label = RegionSelectionOverlay.format_region_label(ScreenRegion(0, 0, 320, 96))

        self.assertEqual(label, "320 x 96")
