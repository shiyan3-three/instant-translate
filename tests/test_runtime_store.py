"""Tests for per-group runtime state storage."""

from __future__ import annotations

import unittest

from app.state.group_state import ScreenRegion
from app.state.runtime_store import RuntimeStore


class RuntimeStoreTests(unittest.TestCase):
    """Verify group slot allocation and lifecycle rules."""

    def test_next_available_group_id_respects_capacity(self) -> None:
        store = RuntimeStore()

        self.assertEqual(store.next_available_group_id(), 1)
        store.save_region(1, ScreenRegion(0, 0, 200, 80))
        store.save_region(2, ScreenRegion(0, 120, 200, 80))
        self.assertEqual(store.next_available_group_id(), 3)
        store.save_region(3, ScreenRegion(0, 240, 200, 80))

        self.assertIsNone(store.next_available_group_id())
        self.assertFalse(store.has_capacity())

    def test_save_region_creates_config_and_runtime_defaults(self) -> None:
        store = RuntimeStore()
        region = ScreenRegion(12, 34, 320, 96)

        store.save_region(2, region)

        self.assertEqual(store.active_group_ids(), [2])
        self.assertEqual(store.regions[2], region)
        self.assertEqual(store.configs[2].target_language, "中文")
        self.assertEqual(store.runtime_states[2].latest_translation_text, "")
