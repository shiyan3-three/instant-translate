"""Per-group runtime state models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScreenRegion:
    """Store fixed screen coordinates for one group."""

    x: int
    y: int
    width: int
    height: int

    def is_valid(self, minimum_size: int = 24) -> bool:
        """Return whether the region is large enough for OCR-oriented selection."""

        return self.width >= minimum_size and self.height >= minimum_size


@dataclass
class GroupConfig:
    """Store per-group preferences for one runtime session."""

    group_id: int
    source_language: str = "English"
    target_language: str = "\u4e2d\u6587"
    translation_dock: str = "bottom"
    paused: bool = False


@dataclass
class GroupRuntimeState:
    """Store mutable OCR and translation state for one group."""

    latest_ocr_text: str = ""
    latest_translation_text: str = ""
    current_request_id: int = 0
