"""Edit-mode controller placeholder."""

from __future__ import annotations


class EditModeController:
    """Track whether overlays are in editable mode."""

    def __init__(self) -> None:
        self.enabled = False

    def toggle(self) -> bool:
        """Toggle edit mode and return the new state."""
        self.enabled = not self.enabled
        return self.enabled
