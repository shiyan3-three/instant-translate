"""Application-level dependency container."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.hotkeys import HotkeyMap
from app.settings import AppSettings


@dataclass
class ApplicationContext:
    """Hold top-level settings and runtime counters."""

    settings: AppSettings = field(default_factory=AppSettings)
    hotkeys: HotkeyMap = field(default_factory=HotkeyMap)
    active_group_count: int = 0
    edit_mode_enabled: bool = False
    default_source_language: str = "English"
    default_target_language: str = "中文"
    status_message: str = "\u684c\u9762\u58f3\u5df2\u5c31\u7eea\uff0c\u7b49\u5f85\u521b\u5efa\u9009\u533a\u3002"

    def note_create_selection_requested(self) -> None:
        """Track placeholder feedback for the create-selection hotkey."""

        self.status_message = (
            "\u5df2\u89e6\u53d1\u65b0\u5efa\u9009\u533a\u5feb\u6377\u952e\uff0c"
            "\u6846\u9009\u5c42\u5c06\u5728\u4e0b\u4e00\u9636\u6bb5\u63a5\u5165\u3002"
        )

    def toggle_edit_mode(self) -> bool:
        """Flip edit mode and return the new state."""

        self.edit_mode_enabled = not self.edit_mode_enabled
        if self.edit_mode_enabled:
            self.status_message = "\u7f16\u8f91\u6a21\u5f0f\u5df2\u5f00\u542f\u3002"
        else:
            self.status_message = "\u7f16\u8f91\u6a21\u5f0f\u5df2\u5173\u95ed\u3002"
        return self.edit_mode_enabled
