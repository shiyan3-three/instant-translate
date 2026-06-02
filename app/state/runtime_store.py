"""Runtime state store for up to three active selection groups."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.state.group_state import GroupConfig, GroupRuntimeState, ScreenRegion


@dataclass
class RuntimeStore:
    """Hold per-group state for up to three active groups."""

    max_groups: int = 3
    regions: dict[int, ScreenRegion] = field(default_factory=dict)
    configs: dict[int, GroupConfig] = field(default_factory=dict)
    runtime_states: dict[int, GroupRuntimeState] = field(default_factory=dict)
    translation_window_positions: dict[int, tuple[int, int]] = field(default_factory=dict)

    def active_group_ids(self) -> list[int]:
        """Return active group identifiers in order."""

        return sorted(self.regions)

    def next_available_group_id(self) -> int | None:
        """Return the next empty group slot, or None when full."""

        for group_id in range(1, self.max_groups + 1):
            if group_id not in self.regions:
                return group_id
        return None

    def has_capacity(self) -> bool:
        """Return whether another group can be created."""

        return self.next_available_group_id() is not None

    def save_region(self, group_id: int, region: ScreenRegion) -> None:
        """Create or replace one active group region and its defaults."""

        self.regions[group_id] = region
        self.configs.setdefault(group_id, GroupConfig(group_id=group_id))
        self.runtime_states.setdefault(group_id, GroupRuntimeState())

    def set_translation_window_position(self, group_id: int, x: int, y: int) -> None:
        """Remember a user-dragged translation window location."""

        self.translation_window_positions[group_id] = (x, y)

    def clear_translation_window_position(self, group_id: int) -> None:
        """Clear a user override so dock-based positioning resumes."""

        self.translation_window_positions.pop(group_id, None)

    def remove_group(self, group_id: int) -> None:
        """Remove a group and all related runtime state."""

        self.regions.pop(group_id, None)
        self.configs.pop(group_id, None)
        self.runtime_states.pop(group_id, None)
        self.translation_window_positions.pop(group_id, None)
