"""Region change detection for OCR refresh gating."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b

from app.capture.screen_capture import CapturedRegionFrame


@dataclass
class RegionChangeDetector:
    """Compare captured frames and decide whether OCR should rerun."""

    _signatures: dict[int, bytes] = field(default_factory=dict)

    def should_process(self, group_id: int, frame: CapturedRegionFrame) -> bool:
        """Return whether one group frame is new enough to process again."""

        signature = self._build_signature(frame)
        previous_signature = self._signatures.get(group_id)
        self._signatures[group_id] = signature
        return previous_signature != signature

    def reset_group(self, group_id: int) -> None:
        """Forget the last frame signature for one group."""

        self._signatures.pop(group_id, None)

    @staticmethod
    def _build_signature(frame: CapturedRegionFrame) -> bytes:
        digest = blake2b(digest_size=16)
        digest.update(frame.width.to_bytes(4, "big", signed=False))
        digest.update(frame.height.to_bytes(4, "big", signed=False))
        digest.update(frame.pixel_bytes)
        return digest.digest()
