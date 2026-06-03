"""Region change detection for OCR refresh gating."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.capture.screen_capture import CapturedRegionFrame


@dataclass(frozen=True)
class _FrameSignature:
    """Compact luminance signature for one captured frame."""

    width: int
    height: int
    samples: bytes


@dataclass
class RegionChangeDetector:
    """Compare captured frames and decide whether OCR should rerun."""

    sample_stride: int = 1
    min_changed_ratio: float = 0.003
    pixel_delta_threshold: int = 8
    _signatures: dict[int, _FrameSignature] = field(default_factory=dict)

    def should_process(self, group_id: int, frame: CapturedRegionFrame) -> bool:
        """Return whether one group frame is new enough to process again."""

        signature = self._build_signature(frame)
        previous_signature = self._signatures.get(group_id)
        self._signatures[group_id] = signature
        if previous_signature is None:
            return True
        return self._has_meaningful_change(previous_signature, signature)

    def reset_group(self, group_id: int) -> None:
        """Forget the last frame signature for one group."""

        self._signatures.pop(group_id, None)

    def _build_signature(self, frame: CapturedRegionFrame) -> _FrameSignature:
        stride = max(1, self.sample_stride)
        step = 4 * stride
        pixels = frame.pixel_bytes
        samples = bytearray()

        for offset in range(0, max(0, len(pixels) - 2), step):
            red = pixels[offset]
            green = pixels[offset + 1]
            blue = pixels[offset + 2]
            samples.append((red * 77 + green * 150 + blue * 29) >> 8)

        return _FrameSignature(
            width=frame.width,
            height=frame.height,
            samples=bytes(samples),
        )

    def _has_meaningful_change(self, previous: _FrameSignature, current: _FrameSignature) -> bool:
        if previous.width != current.width or previous.height != current.height:
            return True
        if previous.samples == current.samples:
            return False
        if not previous.samples or len(previous.samples) != len(current.samples):
            return True

        changed = sum(
            1
            for before, after in zip(previous.samples, current.samples)
            if abs(before - after) >= self.pixel_delta_threshold
        )
        changed_ratio = changed / len(current.samples)
        return changed_ratio >= self.min_changed_ratio
