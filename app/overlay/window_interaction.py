"""Native input-pass-through helpers for overlay windows."""

from __future__ import annotations

import ctypes

from PySide6.QtWidgets import QWidget

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011
SWP_FRAMECHANGED = 0x0020
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_FLAGS = SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER


def set_window_click_through(widget: QWidget, enabled: bool) -> None:
    """Toggle Windows native click-through for a top-level overlay widget."""

    setattr(widget, "_input_passthrough_enabled", enabled)
    try:
        user32 = ctypes.windll.user32
    except AttributeError:
        return

    hwnd = int(widget.winId())
    ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ex_style |= WS_EX_LAYERED
    if enabled:
        ex_style |= WS_EX_TRANSPARENT
    else:
        ex_style &= ~WS_EX_TRANSPARENT

    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)
    user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, SWP_FLAGS)


def is_window_click_through(widget: QWidget) -> bool:
    """Return whether a widget currently has native click-through enabled."""

    try:
        user32 = ctypes.windll.user32
    except AttributeError:
        return bool(getattr(widget, "_input_passthrough_enabled", False))

    hwnd = int(widget.winId())
    ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    return bool(ex_style & WS_EX_TRANSPARENT)


def set_window_excluded_from_capture(widget: QWidget, enabled: bool) -> bool:
    """Record capture-exclusion intent without calling risky native APIs.

    ``SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`` looked attractive for
    hiding overlays from OCR screenshots, but it can crash or behave
    inconsistently across real Windows/Qt/frozen-app combinations. OCR avoids
    edit-mode overlay pollution by hiding inner selection badges while polling
    continues.
    """

    setattr(widget, "_capture_excluded_enabled", enabled)
    return False


def is_window_excluded_from_capture(widget: QWidget) -> bool:
    """Return whether capture exclusion is intended/enabled for a widget."""

    return bool(getattr(widget, "_capture_excluded_enabled", False))
