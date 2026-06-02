"""Shared helpers for desktop UI tests."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication


def ensure_qapplication() -> QApplication:
    """Return a singleton QApplication for widget-based tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
