"""Runtime compatibility checks kept free of GUI and optional dependencies."""

from __future__ import annotations

import sys
from collections.abc import Sequence


MIN_SUPPORTED_PYTHON = (3, 10)
MAX_SUPPORTED_PYTHON = (3, 12)
PACKAGING_PYTHON = (3, 12)


def is_supported_python(version: Sequence[int] | None = None) -> bool:
    """Return whether a Python version is within the published source range."""

    candidate = tuple(version or sys.version_info)
    major_minor = candidate[:2]
    return MIN_SUPPORTED_PYTHON <= major_minor <= MAX_SUPPORTED_PYTHON


def ensure_supported_python(version: Sequence[int] | None = None) -> None:
    """Fail before importing desktop/translation dependencies on bad runtimes."""

    candidate = tuple(version or sys.version_info)
    if is_supported_python(candidate):
        return
    actual = ".".join(str(part) for part in candidate[:3])
    raise RuntimeError(
        "Instant Translate requires Python 3.10 through 3.12 "
        f"(detected {actual}). Packaging and CI use Python "
        f"{PACKAGING_PYTHON[0]}.{PACKAGING_PYTHON[1]}."
    )
