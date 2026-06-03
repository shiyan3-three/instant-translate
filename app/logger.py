"""Dual-file rolling logger for the translation pipeline.

pipeline.log  — INFO+  level: user-facing events (OCR text, translations, errors).
debug.log     — DEBUG+ level: everything including per-tick timing for performance analysis.

Each file auto-rotates at 1 MiB, keeping one backup (``.1.log``).
"""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _log_root() -> Path:
    try:
        import sys

        if getattr(sys, "frozen", False):
            root = Path(sys.executable).parent
        else:
            root = Path(__file__).resolve().parent.parent
    except Exception:
        root = Path.cwd()

    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root / "logs"


# ---------- lazy singletons ----------

_pipeline_logger: logging.Logger | None = None
_debug_logger: logging.Logger | None = None
_initialised = False


def _init_loggers() -> None:
    global _pipeline_logger, _debug_logger, _initialised
    if _initialised:
        return
    _initialised = True

    base = _log_root()

    # -- pipeline log (INFO+) ------------------------------------------------
    pl = logging.getLogger("instant-translate.pipeline")
    pl.setLevel(logging.INFO)
    pl.propagate = False
    _add_rolling_handler(pl, base / "pipeline.log", logging.INFO)
    _pipeline_logger = pl

    # -- debug log (DEBUG+) --------------------------------------------------
    dl = logging.getLogger("instant-translate.debug")
    dl.setLevel(logging.DEBUG)
    dl.propagate = False
    _add_rolling_handler(dl, base / "debug.log", logging.DEBUG)
    _debug_logger = dl

    # session separator
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for lg in (pl, dl):
        lg.info("─── 启动 %s ───", ts)


def _add_rolling_handler(logger: logging.Logger, path: Path, level: int) -> None:
    h = RotatingFileHandler(
        str(path),
        maxBytes=1_048_576,  # 1 MiB
        backupCount=1,
        encoding="utf-8",
    )
    h.setLevel(level)
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(h)


def get_logger() -> logging.Logger:
    """Return the *pipeline* logger (INFO+, user-facing).

    For debug-level performance data use ``get_debug_logger()``.
    """
    _init_loggers()
    return _pipeline_logger  # type: ignore[return-value]


def get_debug_logger() -> logging.Logger:
    """Return the *debug* logger (DEBUG+, full timeline)."""
    _init_loggers()
    return _debug_logger  # type: ignore[return-value]
