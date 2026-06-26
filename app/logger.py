"""Dual-file rolling logger for the translation pipeline.

Logs are organized by date: logs/YYYY/MM/DD.log and logs/YYYY/MM/DD-2.log (if multiple runs per day).

pipeline.log  — INFO+  level: user-facing events (OCR text, translations, errors).
debug.log     — DEBUG+ level: everything including per-tick timing for performance analysis.
"""

from __future__ import annotations

import logging
from datetime import datetime
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


def _get_log_path(log_type: str) -> Path:
    """Get log file path with date-based directory structure.
    
    Returns: logs/<log_type>/YYYY/MM/DD.log (or DD-2.log if exists)
    """
    base = _log_root()
    now = datetime.now()
    
    # Create directory structure: logs/<log_type>/YYYY/MM/
    log_dir = base / log_type / str(now.year) / f"{now.month:02d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next available file: DD.log, DD-2.log, DD-3.log...
    day = f"{now.day:02d}"
    log_file = log_dir / f"{day}.log"
    
    if not log_file.exists():
        return log_file
    
    # File exists, find next number
    counter = 2
    while True:
        log_file = log_dir / f"{day}-{counter}.log"
        if not log_file.exists():
            return log_file
        counter += 1


# ---------- lazy singletons ----------

_pipeline_logger: logging.Logger | None = None
_debug_logger: logging.Logger | None = None
_initialised = False


def _init_loggers() -> None:
    global _pipeline_logger, _debug_logger, _initialised
    if _initialised and _pipeline_logger is not None and _debug_logger is not None:
        return

    # -- pipeline log (INFO+) ------------------------------------------------
    pl = logging.getLogger("instant-translate.pipeline")
    pl.setLevel(logging.INFO)
    pl.propagate = False
    pipeline_path = _get_log_path("pipeline")
    _add_file_handler(pl, pipeline_path, logging.INFO)
    _pipeline_logger = pl

    # -- debug log (DEBUG+) --------------------------------------------------
    dl = logging.getLogger("instant-translate.debug")
    dl.setLevel(logging.DEBUG)
    dl.propagate = False
    debug_path = _get_log_path("debug")
    _add_file_handler(dl, debug_path, logging.DEBUG)
    _debug_logger = dl

    # session separator
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for lg in (pl, dl):
        lg.info("─── 启动 %s ───", ts)
    _initialised = True


def _add_file_handler(logger: logging.Logger, path: Path, level: int) -> None:
    h = logging.FileHandler(
        str(path),
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
