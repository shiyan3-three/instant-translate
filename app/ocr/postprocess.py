"""OCR text cleanup and duplicate detection utilities."""

from __future__ import annotations

import re
import unicodedata

_SIZE_BADGE_RE = re.compile(r"\b\d{2,5}\s*[xX×]\s*\d{2,5}\b")
_TOOLBAR_WORDS = ("重选", "暂停", "继续", "删除")
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")


def normalize_ocr_text(raw_text: str) -> str:
    """Return OCR text cleaned enough to feed into translation."""

    if not raw_text:
        return ""

    text = unicodedata.normalize("NFKC", raw_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    clean_lines: list[str] = []
    for line in text.split("\n"):
        line = _SIZE_BADGE_RE.sub("", line)
        line = _WHITESPACE_RE.sub(" ", line).strip()
        if not line:
            continue
        if _is_toolbar_noise(line):
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def is_duplicate_ocr_text(current: str, previous: str) -> bool:
    """Return True when two OCR outputs differ only by cleanup-level noise."""

    current_norm = _comparison_key(normalize_ocr_text(current))
    previous_norm = _comparison_key(normalize_ocr_text(previous))
    return bool(current_norm and previous_norm and current_norm == previous_norm)


def _comparison_key(text: str) -> str:
    """Make a stable key for OCR duplicate checks."""

    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _is_toolbar_noise(line: str) -> bool:
    """Detect a line that is only the selection-box toolbar text."""

    compact = re.sub(r"\s+", "", line)
    return bool(compact) and all(word in _TOOLBAR_WORDS for word in _split_toolbar_words(compact))


def _split_toolbar_words(compact: str) -> list[str]:
    """Split concatenated toolbar labels while rejecting unknown text."""

    words: list[str] = []
    cursor = 0
    while cursor < len(compact):
        for word in _TOOLBAR_WORDS:
            if compact.startswith(word, cursor):
                words.append(word)
                cursor += len(word)
                break
        else:
            return [compact]
    return words
