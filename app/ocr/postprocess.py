"""OCR text cleanup and duplicate detection utilities."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_TOOLBAR_WORDS = (
    "重选",
    "暂停",
    "继续",
    "删除",
    "翻译有误",
    "OCR",
    "中文",
    "日本語",
    "日本语",
    "English",
    "上",
    "下",
    "左",
    "右",
)
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_SIZE_BADGE_RE = re.compile(r"\d{2,5}\s*[xX\u00d7\u8133]\s*\d{1,5}|\d{1,5}\s*[xX\u00d7\u8133]\s*\d{2,5}")
_SEARCH_BAR_RE = re.compile(r"^(?:[Qq]\s*)?(?:搜索|搜寻|搜尋|Search)$", re.I)
_LANGUAGE_PAIR_RE = re.compile(
    r"^(?:中文|中国語|Chinese|English|日本語|日本语|Japanese|英语|英文|日语|日文)"
    r"\s*(?:=>|->|→|到|to)\s*"
    r"(?:中文|中国語|Chinese|English|日本語|日本语|Japanese|英语|英文|日语|日文)$",
    re.I,
)
_CONTENT_RE = re.compile(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_PUNCT_OR_SYMBOL_RE = re.compile(r"[\W_]+", re.UNICODE)
_STRUCTURAL_LABEL_RE = re.compile(
    r"^\([A-Za-z][A-Za-z0-9 _:/.-]{1,24}\)$"
)
_STRUCTURAL_RESOURCE_RE = re.compile(
    r"^[A-Z]{1,8}-\d{1,5}\s+\([A-Za-z][A-Za-z0-9 _:/.-]{1,24}\)$"
)
_SOCIAL_METRIC_RE = re.compile(
    r"^(?:关注|粉丝|获赞|点赞|赞|评论|收藏|分享|转发|播放|浏览)"
    r"\s*[:：]?\s*[\d.,]+(?:[wW万kK千]+)?$"
)


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
        if _is_search_bar_noise(line):
            continue
        if _is_language_pair_noise(line):
            continue
        if _is_social_metric_noise(line):
            continue
        if _is_structural_ui_noise(line):
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def is_duplicate_ocr_text(current: str, previous: str) -> bool:
    """Return True when two OCR outputs differ only by cleanup-level noise."""

    current_norm = _comparison_key(normalize_ocr_text(current))
    previous_norm = _comparison_key(normalize_ocr_text(previous))
    return bool(current_norm and previous_norm and current_norm == previous_norm)


def is_suspicious_ocr_text(text: str, source_language: str = "") -> bool:
    """Return True when OCR text is likely overlay noise or a broken frame."""

    clean = normalize_ocr_text(text)
    key = _comparison_key(clean)
    compact = re.sub(r"\s+", "", key)
    if not compact:
        return True
    if compact.isdigit():
        return True
    if not _CONTENT_RE.search(compact):
        return True
    if _PUNCT_OR_SYMBOL_RE.sub("", compact) == "":
        return True

    cjk_count = len(_CJK_RE.findall(compact))
    kana_count = len(_KANA_RE.findall(compact))
    latin_count = len(_LATIN_RE.findall(compact))
    lang = source_language.lower()

    if ("中文" in source_language or "chinese" in lang) and kana_count and len(compact) <= 12:
        return True
    if ("english" in lang or source_language == "English") and (cjk_count + kana_count) > max(latin_count, 1):
        return True

    return False


def is_stable_ocr_text(current: str, previous: str) -> bool:
    """Return True when two OCR readings are close enough to trust."""

    current_key = _comparison_key(normalize_ocr_text(current))
    previous_key = _comparison_key(normalize_ocr_text(previous))
    if not current_key or not previous_key:
        return False
    if current_key == previous_key:
        return True

    current_len = len(current_key)
    previous_len = len(previous_key)
    if min(current_len, previous_len) < 8:
        return False

    length_ratio = min(current_len, previous_len) / max(current_len, previous_len)
    if length_ratio < 0.9:
        return False

    return SequenceMatcher(None, current_key, previous_key).ratio() >= 0.97


def _comparison_key(text: str) -> str:
    """Make a stable key for OCR duplicate checks."""

    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _is_toolbar_noise(line: str) -> bool:
    """Detect a line that is only the selection-box toolbar text."""

    compact = re.sub(r"\s+", "", line)
    return bool(compact) and all(word in _TOOLBAR_WORDS for word in _split_toolbar_words(compact))


def _is_social_metric_noise(line: str) -> bool:
    """Detect common social-video counters accidentally captured near subtitles."""

    compact = re.sub(r"\s+", "", line)
    return bool(compact) and bool(_SOCIAL_METRIC_RE.match(compact))


def _is_search_bar_noise(line: str) -> bool:
    """Detect short search-box labels accidentally captured near the selection."""

    compact = re.sub(r"\s+", "", line)
    return bool(compact) and bool(_SEARCH_BAR_RE.match(compact))


def _is_language_pair_noise(line: str) -> bool:
    """Detect the app's own language-pair label when the translation box is captured."""

    compact = re.sub(r"\s+", "", line)
    return bool(compact) and bool(_LANGUAGE_PAIR_RE.match(compact))


def _is_structural_ui_noise(line: str) -> bool:
    """Reject only standalone UI/resource shapes, not ordinary technical prose.

    These patterns intentionally require the whole OCR line to be a short
    structural token. A sentence such as ``check cache timing`` or ``fix Bug
    123`` therefore remains translatable.
    """

    compact = re.sub(r"\s+", " ", line).strip()
    return bool(
        _STRUCTURAL_LABEL_RE.fullmatch(compact)
        or _STRUCTURAL_RESOURCE_RE.fullmatch(compact)
    )


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
