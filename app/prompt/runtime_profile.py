"""Confirmed deterministic runtime profile for low-latency translation.

The runtime profile is intentionally narrower than the compiled prompt.  It is
allowed to carry draft semantic/style preferences, but the current confirmed
contract only permits deterministic completeness checks.  It never carries
local format policy or glossary/reference data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


MAX_DIRECTIVES_PER_KIND = 16
MAX_DIRECTIVE_LENGTH = 240
MAX_LANGUAGE_LENGTH = 64
PROFILE_VERSION = 1
_ALLOWED_FIELDS = frozenset(
    {
        "version",
        "source_language",
        "target_language",
        "semantic_directives",
        "style_directives",
        "completeness_checks",
        "prompt_digest",
        "policy_digest",
        "reference_digest",
        "confirmed",
        "source",
    }
)
_RUNTIME_TAG_RE = re.compile(
    r"</?(?:runtime_profile|ocr_text|rule_checklist)",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


_FORMAT_OR_REFERENCE_MARKERS = tuple(
    item.casefold()
    for item in (
        "allowed_characters",
        "term_wrapper",
        "literal_replace",
        "forbidden_literals",
        "separator",
        "punctuation",
        "hiragana",
        "katakana",
        "kanji",
        "square bracket",
        "bracketed term",
        "glossary",
        "fixed reading",
        "term reading",
        "terminology list",
        "source=>target",
        "source => target",
        "source -> target",
        "=>",
        "->",
        "[]",
        "\u5e73\u5047\u540d",  # hiragana
        "\u7247\u5047\u540d",  # katakana
        "\u6c49\u5b57",
        "\u6f22\u5b57",
        "\u65b9\u62ec\u53f7",
        "\u7a7a\u683c",
        "\u672f\u8bed",
        "\u8a9e\u5f59",
        "\u8bfb\u6cd5",
    )
)


def digest_text(value: str) -> str:
    """Return a stable SHA-256 digest for arbitrary runtime-profile inputs."""

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def context_digest(*parts: str) -> str:
    """Hash the prompt/policy/reference context a profile was confirmed for."""

    payload = json.dumps([str(part) for part in parts], ensure_ascii=False, separators=(",", ":"))
    return digest_text(payload)


@dataclass(frozen=True)
class RuntimeProfile:
    """User-confirmed semantic/style preferences projected into runtime."""

    version: int = PROFILE_VERSION
    source_language: str = ""
    target_language: str = ""
    semantic_directives: tuple[str, ...] = ()
    style_directives: tuple[str, ...] = ()
    completeness_checks: tuple[str, ...] = ()
    prompt_digest: str = ""
    policy_digest: str = ""
    reference_digest: str = ""
    confirmed: bool = False
    source: str = "user_confirmed"

    def __post_init__(self) -> None:
        """Keep direct construction from bypassing the confirmed type boundary."""

        if not isinstance(self.confirmed, bool):
            raise ValueError("confirmed must be a boolean")

    @property
    def is_empty(self) -> bool:
        return not (
            self.semantic_directives
            or self.style_directives
            or self.completeness_checks
        )

    @property
    def is_usable(self) -> bool:
        return (
            self.confirmed is True
            and bool(self.source_language)
            and bool(self.target_language)
            and bool(self.completeness_checks)
            and not self.semantic_directives
            and not self.style_directives
            and bool(_SHA256_RE.fullmatch(self.prompt_digest))
            and bool(_SHA256_RE.fullmatch(self.policy_digest))
            and bool(_SHA256_RE.fullmatch(self.reference_digest))
        )

    def matches_language_pair(self, source_language: str, target_language: str) -> bool:
        if not self.source_language or not self.target_language:
            return False
        return (
            _language_key(self.source_language) == _language_key(source_language)
            and _language_key(self.target_language) == _language_key(target_language)
        )

    def matches_context(
        self,
        *,
        prompt_digest: str = "",
        policy_digest: str = "",
        reference_digest: str = "",
    ) -> bool:
        """Require all confirmed context digests to match exactly."""

        if not self.is_usable:
            return False
        return (
            self.prompt_digest == prompt_digest
            and self.policy_digest == policy_digest
            and self.reference_digest == reference_digest
        )

    def render(self) -> str:
        """Render a compact block suitable for future prompt injection."""

        if not self.is_usable:
            return ""
        lines = [
            "<RUNTIME_PROFILE>",
            f"Language pair: {self.source_language} -> {self.target_language}",
        ]
        _append_section(lines, "Completeness checks", self.completeness_checks)
        lines.append("</RUNTIME_PROFILE>")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "semantic_directives": list(self.semantic_directives),
            "style_directives": list(self.style_directives),
            "completeness_checks": list(self.completeness_checks),
            "prompt_digest": self.prompt_digest,
            "policy_digest": self.policy_digest,
            "reference_digest": self.reference_digest,
            "confirmed": self.confirmed,
            "source": self.source,
        }

    def digest(self) -> str:
        raw = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return digest_text(raw)

    @classmethod
    def from_dict(cls, raw: Any, *, strict: bool = False) -> RuntimeProfile:
        if not isinstance(raw, dict):
            if strict:
                raise ValueError("runtime profile must be an object")
            return cls()
        if strict:
            unknown = sorted(set(raw) - _ALLOWED_FIELDS)
            if unknown:
                raise ValueError(f"unknown runtime profile fields: {', '.join(unknown)}")
        if raw.get("version", PROFILE_VERSION) != PROFILE_VERSION:
            if strict:
                raise ValueError("unsupported runtime profile version")
            return cls()

        try:
            confirmed_raw = raw.get("confirmed", False)
            if not isinstance(confirmed_raw, bool):
                if strict:
                    raise ValueError("confirmed must be a boolean")
                confirmed_raw = False
            profile = cls(
                version=PROFILE_VERSION,
                source_language=_clean_text(
                    raw.get("source_language", ""),
                    field="source_language",
                    limit=MAX_LANGUAGE_LENGTH,
                ),
                target_language=_clean_text(
                    raw.get("target_language", ""),
                    field="target_language",
                    limit=MAX_LANGUAGE_LENGTH,
                ),
                semantic_directives=_directive_list(
                    raw.get("semantic_directives", []),
                    field="semantic_directives",
                    strict=strict,
                ),
                style_directives=_directive_list(
                    raw.get("style_directives", []),
                    field="style_directives",
                    strict=strict,
                ),
                completeness_checks=_directive_list(
                    raw.get("completeness_checks", []),
                    field="completeness_checks",
                    strict=strict,
                ),
                prompt_digest=_clean_digest(raw.get("prompt_digest", "")),
                policy_digest=_clean_digest(raw.get("policy_digest", "")),
                reference_digest=_clean_digest(raw.get("reference_digest", "")),
                confirmed=confirmed_raw,
                source=_clean_text(raw.get("source", "user_confirmed"), field="source", limit=64)
                or "user_confirmed",
            )
        except ValueError:
            if strict:
                raise
            return cls()

        if profile.confirmed:
            confirmed_error = _confirmed_profile_error(profile)
            if confirmed_error:
                if strict:
                    raise ValueError(confirmed_error)
                return cls()
        return profile


def _append_section(lines: list[str], title: str, values: Iterable[str]) -> None:
    values = tuple(values)
    if not values:
        return
    lines.append(f"{title}:")
    lines.extend(f"- {item}" for item in values)


def _directive_list(raw: Any, *, field: str, strict: bool) -> tuple[str, ...]:
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, tuple) and not strict:
        rows = list(raw)
    else:
        if strict:
            raise ValueError(f"{field} must be a list")
        return ()

    if len(rows) > MAX_DIRECTIVES_PER_KIND:
        if strict:
            raise ValueError(f"{field} has too many directives")
        rows = rows[:MAX_DIRECTIVES_PER_KIND]

    result: list[str] = []
    for item in rows:
        if not isinstance(item, str):
            if strict:
                raise ValueError(f"{field} items must be non-empty strings")
            continue
        try:
            text = _clean_text(item, field=field, limit=MAX_DIRECTIVE_LENGTH)
        except ValueError:
            if strict:
                raise
            continue
        if not text:
            if strict:
                raise ValueError(f"{field} items must be non-empty strings")
            continue
        if _looks_like_format_or_reference_rule(text) or _looks_like_runtime_tag(text):
            if strict:
                raise ValueError(
                    f"{field} must not contain format, reference, or runtime-tag rules"
                )
            continue
        if text not in result:
            result.append(text)
    return tuple(result)


def _clean_text(value: Any, *, field: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        raise ValueError(f"{field} is too long")
    return text


def _clean_digest(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("runtime profile digest must be a string")
    text = value.strip()
    if not text:
        return ""
    if not _SHA256_RE.fullmatch(text):
        raise ValueError("runtime profile digest must be 64-character lowercase SHA-256 hex")
    return text


def _looks_like_format_or_reference_rule(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in _FORMAT_OR_REFERENCE_MARKERS)


def _looks_like_runtime_tag(text: str) -> bool:
    return bool(_RUNTIME_TAG_RE.search(text))


def _confirmed_profile_error(profile: RuntimeProfile) -> str:
    if not profile.source_language or not profile.target_language:
        return "confirmed runtime profile requires source and target languages"
    if not profile.prompt_digest or not profile.policy_digest or not profile.reference_digest:
        return "confirmed runtime profile requires all three context digests"
    if profile.semantic_directives or profile.style_directives:
        return "confirmed runtime profile currently permits completeness_checks only"
    if not profile.completeness_checks:
        return "confirmed runtime profile requires at least one completeness check"
    return ""


def _language_key(value: str) -> str:
    text = str(value).strip().casefold()
    aliases = {
        "chinese": "chinese",
        "\u4e2d\u6587": "chinese",
        "\u6c49\u8bed": "chinese",
        "\u6f22\u8a9e": "chinese",
        "japanese": "japanese",
        "\u65e5\u8bed": "japanese",
        "\u65e5\u672c\u8a9e": "japanese",
        "\u65e5\u672c\u8bed": "japanese",
        "english": "english",
        "\u82f1\u8bed": "english",
        "\u82f1\u8a9e": "english",
    }
    return aliases.get(text, text)
