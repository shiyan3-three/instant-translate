"""Local protected-term placeholder helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass


_ASCII_RUN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z0-9][A-Za-z0-9+#._/-]*"
    r"(?:\s+[A-Za-z0-9][A-Za-z0-9+#._/-]*)*"
    r"(?![A-Za-z0-9_])"
)

_LOWERCASE_TECH_TERMS = {
    "ai",
    "api",
    "ocr",
    "ui",
    "ux",
    "json",
    "xml",
    "html",
    "css",
    "js",
    "ts",
    "sql",
    "http",
    "https",
    "url",
    "id",
}


@dataclass(frozen=True)
class PlaceholderPayload:
    """Text with protected terms replaced by placeholders."""

    text: str
    placeholders: dict[str, str]

    @property
    def has_placeholders(self) -> bool:
        return bool(self.placeholders)


class TermPlaceholder:
    """Protect explicit ASCII/code-like source terms from model rewriting."""

    LEFT = "⟦"
    RIGHT = "⟧"

    @classmethod
    def protect(cls, text: str) -> PlaceholderPayload:
        terms = cls.extract_terms(text)
        if not terms:
            return PlaceholderPayload(text=text, placeholders={})

        protected = text
        placeholders: dict[str, str] = {}
        for index, term in enumerate(terms):
            placeholder = f"{cls.LEFT}{index}{cls.RIGHT}"
            protected = protected.replace(term, placeholder)
            placeholders[placeholder] = term

        return PlaceholderPayload(text=protected, placeholders=placeholders)

    @staticmethod
    def restore(text: str, placeholders: dict[str, str]) -> str:
        restored = text
        for placeholder, term in placeholders.items():
            restored = restored.replace(placeholder, term)
        return restored

    @classmethod
    def extract_terms(cls, text: str) -> list[str]:
        """Return source terms that are likely user-visible technical tokens.

        This intentionally avoids protecting every lowercase English word.
        It focuses on code-like, acronym, camel-case, digit-bearing, or
        title-case multi-word terms such as ``API``, ``Team70r``,
        ``PolicyCompiler``, and ``Pull Request``.
        """

        if not cls._contains_cjk(text):
            return []

        seen: set[str] = set()
        terms: list[str] = []
        for match in _ASCII_RUN_RE.finditer(text):
            candidate = match.group(0).strip()
            if not candidate or candidate in seen:
                continue
            if cls._should_protect(candidate):
                seen.add(candidate)
                terms.append(candidate)

        terms.sort(key=len, reverse=True)
        return terms

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    @staticmethod
    def _should_protect(candidate: str) -> bool:
        compact = candidate.replace(" ", "")
        if not compact:
            return False
        if any(ch.isdigit() for ch in compact):
            return True
        if any(ch in compact for ch in "#+._/-"):
            return True
        if any(ch.isupper() for ch in compact):
            return True
        if compact.lower() in _LOWERCASE_TECH_TERMS:
            return True
        parts = candidate.split()
        return len(parts) > 1 and all(part[:1].isupper() for part in parts)
