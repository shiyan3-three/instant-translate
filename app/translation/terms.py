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

    @staticmethod
    def empty(text: str) -> PlaceholderPayload:
        """Return an unchanged payload when verbatim preservation is disallowed."""

        return PlaceholderPayload(text=text, placeholders={})

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

    @classmethod
    def extract_prompt_terms(cls, text: str, *, reference_text: str = "") -> list[str]:
        """Return source terms worth showing to the translator prompt.

        This combines code-like source tokens with terms explicitly provided
        by the user prompt/profile, such as glossary table entries.  It does
        not rely on a built-in Chinese domain vocabulary; unknown domain terms
        are left to the model-side term inference instruction.
        """

        if not cls._contains_cjk(text):
            return []

        seen: set[str] = set()
        terms: list[str] = []
        for term in cls.extract_terms(text):
            seen.add(term)
            terms.append(term)
        for term in cls.extract_reference_terms(reference_text):
            if term in text and term not in seen:
                seen.add(term)
                terms.append(term)

        terms.sort(key=len, reverse=True)
        return terms

    @classmethod
    def extract_reference_terms(cls, reference_text: str) -> list[str]:
        """Extract user-specified source glossary terms from prompt text."""

        seen: set[str] = set()
        terms: list[str] = []
        for line in str(reference_text or "").splitlines():
            for candidate in cls._reference_terms_from_line(line):
                if candidate not in seen:
                    seen.add(candidate)
                    terms.append(candidate)
        terms.sort(key=len, reverse=True)
        return terms

    @classmethod
    def _reference_terms_from_line(cls, line: str) -> list[str]:
        stripped = line.strip()
        if not stripped:
            return []

        terms: list[str] = []
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [
                cell.strip().strip("`")
                for cell in stripped.strip("|").split("|")
            ]
            if len(cells) >= 2 and cls._looks_like_reference_source_term(cells[0]):
                terms.append(cells[0])

        arrow_match = re.match(
            r"^[\-*]?\s*([A-Za-z0-9+#._/\-\u4e00-\u9fff ]{1,40})\s*(?:->|=>|:|：)\s*\S+",
            stripped,
        )
        if arrow_match:
            candidate = arrow_match.group(1).strip().strip("`")
            if cls._looks_like_reference_source_term(candidate):
                terms.append(candidate)
        return terms

    @classmethod
    def _looks_like_reference_source_term(cls, value: str) -> bool:
        candidate = value.strip()
        if not candidate or len(candidate) > 40:
            return False
        lowered = candidate.casefold()
        if lowered in {"chinese term", "source", "input", "term", "原文", "源词", "术语"}:
            return False
        if set(candidate) <= {"-", "—", " ", ":"}:
            return False
        return cls._contains_cjk(candidate) or cls._should_protect(candidate)

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    @staticmethod
    def _should_protect(candidate: str) -> bool:
        compact = candidate.replace(" ", "")
        if not compact:
            return False
        # Pure-numeric expressions (42, 4.0, 70-80, 2026-07-03) are not
        # technical tokens; the model should translate numbers normally
        # without wrapping them in term brackets.
        if not any(ch.isalpha() for ch in compact):
            return False
        if re.fullmatch(r"REF_\d+", compact):
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
