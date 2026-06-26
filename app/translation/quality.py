"""Local translation output validation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#._/-]*(?:\s+[A-Za-z0-9][A-Za-z0-9+#._/-]*)*")


@dataclass(frozen=True)
class ValidationResult:
    """Result of a local output validation pass."""

    ok: bool
    reason: str = ""


class OutputValidator:
    """Validate model output against locally checkable constraints."""

    @classmethod
    def validate(
        cls,
        output: str,
        *,
        source_text: str,
        system_prompt: str,
        protected_terms: list[str] | None = None,
    ) -> ValidationResult:
        if cls._requires_hiragana(system_prompt):
            return cls._validate_hiragana_output(
                output,
                source_text=source_text,
                protected_terms=protected_terms or [],
            )
        return ValidationResult(ok=True)

    @staticmethod
    def _requires_hiragana(system_prompt: str) -> bool:
        lowered = system_prompt.lower()
        return "平假名" in system_prompt or "hiragana" in lowered

    @classmethod
    def _validate_hiragana_output(
        cls,
        output: str,
        *,
        source_text: str,
        protected_terms: list[str],
    ) -> ValidationResult:
        for term in protected_terms:
            if term and term not in output:
                return ValidationResult(
                    ok=False,
                    reason=f"missing protected term: {term}",
                )

        remaining = output

        allowed_ascii = cls._allowed_ascii_spans(source_text, protected_terms)
        for span in sorted(allowed_ascii, key=len, reverse=True):
            if span:
                remaining = remaining.replace(span, "")

        unexpected_ascii = _ASCII_TOKEN_RE.findall(remaining)
        if unexpected_ascii:
            return ValidationResult(
                ok=False,
                reason=f"unexpected ASCII token: {unexpected_ascii[0]}",
            )

        for ch in remaining:
            if cls._is_allowed_hiragana_char(ch):
                continue
            if cls._is_allowed_punctuation_or_space(ch):
                continue
            return ValidationResult(
                ok=False,
                reason=f"unexpected character U+{ord(ch):04X}: {ch}",
            )

        return ValidationResult(ok=True)

    @staticmethod
    def _allowed_ascii_spans(source_text: str, protected_terms: list[str]) -> set[str]:
        spans = set(protected_terms)
        spans.update(match.group(0) for match in _ASCII_TOKEN_RE.finditer(source_text))
        return {span for span in spans if span}

    @staticmethod
    def _is_allowed_hiragana_char(ch: str) -> bool:
        return "\u3040" <= ch <= "\u309f"

    @staticmethod
    def _is_allowed_punctuation_or_space(ch: str) -> bool:
        if ch.isspace():
            return True
        # Punctuation and symbols are allowed because OCR snippets often
        # contain preserved numbers, commas, brackets, and sentence marks.
        category = unicodedata.category(ch)
        return category.startswith("P") or category.startswith("S")


class OutputNormalizer:
    """Apply deterministic local formatting fixes before model fallback."""

    @classmethod
    def normalize(cls, output: str, *, system_prompt: str) -> str:
        if not OutputValidator._requires_hiragana(system_prompt):
            return output
        return cls._normalize_hiragana_output(output)

    @classmethod
    def _normalize_hiragana_output(cls, output: str) -> str:
        chars: list[str] = []
        for ch in output:
            if cls._is_convertible_katakana(ch):
                chars.append(chr(ord(ch) - 0x60))
                continue
            if ch == "ー":
                chars.append(cls._long_vowel_replacement(chars))
                continue
            chars.append(ch)
        return "".join(chars)

    @staticmethod
    def _is_convertible_katakana(ch: str) -> bool:
        return "\u30a1" <= ch <= "\u30f6"

    @staticmethod
    def _long_vowel_replacement(previous_chars: list[str]) -> str:
        for prev in reversed(previous_chars):
            if "\u30a1" <= prev <= "\u30f6":
                prev = chr(ord(prev) - 0x60)
            if not ("\u3041" <= prev <= "\u3096"):
                continue
            if prev in "あぁかがさざただなはばぱまやゃらわ":
                return "あ"
            if prev in "いぃきぎしじちぢにひびぴみり":
                return "い"
            if prev in "うぅくぐすずつづぬふぶぷむゆゅる":
                return "う"
            if prev in "えぇけげせぜてでねへべぺめれ":
                return "え"
            if prev in "おぉこごそぞとどのほぼぽもよょろを":
                return "う"
        return ""
