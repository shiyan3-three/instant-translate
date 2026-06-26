"""Tests for local translation quality helpers."""

from __future__ import annotations

import unittest

from app.translation.quality import OutputNormalizer, OutputValidator
from app.translation.terms import TermPlaceholder


class TermPlaceholderTests(unittest.TestCase):
    """Verify protected term extraction and restoration."""

    def test_extracts_code_like_terms_without_protecting_all_lowercase_words(self) -> None:
        terms = TermPlaceholder.extract_terms(
            "请保持 Claude Code、OpenCode、Codex、API 和 thinking disabled 不变。"
        )

        self.assertIn("Claude Code", terms)
        self.assertIn("OpenCode", terms)
        self.assertIn("Codex", terms)
        self.assertIn("API", terms)
        self.assertNotIn("thinking disabled", terms)

    def test_extracts_lowercase_technical_terms_without_all_lowercase_phrases(self) -> None:
        terms = TermPlaceholder.extract_terms(
            "单纯靠ai是不行的，因为ocr和api都会有问题，但 ordinary words 不需要保护。"
        )

        self.assertIn("ai", terms)
        self.assertIn("ocr", terms)
        self.assertIn("api", terms)
        self.assertNotIn("ordinary words", terms)

    def test_protect_and_restore_terms(self) -> None:
        payload = TermPlaceholder.protect("Team70r 的价格大概是 70-80r。")

        self.assertIn("⟦0⟧", payload.text)
        self.assertIn("⟦1⟧", payload.text)

        restored = TermPlaceholder.restore(payload.text, payload.placeholders)

        self.assertEqual(restored, "Team70r 的价格大概是 70-80r。")


class OutputValidatorTests(unittest.TestCase):
    """Verify locally checkable format constraints."""

    def test_accepts_hiragana_with_source_ascii_terms(self) -> None:
        result = OutputValidator.validate(
            "Team70r のかかくは 70-80r です。",
            source_text="Team70r 的价格区间大概是 70-80r。",
            system_prompt="输出只能由平假名构成。",
            protected_terms=["Team70r", "70-80r"],
        )

        self.assertTrue(result.ok, result.reason)

    def test_rejects_katakana_and_long_vowel_mark(self) -> None:
        result = OutputValidator.validate(
            "つーる",
            source_text="工具",
            system_prompt="output must be hiragana only",
        )

        self.assertFalse(result.ok)
        self.assertIn("U+30FC", result.reason)

    def test_rejects_unexpected_ascii_token(self) -> None:
        result = OutputValidator.validate(
            "API です",
            source_text="接口",
            system_prompt="输出只能由平假名构成。",
        )

        self.assertFalse(result.ok)
        self.assertIn("unexpected ASCII", result.reason)

    def test_rejects_missing_protected_term(self) -> None:
        result = OutputValidator.validate(
            "ええぴいあい です",
            source_text="保持 API 不变。",
            system_prompt="输出只能由平假名构成。",
            protected_terms=["API"],
        )

        self.assertFalse(result.ok)
        self.assertIn("missing protected term", result.reason)


class OutputNormalizerTests(unittest.TestCase):
    """Verify deterministic script repairs."""

    def test_converts_katakana_and_long_vowel_mark_for_hiragana_prompt(self) -> None:
        normalized = OutputNormalizer.normalize(
            "ツールとショートカットキー",
            system_prompt="输出只能由平假名构成。",
        )

        self.assertEqual(normalized, "つうるとしょうとかっときい")

    def test_leaves_output_unchanged_without_hiragana_constraint(self) -> None:
        normalized = OutputNormalizer.normalize(
            "ツール",
            system_prompt="Translate to Japanese.",
        )

        self.assertEqual(normalized, "ツール")


if __name__ == "__main__":
    unittest.main()
