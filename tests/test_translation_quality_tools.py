"""Tests for local translation quality helpers."""

from __future__ import annotations

import unittest

from app.translation.quality import OutputNormalizer, OutputValidator
from app.translation.terms import TermPlaceholder
from app.prompt.policy import ConstraintPolicy, ConstraintPolicyCompiler


def _hiragana_term_policy() -> ConstraintPolicy:
    return ConstraintPolicy.from_dict(
        {
            "version": 1,
            "rules": [
                {
                    "type": "allowed_characters",
                    "params": {
                        "scripts": ["hiragana"],
                        "literals": ["[", "]"],
                        "allow_whitespace": True,
                    },
                    "enforcement": "both",
                    "scope": {},
                },
                {
                    "type": "term_wrapper",
                    "params": {"left": "[", "right": "]", "domain": "software_engineering"},
                    "enforcement": "both",
                    "scope": {},
                },
            ],
        },
        strict=True,
    )


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

    def test_pure_numeric_expressions_are_not_protected(self) -> None:
        """Numbers without letters are not technical tokens."""
        for source, snippet in [
            ("第42课", "42"),
            ("3个小时", "3"),
            ("价格4.0元", "4.0"),
            ("70-80分", "70-80"),
            ("日期2026-07-03", "2026-07-03"),
        ]:
            terms = TermPlaceholder.extract_terms(source)
            self.assertNotIn(
                snippet,
                terms,
                f"{snippet!r} should not be treated as a technical term in {source!r}",
            )

    def test_mixed_alphanumeric_technical_identifiers_are_protected(self) -> None:
        """Letter-bearing identifiers with digits are still technical tokens."""
        terms = TermPlaceholder.extract_terms(
            "请保持 Python3、HTTP/2、v2.1 和 CUDA12 不变。"
        )
        self.assertIn("Python3", terms)
        self.assertIn("HTTP/2", terms)
        self.assertIn("v2.1", terms)
        self.assertIn("CUDA12", terms)

    def test_prompt_terms_include_user_reference_terms_without_builtin_domain_vocabulary(self) -> None:
        source = "这个接口部署在服务器上,请检查页面。"
        reference = (
            "| Chinese Term | Output |\n"
            "|--------------|--------|\n"
            "| 接口 | [いんたーふぇーす] |\n"
            "| 服务器 | [さーばー] |\n"
        )

        protected_terms = TermPlaceholder.extract_terms(source)
        prompt_terms = TermPlaceholder.extract_prompt_terms(source, reference_text=reference)

        self.assertEqual(protected_terms, [])
        self.assertIn("接口", prompt_terms)
        self.assertIn("服务器", prompt_terms)
        self.assertNotIn("页面", prompt_terms)

    def test_prompt_terms_parse_user_arrow_glossary_lines(self) -> None:
        prompt_terms = TermPlaceholder.extract_prompt_terms(
            "请修复这个配置。",
            reference_text="- 配置 -> [せってい]",
        )

        self.assertEqual(prompt_terms, ["配置"])


class OutputValidatorTests(unittest.TestCase):
    """Verify locally checkable format constraints."""

    def test_rejects_source_ascii_when_hiragana_is_a_hard_requirement(self) -> None:
        result = OutputValidator.validate(
            "Team70r のかかくは 70-80r です。",
            source_text="Team70r 的价格区间大概是 70-80r。",
            system_prompt="输出只能由平假名构成。",
            protected_terms=["Team70r", "70-80r"],
        )

        self.assertFalse(result.ok)
        self.assertIn("allowed_characters", result.reason)

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
        self.assertIn("allowed_characters", result.reason)

    def test_rejects_missing_protected_term(self) -> None:
        result = OutputValidator.validate(
            "ええぴいあい です",
            source_text="保持 API 不变。",
            system_prompt="输出只能由平假名构成。",
            protected_terms=["API"],
        )

        self.assertFalse(result.ok)
        self.assertIn("missing protected term", result.reason)

    def test_accepts_wrapped_hiragana_technical_term_and_double_spaces(self) -> None:
        prompt = (
            "如果中文翻译为日语时只能由平假名构成，"
            "软件工程术语使用[]包裹，每个单词之间至少两个空格。"
        )
        policy = ConstraintPolicyCompiler.compile(prompt)

        result = OutputValidator.validate(
            "この  [ええぴいあい]  は  かんせいした",
            source_text="这个API已经完成",
            system_prompt=prompt,
            policy=policy,
            technical_terms=["API"],
        )

        self.assertTrue(result.ok, result.reason)

    def test_accepts_wrapped_hiragana_technical_term_with_internal_spaces(self) -> None:
        result = OutputValidator.validate(
            "\u3053\u306e [\u3070\u3063\u304f\u3048\u3093\u3069  \u305f\u3059\u304f] \u3067\u3059",
            source_text="backend task",
            system_prompt="Translate Chinese to Japanese.",
            policy=_hiragana_term_policy(),
            technical_terms=["backend task"],
        )

        self.assertTrue(result.ok, result.reason)

    def test_term_wrapper_rejects_raw_technical_term_left_outside_brackets(self) -> None:
        policy = ConstraintPolicy.from_dict(
            {
                "version": 1,
                "rules": [
                    {
                        "type": "term_wrapper",
                        "params": {"left": "[", "right": "]"},
                        "enforcement": "both",
                        "scope": {},
                        "source_text": "technical terms use brackets",
                    }
                ],
            },
            strict=True,
        )

        result = OutputValidator.validate(
            "API [ばっくえんど]",
            source_text="这个API由后端完成",
            system_prompt="Translate Chinese to Japanese.",
            policy=policy,
            technical_terms=["API"],
        )

        self.assertFalse(result.ok)
        self.assertIn("left unwrapped: API", result.reason)

    def test_strict_term_wrapper_rejects_extra_generated_cjk_term(self) -> None:
        policy = ConstraintPolicy.from_dict(
            {
                "version": 1,
                "rules": [
                    {
                        "type": "term_wrapper",
                        "params": {
                            "left": "[",
                            "right": "]",
                            "selection_mode": "references_only",
                        },
                        "enforcement": "both",
                        "scope": {},
                    }
                ],
            },
            strict=True,
        )

        result = OutputValidator.validate(
            "これは [ばっくぐらうんどたすく] です",
            source_text="后台任务还没有完全恢复。",
            system_prompt="Translate Chinese to Japanese.",
            policy=policy,
            expected_wrapped_targets=[],
            extra_wrapped_term_budget=0,
        )

        self.assertFalse(result.ok)
        self.assertIn("unexpected generated wrapped term", result.reason)

    def test_strict_term_wrapper_accepts_confirmed_reference_target_only(self) -> None:
        policy = ConstraintPolicy.from_dict(
            {
                "version": 1,
                "rules": [
                    {
                        "type": "term_wrapper",
                        "params": {
                            "left": "[",
                            "right": "]",
                            "selection_mode": "references_only",
                        },
                        "enforcement": "both",
                        "scope": {},
                    }
                ],
            },
            strict=True,
        )

        result = OutputValidator.validate(
            "この [いんたあふぇえす] は かんせいした",
            source_text="这个接口已经完成。",
            system_prompt="Translate Chinese to Japanese.",
            policy=policy,
            expected_wrapped_targets=["[いんたあふぇえす]"],
            extra_wrapped_term_budget=0,
        )

        self.assertTrue(result.ok, result.reason)

    def test_scoped_separator_checks_only_between_bracketed_terms(self) -> None:
        policy = ConstraintPolicy.from_dict(
            {
                "version": 1,
                "rules": [
                    {
                        "type": "separator",
                        "params": {
                            "min_spaces": 2,
                            "normalize_to": 2,
                            "scope": "between_bracketed_terms",
                        },
                        "enforcement": "both",
                        "scope": {},
                    }
                ],
            },
            strict=True,
        )

        bad = OutputValidator.validate(
            "[ええ] [びい] と すすめる",
            source_text="A和B",
            system_prompt="Translate Chinese to Japanese.",
            policy=policy,
        )
        good = OutputValidator.validate(
            "[ええ]  [びい] と すすめる",
            source_text="A和B",
            system_prompt="Translate Chinese to Japanese.",
            policy=policy,
        )

        self.assertFalse(bad.ok)
        self.assertIn("between bracketed terms", bad.reason)
        self.assertTrue(good.ok, good.reason)


class OutputNormalizerTests(unittest.TestCase):
    """Verify deterministic script repairs."""

    def test_converts_katakana_and_long_vowel_mark_for_hiragana_prompt(self) -> None:
        normalized = OutputNormalizer.normalize(
            "ツールとショートカットキー",
            system_prompt="输出只能由平假名构成。",
        )

        self.assertEqual(normalized, "つうるとしょおとかっときい")

    def test_converts_ascii_numbers_for_hiragana_prompt_before_validation(self) -> None:
        normalized = OutputNormalizer.normalize(
            "ふか は 16 で 0.8",
            system_prompt="输出只能由平假名构成。",
        )

        self.assertEqual(normalized, "ふか は じゅうろく で れいてんはち")
        self.assertNotRegex(normalized, r"\d")
        result = OutputValidator.validate(
            normalized,
            source_text="平均负载是16和0.8。",
            system_prompt="输出只能由平假名构成。",
        )
        self.assertTrue(result.ok, result.reason)

    def test_policy_normalizes_punctuation_and_space_width(self) -> None:
        prompt = (
            "如果中文翻译为日语时只能由平假名构成，"
            "软件工程术语使用[]包裹，每个单词之间至少两个空格。"
        )
        policy = ConstraintPolicyCompiler.compile(prompt)

        normalized = OutputNormalizer.normalize_with_policy(
            "この API は、[バックエンド] チームです。",
            system_prompt=prompt,
            policy=policy,
        )

        self.assertEqual(normalized, "この  API  は  [ばっくえんど]  ちいむです")

    def test_normalizes_known_traditional_variants_for_chinese_target(self) -> None:
        normalized = OutputNormalizer.normalize(
            "請確認這個漢字。",
            system_prompt="Translate from Japanese to 中文.",
        )

        self.assertEqual(normalized, "请确认这个汉字。")

    def test_does_not_simplify_variants_for_non_chinese_target(self) -> None:
        normalized = OutputNormalizer.normalize(
            "漢字",
            system_prompt="Translate from Chinese to Japanese.",
        )

        self.assertEqual(normalized, "漢字")

    def test_scoped_separator_normalizes_only_between_bracketed_terms(self) -> None:
        policy = ConstraintPolicy.from_dict(
            {
                "version": 1,
                "rules": [
                    {
                        "type": "separator",
                        "params": {
                            "min_spaces": 2,
                            "normalize_to": 2,
                            "scope": "between_bracketed_terms",
                        },
                        "enforcement": "both",
                        "scope": {},
                    }
                ],
            },
            strict=True,
        )

        normalized = OutputNormalizer.normalize_with_policy(
            "[ええ] [びい] と すすめる",
            system_prompt="Translate Chinese to Japanese.",
            policy=policy,
        )

        self.assertEqual(normalized, "[ええ]  [びい] と すすめる")

    def test_normalizes_spaces_inside_hiragana_wrapped_terms(self) -> None:
        normalized = OutputNormalizer.normalize_with_policy(
            "\u3053\u306e [\u30d0\u30c3\u30af\u30a8\u30f3\u30c9  \u30bf\u30b9\u30af] \u3067\u3059",
            system_prompt="Translate Chinese to Japanese.",
            policy=_hiragana_term_policy(),
        )

        self.assertEqual(
            normalized,
            "\u3053\u306e [\u3070\u3063\u304f\u3048\u3093\u3069\u305f\u3059\u304f] \u3067\u3059",
        )

    def test_leaves_output_unchanged_without_hiragana_constraint(self) -> None:
        normalized = OutputNormalizer.normalize(
            "ツール",
            system_prompt="Translate to Japanese.",
        )

        self.assertEqual(normalized, "ツール")


if __name__ == "__main__":
    unittest.main()
