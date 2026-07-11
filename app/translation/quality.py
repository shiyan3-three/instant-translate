"""Local translation output validation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from app.prompt.policy import ConstraintPolicy, ConstraintPolicyCompiler


_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#._/-]*(?:\s+[A-Za-z0-9][A-Za-z0-9+#._/-]*)*")

_SIMPLIFIED_VARIANT_MAP = str.maketrans(
    {
        "請": "请",
        "漢": "汉",
        "無": "无",
        "裏": "里",
        "為": "为",
        "個": "个",
        "數": "数",
        "據": "据",
        "這": "这",
        "後": "后",
        "過": "过",
        "開": "开",
        "發": "发",
        "檔": "档",
        "確": "确",
        "認": "认",
        "體": "体",
        "應": "应",
        "異": "异",
        "權": "权",
        "錄": "录",
    }
)


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
        policy: ConstraintPolicy | None = None,
        technical_terms: list[str] | None = None,
        expected_wrapped_targets: list[str] | None = None,
        extra_wrapped_term_budget: int | None = None,
    ) -> ValidationResult:
        effective = cls._effective_policy(system_prompt, policy)
        protected = protected_terms or []
        for term in protected:
            if term and term not in output:
                return ValidationResult(ok=False, reason=f"missing protected term: {term}")

        result = cls._validate_policy(
            output,
            source_text=source_text,
            policy=effective,
            technical_terms=technical_terms or [],
            expected_wrapped_targets=expected_wrapped_targets,
            extra_wrapped_term_budget=extra_wrapped_term_budget,
        )
        if not result.ok:
            return result

        # Backward-compatible leakage check for legacy hiragana prompts that
        # predate policy sidecars.
        if cls._requires_hiragana(system_prompt, policy=effective):
            remaining = output
            for span in sorted(protected, key=len, reverse=True):
                if span:
                    remaining = remaining.replace(span, "")
            unexpected_ascii = _ASCII_TOKEN_RE.findall(remaining)
            if unexpected_ascii:
                return ValidationResult(
                    ok=False,
                    reason=f"unexpected ASCII token: {unexpected_ascii[0]}",
                )
        return ValidationResult(ok=True)

    @staticmethod
    def _requires_hiragana(
        system_prompt: str,
        *,
        policy: ConstraintPolicy | None = None,
    ) -> bool:
        if policy is not None and policy.has_script("hiragana"):
            return True
        lowered = system_prompt.lower()
        return "平假名" in system_prompt or "hiragana" in lowered

    @classmethod
    def _validate_policy(
        cls,
        output: str,
        *,
        source_text: str,
        policy: ConstraintPolicy,
        technical_terms: list[str],
        expected_wrapped_targets: list[str] | None,
        extra_wrapped_term_budget: int | None,
    ) -> ValidationResult:
        for rule in policy.rules_of_type("allowed_characters", local_only=True):
            scripts = {str(item).casefold() for item in rule.params.get("scripts", [])}
            literals = {str(item) for item in rule.params.get("literals", [])}
            allow_whitespace = bool(rule.params.get("allow_whitespace", False))
            for ch in output:
                if ch in literals:
                    continue
                if allow_whitespace and ch.isspace():
                    continue
                if any(cls._char_in_script(ch, script) for script in scripts):
                    continue
                return ValidationResult(
                    ok=False,
                    reason=f"allowed_characters: unexpected U+{ord(ch):04X} {ch}",
                )

        for rule in policy.rules_of_type("separator", local_only=True):
            minimum = max(0, int(rule.params.get("min_spaces", 0) or 0))
            separator_scope = str(rule.params.get("scope", "global"))
            if output != output.strip():
                return ValidationResult(ok=False, reason="separator: edge whitespace")
            if minimum:
                if separator_scope == "between_bracketed_terms":
                    left = str(rule.params.get("left", "["))
                    right = str(rule.params.get("right", "]"))
                    pattern = re.escape(right) + r"( *)" + re.escape(left)
                    for run in re.finditer(pattern, output):
                        if len(run.group(1)) < minimum:
                            return ValidationResult(
                                ok=False,
                                reason=f"separator: expected at least {minimum} spaces between bracketed terms",
                            )
                else:
                    for run in re.finditer(r" +", output):
                        if len(run.group(0)) < minimum:
                            return ValidationResult(
                                ok=False,
                                reason=f"separator: expected at least {minimum} spaces",
                            )

        for rule in policy.rules_of_type("term_wrapper", local_only=True):
            left = str(rule.params.get("left", "["))
            right = str(rule.params.get("right", "]"))
            if not left or not right:
                continue
            if output.count(left) != output.count(right):
                return ValidationResult(ok=False, reason="term_wrapper: unbalanced delimiters")
            pattern = re.escape(left) + r"([^" + re.escape(left + right) + r"]+)" + re.escape(right)
            wrapped = re.findall(pattern, output)
            if any(not value.strip() for value in wrapped):
                return ValidationResult(ok=False, reason="term_wrapper: empty wrapped term")
            if policy.has_script("hiragana") and any(
                not all(cls._is_allowed_wrapped_hiragana_char(ch) for ch in value)
                for value in wrapped
            ):
                return ValidationResult(ok=False, reason="term_wrapper: non-hiragana content")
            if extra_wrapped_term_budget is not None:
                expected_wrapped = cls._wrapped_terms_from_targets(
                    expected_wrapped_targets or [],
                    left=left,
                    right=right,
                )
                expected_counts = Counter(expected_wrapped)
                actual_counts = Counter(wrapped)
                unexpected_count = sum((actual_counts - expected_counts).values())
                allowed_extra = max(0, int(extra_wrapped_term_budget))
                if unexpected_count > allowed_extra:
                    return ValidationResult(
                        ok=False,
                        reason=(
                            "term_wrapper: unexpected generated wrapped term "
                            f"(allowed extra={allowed_extra}, actual extra={unexpected_count})"
                        ),
                    )
                minimum_wrapped = len(expected_wrapped) + allowed_extra
                if len(wrapped) < minimum_wrapped:
                    return ValidationResult(
                        ok=False,
                        reason=(
                            "term_wrapper: expected wrapped references and source technical terms "
                            f"(expected at least {minimum_wrapped}, actual={len(wrapped)})"
                        ),
                    )
            unwrapped_output = cls._remove_wrapped_segments(output, left, right)
            for term in technical_terms:
                if term and term in unwrapped_output:
                    return ValidationResult(
                        ok=False,
                        reason=f"term_wrapper: source technical term was left unwrapped: {term}",
                    )
            if technical_terms and len(wrapped) < len(technical_terms):
                return ValidationResult(
                    ok=False,
                    reason="term_wrapper: source technical term was not wrapped",
                )

        for rule in policy.rules_of_type("max_length", local_only=True):
            maximum = int(rule.params.get("characters", 0) or 0)
            if maximum > 0 and len(output) > maximum:
                return ValidationResult(ok=False, reason=f"max_length: exceeds {maximum}")

        for rule in policy.rules_of_type("preserve", local_only=True):
            for literal in rule.params.get("values", []):
                text = str(literal)
                if text and text in source_text and text not in output:
                    return ValidationResult(ok=False, reason=f"preserve: missing {text}")

        for rule in policy.rules_of_type("line_breaks", local_only=True):
            mode = str(rule.params.get("mode", "preserve"))
            if mode == "preserve" and output.count("\n") != source_text.count("\n"):
                return ValidationResult(ok=False, reason="line_breaks: source layout changed")
            if mode == "strip" and "\n" in output:
                return ValidationResult(ok=False, reason="line_breaks: newline is forbidden")
            if mode == "single" and "\n\n" in output:
                return ValidationResult(ok=False, reason="line_breaks: repeated newline")

        for rule in policy.rules_of_type("case", local_only=True):
            mode = str(rule.params.get("mode", "preserve"))
            if mode == "lower" and output != output.lower():
                return ValidationResult(ok=False, reason="case: expected lowercase")
            if mode == "upper" and output != output.upper():
                return ValidationResult(ok=False, reason="case: expected uppercase")

        for rule in policy.rules_of_type("forbidden_literals", local_only=True):
            for literal in rule.params.get("values", []):
                text = str(literal)
                if text and text in output:
                    return ValidationResult(ok=False, reason=f"forbidden literal: {text}")

        for rule in policy.rules_of_type("literal_replace", local_only=True):
            source = str(rule.params.get("source", ""))
            target = str(rule.params.get("target", ""))
            if source and source in source_text and target and target not in output:
                return ValidationResult(ok=False, reason=f"literal_replace: missing {target}")

        return ValidationResult(ok=True)

    @staticmethod
    def _wrapped_terms_from_targets(
        targets: list[str],
        *,
        left: str,
        right: str,
    ) -> list[str]:
        if not left or not right:
            return []
        pattern = re.escape(left) + r"([^" + re.escape(left + right) + r"]+)" + re.escape(right)
        terms: list[str] = []
        for target in targets:
            terms.extend(re.findall(pattern, str(target)))
        return terms

    @staticmethod
    def _effective_policy(
        system_prompt: str,
        policy: ConstraintPolicy | None,
    ) -> ConstraintPolicy:
        if policy is not None and policy.rules:
            return policy
        return ConstraintPolicyCompiler.compile(system_prompt)

    @staticmethod
    def _char_in_script(ch: str, script: str) -> bool:
        if script == "hiragana":
            return "\u3041" <= ch <= "\u3096"
        if script == "katakana":
            return "\u30a1" <= ch <= "\u30fa" or ch == "ー"
        if script == "han":
            return "\u3400" <= ch <= "\u9fff"
        if script == "latin":
            return ch.isascii() and ch.isalpha()
        if script == "digits":
            return ch.isdigit()
        return False

    @staticmethod
    def _allowed_ascii_spans(source_text: str, protected_terms: list[str]) -> set[str]:
        spans = set(protected_terms)
        spans.update(match.group(0) for match in _ASCII_TOKEN_RE.finditer(source_text))
        return {span for span in spans if span}

    @staticmethod
    def _remove_wrapped_segments(output: str, left: str, right: str) -> str:
        pattern = re.escape(left) + r".*?" + re.escape(right)
        return re.sub(pattern, "", output)

    @staticmethod
    def _is_allowed_hiragana_char(ch: str) -> bool:
        return "\u3040" <= ch <= "\u309f"

    @staticmethod
    def _is_allowed_wrapped_hiragana_char(ch: str) -> bool:
        return ch.isspace() or OutputValidator._is_allowed_hiragana_char(ch)

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

    _DIGIT_HIRAGANA = {
        "0": "れい",
        "1": "いち",
        "2": "に",
        "3": "さん",
        "4": "よん",
        "5": "ご",
        "6": "ろく",
        "7": "なな",
        "8": "はち",
        "9": "きゅう",
    }

    @classmethod
    def normalize(cls, output: str, *, system_prompt: str) -> str:
        return cls.normalize_with_policy(output, system_prompt=system_prompt)

    @classmethod
    def normalize_with_policy(
        cls,
        output: str,
        *,
        system_prompt: str,
        policy: ConstraintPolicy | None = None,
    ) -> str:
        effective = OutputValidator._effective_policy(system_prompt, policy)
        value = output
        if cls._target_prefers_simplified_chinese(system_prompt):
            value = cls._normalize_simplified_variants(value)
        if OutputValidator._requires_hiragana(system_prompt, policy=effective):
            value = cls._normalize_hiragana_output(value)

        punctuation_rules = effective.rules_of_type("punctuation", local_only=True)
        if punctuation_rules:
            allowed = {
                str(item)
                for rule in punctuation_rules
                for item in rule.params.get("allowed", [])
            }
            has_separator = bool(effective.rules_of_type("separator", local_only=True))
            chars: list[str] = []
            for ch in value:
                if ch in allowed or not unicodedata.category(ch).startswith(("P", "S")):
                    chars.append(ch)
                elif has_separator:
                    chars.append(" ")
            value = "".join(chars)

        line_break_rules = effective.rules_of_type("line_breaks", local_only=True)
        line_mode = str(line_break_rules[0].params.get("mode", "")) if line_break_rules else ""
        if line_mode == "strip":
            value = re.sub(r"\r?\n", " ", value)
        elif line_mode == "single":
            value = re.sub(r"(?:\r?\n)+", "\n", value)

        separator_rules = effective.rules_of_type("separator", local_only=True)
        if separator_rules:
            bracket_rules = [
                rule
                for rule in separator_rules
                if str(rule.params.get("scope", "global")) == "between_bracketed_terms"
            ]
            global_rules = [
                rule
                for rule in separator_rules
                if str(rule.params.get("scope", "global")) != "between_bracketed_terms"
            ]
            if global_rules:
                width = max(
                    1,
                    max(int(rule.params.get("normalize_to", 1) or 1) for rule in global_rules),
                )
                whitespace_pattern = r"[ \t\f\v]+" if line_mode in {"preserve", "single"} else r"\s+"
                value = re.sub(whitespace_pattern, " " * width, value).strip()
            else:
                value = value.strip()
            for rule in bracket_rules:
                width = max(1, int(rule.params.get("normalize_to", 1) or 1))
                left = str(rule.params.get("left", "["))
                right = str(rule.params.get("right", "]"))
                value = re.sub(
                    re.escape(right) + r"\s*" + re.escape(left),
                    right + (" " * width) + left,
                    value,
                )

        if effective.has_script("hiragana"):
            for rule in effective.rules_of_type("term_wrapper", local_only=True):
                value = cls._normalize_hiragana_wrapped_terms(
                    value,
                    left=str(rule.params.get("left", "[")),
                    right=str(rule.params.get("right", "]")),
                )

        case_rules = effective.rules_of_type("case", local_only=True)
        if case_rules:
            mode = str(case_rules[0].params.get("mode", "preserve"))
            if mode == "lower":
                value = value.lower()
            elif mode == "upper":
                value = value.upper()
        return value

    @classmethod
    def _normalize_hiragana_output(cls, output: str) -> str:
        output = cls._normalize_ascii_numbers_to_hiragana(output)
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
    def _normalize_hiragana_wrapped_terms(output: str, *, left: str, right: str) -> str:
        if not left or not right:
            return output
        pattern = re.escape(left) + r"([^" + re.escape(left + right) + r"]+)" + re.escape(right)

        def compact(match: re.Match[str]) -> str:
            return left + re.sub(r"\s+", "", match.group(1)) + right

        return re.sub(pattern, compact, output)

    @staticmethod
    def _target_prefers_simplified_chinese(system_prompt: str) -> bool:
        lowered = system_prompt.lower()
        return (
            "to chinese" in lowered
            or "to simplified chinese" in lowered
            or "to 中文" in system_prompt
            or "to 简体中文" in system_prompt
            or "to 简体" in system_prompt
            or "目标语言：中文" in system_prompt
            or "目标语言: 中文" in system_prompt
        )

    @staticmethod
    def _normalize_simplified_variants(output: str) -> str:
        return output.translate(_SIMPLIFIED_VARIANT_MAP)

    @classmethod
    def _normalize_ascii_numbers_to_hiragana(cls, output: str) -> str:
        return re.sub(
            r"\d+(?:\.\d+)?",
            lambda match: cls._ascii_number_to_hiragana(match.group(0)),
            output,
        )

    @classmethod
    def _ascii_number_to_hiragana(cls, value: str) -> str:
        if "." in value:
            integer, fraction = value.split(".", 1)
            fraction_reading = "".join(cls._DIGIT_HIRAGANA[digit] for digit in fraction)
            return f"{cls._integer_to_hiragana(integer)}てん{fraction_reading}"
        return cls._integer_to_hiragana(value)

    @classmethod
    def _integer_to_hiragana(cls, value: str) -> str:
        number = int(value or "0")
        if number == 0:
            return cls._DIGIT_HIRAGANA["0"]
        if number > 9999:
            return "".join(cls._DIGIT_HIRAGANA[digit] for digit in value)

        parts: list[str] = []
        thousands, remainder = divmod(number, 1000)
        hundreds, remainder = divmod(remainder, 100)
        tens, ones = divmod(remainder, 10)

        if thousands:
            if thousands == 1:
                parts.append("せん")
            elif thousands == 3:
                parts.append("さんぜん")
            elif thousands == 8:
                parts.append("はっせん")
            else:
                parts.append(cls._DIGIT_HIRAGANA[str(thousands)] + "せん")
        if hundreds:
            if hundreds == 1:
                parts.append("ひゃく")
            elif hundreds == 3:
                parts.append("さんびゃく")
            elif hundreds == 6:
                parts.append("ろっぴゃく")
            elif hundreds == 8:
                parts.append("はっぴゃく")
            else:
                parts.append(cls._DIGIT_HIRAGANA[str(hundreds)] + "ひゃく")
        if tens:
            if tens == 1:
                parts.append("じゅう")
            else:
                parts.append(cls._DIGIT_HIRAGANA[str(tens)] + "じゅう")
        if ones:
            parts.append(cls._DIGIT_HIRAGANA[str(ones)])
        return "".join(parts)

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
                return "お"
        return ""
