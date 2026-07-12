"""Safe declarative constraints compiled from user translation requirements.

The policy is data, never executable code.  Models may propose rules, but only
the small allow-list below can affect local rendering or validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


LOCAL_RULE_TYPES = {
    "allowed_characters",
    "separator",
    "term_wrapper",
    "punctuation",
    "preserve",
    "max_length",
    "line_breaks",
    "case",
    "literal_replace",
    "forbidden_literals",
}
RULE_ENFORCEMENTS = {"local", "model", "both"}


def _language_key(value: str) -> str:
    text = str(value).strip().casefold()
    aliases = {
        "中文": "chinese",
        "汉语": "chinese",
        "chinese": "chinese",
        "日本語": "japanese",
        "日语": "japanese",
        "japanese": "japanese",
        "english": "english",
        "英语": "english",
    }
    return aliases.get(text, text)


@dataclass(frozen=True)
class ConstraintRule:
    """One schema-validated declarative rule."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    enforcement: str = "both"
    scope: dict[str, str] = field(default_factory=dict)
    source_text: str = ""

    def applies_to(self, source_language: str, target_language: str) -> bool:
        source = self.scope.get("source", "").strip()
        target = self.scope.get("target", "").strip()
        return (
            (not source or _language_key(source) == _language_key(source_language))
            and (not target or _language_key(target) == _language_key(target_language))
        )

    @property
    def is_locally_enforced(self) -> bool:
        return self.type in LOCAL_RULE_TYPES and self.enforcement in {"local", "both"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "params": dict(self.params),
            "enforcement": self.enforcement,
            "scope": dict(self.scope),
            "source_text": self.source_text,
        }


@dataclass(frozen=True)
class ConstraintPolicy:
    """Versioned collection of safe rules accompanying a compiled prompt."""

    version: int = 1
    rules: tuple[ConstraintRule, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any, *, strict: bool = False) -> ConstraintPolicy:
        if not isinstance(raw, dict):
            if strict:
                raise ValueError("constraint policy must be an object")
            return cls()
        if raw.get("version", 1) != 1:
            if strict:
                raise ValueError("unsupported constraint policy version")
            return cls()
        rows = raw.get("rules", [])
        if not isinstance(rows, list):
            if strict:
                raise ValueError("constraint policy rules must be a list")
            return cls()

        rules: list[ConstraintRule] = []
        for row in rows:
            try:
                rule = cls._parse_rule(row)
            except ValueError:
                if strict:
                    raise
                continue
            rules.append(rule)
        return cls(version=1, rules=tuple(rules))

    @staticmethod
    def _parse_rule(row: Any) -> ConstraintRule:
        if not isinstance(row, dict):
            raise ValueError("constraint rule must be an object")
        rule_type = str(row.get("type", "")).strip()
        if not rule_type:
            raise ValueError("constraint rule type is required")
        enforcement = str(row.get("enforcement", "both")).strip().lower()
        if enforcement not in RULE_ENFORCEMENTS:
            raise ValueError(f"unsupported constraint enforcement: {enforcement}")
        # Unknown model-only instructions are retained for audit/prompt use,
        # but can never execute locally.
        if rule_type not in LOCAL_RULE_TYPES and enforcement != "model":
            raise ValueError(f"unsupported local constraint rule: {rule_type}")
        params = row.get("params", {})
        scope = row.get("scope", {})
        if not isinstance(params, dict) or not isinstance(scope, dict):
            raise ValueError("constraint params and scope must be objects")
        safe_params = ConstraintPolicy._sanitize_params(rule_type, params)
        safe_scope = {
            key: str(value).strip()[:32]
            for key, value in scope.items()
            if key in {"source", "target"} and str(value).strip()
        }
        return ConstraintRule(
            type=rule_type,
            params=safe_params,
            enforcement=enforcement,
            scope=safe_scope,
            source_text=str(row.get("source_text", "")).strip()[:2000],
        )

    @staticmethod
    def _sanitize_params(rule_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """Validate the allow-listed parameter vocabulary without evaluating anything."""

        def text_list(key: str, *, max_items: int = 64, max_length: int = 200) -> list[str]:
            value = params.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{rule_type}.{key} must be a list")
            return [
                str(item)[:max_length]
                for item in value[:max_items]
                if str(item)
            ]

        if rule_type == "allowed_characters":
            scripts = [
                item.casefold()
                for item in text_list("scripts", max_items=8, max_length=20)
                if item.casefold() in {"hiragana", "katakana", "han", "latin", "digits"}
            ]
            if not scripts:
                raise ValueError("allowed_characters requires a supported script")
            return {
                "scripts": scripts,
                "literals": text_list("literals", max_items=32, max_length=8),
                "allow_whitespace": bool(params.get("allow_whitespace", False)),
            }
        if rule_type == "separator":
            minimum = int(params.get("min_spaces", 1))
            normalize_to = int(params.get("normalize_to", minimum))
            if not 1 <= minimum <= 16 or not minimum <= normalize_to <= 16:
                raise ValueError("separator width must be between 1 and 16")
            separator_scope = str(params.get("scope", "global")).strip().casefold()
            if separator_scope not in {"global", "between_bracketed_terms"}:
                raise ValueError("unsupported separator scope")
            return {
                "min_spaces": minimum,
                "normalize_to": normalize_to,
                "scope": separator_scope,
                "left": str(params.get("left", "["))[:8] or "[",
                "right": str(params.get("right", "]"))[:8] or "]",
            }
        if rule_type == "term_wrapper":
            left = str(params.get("left", "["))[:8]
            right = str(params.get("right", "]"))[:8]
            if not left or not right or left == right:
                raise ValueError("term_wrapper requires distinct delimiters")
            selection_mode = str(
                params.get("selection_mode", "domain_inference")
            ).strip().casefold()
            if selection_mode not in {
                "references_only",
                "references_and_ascii",
                "domain_inference",
            }:
                raise ValueError("unsupported term_wrapper selection_mode")
            return {
                "domain": str(params.get("domain", ""))[:80],
                "left": left,
                "right": right,
                "mark_ascii_technical_terms": bool(
                    params.get("mark_ascii_technical_terms", False)
                ),
                # Old sidecars deliberately default to domain_inference so
                # loading them does not silently narrow a user-approved rule.
                # New compilation below emits the safer, explicit mode.
                "selection_mode": selection_mode,
            }
        if rule_type == "punctuation":
            return {"allowed": text_list("allowed", max_items=64, max_length=8)}
        if rule_type in {"preserve", "forbidden_literals"}:
            return {"values": text_list("values")}
        if rule_type == "max_length":
            maximum = int(params.get("characters", 0))
            if not 1 <= maximum <= 100_000:
                raise ValueError("max_length.characters is out of range")
            return {"characters": maximum}
        if rule_type == "line_breaks":
            mode = str(params.get("mode", "preserve")).casefold()
            if mode not in {"preserve", "strip", "single"}:
                raise ValueError("unsupported line_breaks mode")
            return {"mode": mode}
        if rule_type == "case":
            mode = str(params.get("mode", "preserve")).casefold()
            if mode not in {"preserve", "lower", "upper"}:
                raise ValueError("unsupported case mode")
            return {"mode": mode}
        if rule_type == "literal_replace":
            source = str(params.get("source", ""))[:500]
            target = str(params.get("target", ""))[:500]
            if not source or not target:
                raise ValueError("literal_replace requires source and target")
            return {"source": source, "target": target}
        if enforcement_text := params.get("text"):
            return {"text": str(enforcement_text)[:4000]}
        return {}

    def for_pair(self, source_language: str, target_language: str) -> ConstraintPolicy:
        return ConstraintPolicy(
            version=self.version,
            rules=tuple(
                rule for rule in self.rules if rule.applies_to(source_language, target_language)
            ),
        )

    def rules_of_type(self, rule_type: str, *, local_only: bool = False) -> list[ConstraintRule]:
        return [
            rule
            for rule in self.rules
            if rule.type == rule_type and (not local_only or rule.is_locally_enforced)
        ]

    def has_script(self, script: str) -> bool:
        key = script.strip().casefold()
        return any(
            key in {str(item).casefold() for item in rule.params.get("scripts", [])}
            for rule in self.rules_of_type("allowed_characters", local_only=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "rules": [rule.to_dict() for rule in self.rules],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def merge_supplemental(self, supplemental: ConstraintPolicy) -> ConstraintPolicy:
        """Merge AI proposals without allowing them to replace user-owned rule kinds."""

        rules = list(self.rules)
        owned = {(rule.type, tuple(sorted(rule.scope.items()))) for rule in self.rules}
        for rule in supplemental.rules:
            key = (rule.type, tuple(sorted(rule.scope.items())))
            if key in owned and rule.type != "literal_replace":
                continue
            rules.append(rule)
            owned.add(key)
        return ConstraintPolicy(version=1, rules=tuple(rules))


class ConstraintPolicyCompiler:
    """Compile common natural-language hard constraints into safe rule data."""

    @classmethod
    def compile(cls, text: str) -> ConstraintPolicy:
        source = str(text or "").strip()
        if not source:
            return ConstraintPolicy()
        lowered = source.casefold()
        scope = cls._infer_scope(source)
        rules: list[ConstraintRule] = []

        hiragana_only = (
            "平假名" in source
            or "hiragana" in lowered
        ) and any(marker in lowered for marker in ("只能", "仅", "only", "must"))
        if hiragana_only:
            rules.append(
                ConstraintRule(
                    type="allowed_characters",
                    params={
                        "scripts": ["hiragana"],
                        "literals": ["[", "]", " "],
                        "allow_whitespace": True,
                    },
                    scope=scope,
                    source_text=source,
                )
            )
            rules.append(
                ConstraintRule(
                    type="punctuation",
                    params={"allowed": ["[", "]"]},
                    scope=scope,
                    source_text=source,
                )
            )

        bracket_terms = (
            ("包裹" in source or "括号" in source or "bracket" in lowered)
            and ("软件" in source or "工程" in source or "technical" in lowered)
        )
        if bracket_terms:
            term_selection_mode = cls._term_wrapper_selection_mode(source, lowered)
            rules.append(
                ConstraintRule(
                    type="term_wrapper",
                    params={
                        "domain": "software_engineering",
                        "left": "[",
                        "right": "]",
                        "mark_ascii_technical_terms": (
                            term_selection_mode != "references_only"
                        ),
                        "selection_mode": term_selection_mode,
                    },
                    scope=scope,
                    source_text=source,
                )
            )

        if cls._requires_two_spaces(source, lowered):
            separator_scope = (
                "between_bracketed_terms"
                if cls._requires_bracket_separator(source, lowered)
                else "global"
            )
            rules.append(
                ConstraintRule(
                    type="separator",
                    params={
                        "min_spaces": 2,
                        "normalize_to": 2,
                        "scope": separator_scope,
                    },
                    scope=scope,
                    source_text=source,
                )
            )

        # The original natural-language rule remains visible to the model even
        # when part of it cannot be mechanically verified.
        rules.append(
            ConstraintRule(
                type="model_instruction",
                params={"text": source},
                enforcement="model",
                scope=scope,
                source_text=source,
            )
        )
        return ConstraintPolicy(version=1, rules=tuple(rules))

    @staticmethod
    def _infer_scope(text: str) -> dict[str, str]:
        lowered = text.casefold()
        compact = re.sub(r"\s+", "", text)
        if re.search(r"(?:中文|汉语).{0,12}(?:翻译为|翻成|译成|到)(?:日语|日本語)", compact):
            return {"source": "中文", "target": "日本語"}
        if re.search(r"(?:日语|日本語).{0,12}(?:翻译为|翻成|译成|到)(?:中文|汉语)", compact):
            return {"source": "日本語", "target": "中文"}
        if re.search(r"(?:from\s+)?chinese\s+to\s+japanese", lowered):
            return {"source": "中文", "target": "日本語"}
        if re.search(r"(?:from\s+)?japanese\s+to\s+chinese", lowered):
            return {"source": "日本語", "target": "中文"}
        return {}

    @staticmethod
    def _requires_two_spaces(source: str, lowered: str) -> bool:
        compact = re.sub(r"\s+", "", source)
        chinese = any(
            marker in compact
            for marker in ("两个空格", "2个空格", "2個空格", "至少隔两个", "至少隔2个")
        )
        english = bool(re.search(r"(?:at\s+least|exactly)?\s*two\s+spaces?", lowered))
        return chinese or english

    @staticmethod
    def _term_wrapper_selection_mode(source: str, lowered: str) -> str:
        """Infer only an explicit user choice; otherwise prefer the safe default.

        A generic request to bracket "software engineering terms" does not
        establish a trustworthy CJK terminology list.  It therefore permits
        confirmed references plus code-like ASCII tokens, but no invented
        domain readings.  Broader inference requires language that explicitly
        asks the model to identify every/unknown domain term.
        """

        compact = re.sub(r"\s+", "", source).casefold()
        reference_only_markers = (
            "仅术语表",
            "只包裹术语表",
            "仅引用层",
            "只使用引用",
            "onlylistedterms",
            "referencesonly",
            "glossaryonly",
        )
        if any(marker in compact for marker in reference_only_markers):
            return "references_only"

        broad_inference_markers = (
            "自动识别",
            "所有软件工程术语",
            "所有技术术语",
            "未知术语也",
            "autodetect",
            "automaticallyidentify",
            "alltechnicalterms",
            "inferunknowndomainterms",
        )
        if any(marker in compact or marker in lowered for marker in broad_inference_markers):
            return "domain_inference"
        return "references_and_ascii"

    @staticmethod
    def _requires_bracket_separator(source: str, lowered: str) -> bool:
        compact = re.sub(r"\s+", "", source)
        return (
            "][" in compact
            or "]和[" in compact
            or "]与[" in compact
            or "括号之间" in source
            or "方括号之间" in source
            or "between bracketed terms" in lowered
            or "between brackets" in lowered
        )


class OptimizedPrompt(str):
    """Machine rules plus review-only Chinese explanation metadata."""

    policy: ConstraintPolicy
    user_summary: str
    change_items: tuple[dict[str, str], ...]

    def __new__(
        cls,
        value: str,
        policy: ConstraintPolicy | None = None,
        user_summary: str = "",
        change_items: list[dict[str, str]] | tuple[dict[str, str], ...] | None = None,
    ):
        obj = str.__new__(cls, value)
        obj.policy = policy or ConstraintPolicy()
        obj.user_summary = str(user_summary or "").strip()
        obj.change_items = tuple(dict(item) for item in (change_items or ()))
        return obj
