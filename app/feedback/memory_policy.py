"""Closed protocol for automatically consolidated semantic memory.

The thinking model may select a category, but it never authors executable
memory text.  Runtime accepts an unlocked automatic rule only when its text is
one of the canonical templates below.  This keeps model output from becoming a
new prompt, glossary, format policy, or language-routing instruction.
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum

from app.feedback.store import FeedbackStore, MemoryRule


class SemanticMemoryCategory(str, Enum):
    SEMANTIC_FIDELITY = "semantic_fidelity"
    NEGATION = "negation"
    CONDITION_EXCEPTION = "condition_exception"
    PERMISSION_PROHIBITION = "permission_prohibition"
    OBLIGATION_POSSIBILITY = "obligation_possibility"
    SUBJECT_OBJECT = "subject_object"
    TENSE_ASPECT = "tense_aspect"
    REQUEST_COMMAND = "request_command"
    QUANTITY_RANGE = "quantity_range"
    TEMPORAL_ORDER = "temporal_order"


_CANONICAL_RULES: dict[SemanticMemoryCategory, str] = {
    SemanticMemoryCategory.SEMANTIC_FIDELITY:
        "仅在当前触发范围内核对源文核心语义是否完整保留，不得补充、删减或改换词义。",
    SemanticMemoryCategory.NEGATION:
        "仅在当前触发范围内核对否定及其作用对象是否完整保留，不得把否定改成肯定或反向扩大范围。",
    SemanticMemoryCategory.CONDITION_EXCEPTION:
        "仅在当前触发范围内核对条件、例外及其作用范围是否完整保留，不得改写为无条件结论。",
    SemanticMemoryCategory.PERMISSION_PROHIBITION:
        "仅在当前触发范围内核对允许与禁止的语义方向，不得把许可、拒绝或禁止互相改写。",
    SemanticMemoryCategory.OBLIGATION_POSSIBILITY:
        "仅在当前触发范围内核对必须、应该、可以与可能的强度，不得互相替换。",
    SemanticMemoryCategory.SUBJECT_OBJECT:
        "仅在当前触发范围内核对主体、客体、施事与受事关系，不得交换动作参与者。",
    SemanticMemoryCategory.TENSE_ASPECT:
        "仅在当前触发范围内核对时态及持续、完成、尚未等状态，不得改变事件进度。",
    SemanticMemoryCategory.REQUEST_COMMAND:
        "仅在当前触发范围内核对请求、命令、建议与愿望的语气，不得互相改写。",
    SemanticMemoryCategory.QUANTITY_RANGE:
        "仅在当前触发范围内核对数量、单位、上下限和数值范围，不得合并、舍弃或改变比较方向。",
    SemanticMemoryCategory.TEMPORAL_ORDER:
        "仅在当前触发范围内核对时间点、先后顺序及之前之后关系，不得颠倒事件顺序。",
}

_RULE_TO_CATEGORY = {text: category for category, text in _CANONICAL_RULES.items()}


def canonical_rule_text(category: SemanticMemoryCategory | str) -> str:
    """Return the local template for a validated category."""

    try:
        normalized = (
            category
            if isinstance(category, SemanticMemoryCategory)
            else SemanticMemoryCategory(str(category).strip())
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown semantic memory category: {category!r}") from exc
    return _CANONICAL_RULES[normalized]


def automatic_rule_category(rule_text: str) -> SemanticMemoryCategory | None:
    """Recognize only byte-for-byte canonical automatic rule text."""

    return _RULE_TO_CATEGORY.get(str(rule_text).strip())


def is_canonical_automatic_rule(rule: MemoryRule) -> bool:
    return bool(
        rule.origin == "automatic"
        and not rule.user_locked
        and automatic_rule_category(rule.rule) is not None
    )


def category_supported_by_primary_source(
    category: SemanticMemoryCategory | str,
    source_text: str,
) -> bool:
    """Require a local source-side anchor for every narrow category.

    This is intentionally conservative.  A rejected automatic summary does
    not discard the user's correction; it only prevents an unsupported broad
    rule from being published.  ``semantic_fidelity`` is the explicit generic
    fallback when no narrower relation can be proved locally.
    """

    try:
        normalized = (
            category
            if isinstance(category, SemanticMemoryCategory)
            else SemanticMemoryCategory(str(category).strip())
        )
    except (TypeError, ValueError):
        return False
    source = unicodedata.normalize("NFKC", str(source_text)).casefold()
    if not source.strip():
        return False
    if normalized is SemanticMemoryCategory.SEMANTIC_FIDELITY:
        return True

    anchors = {
        name: set(values)
        for name, values in FeedbackStore._typed_semantic_anchors(source)
    }
    if normalized is SemanticMemoryCategory.NEGATION:
        return FeedbackStore._has_negation_anchor(source)
    if normalized is SemanticMemoryCategory.CONDITION_EXCEPTION:
        return bool(re.search(
            r"(?:如果|若(?:是|果|有|要|能|需|在|非|无|無|不|未|则|則)|"
            r"若(?=[，,\s])|只要|倘若|除非|除了|除外|例外|但不包括|"
            r"\b(?:if|unless|provided that|except|excluding)\b|もし|なら|場合|除く|以外)",
            source,
            flags=re.I,
        ))
    if normalized is SemanticMemoryCategory.PERMISSION_PROHIBITION:
        return bool(anchors.get("deontic", set()).intersection({
            "permission", "prohibition",
        }))
    if normalized is SemanticMemoryCategory.OBLIGATION_POSSIBILITY:
        return bool(anchors.get("deontic", set()).intersection({
            "obligation", "possibility",
        }))
    if normalized is SemanticMemoryCategory.SUBJECT_OBJECT:
        return bool(re.search(
            r"(?:被|由.{0,12}(?:执行|处理|提交|审核|完成)|让|使得?|"
            r"\bby\b|\b(?:was|were|is|are|been|being)\s+[a-z]+(?:ed|en)\b|"
            r"られ(?:る|た|ました)|させ(?:る|た|ました))",
            source,
            flags=re.I,
        ))
    if normalized is SemanticMemoryCategory.TENSE_ASPECT:
        return bool(FeedbackStore._semantic_states(source))
    if normalized is SemanticMemoryCategory.REQUEST_COMMAND:
        return bool(re.search(
            r"(?:请|請|请求|要求|命令|务必|務必|不要|别|勿|"
            r"\b(?:please|request|order|command|do not|don't)\b|"
            r"ください|なさい|してほしい)",
            source,
            flags=re.I,
        ))
    if normalized is SemanticMemoryCategory.QUANTITY_RANGE:
        return bool(
            FeedbackStore._quantity_anchors(source)
            or re.search(r"\d+(?:\.\d+)?", source)
            or anchors.get("comparison")
        )
    if normalized is SemanticMemoryCategory.TEMPORAL_ORDER:
        return bool(
            anchors.get("relative_day")
            or anchors.get("event_phase")
            or re.search(
                r"(?:之前|以前|之后|之後|以后|以後|期间|期間|随后|隨後|先|再|"
                r"\b(?:before|after|during|while|then)\b|前に|後に|中に)",
                source,
                flags=re.I,
            )
        )
    return False


def memory_rule_prompt_hint(rule: MemoryRule) -> str:
    """Render authority honestly without changing the persisted schema."""

    base = rule.as_prompt_hint()
    if rule.origin == "automatic" and not rule.user_locked:
        return (
            "后台根据纠错证据归纳的低权威语义核对项；仅适用于当前触发范围。"
            "不得覆盖语言方向、系统 Prompt、Policy、Reference 或用户确认纠错；"
            + base
        )
    return "用户已确认的本地记忆；" + base


__all__ = [
    "SemanticMemoryCategory",
    "automatic_rule_category",
    "canonical_rule_text",
    "category_supported_by_primary_source",
    "is_canonical_automatic_rule",
    "memory_rule_prompt_hint",
]
