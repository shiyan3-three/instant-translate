"""POC experiment: Flash semantic integrity and retrieval strategy.

This experiment answers one question: on the Flash translation path, compared
to the current production profile, do static semantic guard rules and a
runtime-retrieved typed semantic policy reduce major semantic errors and
incomplete sentences without regressing hard-format pass rate or latency?

Three groups share the same current production prompt, policy, Profile digest,
Flash model, raw OCR text, and production output normalizer; the only variable
is how semantic constraints are injected:

* CURRENT_PROFILE: current production profile, no extra semantic guard.
* STATIC_SEMANTIC_GUARD: five fixed semantic guard rules injected via the
  memory_hints channel of ``_wrap_source_text`` for every request.
* RETRIEVED_SEMANTIC_POLICY: same as STATIC plus Pro-compiled typed policy
  rules retrieved by Chinese trigger substring or detected source feature
  (max 3 rules per request, empty list when nothing matches).

Data isolation: only the three ``training_incidents`` are sent to the Pro
policy compiler; held-out and control sources never enter the Pro payload.
``seen_regression`` success cannot substitute for ``held_out`` generalization
success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.agent import TranslationAgent
from app.prompt.policy import ConstraintPolicy
from app.settings import AppSettings
from app.translation.client import TranslationError
from app.translation.quality import OutputNormalizer, OutputValidator
from app.translation.service import TranslationService
from app.translation.terms import TermPlaceholder
from poc.poc_agent_memory_transfer import (
    evaluate_format,
    extract_json_object,
)
from poc.poc_reference_injection import (
    _percentile,
    _send_request,
    estimate_tokens,
)


GROUPS = ("CURRENT_PROFILE", "STATIC_SEMANTIC_GUARD", "RETRIEVED_SEMANTIC_POLICY")

EXPECTED_CASE_SOURCES = (
    "今天是2026年7月3日，我准备学习第42课。",
    "小李明明把书放在桌子上。",
    "商品原价是4.0元，成绩在70到80分之间。",
    "我准备明天开始学习第九课。",
    "小王明明把钥匙放进抽屉了。",
    "明明已经测试过了，为什么还会失败？",
    "会议定在2026年7月4日下午4点30分。",
    "温度保持在4.0度，误差不能超过0.5度。",
    "完成这项工作大约需要3个小时。",
    "虽然测试通过了，但是现在还不能发布。",
    "如果今晚发布失败，马上回滚到上一个版本。",
    "今天天气很好，我们下午去公园散步吧。",
)
EXPECTED_CASE_SCOPES = (
    "seen_regression",
    "seen_regression",
    "seen_regression",
    "held_out",
    "held_out",
    "held_out",
    "held_out",
    "held_out",
    "held_out",
    "control",
    "control",
    "control",
)

STATIC_GUARD_RULES = (
    "翻译源文全部命题，输出必须以完整日语谓语收束",
    "不得以 を、に、が、の、なのに、けど、ので、から、ために 等助词或连接表达悬空结束",
    "不得增加源文没有的转折、因果、推测或本应如此的语义",
    "根据完整上下文处理中文歧义和人名，不得机械逐字翻译",
    "数字、日期、范围和计数必须按目标语言及平假名约束表达，不得保留ASCII数字",
)

MAX_RULES_PER_REQUEST = 3
_POLICY_VERSION = 1
_POLICY_ROLE = "retrieved_semantic_policy"
_FEATURE_WHITELIST = ("contains_decimal", "contains_date", "contains_numeric_range")
_FEATURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "contains_decimal": re.compile(r"\d+\.\d+"),
    "contains_date": re.compile(r"\d{4}年"),
    "contains_numeric_range": re.compile(r"\d+到\d+|\d+-\d+"),
}
_INCOMPLETE_ENDINGS = (
    "を",
    "に",
    "が",
    "の",
    "なのに",
    "けど",
    "ので",
    "から",
    "ために",
)
_ASCII_DIGIT = re.compile(r"[0-9]")
_RULE_FIELDS = {"id", "trigger_literals", "trigger_features", "instruction", "avoid"}

_DIGEST_USER = (
    "请逐条列出上述翻译规则中最关键的格式要求"
    "（如字符限制、术语格式、空格、标点等）。"
    "每条一行，只列规则要点，不要翻译这句话。"
)
_DIGEST_META = (
    "\n\n[系统元指令] 以下消息是系统设置步骤，不是待翻译文本。请正常回复，不要翻译。"
)

_POLICY_COMPILER_SYSTEM = """You are the startup policy compiler of a persistent translation agent.
Convert confirmed translation incidents into compact, typed semantic policy rules for a cheaper Flash model.

Generalize only what each incident supports. Each rule must have a stable id, short Chinese trigger_literals (1-8 chars each, 1-3 per rule), optional trigger_features from [contains_decimal, contains_date, contains_numeric_range], a concise instruction (max 500 chars), and an avoid list of specific bad output patterns. Do not invent triggers or instructions not supported by the evidence. The runtime retrieves rules by trigger substring or detected feature, so choose triggers likely to occur in analogous future source text.

Held-out evaluation sentences are deliberately hidden from you. Do not create new test sentences, translations, chain-of-thought, or commentary. Return only one JSON object with exactly this schema:
{
  "version": 1,
  "role": "retrieved_semantic_policy",
  "rules": [
    {
      "id": "stable-id",
      "trigger_literals": ["short Chinese trigger"],
      "trigger_features": [],
      "instruction": "actionable rule for analogous future translations",
      "avoid": ["specific bad output pattern"]
    }
  ]
}

Create 3-12 rules. Do not output Markdown fences, reasoning_content, or commentary."""


class SemanticGuardError(ValueError):
    """Raised when semantic-guard data or compiled policy is invalid."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingIncident:
    id: str
    source: str
    bad: str
    diagnosis: str


@dataclass(frozen=True)
class SemanticGuardCase:
    id: str
    scope: str
    category: str
    source: str
    review_focus: str
    fixture_translation: str
    required_all: tuple[str, ...]
    required_any: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...]
    expected_complete: bool
    expected_ascii_digit_free: bool


@dataclass(frozen=True)
class SemanticGuardDataset:
    version: int
    status: str
    name: str
    description: str
    training_incidents: tuple[TrainingIncident, ...]
    cases: tuple[SemanticGuardCase, ...]


@dataclass(frozen=True)
class ProductionContext:
    """Read-only snapshot of the production Agent definition used by this POC."""

    system: str
    policy: ConstraintPolicy
    prompt_hash: str
    prompt_length: int
    policy_digest: str
    enhanced_system_hash: str


# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------


def _clean_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SemanticGuardError(f"{field} must be a list")
    return tuple(str(item).strip() for item in value if str(item).strip())


def load_semantic_guard_dataset(path: Path) -> SemanticGuardDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SemanticGuardError("semantic guard dataset must be a JSON object")

    incidents_raw = raw.get("training_incidents")
    if not isinstance(incidents_raw, list):
        raise SemanticGuardError("training_incidents must be a list")
    incidents = []
    for item in incidents_raw:
        if not isinstance(item, dict):
            raise SemanticGuardError("each training incident must be an object")
        incidents.append(
            TrainingIncident(
                id=str(item.get("id", "")).strip(),
                source=str(item.get("source", "")).strip(),
                bad=str(item.get("bad", "")).strip(),
                diagnosis=str(item.get("diagnosis", "")).strip(),
            )
        )

    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise SemanticGuardError("cases must be a list")
    cases = []
    for item in cases_raw:
        if not isinstance(item, dict):
            raise SemanticGuardError("each case must be an object")
        required_any_raw = item.get("required_any", [])
        if not isinstance(required_any_raw, list):
            raise SemanticGuardError(f"{item.get('id', '?')}.required_any must be a list")
        required_any = tuple(
            _clean_list(group, f"{item.get('id', '?')}.required_any group")
            for group in required_any_raw
            if isinstance(group, list) and group
        )
        cases.append(
            SemanticGuardCase(
                id=str(item.get("id", "")).strip(),
                scope=str(item.get("scope", "")).strip(),
                category=str(item.get("category", "")).strip(),
                source=str(item.get("source", "")).strip(),
                review_focus=str(item.get("review_focus", "")).strip(),
                fixture_translation=str(item.get("fixture_translation", "")).strip(),
                required_all=_clean_list(item.get("required_all", []), f"{item.get('id', '?')}.required_all"),
                required_any=required_any,
                forbidden=_clean_list(item.get("forbidden", []), f"{item.get('id', '?')}.forbidden"),
                expected_complete=bool(item.get("expected_complete", True)),
                expected_ascii_digit_free=bool(item.get("expected_ascii_digit_free", True)),
            )
        )

    dataset = SemanticGuardDataset(
        version=int(raw.get("version", 0)),
        status=str(raw.get("status", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        training_incidents=tuple(incidents),
        cases=tuple(cases),
    )
    validate_semantic_guard_dataset(dataset)
    return dataset


def validate_semantic_guard_dataset(dataset: SemanticGuardDataset) -> None:
    errors: list[str] = []
    if dataset.version != 1 or dataset.status != "frozen_poc":
        errors.append("dataset must be frozen_poc version 1")
    if len(dataset.training_incidents) != 3:
        errors.append(
            f"exactly 3 training incidents are required, found {len(dataset.training_incidents)}"
        )
    if len(dataset.cases) != 12:
        errors.append(f"exactly 12 cases are required, found {len(dataset.cases)}")

    incident_ids = [ti.id for ti in dataset.training_incidents]
    if any(not item for item in incident_ids) or len(incident_ids) != len(set(incident_ids)):
        errors.append("training incident ids must be non-empty and unique")

    case_ids = [case.id for case in dataset.cases]
    case_sources = [case.source for case in dataset.cases]
    if any(not item for item in case_ids) or len(case_ids) != len(set(case_ids)):
        errors.append("case ids must be non-empty and unique")
    if any(not item for item in case_sources) or len(case_sources) != len(set(case_sources)):
        errors.append("case sources must be non-empty and unique")
    if tuple(case_sources) != EXPECTED_CASE_SOURCES:
        errors.append("case sources or order differ from the frozen production benchmark")
    if tuple(case.scope for case in dataset.cases) != EXPECTED_CASE_SCOPES:
        errors.append("case scope mapping differs from the frozen production benchmark")

    for case in dataset.cases:
        if not case.scope or not case.category or not case.review_focus:
            errors.append(f"{case.id} is missing scope, category, or review_focus")
        if not case.fixture_translation:
            errors.append(f"{case.id} is missing fixture_translation")
        elif not evaluate_format(case.fixture_translation)["ok"]:
            errors.append(f"{case.id} fixture translation violates output format")
        if "[" in case.fixture_translation or "]" in case.fixture_translation:
            errors.append(f"{case.id} fixture must not wrap pure numeric expressions")
        incident_sources = {ti.source for ti in dataset.training_incidents}
        if case.scope == "seen_regression" and case.source not in incident_sources:
            errors.append(f"{case.id} seen_regression source must match a training incident")
        if case.scope != "seen_regression" and case.source in incident_sources:
            errors.append(f"{case.id} non-regression source must not match a training incident")

    for ti in dataset.training_incidents:
        if not ti.source or not ti.bad or not ti.diagnosis:
            errors.append(f"training incident {ti.id} has empty source, bad, or diagnosis")
        if not any(
            case.source == ti.source and case.scope == "seen_regression"
            for case in dataset.cases
        ):
            errors.append(f"training incident {ti.id} has no matching seen_regression case")

    if errors:
        raise SemanticGuardError("semantic guard dataset is invalid:\n- " + "\n- ".join(errors))


# ---------------------------------------------------------------------------
# Production context and message building
# ---------------------------------------------------------------------------


def build_production_context(settings: AppSettings) -> ProductionContext:
    """Resolve the same prompt, policy and enhanced system used by production."""

    service = TranslationService(settings)
    try:
        definition = service._resolve_agent_definition("中文", "日本語")
    finally:
        service.shutdown()

    preserve_placeholders = not definition.policy.has_script("hiragana")
    enhanced_system = TranslationAgent._append_runtime_tool_rules(
        TranslationAgent._rewrite_constraints(definition.prompt),
        preserve_placeholders=preserve_placeholders,
    )
    return ProductionContext(
        system=enhanced_system,
        policy=definition.policy,
        prompt_hash=definition.runtime_meta.prompt_hash,
        prompt_length=len(definition.prompt),
        policy_digest=definition.policy.digest(),
        enhanced_system_hash=hashlib.sha256(enhanced_system.encode("utf-8")).hexdigest(),
    )


def build_digest_payload(context: ProductionContext, thinking_model: str) -> dict[str, Any]:
    """Build the exact visible Profile digest request used by TranslationAgent."""

    return {
        "model": thinking_model,
        "messages": [
            {"role": "system", "content": context.system + _DIGEST_META},
            {"role": "user", "content": _DIGEST_USER},
        ],
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }


# ---------------------------------------------------------------------------
# Typed policy: compilation, validation, retrieval
# ---------------------------------------------------------------------------


def build_policy_source(dataset: SemanticGuardDataset) -> str:
    """Return training evidence only; held-out sources stay hidden."""

    evidence = [
        {
            "incident_id": incident.id,
            "source": incident.source,
            "bad_translation": incident.bad,
            "diagnosis": incident.diagnosis,
        }
        for incident in dataset.training_incidents
    ]
    return (
        "Compile these confirmed translation incidents into typed semantic policy rules.\n\n"
        + json.dumps({"incidents": evidence}, ensure_ascii=False, indent=2)
    )


def detect_features(source: str) -> list[str]:
    """Detect numeric features present in the Chinese source text."""

    return [
        feature
        for feature, pattern in _FEATURE_PATTERNS.items()
        if pattern.search(source)
    ]


def _validate_trigger_literals(value: Any, rule_id: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SemanticGuardError(
            f"policy rule {rule_id} trigger_literals must be a non-empty list"
        )
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text or len(text) > 8:
            raise SemanticGuardError(
                f"policy rule {rule_id} trigger_literal must be 1-8 chars: {text!r}"
            )
        if not any("\u4e00" <= ch <= "\u9fff" for ch in text):
            raise SemanticGuardError(
                f"policy rule {rule_id} trigger_literal must contain Chinese: {text!r}"
            )
        result.append(text)
    if len(result) > 3:
        result = result[:3]
    return result


def _validate_trigger_features(value: Any, rule_id: str) -> list[str]:
    if not isinstance(value, list):
        raise SemanticGuardError(
            f"policy rule {rule_id} trigger_features must be a list"
        )
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text not in _FEATURE_WHITELIST:
            raise SemanticGuardError(
                f"policy rule {rule_id} trigger_features has unknown feature: {text!r}"
            )
        if text not in result:
            result.append(text)
    return result


def validate_typed_policy(
    raw: dict[str, Any],
    *,
    dataset: SemanticGuardDataset,
) -> dict[str, Any]:
    if raw.get("version") != _POLICY_VERSION:
        raise SemanticGuardError(f"typed policy version must be {_POLICY_VERSION}")
    if raw.get("role") != _POLICY_ROLE:
        raise SemanticGuardError(f"typed policy role must be {_POLICY_ROLE}")

    rules_raw = raw.get("rules")
    if not isinstance(rules_raw, list) or not 3 <= len(rules_raw) <= 12:
        raise SemanticGuardError("typed policy rules must contain 3-12 items")

    seen_ids: set[str] = set()
    rules: list[dict[str, Any]] = []
    for index, item in enumerate(rules_raw, 1):
        if not isinstance(item, dict) or set(item) != _RULE_FIELDS:
            raise SemanticGuardError(
                f"policy rule {index} has invalid fields; expected {sorted(_RULE_FIELDS)}"
            )
        rule_id = str(item["id"]).strip()
        if not rule_id:
            raise SemanticGuardError(f"policy rule {index} id must not be empty")
        if rule_id in seen_ids:
            raise SemanticGuardError(f"policy rule id must be unique: {rule_id}")
        seen_ids.add(rule_id)

        trigger_literals = _validate_trigger_literals(item["trigger_literals"], rule_id)
        trigger_features = _validate_trigger_features(item["trigger_features"], rule_id)

        instruction = str(item["instruction"]).strip()
        if not instruction or len(instruction) > 500:
            raise SemanticGuardError(
                f"policy rule {rule_id} instruction must be 1-500 chars"
            )

        avoid_raw = item.get("avoid")
        if not isinstance(avoid_raw, list):
            raise SemanticGuardError(f"policy rule {rule_id} avoid must be a list")
        avoid = [str(a).strip() for a in avoid_raw if str(a).strip()]
        if not 1 <= len(avoid) <= 8:
            raise SemanticGuardError(f"policy rule {rule_id} needs 1-8 avoid items")

        rules.append(
            {
                "id": rule_id,
                "trigger_literals": trigger_literals,
                "trigger_features": trigger_features,
                "instruction": instruction,
                "avoid": avoid,
            }
        )

    policy = {
        "version": _POLICY_VERSION,
        "role": _POLICY_ROLE,
        "max_rules_per_request": MAX_RULES_PER_REQUEST,
        "rules": rules,
    }
    rendered = json.dumps(policy, ensure_ascii=False)
    if "```" in rendered:
        raise SemanticGuardError("typed policy must not contain markdown fences")
    if "reasoning_content" in rendered:
        raise SemanticGuardError("typed policy must not contain reasoning_content")

    held_out_sources = [
        case.source for case in dataset.cases if case.scope != "seen_regression"
    ]
    leaked = [src for src in held_out_sources if src in rendered]
    if leaked:
        raise SemanticGuardError(
            "typed policy leaked held-out sources: " + ", ".join(leaked)
        )
    return policy


def select_policy_rules(policy: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Retrieve rules whose trigger_literals or trigger_features match the source."""

    source_features = set(detect_features(source))
    selected: list[dict[str, Any]] = []
    for rule in policy["rules"]:
        literals = rule.get("trigger_literals", [])
        features = rule.get("trigger_features", [])
        if any(lit in source for lit in literals) or any(f in source_features for f in features):
            selected.append(rule)
    return selected[:MAX_RULES_PER_REQUEST]


def make_dry_typed_policy() -> dict[str, Any]:
    """Deterministic typed policy with 3 rules matching the training incidents."""

    return {
        "version": _POLICY_VERSION,
        "role": _POLICY_ROLE,
        "max_rules_per_request": MAX_RULES_PER_REQUEST,
        "rules": [
            {
                "id": "sg:complete_predicate",
                "trigger_literals": ["准备"],
                "trigger_features": [],
                "instruction": (
                    "翻译源文全部命题，输出必须以完整日语谓语收束；"
                    "不得以助词或连接表达悬空结束。"
                ),
                "avoid": ["じゅんび を"],
            },
            {
                "id": "sg:no_injected_contrast",
                "trigger_literals": ["明明"],
                "trigger_features": [],
                "instruction": (
                    "不得增加源文没有的转折、因果、推测或本应如此的语义；"
                    "源文仅陈述已完成的动作时不得追加はずなのに等表达。"
                ),
                "avoid": ["はず なのに", "はず", "なのに"],
            },
            {
                "id": "sg:no_ascii_digits",
                "trigger_literals": ["原价", "成绩"],
                "trigger_features": [
                    "contains_decimal",
                    "contains_date",
                    "contains_numeric_range",
                ],
                "instruction": (
                    "数字、日期、范围和计数必须按目标语言及平假名约束表达，"
                    "不得保留ASCII数字。"
                ),
                "avoid": ["4.0", "70", "80"],
            },
        ],
    }


def make_dry_checklist() -> str:
    return (
        "1. 出力はひらがな、角括弧、二重スペースのみ\n"
        "2. 角括弧内はひらがなのみ\n"
        "3. 数字はひらがなで表現し、ASCII数字を残さない\n"
        "4. 文は完全な述語で終わる\n"
        "5. OCR_TEXTのみを翻訳し、最終結果を <final> タグで出力する"
    )


def render_rule_hint(rule: dict[str, Any]) -> str:
    """Render a typed policy rule as a memory-hint string for Flash."""

    hint = f"[{rule['id']}] {rule['instruction']}"
    if rule.get("avoid"):
        hint += "（避免: " + "、".join(rule["avoid"]) + "）"
    return hint


# ---------------------------------------------------------------------------
# Message construction per group
# ---------------------------------------------------------------------------


def build_group_messages(
    *,
    context: ProductionContext,
    digest_user: str,
    checklist: str,
    case: SemanticGuardCase,
    group: str,
    policy: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], list[str]]:
    if group not in GROUPS:
        raise ValueError(f"unknown semantic guard group: {group}")

    memory_hints: list[str] = []
    retrieved_ids: list[str] = []

    if group in ("STATIC_SEMANTIC_GUARD", "RETRIEVED_SEMANTIC_POLICY"):
        memory_hints.extend(STATIC_GUARD_RULES)

    if group == "RETRIEVED_SEMANTIC_POLICY" and policy:
        selected = select_policy_rules(policy, case.source)
        retrieved_ids = [rule["id"] for rule in selected]
        for rule in selected:
            memory_hints.append(render_rule_hint(rule))

    preserve_verbatim = not context.policy.has_script("hiragana")
    protected = (
        TermPlaceholder.protect(case.source)
        if preserve_verbatim
        else TermPlaceholder.empty(case.source)
    )
    technical_terms: list[str] = []
    term_wrapper_domains: list[str] = []
    if context.policy.rules_of_type("term_wrapper", local_only=True):
        technical_terms = TermPlaceholder.extract_prompt_terms(
            case.source,
            reference_text=context.system,
        )
        term_wrapper_domains = [
            str(rule.params.get("domain", "")).strip()
            for rule in context.policy.rules_of_type("term_wrapper", local_only=True)
            if str(rule.params.get("domain", "")).strip()
        ]

    ocr_user = TranslationAgent._wrap_source_text(
        protected.text,
        memory_hints=memory_hints,
        rule_checklist=checklist,
        technical_terms=technical_terms,
        term_wrapper_domains=term_wrapper_domains,
    )
    messages = [
        {"role": "system", "content": context.system},
        {"role": "user", "content": digest_user},
        {"role": "assistant", "content": checklist},
        {"role": "user", "content": ocr_user},
    ]
    return messages, retrieved_ids


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_production_output(
    raw_response: str,
    *,
    case: SemanticGuardCase,
    context: ProductionContext,
) -> dict[str, Any]:
    """Apply the production fast-path extraction, normalization and validation."""

    if not raw_response:
        return {
            "extracted": "",
            "translation": "",
            "normalization_changed": False,
            "validation": {"ok": False, "reason": "empty_output"},
        }

    extracted = TranslationAgent._extract_final(raw_response)
    preserve_verbatim = not context.policy.has_script("hiragana")
    protected = (
        TermPlaceholder.protect(case.source)
        if preserve_verbatim
        else TermPlaceholder.empty(case.source)
    )
    restored = TermPlaceholder.restore(extracted, protected.placeholders)
    translation = OutputNormalizer.normalize_with_policy(
        restored,
        system_prompt=context.system,
        policy=context.policy,
    )
    technical_terms: list[str] = []
    if context.policy.rules_of_type("term_wrapper", local_only=True):
        technical_terms = TermPlaceholder.extract_terms(case.source)
    validation = OutputValidator.validate(
        translation,
        source_text=case.source,
        system_prompt=context.system,
        protected_terms=list(protected.placeholders.values()),
        policy=context.policy,
        technical_terms=technical_terms,
    )
    return {
        "extracted": extracted,
        "translation": translation,
        "normalization_changed": translation != extracted,
        "validation": {"ok": validation.ok, "reason": validation.reason},
    }


def _canonical(text: str) -> str:
    return re.sub(r"[\[\]\s]", "", text).casefold()


def evaluate_anchors(case: SemanticGuardCase, translation: str) -> dict[str, Any]:
    canonical = _canonical(translation)
    required_all = {
        item: _canonical(item) in canonical for item in case.required_all
    }
    required_any = [
        {
            "options": list(group),
            "ok": any(_canonical(item) in canonical for item in group),
        }
        for group in case.required_any
    ]
    forbidden = {
        item: _canonical(item) not in canonical for item in case.forbidden
    }
    checks = [
        *required_all.values(),
        *(item["ok"] for item in required_any),
        *forbidden.values(),
    ]
    return {
        "ok": all(checks),
        "required_all": required_all,
        "required_any": required_any,
        "forbidden_absent": forbidden,
        "quality_claim": False,
    }


def diagnose_completeness(translation: str) -> dict[str, Any]:
    if not translation or not translation.strip():
        return {"complete": False, "reason": "empty_output", "quality_claim": False}
    stripped = translation.strip()
    for ending in _INCOMPLETE_ENDINGS:
        if stripped.endswith(ending):
            return {
                "complete": False,
                "reason": f"ends_with_{ending}",
                "quality_claim": False,
            }
    return {"complete": True, "reason": "", "quality_claim": False}


def check_ascii_digit_leak(translation: str) -> bool:
    return bool(_ASCII_DIGIT.search(translation))


# ---------------------------------------------------------------------------
# Summary and blind output
# ---------------------------------------------------------------------------


def _subset_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "requests": len(records),
        "production_validation_fail": sum(
            not r["production_validation"]["ok"] for r in records
        ),
        "anchor_pass": sum(bool(r["anchor_evaluation"]["ok"]) for r in records),
        "incomplete": sum(not r["sentence_complete_diagnostic"]["complete"] for r in records),
        "digit_leak": sum(r["ascii_digit_leak"] for r in records),
        "errors": sum(bool(r.get("error")) for r in records),
    }


def summarize(records: list[dict[str, Any]], *, repetitions: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [r for r in records if r["group"] == group]
        completed = [r for r in selected if r.get("normalized_translation")]
        latencies = [float(r["latency_s"]) for r in completed]

        by_scope: dict[str, Any] = {}
        for scope in ("seen_regression", "held_out", "control"):
            subset = [r for r in completed if r["scope"] == scope]
            by_scope[scope] = _subset_stats(subset)

        by_category: dict[str, Any] = {}
        for category in sorted({r["category"] for r in completed}):
            subset = [r for r in completed if r["category"] == category]
            by_category[category] = _subset_stats(subset)

        groups[group] = {
            "requests": len(selected),
            "completed": len(completed),
            "errors": sum(bool(r.get("error")) for r in selected),
            "production_validation_pass": sum(
                bool(r["production_validation"]["ok"]) for r in completed
            ),
            "format_pass": sum(
                bool(r["hard_constraint_evaluation"]["ok"]) for r in completed
            ),
            "anchor_pass": sum(bool(r["anchor_evaluation"]["ok"]) for r in completed),
            "incomplete_count": sum(
                not r["sentence_complete_diagnostic"]["complete"] for r in completed
            ),
            "digit_leak_count": sum(r["ascii_digit_leak"] for r in completed),
            "provider_prompt_tokens": sum(
                int((r.get("usage") or {}).get("prompt_tokens", 0) or 0)
                for r in completed
            ),
            "estimated_input_tokens_mean": round(
                statistics.fmean(float(r["estimated_input_tokens"]) for r in selected), 1
            )
            if selected
            else None,
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "by_scope": by_scope,
            "by_category": by_category,
        }
    return {
        "type": "summary",
        "repetitions": repetitions,
        "groups": groups,
        "decision_gate": {
            "STATIC_SEMANTIC_GUARD_preferred": (
                "If semantic score gap with RETRIEVED_SEMANTIC_POLICY is <= 0.25 and "
                "major error counts are equal, adopt the static guard for simplicity."
            ),
            "RETRIEVED_SEMANTIC_POLICY_wins": (
                "Integrate Pro-compiled typed policy with trigger retrieval only if "
                "held-out semantic mean beats static by >= 0.4, with fewer major errors "
                "and incomplete sentences, format pass drop <= 1, and P95 latency rise <= 30%."
            ),
            "neither_beats_CURRENT_PROFILE": (
                "If both guard approaches still produce material incomplete sentences, "
                "stop stacking prompts and validate lightweight semantic checks or "
                "targeted routing to Pro."
            ),
        },
    }


def write_blind(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
    rows = [
        {
            "blind_id": r["blind_id"],
            "case_id": r["case_id"],
            "category": r["category"],
            "source": r["source"],
            "review_focus": r["review_focus"],
            "translation": r["normalized_translation"],
            "semantic_fidelity_0_to_5": None,
            "naturalness_0_to_5": None,
            "review_notes": "",
        }
        for r in records
        if r.get("normalized_translation")
    ]
    random.Random(seed ^ 0x5C17).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("poc/results")
    return (
        root / f"semantic-guard-{stamp}.jsonl",
        root / f"semantic-guard-state-{stamp}.json",
        root / f"semantic-guard-blind-{stamp}.jsonl",
    )


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_path = Path(args.dataset)
    dataset = load_semantic_guard_dataset(dataset_path)

    # Verify data isolation: training sources vs held-out/control sources.
    incident_sources = {ti.source for ti in dataset.training_incidents}
    for case in dataset.cases:
        if case.scope != "seen_regression" and case.source in incident_sources:
            raise SemanticGuardError(
                f"held-out/control source leaked into training incidents: {case.id}"
            )

    settings = AppSettings.load()
    context = build_production_context(settings)
    fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
    thinking_model = (
        args.thinking_model.strip()
        or settings.ai.thinking_model_name
        or "deepseek-v4-pro"
    )
    if not args.dry_run and (
        not settings.ai.base_url
        or not settings.ai.api_key
        or not fast_model
        or not thinking_model
    ):
        raise SemanticGuardError("API base URL, key, fast model, and thinking model must be configured")

    default_output, default_state, default_blind = _default_paths()
    output_path = Path(args.output) if args.output else default_output
    state_path = Path(args.state_output) if args.state_output else default_state
    blind_path = Path(args.blind_output) if args.blind_output else default_blind
    for path in (output_path, state_path, blind_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    training_hash = hashlib.sha256(
        json.dumps(
            [
                {"id": ti.id, "source": ti.source, "bad": ti.bad, "diagnosis": ti.diagnosis}
                for ti in dataset.training_incidents
            ],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    # --- Pro call #1: profile checklist (digest) ---
    digest_payload = build_digest_payload(context, thinking_model)
    digest_started = time.perf_counter()
    if args.dry_run:
        checklist = make_dry_checklist()
        digest_response = checklist
        digest_usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        digest_latency = 0.0
    else:
        digest_response, digest_usage = _send_request(
            base_url=settings.ai.base_url,
            api_key=settings.ai.api_key,
            payload=digest_payload,
            timeout_seconds=args.pro_timeout,
        )
        digest_latency = round(time.perf_counter() - digest_started, 3)
        checklist = digest_response.strip()

    # --- Pro call #2: typed semantic policy ---
    policy_source = build_policy_source(dataset)
    policy_payload = {
        "model": thinking_model,
        "messages": [
            {"role": "system", "content": _POLICY_COMPILER_SYSTEM},
            {"role": "user", "content": policy_source},
        ],
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }
    policy_started = time.perf_counter()
    if args.dry_run:
        policy = validate_typed_policy(make_dry_typed_policy(), dataset=dataset)
        policy_response = json.dumps(policy, ensure_ascii=False)
        policy_usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        policy_latency = 0.0
    else:
        policy_response, policy_usage = _send_request(
            base_url=settings.ai.base_url,
            api_key=settings.ai.api_key,
            payload=policy_payload,
            timeout_seconds=args.pro_timeout,
        )
        policy_latency = round(time.perf_counter() - policy_started, 3)
        policy = validate_typed_policy(extract_json_object(policy_response), dataset=dataset)

    # --- Write state file ---
    state = {
        "metadata": {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": context.prompt_hash,
            "prompt_length": context.prompt_length,
            "policy_digest": context.policy_digest,
            "enhanced_system_hash": context.enhanced_system_hash,
            "fast_model": fast_model,
            "thinking_model": thinking_model,
            "seed": args.seed,
            "contains_reasoning_content": False,
            "training_hash": training_hash,
        },
        "profile_checklist": checklist,
        "typed_policy": policy,
    }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    run_id = datetime.now(timezone.utc).isoformat()
    metadata = {
        "type": "run",
        "run_id": run_id,
        "experiment": "flash_semantic_integrity_retrieval_strategy",
        "groups": list(GROUPS),
        "cases": len(dataset.cases),
        "repetitions": args.repetitions,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "prompt_hash": context.prompt_hash,
        "prompt_length": context.prompt_length,
        "policy_digest": context.policy_digest,
        "enhanced_system_hash": context.enhanced_system_hash,
        "pro_sees_held_out_sources": False,
        "reasoning_content_replayed": False,
        "api_calls_expected": {
            "pro": 0 if args.dry_run else 2,
            "flash": len(dataset.cases) * len(GROUPS) * args.repetitions,
        },
        "dry_run": args.dry_run,
        "static_guard_rule_count": len(STATIC_GUARD_RULES),
        "max_rules_per_request": MAX_RULES_PER_REQUEST,
    }
    bootstrap_record = {
        "type": "bootstrap",
        "digest": {
            "payload": digest_payload,
            "response": digest_response,
            "checklist": checklist,
            "usage": digest_usage,
            "latency_s": digest_latency,
        },
        "policy": {
            "payload": policy_payload,
            "response": policy_response,
            "typed_policy": policy,
            "usage": policy_usage,
            "latency_s": policy_latency,
        },
    }

    # --- Build and shuffle schedule ---
    rng = random.Random(args.seed)
    schedule = [
        (repetition, group, case)
        for repetition in range(1, args.repetitions + 1)
        for case in dataset.cases
        for group in GROUPS
    ]
    rng.shuffle(schedule)

    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        handle.write(json.dumps(bootstrap_record, ensure_ascii=False) + "\n")

        for index, (repetition, group, case) in enumerate(schedule, 1):
            messages, retrieved_ids = build_group_messages(
                context=context,
                digest_user=_DIGEST_USER,
                checklist=checklist,
                case=case,
                group=group,
                policy=policy,
            )
            payload = {
                "model": fast_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 4096,
                "thinking": {"type": "disabled"},
            }
            started = time.perf_counter()
            error: str | None = None
            usage: dict[str, Any] = {}
            raw_response = ""
            if args.dry_run:
                raw_response = case.fixture_translation
                latency = 0.0
            else:
                try:
                    raw_response, usage = _send_request(
                        base_url=settings.ai.base_url,
                        api_key=settings.ai.api_key,
                        payload=payload,
                        timeout_seconds=args.flash_timeout,
                    )
                    latency = round(time.perf_counter() - started, 3)
                except TranslationError as exc:
                    raw_response = ""
                    latency = round(time.perf_counter() - started, 3)
                    error = str(exc)

            production_output = evaluate_production_output(
                raw_response,
                case=case,
                context=context,
            )
            translation = production_output["translation"]
            hard = (
                evaluate_format(translation)
                if translation
                else {"ok": False, "checks": {}}
            )
            anchor = (
                evaluate_anchors(case, translation)
                if translation
                else {"ok": False, "quality_claim": False}
            )
            completeness = diagnose_completeness(translation)
            digit_leak = check_ascii_digit_leak(translation)

            record = {
                "type": "result",
                "blind_id": hashlib.sha256(
                    f"{run_id}:{group}:{case.id}:{repetition}".encode("utf-8")
                ).hexdigest()[:12],
                "request_index": index,
                "group": group,
                "case_id": case.id,
                "scope": case.scope,
                "category": case.category,
                "repetition": repetition,
                "source": case.source,
                "review_focus": case.review_focus,
                "payload": payload,
                "raw_response": raw_response,
                "extracted_response": production_output["extracted"],
                "normalized_translation": translation,
                "normalization_changed": production_output["normalization_changed"],
                "production_validation": production_output["validation"],
                "hard_constraint_evaluation": hard,
                "anchor_evaluation": anchor,
                "sentence_complete_diagnostic": completeness,
                "ascii_digit_leak": digit_leak,
                "retrieved_rule_ids": retrieved_ids,
                "estimated_input_tokens": estimate_tokens(
                    "\n".join(message["content"] for message in messages)
                ),
                "latency_s": latency,
                "usage": usage,
                "error": error,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{index:02d}/{len(schedule)} {group} {case.id} "
                f"rules={len(retrieved_ids)} elapsed={latency:.3f}s ok={error is None}"
            )
            if args.delay and not args.dry_run:
                time.sleep(args.delay)

        summary = summarize(records, repetitions=args.repetitions)
        summary["bootstrap"] = {
            "digest_latency_s": digest_latency,
            "digest_usage": digest_usage,
            "policy_latency_s": policy_latency,
            "policy_usage": policy_usage,
            "state_output": str(state_path.resolve()),
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    write_blind(blind_path, records, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Policy state: {state_path.resolve()}")
    print(f"Blind review: {blind_path.resolve()}")
    return output_path, state_path, blind_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="poc/data/semantic_guard_dataset.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--state-output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--thinking-model", default="")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--pro-timeout", type=float, default=180.0)
    parser.add_argument("--flash-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions != 2:
        raise SystemExit("formal semantic-guard POC requires exactly 2 repetitions")
    try:
        run(args)
    except (SemanticGuardError, OSError, json.JSONDecodeError, TranslationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
