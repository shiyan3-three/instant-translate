"""Blind POC for replacement-style Runtime Profile handoff.

CURRENT_RUNTIME reproduces the production Pro format digest plus the full
runtime prompt.  PRO_RUNTIME_PROJECTION instead gives Flash a compact system
made from the fixed template, the active local Policy, and a source-grounded
Pro projection.  Both groups reuse production request wrapping, reference
matching, memory retrieval, normalization, and validation.  Evaluation sources
never enter either Pro payload.  There is no judge and no retry path.
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

import httpx

from app.agent.agent import TranslationAgent
from app.feedback.store import FeedbackStore
from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.prompt.policy import LOCAL_RULE_TYPES, ConstraintPolicy
from app.prompt.storage import PromptStorage
from app.reference_layer import ReferencePackage, ReferenceStore
from app.settings import AppSettings
from app.translation.client import TranslationError
from app.translation.quality import OutputNormalizer
from app.translation.service import TranslationService
from app.translation.terms import TermPlaceholder
from poc.poc_agent_memory_transfer import extract_json_object
from poc.poc_agent_semantic_profile import (
    ProductionContext,
    RuntimeCaseContext,
    evaluate_output,
    make_dry_format_profile,
    validate_format_profile,
)
from poc.poc_reference_injection import _percentile, estimate_tokens


GROUPS = ("CURRENT_RUNTIME", "PRO_RUNTIME_PROJECTION")
PROJECTION_PROTOCOL_VERSION = 3
PROJECTION_PROTOCOL_CHANGE = (
    "Reuse the already validated CURRENT_RUNTIME profile from the pre-Flash v2 checkpoint; "
    "issue only one new Projection Pro call; preserve HTTP-success diagnostics before "
    "rejecting empty content."
)
PROFILE_DIRECTIVE_FIELDS = (
    "semantic_directives",
    "style_directives",
    "completeness_directives",
)
PROFILE_TOP_LEVEL_KEYS = {
    "version",
    "role",
    "language_pair",
    *PROFILE_DIRECTIVE_FIELDS,
}
DIRECTIVE_KEYS = {"instruction", "evidence_layer", "evidence_quote"}
EVIDENCE_LAYERS = {"user_constraint", "ai_optimization"}
DATASET_PATH = Path("poc/data/agent_runtime_projection_dataset.json")
EXPECTED_DATASET_SHA256 = "d0f9ca8cfd9a72793b10d4fd089c3c6b9485bc60108a539d5872279afe37154b"
LEGACY_DATASET_PATHS = (
    Path("poc/data/semantic_profile_dataset.json"),
    Path("poc/data/agent_strategy_transfer_dataset.json"),
    Path("poc/data/semantic_guard_dataset.json"),
)
EXPECTED_CASE_SOURCES = (
    "如果健康检查没有通过，就不要切换到新的节点。",
    "请先保存当前设置，再关闭这个窗口。",
    "通知显示上传成功，远端目录里却找不到文件。",
    "由于权限校验失败，这次同步没有继续执行。",
    "用户已经退出登录，页面右上角仍然显示他的头像。",
    "连接等待了十五秒后超时，请检查代理配置。",
    "处理任务已经停止，但消息队列还在继续增长。",
    "请在今天六点之前完成三份报告的审核。",
    "她把洗好的杯子放回柜子里。",
    "吃完晚饭以后，我们一起去散步。",
    "请把靠窗的那本蓝色笔记本递给我。",
    "这家书店每周一上午十点开门。",
)
EXPECTED_CASE_SCOPES = ("runtime_regression",) * 8 + ("control",) * 4

_DIGEST_META = (
    "\n\n[系统元指令] 以下消息是系统设置步骤，不是待翻译文本。"
    "请正常回复，不要翻译。"
)
_FORMAT_PROFILE_USER = (
    "请逐条列出上述翻译规则中最关键的格式要求"
    "（如字符限制、术语格式、空格、标点等）。"
    "每条一行，只列规则要点，不要翻译这句话。"
)
_PROJECTION_COMPILER_SYSTEM = """You compile already-confirmed translation instructions into a compact runtime projection for a fast model.

Return one JSON object only. The exact schema is:
{
  "version": 1,
  "role": "runtime_profile_projection",
  "language_pair": {"source": "中文", "target": "日本語"},
  "semantic_directives": [
    {"instruction": "...", "evidence_layer": "user_constraint or ai_optimization", "evidence_quote": "exact continuous quote"}
  ],
  "style_directives": [],
  "completeness_directives": []
}

Each directive must be supported by an exact continuous quote from the named input layer. Only reorganize and clarify existing confirmed content. Do not add a preference, rule, domain assumption, term, reading, mapping, example, test sentence, translation, or outside knowledge. Do not rewrite the Policy. Do not create glossary tables, terminology lists, fixed readings, source-to-target pairs, Markdown tables, or chain-of-thought. Each list has 0-6 items. Keep instructions concise."""

_PROJECTION_COMPILER_SYSTEM += """

Do not put output-format requirements in any Profile directive. Character sets, hiragana, katakana, kanji, romaji, brackets, term wrapping, whitespace, punctuation, tokenization, word boundaries, and lexical-unit rules belong exclusively to ACTIVE_POLICY and must not be repeated in AGENT_PROFILE.

Protocol validation requirements:
- The three directive lists must contain at least 1 item in total.
- Every instruction must be non-empty and no longer than 240 characters.
- Every evidence_quote must be non-empty and no longer than 160 characters.
- Choose the shortest continuous quote from the named input layer that is sufficient to support the instruction.
- Do not copy a whole paragraph or a whole long rule when a shorter continuous quote is sufficient.
- If no legal short evidence quote supports a directive, do not generate that directive.
- Before returning, verify the exact Schema, all non-empty requirements, both length limits, and the total directive count.
- Never truncate a value to make it pass validation; return only values that already satisfy the limits."""

_FORBIDDEN_PROFILE_PATTERNS = (
    r"\bglossar(?:y|ies)\b",
    r"\bterminology\s+lists?\b",
    r"\bfixed\s+readings?\b",
    r"\bsource\s*[-=]>\s*target\b",
    r"\bexamples?\s*:",
    r"\btest\s+sentences?\b",
    r"\btranslation\s+results?\b",
    r"\bchain[- ]of[- ]thought\b",
    r"术语表",
    r"词汇表",
    r"固定(?:读法|译法)",
    r"(?:例如|比如)",
    r"例句",
    r"测试句",
    r"翻译结果",
    r"思维链",
)

_FORBIDDEN_FORMAT_PATTERNS = (
    r"\bhiragana\b",
    r"\bkatakana\b",
    r"\bkanji\b",
    r"\bromaji\b",
    r"平假名",
    r"括号",
    r"\bbrackets?\b",
    r"空格",
    r"\bspaces?\b",
    r"\bpunctuation\b",
    r"u\+",
    r"\btokeni[sz]e\b",
    r"\bword\s+boundar(?:y|ies)\b",
    r"\blexical\s+units?\b",
)


class RuntimeProjectionPocError(ValueError):
    """Raised for invalid experiment data or ungrounded Pro output."""


@dataclass(frozen=True)
class ProjectionCase:
    id: str
    scope: str
    category: str
    source: str
    review_focus: str
    fixture_translation: str


@dataclass(frozen=True)
class ProjectionDataset:
    version: int
    status: str
    name: str
    description: str
    cases: tuple[ProjectionCase, ...]


@dataclass(frozen=True)
class ProjectionContext:
    production: ProductionContext
    user_constraint_layer: str
    ai_optimization_layer: str
    minimal_policy: dict[str, Any]
    compiled_prompt: str
    feedback_store: FeedbackStore


@dataclass(frozen=True)
class FrozenRequestContext:
    """One case's production retrieval state, frozen before Flash jobs start."""

    runtime_case: RuntimeCaseContext
    current_user_content: str
    projection_user_content: str
    matched_memory_ids: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path) -> ProjectionDataset:
    if path.resolve() == DATASET_PATH.resolve():
        actual_hash = _sha256_file(path)
        if actual_hash != EXPECTED_DATASET_SHA256:
            raise RuntimeProjectionPocError(
                f"dataset SHA256 changed: expected={EXPECTED_DATASET_SHA256} actual={actual_hash}"
            )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise RuntimeProjectionPocError("dataset must be an object containing cases")
    dataset = ProjectionDataset(
        version=int(raw.get("version", 0)),
        status=str(raw.get("status", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        cases=tuple(
            ProjectionCase(
                id=str(item.get("id", "")).strip(),
                scope=str(item.get("scope", "")).strip(),
                category=str(item.get("category", "")).strip(),
                source=str(item.get("source", "")).strip(),
                review_focus=str(item.get("review_focus", "")).strip(),
                fixture_translation=str(item.get("fixture_translation", "")).strip(),
            )
            for item in raw["cases"]
            if isinstance(item, dict)
        ),
    )
    validate_dataset(dataset)
    return dataset


def _legacy_sources() -> set[str]:
    sources: set[str] = set()
    for path in LEGACY_DATASET_PATHS:
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw.get("cases", []):
            if isinstance(item, dict) and str(item.get("source", "")).strip():
                sources.add(str(item["source"]).strip())
    return sources


def validate_dataset(dataset: ProjectionDataset) -> None:
    errors: list[str] = []
    if dataset.version != 1 or dataset.status != "frozen_poc":
        errors.append("dataset must be frozen_poc version 1")
    if tuple(case.source for case in dataset.cases) != EXPECTED_CASE_SOURCES:
        errors.append("case source list/order differs from the frozen benchmark")
    if tuple(case.scope for case in dataset.cases) != EXPECTED_CASE_SCOPES:
        errors.append("case scope list/order differs from the frozen benchmark")
    ids = [case.id for case in dataset.cases]
    sources = [case.source for case in dataset.cases]
    if len(ids) != 12 or len(set(ids)) != 12 or any(not value for value in ids):
        errors.append("dataset must contain exactly 12 unique non-empty ids")
    if len(set(sources)) != 12:
        errors.append("all evaluation sources must be unique")
    overlap = sorted(set(sources) & _legacy_sources())
    if overlap:
        errors.append(f"evaluation sources overlap legacy datasets: {overlap}")
    for case in dataset.cases:
        if not case.category or not case.review_focus or not case.fixture_translation:
            errors.append(f"{case.id} is missing category, review focus, or dry fixture")
    if errors:
        raise RuntimeProjectionPocError("; ".join(errors))


def _read_compiled_prompt(settings: AppSettings) -> str:
    """Read the confirmed Markdown without invoking migration/write paths."""

    storage = PromptStorage()
    path = storage.existing_compiled_prompt_path(settings.prompt.compiled_prompt_path)
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _extract_markdown_layer(raw: str, title: str) -> str:
    marker = f"## {title}"
    match = re.search(rf"(?m)^{re.escape(marker)}\s*$", raw)
    if not match:
        return ""
    remainder = raw[match.end():]
    end = re.search(r"(?m)^##\s+", remainder)
    section = remainder[: end.start()] if end else remainder
    section = section.strip()
    # Compiled prompts put a system-owned explanatory paragraph before the
    # actual confirmed layer content. It is not user evidence.
    if section.startswith("This layer") and "\n\n" in section:
        section = section.split("\n\n", 1)[1].strip()
    return section


def extract_confirmed_layers(raw: str, settings: AppSettings) -> tuple[str, str]:
    if raw:
        user_layer = _extract_markdown_layer(raw, "User Constraint Layer")
        ai_layer = _extract_markdown_layer(raw, "AI Optimization Layer")
    else:
        user_layer = ""
        ai_layer = ""
    if not user_layer:
        user_layer = settings.prompt.constraints_text.strip()
    return user_layer, ai_layer


def minimal_policy_dict(policy: ConstraintPolicy) -> dict[str, Any]:
    return {
        "version": policy.version,
        "rules": [
            {
                "type": rule.type,
                "params": dict(rule.params),
                "enforcement": rule.enforcement,
                "scope": dict(rule.scope),
            }
            for rule in policy.rules
            if rule.type in LOCAL_RULE_TYPES
        ],
    }


def build_projection_context(settings: AppSettings) -> ProjectionContext:
    compiled_prompt = _read_compiled_prompt(settings)
    service = TranslationService(settings)
    try:
        definition = service._resolve_agent_definition("中文", "日本語")
    finally:
        service.shutdown()
    term_wrapper_mode = TranslationAgent._term_wrapper_selection_mode(definition.policy)
    enhanced_system = TranslationAgent._append_runtime_tool_rules(
        TranslationAgent._rewrite_constraints(definition.prompt),
        preserve_placeholders=not definition.policy.has_script("hiragana"),
        term_wrapper_mode=term_wrapper_mode,
    )
    production = ProductionContext(
        system=enhanced_system,
        policy=definition.policy,
        references=definition.reference_package,
        prompt_hash=definition.runtime_meta.prompt_hash,
        prompt_length=len(definition.prompt),
        policy_digest=definition.policy.digest(),
        reference_digest=definition.reference_package.digest(),
        enhanced_system_hash=hashlib.sha256(enhanced_system.encode("utf-8")).hexdigest(),
        term_wrapper_mode=term_wrapper_mode,
    )
    user_layer, ai_layer = extract_confirmed_layers(compiled_prompt, settings)
    return ProjectionContext(
        production=production,
        user_constraint_layer=user_layer,
        ai_optimization_layer=ai_layer,
        minimal_policy=minimal_policy_dict(definition.policy),
        compiled_prompt=compiled_prompt,
        feedback_store=FeedbackStore(),
    )


def build_pro_payloads(
    context: ProjectionContext,
    thinking_model: str,
) -> dict[str, dict[str, Any]]:
    projection_input = {
        "source_language": "中文",
        "target_language": "日本語",
        "user_constraint_layer": context.user_constraint_layer,
        "ai_optimization_layer": context.ai_optimization_layer,
        "active_policy": context.minimal_policy,
        "runtime_context_note": (
            "Knowledge Reference and confirmed feedback memory are injected per request; "
            "do not reproduce or invent them in this profile."
        ),
    }
    return {
        "CURRENT_RUNTIME": {
            "model": thinking_model,
            "messages": [
                {"role": "system", "content": context.production.system + _DIGEST_META},
                {"role": "user", "content": _FORMAT_PROFILE_USER},
            ],
            "temperature": 0.0,
            "max_tokens": 8192,
            "thinking": {"type": "enabled"},
        },
        "PRO_RUNTIME_PROJECTION": {
            "model": thinking_model,
            "messages": [
                {"role": "system", "content": _PROJECTION_COMPILER_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(projection_input, ensure_ascii=False, indent=2),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 8192,
            "thinking": {"type": "enabled"},
        },
    }


def load_reused_current_profile_state(
    path: Path,
    *,
    context: ProjectionContext,
    dataset: ProjectionDataset,
    fast_model: str,
    thinking_model: str,
) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RuntimeProjectionPocError(f"reused CURRENT profile state does not exist: {path}")
    resolved = path.resolve()
    try:
        raw_bytes = resolved.read_bytes()
        saved = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeProjectionPocError(f"cannot read reused CURRENT profile state: {exc}") from exc
    if not isinstance(saved, dict):
        raise RuntimeProjectionPocError("reused CURRENT profile state must be an object")
    if saved.get("projection_protocol_version") != 2:
        raise RuntimeProjectionPocError("reused CURRENT profile state is not protocol v2")
    metadata = saved.get("metadata")
    profiles = saved.get("profiles")
    if not isinstance(metadata, dict) or not isinstance(profiles, dict):
        raise RuntimeProjectionPocError("reused CURRENT profile state is missing metadata/profiles")
    current = profiles.get("CURRENT_RUNTIME")
    if not isinstance(current, dict) or current.get("validation_status") != "passed":
        raise RuntimeProjectionPocError("reused CURRENT_RUNTIME validation_status is not passed")

    expected_metadata = {
        "prompt_hash": context.production.prompt_hash,
        "policy_digest": context.production.policy_digest,
        "reference_digest": context.production.reference_digest,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
    }
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if actual != expected:
            raise RuntimeProjectionPocError(
                f"reused CURRENT profile {key} mismatch: expected={expected!r} actual={actual!r}"
            )

    try:
        raw_profile = validate_format_profile(current.get("raw_response", ""), dataset=dataset)
        rendered_profile = validate_format_profile(
            current.get("rendered_profile", ""),
            dataset=dataset,
        )
    except ValueError as exc:
        raise RuntimeProjectionPocError(
            f"reused CURRENT profile failed format revalidation: {exc}"
        ) from exc
    if raw_profile != rendered_profile:
        raise RuntimeProjectionPocError("reused CURRENT raw/rendered profiles differ")
    profile_hash = hashlib.sha256(rendered_profile.encode("utf-8")).hexdigest()
    stored_hash = str(current.get("rendered_profile_hash", "")).strip()
    if stored_hash and stored_hash != profile_hash:
        raise RuntimeProjectionPocError("reused CURRENT rendered profile hash mismatch")

    profile_record = dict(current)
    profile_record.update(
        {
            "group": "CURRENT_RUNTIME",
            "raw_response": raw_profile,
            "rendered_profile": rendered_profile,
            "rendered_profile_hash": profile_hash,
            "validation_status": "passed",
            "reused": True,
            "reused_from_state": str(resolved),
        }
    )
    return {
        "absolute_path": str(resolved),
        "state_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "profile_hash": profile_hash,
        "validation": "passed",
        "profile_record": profile_record,
    }


def _contains_forbidden_profile_content(text: str) -> bool:
    if any(marker in text for marker in ("->", "=>", "→")):
        return True
    if re.search(r"(?m)^\s*\|.*\|\s*$", text):
        return True
    return any(re.search(pattern, text, re.I) for pattern in _FORBIDDEN_PROFILE_PATTERNS)


def _contains_forbidden_format_concept(text: str) -> bool:
    if "[" in text or "]" in text:
        return True
    return any(re.search(pattern, text, re.I) for pattern in _FORBIDDEN_FORMAT_PATTERNS)


def validate_projection_profile(
    raw: Any,
    *,
    context: ProjectionContext,
    dataset: ProjectionDataset,
) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != PROFILE_TOP_LEVEL_KEYS:
        raise RuntimeProjectionPocError("projection profile top-level keys do not match schema")
    if raw.get("version") != 1 or raw.get("role") != "runtime_profile_projection":
        raise RuntimeProjectionPocError("projection profile version/role is invalid")
    pair = raw.get("language_pair")
    if not isinstance(pair, dict) or set(pair) != {"source", "target"}:
        raise RuntimeProjectionPocError("language_pair schema is invalid")
    if pair != {"source": "中文", "target": "日本語"}:
        raise RuntimeProjectionPocError("language_pair does not match the experiment")

    layers = {
        "user_constraint": context.user_constraint_layer,
        "ai_optimization": context.ai_optimization_layer,
    }
    clean: dict[str, Any] = {
        "version": 1,
        "role": "runtime_profile_projection",
        "language_pair": dict(pair),
    }
    directive_count = 0
    for field in PROFILE_DIRECTIVE_FIELDS:
        values = raw.get(field)
        if not isinstance(values, list) or len(values) > 6:
            raise RuntimeProjectionPocError(f"{field} must contain 0-6 directives")
        clean_values: list[dict[str, str]] = []
        for value in values:
            if not isinstance(value, dict) or set(value) != DIRECTIVE_KEYS:
                raise RuntimeProjectionPocError(f"{field} directive keys are invalid")
            instruction = str(value.get("instruction", "")).strip()
            evidence_layer = str(value.get("evidence_layer", "")).strip()
            evidence_quote = str(value.get("evidence_quote", "")).strip()
            if not instruction or len(instruction) > 240:
                raise RuntimeProjectionPocError(f"{field} has an empty or oversized instruction")
            if evidence_layer not in EVIDENCE_LAYERS:
                raise RuntimeProjectionPocError(f"{field} has an invalid evidence_layer")
            if not evidence_quote or len(evidence_quote) > 160:
                raise RuntimeProjectionPocError(f"{field} has an empty or oversized evidence_quote")
            if evidence_quote not in layers[evidence_layer]:
                raise RuntimeProjectionPocError(f"{field} evidence_quote is not in its source layer")
            if _contains_forbidden_profile_content(instruction + "\n" + evidence_quote):
                raise RuntimeProjectionPocError(f"{field} contains forbidden generated knowledge")
            if _contains_forbidden_format_concept(instruction):
                raise RuntimeProjectionPocError(f"{field} contains an ACTIVE_POLICY format concept")
            clean_values.append(
                {
                    "instruction": instruction,
                    "evidence_layer": evidence_layer,
                    "evidence_quote": evidence_quote,
                }
            )
            directive_count += 1
        clean[field] = clean_values

    if directive_count < 1:
        raise RuntimeProjectionPocError("projection profile must contain at least one directive")

    serialized = json.dumps(clean, ensure_ascii=False)
    for case in dataset.cases:
        if case.source in serialized:
            raise RuntimeProjectionPocError(f"evaluation source leaked into profile: {case.id}")
    rendered = render_projection_profile(clean)
    if len(rendered) > 1800:
        raise RuntimeProjectionPocError("rendered Runtime Profile exceeds 1800 characters")
    return clean


def projection_evidence_audit(profile: dict[str, Any], context: ProjectionContext) -> dict[str, Any]:
    layers = {
        "user_constraint": context.user_constraint_layer,
        "ai_optimization": context.ai_optimization_layer,
    }
    directives = [item for field in PROFILE_DIRECTIVE_FIELDS for item in profile[field]]
    valid_quotes = sum(
        item["evidence_quote"] in layers[item["evidence_layer"]]
        for item in directives
    )
    return {
        "valid": valid_quotes == len(directives),
        "directive_count": len(directives),
        "validated_evidence_quotes": valid_quotes,
        "rendered_contains_evidence_fields": False,
        "forbidden_generated_knowledge": _contains_forbidden_profile_content(
            render_projection_profile(profile)
        ),
        "forbidden_format_concepts": any(
            _contains_forbidden_format_concept(item["instruction"])
            for item in directives
        ),
    }


def render_projection_profile(profile: dict[str, Any]) -> str:
    lines: list[str] = []
    for field in PROFILE_DIRECTIVE_FIELDS:
        values = profile[field]
        if not values:
            continue
        lines.append(f"[{field}]")
        lines.extend(f"- {item['instruction']}" for item in values)
    return "\n".join(lines).strip()


def _safe_evidence_quote(layer: str) -> str:
    preferred: list[str] = []
    fallback: list[str] = []
    for raw_line in layer.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or _contains_forbidden_profile_content(line):
            continue
        complete = re.match(r"^(.{1,159}?[。！？.!?])(?:\s|$)", line)
        quote = (complete.group(1) if complete else line[:160]).strip()
        if not quote:
            continue
        fallback.append(quote)
        if any(marker in line.casefold() for marker in ("natural", "meaning", "faithful", "自然", "原意", "含义")):
            preferred.append(quote)
    return (preferred or fallback or [""])[0]


def make_dry_projection_profile(context: ProjectionContext) -> dict[str, Any]:
    layer_name = "ai_optimization"
    quote = _safe_evidence_quote(context.ai_optimization_layer)
    if not quote:
        layer_name = "user_constraint"
        quote = _safe_evidence_quote(context.user_constraint_layer)
    directives: list[dict[str, str]] = []
    if quote:
        instruction = re.sub(r"^[-*]\s*", "", quote).strip()
        directives.append(
            {
                "instruction": instruction,
                "evidence_layer": layer_name,
                "evidence_quote": quote,
            }
        )
    return {
        "version": 1,
        "role": "runtime_profile_projection",
        "language_pair": {"source": "中文", "target": "日本語"},
        "semantic_directives": directives,
        "style_directives": [],
        "completeness_directives": [],
    }


def _term_wrapper_domains(policy: ConstraintPolicy) -> list[str]:
    return [
        str(rule.params.get("domain", "")).strip()
        for rule in policy.rules_of_type("term_wrapper", local_only=True)
        if str(rule.params.get("domain", "")).strip()
    ]


def freeze_runtime_case(
    context: ProjectionContext,
    case: ProjectionCase,
    *,
    current_format_profile: str,
) -> FrozenRequestContext:
    production = context.production
    plan = ReferenceStore(production.references.entries).protect(case.source)
    reference_hints = production.references.runtime_hints(
        case.source,
        matched_entries=plan.matched_entries,
    )
    memory_rules = context.feedback_store.match_memory_rules(
        case.source,
        source_language="中文",
        target_language="日本語",
        limit=3,
    )
    memory_hints = [rule.as_prompt_hint() for rule in memory_rules]
    preserve_verbatim = not production.policy.has_script("hiragana")
    protected = (
        TermPlaceholder.protect(plan.source)
        if preserve_verbatim
        else TermPlaceholder.empty(plan.source)
    )
    technical_terms: list[str] = []
    if (
        production.term_wrapper_mode != "references_only"
        and TranslationAgent._term_wrapper_marks_ascii_terms(production.policy)
    ):
        technical_terms = TermPlaceholder.extract_terms(case.source)
    strict_mode = production.term_wrapper_mode in {"references_only", "references_and_ascii"}
    expected_targets = tuple(
        OutputNormalizer.normalize_with_policy(
            target,
            system_prompt=production.system,
            policy=production.policy,
        )
        for target in plan.replacements.values()
    ) if strict_mode else None
    extra_budget = (
        len(technical_terms)
        if production.term_wrapper_mode == "references_and_ascii"
        else 0
    ) if strict_mode else None
    wrap_kwargs = {
        "memory_hints": memory_hints,
        "reference_hints": reference_hints,
        "technical_terms": technical_terms,
        "term_wrapper_domains": _term_wrapper_domains(production.policy),
        "term_wrapper_mode": production.term_wrapper_mode,
        "placeholder_protocol": plan.protocol_block(),
    }
    projection_user_content = TranslationAgent._wrap_source_text(
        protected.text,
        **wrap_kwargs,
    )
    current_user_content = TranslationAgent._wrap_source_text(
        protected.text,
        rule_checklist=current_format_profile,
        **wrap_kwargs,
    )
    runtime_case = RuntimeCaseContext(
        plan=plan,
        protected_text=protected.text,
        protected_terms=tuple(protected.placeholders.values()),
        technical_terms=tuple(technical_terms),
        expected_wrapped_targets=expected_targets,
        extra_wrapped_term_budget=extra_budget,
        user_content=projection_user_content,
    )
    return FrozenRequestContext(
        runtime_case=runtime_case,
        current_user_content=current_user_content,
        projection_user_content=projection_user_content,
        matched_memory_ids=tuple(rule.id for rule in memory_rules),
    )


def build_projection_system(context: ProjectionContext, rendered_profile: str) -> str:
    policy_json = json.dumps(
        context.minimal_policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    base = (
        f"{DEFAULT_BASE_PROMPT}\n\n"
        f"<ACTIVE_POLICY>\n{policy_json}\n</ACTIVE_POLICY>\n\n"
        "ACTIVE_POLICY is the user-confirmed authoritative output contract. "
        "Execute every rule by its type and params. AGENT_PROFILE contains only semantic "
        "and style guidance and must never override ACTIVE_POLICY.\n\n"
        f"<AGENT_PROFILE>\n{rendered_profile}\n</AGENT_PROFILE>\n\n"
        "Translate from 中文 to 日本語."
    )
    return TranslationAgent._append_runtime_tool_rules(
        base,
        preserve_placeholders=not context.production.policy.has_script("hiragana"),
        term_wrapper_mode=context.production.term_wrapper_mode,
    )


def build_flash_messages(
    *,
    group: str,
    context: ProjectionContext,
    frozen_request: FrozenRequestContext,
    current_format_profile: str,
    rendered_projection: str,
) -> list[dict[str, str]]:
    if group == "CURRENT_RUNTIME":
        return [
            {"role": "system", "content": context.production.system},
            {"role": "user", "content": _FORMAT_PROFILE_USER},
            {"role": "assistant", "content": current_format_profile},
            {"role": "user", "content": frozen_request.current_user_content},
        ]
    if group == "PRO_RUNTIME_PROJECTION":
        return [
            {
                "role": "system",
                "content": build_projection_system(context, rendered_projection),
            },
            {"role": "user", "content": frozen_request.projection_user_content},
        ]
    raise RuntimeProjectionPocError(f"unknown group: {group}")


def _send_diagnostic_request(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Return auditable 2xx diagnostics without persisting reasoning content."""

    started = time.perf_counter()
    try:
        response = httpx.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            raise TypeError("response choice.message is not an object")
        raw_content = message.get("content")
        content = "" if raw_content is None else str(raw_content)
        raw_reasoning = message.get("reasoning_content")
        reasoning_len = len("" if raw_reasoning is None else str(raw_reasoning))
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return {
            "http_status": int(response.status_code),
            "message_keys": sorted(str(key) for key in message),
            "content_len": len(content),
            "reasoning_len": reasoning_len,
            "finish_reason": choice.get("finish_reason"),
            "usage": dict(usage),
            "latency_s": round(time.perf_counter() - started, 3),
            "content": content,
        }
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise TranslationError(str(exc)) from exc


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("poc/results")
    return (
        root / f"runtime-projection-{stamp}.jsonl",
        root / f"runtime-projection-state-{stamp}.json",
        root / f"runtime-projection-blind-{stamp}.jsonl",
    )


def _summarize(records: list[dict[str, Any]], repetitions: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [row for row in records if row["group"] == group]
        completed = [row for row in selected if not row.get("error")]
        latencies = [float(row["latency_s"]) for row in completed]
        finish_reasons: dict[str, int] = {}
        for row in completed:
            key = str(row.get("finish_reason") or "missing")
            finish_reasons[key] = finish_reasons.get(key, 0) + 1
        groups[group] = {
            "completed": len(completed),
            "expected": len(selected),
            "production_validation_pass": sum(
                bool(row["production_validation"]["ok"]) for row in completed
            ),
            "finish_reasons": finish_reasons,
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "provider_prompt_tokens": sum(
                int((row.get("usage") or {}).get("prompt_tokens", 0) or 0)
                for row in completed
            ),
            "estimated_input_tokens_mean": round(
                statistics.fmean(float(row["estimated_input_tokens"]) for row in selected),
                1,
            ) if selected else None,
        }
    return {
        "type": "summary",
        "repetitions": repetitions,
        "groups": groups,
        "semantic_scores_are_manual_only": True,
        "decision_gate": {
            "integrate_projection_only_if": (
                "Blind semantic fidelity improves by >=0.4, naturalness improves by >=0.3, "
                "runtime-regression major semantic errors fall by >=30%, production format pass "
                "drops by at most 1 result, Flash P95 rises by <=15%, mean estimated input tokens "
                "fall by >=20%, and the Profile has zero ungrounded directives, term mappings, "
                "or evaluation-source leakage."
            ),
            "otherwise": (
                "Do not modify the production Agent Profile or real Session; move next to Flash "
                "model capability or the user-initiated optimization path."
            ),
        },
    }


def write_blind(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
    rows = [
        {
            "blind_id": row["blind_id"],
            "source": row["source"],
            "review_focus": row["review_focus"],
            "translation": row["normalized_translation"],
            "semantic_fidelity_0_to_5": None,
            "naturalness_0_to_5": None,
            "meaning_error": "",
            "review_notes": "",
        }
        for row in records
    ]
    random.Random(seed ^ 0xB71D).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_state_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mark_state_failed(
    state: dict[str, Any],
    state_path: Path,
    *,
    stage: str,
    group: str,
    error: Exception,
) -> None:
    state.update(
        {
            "status": "failed",
            "failure_stage": stage,
            "failure_group": group,
            "error": str(error),
        }
    )
    _write_state_checkpoint(state_path, state)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.repetitions != 2:
        raise RuntimeProjectionPocError("this frozen POC requires exactly 2 repetitions")
    dataset = load_dataset(Path(args.dataset))
    settings = AppSettings.load()
    context = build_projection_context(settings)
    fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
    thinking_model = (
        args.thinking_model.strip()
        or settings.ai.thinking_model_name
        or "deepseek-v4-pro"
    )
    if not args.dry_run and (
        not settings.ai.base_url or not settings.ai.api_key or not fast_model or not thinking_model
    ):
        raise RuntimeProjectionPocError(
            "API base URL, key, fast model, and thinking model are required"
        )
    reuse_value = str(getattr(args, "reuse_current_profile_state", "") or "").strip()
    if not reuse_value:
        raise RuntimeProjectionPocError("--reuse-current-profile-state is required for protocol v3")
    reused_current = load_reused_current_profile_state(
        Path(reuse_value),
        context=context,
        dataset=dataset,
        fast_model=fast_model,
        thinking_model=thinking_model,
    )

    defaults = _default_paths()
    output_path = Path(args.output) if args.output else defaults[0]
    state_path = Path(args.state_output) if args.state_output else defaults[1]
    blind_path = Path(args.blind_output) if args.blind_output else defaults[2]
    for path in (output_path, state_path, blind_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    pro_payloads = build_pro_payloads(context, thinking_model)
    serialized_payloads = json.dumps(pro_payloads, ensure_ascii=False)
    for case in dataset.cases:
        if case.source in serialized_payloads:
            raise RuntimeProjectionPocError(f"evaluation source leaked into Pro payload: {case.id}")

    current_system = context.production.system
    formal_call_design = {
        "reused_current_pro_profiles": 1,
        "new_projection_pro_calls": 1,
        "flash": 48,
        "pro_judge": 0,
        "retry": 0,
    }
    state: dict[str, Any] = {
        "status": "running",
        "projection_protocol_version": PROJECTION_PROTOCOL_VERSION,
        "projection_protocol_change": PROJECTION_PROTOCOL_CHANGE,
        "reused_current_profile_state": reused_current["absolute_path"],
        "reused_current_profile_state_sha256": reused_current["state_sha256"],
        "reused_current_profile_hash": reused_current["profile_hash"],
        "reused_current_profile_validation": reused_current["validation"],
        "reused_current_profile_state": reused_current["absolute_path"],
        "reused_current_profile_state_sha256": reused_current["state_sha256"],
        "reused_current_profile_hash": reused_current["profile_hash"],
        "reused_current_profile_validation": reused_current["validation"],
        "attempted_pro_calls": 0,
        "usable_pro_responses": 0,
        "attempted_flash_calls": 0,
        "usable_flash_responses": 0,
        "executed_pro_calls": 0,
        "executed_flash_calls": 0,
        "metadata": {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "prompt_hash": context.production.prompt_hash,
            "prompt_length": context.production.prompt_length,
            "policy_digest": context.production.policy_digest,
            "reference_digest": context.production.reference_digest,
            "term_wrapper_mode": context.production.term_wrapper_mode,
            "fast_model": fast_model,
            "thinking_model": thinking_model,
            "projection_protocol_version": PROJECTION_PROTOCOL_VERSION,
            "projection_protocol_change": PROJECTION_PROTOCOL_CHANGE,
            "reused_current_profile_state": reused_current["absolute_path"],
            "reused_current_profile_state_sha256": reused_current["state_sha256"],
            "reused_current_profile_hash": reused_current["profile_hash"],
            "reused_current_profile_validation": reused_current["validation"],
            "pro_sees_evaluation_sources": False,
            "formal_call_design": formal_call_design,
            "dry_run": args.dry_run,
        },
        "system_prompts": {
            "CURRENT_RUNTIME": {
                "sha256": hashlib.sha256(current_system.encode("utf-8")).hexdigest(),
                "length": len(current_system),
            },
            "PRO_RUNTIME_PROJECTION": {
                "sha256": "",
                "length": 0,
                "status": "pending_profile_validation",
            },
        },
        "profiles": {
            "CURRENT_RUNTIME": reused_current["profile_record"],
        },
        "frozen_request_contexts": {},
        "flash_diagnostics": [],
    }
    _write_state_checkpoint(state_path, state)

    profile_records: dict[str, Any] = state["profiles"]
    projection_payload = pro_payloads["PRO_RUNTIME_PROJECTION"]
    if args.dry_run:
        diagnostic = {
            "http_status": None,
            "message_keys": ["content"],
            "content_len": 0,
            "reasoning_len": 0,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_s": 0.0,
            "content": json.dumps(make_dry_projection_profile(context), ensure_ascii=False),
        }
        diagnostic["content_len"] = len(diagnostic["content"])
    else:
        state["attempted_pro_calls"] += 1
        _write_state_checkpoint(state_path, state)
        try:
            diagnostic = _send_diagnostic_request(
                base_url=settings.ai.base_url,
                api_key=settings.ai.api_key,
                payload=projection_payload,
                timeout_seconds=args.pro_timeout,
            )
        except Exception as exc:
            _mark_state_failed(
                state,
                state_path,
                stage="projection_pro_request",
                group="PRO_RUNTIME_PROJECTION",
                error=exc,
            )
            raise

    checkpoint: dict[str, Any] = {
        "group": "PRO_RUNTIME_PROJECTION",
        "payload": projection_payload,
        "raw_response": diagnostic["content"],
        "http_status": diagnostic["http_status"],
        "message_keys": diagnostic["message_keys"],
        "content_len": diagnostic["content_len"],
        "reasoning_len": diagnostic["reasoning_len"],
        "usage": diagnostic["usage"],
        "finish_reason": diagnostic["finish_reason"],
        "latency_s": diagnostic["latency_s"],
        "validation_status": "pending",
    }
    profile_records["PRO_RUNTIME_PROJECTION"] = checkpoint
    _write_state_checkpoint(state_path, state)

    if not diagnostic["content"].strip():
        checkpoint["validation_status"] = "failed"
        exc = TranslationError("Projection Pro returned empty content.")
        checkpoint["validation_error"] = str(exc)
        _mark_state_failed(
            state,
            state_path,
            stage="projection_pro_empty_content",
            group="PRO_RUNTIME_PROJECTION",
            error=exc,
        )
        raise exc

    if not args.dry_run:
        state["usable_pro_responses"] += 1
        state["executed_pro_calls"] = state["attempted_pro_calls"]
        _write_state_checkpoint(state_path, state)

    try:
        parsed = validate_projection_profile(
            extract_json_object(diagnostic["content"]),
            context=context,
            dataset=dataset,
        )
        rendered = render_projection_profile(parsed)
        evidence_validation = projection_evidence_audit(parsed, context)
        if not evidence_validation["valid"]:
            raise RuntimeProjectionPocError("projection evidence audit failed")
    except Exception as exc:
        checkpoint["validation_status"] = "failed"
        checkpoint["validation_error"] = str(exc)
        _mark_state_failed(
            state,
            state_path,
            stage="projection_profile_validation",
            group="PRO_RUNTIME_PROJECTION",
            error=exc,
        )
        raise

    checkpoint.update(
        {
            "parsed_profile": parsed,
            "rendered_profile": rendered,
            "rendered_profile_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "evidence_validation": evidence_validation,
            "validation_status": "passed",
        }
    )
    _write_state_checkpoint(state_path, state)

    # Freeze every mutable per-request dependency once, before the 48 Flash
    # jobs are built. Repetitions and groups must never re-read feedback or
    # re-run reference matching against state that may change mid-experiment.
    frozen_requests = {
        case.id: freeze_runtime_case(
            context,
            case,
            current_format_profile=profile_records["CURRENT_RUNTIME"]["rendered_profile"],
        )
        for case in dataset.cases
    }

    projection_system = build_projection_system(
        context,
        profile_records["PRO_RUNTIME_PROJECTION"]["rendered_profile"],
    )
    state["system_prompts"]["PRO_RUNTIME_PROJECTION"] = {
        "sha256": hashlib.sha256(projection_system.encode("utf-8")).hexdigest(),
        "length": len(projection_system),
        "status": "ready",
    }
    state["frozen_request_contexts"] = {
            case.id: {
                "current_user_content_hash": hashlib.sha256(
                    frozen_requests[case.id].current_user_content.encode("utf-8")
                ).hexdigest(),
                "projection_user_content_hash": hashlib.sha256(
                    frozen_requests[case.id].projection_user_content.encode("utf-8")
                ).hexdigest(),
                "matched_reference_ids": [
                    entry.id for entry in frozen_requests[case.id].runtime_case.plan.matched_entries
                ],
                "matched_memory_ids": list(frozen_requests[case.id].matched_memory_ids),
            }
            for case in dataset.cases
    }
    _write_state_checkpoint(state_path, state)

    metadata = {
        "type": "run",
        "experiment": "runtime_profile_replacement_handoff",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "groups": list(GROUPS),
        "cases": len(dataset.cases),
        "repetitions": args.repetitions,
        "runtime_regressions": 8,
        "controls": 4,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "projection_protocol_version": PROJECTION_PROTOCOL_VERSION,
        "projection_protocol_change": PROJECTION_PROTOCOL_CHANGE,
        "prompt_hash": context.production.prompt_hash,
        "policy_digest": context.production.policy_digest,
        "reference_digest": context.production.reference_digest,
        "pro_sees_evaluation_sources": False,
        "pro_judge_calls": 0,
        "automatic_retries": 0,
        "formal_call_design": formal_call_design,
        "attempted_api_calls": {
            "pro": state["attempted_pro_calls"],
            "flash": state["attempted_flash_calls"],
        },
        "usable_api_responses": {
            "pro": state["usable_pro_responses"],
            "flash": state["usable_flash_responses"],
        },
        "executed_api_calls": {
            "pro": state["executed_pro_calls"],
            "flash": state["executed_flash_calls"],
        },
        "dry_run": args.dry_run,
    }

    jobs = [
        (case, group, repetition)
        for repetition in range(1, args.repetitions + 1)
        for case in dataset.cases
        for group in GROUPS
    ]
    random.Random(args.seed).shuffle(jobs)
    records: list[dict[str, Any]] = []
    for index, (case, group, repetition) in enumerate(jobs, 1):
        frozen_request = frozen_requests[case.id]
        runtime_case = frozen_request.runtime_case
        messages = build_flash_messages(
            group=group,
            context=context,
            frozen_request=frozen_request,
            current_format_profile=profile_records["CURRENT_RUNTIME"]["rendered_profile"],
            rendered_projection=profile_records["PRO_RUNTIME_PROJECTION"]["rendered_profile"],
        )
        payload = {
            "model": fast_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4096,
            "thinking": {"type": "disabled"},
        }
        started = time.perf_counter()
        error = ""
        try:
            if args.dry_run:
                raw_response = case.fixture_translation
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                finish_reason = "stop"
                latency = 0.0
            else:
                state["attempted_flash_calls"] += 1
                _write_state_checkpoint(state_path, state)
                flash_diagnostic = _send_diagnostic_request(
                    base_url=settings.ai.base_url,
                    api_key=settings.ai.api_key,
                    payload=payload,
                    timeout_seconds=args.flash_timeout,
                )
                state["flash_diagnostics"].append(
                    {
                        "case_id": case.id,
                        "group": group,
                        "repetition": repetition,
                        "http_status": flash_diagnostic["http_status"],
                        "message_keys": flash_diagnostic["message_keys"],
                        "content_len": flash_diagnostic["content_len"],
                        "reasoning_len": flash_diagnostic["reasoning_len"],
                        "finish_reason": flash_diagnostic["finish_reason"],
                        "usage": flash_diagnostic["usage"],
                        "latency_s": flash_diagnostic["latency_s"],
                    }
                )
                _write_state_checkpoint(state_path, state)
                raw_response = flash_diagnostic["content"]
                usage = flash_diagnostic["usage"]
                finish_reason = flash_diagnostic["finish_reason"]
                latency = flash_diagnostic["latency_s"]
                if not raw_response.strip():
                    raise TranslationError("Flash returned empty content.")
                state["usable_flash_responses"] += 1
                state["executed_flash_calls"] = state["attempted_flash_calls"]
                _write_state_checkpoint(state_path, state)
            evaluation = evaluate_output(
                raw_response,
                case=case,
                context=context.production,
                runtime_case=runtime_case,
            )
        except TranslationError as exc:
            raw_response = ""
            usage = {}
            finish_reason = None
            latency = round(time.perf_counter() - started, 3)
            error = str(exc)
            evaluation = {
                "translation": "",
                "normalization_changed": False,
                "validation": {"ok": False, "reason": "request_error"},
            }
        blind_id = hashlib.sha256(
            f"{args.seed}:{case.id}:{group}:{repetition}".encode("utf-8")
        ).hexdigest()[:16]
        record = {
            "type": "result",
            "blind_id": blind_id,
            "group": group,
            "case_id": case.id,
            "scope": case.scope,
            "category": case.category,
            "source": case.source,
            "review_focus": case.review_focus,
            "repetition": repetition,
            "raw_response": raw_response,
            "normalized_translation": evaluation["translation"],
            "normalization_changed": evaluation["normalization_changed"],
            "production_validation": evaluation["validation"],
            "finish_reason": finish_reason,
            "latency_s": latency,
            "usage": usage,
            "estimated_input_tokens": estimate_tokens(
                json.dumps(messages, ensure_ascii=False)
            ),
            "matched_reference_ids": [entry.id for entry in runtime_case.plan.matched_entries],
            "matched_memory_ids": list(frozen_request.matched_memory_ids),
            "source_technical_terms": list(runtime_case.technical_terms),
            "error": error,
        }
        records.append(record)
        print(
            f"{index:02d}/{len(jobs)} {group:<24} {case.id} rep={repetition} "
            f"ok={not error} validation={record['production_validation']['ok']} "
            f"finish={finish_reason!r} latency={latency:.3f}s"
        )
        if args.delay > 0 and not args.dry_run:
            time.sleep(args.delay)

    summary = _summarize(records, args.repetitions)
    metadata["attempted_api_calls"] = {
        "pro": state["attempted_pro_calls"],
        "flash": state["attempted_flash_calls"],
    }
    metadata["usable_api_responses"] = {
        "pro": state["usable_pro_responses"],
        "flash": state["usable_flash_responses"],
    }
    metadata["executed_api_calls"] = {
        "pro": state["executed_pro_calls"],
        "flash": state["executed_flash_calls"],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        handle.write(json.dumps({"type": "profiles", "profiles": profile_records}, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    write_blind(blind_path, records, seed=args.seed)
    state["status"] = "completed"
    _write_state_checkpoint(state_path, state)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Profile state: {state_path.resolve()}")
    print(f"Blind review: {blind_path.resolve()}")
    return output_path, state_path, blind_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DATASET_PATH).replace("\\", "/"))
    parser.add_argument("--output", default="")
    parser.add_argument("--state-output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--reuse-current-profile-state", default="")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--thinking-model", default="")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--pro-timeout", type=float, default=180.0)
    parser.add_argument("--flash-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions != 2:
        raise SystemExit("This frozen POC requires --repetitions 2")
    run(args)


if __name__ == "__main__":
    main()
