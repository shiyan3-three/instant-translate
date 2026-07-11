"""Blind A/B POC for Pro-compiled semantic Agent Profiles.

The experiment compares two Pro outputs built from the same production prompt:

* FORMAT_PROFILE: the current format-focused digest instruction.
* SEMANTIC_PROFILE: a validated structured profile covering semantic fidelity,
  target-language naturalness, terminology strategy, ambiguity, completeness,
  and the existing format checklist.

All Flash requests use the same production system prompt, policy, references,
model parameters, normalized profile delivery, OCR source, and local output
pipeline.  Only the compiled profile content differs.  Evaluation cases never
enter either Pro request.  There is no Pro judge and no automatic retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.agent.agent import TranslationAgent
from app.prompt.policy import ConstraintPolicy
from app.reference_layer import ReferencePackage, ReferencePlan, ReferenceStore
from app.settings import AppSettings
from app.translation.client import TranslationError
from app.translation.quality import OutputNormalizer, OutputValidator
from app.translation.service import TranslationService
from app.translation.terms import TermPlaceholder
from poc.poc_agent_memory_transfer import extract_json_object
from poc.poc_reference_injection import _percentile, estimate_tokens


GROUPS = ("FORMAT_PROFILE", "SEMANTIC_PROFILE")
PROFILE_FIELDS = (
    "semantic_principles",
    "target_naturalness",
    "terminology_strategy",
    "ambiguity_strategy",
    "completeness_checks",
    "format_checklist",
)
EXPECTED_CASE_SOURCES = (
    "服务器刚刚重启，后台任务还没有完全恢复。",
    "这个页面显示成功了，但数据库里没有写入记录。",
    "请把 API 网关的超时时间从 30 秒改成 5 秒。",
    "缓存命中率突然下降，可能是配置文件没有加载。",
    "用户已经取消订单，界面却还显示正在支付。",
    "如果脚本执行失败，不要重复提交同一个请求。",
    "这次更新只改了前端样式，没有影响核心逻辑。",
    "日志里显示任务完成了，实际上队列还在阻塞。",
    "明天下午三点前，把会议纪要整理完。",
    "雨停以后，我们再去附近的商店。",
    "她把钥匙放在桌上，然后关掉了房间里的灯。",
    "这趟列车比预定时间晚了二十分钟。",
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
_SEMANTIC_PROFILE_USER = """Compile the confirmed translation prompt above into one compact Agent Profile for a fast translation model.

Return only one JSON object with exactly this schema:
{
  "version": 1,
  "role": "translation_semantic_profile",
  "semantic_principles": ["..."],
  "target_naturalness": ["..."],
  "terminology_strategy": ["..."],
  "ambiguity_strategy": ["..."],
  "completeness_checks": ["..."],
  "format_checklist": ["..."]
}

Requirements:
- Each list must contain 1-8 concise, actionable rules. Each rule is at most 300 characters.
- Preserve meaning, predicate, negation, contrast, cause, uncertainty, time, quantity, and sentence completeness before applying format constraints.
- Describe how to choose natural target-language expressions from context, especially in software/UI text, without mechanically reading Chinese characters or shortening concepts.
- User-confirmed Knowledge References are authoritative. Unknown ordinary words must be translated naturally.
- Do not create glossary tables, source-to-target mappings, fixed readings, terminology lists, examples, test sentences, translations, or chain-of-thought.
- Do not weaken the original user constraints. Format requirements belong in format_checklist.
- Return JSON only, without Markdown fences or commentary."""


class SemanticProfilePocError(ValueError):
    """Raised for invalid POC data or model-produced profile state."""


@dataclass(frozen=True)
class ProfileCase:
    id: str
    scope: str
    category: str
    source: str
    review_focus: str
    fixture_translation: str


@dataclass(frozen=True)
class ProfileDataset:
    version: int
    status: str
    name: str
    description: str
    cases: tuple[ProfileCase, ...]


@dataclass(frozen=True)
class ProductionContext:
    system: str
    policy: ConstraintPolicy
    references: ReferencePackage
    prompt_hash: str
    prompt_length: int
    policy_digest: str
    reference_digest: str
    enhanced_system_hash: str
    term_wrapper_mode: str


@dataclass(frozen=True)
class RuntimeCaseContext:
    plan: ReferencePlan
    protected_text: str
    protected_terms: tuple[str, ...]
    technical_terms: tuple[str, ...]
    expected_wrapped_targets: tuple[str, ...] | None
    extra_wrapped_term_budget: int | None
    user_content: str


def load_dataset(path: Path) -> ProfileDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SemanticProfilePocError("dataset must be a JSON object")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise SemanticProfilePocError("dataset.cases must be a list")
    cases: list[ProfileCase] = []
    for item in cases_raw:
        if not isinstance(item, dict):
            raise SemanticProfilePocError("each case must be an object")
        cases.append(
            ProfileCase(
                id=str(item.get("id", "")).strip(),
                scope=str(item.get("scope", "")).strip(),
                category=str(item.get("category", "")).strip(),
                source=str(item.get("source", "")).strip(),
                review_focus=str(item.get("review_focus", "")).strip(),
                fixture_translation=str(item.get("fixture_translation", "")).strip(),
            )
        )
    dataset = ProfileDataset(
        version=int(raw.get("version", 0)),
        status=str(raw.get("status", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        cases=tuple(cases),
    )
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: ProfileDataset) -> None:
    errors: list[str] = []
    if dataset.version != 1 or dataset.status != "frozen_poc":
        errors.append("dataset must be frozen_poc version 1")
    if tuple(case.source for case in dataset.cases) != EXPECTED_CASE_SOURCES:
        errors.append("case source list/order differs from the frozen benchmark")
    if tuple(case.scope for case in dataset.cases) != EXPECTED_CASE_SCOPES:
        errors.append("case scope list/order differs from the frozen benchmark")
    ids = [case.id for case in dataset.cases]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        errors.append("case ids must be non-empty and unique")
    for case in dataset.cases:
        if not case.category or not case.review_focus or not case.fixture_translation:
            errors.append(f"{case.id} is missing category, review focus, or fixture")
    if errors:
        raise SemanticProfilePocError("; ".join(errors))


def build_production_context(settings: AppSettings) -> ProductionContext:
    service = TranslationService(settings)
    try:
        definition = service._resolve_agent_definition("中文", "日本語")
        references = service._current_reference_package()
    finally:
        service.shutdown()
    term_wrapper_mode = TranslationAgent._term_wrapper_selection_mode(definition.policy)
    enhanced_system = TranslationAgent._append_runtime_tool_rules(
        TranslationAgent._rewrite_constraints(definition.prompt),
        preserve_placeholders=not definition.policy.has_script("hiragana"),
        term_wrapper_mode=term_wrapper_mode,
    )
    return ProductionContext(
        system=enhanced_system,
        policy=definition.policy,
        references=references,
        prompt_hash=definition.runtime_meta.prompt_hash,
        prompt_length=len(definition.prompt),
        policy_digest=definition.policy.digest(),
        reference_digest=references.digest(),
        enhanced_system_hash=hashlib.sha256(enhanced_system.encode("utf-8")).hexdigest(),
        term_wrapper_mode=term_wrapper_mode,
    )


def build_pro_payloads(context: ProductionContext, thinking_model: str) -> dict[str, dict[str, Any]]:
    common_system = context.system + _DIGEST_META
    return {
        "FORMAT_PROFILE": {
            "model": thinking_model,
            "messages": [
                {"role": "system", "content": common_system},
                {"role": "user", "content": _FORMAT_PROFILE_USER},
            ],
            "temperature": 0.0,
            "max_tokens": 8192,
            "thinking": {"type": "enabled"},
        },
        "SEMANTIC_PROFILE": {
            "model": thinking_model,
            "messages": [
                {"role": "system", "content": common_system},
                {"role": "user", "content": _SEMANTIC_PROFILE_USER},
            ],
            "temperature": 0.0,
            "max_tokens": 8192,
            "thinking": {"type": "enabled"},
        },
    }


def make_dry_format_profile() -> str:
    return (
        "- 输出遵守用户要求的字符范围\n"
        "- 用户确认的技术术语使用指定包裹符\n"
        "- 按用户要求处理空格与标点\n"
        "- 最终只输出译文"
    )


def make_dry_semantic_profile() -> dict[str, Any]:
    return {
        "version": 1,
        "role": "translation_semantic_profile",
        "semantic_principles": [
            "先保留完整命题、谓语、否定、转折、因果、推测、时间和数量，再应用格式约束。",
            "不得逐字替换源语言结构，也不得增加源文没有的关系。",
        ],
        "target_naturalness": [
            "按上下文选择目标语言母语者自然使用的完整表达。",
            "软件与界面文本应使用对应语境中的自然概念，不按中文字面读音机械转换。",
        ],
        "terminology_strategy": [
            "用户确认的知识引用优先；未知普通词自然翻译，不制造固定读法。",
        ],
        "ambiguity_strategy": [
            "结合整句谓语和领域上下文消解多义词，不孤立翻译单个词。",
        ],
        "completeness_checks": [
            "最终译文必须表达源文的完整谓语，不得以未完成词干或悬空关系结束。",
        ],
        "format_checklist": [
            "遵守用户确认的字符、术语包裹、空格和标点要求。",
        ],
    }


def validate_semantic_profile(raw: Any, *, dataset: ProfileDataset) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SemanticProfilePocError("semantic profile must be an object")
    expected_keys = {"version", "role", *PROFILE_FIELDS}
    if set(raw) != expected_keys:
        raise SemanticProfilePocError("semantic profile keys do not match the frozen schema")
    if raw.get("version") != 1 or raw.get("role") != "translation_semantic_profile":
        raise SemanticProfilePocError("semantic profile version/role is invalid")
    clean: dict[str, Any] = {"version": 1, "role": "translation_semantic_profile"}
    combined: list[str] = []
    for field in PROFILE_FIELDS:
        values = raw.get(field)
        if not isinstance(values, list) or not 1 <= len(values) <= 8:
            raise SemanticProfilePocError(f"{field} must contain 1-8 rules")
        rules: list[str] = []
        for value in values:
            text = str(value).strip()
            if not text or len(text) > 300:
                raise SemanticProfilePocError(f"{field} contains an empty or oversized rule")
            if any(marker in text for marker in ("->", "=>", "→")):
                raise SemanticProfilePocError(f"{field} contains a forbidden term mapping")
            rules.append(text)
            combined.append(text)
        clean[field] = rules
    serialized = json.dumps(clean, ensure_ascii=False)
    if len(serialized) > 6000:
        raise SemanticProfilePocError("semantic profile is too large")
    for case in dataset.cases:
        if case.source in serialized:
            raise SemanticProfilePocError(f"evaluation case leaked into semantic profile: {case.id}")
    return clean


def validate_format_profile(text: str, *, dataset: ProfileDataset) -> str:
    value = str(text or "").strip()
    if not value or len(value) > 6000:
        raise SemanticProfilePocError("format profile is empty or too large")
    for case in dataset.cases:
        if case.source in value:
            raise SemanticProfilePocError(f"evaluation case leaked into format profile: {case.id}")
    return value


def render_profile(group: str, profile: str | dict[str, Any]) -> str:
    if group == "FORMAT_PROFILE":
        return str(profile).strip()
    if group != "SEMANTIC_PROFILE" or not isinstance(profile, dict):
        raise SemanticProfilePocError(f"invalid profile for group {group}")
    lines: list[str] = []
    for field in PROFILE_FIELDS:
        lines.append(f"[{field}]")
        lines.extend(f"- {rule}" for rule in profile[field])
    return "\n".join(lines)


def _term_wrapper_domains(policy: ConstraintPolicy) -> list[str]:
    return [
        str(rule.params.get("domain", "")).strip()
        for rule in policy.rules_of_type("term_wrapper", local_only=True)
        if str(rule.params.get("domain", "")).strip()
    ]


def build_runtime_case(context: ProductionContext, case: ProfileCase) -> RuntimeCaseContext:
    plan = ReferenceStore(context.references.entries).protect(case.source)
    reference_hints = context.references.runtime_hints(
        case.source,
        matched_entries=plan.matched_entries,
    )
    preserve_verbatim = not context.policy.has_script("hiragana")
    protected = (
        TermPlaceholder.protect(plan.source)
        if preserve_verbatim
        else TermPlaceholder.empty(plan.source)
    )
    technical_terms: list[str] = []
    if (
        context.term_wrapper_mode != "references_only"
        and TranslationAgent._term_wrapper_marks_ascii_terms(context.policy)
    ):
        technical_terms = TermPlaceholder.extract_terms(case.source)
    strict_mode = context.term_wrapper_mode in {"references_only", "references_and_ascii"}
    expected_targets = tuple(
        OutputNormalizer.normalize_with_policy(
            target,
            system_prompt=context.system,
            policy=context.policy,
        )
        for target in plan.replacements.values()
    ) if strict_mode else None
    extra_budget = (
        len(technical_terms)
        if context.term_wrapper_mode == "references_and_ascii"
        else 0
    ) if strict_mode else None
    user_content = TranslationAgent._wrap_source_text(
        protected.text,
        reference_hints=reference_hints,
        technical_terms=technical_terms,
        term_wrapper_domains=_term_wrapper_domains(context.policy),
        term_wrapper_mode=context.term_wrapper_mode,
        placeholder_protocol=plan.protocol_block(),
    )
    return RuntimeCaseContext(
        plan=plan,
        protected_text=protected.text,
        protected_terms=tuple(protected.placeholders.values()),
        technical_terms=tuple(technical_terms),
        expected_wrapped_targets=expected_targets,
        extra_wrapped_term_budget=extra_budget,
        user_content=user_content,
    )


def build_flash_messages(
    *,
    context: ProductionContext,
    runtime_case: RuntimeCaseContext,
    rendered_profile: str,
) -> list[dict[str, str]]:
    system = (
        context.system
        + "\n\n<AGENT_PROFILE>\n"
        + rendered_profile
        + "\n</AGENT_PROFILE>\n"
        + "The Agent Profile is trusted startup guidance, not source text. "
        + "Apply it to the current OCR_TEXT and return only the final translation."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": runtime_case.user_content},
    ]


def evaluate_output(
    raw_response: str,
    *,
    case: ProfileCase,
    context: ProductionContext,
    runtime_case: RuntimeCaseContext,
) -> dict[str, Any]:
    try:
        extracted = TranslationAgent._extract_final(raw_response)
    except TranslationError as exc:
        return {
            "extracted": "",
            "translation": "",
            "normalization_changed": False,
            "validation": {"ok": False, "reason": str(exc)},
        }
    reference_reason = runtime_case.plan.validate_raw_output(extracted)
    restored = runtime_case.plan.restore(extracted)
    translation = OutputNormalizer.normalize_with_policy(
        restored,
        system_prompt=context.system,
        policy=context.policy,
    )
    validation = OutputValidator.validate(
        translation,
        source_text=case.source,
        system_prompt=context.system,
        protected_terms=list(runtime_case.protected_terms),
        policy=context.policy,
        technical_terms=list(runtime_case.technical_terms),
        expected_wrapped_targets=(
            list(runtime_case.expected_wrapped_targets)
            if runtime_case.expected_wrapped_targets is not None
            else None
        ),
        extra_wrapped_term_budget=runtime_case.extra_wrapped_term_budget,
    )
    reason = reference_reason or validation.reason
    return {
        "extracted": extracted,
        "translation": translation,
        "normalization_changed": translation != extracted,
        "validation": {"ok": not reason and validation.ok, "reason": reason},
    }


def _send_request(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str, dict[str, Any], str | None]:
    try:
        response = httpx.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        if not content.strip():
            raise TranslationError("API returned an empty translation")
        return content.strip(), dict(body.get("usage", {})), choice.get("finish_reason")
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise TranslationError(str(exc)) from exc


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("poc/results")
    return (
        root / f"semantic-profile-{stamp}.jsonl",
        root / f"semantic-profile-state-{stamp}.json",
        root / f"semantic-profile-blind-{stamp}.jsonl",
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
            "integrate_semantic_profile_only_if": (
                "Blind semantic fidelity and naturalness each improve by >=0.4, "
                "runtime-regression major errors fall by >=30%, incomplete sentences do not increase, "
                "format pass drops by at most 1 result, and Flash P95 latency rises by <=30%."
            ),
            "otherwise": "Do not replace the production Agent Profile.",
        },
    }


def write_blind(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
    rows = [
        {
            "blind_id": row["blind_id"],
            "case_id": row["case_id"],
            "scope": row["scope"],
            "category": row["category"],
            "source": row["source"],
            "review_focus": row["review_focus"],
            "translation": row["normalized_translation"],
            "semantic_fidelity_0_to_5": None,
            "naturalness_0_to_5": None,
            "terminology_0_to_5": None,
            "complete_sentence": None,
            "major_error": None,
            "review_notes": "",
        }
        for row in records
        if row.get("normalized_translation")
    ]
    random.Random(seed ^ 0xA63F).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset = load_dataset(Path(args.dataset))
    settings = AppSettings.load()
    context = build_production_context(settings)
    fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
    thinking_model = (
        args.thinking_model.strip()
        or settings.ai.thinking_model_name
        or "deepseek-v4-pro"
    )
    if not args.dry_run and (
        not settings.ai.base_url or not settings.ai.api_key or not fast_model or not thinking_model
    ):
        raise SemanticProfilePocError("API base URL, key, fast model, and thinking model are required")

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
            raise SemanticProfilePocError(f"evaluation source leaked into Pro payload: {case.id}")

    profiles: dict[str, str | dict[str, Any]] = {}
    profile_records: dict[str, Any] = {}
    for group in GROUPS:
        started = time.perf_counter()
        if args.dry_run:
            raw_response = (
                make_dry_format_profile()
                if group == "FORMAT_PROFILE"
                else json.dumps(make_dry_semantic_profile(), ensure_ascii=False)
            )
            usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            finish_reason = "stop"
            latency = 0.0
        else:
            raw_response, usage, finish_reason = _send_request(
                base_url=settings.ai.base_url,
                api_key=settings.ai.api_key,
                payload=pro_payloads[group],
                timeout_seconds=args.pro_timeout,
            )
            latency = round(time.perf_counter() - started, 3)
        if group == "FORMAT_PROFILE":
            parsed_profile: str | dict[str, Any] = validate_format_profile(
                raw_response,
                dataset=dataset,
            )
        else:
            parsed_profile = validate_semantic_profile(
                extract_json_object(raw_response),
                dataset=dataset,
            )
        profiles[group] = parsed_profile
        rendered = render_profile(group, parsed_profile)
        profile_records[group] = {
            "payload": pro_payloads[group],
            "raw_response": raw_response,
            "parsed_profile": parsed_profile,
            "rendered_profile": rendered,
            "rendered_profile_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "usage": usage,
            "finish_reason": finish_reason,
            "latency_s": latency,
        }

    state = {
        "metadata": {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_hash": context.prompt_hash,
            "prompt_length": context.prompt_length,
            "policy_digest": context.policy_digest,
            "reference_digest": context.reference_digest,
            "enhanced_system_hash": context.enhanced_system_hash,
            "term_wrapper_mode": context.term_wrapper_mode,
            "fast_model": fast_model,
            "thinking_model": thinking_model,
            "profile_delivery": "single_system_agent_profile_block_v1",
            "pro_sees_evaluation_sources": False,
            "dry_run": args.dry_run,
        },
        "profiles": profile_records,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "type": "run",
        "experiment": "pro_semantic_agent_profile_ab",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "groups": list(GROUPS),
        "cases": len(dataset.cases),
        "repetitions": args.repetitions,
        "runtime_regressions": 8,
        "controls": 4,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "prompt_hash": context.prompt_hash,
        "policy_digest": context.policy_digest,
        "reference_digest": context.reference_digest,
        "term_wrapper_mode": context.term_wrapper_mode,
        "pro_sees_evaluation_sources": False,
        "pro_judge_calls": 0,
        "automatic_retries": 0,
        "api_calls_expected": {
            "pro": 0 if args.dry_run else 2,
            "flash": len(dataset.cases) * len(GROUPS) * args.repetitions,
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
    total = len(jobs)
    for index, (case, group, repetition) in enumerate(jobs, 1):
        runtime_case = build_runtime_case(context, case)
        rendered_profile = profile_records[group]["rendered_profile"]
        messages = build_flash_messages(
            context=context,
            runtime_case=runtime_case,
            rendered_profile=rendered_profile,
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
                raw_response, usage, finish_reason = _send_request(
                    base_url=settings.ai.base_url,
                    api_key=settings.ai.api_key,
                    payload=payload,
                    timeout_seconds=args.flash_timeout,
                )
                latency = round(time.perf_counter() - started, 3)
            evaluation = evaluate_output(
                raw_response,
                case=case,
                context=context,
                runtime_case=runtime_case,
            )
        except TranslationError as exc:
            raw_response = ""
            usage = {}
            finish_reason = None
            latency = round(time.perf_counter() - started, 3)
            error = str(exc)
            evaluation = {
                "extracted": "",
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
            "source_technical_terms": list(runtime_case.technical_terms),
            "error": error,
        }
        records.append(record)
        print(
            f"{index:02d}/{total} {group:<16} {case.id} rep={repetition} "
            f"ok={not error} validation={record['production_validation']['ok']} "
            f"finish={finish_reason!r} latency={latency:.3f}s"
        )
        if args.delay > 0 and not args.dry_run:
            time.sleep(args.delay)

    summary = _summarize(records, args.repetitions)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        handle.write(json.dumps({"type": "profiles", "profiles": profile_records}, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    write_blind(blind_path, records, seed=args.seed)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Profile state: {state_path.resolve()}")
    print(f"Blind review: {blind_path.resolve()}")
    return output_path, state_path, blind_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="poc/data/semantic_profile_dataset.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--state-output", default="")
    parser.add_argument("--blind-output", default="")
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
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
