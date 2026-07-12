"""Zero-Pro Flash A/B POC for one completeness-only RuntimeProfile check."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.feedback.store import FeedbackStore
from app.agent.session_store import AgentProfileMeta, AgentSessionStore
from app.prompt.runtime_profile import RuntimeProfile
from app.reference_layer import ReferenceEntry, ReferencePackage
from app.settings import AppSettings
from app.translation.client import TranslationError
from app.translation.service import TranslationService
from poc.poc_agent_runtime_projection import (
    FrozenRequestContext,
    ProjectionContext,
    _send_diagnostic_request,
    build_projection_context,
    freeze_runtime_case,
)
from poc.poc_agent_semantic_profile import evaluate_output
from poc.poc_reference_injection import _percentile, estimate_tokens


GROUPS = ("CURRENT_RUNTIME", "COMPLETENESS_ONLY_RUNTIME")
DATASET_PATH = Path("poc/data/runtime_profile_completeness_dataset.json")
EXPECTED_DATASET_SHA256 = "f927f981ffa128e300ae7ef14e0b6b66ad69cd559cc6a881753b0233418dd57d"
LEGACY_DATASET_PATHS = tuple(Path("poc/data") / name for name in (
    "semantic_profile_dataset.json",
    "agent_strategy_transfer_dataset.json",
    "semantic_guard_dataset.json",
    "agent_runtime_projection_dataset.json",
    "runtime_profile_ab_dataset.json",
))
EXPECTED_CASE_SOURCES = (
    "审核结果尚未写入记录，因此现在不能关闭页面。",
    "如果校验没有完成，就不要发送确认通知。",
    "除非负责人今天批准，否则变更不会生效。",
    "报告已经生成，但签名检查仍然一直运行。",
    "旧凭证被管理员撤销后，新凭证才被发放。",
    "值班人员通知开发者暂停服务，开发者随后执行了操作。",
    "回家以前，请顺路买一盒牛奶。",
    "妹妹把钥匙放在玄关的小篮子里了。",
    "明天下午三点，我们在图书馆门口见面。",
    "院子里的花昨晚被风吹倒了两盆。",
    "请比较配置快照与当前配置文件，不要直接覆盖原文件。",
    "调度批次进入队列以后，后台任务只提交了一次。",
    "午夜窗口内连接发生超时，但客户端没有再次提交请求。",
    "备用链路已经连接成功，主连接却仍然处于超时状态。",
    "审计批次没有完成时，不得归档这份记录。",
    "回放窗口已经关闭，但核对动作还没结束。",
)
EXPECTED_CASE_SCOPES = (
    ("runtime_regression",) * 6
    + ("control",) * 4
    + ("lexical_trap",) * 4
    + ("reference_memory",) * 2
)
REPETITIONS = 2
RANDOM_SEED = 20260711
COMPLETENESS_CHECK = (
    "输出前仅核对源文中明确的否定、条件、例外、时间、数量、主体、动作及持续或完成状态"
    "是否完整保留；不得补充含义或改变词义。"
)
FORMAL_CALL_DESIGN = {"pro": 0, "flash": 64, "pro_judge": 0, "retry": 0}


class CompletenessPocError(ValueError):
    """Raised for invalid frozen inputs or experiment configuration."""


@dataclass(frozen=True)
class CompletenessCase:
    id: str
    scope: str
    category: str
    source: str
    review_focus: str
    fixture_translation: str


@dataclass(frozen=True)
class CompletenessDataset:
    version: int
    status: str
    name: str
    description: str
    cases: tuple[CompletenessCase, ...]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path) -> CompletenessDataset:
    if path.resolve() == DATASET_PATH.resolve():
        actual = _sha256_file(path)
        if actual != EXPECTED_DATASET_SHA256:
            raise CompletenessPocError(
                f"dataset SHA256 changed: expected={EXPECTED_DATASET_SHA256} actual={actual}"
            )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise CompletenessPocError("dataset must contain a cases list")
    dataset = CompletenessDataset(
        version=int(raw.get("version", 0)),
        status=str(raw.get("status", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        cases=tuple(
            CompletenessCase(
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
        sources.update(
            str(row.get("source", "")).strip()
            for row in raw.get("cases", [])
            if isinstance(row, dict) and str(row.get("source", "")).strip()
        )
    return sources


def validate_dataset(dataset: CompletenessDataset) -> None:
    errors: list[str] = []
    if dataset.version != 1 or dataset.status != "frozen_poc":
        errors.append("dataset must be frozen_poc version 1")
    if tuple(case.source for case in dataset.cases) != EXPECTED_CASE_SOURCES:
        errors.append("source list/order differs from frozen benchmark")
    if tuple(case.scope for case in dataset.cases) != EXPECTED_CASE_SCOPES:
        errors.append("scope list/order differs from frozen benchmark")
    if len(dataset.cases) != 16:
        errors.append("dataset must contain exactly 16 cases")
    if len({case.id for case in dataset.cases}) != 16:
        errors.append("case ids must be unique")
    if len({case.source for case in dataset.cases}) != 16:
        errors.append("case sources must be unique")
    overlap = sorted(set(EXPECTED_CASE_SOURCES) & _legacy_sources())
    if overlap:
        errors.append(f"sources overlap prior POCs: {overlap}")
    for case in dataset.cases:
        if not case.category or not case.review_focus or not case.fixture_translation:
            errors.append(f"{case.id} is incomplete")
    if errors:
        raise CompletenessPocError("; ".join(errors))


def build_confirmed_runtime_profile(
    context: ProjectionContext,
    dataset: CompletenessDataset,
) -> RuntimeProfile:
    profile = RuntimeProfile.from_dict(
        {
            "version": 1,
            "source_language": "中文",
            "target_language": "日本語",
            "semantic_directives": [],
            "style_directives": [],
            "completeness_checks": [COMPLETENESS_CHECK],
            "prompt_digest": context.production.prompt_hash,
            "policy_digest": context.production.policy_digest,
            "reference_digest": context.production.reference_digest,
            "confirmed": True,
            "source": "poc_local_completeness_only",
        },
        strict=True,
    )
    if not profile.is_usable or not profile.matches_language_pair("中文", "日本語"):
        raise CompletenessPocError("fixed completeness RuntimeProfile is not usable")
    if not profile.matches_context(
        prompt_digest=context.production.prompt_hash,
        policy_digest=context.production.policy_digest,
        reference_digest=context.production.reference_digest,
    ):
        raise CompletenessPocError("fixed RuntimeProfile does not match POC context")
    rendered = profile.render()
    for case in dataset.cases:
        if case.source in rendered:
            raise CompletenessPocError(f"evaluation source leaked into profile: {case.id}")
    return profile


def load_production_agent_profile(
    settings: AppSettings,
    *,
    session_store: AgentSessionStore | None = None,
) -> tuple[AgentProfileMeta, tuple[dict[str, str], ...], dict[str, Any]]:
    """Load the exact current three-message bootstrap without creating one."""

    service = TranslationService(settings)
    try:
        definition = service._resolve_agent_definition("中文", "日本語")
    finally:
        service.shutdown()
    meta = definition.profile_meta
    store = session_store or AgentSessionStore()
    profile_path = store._profile_path(meta)
    if not profile_path.exists():
        raise CompletenessPocError(
            f"matching production Agent Profile is missing: key={meta.key()}"
        )
    messages = store.load_profile(meta)
    if messages is None:
        raise CompletenessPocError(
            f"matching production Agent Profile is invalid: key={meta.key()}"
        )
    roles = [message.get("role") for message in messages]
    if len(messages) != 3 or roles != ["system", "user", "assistant"]:
        raise CompletenessPocError(
            "production Agent Profile must contain exactly system/user/assistant bootstrap"
        )
    bootstrap: list[dict[str, str]] = []
    message_audit: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise CompletenessPocError(
                f"production Agent Profile message {index} has empty content"
            )
        role = str(message["role"])
        bootstrap.append({"role": role, "content": content})
        message_audit.append(
            {
                "index": index,
                "role": role,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "length": len(content),
            }
        )
    return meta, tuple(bootstrap), {
        "status": "loaded",
        "key": meta.key(),
        "path": str(profile_path.resolve()),
        "source_language": meta.source_language,
        "target_language": meta.target_language,
        "prompt_hash": meta.prompt_hash,
        "fast_model": meta.fast_model,
        "thinking_model": meta.thinking_model,
        "messages": message_audit,
    }


def build_isolated_fixture_context(
    settings: AppSettings,
    fixture_root: Path,
) -> tuple[ProjectionContext, dict[str, Any]]:
    """Merge deterministic POC-only references and memory without touching settings."""

    context = build_projection_context(settings)
    fixture_references = ReferencePackage(
        entries=(
            ReferenceEntry("fixture_config_snapshot", "配置快照", "[せっていすなっぷしょっと]"),
            ReferenceEntry("fixture_schedule_batch", "调度批次", "[すけじゅうるばっち]"),
            ReferenceEntry("fixture_audit_batch", "审计批次", "[かんさばっち]"),
            ReferenceEntry("fixture_replay_window", "回放窗口", "[りぷれいまど]"),
        )
    )
    merged_references = ReferencePackage.merge(
        (context.production.references, fixture_references)
    )
    production = replace(
        context.production,
        references=merged_references,
        reference_digest=merged_references.digest(),
    )

    feedback_store = FeedbackStore(fixture_root / "feedback")
    fixture_memories = (
        ("午夜窗口", "核对午夜窗口中的时间范围是否完整保留。"),
        ("备用链路", "核对备用链路与主连接的主体和状态是否分别保留。"),
        ("回放窗口", "核对回放窗口与核对动作的完成状态是否分别保留。"),
    )
    memory_ids: list[str] = []
    for index, (trigger, rule) in enumerate(fixture_memories, 1):
        feedback = feedback_store.add_feedback(
            group_id=9000 + index,
            source_language="中文",
            target_language="日本語",
            ocr_text=trigger,
            translation_text="fixture",
            note="isolated completeness POC fixture",
        )
        memory = feedback_store.approve_feedback(
            feedback.id,
            trigger=trigger,
            rule=rule,
        )
        memory_ids.append(memory.id)

    return replace(
        context,
        production=production,
        feedback_store=feedback_store,
    ), {
        "kind": "isolated_poc_fixture",
        "settings_modified": False,
        "real_user_data_modified": False,
        "reference_ids": [entry.id for entry in fixture_references.entries],
        "memory_ids": memory_ids,
    }


def freeze_case_contexts(
    context: ProjectionContext,
    dataset: CompletenessDataset,
    *,
    rule_checklist: str,
) -> dict[str, FrozenRequestContext]:
    if not rule_checklist.strip():
        raise CompletenessPocError("production Agent Profile checklist is empty")
    return {
        case.id: freeze_runtime_case(
            context,
            case,
            current_format_profile=rule_checklist,
        )
        for case in dataset.cases
    }


def build_flash_messages(
    *,
    group: str,
    context: ProjectionContext,
    frozen: FrozenRequestContext,
    runtime_profile: RuntimeProfile,
    production_bootstrap: tuple[dict[str, str], ...],
) -> list[dict[str, str]]:
    del context
    if len(production_bootstrap) != 3:
        raise CompletenessPocError("production bootstrap must contain three messages")
    messages = [dict(message) for message in production_bootstrap]
    if group == "COMPLETENESS_ONLY_RUNTIME":
        rendered = runtime_profile.render()
        if not rendered:
            raise CompletenessPocError("RuntimeProfile rendered empty")
        messages[0] = {
            "role": "system",
            "content": f"{messages[0]['content']}\n\n{rendered}",
        }
    elif group != "CURRENT_RUNTIME":
        raise CompletenessPocError(f"unknown group: {group}")
    messages.append({"role": "user", "content": frozen.current_user_content})
    return messages


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("poc/results")
    return (
        root / f"runtime-profile-completeness-{stamp}.jsonl",
        root / f"runtime-profile-completeness-state-{stamp}.json",
        root / f"runtime-profile-completeness-blind-{stamp}.jsonl",
    )


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _provider_prompt_tokens(rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["usage"]["prompt_tokens"])
        for row in rows
        if isinstance(row.get("usage"), dict)
        and isinstance(row["usage"].get("prompt_tokens"), (int, float))
        and row["usage"]["prompt_tokens"] > 0
    ]
    return round(statistics.fmean(values), 1) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    means: dict[str, float] = {}
    for group in GROUPS:
        selected = [row for row in records if row["group"] == group]
        completed = [row for row in selected if not row["error"]]
        latencies = [float(row["latency_s"]) for row in completed]
        mean_tokens = statistics.fmean(
            float(row["estimated_input_tokens"]) for row in selected
        )
        means[group] = mean_tokens
        groups[group] = {
            "completed": len(completed),
            "expected": len(selected),
            "validation_pass": sum(
                bool(row["validation_result"]["ok"]) for row in completed
            ),
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "estimated_input_tokens_mean": round(mean_tokens, 1),
            "provider_prompt_tokens": _provider_prompt_tokens(completed),
            "injected_runtime_profile": group == "COMPLETENESS_ONLY_RUNTIME",
        }
    current = means["CURRENT_RUNTIME"]
    delta = ((means["COMPLETENESS_ONLY_RUNTIME"] - current) / current * 100) if current else 0.0
    likely_failed = delta > 5.0
    return {
        "type": "summary",
        "groups": groups,
        "estimated_input_token_delta_percent": round(delta, 2),
        "token_gate_likely_failed_before_formal_run": likely_failed,
        "token_gate_status": (
            "token gate likely failed before formal run"
            if likely_failed
            else "dry-run estimate is within the 5% token gate"
        ),
        "pro_calls": 0,
        "pro_payloads": 0,
        "automatic_retries": 0,
        "semantic_scores_are_manual_only": True,
    }


def write_blind(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
    rows = [
        {
            "blind_id": row["blind_id"],
            "source_text": row["source_text"],
            "scope": row["scope"],
            "category": row["category"],
            "review_focus": row["review_focus"],
            "translation": row["normalized_translation"],
            "major_completeness_error": None,
            "major_lexical_error": None,
            "semantic_fidelity_0_to_5": None,
            "review_notes": "",
        }
        for row in records
    ]
    random.Random(seed ^ 0xC011).shuffle(rows)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.repetitions != REPETITIONS:
        raise CompletenessPocError("this frozen POC requires exactly 2 repetitions")
    dataset = load_dataset(Path(args.dataset))
    settings = AppSettings.load()
    fixture_temp = tempfile.TemporaryDirectory(prefix="instant-translate-completeness-poc-")
    try:
        context, fixture_metadata = build_isolated_fixture_context(
            settings, Path(fixture_temp.name)
        )
        profile_meta, production_bootstrap, production_profile_audit = (
            load_production_agent_profile(settings)
        )
        if profile_meta.prompt_hash != context.production.prompt_hash:
            raise CompletenessPocError(
                "loaded production Agent Profile does not match current POC prompt hash"
            )
        runtime_profile = build_confirmed_runtime_profile(context, dataset)
        frozen = freeze_case_contexts(
            context,
            dataset,
            rule_checklist=production_bootstrap[2]["content"],
        )
        fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
        if not args.dry_run and not (settings.ai.base_url and settings.ai.api_key and fast_model):
            raise CompletenessPocError("API base URL, key, and Flash model are required")

        defaults = _default_paths()
        output_path = Path(args.output) if args.output else defaults[0]
        state_path = Path(args.state_output) if args.state_output else defaults[1]
        blind_path = Path(args.blind_output) if args.blind_output else defaults[2]
        for path in (output_path, state_path, blind_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        group_systems = {
            group: build_flash_messages(
                group=group,
                context=context,
                frozen=frozen[dataset.cases[0].id],
                runtime_profile=runtime_profile,
                production_bootstrap=production_bootstrap,
            )[0]["content"]
            for group in GROUPS
        }
        frozen_state = {
            case.id: {
                "user_content_sha256": hashlib.sha256(
                    frozen[case.id].current_user_content.encode("utf-8")
                ).hexdigest(),
                "matched_reference_ids": [
                    entry.id for entry in frozen[case.id].runtime_case.plan.matched_entries
                ],
                "matched_memory_ids": list(frozen[case.id].matched_memory_ids),
                "source_technical_terms": list(frozen[case.id].runtime_case.technical_terms),
            }
            for case in dataset.cases
        }
        state: dict[str, Any] = {
            "status": "running",
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "groups": list(GROUPS),
            "formal_call_design": dict(FORMAL_CALL_DESIGN),
            "dry_run": args.dry_run,
            "metadata": {
                "version": 1,
                "experiment": "runtime_profile_completeness_zero_pro_flash_ab",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "groups": list(GROUPS),
                "cases": 16,
                "repetitions": REPETITIONS,
                "seed": args.seed,
                "fast_model": fast_model,
                "formal_call_design": dict(FORMAL_CALL_DESIGN),
                "dry_run": args.dry_run,
                "fixture_source": fixture_metadata,
                "production_agent_profile": production_profile_audit,
            },
            "pro_calls": 0,
            "pro_payloads": [],
            "attempted_flash_calls": 0,
            "usable_flash_responses": 0,
            "runtime_profile": {
                "data": runtime_profile.to_dict(),
                "rendered": runtime_profile.render(),
                "digest": runtime_profile.digest(),
            },
            "production_agent_profile": production_profile_audit,
            "frozen_request_contexts": frozen_state,
            "system_prompts": {
                group: {
                    "sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
                    "length": len(system),
                    "injected_runtime_profile": group == "COMPLETENESS_ONLY_RUNTIME",
                }
                for group, system in group_systems.items()
            },
        }
        _write_state(state_path, state)

        metadata = {
            "type": "run",
            **state["metadata"],
            "pro_calls": 0,
            "pro_payloads": [],
        }
        jobs = [
            (case, group, repetition)
            for repetition in range(1, REPETITIONS + 1)
            for case in dataset.cases
            for group in GROUPS
        ]
        random.Random(args.seed).shuffle(jobs)
        records: list[dict[str, Any]] = []
        for index, (case, group, repetition) in enumerate(jobs, 1):
            frozen_case = frozen[case.id]
            messages = build_flash_messages(
                group=group,
                context=context,
                frozen=frozen_case,
                runtime_profile=runtime_profile,
                production_bootstrap=production_bootstrap,
            )
            payload = {
                "model": fast_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 4096,
                "thinking": {"type": "disabled"},
            }
            error = ""
            if args.dry_run:
                raw_translation = case.fixture_translation
                usage: dict[str, Any] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
                finish_reason: str | None = "stop"
                latency = 0.0
            else:
                state["attempted_flash_calls"] += 1
                _write_state(state_path, state)
                try:
                    diagnostic = _send_diagnostic_request(
                        base_url=settings.ai.base_url,
                        api_key=settings.ai.api_key,
                        payload=payload,
                        timeout_seconds=args.flash_timeout,
                    )
                    raw_translation = diagnostic["content"]
                    usage = diagnostic["usage"]
                    finish_reason = diagnostic["finish_reason"]
                    latency = diagnostic["latency_s"]
                    if not raw_translation.strip():
                        raise TranslationError("Flash returned empty content.")
                    state["usable_flash_responses"] += 1
                    _write_state(state_path, state)
                except TranslationError as exc:
                    raw_translation = ""
                    usage = {}
                    finish_reason = None
                    latency = 0.0
                    error = str(exc)

            evaluation = (
                {
                    "translation": "",
                    "normalization_changed": False,
                    "validation": {"ok": False, "reason": "request_error"},
                }
                if error
                else evaluate_output(
                    raw_translation,
                    case=case,
                    context=context.production,
                    runtime_case=frozen_case.runtime_case,
                )
            )
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
                "source_text": case.source,
                "review_focus": case.review_focus,
                "repetition": repetition,
                "raw_translation": raw_translation,
                "normalized_translation": evaluation["translation"],
                "validation_result": evaluation["validation"],
                "finish_reason": finish_reason,
                "latency_s": latency,
                "usage": usage,
                "estimated_input_tokens": estimate_tokens(
                    json.dumps(messages, ensure_ascii=False)
                ),
                "injected_runtime_profile": group == "COMPLETENESS_ONLY_RUNTIME",
                "matched_reference_ids": [
                    entry.id for entry in frozen_case.runtime_case.plan.matched_entries
                ],
                "matched_memory_ids": list(frozen_case.matched_memory_ids),
                "source_technical_terms": list(frozen_case.runtime_case.technical_terms),
                "error": error,
            }
            records.append(record)
            print(
                f"{index:02d}/{len(jobs)} {group:<27} {case.id} rep={repetition} "
                f"ok={not error} validation={record['validation_result']['ok']} "
                f"finish={finish_reason!r} latency={latency:.3f}s"
            )
            if args.delay > 0 and not args.dry_run:
                time.sleep(args.delay)

        summary = summarize(records)
        metadata["attempted_flash_calls"] = state["attempted_flash_calls"]
        metadata["usable_flash_responses"] = state["usable_flash_responses"]
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        write_blind(blind_path, records, seed=args.seed)
        state["status"] = "completed"
        state["summary"] = summary
        _write_state(state_path, state)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Results: {output_path.resolve()}")
        print(f"State: {state_path.resolve()}")
        print(f"Blind review: {blind_path.resolve()}")
        return output_path, state_path, blind_path
    finally:
        fixture_temp.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DATASET_PATH).replace("\\", "/"))
    parser.add_argument("--output", default="")
    parser.add_argument("--state-output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--flash-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
