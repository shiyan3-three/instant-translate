"""Zero-Pro Flash A/B POC for a confirmed local RuntimeProfile.

CURRENT_RUNTIME reproduces the current production Flash bootstrap and frozen
request wrapper. RUNTIME_PROFILE_RUNTIME uses the identical messages except
that the system message additionally contains RuntimeProfile.render(). No Pro
payload or Pro request exists in this experiment.
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

from app.prompt.runtime_profile import RuntimeProfile
from app.settings import AppSettings
from app.translation.client import TranslationError
from poc.poc_agent_runtime_projection import (
    FrozenRequestContext,
    ProjectionContext,
    _send_diagnostic_request,
    build_projection_context,
    freeze_runtime_case,
)
from poc.poc_agent_semantic_profile import evaluate_output
from poc.poc_reference_injection import _percentile, estimate_tokens


GROUPS = ("CURRENT_RUNTIME", "RUNTIME_PROFILE_RUNTIME")
DATASET_PATH = Path("poc/data/runtime_profile_ab_dataset.json")
EXPECTED_DATASET_SHA256 = "808e75619c1518d68539862e1f8e2002a321e04bd87947ba88c154aee3cd7863"
LEGACY_DATASET_PATHS = (
    Path("poc/data/semantic_profile_dataset.json"),
    Path("poc/data/agent_strategy_transfer_dataset.json"),
    Path("poc/data/semantic_guard_dataset.json"),
    Path("poc/data/agent_runtime_projection_dataset.json"),
)
EXPECTED_CASE_SOURCES = (
    "监控面板已经恢复正常，但告警记录还没有清除。",
    "更新包还没下载完，安装任务仍然在等待。",
    "用户关闭了弹窗，界面上仍然显示加载状态。",
    "这个服务从早上开始一直重复提交同一个任务。",
    "如果系统没有收到确认消息，就保持当前页面不变。",
    "只要队列里还有任务，处理程序就继续运行。",
    "除非连接在十秒内恢复，否则这次请求会超时。",
    "配置文件被后台任务覆盖后，系统重新启动了服务。",
    "她已经把借来的雨伞还给邻居了。",
    "只要天气晴朗，我们就去河边散步。",
    "晚饭做好以后，请先叫孩子们洗手。",
    "那只猫一直趴在窗边看外面的鸟。",
)
EXPECTED_CASE_SCOPES = ("runtime_regression",) * 8 + ("control",) * 4
REPETITIONS = 2
RANDOM_SEED = 20260710
CURRENT_RULE_CHECKLIST = (
    "- 输出只能由用户确认的字符范围构成\n"
    "- 只处理用户确认或本地识别出的技术项\n"
    "- 遵守用户确认的空白与标点要求\n"
    "- 不添加解释、注释或扩展内容，只输出翻译文本"
)
_FORMAT_PROFILE_USER = (
    "请逐条列出上述翻译规则中最关键的格式要求"
    "（如字符限制、术语格式、空格、标点等）。"
    "每条一行，只列规则要点，不要翻译这句话。"
)


class RuntimeProfileAbPocError(ValueError):
    """Raised for invalid frozen inputs or experiment configuration."""


@dataclass(frozen=True)
class RuntimeProfileCase:
    id: str
    scope: str
    category: str
    source: str
    review_focus: str
    fixture_translation: str


@dataclass(frozen=True)
class RuntimeProfileDataset:
    version: int
    status: str
    name: str
    description: str
    cases: tuple[RuntimeProfileCase, ...]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path) -> RuntimeProfileDataset:
    if path.resolve() == DATASET_PATH.resolve():
        actual = _sha256_file(path)
        if actual != EXPECTED_DATASET_SHA256:
            raise RuntimeProfileAbPocError(
                f"dataset SHA256 changed: expected={EXPECTED_DATASET_SHA256} actual={actual}"
            )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise RuntimeProfileAbPocError("dataset must be an object containing cases")
    dataset = RuntimeProfileDataset(
        version=int(raw.get("version", 0)),
        status=str(raw.get("status", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        cases=tuple(
            RuntimeProfileCase(
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


def validate_dataset(dataset: RuntimeProfileDataset) -> None:
    errors: list[str] = []
    if dataset.version != 1 or dataset.status != "frozen_poc":
        errors.append("dataset must be frozen_poc version 1")
    if tuple(case.source for case in dataset.cases) != EXPECTED_CASE_SOURCES:
        errors.append("source list/order differs from the frozen benchmark")
    if tuple(case.scope for case in dataset.cases) != EXPECTED_CASE_SCOPES:
        errors.append("scope list/order differs from the frozen benchmark")
    if len(dataset.cases) != 12:
        errors.append("dataset must contain exactly 12 cases")
    if len({case.id for case in dataset.cases}) != 12:
        errors.append("case ids must be unique")
    if len({case.source for case in dataset.cases}) != 12:
        errors.append("case sources must be unique")
    overlap = sorted(set(EXPECTED_CASE_SOURCES) & _legacy_sources())
    if overlap:
        errors.append(f"sources overlap prior POCs: {overlap}")
    for case in dataset.cases:
        if not case.category or not case.review_focus or not case.fixture_translation:
            errors.append(f"{case.id} is incomplete")
    if errors:
        raise RuntimeProfileAbPocError("; ".join(errors))


def build_confirmed_runtime_profile(
    context: ProjectionContext,
    dataset: RuntimeProfileDataset,
) -> RuntimeProfile:
    profile = RuntimeProfile.from_dict(
        {
            "version": 1,
            "source_language": "中文",
            "target_language": "日本語",
            "semantic_directives": [],
            "style_directives": [],
            "completeness_checks": [
                "检查整句的主语、谓语、否定、条件、时间和状态变化是否完整保留。",
                "检查已经完成、尚未完成、仍在持续和长期反复等状态是否被正确区分。",
                "检查技术场景中的普通词是否依据整句含义处理，且没有被误当作专名。",
                "检查条件、例外、被动关系、数量、期限和转折是否完整保留。",
                "检查已经、尚未、仍在和持续状态是否落在正确的动作上。",
            ],
            "prompt_digest": context.production.prompt_hash,
            "policy_digest": context.production.policy_digest,
            "reference_digest": context.production.reference_digest,
            "confirmed": True,
            "source": "poc_local_confirmed_profile",
        },
        strict=True,
    )
    if not profile.is_usable or not profile.matches_language_pair("中文", "日本語"):
        raise RuntimeProfileAbPocError("fixed RuntimeProfile is not usable for 中文→日本語")
    if not profile.matches_context(
        prompt_digest=context.production.prompt_hash,
        policy_digest=context.production.policy_digest,
        reference_digest=context.production.reference_digest,
    ):
        raise RuntimeProfileAbPocError("fixed RuntimeProfile does not match production context")
    rendered = profile.render()
    for case in dataset.cases:
        if case.source in rendered:
            raise RuntimeProfileAbPocError(f"evaluation source leaked into RuntimeProfile: {case.id}")
    return profile


def freeze_case_contexts(
    context: ProjectionContext,
    dataset: RuntimeProfileDataset,
) -> dict[str, FrozenRequestContext]:
    return {
        case.id: freeze_runtime_case(
            context,
            case,
            current_format_profile=CURRENT_RULE_CHECKLIST,
        )
        for case in dataset.cases
    }


def build_flash_messages(
    *,
    group: str,
    context: ProjectionContext,
    frozen: FrozenRequestContext,
    runtime_profile: RuntimeProfile,
) -> list[dict[str, str]]:
    system = context.production.system
    if group == "RUNTIME_PROFILE_RUNTIME":
        rendered = runtime_profile.render()
        if not rendered:
            raise RuntimeProfileAbPocError("RuntimeProfile rendered empty")
        system = (
            f"{system}\n\n{rendered}\n"
            "The confirmed RUNTIME_PROFILE is semantic runtime guidance. "
            "Apply it to OCR_TEXT without weakening the existing system rules."
        )
    elif group != "CURRENT_RUNTIME":
        raise RuntimeProfileAbPocError(f"unknown group: {group}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _FORMAT_PROFILE_USER},
        {"role": "assistant", "content": CURRENT_RULE_CHECKLIST},
        {"role": "user", "content": frozen.current_user_content},
    ]


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("poc/results")
    return (
        root / f"runtime-profile-ab-{stamp}.jsonl",
        root / f"runtime-profile-ab-state-{stamp}.json",
        root / f"runtime-profile-ab-blind-{stamp}.jsonl",
    )


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [row for row in records if row["group"] == group]
        completed = [row for row in selected if not row["error"]]
        latencies = [float(row["latency_s"]) for row in completed]
        groups[group] = {
            "completed": len(completed),
            "expected": len(selected),
            "validation_pass": sum(
                bool(row["validation_result"]["ok"]) for row in completed
            ),
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "estimated_input_tokens_mean": round(
                statistics.fmean(float(row["estimated_input_tokens"]) for row in selected),
                1,
            ) if selected else None,
            "injected_runtime_profile": group == "RUNTIME_PROFILE_RUNTIME",
        }
    return {
        "type": "summary",
        "groups": groups,
        "pro_calls": 0,
        "pro_payloads": 0,
        "automatic_retries": 0,
        "semantic_scores_are_manual_only": True,
        "interpretation": "Do not infer translation quality from dry-run fixtures; use blind review after a separately authorized Flash-only run.",
    }


def write_blind(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
    rows = [
        {
            "blind_id": row["blind_id"],
            "source_text": row["source_text"],
            "review_focus": row["review_focus"],
            "translation": row["normalized_translation"],
            "semantic_fidelity_0_to_5": None,
            "naturalness_0_to_5": None,
            "meaning_error": "",
            "review_notes": "",
        }
        for row in records
    ]
    random.Random(seed ^ 0xC305).shuffle(rows)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.repetitions != REPETITIONS:
        raise RuntimeProfileAbPocError("this frozen POC requires exactly 2 repetitions")
    dataset = load_dataset(Path(args.dataset))
    settings = AppSettings.load()
    context = build_projection_context(settings)
    runtime_profile = build_confirmed_runtime_profile(context, dataset)
    frozen = freeze_case_contexts(context, dataset)
    fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
    if not args.dry_run and not (settings.ai.base_url and settings.ai.api_key and fast_model):
        raise RuntimeProfileAbPocError("API base URL, key, and Flash model are required")

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
        )[0]["content"]
        for group in GROUPS
    }
    state: dict[str, Any] = {
        "status": "running",
        "metadata": {
            "version": 1,
            "experiment": "runtime_profile_zero_pro_flash_ab",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "groups": list(GROUPS),
            "cases": 12,
            "repetitions": REPETITIONS,
            "seed": args.seed,
            "fast_model": fast_model,
            "prompt_hash": context.production.prompt_hash,
            "policy_digest": context.production.policy_digest,
            "reference_digest": context.production.reference_digest,
            "formal_call_design": {
                "pro": 0,
                "flash": 48,
                "pro_judge": 0,
                "retry": 0,
            },
            "dry_run": args.dry_run,
        },
        "pro_payloads": [],
        "pro_calls": 0,
        "attempted_flash_calls": 0,
        "usable_flash_responses": 0,
        "runtime_profile": {
            "data": runtime_profile.to_dict(),
            "rendered": runtime_profile.render(),
            "digest": runtime_profile.digest(),
        },
        "system_prompts": {
            group: {
                "sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
                "length": len(system),
                "injected_runtime_profile": group == "RUNTIME_PROFILE_RUNTIME",
            }
            for group, system in group_systems.items()
        },
        "frozen_request_contexts": {
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
        },
    }
    _write_state(state_path, state)

    metadata = {
        "type": "run",
        **state["metadata"],
        "pro_payloads": [],
        "pro_calls": 0,
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
            finish_reason = "stop"
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

        if error:
            evaluation = {
                "translation": "",
                "normalization_changed": False,
                "validation": {"ok": False, "reason": "request_error"},
            }
        else:
            evaluation = evaluate_output(
                raw_translation,
                case=case,
                context=context.production,
                runtime_case=frozen_case.runtime_case,
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
            "normalization_changed": evaluation["normalization_changed"],
            "validation_result": evaluation["validation"],
            "finish_reason": finish_reason,
            "latency_s": latency,
            "usage": usage,
            "estimated_input_tokens": estimate_tokens(json.dumps(messages, ensure_ascii=False)),
            "injected_runtime_profile": group == "RUNTIME_PROFILE_RUNTIME",
            "matched_reference_ids": [
                entry.id for entry in frozen_case.runtime_case.plan.matched_entries
            ],
            "matched_memory_ids": list(frozen_case.matched_memory_ids),
            "source_technical_terms": list(frozen_case.runtime_case.technical_terms),
            "error": error,
        }
        records.append(record)
        print(
            f"{index:02d}/{len(jobs)} {group:<25} {case.id} rep={repetition} "
            f"ok={not error} validation={record['validation_result']['ok']} "
            f"finish={finish_reason!r} latency={latency:.3f}s"
        )
        if args.delay > 0 and not args.dry_run:
            time.sleep(args.delay)

    summary = _summarize(records)
    metadata["attempted_flash_calls"] = state["attempted_flash_calls"]
    metadata["usable_flash_responses"] = state["usable_flash_responses"]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    write_blind(blind_path, records, seed=args.seed)
    state["status"] = "completed"
    _write_state(state_path, state)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"State: {state_path.resolve()}")
    print(f"Blind review: {blind_path.resolve()}")
    return output_path, state_path, blind_path


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
