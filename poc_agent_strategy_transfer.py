"""Test whether a visible Pro playbook improves held-out Flash translations.

This experiment isolates the only startup-Agent effect that can survive a
normal DeepSeek multi-turn request: visible assistant content.  Both groups
receive the same system prompt and raw startup brief.  The control receives a
neutral acknowledgement, while the Agent group receives a structured playbook
created once by Pro with thinking enabled.  Held-out OCR cases are never sent
to Pro.
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

from app.settings import AppSettings
from app.translation.client import TranslationError
from poc_agent_memory_transfer import (
    build_common_system,
    evaluate_format,
    extract_json_object,
    normalize_surface,
)
from poc_reference_injection import _percentile, _send_request, estimate_tokens, load_dataset


GROUPS = ("RAW_BRIEF", "PRO_PLAYBOOK")
_NEUTRAL_ACK = (
    "Understood. I will use the startup brief and all confirmed rules for later "
    "OCR translation turns, preserve the complete meaning, and return only the translation."
)

_PLAYBOOK_SCHEMA = """Return only one JSON object with exactly this top-level shape:
{
  "version": 1,
  "role": "translation_playbook",
  "domain_sense_decisions": [
    {"source_expression": "Chinese expression", "contextual_meaning": "meaning", "preferred_japanese": ["choice"], "avoid": ["bad literal choice"]}
  ],
  "natural_patterns": [
    {"purpose": "semantic relation", "japanese_pattern": "reusable pattern"}
  ],
  "quality_checks": ["short operational check"],
  "generated_examples": [
    {"source": "new Chinese example invented by you", "target": "hiragana-only Japanese using exactly two ASCII spaces between units"}
  ]
}

Create 6-12 domain sense decisions, 4-8 natural patterns, 4-8 checks, and 6-10
short generated examples. Generalize from the brief; do not merely repeat the
format rules. Focus on choices that prevent literal, semantically incomplete,
or unnatural translation. The held-out evaluation sentences are not present,
so do not claim to have seen them. Do not output chain-of-thought,
reasoning_content, Markdown fences, or commentary."""


class StrategyPocError(ValueError):
    """Raised when the strategy-transfer experiment is not auditable."""


@dataclass(frozen=True)
class StrategyCase:
    id: str
    category: str
    source: str
    review_focus: str
    fixture_translation: str
    required_all: tuple[str, ...]
    required_any: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...]


@dataclass(frozen=True)
class StrategyDataset:
    version: int
    status: str
    name: str
    description: str
    startup_brief: str
    cases: tuple[StrategyCase, ...]


def load_strategy_dataset(path: Path) -> StrategyDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise StrategyPocError("strategy dataset must be a JSON object")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise StrategyPocError("strategy dataset cases must be a list")

    def strings(item: dict[str, Any], key: str) -> tuple[str, ...]:
        value = item.get(key)
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            raise StrategyPocError(f"{item.get('id', '?')}.{key} must be a string list")
        return tuple(x.strip() for x in value if x.strip())

    cases = []
    for item in cases_raw:
        if not isinstance(item, dict):
            raise StrategyPocError("each strategy case must be an object")
        required_any_raw = item.get("required_any")
        if not isinstance(required_any_raw, list) or any(
            not isinstance(group, list) or not group or any(not isinstance(x, str) for x in group)
            for group in required_any_raw
        ):
            raise StrategyPocError(f"{item.get('id', '?')}.required_any is invalid")
        cases.append(
            StrategyCase(
                id=str(item.get("id", "")).strip(),
                category=str(item.get("category", "")).strip(),
                source=str(item.get("source", "")).strip(),
                review_focus=str(item.get("review_focus", "")).strip(),
                fixture_translation=str(item.get("fixture_translation", "")).strip(),
                required_all=strings(item, "required_all"),
                required_any=tuple(
                    tuple(x.strip() for x in group if x.strip()) for group in required_any_raw
                ),
                forbidden=strings(item, "forbidden"),
            )
        )
    dataset = StrategyDataset(
        version=int(raw.get("version", 0)),
        status=str(raw.get("status", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        description=str(raw.get("description", "")).strip(),
        startup_brief=str(raw.get("startup_brief", "")).strip(),
        cases=tuple(cases),
    )
    validate_strategy_dataset(dataset)
    return dataset


def validate_strategy_dataset(dataset: StrategyDataset) -> None:
    errors = []
    if dataset.version != 1 or dataset.status != "frozen_poc":
        errors.append("dataset must be frozen_poc version 1")
    if len(dataset.cases) != 12:
        errors.append(f"exactly 12 held-out cases are required, found {len(dataset.cases)}")
    if not dataset.startup_brief:
        errors.append("startup_brief is empty")
    ids = [case.id for case in dataset.cases]
    sources = [case.source for case in dataset.cases]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        errors.append("case ids must be non-empty and unique")
    if len(set(sources)) != len(sources) or any(not item for item in sources):
        errors.append("case sources must be non-empty and unique")
    for case in dataset.cases:
        if not case.category or not case.review_focus or not case.fixture_translation:
            errors.append(f"{case.id} is missing review metadata or fixture translation")
        if case.source in dataset.startup_brief:
            errors.append(f"held-out source leaked into startup_brief: {case.id}")
        if not evaluate_format(case.fixture_translation)["ok"]:
            errors.append(f"fixture translation violates output format: {case.id}")
    if errors:
        raise StrategyPocError("strategy dataset is invalid:\n- " + "\n- ".join(errors))


def build_session_system(translation_dataset_path: Path) -> tuple[str, str]:
    translation = load_dataset(translation_dataset_path, strict=True)
    common = build_common_system(translation, translation.glossary_sets["20"])
    protocol = """## Persistent Agent startup protocol
- A final <SESSION_BOOTSTRAP> turn is a control request, not OCR text.
- Pro must answer that control turn with the requested visible JSON playbook.
- A final <OCR_TEXT> turn is the only text to translate.
- Historical bootstrap content is trusted session context for later Flash turns.
- Return only the translation when the final turn is OCR_TEXT."""
    result = common + "\n\n" + protocol
    return result, hashlib.sha256(result.encode("utf-8")).hexdigest()


def build_bootstrap_turn(dataset: StrategyDataset) -> str:
    return (
        "<SESSION_BOOTSTRAP>\n"
        "You are the thinking startup turn of a persistent Pro/Flash translation Agent. "
        "Study the user's operating context below and create a visible reusable playbook "
        "for a non-thinking Flash model that will receive later OCR turns in this same session.\n\n"
        "<USER_OPERATING_CONTEXT>\n"
        + dataset.startup_brief
        + "\n</USER_OPERATING_CONTEXT>\n\n"
        + _PLAYBOOK_SCHEMA
        + "\n</SESSION_BOOTSTRAP>"
    )


def validate_playbook(raw: dict[str, Any], dataset: StrategyDataset) -> dict[str, Any]:
    if raw.get("version") != 1 or raw.get("role") != "translation_playbook":
        raise StrategyPocError("Pro playbook has wrong version or role")
    limits = {
        "domain_sense_decisions": (6, 12),
        "natural_patterns": (4, 8),
        "quality_checks": (4, 8),
        "generated_examples": (6, 10),
    }
    for key, (minimum, maximum) in limits.items():
        value = raw.get(key)
        if not isinstance(value, list) or not minimum <= len(value) <= maximum:
            raise StrategyPocError(f"Pro playbook {key} must contain {minimum}-{maximum} items")
    if "reasoning_content" in json.dumps(raw, ensure_ascii=False):
        raise StrategyPocError("Pro playbook must not contain reasoning_content")
    rendered = json.dumps(raw, ensure_ascii=False)
    leaked = [case.id for case in dataset.cases if case.source in rendered]
    if leaked:
        raise StrategyPocError("Pro playbook leaked held-out sources: " + ", ".join(leaked))
    if len(rendered) > 30_000:
        raise StrategyPocError("Pro playbook exceeds 30,000 characters")
    return raw


def make_dry_playbook() -> dict[str, Any]:
    return {
        "version": 1,
        "role": "translation_playbook",
        "domain_sense_decisions": [
            {"source_expression": term, "contextual_meaning": meaning, "preferred_japanese": [jp], "avoid": [avoid]}
            for term, meaning, jp, avoid in [
                ("跑测试", "execute tests", "じっしする", "はしる"),
                ("复现", "reproduce a defect", "さいげんする", "もういちどみる"),
                ("线上", "production environment", "ほんばんかんきょう", "せんじょう"),
                ("回滚", "restore a previous release", "ろおるばっくする", "もどってあるく"),
                ("合并代码", "merge code", "こおどをまあじする", "こおどをあわせる"),
                ("返回响应", "return a response", "れすぽんすをかえす", "へんじしてかえる"),
            ]
        ],
        "natural_patterns": [
            {"purpose": "condition", "japanese_pattern": "もし〜なら"},
            {"purpose": "sequence", "japanese_pattern": "〜したあとで"},
            {"purpose": "contrast", "japanese_pattern": "〜が〜"},
            {"purpose": "prohibition", "japanese_pattern": "〜しないでください"},
        ],
        "quality_checks": [
            "preserve negation",
            "preserve condition and sequence",
            "preserve agent and object",
            "prefer contextual domain meaning",
        ],
        "generated_examples": [
            {"source": f"校准例句{i}", "target": f"こうせい  れいぶん  {word}"}
            for i, word in enumerate(("いち", "に", "さん", "よん", "ご", "ろく"), 1)
        ],
    }


def _ocr(source: str) -> str:
    return f"<OCR_TEXT>\n{source}\n</OCR_TEXT>"


def build_group_messages(
    *,
    system: str,
    bootstrap_turn: str,
    playbook: dict[str, Any],
    case: StrategyCase,
    group: str,
) -> list[dict[str, str]]:
    assistant = (
        _NEUTRAL_ACK
        if group == "RAW_BRIEF"
        else json.dumps(playbook, ensure_ascii=False, indent=2)
    )
    if group not in GROUPS:
        raise ValueError(f"unknown strategy group: {group}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": bootstrap_turn},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": _ocr(case.source)},
    ]


def _canonical(text: str) -> str:
    return "".join(char for char in text.casefold() if char not in "[] \t\r\n")


def evaluate_anchors(case: StrategyCase, translation: str) -> dict[str, Any]:
    canonical = _canonical(translation)
    required_all = {item: _canonical(item) in canonical for item in case.required_all}
    required_any = [
        {"options": list(group), "ok": any(_canonical(item) in canonical for item in group)}
        for group in case.required_any
    ]
    forbidden = {item: _canonical(item) not in canonical for item in case.forbidden}
    checks = [*required_all.values(), *(item["ok"] for item in required_any), *forbidden.values()]
    return {
        "ok": all(checks),
        "required_all": required_all,
        "required_any": required_any,
        "forbidden_absent": forbidden,
        "quality_claim": False,
    }


def summarize(records: list[dict[str, Any]], repetitions: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [row for row in records if row["group"] == group]
        completed = [row for row in selected if row.get("translation")]
        latencies = [float(row["latency_s"]) for row in completed]
        stable = 0
        for case_id in sorted({row["case_id"] for row in completed}):
            outputs = [row["translation"] for row in completed if row["case_id"] == case_id]
            stable += len(outputs) == repetitions and len(set(outputs)) == 1
        groups[group] = {
            "requests": len(selected),
            "completed": len(completed),
            "errors": sum(bool(row.get("error")) for row in selected),
            "format_pass": sum(bool(row["hard_constraint_evaluation"]["ok"]) for row in completed),
            "diagnostic_anchor_pass": sum(bool(row["anchor_evaluation"]["ok"]) for row in completed),
            "provider_prompt_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens", 0) or 0) for row in completed),
            "estimated_input_tokens_mean": round(statistics.fmean(row["estimated_input_tokens"] for row in selected), 1) if selected else None,
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "exact_repeat_stability": f"{stable}/{len({row['case_id'] for row in completed})}" if completed else None,
            "semantic_fidelity": "NOT_SCORED_USE_BLIND_REVIEW",
            "naturalness": "NOT_SCORED_USE_BLIND_REVIEW",
        }
    return {
        "type": "summary",
        "repetitions": repetitions,
        "groups": groups,
        "decision_gate": {
            "PRO_PLAYBOOK": "Integrate cached Pro startup playbook only if blind semantic quality improves materially without naturalness or hard-format regression.",
            "RAW_BRIEF": "If equal or better, startup Pro does not justify latency; retain Pro for explicit thinking mode and confirmed-feedback maintenance.",
        },
    }


def write_blind(path: Path, records: list[dict[str, Any]], seed: int) -> None:
    rows = [
        {
            "blind_id": row["blind_id"],
            "case_id": row["case_id"],
            "category": row["category"],
            "source": row["source"],
            "review_focus": row["review_focus"],
            "translation": row["translation"],
            "semantic_fidelity_0_to_5": None,
            "naturalness_0_to_5": None,
            "meaning_error": "",
            "review_notes": "",
        }
        for row in records
        if row.get("translation")
    ]
    random.Random(seed ^ 0x51A7).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("logs/poc")
    return (
        root / f"agent-strategy-{stamp}.jsonl",
        root / f"agent-strategy-session-{stamp}.json",
        root / f"agent-strategy-blind-{stamp}.jsonl",
    )


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    strategy_path = Path(args.dataset)
    translation_path = Path(args.translation_dataset)
    dataset = load_strategy_dataset(strategy_path)
    system, prompt_hash = build_session_system(translation_path)
    bootstrap_turn = build_bootstrap_turn(dataset)
    for case in dataset.cases:
        if case.source in bootstrap_turn:
            raise StrategyPocError(f"held-out source leaked into Pro request: {case.id}")

    settings = AppSettings.load()
    fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
    thinking_model = args.thinking_model.strip() or settings.ai.thinking_model_name or "deepseek-v4-pro"
    if not args.dry_run and (not settings.ai.base_url or not settings.ai.api_key):
        raise StrategyPocError("API base URL and key must be configured")

    default_output, default_session, default_blind = _default_paths()
    output_path = Path(args.output) if args.output else default_output
    session_path = Path(args.session_output) if args.session_output else default_session
    blind_path = Path(args.blind_output) if args.blind_output else default_blind
    for path in (output_path, session_path, blind_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    pro_payload = {
        "model": thinking_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": bootstrap_turn},
        ],
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }
    pro_started = time.perf_counter()
    if args.dry_run:
        playbook = validate_playbook(make_dry_playbook(), dataset)
        pro_response = json.dumps(playbook, ensure_ascii=False)
        pro_usage: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        pro_latency = 0.0
    else:
        pro_response, pro_usage = _send_request(
            base_url=settings.ai.base_url,
            api_key=settings.ai.api_key,
            payload=pro_payload,
            timeout_seconds=args.pro_timeout,
        )
        pro_latency = round(time.perf_counter() - pro_started, 3)
        playbook = validate_playbook(extract_json_object(pro_response), dataset)

    canonical_playbook = json.dumps(playbook, ensure_ascii=False, indent=2)
    session = {
        "metadata": {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_hash": hashlib.sha256(strategy_path.read_bytes()).hexdigest(),
            "prompt_hash": prompt_hash,
            "thinking_model": thinking_model,
            "fast_model": fast_model,
            "contains_reasoning_content": False,
            "held_out_case_count": len(dataset.cases),
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": bootstrap_turn},
            {"role": "assistant", "content": canonical_playbook},
        ],
    }
    session_path.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run_id = datetime.now(timezone.utc).isoformat()
    metadata = {
        "type": "run",
        "run_id": run_id,
        "experiment": "visible_pro_strategy_transfer",
        "groups": list(GROUPS),
        "cases": len(dataset.cases),
        "repetitions": args.repetitions,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "pro_sees_held_out_sources": False,
        "reasoning_content_replayed": False,
        "api_calls_expected": {"pro": 1, "flash": len(dataset.cases) * len(GROUPS) * args.repetitions},
        "dry_run": args.dry_run,
    }
    bootstrap_record = {
        "type": "bootstrap",
        "payload": pro_payload,
        "response": pro_response,
        "playbook": playbook,
        "usage": pro_usage,
        "latency_s": pro_latency,
    }
    records: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    schedule = [
        (repetition, group, case)
        for repetition in range(1, args.repetitions + 1)
        for case in dataset.cases
        for group in GROUPS
    ]
    rng.shuffle(schedule)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        handle.write(json.dumps(bootstrap_record, ensure_ascii=False) + "\n")
        for index, (repetition, group, case) in enumerate(schedule, 1):
            messages = build_group_messages(
                system=system,
                bootstrap_turn=bootstrap_turn,
                playbook=playbook,
                case=case,
                group=group,
            )
            payload = {
                "model": fast_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 4096,
                "thinking": {"type": "disabled"},
            }
            started = time.perf_counter()
            error = None
            usage: dict[str, Any] = {}
            if args.dry_run:
                response = case.fixture_translation
                latency = 0.0
            else:
                try:
                    response, usage = _send_request(
                        base_url=settings.ai.base_url,
                        api_key=settings.ai.api_key,
                        payload=payload,
                        timeout_seconds=args.flash_timeout,
                    )
                    latency = round(time.perf_counter() - started, 3)
                except TranslationError as exc:
                    response = ""
                    latency = round(time.perf_counter() - started, 3)
                    error = str(exc)
            normalized = normalize_surface(response) if response else {"translation": "", "changed": False}
            translation = normalized["translation"]
            hard = evaluate_format(translation) if translation else {"ok": False, "checks": {}}
            anchor = evaluate_anchors(case, translation) if translation else {"ok": False, "quality_claim": False}
            record = {
                "type": "result",
                "blind_id": hashlib.sha256(f"{run_id}:{group}:{case.id}:{repetition}".encode()).hexdigest()[:12],
                "request_index": index,
                "group": group,
                "case_id": case.id,
                "category": case.category,
                "repetition": repetition,
                "source": case.source,
                "review_focus": case.review_focus,
                "payload": payload,
                "response": response,
                "translation": translation,
                "normalization": normalized,
                "hard_constraint_evaluation": hard,
                "anchor_evaluation": anchor,
                "estimated_input_tokens": estimate_tokens("\n".join(message["content"] for message in messages)),
                "usage": usage,
                "latency_s": latency,
                "error": error,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index:02d}/{len(schedule)} {group} {case.id} elapsed={latency:.3f}s ok={not error}")
            if args.delay and not args.dry_run:
                time.sleep(args.delay)
        summary = summarize(records, args.repetitions)
        summary["bootstrap"] = {
            "latency_s": pro_latency,
            "usage": pro_usage,
            "session_output": str(session_path.resolve()),
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    write_blind(blind_path, records, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Session: {session_path.resolve()}")
    print(f"Blind review: {blind_path.resolve()}")
    return output_path, session_path, blind_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="poc_data/agent_strategy_transfer_dataset.json")
    parser.add_argument("--translation-dataset", default="poc_data/reference_poc_dataset.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--session-output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--thinking-model", default="")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--pro-timeout", type=float, default=180.0)
    parser.add_argument("--flash-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions != 2:
        raise SystemExit("formal strategy-transfer POC requires exactly 2 repetitions")
    try:
        run(args)
    except (StrategyPocError, OSError, json.JSONDecodeError, TranslationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
