"""POC 5: replay visible Pro conversation history to a non-thinking Flash model.

Unlike the structured-state experiments, this POC preserves the exact visible
assistant ``content`` produced by Pro and replays it with its original role.
It never sends or stores ``reasoning_content``. No retrieval, placeholders,
fuzzy matching, checker model, or repair pass is used.

Groups:

* DIRECT: rules/references and OCR are sent directly in one user turn.
* NEUTRAL_SESSION: rules -> generic assistant acknowledgement -> OCR.
* PRO_SESSION: rules -> Pro's visible final response -> OCR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.settings import AppSettings
from app.translation.client import TranslationError
from poc.poc_reference_injection import (
    GlossaryEntry,
    PocCase,
    PocDataset,
    _percentile,
    _send_request,
    estimate_tokens,
    evaluate_response,
    load_dataset,
)


GROUPS = ("DIRECT", "NEUTRAL_SESSION", "PRO_SESSION")

_INITIALIZER_SYSTEM = """You are the thinking startup turn of a persistent translation session.
Read the user's complete translation rules and references carefully.
Think privately, then return one visible operational handoff for the assistant that will answer later OCR translation turns.
The handoff should confirm the real goal, resolve priorities, explain how to preserve meaning while following format rules, and reuse only confirmed examples from the package.
Do not translate any unseen test sentence. Do not output JSON. Do not mention hidden reasoning or chain-of-thought.
Your final visible response will be stored verbatim as an assistant message and shown to a non-thinking Flash model in later turns.
"""

_NEUTRAL_ACK = (
    "Understood. I will translate subsequent OCR text according to the supplied "
    "rules and references, preserve the complete meaning, and return only the translation."
)

_DRY_PRO_CONTENT = """I understand the task. I will preserve the complete Chinese meaning in natural Japanese, then apply the confirmed hiragana, glossary-bracketing, and spacing rules. I will treat OCR uncertainty conservatively and return only the translation. Confirmed example: 打开软件 -> [そふとうぇあ]を  ひらく."""


class SessionReplayError(ValueError):
    """Raised when a visible session transcript cannot be safely reused."""


def _entry_json(entry: GlossaryEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "source": entry.source,
        "target": entry.target,
        "aliases": list(entry.aliases),
        "format": entry.format,
        "risk": entry.risk,
    }


def build_rule_package(
    dataset: PocDataset,
    entries: tuple[GlossaryEntry, ...],
) -> str:
    """Return the complete visible user turn; test cases are never included."""

    package = {
        "task": "Prepare for later Chinese OCR to Japanese translation turns.",
        "raw_user_constraints": dataset.raw_user_constraints,
        "confirmed_global_rules": list(dataset.global_rules),
        "confirmed_style_rules": list(dataset.style_rules),
        "forbidden_outputs": list(dataset.forbidden_outputs),
        "confirmed_examples": list(dataset.examples),
        "reference_glossary": [_entry_json(entry) for entry in entries],
        "source_language": "Chinese OCR text",
        "target_language": "Japanese",
    }
    return (
        "These are my complete translation rules and references. Understand them now and "
        "apply them to every later OCR message in this conversation.\n\n"
        + json.dumps(package, ensure_ascii=False, indent=2)
    )


def validate_visible_content(content: str) -> str:
    content = content.strip()
    if not content:
        raise SessionReplayError("Pro returned empty visible content")
    if len(content) > 20_000:
        raise SessionReplayError("Pro visible content exceeds 20,000 characters")
    return content


def build_transcript(
    dataset: PocDataset,
    rule_package: str,
    pro_content: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": dataset.base_system_prompt},
        {"role": "user", "content": rule_package},
        {"role": "assistant", "content": validate_visible_content(pro_content)},
    ]


def save_transcript(
    path: Path,
    messages: list[dict[str, str]],
    *,
    dataset_sha256: str,
    thinking_model: str,
    source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": dataset_sha256,
            "thinking_model": thinking_model,
            "source": source,
            "contains_reasoning_content": False,
        },
        "messages": messages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_transcript(path: Path, *, dataset_sha256: str) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    if metadata.get("dataset_sha256") != dataset_sha256:
        raise SessionReplayError("visible transcript belongs to a different prompt package")
    if metadata.get("contains_reasoning_content") is not False:
        raise SessionReplayError("transcript must explicitly exclude reasoning_content")
    messages = raw.get("messages") if isinstance(raw, dict) else None
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise SessionReplayError("transcript must contain system -> user -> assistant messages")
    cleaned = []
    for item in messages:
        if set(item) != {"role", "content"}:
            raise SessionReplayError("transcript messages may contain only role and content")
        content = str(item["content"]).strip()
        if not content:
            raise SessionReplayError("transcript contains empty message content")
        cleaned.append({"role": item["role"], "content": content})
    return cleaned


def _ocr_turn(case: PocCase) -> str:
    return f"<OCR_TEXT>\n{case.source}\n</OCR_TEXT>"


def build_runtime_messages(
    dataset: PocDataset,
    rule_package: str,
    pro_transcript: list[dict[str, str]],
    case: PocCase,
    group: str,
) -> list[dict[str, str]]:
    ocr = _ocr_turn(case)
    if group == "DIRECT":
        return [
            {"role": "system", "content": dataset.base_system_prompt},
            {"role": "user", "content": rule_package + "\n\n" + ocr},
        ]
    if group == "NEUTRAL_SESSION":
        return [
            {"role": "system", "content": dataset.base_system_prompt},
            {"role": "user", "content": rule_package},
            {"role": "assistant", "content": _NEUTRAL_ACK},
            {"role": "user", "content": ocr},
        ]
    if group == "PRO_SESSION":
        return [*pro_transcript, {"role": "user", "content": ocr}]
    raise ValueError(f"unknown group: {group}")


def evaluate_hard_constraints(
    dataset: PocDataset,
    case: PocCase,
    translation: str,
) -> dict[str, Any]:
    """Apply dataset-wide hard rules to every category, not only format cases."""

    base = evaluate_response(
        dataset,
        case,
        dataset.glossary_sets["200"],
        translation,
        {},
    )
    script_ok = bool(re.fullmatch(r"[\u3041-\u3096\[\] ]+", translation))
    edge_ok = translation == translation.strip()
    space_runs = [len(match.group(0)) for match in re.finditer(r" +", translation)]
    spaces_exact = all(length == 2 for length in space_runs)
    spacing_required = case.id != "format_01"
    spacing_present = bool(space_runs) if spacing_required else True
    checks = {
        "hiragana_brackets_spaces_only": script_ok,
        "no_edge_whitespace": edge_ok,
        "all_existing_space_runs_are_exactly_two": spaces_exact,
        "double_space_present_when_required": spacing_present,
        "case_specific_and_terminology_checks": bool(base["machine_ok"]),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "space_run_lengths": space_runs,
        "missing_expected_terms": base["missing_expected_terms"],
        "present_forbidden_outputs": base["present_forbidden_outputs"],
    }


def summarize(records: list[dict[str, Any]], *, repetitions: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [record for record in records if record["group"] == group]
        completed = [record for record in selected if record.get("response") is not None]
        latencies = [float(record["latency_s"]) for record in completed]
        stable_cases = 0
        comparable_cases = 0
        for case_id in sorted({record["case_id"] for record in completed}):
            outputs = [
                record["response"]
                for record in completed
                if record["case_id"] == case_id
            ]
            if len(outputs) == repetitions:
                comparable_cases += 1
                stable_cases += len(set(outputs)) == 1
        groups[group] = {
            "requests": len(selected),
            "completed": len(completed),
            "errors": sum(bool(record.get("error")) for record in selected),
            "hard_constraint_pass": sum(
                bool(record["hard_constraint_evaluation"]["ok"]) for record in completed
            ),
            "missing_expected_terms": sum(
                len(record["hard_constraint_evaluation"]["missing_expected_terms"])
                for record in completed
            ),
            "provider_prompt_tokens": sum(
                int(record.get("usage", {}).get("prompt_tokens", 0) or 0)
                for record in completed
            ),
            "estimated_input_tokens_mean": round(
                statistics.fmean(record["estimated_input_tokens"] for record in selected), 1
            ) if selected else None,
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "exact_repeat_stability": (
                f"{stable_cases}/{comparable_cases}" if comparable_cases else None
            ),
            "semantic_fidelity": "NOT_SCORED_USE_BLIND_REVIEW",
            "constraint_aware_naturalness": "NOT_SCORED_USE_BLIND_REVIEW",
        }
    return {"type": "summary", "repetitions": repetitions, "groups": groups}


def write_blind_review(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
    rows = []
    for record in records:
        if record.get("response") is None:
            continue
        rows.append(
            {
                "blind_id": record["blind_id"],
                "case_id": record["case_id"],
                "category": record["category"],
                "source": record["source"],
                "translation": record["response"],
                "confirmed_user_constraints": (
                    "Judge naturalness under the required all-hiragana, glossary-bracket, "
                    "and exact-double-space format. Do not reward breaking a hard rule."
                ),
                "expected_fixed_outputs": record["expected_fixed_outputs"],
                "semantic_fidelity_0_to_5": None,
                "naturalness_0_to_5": None,
                "meaning_error": "",
                "review_notes": "",
            }
        )
    random.Random(seed ^ 0x51A7).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("logs") / "poc"
    return (
        root / f"session-replay-{stamp}.jsonl",
        root / f"session-replay-transcript-{stamp}.json",
        root / f"session-replay-blind-{stamp}.jsonl",
    )


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path, strict=True)
    entries = dataset.glossary_sets["200"]
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    output_default, transcript_default, blind_default = _default_paths()
    output_path = Path(args.output) if args.output else output_default
    transcript_path = Path(args.transcript_output) if args.transcript_output else transcript_default
    blind_path = Path(args.blind_output) if args.blind_output else blind_default
    output_path.parent.mkdir(parents=True, exist_ok=True)

    settings = AppSettings.load()
    fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
    thinking_model = (
        args.thinking_model.strip()
        or settings.ai.thinking_model_name
        or "deepseek-v4-pro"
    )
    if not args.dry_run and (
        not settings.ai.base_url or not settings.ai.api_key or not fast_model or not thinking_model
    ):
        raise SystemExit("API base URL, key, fast model, and thinking model must be configured")

    rule_package = build_rule_package(dataset, entries)
    initializer_messages = [
        {"role": "system", "content": _INITIALIZER_SYSTEM},
        {"role": "user", "content": rule_package},
    ]
    initializer_payload = {
        "model": thinking_model,
        "messages": initializer_messages,
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }
    bootstrap_record: dict[str, Any] = {
        "type": "bootstrap",
        "payload": initializer_payload,
        "visible_content": None,
        "usage": {},
        "latency_s": None,
        "source": "",
        "reasoning_content_stored": False,
    }

    if args.transcript:
        pro_transcript = load_transcript(Path(args.transcript), dataset_sha256=dataset_sha256)
        pro_content = pro_transcript[-1]["content"]
        bootstrap_record["source"] = str(Path(args.transcript).resolve())
    elif args.dry_run:
        pro_content = _DRY_PRO_CONTENT
        pro_transcript = build_transcript(dataset, rule_package, pro_content)
        bootstrap_record["source"] = "deterministic_dry_run_visible_content"
    else:
        started = time.perf_counter()
        pro_content, usage = _send_request(
            base_url=settings.ai.base_url,
            api_key=settings.ai.api_key,
            payload=initializer_payload,
            timeout_seconds=args.bootstrap_timeout,
        )
        bootstrap_record["latency_s"] = round(time.perf_counter() - started, 3)
        bootstrap_record["usage"] = usage
        bootstrap_record["source"] = "thinking_model_visible_content"
        pro_transcript = build_transcript(dataset, rule_package, pro_content)
    bootstrap_record["visible_content"] = validate_visible_content(pro_content)
    save_transcript(
        transcript_path,
        pro_transcript,
        dataset_sha256=dataset_sha256,
        thinking_model=thinking_model,
        source=bootstrap_record["source"],
    )

    if args.bootstrap_only:
        print(f"Visible transcript: {transcript_path.resolve()}")
        return output_path, transcript_path, blind_path

    run_id = datetime.now(timezone.utc).isoformat()
    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    cases = dataset.cases[: args.limit or None]
    entries_by_id = {entry.id: entry for entry in entries}
    metadata = {
        "type": "run",
        "run_id": run_id,
        "experiment": "visible_session_replay",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "translation_thinking": "disabled",
        "reasoning_content_replayed": False,
        "groups": GROUPS,
        "repetitions": args.repetitions,
        "dry_run": args.dry_run,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        handle.write(json.dumps(bootstrap_record, ensure_ascii=False) + "\n")
        request_index = 0
        for repetition in range(1, args.repetitions + 1):
            for case in cases:
                ordered_groups = list(GROUPS)
                rng.shuffle(ordered_groups)
                for group in ordered_groups:
                    request_index += 1
                    messages = build_runtime_messages(
                        dataset,
                        rule_package,
                        pro_transcript,
                        case,
                        group,
                    )
                    payload = {
                        "model": fast_model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 4096,
                        "thinking": {"type": "disabled"},
                    }
                    blind_id = hashlib.sha256(
                        f"{run_id}:{request_index}:{args.seed}".encode("utf-8")
                    ).hexdigest()[:12]
                    record: dict[str, Any] = {
                        "type": "result",
                        "run_id": run_id,
                        "request_index": request_index,
                        "blind_id": blind_id,
                        "repetition": repetition,
                        "group": group,
                        "case_id": case.id,
                        "category": case.category,
                        "source": case.source,
                        "expected_fixed_outputs": [
                            entries_by_id[entry_id].target
                            for entry_id in case.expected_entry_ids
                            if entry_id in entries_by_id
                        ],
                        "estimated_input_tokens": estimate_tokens(
                            "\n".join(message["content"] for message in messages)
                        ),
                        "payload": payload,
                        "response": None,
                        "usage": {},
                        "latency_s": None,
                        "error": None,
                        "evaluation": None,
                        "hard_constraint_evaluation": None,
                    }
                    if not args.dry_run:
                        started = time.perf_counter()
                        try:
                            response, usage = _send_request(
                                base_url=settings.ai.base_url,
                                api_key=settings.ai.api_key,
                                payload=payload,
                                timeout_seconds=args.translation_timeout,
                            )
                            record["response"] = response
                            record["usage"] = usage
                            record["evaluation"] = evaluate_response(
                                dataset, case, entries, response, {}
                            )
                            record["hard_constraint_evaluation"] = evaluate_hard_constraints(
                                dataset, case, response
                            )
                        except TranslationError as exc:
                            record["error"] = str(exc)
                        record["latency_s"] = round(time.perf_counter() - started, 3)
                        print(
                            f"{request_index:03d} repeat={repetition} group={group} "
                            f"case={case.id} elapsed={record['latency_s']:.3f}s "
                            f"ok={record['error'] is None}"
                        )
                        if args.delay:
                            time.sleep(args.delay)
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()

        summary = summarize(records, repetitions=args.repetitions)
        summary["run_id"] = run_id
        summary["bootstrap"] = {
            "source": bootstrap_record["source"],
            "latency_s": bootstrap_record["latency_s"],
            "usage": bootstrap_record["usage"],
            "transcript": str(transcript_path.resolve()),
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    write_blind_review(blind_path, records, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Visible transcript: {transcript_path.resolve()}")
    print(f"Blind review: {blind_path.resolve()}")
    return output_path, transcript_path, blind_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="poc/data/reference_poc_dataset.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--transcript-output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--transcript", default="", help="reuse visible transcript; skip Pro")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--thinking-model", default="")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="debug only")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--bootstrap-timeout", type=float, default=180.0)
    parser.add_argument("--translation-timeout", type=float, default=60.0)
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be at least 1")
    try:
        run(args)
    except (SessionReplayError, json.JSONDecodeError, OSError, TranslationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
