"""POC 6.1 canary: typed reference slots plus deterministic output safety.

The canary reuses both DIRECT and Agent-v6 evidence from POC 6, applies the new
safe formatter offline to the old Agent output, and sends only eight difficult
cases twice through Flash with typed reference slots.  This isolates the value
of semantic slot metadata from the value of deterministic formatting.

No Pro request is made: the persisted POC-6 Agent memory is reused.  The formal
run therefore makes exactly 16 new Flash requests.
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
from typing import Any, Iterable

from app.settings import AppSettings
from app.translation.client import TranslationError
from poc_agent_bootstrap import find_fuzzy_candidates
from poc_agent_memory import (
    AgentMemoryError,
    compile_agent_system,
    load_agent_memory,
)
from poc_reference_injection import (
    GlossaryEntry,
    PocCase,
    PocDataset,
    _percentile,
    _send_request,
    estimate_tokens,
    evaluate_response,
    load_dataset,
    match_glossary,
    protect_glossary_terms,
    restore_placeholders,
)
from poc_session_replay import evaluate_hard_constraints


GROUPS = ("DIRECT_BASELINE", "AGENT_V6_NORMALIZED", "AGENT_V61_TYPED")
CANARY_CASE_IDS = (
    "semantic_02",
    "semantic_03",
    "terminology_04",
    "style_02",
    "style_03",
    "style_04",
    "ocr_noise_03",
    "ocr_noise_04",
)

_ACTION_ENTRY_IDS = {
    "se_compile",
    "se_debug",
    "se_develop",
    "se_deploy",
    "se_test",
}
_PUNCTUATION_PATTERN = re.compile(r"[、。，．,.！？!?；;：:]+")
_BRACKET_PATTERN = re.compile(r"\[[^\[\]]+\]")


class CanaryError(ValueError):
    """Raised when canary evidence or configuration is invalid."""


def select_canary_cases(dataset: PocDataset) -> tuple[PocCase, ...]:
    by_id = {case.id: case for case in dataset.cases}
    missing = [case_id for case_id in CANARY_CASE_IDS if case_id not in by_id]
    if missing:
        raise CanaryError("dataset is missing canary cases: " + ", ".join(missing))
    return tuple(by_id[case_id] for case_id in CANARY_CASE_IDS)


def semantic_type(entry: GlossaryEntry) -> tuple[str, str]:
    """Return a compact semantic type and grammar hint for one local term."""

    if entry.id in _ACTION_ENTRY_IDS:
        return (
            "action_concept",
            (
                "Place this slot where the action belongs. Add hiragana such as する, "
                "される, or すれば outside the slot when tense, voice, or condition requires it. "
                "Do not concatenate the action slot directly to a noun slot."
            ),
        )
    return (
        "noun_concept",
        (
            "Treat this slot as a noun or named concept. Attach the grammatically required "
            "hiragana particle outside the slot."
        ),
    )


def _example_messages(dataset: PocDataset, memory: dict[str, Any]) -> list[dict[str, str]]:
    examples = [
        {
            "source": str(item.get("source", "")).strip(),
            "target": str(item.get("target", "")).strip(),
        }
        for item in dataset.examples
    ]
    examples.extend(
        {"source": item["source"], "target": item["target"]}
        for item in memory["derived_demonstrations"]
    )
    messages: list[dict[str, str]] = []
    for item in examples:
        if not item["source"] or not item["target"]:
            continue
        messages.extend(
            (
                {
                    "role": "user",
                    "content": f"<MEMORY_EXAMPLE>\n{item['source']}\n</MEMORY_EXAMPLE>",
                },
                {"role": "assistant", "content": item["target"]},
            )
        )
    return messages


def _typed_reference_lines(
    placeholders: dict[str, str],
    placeholder_entry_ids: dict[str, str],
    entries_by_id: dict[str, GlossaryEntry],
) -> list[str]:
    lines = []
    for placeholder, target in placeholders.items():
        entry_id = placeholder_entry_ids[placeholder]
        entry = entries_by_id[entry_id]
        kind, hint = semantic_type(entry)
        lines.append(
            "\n".join(
                (
                    f"- slot: {placeholder}",
                    f"  source_meaning: {entry.source}",
                    f"  restored_target: {target}",
                    f"  semantic_type: {kind}",
                    f"  grammar_hint: {hint}",
                )
            )
        )
    return lines


def build_typed_agent_request(
    dataset: PocDataset,
    memory: dict[str, Any],
    case: PocCase,
    entries: tuple[GlossaryEntry, ...],
) -> dict[str, Any]:
    """Build a compact request whose placeholders retain source meaning and type."""

    matched = match_glossary(case.source, entries)
    protected = protect_glossary_terms(case.source, matched)
    entries_by_id = {entry.id: entry for entry in entries}
    fuzzy = (
        find_fuzzy_candidates(case.source, entries, max_candidates=2)
        if case.category == "ocr_noise"
        else ()
    )
    sections: list[str] = []
    typed_lines = _typed_reference_lines(
        protected.replacements,
        protected.entry_ids,
        entries_by_id,
    )
    if typed_lines:
        sections.append(
            "<TYPED_LOCAL_REFERENCES>\n"
            "Each slot in OCR_TEXT is immutable. Copy every slot exactly once, in the "
            "grammatically correct position, and translate all surrounding content. The "
            "application restores the target after your response. Use source_meaning and "
            "semantic_type to build natural Japanese grammar; never output restored_target "
            "beside its slot.\n"
            + "\n".join(typed_lines)
            + "\n</TYPED_LOCAL_REFERENCES>"
        )
    if fuzzy:
        candidate_lines = []
        for item in fuzzy:
            entry = entries_by_id[item.entry_id]
            kind, hint = semantic_type(entry)
            candidate_lines.append(
                f"- observed `{item.observed}` may mean `{item.source}` -> {item.target}; "
                f"semantic_type={kind}; grammar_hint={hint}"
            )
        sections.append(
            "<OCR_CANDIDATES>\n"
            "These are non-binding. Use a candidate only when the whole sentence makes the "
            "correction unambiguous; otherwise ignore it.\n"
            + "\n".join(candidate_lines)
            + "\n</OCR_CANDIDATES>"
        )
    sections.append(f"<OCR_TEXT>\n{protected.source}\n</OCR_TEXT>")
    system = (
        compile_agent_system(dataset, memory)
        + "\n\n## Runtime output contract\n"
        "Before returning, silently verify that prose contains only hiragana, every space "
        "run contains exactly two ASCII spaces, there is no punctuation, every reference "
        "slot occurs exactly once, and no ordinary word is wrapped in square brackets."
    )
    messages = [
        {"role": "system", "content": system},
        *_example_messages(dataset, memory),
        {"role": "user", "content": "\n\n".join(sections)},
    ]
    return {
        "messages": messages,
        "matched_entry_ids": [entry.id for entry in matched],
        "fuzzy_candidates": [item.__dict__ for item in fuzzy],
        "protected_source": protected.source,
        "placeholders": protected.replacements,
        "placeholder_entry_ids": protected.entry_ids,
        "allowed_bracket_targets": sorted(
            set(protected.replacements.values())
            | {item.target for item in fuzzy}
        ),
        "estimated_input_tokens": estimate_tokens(
            "\n".join(message["content"] for message in messages)
        ),
    }


def normalize_translation(
    raw_response: str,
    placeholders: dict[str, str],
    *,
    allowed_bracket_targets: Iterable[str],
) -> dict[str, Any]:
    """Apply only deterministic presentation fixes that preserve lexical content."""

    text = raw_response.strip()
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        last_fence = text.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            text = text[first_newline + 1:last_fence].strip()
    restored = restore_placeholders(text, placeholders)
    allowed = {str(item) for item in allowed_bracket_targets if str(item)}
    unauthorized: list[str] = []

    def unwrap(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in allowed:
            return token
        unauthorized.append(token)
        return token[1:-1]

    unwrapped = _BRACKET_PATTERN.sub(unwrap, restored)
    punctuation = _PUNCTUATION_PATTERN.findall(unwrapped)
    without_punctuation = _PUNCTUATION_PATTERN.sub(" ", unwrapped)
    whitespace_runs = re.findall(r"\s+", without_punctuation)
    normalized = re.sub(r"\s+", "  ", without_punctuation).strip()
    return {
        "translation": normalized,
        "restored_before_normalization": restored,
        "changed": normalized != restored,
        "unauthorized_brackets_unwrapped": unauthorized,
        "punctuation_removed": punctuation,
        "whitespace_runs_changed": sum(run != "  " for run in whitespace_runs),
    }


def evaluate_strict_constraints(
    dataset: PocDataset,
    case: PocCase,
    translation: str,
    *,
    allowed_bracket_targets: Iterable[str],
) -> dict[str, Any]:
    """Extend the global hard evaluator with balanced, authorized brackets."""

    base = evaluate_hard_constraints(dataset, case, translation)
    allowed = {str(item) for item in allowed_bracket_targets if str(item)}
    bracket_tokens = _BRACKET_PATTERN.findall(translation)
    balanced = (
        translation.count("[") == len(bracket_tokens)
        and translation.count("]") == len(bracket_tokens)
    )
    unauthorized = [token for token in bracket_tokens if token not in allowed]
    checks = dict(base["checks"])
    checks["balanced_brackets"] = balanced
    checks["authorized_brackets_only"] = not unauthorized
    return {
        **base,
        "ok": all(checks.values()),
        "checks": checks,
        "allowed_bracket_targets": sorted(allowed),
        "unauthorized_brackets": unauthorized,
    }


def _allowed_targets_for_case(
    case: PocCase,
    entries: tuple[GlossaryEntry, ...],
) -> list[str]:
    targets = {entry.target for entry in match_glossary(case.source, entries)}
    if case.category == "ocr_noise":
        targets.update(
            item.target for item in find_fuzzy_candidates(case.source, entries, max_candidates=2)
        )
    return sorted(targets)


def load_v6_evidence(
    path: Path,
    *,
    dataset_sha256: str,
    cases: tuple[PocCase, ...],
    repetitions: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = next((row for row in rows if row.get("type") == "run"), None)
    if not isinstance(metadata, dict):
        raise CanaryError("POC-6 evidence has no run metadata")
    if metadata.get("experiment") != "compiled_persistent_agent_memory":
        raise CanaryError("baseline is not a POC-6 Agent-memory run")
    if metadata.get("dataset_sha256") != dataset_sha256:
        raise CanaryError("POC-6 evidence uses a different dataset")
    if metadata.get("repetitions") != repetitions:
        raise CanaryError("POC-6 evidence repetition count does not match")

    case_ids = {case.id for case in cases}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for group in ("DIRECT_BASELINE", "AGENT_MEMORY"):
        selected = [
            row
            for row in rows
            if row.get("type") == "result"
            and row.get("group") == group
            and row.get("case_id") in case_ids
        ]
        expected = len(cases) * repetitions
        if len(selected) != expected:
            raise CanaryError(f"{group} needs {expected} records, found {len(selected)}")
        identities: set[tuple[str, int]] = set()
        for record in selected:
            identity = (str(record.get("case_id")), int(record.get("repetition", 0)))
            if identity in identities:
                raise CanaryError(f"{group} has duplicate record {identity}")
            identities.add(identity)
            response = record.get("response")
            if record.get("error") or not isinstance(response, str) or not response.strip():
                raise CanaryError(f"{group} record {identity} is incomplete")
        grouped[group] = selected
    return metadata, grouped


def _blind_id(run_id: str, group: str, case_id: str, repetition: int) -> str:
    return hashlib.sha256(
        f"{run_id}:{group}:{case_id}:{repetition}".encode("utf-8")
    ).hexdigest()[:12]


def prepare_reused_records(
    evidence: dict[str, list[dict[str, Any]]],
    *,
    dataset: PocDataset,
    cases: tuple[PocCase, ...],
    entries: tuple[GlossaryEntry, ...],
    run_id: str,
    baseline_path: Path,
) -> list[dict[str, Any]]:
    by_case = {case.id: case for case in cases}
    order = {case.id: index for index, case in enumerate(cases)}
    records: list[dict[str, Any]] = []
    for source_group, target_group in (
        ("DIRECT_BASELINE", "DIRECT_BASELINE"),
        ("AGENT_MEMORY", "AGENT_V6_NORMALIZED"),
    ):
        selected = sorted(
            evidence[source_group],
            key=lambda row: (int(row["repetition"]), order[row["case_id"]]),
        )
        for source in selected:
            case = by_case[source["case_id"]]
            restored = (source.get("evaluation") or {}).get(
                "restored_response", source["response"]
            )
            allowed = _allowed_targets_for_case(case, entries)
            if target_group == "AGENT_V6_NORMALIZED":
                formatting = normalize_translation(
                    source["response"],
                    source.get("placeholders") or {},
                    allowed_bracket_targets=allowed,
                )
                translation = formatting["translation"]
            else:
                formatting = {
                    "translation": restored,
                    "restored_before_normalization": restored,
                    "changed": False,
                    "unauthorized_brackets_unwrapped": [],
                    "punctuation_removed": [],
                    "whitespace_runs_changed": 0,
                }
                translation = restored
            evaluation = evaluate_response(dataset, case, entries, translation, {})
            record = dict(source)
            record.update(
                {
                    "run_id": run_id,
                    "group": target_group,
                    "blind_id": _blind_id(
                        run_id, target_group, case.id, int(source["repetition"])
                    ),
                    "baseline_original_blind_id": source.get("blind_id"),
                    "baseline_source": str(baseline_path.resolve()),
                    "final_translation": translation,
                    "normalization": formatting,
                    "evaluation": evaluation,
                    "hard_constraint_evaluation": evaluate_strict_constraints(
                        dataset,
                        case,
                        translation,
                        allowed_bracket_targets=allowed,
                    ),
                }
            )
            records.append(record)
    for index, record in enumerate(records, 1):
        record["request_index"] = index
    return records


def summarize(records: list[dict[str, Any]], *, repetitions: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [record for record in records if record["group"] == group]
        completed = [record for record in selected if record.get("final_translation")]
        latencies = [float(record["latency_s"]) for record in completed]
        comparable = 0
        stable = 0
        for case_id in sorted({record["case_id"] for record in completed}):
            outputs = [
                record["final_translation"]
                for record in completed
                if record["case_id"] == case_id
            ]
            if len(outputs) == repetitions:
                comparable += 1
                stable += len(set(outputs)) == 1
        hard_pass = sum(
            bool(record["hard_constraint_evaluation"]["ok"]) for record in completed
        )
        groups[group] = {
            "requests": len(selected),
            "completed": len(completed),
            "errors": sum(bool(record.get("error")) for record in selected),
            "strict_hard_pass": hard_pass,
            "strict_hard_pass_rate": round(hard_pass / len(completed), 3)
            if completed
            else None,
            "missing_expected_terms": sum(
                len(record["hard_constraint_evaluation"]["missing_expected_terms"])
                for record in completed
            ),
            "normalization_changed": sum(
                bool(record.get("normalization", {}).get("changed"))
                for record in completed
            ),
            "provider_prompt_tokens": sum(
                int(record.get("usage", {}).get("prompt_tokens", 0) or 0)
                for record in completed
            ),
            "estimated_input_tokens_mean": round(
                statistics.fmean(record["estimated_input_tokens"] for record in selected),
                1,
            )
            if selected
            else None,
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "exact_repeat_stability": f"{stable}/{comparable}" if comparable else None,
            "semantic_fidelity": "NOT_SCORED_USE_BLIND_REVIEW",
            "constraint_aware_naturalness": "NOT_SCORED_USE_BLIND_REVIEW",
        }
    return {"type": "summary", "repetitions": repetitions, "groups": groups}


def write_blind_review(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
    rows = []
    for record in records:
        if not record.get("final_translation"):
            continue
        rows.append(
            {
                "blind_id": record["blind_id"],
                "case_id": record["case_id"],
                "category": record["category"],
                "source": record["source"],
                "translation": record["final_translation"],
                "confirmed_user_constraints": (
                    "Judge naturalness under the required all-hiragana, authorized-term "
                    "brackets, and exact-double-space format. Do not reward breaking a hard rule."
                ),
                "expected_fixed_outputs": record.get("expected_fixed_outputs", []),
                "semantic_fidelity_0_to_5": None,
                "naturalness_0_to_5": None,
                "meaning_error": "",
                "review_notes": "",
            }
        )
    random.Random(seed ^ 0x61CA).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_paths() -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("logs") / "poc"
    return (
        root / f"agent-memory-canary-{stamp}.jsonl",
        root / f"agent-memory-canary-blind-{stamp}.jsonl",
    )


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path, strict=True)
    entries = dataset.glossary_sets["200"]
    cases = select_canary_cases(dataset)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    memory = load_agent_memory(
        Path(args.memory),
        dataset_sha256=dataset_sha256,
        forbidden_example_sources=(case.source for case in dataset.cases),
    )
    baseline_path = Path(args.baseline)
    baseline_metadata, evidence = load_v6_evidence(
        baseline_path,
        dataset_sha256=dataset_sha256,
        cases=cases,
        repetitions=args.repetitions,
    )
    output_default, blind_default = _default_paths()
    output_path = Path(args.output) if args.output else output_default
    blind_path = Path(args.blind_output) if args.blind_output else blind_default
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blind_path.parent.mkdir(parents=True, exist_ok=True)

    settings = AppSettings.load()
    fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
    baseline_model = str(baseline_metadata.get("fast_model", "")).strip()
    if baseline_model and fast_model != baseline_model:
        raise CanaryError(
            f"fast model must match baseline ({baseline_model}), got {fast_model}"
        )
    if not args.dry_run and (
        not settings.ai.base_url or not settings.ai.api_key or not fast_model
    ):
        raise SystemExit("API base URL, key, and fast model must be configured")

    run_id = datetime.now(timezone.utc).isoformat()
    records = prepare_reused_records(
        evidence,
        dataset=dataset,
        cases=cases,
        entries=entries,
        run_id=run_id,
        baseline_path=baseline_path,
    )
    metadata = {
        "type": "run",
        "run_id": run_id,
        "experiment": "agent_memory_typed_reference_canary",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "baseline": str(baseline_path.resolve()),
        "memory": str(Path(args.memory).resolve()),
        "fast_model": fast_model,
        "translation_thinking": "disabled",
        "groups": GROUPS,
        "case_ids": CANARY_CASE_IDS,
        "repetitions": args.repetitions,
        "new_flash_api_requests": len(cases) * args.repetitions,
        "dry_run": args.dry_run,
        "success_gate": {
            "typed_strict_hard_pass_min": 13,
            "typed_missing_expected_terms_max": 0,
            "typed_latency_p50_s_max": 3.0,
            "critical_cases_better_than_v6": ["terminology_04", "ocr_noise_04"],
            "blind_semantic_and_naturalness_must_beat_v6_normalized": True,
        },
    }
    rng = random.Random(args.seed)
    entries_by_id = {entry.id: entry for entry in entries}
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        request_index = len(records)
        new_request_count = 0
        for repetition in range(1, args.repetitions + 1):
            ordered_cases = list(cases)
            rng.shuffle(ordered_cases)
            for case in ordered_cases:
                request_index += 1
                new_request_count += 1
                built = build_typed_agent_request(dataset, memory, case, entries)
                payload = {
                    "model": fast_model,
                    "messages": built["messages"],
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "thinking": {"type": "disabled"},
                }
                record: dict[str, Any] = {
                    "type": "result",
                    "run_id": run_id,
                    "request_index": request_index,
                    "blind_id": _blind_id(
                        run_id, "AGENT_V61_TYPED", case.id, repetition
                    ),
                    "repetition": repetition,
                    "group": "AGENT_V61_TYPED",
                    "case_id": case.id,
                    "category": case.category,
                    "source": case.source,
                    "expected_fixed_outputs": [
                        entries_by_id[entry_id].target
                        for entry_id in case.expected_entry_ids
                        if entry_id in entries_by_id
                    ],
                    "matched_entry_ids": built["matched_entry_ids"],
                    "fuzzy_candidates": built["fuzzy_candidates"],
                    "protected_source": built["protected_source"],
                    "placeholders": built["placeholders"],
                    "placeholder_entry_ids": built["placeholder_entry_ids"],
                    "allowed_bracket_targets": built["allowed_bracket_targets"],
                    "estimated_input_tokens": built["estimated_input_tokens"],
                    "payload": payload,
                    "response": None,
                    "final_translation": None,
                    "normalization": None,
                    "usage": {},
                    "latency_s": None,
                    "error": None,
                    "raw_evaluation": None,
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
                        raw_evaluation = evaluate_response(
                            dataset,
                            case,
                            entries,
                            response,
                            built["placeholders"],
                        )
                        formatting = normalize_translation(
                            response,
                            built["placeholders"],
                            allowed_bracket_targets=built["allowed_bracket_targets"],
                        )
                        translation = formatting["translation"]
                        record["response"] = response
                        record["final_translation"] = translation
                        record["normalization"] = formatting
                        record["usage"] = usage
                        record["raw_evaluation"] = raw_evaluation
                        record["evaluation"] = evaluate_response(
                            dataset, case, entries, translation, {}
                        )
                        record["hard_constraint_evaluation"] = evaluate_strict_constraints(
                            dataset,
                            case,
                            translation,
                            allowed_bracket_targets=built["allowed_bracket_targets"],
                        )
                    except TranslationError as exc:
                        record["error"] = str(exc)
                    record["latency_s"] = round(time.perf_counter() - started, 3)
                    print(
                        f"{new_request_count:02d}/{len(cases) * args.repetitions} "
                        f"repetition={repetition} case={case.id} "
                        f"elapsed={record['latency_s']:.3f}s ok={record['error'] is None}"
                    )
                    if args.delay:
                        time.sleep(args.delay)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

        summary = summarize(records, repetitions=args.repetitions)
        summary["run_id"] = run_id
        summary["api_calls_this_run"] = {
            "pro": 0,
            "flash": 0 if args.dry_run else len(cases) * args.repetitions,
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    write_blind_review(blind_path, records, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Blind review: {blind_path.resolve()}")
    return output_path, blind_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="completed POC-6 result JSONL")
    parser.add_argument("--memory", required=True, help="persisted POC-6 Agent memory JSON")
    parser.add_argument("--dataset", default="poc_data/reference_poc_dataset.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--translation-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions != 2:
        raise SystemExit("formal canary requires exactly 2 repetitions")
    try:
        run(args)
    except (AgentMemoryError, CanaryError, json.JSONDecodeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
