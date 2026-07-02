"""POC 4: Pro bootstrap state -> persistent session -> Flash translation.

This experiment tests the project's central agent hypothesis without changing
the desktop app.  Pro is called once to turn the complete prompt/reference
package into explicit session state.  All per-case translations are then made
by the configured Flash model with thinking disabled.

The three Flash groups are:

* CURRENT_FULL: current-style full 200-entry reference in the system prompt.
* RETRIEVAL_RAW: deterministic relevant-reference injection without Pro state.
* AGENT_BOOTSTRAP: the same retrieval path plus persisted Pro-understood state.

Machine checks are reported only as constraint/terminology checks.  Semantic
fidelity and naturalness are exported for blind human/Pro review and are never
presented as automatic translation-quality scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.settings import AppSettings
from app.translation.client import TranslationError
from poc_reference_injection import (
    GlossaryEntry,
    PocCase,
    PocDataset,
    _percentile,
    _send_request,
    build_request,
    build_stable_system,
    estimate_tokens,
    evaluate_response,
    load_dataset,
    match_glossary,
    protect_glossary_terms,
)


GROUPS = ("CURRENT_FULL", "RETRIEVAL_RAW", "AGENT_BOOTSTRAP")

_BOOTSTRAP_SYSTEM = """You are the startup planning phase of a persistent translation agent.
Read the complete user prompt package and convert your understanding into explicit JSON state for a cheaper Flash model.
The Flash model cannot see your hidden reasoning, so every useful conclusion must be represented in the JSON fields.
Preserve confirmed user requirements exactly. Do not weaken, silently replace, or invent requirements.
Do not translate test sentences and do not include chain-of-thought or commentary.
The glossary itself is retrieved deterministically at runtime; summarize how it should be used rather than copying all entries.

The confirmed rules are already injected separately at runtime. Do not restate
or paraphrase them. Return only additional understanding that helps Flash apply
those rules to meaning, style, OCR uncertainty, and known failure modes.

Return one JSON object with exactly this shape:
{
  "version": 2,
  "intent_summary": "concise description of the user's real translation goal",
  "semantic_priorities": ["meaning distinction Flash must preserve"],
  "style_guidance": ["non-redundant naturalness or tone insight"],
  "ocr_guidance": ["how to use uncertain OCR candidates without inventing context"],
  "risk_notes": ["known ambiguity or failure mode Flash should avoid"]
}
"""

_LIST_FIELDS = (
    "semantic_priorities",
    "style_guidance",
    "ocr_guidance",
    "risk_notes",
)


class BootstrapStateError(ValueError):
    """Raised when Pro does not return usable persistent state."""


def _entry_json(entry: GlossaryEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "source": entry.source,
        "target": entry.target,
        "aliases": list(entry.aliases),
        "format": entry.format,
        "risk": entry.risk,
    }


def build_bootstrap_source(
    dataset: PocDataset,
    entries: tuple[GlossaryEntry, ...],
) -> str:
    """Build the complete startup package. Test cases are intentionally absent."""

    package = {
        "fixed_template": dataset.base_system_prompt,
        "raw_user_constraints": dataset.raw_user_constraints,
        "user_confirmed_rules": list(dataset.global_rules),
        "style_rules": list(dataset.style_rules),
        "forbidden_outputs": list(dataset.forbidden_outputs),
        "confirmed_examples": list(dataset.examples),
        "reference_glossary": [_entry_json(entry) for entry in entries],
        "confirmed_corrections": [],
        "runtime_model": "Flash with thinking disabled",
    }
    return (
        "Compile this source package into persistent translation-agent state. "
        "The state will be stored locally and reused across application restarts.\n\n"
        + json.dumps(package, ensure_ascii=False, indent=2)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a single JSON object, tolerating an accidental Markdown fence."""

    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            stripped = stripped[first_newline + 1:last_fence].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise BootstrapStateError("Pro response does not contain a JSON object")
    try:
        value = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise BootstrapStateError(f"Pro state is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BootstrapStateError("Pro state must be a JSON object")
    return value


def _clean_string_list(value: Any, field: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise BootstrapStateError(f"{field} must be a list")
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    if required and not cleaned:
        raise BootstrapStateError(f"{field} must not be empty")
    if len(cleaned) > 40:
        raise BootstrapStateError(f"{field} has too many entries ({len(cleaned)})")
    return cleaned


def validate_bootstrap_state(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize Pro output before it becomes runtime state."""

    if raw.get("version") != 2:
        raise BootstrapStateError("version must be 2")
    intent = str(raw.get("intent_summary", "")).strip()
    if not intent:
        raise BootstrapStateError("intent_summary must not be empty")
    state: dict[str, Any] = {"version": 2, "intent_summary": intent}
    for field in _LIST_FIELDS:
        state[field] = _clean_string_list(
            raw.get(field, []),
            field,
            required=field in {"semantic_priorities"},
        )

    serialized = json.dumps(state, ensure_ascii=False)
    if len(serialized) > 30_000:
        raise BootstrapStateError("compiled state exceeds 30,000 characters")
    return state


def make_dry_run_state(dataset: PocDataset) -> dict[str, Any]:
    """Create deterministic fixture state for offline request-generation tests."""

    return validate_bootstrap_state(
        {
            "version": 2,
            "intent_summary": (
                "Translate OCR-derived Chinese into faithful, natural Japanese while "
                "obeying the user's confirmed script, glossary, and spacing rules."
            ),
            "semantic_priorities": [
                "Preserve the complete source meaning",
                "Preserve negation, conditions, quantities, and speaker intent",
            ],
            "style_guidance": [
                "Use natural Japanese word order instead of mirroring Chinese syntax",
            ],
            "ocr_guidance": [
                "Correct an OCR error only when nearby context makes the intended text clear.",
                "Do not invent missing context when an OCR correction is uncertain.",
            ],
            "risk_notes": [
                "Do not output only glossary terms; translate all surrounding content.",
                "Treat OCR candidates as hints, not mandatory corrections.",
            ],
        }
    )


def _section(title: str, values: Iterable[str]) -> str:
    rows = [str(value).strip() for value in values if str(value).strip()]
    if not rows:
        return ""
    return title + "\n" + "\n".join(f"{index}. {row}" for index, row in enumerate(rows, 1))


def compile_agent_base(dataset: PocDataset, state: dict[str, Any]) -> str:
    """Compile stable state into the system prefix consumed by Flash."""

    parts = [dataset.base_system_prompt]
    original_sections = (
        _section("## Confirmed user rules", dataset.global_rules),
        _section("## Confirmed style rules", dataset.style_rules),
        _section("## Confirmed forbidden outputs", dataset.forbidden_outputs),
    )
    parts.extend(section for section in original_sections if section)
    parts.append("## Pro-initialized persistent session")
    parts.append("User intent:\n" + state["intent_summary"])
    for title, field in (
        ("Semantic priorities", "semantic_priorities"),
        ("Style understanding", "style_guidance"),
        ("OCR understanding", "ocr_guidance"),
        ("Known risks", "risk_notes"),
    ):
        section = _section(f"### {title}", state[field])
        if section:
            parts.append(section)
    parts.append(
        "This session state was produced once during startup. Apply it to every "
        "translation without explaining or restating it."
    )
    return "\n\n".join(parts)


def make_agent_dataset(dataset: PocDataset, state: dict[str, Any]) -> PocDataset:
    return replace(
        dataset,
        base_system_prompt=compile_agent_base(dataset, state),
        global_rules=(),
        style_rules=(),
        forbidden_outputs=(),
        examples=(),
    )


@dataclass(frozen=True)
class FuzzyCandidate:
    """One non-binding glossary candidate for OCR-damaged source text."""

    entry_id: str
    observed: str
    source: str
    target: str
    distance: int


def _is_cjk_text(text: str) -> bool:
    return bool(text) and all("\u4e00" <= char <= "\u9fff" for char in text)


def _edit_distance(left: str, right: str, *, maximum: int = 1) -> int:
    """Return Levenshtein distance, stopping rows that cannot meet maximum."""

    if abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        current = [index]
        for column, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        if min(current) > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def _exact_spans(source: str, entries: Iterable[GlossaryEntry]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for entry in entries:
        for surface in entry.surfaces:
            start = source.find(surface)
            while start >= 0:
                spans.append((start, start + len(surface)))
                start = source.find(surface, start + 1)
    return spans


def find_fuzzy_candidates(
    source: str,
    entries: tuple[GlossaryEntry, ...],
    *,
    max_candidates: int = 6,
) -> tuple[FuzzyCandidate, ...]:
    """Find edit-distance-1 CJK candidates outside already exact-matched spans."""

    exact = match_glossary(source, entries)
    exact_ids = {entry.id for entry in exact}
    spans = _exact_spans(source, exact)
    candidates: list[tuple[int, int, int, FuzzyCandidate]] = []
    for entry in entries:
        if entry.id in exact_ids:
            continue
        best: tuple[int, int, str] | None = None
        for surface in entry.surfaces:
            if len(surface) < 2 or not _is_cjk_text(surface):
                continue
            for window_length in {len(surface) - 1, len(surface), len(surface) + 1}:
                if window_length < 2:
                    continue
                for start in range(0, len(source) - window_length + 1):
                    end = start + window_length
                    if any(start < span_end and end > span_start for span_start, span_end in spans):
                        continue
                    observed = source[start:end]
                    if not _is_cjk_text(observed):
                        continue
                    distance = _edit_distance(observed, surface, maximum=1)
                    if distance != 1:
                        continue
                    choice = (distance, start, observed)
                    if best is None or choice < best:
                        best = choice
        if best is not None:
            distance, start, observed = best
            candidate = FuzzyCandidate(entry.id, observed, entry.source, entry.target, distance)
            candidates.append((distance, -len(entry.source), start, candidate))
    candidates.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in candidates[:max_candidates])


def build_safe_retrieval_request(
    dataset: PocDataset,
    case: PocCase,
    entries: tuple[GlossaryEntry, ...],
) -> dict[str, Any]:
    """Build a no-double-instruction placeholder request plus optional OCR hints."""

    matched = match_glossary(case.source, entries)
    plan = protect_glossary_terms(case.source, matched)
    # The case category stands in for an upstream OCR-uncertainty signal in
    # this controlled POC. A production path should use OCR confidence/change
    # metadata and must not run edit-distance retrieval on every clean input.
    fuzzy = (
        find_fuzzy_candidates(case.source, entries, max_candidates=2)
        if case.category == "ocr_noise"
        else ()
    )
    protocol = (
        "<PLACEHOLDER_PROTOCOL>\n"
        "Each token shaped like ⟦REF_n⟧ already represents one fixed term that the local "
        "application restores after your response. Copy every placeholder exactly once. "
        "Never translate, expand, duplicate, omit, or explain a placeholder. Do not output "
        "a glossary target next to it. Translate all surrounding text naturally.\n"
        "Protocol example: OCR `打开⟦REF_0⟧` -> output `⟦REF_0⟧を  ひらく`.\n"
        "</PLACEHOLDER_PROTOCOL>"
    )
    sections = [protocol] if plan.replacements else []
    if fuzzy:
        lines = [
            (
                f"- observed `{item.observed}` may be `{item.source}` -> {item.target} "
                f"(edit distance {item.distance})"
            )
            for item in fuzzy
        ]
        sections.append(
            "<OCR_CANDIDATES>\n"
            "These are non-binding hints. Use a candidate only when sentence context makes "
            "the correction clear; otherwise ignore it.\n"
            + "\n".join(lines)
            + "\n</OCR_CANDIDATES>"
        )
    ocr_block = f"<OCR_TEXT>\n{plan.source}\n</OCR_TEXT>"
    user = "\n\n".join((*sections, ocr_block))
    system = build_stable_system(dataset)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "matched_entry_ids": [entry.id for entry in matched],
        "fuzzy_candidates": [item.__dict__ for item in fuzzy],
        "protected_source": plan.source,
        "placeholders": plan.replacements,
        "placeholder_entry_ids": plan.entry_ids,
        "estimated_input_tokens": estimate_tokens(system + "\n" + user),
    }


def build_group_request(
    dataset: PocDataset,
    agent_dataset: PocDataset,
    case: PocCase,
    entries: tuple[GlossaryEntry, ...],
    group: str,
) -> dict[str, Any]:
    if group == "CURRENT_FULL":
        return build_request(dataset, case, entries, "A")
    if group == "RETRIEVAL_RAW":
        return build_safe_retrieval_request(dataset, case, entries)
    if group == "AGENT_BOOTSTRAP":
        return build_safe_retrieval_request(agent_dataset, case, entries)
    raise ValueError(f"unknown group: {group}")


def _load_state(path: Path, *, expected_dataset_sha256: str = "") -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "state" in raw:
        metadata = raw.get("metadata", {})
        state_hash = str(metadata.get("dataset_sha256", "")).strip()
        if expected_dataset_sha256 and state_hash and state_hash != expected_dataset_sha256:
            raise BootstrapStateError(
                "persisted state was compiled from a different dataset; run Pro bootstrap again"
            )
        raw = raw["state"]
    if not isinstance(raw, dict):
        raise BootstrapStateError("state file must contain an object")
    return validate_bootstrap_state(raw)


def _save_state(
    path: Path,
    state: dict[str, Any],
    *,
    dataset_sha256: str,
    thinking_model: str,
    source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": dataset_sha256,
            "thinking_model": thinking_model,
            "source": source,
        },
        "state": state,
    }
    path.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("logs") / "poc"
    return (
        root / f"agent-bootstrap-{stamp}.jsonl",
        root / f"agent-bootstrap-state-{stamp}.json",
        root / f"agent-bootstrap-blind-{stamp}.jsonl",
    )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [record for record in records if record["group"] == group]
        completed = [record for record in selected if record.get("response") is not None]
        evaluations = [record["evaluation"] for record in completed]
        latencies = [float(record["latency_s"]) for record in completed]
        groups[group] = {
            "requests": len(selected),
            "completed": len(completed),
            "errors": sum(bool(record.get("error")) for record in selected),
            "machine_constraint_pass": sum(item["machine_ok"] for item in evaluations),
            "missing_expected_terms": sum(
                len(item["missing_expected_terms"]) for item in evaluations
            ),
            "provider_prompt_tokens": sum(
                int(record.get("usage", {}).get("prompt_tokens", 0) or 0)
                for record in completed
            ),
            "estimated_input_tokens_mean": round(
                statistics.fmean(record["estimated_input_tokens"] for record in selected),
                1,
            ) if selected else None,
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "semantic_fidelity": "NOT_SCORED_USE_BLIND_REVIEW",
            "naturalness": "NOT_SCORED_USE_BLIND_REVIEW",
        }
    return {"type": "summary", "groups": groups}


def _write_blind_review(
    path: Path,
    records: list[dict[str, Any]],
    *,
    seed: int,
) -> None:
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
                "translation": record["evaluation"]["restored_response"],
                "semantic_fidelity_0_to_5": None,
                "naturalness_0_to_5": None,
                "meaning_error": "",
                "review_notes": "",
            }
        )
    random.Random(seed ^ 0xA63E).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path, strict=True)
    entries = dataset.glossary_sets["200"]
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    output_default, state_default, blind_default = _default_paths()
    output_path = Path(args.output) if args.output else output_default
    state_output = Path(args.state_output) if args.state_output else state_default
    blind_output = Path(args.blind_output) if args.blind_output else blind_default
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

    bootstrap_messages = [
        {"role": "system", "content": _BOOTSTRAP_SYSTEM},
        {"role": "user", "content": build_bootstrap_source(dataset, entries)},
    ]
    bootstrap_payload = {
        "model": thinking_model,
        "messages": bootstrap_messages,
        "temperature": 0.0,
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }
    bootstrap_record: dict[str, Any] = {
        "type": "bootstrap",
        "payload": bootstrap_payload,
        "response": None,
        "usage": {},
        "latency_s": None,
        "state_source": "",
    }

    if args.state:
        state = _load_state(
            Path(args.state),
            expected_dataset_sha256=dataset_sha256,
        )
        bootstrap_record["state_source"] = str(Path(args.state).resolve())
    elif args.dry_run:
        state = make_dry_run_state(dataset)
        bootstrap_record["state_source"] = "deterministic_dry_run_fixture"
    else:
        started = time.perf_counter()
        response, usage = _send_request(
            base_url=settings.ai.base_url,
            api_key=settings.ai.api_key,
            payload=bootstrap_payload,
            timeout_seconds=args.bootstrap_timeout,
        )
        bootstrap_record["latency_s"] = round(time.perf_counter() - started, 3)
        bootstrap_record["response"] = response
        bootstrap_record["usage"] = usage
        bootstrap_record["state_source"] = "thinking_model_api"
        state = validate_bootstrap_state(extract_json_object(response))

    bootstrap_record["state"] = state
    _save_state(
        state_output,
        state,
        dataset_sha256=dataset_sha256,
        thinking_model=thinking_model,
        source=bootstrap_record["state_source"],
    )

    if args.bootstrap_only:
        print(f"Bootstrap state: {state_output.resolve()}")
        return output_path, state_output, blind_output

    agent_dataset = make_agent_dataset(dataset, state)
    rng = random.Random(args.seed)
    run_id = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    metadata = {
        "type": "run",
        "run_id": run_id,
        "experiment": "agent_bootstrap_handoff",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "dataset_status": dataset.status,
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "translation_thinking": "disabled",
        "glossary_scale": 200,
        "groups": GROUPS,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "semantic_quality": "not automatically scored",
    }

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        handle.write(json.dumps(bootstrap_record, ensure_ascii=False) + "\n")
        request_index = 0
        cases = dataset.cases[: args.limit or None]
        for case in cases:
            ordered_groups = list(GROUPS)
            rng.shuffle(ordered_groups)
            for group in ordered_groups:
                request_index += 1
                built = build_group_request(dataset, agent_dataset, case, entries, group)
                payload = {
                    "model": fast_model,
                    "messages": built["messages"],
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
                    "group": group,
                    "case_id": case.id,
                    "category": case.category,
                    "source": case.source,
                    "matched_entry_ids": built["matched_entry_ids"],
                    "fuzzy_candidates": built.get("fuzzy_candidates", []),
                    "protected_source": built["protected_source"],
                    "placeholders": built["placeholders"],
                    "estimated_input_tokens": built["estimated_input_tokens"],
                    "payload": payload,
                    "response": None,
                    "usage": {},
                    "latency_s": None,
                    "error": None,
                    "evaluation": None,
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
                            dataset,
                            case,
                            entries,
                            response,
                            built["placeholders"],
                        )
                    except TranslationError as exc:
                        record["error"] = str(exc)
                    record["latency_s"] = round(time.perf_counter() - started, 3)
                    print(
                        f"{request_index:03d} group={group} case={case.id} "
                        f"elapsed={record['latency_s']:.3f}s ok={record['error'] is None}"
                    )
                    if args.delay:
                        time.sleep(args.delay)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

        summary = summarize(records)
        summary["run_id"] = run_id
        summary["bootstrap"] = {
            "state_source": bootstrap_record["state_source"],
            "latency_s": bootstrap_record["latency_s"],
            "usage": bootstrap_record["usage"],
            "state_output": str(state_output.resolve()),
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    _write_blind_review(blind_output, records, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Bootstrap state: {state_output.resolve()}")
    print(f"Blind review: {blind_output.resolve()}")
    return output_path, state_output, blind_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="poc_data/reference_poc_dataset.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--state-output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--state", default="", help="reuse an existing persisted state; skip Pro")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--thinking-model", default="")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--limit", type=int, default=0, help="debug only; formal run uses all cases")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--bootstrap-timeout", type=float, default=180.0)
    parser.add_argument("--translation-timeout", type=float, default=60.0)
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (BootstrapStateError, json.JSONDecodeError, OSError, TranslationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
