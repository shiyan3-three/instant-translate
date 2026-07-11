"""POC 6: compiled persistent Agent memory -> fast Flash translation.

This experiment reuses the completed DIRECT records from POC 5 and sends only
one new group to the API.  Pro runs once at startup and compiles an operational
memory containing a semantic playbook, execution plan, failure guards, and
derived practice examples.  Flash receives that memory as a stable system
prefix plus visible example turns, then translates with thinking disabled.

The full 200-entry glossary is read by Pro once.  At translation time the local
application injects only exact matches and conservative OCR candidates.  No
hidden ``reasoning_content`` is persisted or replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.settings import AppSettings
from app.translation.client import TranslationError
from poc.poc_agent_bootstrap import find_fuzzy_candidates
from poc.poc_reference_injection import (
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
)
from poc.poc_session_replay import evaluate_hard_constraints


GROUPS = ("DIRECT_BASELINE", "AGENT_MEMORY")
MEMORY_VERSION = 3
_SEMANTIC_KEYS = (
    "negation_and_scope",
    "conditions_and_consequences",
    "quantities_and_frequency",
    "requests_and_tone",
    "software_relations",
)
_MEMORY_FIELDS = {
    "version",
    "mission",
    "priority_stack",
    "execution_plan",
    "semantic_playbook",
    "format_plan",
    "ocr_plan",
    "failure_guards",
    "derived_demonstrations",
}

_MEMORY_COMPILER_SYSTEM = """You are the startup thinking phase of a persistent translation agent.
Read the complete user rules, approved examples, and reference glossary. Think privately, then compile an operational memory for a cheaper Flash model.

Do not merely summarize or restate the rules. Convert them into an executable translation procedure that helps Flash preserve meaning, scope, conditions, quantities, tone, and software relationships while still obeying the exact output format.

The full glossary is indexed locally and is not copied into memory. You may create 3 to 6 short practice demonstrations that do not copy any supplied example. Their targets must obey the user's output contract: only hiragana, square brackets, and ASCII spaces; every existing space run must contain exactly two spaces; no leading or trailing spaces. Do not invent user preferences or accepted corrections.

Return only one JSON object. Do not include Markdown, commentary, chain-of-thought, or reasoning_content. Use exactly this schema:
{
  "version": 3,
  "mission": "the real translation objective",
  "priority_stack": ["ordered decision priority"],
  "execution_plan": ["concrete step Flash should execute silently"],
  "semantic_playbook": {
    "negation_and_scope": "how to preserve it",
    "conditions_and_consequences": "how to preserve them",
    "quantities_and_frequency": "how to preserve them",
    "requests_and_tone": "how to preserve them",
    "software_relations": "how to preserve them"
  },
  "format_plan": ["concrete formatting step"],
  "ocr_plan": ["conservative OCR decision step"],
  "failure_guards": ["specific failure and prevention"],
  "derived_demonstrations": [
    {"source": "new short Chinese practice input", "target": "valid Japanese output", "lesson": "what this demonstrates"}
  ]
}
"""


class AgentMemoryError(ValueError):
    """Raised when compiled memory or reused baseline evidence is invalid."""


def _entry_json(entry: GlossaryEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "source": entry.source,
        "target": entry.target,
        "aliases": list(entry.aliases),
        "format": entry.format,
        "risk": entry.risk,
    }


def build_memory_source(
    dataset: PocDataset,
    entries: tuple[GlossaryEntry, ...],
) -> str:
    """Build the startup package without exposing any test case."""

    package = {
        "raw_user_constraints": dataset.raw_user_constraints,
        "base_task": dataset.base_system_prompt,
        "confirmed_global_rules": list(dataset.global_rules),
        "confirmed_style_rules": list(dataset.style_rules),
        "forbidden_outputs": list(dataset.forbidden_outputs),
        "approved_examples": list(dataset.examples),
        "accepted_corrections": [],
        "reference_glossary": [_entry_json(entry) for entry in entries],
        "runtime": {
            "model": "Flash",
            "thinking": "disabled",
            "memory_storage": "local persistent session",
            "glossary_delivery": "only locally retrieved entries per OCR request",
        },
    }
    return (
        "Compile the following user-owned translation package into operational "
        "Agent memory. Test sentences are not included.\n\n"
        + json.dumps(package, ensure_ascii=False, indent=2)
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object while tolerating an accidental Markdown fence."""

    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            stripped = stripped[first_newline + 1:last_fence].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise AgentMemoryError("Pro response does not contain a JSON object")
    try:
        value = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise AgentMemoryError(f"Pro memory is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentMemoryError("Pro memory must be a JSON object")
    return value


def _clean_text(value: Any, field: str, *, maximum: int = 2_000) -> str:
    text = str(value).strip()
    if not text:
        raise AgentMemoryError(f"{field} must not be empty")
    if len(text) > maximum:
        raise AgentMemoryError(f"{field} exceeds {maximum} characters")
    return text


def _clean_list(
    value: Any,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list):
        raise AgentMemoryError(f"{field} must be a list")
    rows = [_clean_text(item, field, maximum=1_000) for item in value]
    if not minimum <= len(rows) <= maximum:
        raise AgentMemoryError(
            f"{field} must contain {minimum}-{maximum} items, found {len(rows)}"
        )
    return rows


def _valid_example_target(target: str) -> bool:
    if not re.fullmatch(r"[\u3041-\u3096\[\] ]+", target):
        return False
    if target != target.strip():
        return False
    return all(len(match.group(0)) == 2 for match in re.finditer(r" +", target))


def validate_agent_memory(
    raw: dict[str, Any],
    *,
    forbidden_example_sources: Iterable[str] = (),
) -> dict[str, Any]:
    """Normalize Pro output and reject non-operational or leaked memory."""

    extra = set(raw) - _MEMORY_FIELDS
    missing = _MEMORY_FIELDS - set(raw)
    if extra or missing:
        raise AgentMemoryError(
            f"memory schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if raw.get("version") != MEMORY_VERSION:
        raise AgentMemoryError(f"version must be {MEMORY_VERSION}")

    playbook = raw.get("semantic_playbook")
    if not isinstance(playbook, dict) or set(playbook) != set(_SEMANTIC_KEYS):
        raise AgentMemoryError("semantic_playbook has missing or unknown fields")
    cleaned_playbook = {
        key: _clean_text(playbook[key], f"semantic_playbook.{key}")
        for key in _SEMANTIC_KEYS
    }

    demonstrations = raw.get("derived_demonstrations")
    if not isinstance(demonstrations, list) or not 3 <= len(demonstrations) <= 6:
        raise AgentMemoryError("derived_demonstrations must contain 3-6 items")
    forbidden = {str(item).strip() for item in forbidden_example_sources if str(item).strip()}
    cleaned_demonstrations = []
    seen_sources: set[str] = set()
    for index, item in enumerate(demonstrations, 1):
        if not isinstance(item, dict) or set(item) != {"source", "target", "lesson"}:
            raise AgentMemoryError(f"derived_demonstrations[{index}] has invalid fields")
        source = _clean_text(item["source"], f"demonstration {index} source", maximum=120)
        target = _clean_text(item["target"], f"demonstration {index} target", maximum=240)
        lesson = _clean_text(item["lesson"], f"demonstration {index} lesson", maximum=500)
        if source in forbidden:
            raise AgentMemoryError(f"derived demonstration leaks test source: {source}")
        if source in seen_sources:
            raise AgentMemoryError("derived demonstration sources must be unique")
        if not _valid_example_target(target):
            raise AgentMemoryError(
                f"derived demonstration {index} target violates the output contract"
            )
        seen_sources.add(source)
        cleaned_demonstrations.append(
            {"source": source, "target": target, "lesson": lesson}
        )

    memory = {
        "version": MEMORY_VERSION,
        "mission": _clean_text(raw["mission"], "mission"),
        "priority_stack": _clean_list(
            raw["priority_stack"], "priority_stack", minimum=4, maximum=10
        ),
        "execution_plan": _clean_list(
            raw["execution_plan"], "execution_plan", minimum=5, maximum=14
        ),
        "semantic_playbook": cleaned_playbook,
        "format_plan": _clean_list(
            raw["format_plan"], "format_plan", minimum=3, maximum=10
        ),
        "ocr_plan": _clean_list(raw["ocr_plan"], "ocr_plan", minimum=2, maximum=8),
        "failure_guards": _clean_list(
            raw["failure_guards"], "failure_guards", minimum=4, maximum=14
        ),
        "derived_demonstrations": cleaned_demonstrations,
    }
    if len(json.dumps(memory, ensure_ascii=False)) > 25_000:
        raise AgentMemoryError("compiled Agent memory exceeds 25,000 characters")
    return memory


def make_dry_run_memory(dataset: PocDataset) -> dict[str, Any]:
    """Deterministic operational memory used only for offline payload checks."""

    raw = {
        "version": MEMORY_VERSION,
        "mission": (
            "Preserve the complete Chinese OCR meaning in natural Japanese, then apply "
            "the user's confirmed script, terminology, and spacing contract."
        ),
        "priority_stack": [
            "Preserve propositions, negation, scope, conditions, quantities, and intent.",
            "Choose natural Japanese relationships and word order.",
            "Apply locally supplied terminology exactly once.",
            "Apply the exact output-format contract and return only the translation.",
        ],
        "execution_plan": [
            "Read the entire OCR text before drafting.",
            "Identify predicate, arguments, modifiers, and discourse relationship.",
            "Resolve only OCR damage supported by sentence context.",
            "Draft a complete natural Japanese sentence without dropping information.",
            "Preserve every local reference placeholder exactly once.",
            "Convert prose to the required script and segment it with exact spacing.",
            "Silently verify meaning and formatting before returning the translation.",
        ],
        "semantic_playbook": {
            "negation_and_scope": "Keep the negation attached to the same predicate and preserve prohibitions.",
            "conditions_and_consequences": "Render both the condition and its consequence without reversing them.",
            "quantities_and_frequency": "Preserve every number, unit, frequency, and per-event quantity.",
            "requests_and_tone": "Keep requests polite without turning statements into commands.",
            "software_relations": "Preserve source, destination, ownership, direction, and component relations.",
        },
        "format_plan": [
            "Use only hiragana in prose and retain square brackets around fixed terms.",
            "Put exactly two ASCII spaces between lexical units and nowhere else.",
            "Remove labels, explanations, punctuation, and edge whitespace.",
        ],
        "ocr_plan": [
            "Treat fuzzy candidates as optional evidence rather than mandatory replacements.",
            "Correct damaged OCR only when the whole sentence makes one reading clear.",
        ],
        "failure_guards": [
            "Do not return only matched terms.",
            "Do not omit time, polarity, quantity, condition, or consequence.",
            "Do not turn an opaque placeholder into a guessed term.",
            "Do not add ordinary Japanese punctuation or extra spaces.",
        ],
        "derived_demonstrations": [
            {
                "source": "请不要忘记明天的会议",
                "target": "あしたの  かいぎを  わすれないで  ください",
                "lesson": "Preserve prohibition, time, and object.",
            },
            {
                "source": "如果下雪就留在家里",
                "target": "ゆきが  ふったら  いえに  のこります",
                "lesson": "Preserve condition and consequence.",
            },
            {
                "source": "每天服用两次",
                "target": "まいにち  にかい  のんで  ください",
                "lesson": "Preserve frequency and action.",
            },
            {
                "source": "请在下午关门",
                "target": "ごごに  どあを  しめて  ください",
                "lesson": "Preserve time and polite request tone.",
            },
        ],
    }
    return validate_agent_memory(
        raw,
        forbidden_example_sources=(case.source for case in dataset.cases),
    )


def save_agent_memory(
    path: Path,
    memory: dict[str, Any],
    *,
    dataset_sha256: str,
    thinking_model: str,
    source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "metadata": {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": dataset_sha256,
            "thinking_model": thinking_model,
            "source": source,
            "contains_reasoning_content": False,
        },
        "memory": memory,
    }
    path.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")


def load_agent_memory(
    path: Path,
    *,
    dataset_sha256: str,
    forbidden_example_sources: Iterable[str] = (),
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("metadata"), dict):
        raise AgentMemoryError("memory file must contain metadata and memory")
    metadata = raw["metadata"]
    if metadata.get("dataset_sha256") != dataset_sha256:
        raise AgentMemoryError("Agent memory belongs to a different prompt package")
    if metadata.get("contains_reasoning_content") is not False:
        raise AgentMemoryError("Agent memory must explicitly exclude reasoning_content")
    memory = raw.get("memory")
    if not isinstance(memory, dict):
        raise AgentMemoryError("memory file has no memory object")
    return validate_agent_memory(
        memory,
        forbidden_example_sources=forbidden_example_sources,
    )


def _numbered(title: str, values: Iterable[str]) -> str:
    rows = [str(value).strip() for value in values if str(value).strip()]
    return title + "\n" + "\n".join(
        f"{index}. {value}" for index, value in enumerate(rows, 1)
    )


def compile_agent_system(dataset: PocDataset, memory: dict[str, Any]) -> str:
    """Render stable, auditable memory into the Flash system prefix."""

    parts = [
        dataset.base_system_prompt,
        _numbered("## Confirmed user rules", dataset.global_rules),
        _numbered("## Confirmed style rules", dataset.style_rules),
        _numbered("## Forbidden outputs", dataset.forbidden_outputs),
        "## Persistent Agent memory\nMission: " + memory["mission"],
        _numbered("### Decision priority", memory["priority_stack"]),
        _numbered("### Silent execution plan", memory["execution_plan"]),
        "### Semantic playbook\n"
        + "\n".join(
            f"- {key}: {memory['semantic_playbook'][key]}" for key in _SEMANTIC_KEYS
        ),
        _numbered("### Formatting plan", memory["format_plan"]),
        _numbered("### OCR plan", memory["ocr_plan"]),
        _numbered("### Failure guards", memory["failure_guards"]),
        (
            "The Agent memory was compiled by Pro at session startup. It may guide "
            "execution but never override confirmed user rules or per-request references. "
            "Apply it silently and output only the translation."
        ),
    ]
    return "\n\n".join(part for part in parts if part.strip())


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


def build_agent_request(
    dataset: PocDataset,
    memory: dict[str, Any],
    case: PocCase,
    entries: tuple[GlossaryEntry, ...],
) -> dict[str, Any]:
    """Build one compact Agent-memory request without full-glossary repetition."""

    matched = match_glossary(case.source, entries)
    protected = protect_glossary_terms(case.source, matched)
    fuzzy = (
        find_fuzzy_candidates(case.source, entries, max_candidates=2)
        if case.category == "ocr_noise"
        else ()
    )
    sections: list[str] = []
    if protected.replacements:
        sections.append(
            "<LOCAL_REFERENCE_PROTOCOL>\n"
            "Every token shaped like ⟦REF_n⟧ is an immutable local terminology slot. "
            "Copy each slot exactly once in the grammatically correct position. Do not "
            "translate, expand, duplicate, omit, or explain it. The application restores "
            "its user-approved target after the response. Translate all surrounding text.\n"
            "</LOCAL_REFERENCE_PROTOCOL>"
        )
    if fuzzy:
        candidates = "\n".join(
            f"- observed `{item.observed}` may be `{item.source}` -> {item.target}"
            for item in fuzzy
        )
        sections.append(
            "<OCR_CANDIDATES>\n"
            "Non-binding hints: use one only if the entire sentence makes the correction "
            "clear; otherwise ignore it.\n"
            + candidates
            + "\n</OCR_CANDIDATES>"
        )
    sections.append(f"<OCR_TEXT>\n{protected.source}\n</OCR_TEXT>")
    system = compile_agent_system(dataset, memory)
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
        "estimated_input_tokens": estimate_tokens(
            "\n".join(message["content"] for message in messages)
        ),
    }


def load_direct_baseline(
    path: Path,
    *,
    dataset: PocDataset,
    dataset_sha256: str,
    repetitions: int,
    cases: tuple[PocCase, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load complete POC-5 DIRECT evidence and reject incompatible baselines."""

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = next((row for row in rows if row.get("type") == "run"), None)
    if not isinstance(metadata, dict):
        raise AgentMemoryError("baseline file has no run metadata")
    if metadata.get("experiment") != "visible_session_replay":
        raise AgentMemoryError("baseline is not a POC-5 visible-session run")
    if metadata.get("dataset_sha256") != dataset_sha256:
        raise AgentMemoryError("baseline uses a different dataset")
    if metadata.get("repetitions") != repetitions:
        raise AgentMemoryError("baseline repetition count does not match this run")

    selected_ids = {case.id for case in cases}
    by_case = {case.id: case for case in dataset.cases}
    records = [
        row
        for row in rows
        if row.get("type") == "result"
        and row.get("group") == "DIRECT"
        and row.get("case_id") in selected_ids
    ]
    expected = len(cases) * repetitions
    if len(records) != expected:
        raise AgentMemoryError(
            f"baseline needs {expected} DIRECT records, found {len(records)}"
        )
    identities: set[tuple[str, int]] = set()
    for record in records:
        identity = (str(record.get("case_id")), int(record.get("repetition", 0)))
        if identity in identities:
            raise AgentMemoryError(f"baseline has duplicate record {identity}")
        identities.add(identity)
        case = by_case[identity[0]]
        if record.get("source") != case.source:
            raise AgentMemoryError(f"baseline source mismatch for {case.id}")
        response = record.get("response")
        if record.get("error") or not isinstance(response, str) or not response.strip():
            raise AgentMemoryError(f"baseline record {identity} is incomplete")
        if not isinstance(record.get("hard_constraint_evaluation"), dict):
            raise AgentMemoryError(f"baseline record {identity} lacks hard evaluation")
    order = {case.id: index for index, case in enumerate(cases)}
    records.sort(key=lambda row: (int(row["repetition"]), order[row["case_id"]]))
    return metadata, records


def _new_blind_id(run_id: str, group: str, case_id: str, repetition: int) -> str:
    return hashlib.sha256(
        f"{run_id}:{group}:{case_id}:{repetition}".encode("utf-8")
    ).hexdigest()[:12]


def clone_baseline_records(
    records: list[dict[str, Any]],
    *,
    run_id: str,
    baseline_path: Path,
) -> list[dict[str, Any]]:
    cloned = []
    for request_index, source in enumerate(records, 1):
        record = dict(source)
        record["run_id"] = run_id
        record["request_index"] = request_index
        record["baseline_original_blind_id"] = source.get("blind_id")
        record["baseline_source"] = str(baseline_path.resolve())
        record["group"] = "DIRECT_BASELINE"
        record["blind_id"] = _new_blind_id(
            run_id,
            "DIRECT_BASELINE",
            str(record["case_id"]),
            int(record["repetition"]),
        )
        cloned.append(record)
    return cloned


def summarize(records: list[dict[str, Any]], *, repetitions: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [record for record in records if record["group"] == group]
        completed = [record for record in selected if record.get("response") is not None]
        latencies = [float(record["latency_s"]) for record in completed]
        stable = 0
        comparable = 0
        for case_id in sorted({record["case_id"] for record in completed}):
            outputs = [
                record.get("evaluation", {}).get("restored_response", record["response"])
                for record in completed
                if record["case_id"] == case_id
            ]
            if len(outputs) == repetitions:
                comparable += 1
                stable += len(set(outputs)) == 1
        groups[group] = {
            "requests": len(selected),
            "completed": len(completed),
            "errors": sum(bool(record.get("error")) for record in selected),
            "hard_constraint_pass": sum(
                bool(record.get("hard_constraint_evaluation", {}).get("ok"))
                for record in completed
            ),
            "hard_constraint_pass_rate": round(
                sum(
                    bool(record.get("hard_constraint_evaluation", {}).get("ok"))
                    for record in completed
                ) / len(completed),
                3,
            ) if completed else None,
            "missing_expected_terms": sum(
                len(
                    record.get("hard_constraint_evaluation", {}).get(
                        "missing_expected_terms", []
                    )
                )
                for record in completed
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
            "exact_repeat_stability": f"{stable}/{comparable}" if comparable else None,
            "semantic_fidelity": "NOT_SCORED_USE_BLIND_REVIEW",
            "constraint_aware_naturalness": "NOT_SCORED_USE_BLIND_REVIEW",
        }
    return {"type": "summary", "repetitions": repetitions, "groups": groups}


def write_blind_review(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
    rows = []
    for record in records:
        if record.get("response") is None:
            continue
        restored = record.get("evaluation", {}).get(
            "restored_response", record["response"]
        )
        rows.append(
            {
                "blind_id": record["blind_id"],
                "case_id": record["case_id"],
                "category": record["category"],
                "source": record["source"],
                "translation": restored,
                "confirmed_user_constraints": (
                    "Judge naturalness under the required all-hiragana, glossary-bracket, "
                    "and exact-double-space format. Do not reward breaking a hard rule."
                ),
                "expected_fixed_outputs": record.get("expected_fixed_outputs", []),
                "semantic_fidelity_0_to_5": None,
                "naturalness_0_to_5": None,
                "meaning_error": "",
                "review_notes": "",
            }
        )
    random.Random(seed ^ 0xA61E).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("logs") / "poc"
    return (
        root / f"agent-memory-{stamp}.jsonl",
        root / f"agent-memory-state-{stamp}.json",
        root / f"agent-memory-blind-{stamp}.jsonl",
    )


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path, strict=True)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    entries = dataset.glossary_sets["200"]
    cases = dataset.cases[: args.limit or None]
    baseline_path = Path(args.baseline)
    baseline_metadata, baseline_records = load_direct_baseline(
        baseline_path,
        dataset=dataset,
        dataset_sha256=dataset_sha256,
        repetitions=args.repetitions,
        cases=cases,
    )

    output_default, memory_default, blind_default = _default_paths()
    output_path = Path(args.output) if args.output else output_default
    memory_output = Path(args.memory_output) if args.memory_output else memory_default
    blind_output = Path(args.blind_output) if args.blind_output else blind_default
    for path in (output_path, memory_output, blind_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    settings = AppSettings.load()
    fast_model = args.fast_model.strip() or settings.ai.fast_model_name or "deepseek-v4-flash"
    thinking_model = (
        args.thinking_model.strip()
        or settings.ai.thinking_model_name
        or "deepseek-v4-pro"
    )
    baseline_model = str(baseline_metadata.get("fast_model", "")).strip()
    if baseline_model and fast_model != baseline_model:
        raise AgentMemoryError(
            f"fast model must match baseline ({baseline_model}), got {fast_model}"
        )
    if not args.dry_run and (
        not settings.ai.base_url or not settings.ai.api_key or not fast_model or not thinking_model
    ):
        raise SystemExit("API base URL, key, fast model, and thinking model must be configured")

    compiler_messages = [
        {"role": "system", "content": _MEMORY_COMPILER_SYSTEM},
        {"role": "user", "content": build_memory_source(dataset, entries)},
    ]
    compiler_payload = {
        "model": thinking_model,
        "messages": compiler_messages,
        "temperature": 0.0,
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }
    bootstrap: dict[str, Any] = {
        "type": "bootstrap",
        "payload": compiler_payload,
        "response": None,
        "usage": {},
        "latency_s": None,
        "memory_source": "",
        "reasoning_content_stored": False,
    }
    forbidden_sources = tuple(case.source for case in dataset.cases)
    if args.memory:
        memory = load_agent_memory(
            Path(args.memory),
            dataset_sha256=dataset_sha256,
            forbidden_example_sources=forbidden_sources,
        )
        bootstrap["memory_source"] = str(Path(args.memory).resolve())
    elif args.dry_run:
        memory = make_dry_run_memory(dataset)
        bootstrap["memory_source"] = "deterministic_dry_run_fixture"
    else:
        started = time.perf_counter()
        response, usage = _send_request(
            base_url=settings.ai.base_url,
            api_key=settings.ai.api_key,
            payload=compiler_payload,
            timeout_seconds=args.bootstrap_timeout,
        )
        bootstrap["latency_s"] = round(time.perf_counter() - started, 3)
        bootstrap["response"] = response
        bootstrap["usage"] = usage
        bootstrap["memory_source"] = "thinking_model_api"
        memory = validate_agent_memory(
            extract_json_object(response),
            forbidden_example_sources=forbidden_sources,
        )
    bootstrap["memory"] = memory
    save_agent_memory(
        memory_output,
        memory,
        dataset_sha256=dataset_sha256,
        thinking_model=thinking_model,
        source=bootstrap["memory_source"],
    )

    if args.bootstrap_only:
        print(f"Agent memory: {memory_output.resolve()}")
        return output_path, memory_output, blind_output

    run_id = datetime.now(timezone.utc).isoformat()
    records = clone_baseline_records(
        baseline_records,
        run_id=run_id,
        baseline_path=baseline_path,
    )
    rng = random.Random(args.seed)
    metadata = {
        "type": "run",
        "run_id": run_id,
        "experiment": "compiled_persistent_agent_memory",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "baseline": str(baseline_path.resolve()),
        "fast_model": fast_model,
        "thinking_model": thinking_model,
        "translation_thinking": "disabled",
        "reasoning_content_replayed": False,
        "groups": GROUPS,
        "repetitions": args.repetitions,
        "new_flash_api_requests": len(cases) * args.repetitions,
        "reused_baseline_records": len(baseline_records),
        "dry_run": args.dry_run,
        "success_gate": {
            "agent_hard_constraint_pass_rate_min": 0.80,
            "agent_missing_expected_terms_max": 2,
            "agent_latency_p50_s_max": 3.0,
            "semantic_fidelity_improvement_min_0_to_5": 0.5,
            "naturalness_must_not_regress": True,
        },
    }

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        handle.write(json.dumps(bootstrap, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        request_index = len(records)
        for repetition in range(1, args.repetitions + 1):
            ordered_cases = list(cases)
            rng.shuffle(ordered_cases)
            for case in ordered_cases:
                request_index += 1
                built = build_agent_request(dataset, memory, case, entries)
                payload = {
                    "model": fast_model,
                    "messages": built["messages"],
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "thinking": {"type": "disabled"},
                }
                entries_by_id = {entry.id: entry for entry in entries}
                record: dict[str, Any] = {
                    "type": "result",
                    "run_id": run_id,
                    "request_index": request_index,
                    "blind_id": _new_blind_id(
                        run_id, "AGENT_MEMORY", case.id, repetition
                    ),
                    "repetition": repetition,
                    "group": "AGENT_MEMORY",
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
                    "estimated_input_tokens": built["estimated_input_tokens"],
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
                        evaluation = evaluate_response(
                            dataset,
                            case,
                            entries,
                            response,
                            built["placeholders"],
                        )
                        record["response"] = response
                        record["usage"] = usage
                        record["evaluation"] = evaluation
                        record["hard_constraint_evaluation"] = evaluate_hard_constraints(
                            dataset,
                            case,
                            evaluation["restored_response"],
                        )
                    except TranslationError as exc:
                        record["error"] = str(exc)
                    record["latency_s"] = round(time.perf_counter() - started, 3)
                    print(
                        f"{request_index - len(baseline_records):02d}/"
                        f"{len(cases) * args.repetitions} repetition={repetition} "
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
        summary["api_calls_this_run"] = {
            "pro": 0 if args.memory or args.dry_run else 1,
            "flash": 0 if args.dry_run else len(cases) * args.repetitions,
        }
        summary["bootstrap"] = {
            "memory_source": bootstrap["memory_source"],
            "latency_s": bootstrap["latency_s"],
            "usage": bootstrap["usage"],
            "memory_output": str(memory_output.resolve()),
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    write_blind_review(blind_output, records, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Agent memory: {memory_output.resolve()}")
    print(f"Blind review: {blind_output.resolve()}")
    return output_path, memory_output, blind_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="completed POC-5 result JSONL")
    parser.add_argument("--dataset", default="poc/data/reference_poc_dataset.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--memory-output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--memory", default="", help="reuse persisted memory; skip Pro")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--thinking-model", default="")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="debug only")
    parser.add_argument("--seed", type=int, default=20260701)
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
    except (AgentMemoryError, json.JSONDecodeError, OSError, TranslationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
