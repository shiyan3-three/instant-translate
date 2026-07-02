"""POC 3: compare reference injection strategies without changing the app.

The runner intentionally keeps test data outside the script.  A strict run
requires an approved 24-30 case dataset and glossary sets containing exactly
20 and 200 entries.  ``--dry-run`` writes every exact API payload without
sending a request, so prompts can be reviewed before API quota is spent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

from app.settings import AppSettings
from app.translation.client import TranslationError


VARIANTS = ("A", "B2", "B3")
SCALES = ("20", "200")
REQUIRED_CATEGORIES = {
    "semantic",
    "terminology",
    "format",
    "style",
    "forbidden",
    "ocr_noise",
}


class DatasetError(ValueError):
    """Raised when POC data is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class GlossaryEntry:
    id: str
    source: str
    target: str
    aliases: tuple[str, ...] = ()
    format: str = ""
    risk: str = "low"

    @property
    def surfaces(self) -> tuple[str, ...]:
        return tuple(item for item in (self.source, *self.aliases) if item)


@dataclass(frozen=True)
class PocCase:
    id: str
    category: str
    source: str
    expected_entry_ids: tuple[str, ...] = ()
    required_outputs: tuple[str, ...] = ()
    forbidden_outputs: tuple[str, ...] = ()
    checks: dict[str, Any] | None = None
    notes: str = ""


@dataclass(frozen=True)
class PocDataset:
    version: int
    status: str
    name: str
    base_system_prompt: str
    global_rules: tuple[str, ...]
    style_rules: tuple[str, ...]
    forbidden_outputs: tuple[str, ...]
    examples: tuple[dict[str, str], ...]
    glossary_sets: dict[str, tuple[GlossaryEntry, ...]]
    cases: tuple[PocCase, ...]
    raw_user_constraints: str = ""


@dataclass(frozen=True)
class PlaceholderPlan:
    source: str
    replacements: dict[str, str]
    entry_ids: dict[str, str]


def _entry_from_json(raw: dict[str, Any]) -> GlossaryEntry:
    return GlossaryEntry(
        id=str(raw.get("id", "")).strip(),
        source=str(raw.get("source", "")).strip(),
        target=str(raw.get("target", "")).strip(),
        aliases=tuple(str(item).strip() for item in raw.get("aliases", []) if str(item).strip()),
        format=str(raw.get("format", "")).strip(),
        risk=str(raw.get("risk", "low")).strip() or "low",
    )


def load_dataset(path: Path, *, strict: bool = True) -> PocDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    glossary_sets = {
        str(scale): tuple(_entry_from_json(item) for item in entries)
        for scale, entries in raw.get("glossary_sets", {}).items()
    }
    cases = tuple(
        PocCase(
            id=str(item.get("id", "")).strip(),
            category=str(item.get("category", "")).strip(),
            source=str(item.get("source", "")).strip(),
            expected_entry_ids=tuple(item.get("expected_entry_ids", [])),
            required_outputs=tuple(item.get("required_outputs", [])),
            forbidden_outputs=tuple(item.get("forbidden_outputs", [])),
            checks=dict(item.get("checks", {})),
            notes=str(item.get("notes", "")).strip(),
        )
        for item in raw.get("cases", [])
    )
    dataset = PocDataset(
        version=int(raw.get("version", 0)),
        status=str(raw.get("status", "draft")).strip(),
        name=str(raw.get("name", path.stem)).strip(),
        base_system_prompt=str(raw.get("base_system_prompt", "")).strip(),
        global_rules=tuple(raw.get("global_rules", [])),
        style_rules=tuple(raw.get("style_rules", [])),
        forbidden_outputs=tuple(raw.get("forbidden_outputs", [])),
        examples=tuple(raw.get("examples", [])),
        glossary_sets=glossary_sets,
        cases=cases,
        raw_user_constraints=str(raw.get("raw_user_constraints", "")).strip(),
    )
    validate_dataset(dataset, strict=strict)
    return dataset


def validate_dataset(dataset: PocDataset, *, strict: bool = True) -> None:
    errors: list[str] = []
    if dataset.version != 1:
        errors.append("version must be 1")
    if not dataset.base_system_prompt:
        errors.append("base_system_prompt is empty")

    case_ids = [case.id for case in dataset.cases]
    if any(not item for item in case_ids):
        errors.append("every case needs a non-empty id")
    if len(case_ids) != len(set(case_ids)):
        errors.append("case ids must be unique")

    for scale in SCALES:
        entries = dataset.glossary_sets.get(scale, ())
        entry_ids = [entry.id for entry in entries]
        if any(not entry.id or not entry.source or not entry.target for entry in entries):
            errors.append(f"glossary {scale} has an entry missing id/source/target")
        if len(entry_ids) != len(set(entry_ids)):
            errors.append(f"glossary {scale} entry ids must be unique")

    large_ids = {entry.id for entry in dataset.glossary_sets.get("200", ())}
    small_ids = {entry.id for entry in dataset.glossary_sets.get("20", ())}
    if not small_ids.issubset(large_ids):
        errors.append("the 200-entry glossary must contain every 20-entry id")

    for case in dataset.cases:
        if not case.source:
            errors.append(f"case {case.id or '<unknown>'} has empty source")
        unknown = set(case.expected_entry_ids) - small_ids
        if unknown:
            errors.append(
                f"case {case.id} expected ids are absent from the 20-entry glossary: "
                f"{', '.join(sorted(unknown))}"
            )

    if strict:
        if dataset.status != "approved":
            errors.append("status must be 'approved' before a formal run")
        if not 24 <= len(dataset.cases) <= 30:
            errors.append(f"formal run needs 24-30 cases, found {len(dataset.cases)}")
        categories = {case.category for case in dataset.cases}
        missing_categories = REQUIRED_CATEGORIES - categories
        if missing_categories:
            errors.append("missing case categories: " + ", ".join(sorted(missing_categories)))
        for scale, expected_count in (("20", 20), ("200", 200)):
            actual_count = len(dataset.glossary_sets.get(scale, ()))
            if actual_count != expected_count:
                errors.append(
                    f"glossary {scale} must contain {expected_count} entries, found {actual_count}"
                )

    if errors:
        raise DatasetError("POC dataset is not ready:\n- " + "\n- ".join(errors))


def match_glossary(source: str, entries: Iterable[GlossaryEntry]) -> tuple[GlossaryEntry, ...]:
    """Return entries with a source term or alias present in the input."""

    return tuple(entry for entry in entries if any(surface in source for surface in entry.surfaces))


def protect_glossary_terms(source: str, entries: Iterable[GlossaryEntry]) -> PlaceholderPlan:
    """Replace deterministic, non-overlapping glossary matches before translation.

    A surface shared by multiple entries is deliberately not protected: choosing a
    target would hide the ambiguity that the POC is meant to expose.
    """

    entries = tuple(entries)
    surface_owners: dict[str, list[GlossaryEntry]] = {}
    for entry in entries:
        for surface in entry.surfaces:
            if surface in source:
                surface_owners.setdefault(surface, []).append(entry)

    candidates: list[tuple[int, int, GlossaryEntry]] = []
    for surface, owners in surface_owners.items():
        if len(owners) != 1:
            continue
        for found in re.finditer(re.escape(surface), source):
            candidates.append((found.start(), found.end(), owners[0]))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    selected: list[tuple[int, int, GlossaryEntry]] = []
    cursor = -1
    for start, end, entry in candidates:
        if start < cursor:
            continue
        selected.append((start, end, entry))
        cursor = end

    parts: list[str] = []
    replacements: dict[str, str] = {}
    entry_ids: dict[str, str] = {}
    cursor = 0
    for index, (start, end, entry) in enumerate(selected):
        placeholder = f"⟦REF_{index}⟧"
        parts.append(source[cursor:start])
        parts.append(placeholder)
        replacements[placeholder] = entry.target
        entry_ids[placeholder] = entry.id
        cursor = end
    parts.append(source[cursor:])
    return PlaceholderPlan("".join(parts), replacements, entry_ids)


def restore_placeholders(text: str, replacements: dict[str, str]) -> str:
    for placeholder, target in replacements.items():
        text = text.replace(placeholder, target)
    return text


def _numbered(title: str, values: Iterable[str]) -> str:
    values = tuple(str(value).strip() for value in values if str(value).strip())
    if not values:
        return ""
    return title + "\n" + "\n".join(f"{index}. {value}" for index, value in enumerate(values, 1))


def _format_glossary(entries: Iterable[GlossaryEntry]) -> str:
    lines = []
    for entry in entries:
        aliases = f"; aliases: {', '.join(entry.aliases)}" if entry.aliases else ""
        format_note = f"; format: {entry.format}" if entry.format else ""
        lines.append(
            f"- {entry.id}: {entry.source} -> {entry.target}{aliases}{format_note}; risk: {entry.risk}"
        )
    return "## Glossary\n" + ("\n".join(lines) if lines else "(no matched entries)")


def _format_examples(examples: Iterable[dict[str, str]]) -> str:
    rows = []
    for item in examples:
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if source and target:
            rows.append(f"- Source: {source}\n  Target: {target}")
    return "## Examples\n" + ("\n".join(rows) if rows else "(none)")


def build_stable_system(dataset: PocDataset) -> str:
    sections = [dataset.base_system_prompt]
    for title, values in (
        ("## Global rules", dataset.global_rules),
        ("## Style rules", dataset.style_rules),
        ("## Globally forbidden outputs", dataset.forbidden_outputs),
    ):
        section = _numbered(title, values)
        if section:
            sections.append(section)
    sections.append(
        "Any token shaped like ⟦REF_n⟧ is an immutable placeholder. "
        "Copy it exactly once and translate all surrounding content."
    )
    return "\n\n".join(sections)


def build_request(
    dataset: PocDataset,
    case: PocCase,
    entries: tuple[GlossaryEntry, ...],
    variant: str,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    stable_system = build_stable_system(dataset)
    matched = match_glossary(case.source, entries)
    plan = PlaceholderPlan(case.source, {}, {})

    if variant == "A":
        system = "\n\n".join(
            (stable_system, "## Full reference layer", _format_glossary(entries), _format_examples(dataset.examples))
        )
        user = f"<OCR_TEXT>\n{case.source}\n</OCR_TEXT>"
    else:
        plan = protect_glossary_terms(case.source, matched)
        dynamic_reference = "\n\n".join((_format_glossary(matched), _format_examples(dataset.examples)))
        mandatory = (
            "These references are mandatory for this request. Use every relevant mapping, "
            "preserve every placeholder exactly once, and translate the complete OCR text.\n\n"
            + dynamic_reference
        )
        if variant == "B2":
            system = stable_system
            user = (
                "<MANDATORY_REFERENCE>\n"
                + mandatory
                + "\n</MANDATORY_REFERENCE>\n\n<OCR_TEXT>\n"
                + plan.source
                + "\n</OCR_TEXT>"
            )
        else:
            system = stable_system + "\n\n## Dynamic mandatory reference\n" + mandatory
            user = f"<OCR_TEXT>\n{plan.source}\n</OCR_TEXT>"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return {
        "messages": messages,
        "matched_entry_ids": [entry.id for entry in matched],
        "protected_source": plan.source,
        "placeholders": plan.replacements,
        "placeholder_entry_ids": plan.entry_ids,
        "estimated_input_tokens": estimate_tokens(system + "\n" + user),
    }


def estimate_tokens(text: str) -> int:
    cjk = sum(1 for char in text if "\u3400" <= char <= "\u9fff")
    return cjk + math.ceil((len(text) - cjk) / 4)


def evaluate_response(
    dataset: PocDataset,
    case: PocCase,
    entries: tuple[GlossaryEntry, ...],
    raw_response: str,
    placeholders: dict[str, str],
) -> dict[str, Any]:
    placeholder_counts = {key: raw_response.count(key) for key in placeholders}
    restored = restore_placeholders(raw_response, placeholders)
    by_id = {entry.id: entry for entry in entries}
    expected_targets = [by_id[item].target for item in case.expected_entry_ids if item in by_id]
    missing_terms = [target for target in expected_targets if target not in restored]
    forbidden = tuple(dataset.forbidden_outputs) + tuple(case.forbidden_outputs)
    present_forbidden = [item for item in forbidden if item and item in restored]
    missing_required = [item for item in case.required_outputs if item not in restored]

    checks = case.checks or {}
    check_results: dict[str, bool] = {}
    allowed_pattern = str(checks.get("allowed_pattern", "")).strip()
    if allowed_pattern:
        check_results["allowed_pattern"] = bool(re.fullmatch(allowed_pattern, restored))
    if checks.get("no_edge_whitespace"):
        check_results["no_edge_whitespace"] = restored == restored.strip()
    for index, pattern in enumerate(checks.get("required_patterns", []), 1):
        check_results[f"required_pattern_{index}"] = bool(re.search(pattern, restored))
    for index, pattern in enumerate(checks.get("forbidden_patterns", []), 1):
        check_results[f"forbidden_pattern_{index}"] = not bool(re.search(pattern, restored))

    placeholder_ok = all(count == 1 for count in placeholder_counts.values())
    machine_ok = (
        placeholder_ok
        and not missing_terms
        and not present_forbidden
        and not missing_required
        and all(check_results.values())
    )
    return {
        "restored_response": restored,
        "placeholder_counts": placeholder_counts,
        "placeholder_ok": placeholder_ok,
        "missing_expected_terms": missing_terms,
        "present_forbidden_outputs": present_forbidden,
        "missing_required_outputs": missing_required,
        "checks": check_results,
        "machine_ok": machine_ok,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        for variant in VARIANTS:
            selected = [
                record
                for record in records
                if record["scale"] == scale and record["variant"] == variant
            ]
            completed = [record for record in selected if record.get("response") is not None]
            latencies = [float(record["latency_s"]) for record in completed]
            evaluations = [record["evaluation"] for record in completed]
            groups[f"{scale}/{variant}"] = {
                "requests": len(selected),
                "completed": len(completed),
                "errors": sum(1 for record in selected if record.get("error")),
                "machine_pass": sum(1 for item in evaluations if item["machine_ok"]),
                "missing_expected_terms": sum(
                    len(item["missing_expected_terms"]) for item in evaluations
                ),
                "input_tokens_estimated_mean": round(
                    statistics.fmean(record["estimated_input_tokens"] for record in selected), 1
                ) if selected else None,
                "provider_prompt_tokens": sum(
                    int(record.get("usage", {}).get("prompt_tokens", 0) or 0) for record in completed
                ),
                "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
                "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            }
    return {"type": "summary", "groups": groups}


def _send_request(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    try:
        response = httpx.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise TranslationError("API returned an empty translation")
        return content.strip(), dict(body.get("usage", {}))
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise TranslationError(str(exc)) from exc


def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("logs") / "poc" / f"reference-injection-{stamp}.jsonl"


def run(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path, strict=not args.allow_draft)
    if args.validate_only:
        print(
            f"Dataset valid: {len(dataset.cases)} cases, "
            f"20={len(dataset.glossary_sets['20'])}, 200={len(dataset.glossary_sets['200'])}"
        )
        return Path()

    settings = AppSettings.load()
    model = args.model.strip() or settings.ai.fast_model_name
    if not args.dry_run and (not settings.ai.base_url or not settings.ai.api_key or not model):
        raise SystemExit("API base URL, key, and fast model must be configured")

    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    seed = args.seed
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    run_id = datetime.now(timezone.utc).isoformat()
    scales = tuple(args.scales.split(","))
    variants = tuple(args.variants.split(","))
    unknown_scales = set(scales) - set(SCALES)
    unknown_variants = set(variants) - set(VARIANTS)
    if unknown_scales or unknown_variants:
        raise SystemExit(
            f"Unknown scales/variants: {sorted(unknown_scales | unknown_variants)}"
        )

    metadata = {
        "type": "run",
        "run_id": run_id,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "dataset_status": dataset.status,
        "model": model,
        "thinking": "disabled",
        "seed": seed,
        "dry_run": args.dry_run,
        "scales": scales,
        "variants": variants,
    }

    with output_path.open("w", encoding="utf-8") as output:
        output.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        request_index = 0
        for scale in scales:
            entries = dataset.glossary_sets[scale]
            for case in dataset.cases:
                ordered_variants = list(variants)
                rng.shuffle(ordered_variants)
                for variant in ordered_variants:
                    request_index += 1
                    built = build_request(dataset, case, entries, variant)
                    payload = {
                        "model": model,
                        "messages": built["messages"],
                        "temperature": 0.0,
                        "max_tokens": 4096,
                        "thinking": {"type": "disabled"},
                    }
                    record: dict[str, Any] = {
                        "type": "result",
                        "run_id": run_id,
                        "request_index": request_index,
                        "scale": scale,
                        "variant": variant,
                        "case_id": case.id,
                        "category": case.category,
                        "source": case.source,
                        "matched_entry_ids": built["matched_entry_ids"],
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
                                timeout_seconds=args.timeout,
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
                            f"{request_index:03d} scale={scale} variant={variant} "
                            f"case={case.id} elapsed={record['latency_s']:.3f}s "
                            f"ok={record['error'] is None}"
                        )
                        if args.delay:
                            time.sleep(args.delay)
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()

        summary = summarize(records)
        summary["run_id"] = run_id
        output.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {output_path.resolve()}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="poc_data/reference_poc_dataset.json",
        help="POC dataset JSON path",
    )
    parser.add_argument("--output", default="", help="JSONL result path")
    parser.add_argument("--model", default="", help="override configured fast model")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--scales", default=",".join(SCALES))
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="allow incomplete data for local harness inspection only; never use for conclusions",
    )
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (DatasetError, json.JSONDecodeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
