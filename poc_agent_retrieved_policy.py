"""Final decision POC for a retrieved, typed Pro policy.

The previous RAW_BRIEF and PRO_PLAYBOOK results are reused as frozen baselines.
No new Pro call is made.  The saved Pro playbook is filtered to source-backed
domain decisions, stored locally, and reduced to at most three rules relevant
to each current OCR sentence.  Only the RETRIEVED_POLICY Flash group is new.
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
from poc_agent_memory_transfer import evaluate_format, normalize_surface
from poc_agent_strategy_transfer import (
    GROUPS as BASELINE_GROUPS,
    _NEUTRAL_ACK,
    StrategyCase,
    StrategyDataset,
    build_bootstrap_turn,
    build_session_system,
    evaluate_anchors,
    load_strategy_dataset,
)
from poc_reference_injection import _percentile, _send_request, estimate_tokens


GROUPS = (*BASELINE_GROUPS, "RETRIEVED_POLICY")
ALLOWED_TRIGGERS = ("跑", "复现", "上线", "回滚", "合并", "返回", "引起")
MAX_RULES_PER_REQUEST = 3
_POLICY_JAPANESE = re.compile(r"^[\u3041-\u3096\[\] ]+$")


class RetrievedPolicyError(ValueError):
    """Raised when baseline reuse or policy retrieval is not auditable."""


def find_latest_baseline(root: Path = Path("logs/poc")) -> Path:
    candidates = [
        path
        for path in root.glob("agent-strategy-*.jsonl")
        if "blind" not in path.name and "session" not in path.name and "dry" not in path.name
    ]
    if not candidates:
        raise RetrievedPolicyError("no formal agent-strategy baseline was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_baseline(
    path: Path,
    *,
    dataset: StrategyDataset,
    allow_dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = [row for row in rows if row.get("type") == "run"]
    bootstraps = [row for row in rows if row.get("type") == "bootstrap"]
    results = [row for row in rows if row.get("type") == "result"]
    if len(runs) != 1 or len(bootstraps) != 1:
        raise RetrievedPolicyError("baseline must contain exactly one run and bootstrap record")
    run, bootstrap = runs[0], bootstraps[0]
    if run.get("experiment") != "visible_pro_strategy_transfer" or (
        run.get("dry_run") and not allow_dry_run
    ):
        raise RetrievedPolicyError("baseline is not a formal visible strategy run")
    if tuple(run.get("groups", [])) != BASELINE_GROUPS:
        raise RetrievedPolicyError("baseline groups do not match RAW_BRIEF and PRO_PLAYBOOK")
    if run.get("cases") != len(dataset.cases) or run.get("repetitions") != 2:
        raise RetrievedPolicyError("baseline case count or repetitions do not match")
    if len(results) != len(dataset.cases) * len(BASELINE_GROUPS) * 2:
        raise RetrievedPolicyError("baseline must contain exactly 48 result records")

    expected = {
        (group, case.id, repetition)
        for group in BASELINE_GROUPS
        for case in dataset.cases
        for repetition in (1, 2)
    }
    actual = {
        (str(row.get("group")), str(row.get("case_id")), int(row.get("repetition", 0)))
        for row in results
    }
    if actual != expected:
        raise RetrievedPolicyError("baseline group/case/repetition coverage is incomplete")
    sources = {case.id: case.source for case in dataset.cases}
    if any(row.get("source") != sources.get(str(row.get("case_id"))) for row in results):
        raise RetrievedPolicyError("baseline sources differ from the frozen dataset")
    if any(row.get("error") or not row.get("translation") for row in results):
        raise RetrievedPolicyError("baseline contains failed or empty translations")
    if any((row.get("payload") or {}).get("thinking") != {"type": "disabled"} for row in results):
        raise RetrievedPolicyError("baseline contains a Flash request without thinking disabled")
    playbook = bootstrap.get("playbook")
    if not isinstance(playbook, dict):
        raise RetrievedPolicyError("baseline bootstrap has no structured playbook")
    return run, bootstrap, results


def _policy_terms(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RetrievedPolicyError(f"policy {field} must be a non-empty list")
    terms = []
    for item in value:
        text = str(item).strip()
        if not text or not _POLICY_JAPANESE.fullmatch(text):
            raise RetrievedPolicyError(f"policy {field} contains non-hiragana guidance: {text!r}")
        terms.append(text)
    return terms[:3]


def compile_typed_policy(
    playbook: dict[str, Any],
    *,
    dataset: StrategyDataset,
) -> dict[str, Any]:
    """Keep only decisions whose Chinese trigger was explicitly supplied by the user."""

    for trigger in ALLOWED_TRIGGERS:
        if trigger not in dataset.startup_brief:
            raise RetrievedPolicyError(f"allowed trigger is not backed by startup brief: {trigger}")
    decisions = playbook.get("domain_sense_decisions")
    if not isinstance(decisions, list):
        raise RetrievedPolicyError("playbook domain_sense_decisions must be a list")

    rules: list[dict[str, Any]] = []
    seen: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        expression = str(decision.get("source_expression", "")).strip()
        trigger = next(
            (
                candidate
                for candidate in ALLOWED_TRIGGERS
                if candidate == expression or candidate in expression
            ),
            "",
        )
        if not trigger or trigger in seen:
            continue
        meaning = str(decision.get("contextual_meaning", "")).strip()
        if not meaning:
            raise RetrievedPolicyError(f"policy rule {trigger} has no contextual meaning")
        rules.append(
            {
                "rule_id": f"startup:{trigger}",
                "trigger": trigger,
                "contextual_meaning": meaning,
                "preferred_japanese": _policy_terms(
                    decision.get("preferred_japanese"), field=f"{trigger}.preferred_japanese"
                ),
                "avoid": _policy_terms(decision.get("avoid"), field=f"{trigger}.avoid"),
                "evidence": "explicit_user_startup_brief",
                "status": "poc_candidate",
            }
        )
        seen.add(trigger)
    if len(rules) < 5:
        raise RetrievedPolicyError(f"too few source-backed policy rules: {len(rules)}")
    return {
        "version": 1,
        "role": "typed_retrieved_translation_policy",
        "source": "filtered_visible_pro_playbook",
        "max_rules_per_request": MAX_RULES_PER_REQUEST,
        "excluded_playbook_sections": [
            "natural_patterns",
            "quality_checks",
            "generated_examples",
        ],
        "rules": rules,
    }


def select_policy_rules(policy: dict[str, Any], source: str) -> list[dict[str, Any]]:
    selected = [rule for rule in policy["rules"] if rule["trigger"] in source]
    return selected[:MAX_RULES_PER_REQUEST]


def build_retrieved_messages(
    *,
    system: str,
    bootstrap_turn: str,
    policy: dict[str, Any],
    case: StrategyCase,
) -> tuple[list[dict[str, str]], list[str]]:
    selected = select_policy_rules(policy, case.source)
    if selected:
        assistant = json.dumps(
            {
                "version": 1,
                "role": "relevant_translation_policy_slice",
                "instruction": (
                    "Apply only these source-backed semantic decisions when their trigger has "
                    "the stated contextual meaning. Confirmed system rules and glossary remain authoritative."
                ),
                "rules": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
    else:
        assistant = _NEUTRAL_ACK
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": bootstrap_turn},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": f"<OCR_TEXT>\n{case.source}\n</OCR_TEXT>"},
    ]
    return messages, [rule["rule_id"] for rule in selected]


def clone_baseline_records(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    cloned = []
    for row in rows:
        item = dict(row)
        old_blind_id = str(item.get("blind_id", ""))
        item["blind_id"] = hashlib.sha256(
            f"{run_id}:{item['group']}:{item['case_id']}:{item['repetition']}".encode()
        ).hexdigest()[:12]
        item["record_source"] = "reused_frozen_baseline"
        item["baseline_original_blind_id"] = old_blind_id
        cloned.append(item)
    return cloned


def summarize(records: list[dict[str, Any]], *, repetitions: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in GROUPS:
        selected = [row for row in records if row["group"] == group]
        completed = [row for row in selected if row.get("translation")]
        latencies = [float(row["latency_s"]) for row in completed]
        stable = 0
        case_ids = sorted({row["case_id"] for row in completed})
        for case_id in case_ids:
            outputs = [row["translation"] for row in completed if row["case_id"] == case_id]
            stable += len(outputs) == repetitions and len(set(outputs)) == 1
        groups[group] = {
            "requests": len(selected),
            "completed": len(completed),
            "new_api_requests": sum(row.get("record_source") == "new_retrieved_request" for row in selected),
            "errors": sum(bool(row.get("error")) for row in selected),
            "format_pass": sum(bool((row.get("hard_constraint_evaluation") or {}).get("ok")) for row in completed),
            "diagnostic_anchor_pass": sum(bool((row.get("anchor_evaluation") or {}).get("ok")) for row in completed),
            "provider_prompt_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens", 0) or 0) for row in completed),
            "estimated_input_tokens_mean": round(statistics.fmean(float(row["estimated_input_tokens"]) for row in selected), 1) if selected else None,
            "latency_p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "latency_p95_s": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "exact_repeat_stability": f"{stable}/{len(case_ids)}" if case_ids else None,
            "semantic_fidelity": "NOT_SCORED_USE_COMBINED_BLIND_REVIEW",
            "naturalness": "NOT_SCORED_USE_COMBINED_BLIND_REVIEW",
        }
    return {
        "type": "summary",
        "repetitions": repetitions,
        "groups": groups,
        "formal_decision_after_blind_review": {
            "RETRIEVED_POLICY_wins": (
                "Integrate local typed policy storage, relevant-rule retrieval, cached Pro refresh, "
                "and shared fast/thinking session into the application."
            ),
            "RAW_BRIEF_wins": (
                "Do not use Pro startup policy; replay only relevant raw confirmed feedback and "
                "reserve Pro for explicit thinking mode or feedback proposals."
            ),
            "PRO_PLAYBOOK_wins": (
                "Revisit retrieval coverage, but do not integrate the full playbook until its "
                "observed major semantic errors are eliminated."
            ),
        },
    }


def write_combined_blind(path: Path, records: list[dict[str, Any]], *, seed: int) -> None:
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
    random.Random(seed ^ 0x7E71).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("logs/poc")
    return (
        root / f"agent-retrieved-policy-{stamp}.jsonl",
        root / f"agent-retrieved-policy-state-{stamp}.json",
        root / f"agent-retrieved-policy-blind-{stamp}.jsonl",
    )


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_path = Path(args.dataset)
    translation_path = Path(args.translation_dataset)
    dataset = load_strategy_dataset(dataset_path)
    baseline_path = Path(args.baseline) if args.baseline else find_latest_baseline()
    baseline_run, bootstrap, baseline_rows = load_baseline(
        baseline_path,
        dataset=dataset,
        allow_dry_run=args.dry_run,
    )
    policy = compile_typed_policy(bootstrap["playbook"], dataset=dataset)
    system, prompt_hash = build_session_system(translation_path)
    bootstrap_turn = build_bootstrap_turn(dataset)

    settings = AppSettings.load()
    baseline_fast_model = str(baseline_run.get("fast_model", "")).strip()
    fast_model = args.fast_model.strip() or baseline_fast_model or settings.ai.fast_model_name
    if fast_model != baseline_fast_model:
        raise RetrievedPolicyError("new group must use the same fast model as the frozen baseline")
    if not args.dry_run and (not settings.ai.base_url or not settings.ai.api_key):
        raise RetrievedPolicyError("API base URL and key must be configured")

    default_output, default_state, default_blind = _default_paths()
    output_path = Path(args.output) if args.output else default_output
    state_path = Path(args.state_output) if args.state_output else default_state
    blind_path = Path(args.blind_output) if args.blind_output else default_blind
    for path in (output_path, state_path, blind_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).isoformat()
    source_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    state = {
        "metadata": {
            "version": 1,
            "created_at": run_id,
            "baseline_path": str(baseline_path.resolve()),
            "baseline_sha256": source_hash,
            "prompt_hash": prompt_hash,
            "fast_model": fast_model,
            "contains_reasoning_content": False,
        },
        "policy": policy,
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "type": "run",
        "run_id": run_id,
        "experiment": "retrieved_typed_policy_final_decision",
        "baseline": str(baseline_path.resolve()),
        "baseline_run_id": baseline_run.get("run_id"),
        "groups": list(GROUPS),
        "cases": len(dataset.cases),
        "repetitions": 2,
        "fast_model": fast_model,
        "reused_baseline_records": len(baseline_rows),
        "new_flash_api_requests": len(dataset.cases) * 2,
        "new_pro_api_requests": 0,
        "max_rules_per_request": MAX_RULES_PER_REQUEST,
        "reasoning_content_replayed": False,
        "dry_run": args.dry_run,
    }
    records = clone_baseline_records(baseline_rows, run_id=run_id)
    rng = random.Random(args.seed)
    schedule = [
        (repetition, case)
        for repetition in (1, 2)
        for case in dataset.cases
    ]
    rng.shuffle(schedule)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        handle.write(json.dumps({"type": "policy_state", "policy": policy}, ensure_ascii=False) + "\n")
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for index, (repetition, case) in enumerate(schedule, 1):
            messages, selected_ids = build_retrieved_messages(
                system=system,
                bootstrap_turn=bootstrap_turn,
                policy=policy,
                case=case,
            )
            payload = {
                "model": fast_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 4096,
                "thinking": {"type": "disabled"},
            }
            started = time.perf_counter()
            usage: dict[str, Any] = {}
            error = None
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
            anchors = evaluate_anchors(case, translation) if translation else {"ok": False, "quality_claim": False}
            record = {
                "type": "result",
                "blind_id": hashlib.sha256(
                    f"{run_id}:RETRIEVED_POLICY:{case.id}:{repetition}".encode()
                ).hexdigest()[:12],
                "request_index": index,
                "group": "RETRIEVED_POLICY",
                "case_id": case.id,
                "category": case.category,
                "repetition": repetition,
                "source": case.source,
                "review_focus": case.review_focus,
                "record_source": "new_retrieved_request",
                "selected_policy_rule_ids": selected_ids,
                "payload": payload,
                "response": response,
                "translation": translation,
                "normalization": normalized,
                "hard_constraint_evaluation": hard,
                "anchor_evaluation": anchors,
                "estimated_input_tokens": estimate_tokens("\n".join(m["content"] for m in messages)),
                "usage": usage,
                "latency_s": latency,
                "error": error,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{index:02d}/{len(schedule)} RETRIEVED_POLICY {case.id} "
                f"rules={len(selected_ids)} elapsed={latency:.3f}s ok={not error}"
            )
            if args.delay and not args.dry_run:
                time.sleep(args.delay)
        summary = summarize(records, repetitions=2)
        summary["evidence"] = {
            "baseline_reused": str(baseline_path.resolve()),
            "policy_state": str(state_path.resolve()),
            "combined_blind_rows": len(records),
            "new_api_calls": {"pro": 0, "flash": len(dataset.cases) * 2},
        }
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    write_combined_blind(blind_path, records, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output_path.resolve()}")
    print(f"Policy state: {state_path.resolve()}")
    print(f"Combined blind review: {blind_path.resolve()}")
    return output_path, state_path, blind_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="poc_data/agent_strategy_transfer_dataset.json")
    parser.add_argument("--translation-dataset", default="poc_data/reference_poc_dataset.json")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--state-output", default="")
    parser.add_argument("--blind-output", default="")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--flash-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except (RetrievedPolicyError, OSError, json.JSONDecodeError, TranslationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
