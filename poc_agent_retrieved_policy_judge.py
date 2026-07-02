"""Blindly score the final three-group Agent policy benchmark with one Pro call."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.settings import AppSettings
from app.translation.client import TranslationError
from poc_agent_bootstrap_review import (
    aggregate_reviews,
    load_result_mapping,
    load_scored_reviews,
)
from poc_agent_memory_transfer import extract_json_object
from poc_agent_retrieved_policy import GROUPS
from poc_reference_injection import _send_request


_JUDGE_SYSTEM = """You are an independent senior Chinese-to-Japanese translation evaluator.
Evaluate every anonymous translation independently. You do not know which runtime group produced it.

The user's unusual surface contract is intentional: Japanese prose must use hiragana only, software glossary terms may be bracketed, and lexical units use exactly two ASCII spaces. Do not penalize those required constraints. Do penalize kanji, katakana, long-vowel marks, broken spacing, or missing required brackets when they violate the stated contract.

semantic_fidelity_0_to_5:
5 = complete meaning, relations, scope, time, negation, terminology, and tone preserved.
4 = correct meaning with only a small wording or terminology issue.
3 = understandable but has a noticeable omission, ambiguity, or mistranslation.
2 = major semantic error while part of the intended message remains.
1 = mostly wrong.
0 = unrelated or unusable.

naturalness_0_to_5:
5 = natural Japanese under the user's hiragana/spacing constraints.
4 = minor awkwardness.
3 = clearly awkward but understandable.
2 = substantially unnatural or grammatically damaged.
1 = barely readable.
0 = unreadable.

Important examples of errors to catch include translating software-test 跑 as physical running, confusing 今日 with こんにちは, losing negation or sequence, using the wrong medicine counter, omitting time qualifiers, or reversing agent/object relations.

Return only one JSON object. Do not output Markdown or explanations outside JSON."""

_JUDGE_SCHEMA = """Use exactly this shape:
{
  "reviews": [
    {
      "blind_id": "copied exactly from input",
      "semantic_fidelity_0_to_5": 0,
      "naturalness_0_to_5": 0,
      "meaning_error": "concise Chinese description, empty when none",
      "review_notes": "concise Chinese note"
    }
  ]
}
Return exactly one review for every input blind_id. Scores may use 0.5 increments."""


class BlindJudgeError(ValueError):
    """Raised when blind evidence or judge output is incomplete."""


def find_latest_final_results(root: Path = Path("logs/poc")) -> Path:
    candidates = [
        path
        for path in root.glob("agent-retrieved-policy-*.jsonl")
        if "blind" not in path.name and "dry" not in path.name
    ]
    if not candidates:
        raise BlindJudgeError("no formal retrieved-policy result file was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def matching_blind_path(results_path: Path) -> Path:
    suffix = results_path.stem.removeprefix("agent-retrieved-policy-")
    path = results_path.with_name(f"agent-retrieved-policy-blind-{suffix}.jsonl")
    if not path.exists():
        raise BlindJudgeError(f"matching combined blind file does not exist: {path}")
    return path


def load_anonymous_rows(path: Path, *, expected_ids: set[str]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 72:
        raise BlindJudgeError(f"combined blind review must contain 72 rows, found {len(rows)}")
    ids = [str(row.get("blind_id", "")).strip() for row in rows]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise BlindJudgeError("combined blind review contains missing or duplicate blind_id")
    if set(ids) != expected_ids:
        raise BlindJudgeError("combined blind ids do not match final result records")
    forbidden_keys = {"group", "repetition", "record_source", "selected_policy_rule_ids"}
    if any(forbidden_keys.intersection(row) for row in rows):
        raise BlindJudgeError("combined blind review leaks runtime identity")
    required = {"blind_id", "case_id", "category", "source", "review_focus", "translation"}
    if any(not required.issubset(row) for row in rows):
        raise BlindJudgeError("combined blind review is missing evaluation fields")
    return rows


def build_judge_payload_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Strip prior score placeholders and preserve only anonymous evidence."""

    return [
        {
            "blind_id": str(row["blind_id"]),
            "source": str(row["source"]),
            "review_focus": str(row["review_focus"]),
            "translation": str(row["translation"]),
        }
        for row in rows
    ]


def build_judge_user(rows: list[dict[str, Any]]) -> str:
    return (
        _JUDGE_SCHEMA
        + "\n\nAnonymous translations to score:\n"
        + json.dumps(build_judge_payload_rows(rows), ensure_ascii=False, indent=2)
    )


def _score(value: Any, *, blind_id: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlindJudgeError(f"{blind_id} has non-numeric {field}")
    score = float(value)
    if not 0 <= score <= 5 or score * 2 != int(score * 2):
        raise BlindJudgeError(f"{blind_id} has invalid {field}: {score}")
    return score


def validate_judge_reviews(raw: dict[str, Any], *, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    reviews = raw.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != len(expected_ids):
        raise BlindJudgeError(
            f"judge must return {len(expected_ids)} reviews, found "
            f"{len(reviews) if isinstance(reviews, list) else 'non-list'}"
        )
    result: dict[str, dict[str, Any]] = {}
    for row in reviews:
        if not isinstance(row, dict):
            raise BlindJudgeError("each judge review must be an object")
        blind_id = str(row.get("blind_id", "")).strip()
        if not blind_id or blind_id in result:
            raise BlindJudgeError("judge returned a missing or duplicate blind_id")
        result[blind_id] = {
            "semantic_fidelity_0_to_5": _score(
                row.get("semantic_fidelity_0_to_5"), blind_id=blind_id, field="semantic score"
            ),
            "naturalness_0_to_5": _score(
                row.get("naturalness_0_to_5"), blind_id=blind_id, field="naturalness score"
            ),
            "meaning_error": str(row.get("meaning_error", "")).strip(),
            "review_notes": str(row.get("review_notes", "")).strip(),
        }
    missing = expected_ids - set(result)
    unknown = set(result) - expected_ids
    if missing or unknown:
        raise BlindJudgeError(
            f"judge coverage mismatch: missing={len(missing)}, unknown={len(unknown)}"
        )
    return result


def make_dry_reviews(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reviews": [
            {
                "blind_id": row["blind_id"],
                "semantic_fidelity_0_to_5": 4.0,
                "naturalness_0_to_5": 4.0,
                "meaning_error": "",
                "review_notes": "dry-run fixture",
            }
            for row in rows
        ]
    }


def write_scored_review(
    path: Path,
    *,
    anonymous_rows: list[dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    judge_model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in anonymous_rows:
            scored = dict(row)
            scored.update(reviews[str(row["blind_id"])])
            scored["reviewer"] = judge_model
            scored["review_was_group_blind"] = True
            handle.write(json.dumps(scored, ensure_ascii=False) + "\n")


def _default_paths() -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("logs/poc")
    return (
        root / f"agent-retrieved-policy-blind-scored-{stamp}.jsonl",
        root / f"agent-retrieved-policy-review-summary-{stamp}.json",
        root / f"agent-retrieved-policy-judge-audit-{stamp}.json",
    )


def _sum_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    total = {
        field: sum(int(usage.get(field, 0) or 0) for usage in usages)
        for field in fields
    }
    total["reasoning_tokens"] = sum(
        int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
        for usage in usages
    )
    return total


def judge_anonymous_batches(
    rows: list[dict[str, Any]],
    *,
    judge_model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    batch_size: int,
    dry_run: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Judge bounded batches and recursively split a failed/empty batch."""

    if batch_size < 4:
        raise BlindJudgeError("batch size must be at least 4")
    batches = [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]
    all_reviews: dict[str, dict[str, Any]] = {}
    audit_batches: list[dict[str, Any]] = []

    def score_batch(batch: list[dict[str, Any]], label: str) -> None:
        user = build_judge_user(batch)
        payload = {
            "model": judge_model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            "max_tokens": 16384,
            "thinking": {"type": "enabled"},
            "response_format": {"type": "json_object"},
        }
        started = time.perf_counter()
        try:
            if dry_run:
                raw_response = json.dumps(make_dry_reviews(batch), ensure_ascii=False)
                usage: dict[str, Any] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            else:
                raw_response, usage = _send_request(
                    base_url=base_url,
                    api_key=api_key,
                    payload=payload,
                    timeout_seconds=timeout,
                )
            reviews = validate_judge_reviews(
                extract_json_object(raw_response),
                expected_ids={str(row["blind_id"]) for row in batch},
            )
        except (BlindJudgeError, TranslationError, json.JSONDecodeError) as exc:
            if len(batch) <= 4:
                raise BlindJudgeError(
                    f"judge batch {label} failed at minimum safe size: {exc}"
                ) from exc
            midpoint = len(batch) // 2
            print(
                f"Judge batch {label} failed ({exc}); retrying as "
                f"{len(batch[:midpoint])}+{len(batch[midpoint:])} rows"
            )
            score_batch(batch[:midpoint], label + "a")
            score_batch(batch[midpoint:], label + "b")
            return
        elapsed = round(time.perf_counter() - started, 3)
        overlap = set(all_reviews).intersection(reviews)
        if overlap:
            raise BlindJudgeError(f"judge batches returned duplicate ids: {sorted(overlap)}")
        all_reviews.update(reviews)
        audit_batches.append(
            {
                "batch": label,
                "rows": len(batch),
                "input_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
                "latency_s": elapsed,
                "usage": usage,
                "response": raw_response,
            }
        )
        print(f"Judge batch {label} complete: rows={len(batch)} elapsed={elapsed:.3f}s")

    for index, batch in enumerate(batches, 1):
        score_batch(batch, str(index))
    expected = {str(row["blind_id"]) for row in rows}
    if set(all_reviews) != expected:
        raise BlindJudgeError("batched judge did not cover every blind row")
    return all_reviews, audit_batches


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    results_path = Path(args.results) if args.results else find_latest_final_results()
    blind_path = Path(args.blind) if args.blind else matching_blind_path(results_path)
    mapping = load_result_mapping(results_path)
    anonymous_rows = load_anonymous_rows(blind_path, expected_ids=set(mapping))

    settings = AppSettings.load()
    judge_model = args.judge_model.strip() or settings.ai.thinking_model_name or "deepseek-v4-pro"
    if not args.dry_run and (not settings.ai.base_url or not settings.ai.api_key):
        raise BlindJudgeError("API base URL and key must be configured")

    started = time.perf_counter()
    validated, audit_batches = judge_anonymous_batches(
        anonymous_rows,
        judge_model=judge_model,
        base_url=settings.ai.base_url,
        api_key=settings.ai.api_key,
        timeout=args.timeout,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    latency = round(time.perf_counter() - started, 3)
    usage = _sum_usage([batch["usage"] for batch in audit_batches])

    default_scored, default_summary, default_audit = _default_paths()
    scored_path = Path(args.scored_output) if args.scored_output else default_scored
    summary_path = Path(args.summary_output) if args.summary_output else default_summary
    audit_path = Path(args.audit_output) if args.audit_output else default_audit
    write_scored_review(
        scored_path,
        anonymous_rows=anonymous_rows,
        reviews=validated,
        judge_model=judge_model,
    )
    summary = aggregate_reviews(
        mapping,
        load_scored_reviews(scored_path),
        groups=GROUPS,
    )
    summary["judge"] = {
        "model": judge_model,
        "thinking": "enabled",
        "group_blind": True,
        "reviewed": len(validated),
        "api_calls": len(audit_batches),
        "requested_batch_size": args.batch_size,
        "latency_s": latency,
        "usage": usage,
        "source_results": str(results_path.resolve()),
        "source_blind": str(blind_path.resolve()),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "judge_model": judge_model,
            "group_blind": True,
            "contains_reasoning_content": False,
            "review_count": len(validated),
            "api_calls": len(audit_batches),
            "requested_batch_size": args.batch_size,
            "latency_s": latency,
            "usage": usage,
        },
        "batches": audit_batches,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Scored blind review: {scored_path.resolve()}")
    print(f"Review summary: {summary_path.resolve()}")
    print(f"Judge audit: {audit_path.resolve()}")
    return scored_path, summary_path, audit_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="")
    parser.add_argument("--blind", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--scored-output", default="")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--audit-output", default="")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (BlindJudgeError, OSError, json.JSONDecodeError, TranslationError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
