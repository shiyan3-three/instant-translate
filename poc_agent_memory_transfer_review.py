"""Validate blind scores and decide the feedback-memory architecture."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from poc_agent_memory_transfer import GROUPS


class TransferReviewError(ValueError):
    """Raised when blind-review evidence is missing or malformed."""


def _score(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None:
        raise TransferReviewError(f"blind_id {row.get('blind_id')} has no {field}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransferReviewError(f"blind_id {row.get('blind_id')} has invalid {field}")
    score = float(value)
    if not 0 <= score <= 5:
        raise TransferReviewError(
            f"blind_id {row.get('blind_id')} has out-of-range {field}: {score}"
        )
    return score


def load_result_mapping(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") != "result":
            continue
        blind_id = str(row.get("blind_id", "")).strip()
        if not blind_id or blind_id in mapping:
            raise TransferReviewError("results have missing or duplicate blind_id")
        mapping[blind_id] = {
            "group": row["group"],
            "episode_id": row["episode_id"],
            "provenance": row["provenance"],
            "transfer_contract_ok": bool((row.get("transfer_evaluation") or {}).get("ok")),
            "format_ok": bool((row.get("format_evaluation") or {}).get("ok")),
        }
    if not mapping:
        raise TransferReviewError("results contain no result records")
    return mapping


def load_scored_reviews(path: Path) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        blind_id = str(row.get("blind_id", "")).strip()
        if not blind_id or blind_id in reviews:
            raise TransferReviewError("reviews have missing or duplicate blind_id")
        row["semantic_score"] = _score(row, "semantic_fidelity_0_to_5")
        row["naturalness_score"] = _score(row, "naturalness_0_to_5")
        row["preference_score"] = _score(row, "preference_transfer_0_to_5")
        reviews[blind_id] = row
    if not reviews:
        raise TransferReviewError("blind review is empty")
    return reviews


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"reviewed": 0}
    semantic = [row["semantic_score"] for row in rows]
    naturalness = [row["naturalness_score"] for row in rows]
    preference = [row["preference_score"] for row in rows]
    return {
        "reviewed": len(rows),
        "semantic_mean": round(statistics.fmean(semantic), 3),
        "naturalness_mean": round(statistics.fmean(naturalness), 3),
        "preference_transfer_mean": round(statistics.fmean(preference), 3),
        "combined_mean_0_to_15": round(
            statistics.fmean(
                s + n + p for s, n, p in zip(semantic, naturalness, preference)
            ),
            3,
        ),
        "major_semantic_error_count": sum(score <= 2 for score in semantic),
        "transfer_contract_pass": sum(row["transfer_contract_ok"] for row in rows),
        "format_pass": sum(row["format_ok"] for row in rows),
    }


def _decision(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stateless = groups["STATELESS"]
    raw = groups["RAW_SESSION"]
    compiled = groups["COMPILED_MEMORY"]
    compiled_gain = (
        compiled["preference_transfer_mean"] - stateless["preference_transfer_mean"]
    )
    raw_gain = raw["preference_transfer_mean"] - stateless["preference_transfer_mean"]
    compiled_quality_ok = (
        compiled["semantic_mean"] >= raw["semantic_mean"] - 0.25
        and compiled["naturalness_mean"] >= raw["naturalness_mean"] - 0.25
        and compiled["major_semantic_error_count"] == 0
    )
    if compiled_gain >= 0.75 and compiled_quality_ok:
        route = "COMPILED_MEMORY"
        reason = "Pro-compiled memory transfers preferences without meaningful quality regression."
    elif raw_gain >= 0.75 and raw["major_semantic_error_count"] == 0:
        route = "RAW_SESSION"
        reason = "Visible correction history transfers preferences more reliably than compiled memory."
    else:
        route = "NO_RELIABLE_FLASH_TRANSFER"
        reason = "Neither memory path shows the predeclared transfer advantage over stateless Flash."
    return {
        "recommended_route": route,
        "compiled_preference_gain_vs_stateless": round(compiled_gain, 3),
        "raw_preference_gain_vs_stateless": round(raw_gain, 3),
        "compiled_quality_guard_pass": compiled_quality_ok,
        "reason": reason,
    }


def aggregate_reviews(
    mapping: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing = sorted(set(mapping) - set(reviews))
    unknown = sorted(set(reviews) - set(mapping))
    if missing or unknown:
        raise TransferReviewError(
            f"review coverage mismatch: missing={len(missing)}, unknown={len(unknown)}"
        )
    unknown_groups = sorted({row["group"] for row in mapping.values()} - set(GROUPS))
    if unknown_groups:
        raise TransferReviewError("unknown groups: " + ", ".join(unknown_groups))

    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUPS}
    episodes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    provenance: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for blind_id, review in reviews.items():
        identity = mapping[blind_id]
        row = {**review, **identity}
        grouped[identity["group"]].append(row)
        episodes.setdefault(
            identity["episode_id"], {group: [] for group in GROUPS}
        )[identity["group"]].append(row)
        provenance.setdefault(
            identity["provenance"], {group: [] for group in GROUPS}
        )[identity["group"]].append(row)

    group_stats = {group: _stats(rows) for group, rows in grouped.items()}
    return {
        "type": "feedback_memory_transfer_review",
        "evidence": {
            "result_records": len(mapping),
            "scored_review_records": len(reviews),
            "all_scores_persisted": True,
        },
        "groups": group_stats,
        "episodes": {
            episode_id: {group: _stats(rows) for group, rows in group_rows.items()}
            for episode_id, group_rows in sorted(episodes.items())
        },
        "provenance": {
            source: {group: _stats(rows) for group, rows in group_rows.items()}
            for source, group_rows in sorted(provenance.items())
        },
        "decision": _decision(group_stats),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = aggregate_reviews(
            load_result_mapping(Path(args.results)),
            load_scored_reviews(Path(args.review)),
        )
    except (TransferReviewError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
