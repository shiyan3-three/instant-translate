"""Validate persisted blind scores and reveal Agent Bootstrap group statistics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from poc_agent_bootstrap import GROUPS


class ReviewError(ValueError):
    """Raised when blind-review evidence is missing or malformed."""


def _score(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None:
        raise ReviewError(f"blind_id {row.get('blind_id')} has no {field} score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewError(f"blind_id {row.get('blind_id')} has non-numeric {field}")
    score = float(value)
    if not 0 <= score <= 5:
        raise ReviewError(f"blind_id {row.get('blind_id')} has out-of-range {field}: {score}")
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
            raise ReviewError("result file has missing or duplicate blind_id")
        hard_evaluation = row.get("hard_constraint_evaluation") or {}
        fallback_evaluation = row.get("evaluation") or {}
        mapping[blind_id] = {
            "group": row["group"],
            "case_id": row["case_id"],
            "category": row["category"],
            "hard_constraint_ok": bool(
                hard_evaluation.get(
                    "ok",
                    fallback_evaluation.get("machine_ok", False),
                )
            ),
        }
    if not mapping:
        raise ReviewError("result file contains no completed result records")
    return mapping


def load_scored_reviews(path: Path) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        blind_id = str(row.get("blind_id", "")).strip()
        if not blind_id or blind_id in reviews:
            raise ReviewError("blind-review file has missing or duplicate blind_id")
        row["semantic_score"] = _score(row, "semantic_fidelity_0_to_5")
        row["naturalness_score"] = _score(row, "naturalness_0_to_5")
        reviews[blind_id] = row
    if not reviews:
        raise ReviewError("blind-review file is empty")
    return reviews


def aggregate_reviews(
    mapping: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    *,
    groups: tuple[str, ...] = GROUPS,
) -> dict[str, Any]:
    missing = sorted(set(mapping) - set(reviews))
    unknown = sorted(set(reviews) - set(mapping))
    if missing or unknown:
        raise ReviewError(
            f"review coverage mismatch: missing={len(missing)}, unknown={len(unknown)}"
        )

    unknown_groups = sorted({item["group"] for item in mapping.values()} - set(groups))
    if unknown_groups:
        raise ReviewError(f"result file contains unknown groups: {', '.join(unknown_groups)}")

    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in groups}
    category_grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for blind_id, review in reviews.items():
        identity = mapping[blind_id]
        enriched = {**review, **identity}
        grouped[identity["group"]].append(enriched)
        category_grouped.setdefault(identity["category"], {group: [] for group in groups})[
            identity["group"]
        ].append(enriched)

    def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"reviewed": 0}
        semantics = [row["semantic_score"] for row in rows]
        naturalness = [row["naturalness_score"] for row in rows]
        hard_rows = [row for row in rows if row["hard_constraint_ok"]]
        return {
            "reviewed": len(rows),
            "semantic_mean": round(statistics.fmean(semantics), 3),
            "naturalness_mean": round(statistics.fmean(naturalness), 3),
            "combined_mean_0_to_10": round(
                statistics.fmean(s + n for s, n in zip(semantics, naturalness)), 3
            ),
            "full_score_count": sum(s == 5 and n == 5 for s, n in zip(semantics, naturalness)),
            "major_semantic_error_count": sum(s <= 2 for s in semantics),
            "hard_constraint_pass_count": len(hard_rows),
            "hard_constraint_pass_rate": round(len(hard_rows) / len(rows), 3),
            "hard_pass_combined_mean_0_to_10": round(
                statistics.fmean(
                    row["semantic_score"] + row["naturalness_score"]
                    for row in hard_rows
                ),
                3,
            ) if hard_rows else None,
        }

    return {
        "type": "blind_review_summary",
        "evidence": {
            "result_records": len(mapping),
            "scored_review_records": len(reviews),
            "all_scores_persisted": True,
        },
        "groups": {group: stats(rows) for group, rows in grouped.items()},
        "categories": {
            category: {group: stats(rows) for group, rows in groups.items()}
            for category, groups in sorted(category_grouped.items())
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="complete Agent Bootstrap JSONL")
    parser.add_argument("--review", required=True, help="filled blind-review JSONL")
    parser.add_argument("--output", default="", help="optional summary JSON path")
    parser.add_argument(
        "--groups",
        default=",".join(GROUPS),
        help="comma-separated group names expected in the result file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
        if not groups:
            raise ReviewError("--groups must contain at least one name")
        summary = aggregate_reviews(
            load_result_mapping(Path(args.results)),
            load_scored_reviews(Path(args.review)),
            groups=groups,
        )
    except (ReviewError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
