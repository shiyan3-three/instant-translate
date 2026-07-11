"""Tests for auditable Agent Bootstrap blind-review aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from poc.poc_agent_bootstrap_review import ReviewError, aggregate_reviews, load_result_mapping


class ReviewAggregationTests(unittest.TestCase):
    def test_incomplete_result_maps_to_failed_hard_constraint_without_crashing(self) -> None:
        row = {
            "type": "result",
            "blind_id": "a",
            "group": "AGENT_MEMORY",
            "case_id": "c1",
            "category": "semantic",
            "hard_constraint_evaluation": None,
            "evaluation": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            mapping = load_result_mapping(path)

        self.assertFalse(mapping["a"]["hard_constraint_ok"])

    def test_requires_every_score_to_be_persisted(self) -> None:
        mapping = {
            "a": {
                "group": "CURRENT_FULL",
                "case_id": "c1",
                "category": "semantic",
                "hard_constraint_ok": True,
            }
        }

        with self.assertRaisesRegex(ReviewError, "coverage mismatch"):
            aggregate_reviews(mapping, {})

    def test_aggregates_only_after_blind_ids_are_joined(self) -> None:
        mapping = {
            "a": {
                "group": "CURRENT_FULL",
                "case_id": "c1",
                "category": "semantic",
                "hard_constraint_ok": True,
            },
            "b": {
                "group": "AGENT_BOOTSTRAP",
                "case_id": "c1",
                "category": "semantic",
                "hard_constraint_ok": False,
            },
        }
        reviews = {
            "a": {"semantic_score": 5.0, "naturalness_score": 4.0},
            "b": {"semantic_score": 2.0, "naturalness_score": 3.0},
        }

        result = aggregate_reviews(mapping, reviews)

        self.assertTrue(result["evidence"]["all_scores_persisted"])
        self.assertEqual(result["groups"]["CURRENT_FULL"]["combined_mean_0_to_10"], 9.0)
        self.assertEqual(
            result["groups"]["AGENT_BOOTSTRAP"]["major_semantic_error_count"], 1
        )
        self.assertEqual(result["groups"]["CURRENT_FULL"]["hard_constraint_pass_count"], 1)
        self.assertEqual(result["groups"]["AGENT_BOOTSTRAP"]["hard_constraint_pass_count"], 0)


if __name__ == "__main__":
    unittest.main()
