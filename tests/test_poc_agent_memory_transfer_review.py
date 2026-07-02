"""Tests for blind feedback-memory transfer aggregation and routing decision."""

from __future__ import annotations

import unittest

from poc_agent_memory_transfer_review import (
    TransferReviewError,
    aggregate_reviews,
)


def _mapping() -> dict[str, dict]:
    return {
        "s": {
            "group": "STATELESS",
            "episode_id": "e1",
            "provenance": "user_confirmed_feedback",
            "transfer_contract_ok": False,
            "format_ok": True,
        },
        "r": {
            "group": "RAW_SESSION",
            "episode_id": "e1",
            "provenance": "user_confirmed_feedback",
            "transfer_contract_ok": True,
            "format_ok": True,
        },
        "c": {
            "group": "COMPILED_MEMORY",
            "episode_id": "e1",
            "provenance": "user_confirmed_feedback",
            "transfer_contract_ok": True,
            "format_ok": True,
        },
    }


class TransferReviewTests(unittest.TestCase):
    def test_requires_complete_blind_review_coverage(self) -> None:
        with self.assertRaisesRegex(TransferReviewError, "coverage mismatch"):
            aggregate_reviews(_mapping(), {})

    def test_recommends_compiled_memory_when_transfer_gain_is_large(self) -> None:
        reviews = {
            "s": {"semantic_score": 4.0, "naturalness_score": 4.0, "preference_score": 2.0},
            "r": {"semantic_score": 4.5, "naturalness_score": 4.0, "preference_score": 4.0},
            "c": {"semantic_score": 4.5, "naturalness_score": 4.0, "preference_score": 4.5},
        }

        result = aggregate_reviews(_mapping(), reviews)

        self.assertEqual(result["decision"]["recommended_route"], "COMPILED_MEMORY")
        self.assertEqual(result["groups"]["COMPILED_MEMORY"]["transfer_contract_pass"], 1)
        self.assertTrue(result["evidence"]["all_scores_persisted"])

    def test_recommends_raw_session_when_compiled_quality_regresses(self) -> None:
        reviews = {
            "s": {"semantic_score": 4.0, "naturalness_score": 4.0, "preference_score": 2.0},
            "r": {"semantic_score": 4.5, "naturalness_score": 4.5, "preference_score": 4.5},
            "c": {"semantic_score": 2.0, "naturalness_score": 3.0, "preference_score": 4.5},
        }

        result = aggregate_reviews(_mapping(), reviews)

        self.assertEqual(result["decision"]["recommended_route"], "RAW_SESSION")


if __name__ == "__main__":
    unittest.main()
