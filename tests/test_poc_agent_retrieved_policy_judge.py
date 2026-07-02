"""Tests for the anonymous Pro blind judge."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from poc_agent_retrieved_policy_judge import (
    BlindJudgeError,
    build_judge_payload_rows,
    load_anonymous_rows,
    run,
    validate_judge_reviews,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "logs" / "poc" / "agent-retrieved-policy-20260702-213926.jsonl"
BLIND = ROOT / "logs" / "poc" / "agent-retrieved-policy-blind-20260702-213926.jsonl"


class BlindJudgeValidationTests(unittest.TestCase):
    def test_judge_payload_excludes_identity_and_existing_scores(self) -> None:
        rows = [
            {
                "blind_id": "a",
                "source": "中文",
                "translation": "ほんやく",
                "review_focus": "meaning",
                "group": "SECRET",
                "semantic_fidelity_0_to_5": None,
            }
        ]
        payload = build_judge_payload_rows(rows)
        self.assertEqual(
            set(payload[0]),
            {"blind_id", "source", "translation", "review_focus"},
        )

    def test_validator_requires_complete_exact_coverage(self) -> None:
        raw = {
            "reviews": [
                {
                    "blind_id": "a",
                    "semantic_fidelity_0_to_5": 4.5,
                    "naturalness_0_to_5": 4.0,
                    "meaning_error": "",
                    "review_notes": "ok",
                }
            ]
        }
        self.assertIn("a", validate_judge_reviews(raw, expected_ids={"a"}))
        with self.assertRaises(BlindJudgeError):
            validate_judge_reviews(raw, expected_ids={"a", "b"})

    def test_validator_rejects_non_half_step_score(self) -> None:
        raw = {
            "reviews": [
                {
                    "blind_id": "a",
                    "semantic_fidelity_0_to_5": 4.2,
                    "naturalness_0_to_5": 4,
                }
            ]
        }
        with self.assertRaisesRegex(BlindJudgeError, "invalid"):
            validate_judge_reviews(raw, expected_ids={"a"})

    def test_loader_rejects_identity_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind.jsonl"
            row = {
                "blind_id": "a",
                "case_id": "c",
                "category": "x",
                "source": "s",
                "review_focus": "f",
                "translation": "t",
                "group": "leak",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(BlindJudgeError):
                load_anonymous_rows(path, expected_ids={"a"})


@unittest.skipUnless(RESULTS.exists() and BLIND.exists(), "formal combined blind evidence unavailable")
class BlindJudgeDryRunTests(unittest.TestCase):
    def test_dry_run_scores_and_aggregates_all_72_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                results=str(RESULTS),
                blind=str(BLIND),
                judge_model="deepseek-v4-pro",
                scored_output=str(root / "scored.jsonl"),
                summary_output=str(root / "summary.json"),
                audit_output=str(root / "audit.json"),
                timeout=300.0,
                batch_size=24,
                dry_run=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                scored, summary, audit = run(args)
            rows = [json.loads(line) for line in scored.read_text(encoding="utf-8").splitlines()]
            aggregated = json.loads(summary.read_text(encoding="utf-8"))
            audit_data = json.loads(audit.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 72)
        self.assertTrue(all(row["review_was_group_blind"] for row in rows))
        self.assertEqual(aggregated["evidence"]["scored_review_records"], 72)
        self.assertEqual(set(aggregated["groups"]), {"RAW_BRIEF", "PRO_PLAYBOOK", "RETRIEVED_POLICY"})
        self.assertFalse(audit_data["metadata"]["contains_reasoning_content"])
        self.assertEqual(audit_data["metadata"]["api_calls"], 3)


if __name__ == "__main__":
    unittest.main()
