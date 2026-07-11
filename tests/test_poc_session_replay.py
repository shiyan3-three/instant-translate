"""Offline tests for visible Pro-content session replay."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from poc.poc_reference_injection import load_dataset
from poc.poc_session_replay import (
    _DRY_PRO_CONTENT,
    SessionReplayError,
    build_rule_package,
    build_runtime_messages,
    build_transcript,
    evaluate_hard_constraints,
    load_transcript,
    save_transcript,
    summarize,
    write_blind_review,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "poc" / "data" / "reference_poc_dataset.json"


class SessionMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset(DATASET_PATH, strict=True)
        self.package = build_rule_package(self.dataset, self.dataset.glossary_sets["200"])
        self.transcript = build_transcript(self.dataset, self.package, _DRY_PRO_CONTENT)
        self.case = self.dataset.cases[0]

    def test_rule_package_contains_no_test_case(self) -> None:
        self.assertIn("reference_glossary", self.package)
        self.assertNotIn(self.case.source, self.package)

    def test_groups_preserve_expected_role_sequences(self) -> None:
        direct = build_runtime_messages(
            self.dataset, self.package, self.transcript, self.case, "DIRECT"
        )
        neutral = build_runtime_messages(
            self.dataset, self.package, self.transcript, self.case, "NEUTRAL_SESSION"
        )
        pro = build_runtime_messages(
            self.dataset, self.package, self.transcript, self.case, "PRO_SESSION"
        )

        self.assertEqual([item["role"] for item in direct], ["system", "user"])
        self.assertEqual(
            [item["role"] for item in neutral],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(
            [item["role"] for item in pro],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(neutral[1]["content"], pro[1]["content"])
        self.assertNotEqual(neutral[2]["content"], pro[2]["content"])

    def test_persisted_transcript_explicitly_excludes_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            save_transcript(
                path,
                self.transcript,
                dataset_sha256="same",
                thinking_model="pro",
                source="test",
            )
            loaded = load_transcript(path, dataset_sha256="same")
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, self.transcript)
        self.assertFalse(raw["metadata"]["contains_reasoning_content"])
        self.assertTrue(
            all("reasoning_content" not in message for message in raw["messages"])
        )

    def test_rejects_transcript_from_different_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            save_transcript(
                path,
                self.transcript,
                dataset_sha256="old",
                thinking_model="pro",
                source="test",
            )
            with self.assertRaisesRegex(SessionReplayError, "different prompt package"):
                load_transcript(path, dataset_sha256="new")


class HardConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset(DATASET_PATH, strict=True)
        self.case = next(case for case in self.dataset.cases if case.id == "format_02")

    def test_accepts_natural_order_under_confirmed_format(self) -> None:
        result = evaluate_hard_constraints(
            self.dataset,
            self.case,
            "[ぷろぐらむ]を  [こんぱいる]する",
        )

        self.assertTrue(result["ok"], result)

    def test_rejects_kanji_even_when_meaning_is_natural(self) -> None:
        result = evaluate_hard_constraints(self.dataset, self.case, "プログラムを  編译する")

        self.assertFalse(result["checks"]["hiragana_brackets_spaces_only"])
        self.assertFalse(result["ok"])

    def test_rejects_single_space_run(self) -> None:
        result = evaluate_hard_constraints(
            self.dataset,
            self.case,
            "[ぷろぐらむ]を [こんぱいる]する",
        )

        self.assertFalse(result["checks"]["all_existing_space_runs_are_exactly_two"])
        self.assertFalse(result["ok"])


class SessionReportingTests(unittest.TestCase):
    def test_blind_file_hides_group_and_repetition(self) -> None:
        record = {
            "blind_id": "blind",
            "case_id": "c1",
            "category": "semantic",
            "source": "原文",
            "response": "ほんやく",
            "group": "PRO_SESSION",
            "repetition": 2,
            "expected_fixed_outputs": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind.jsonl"
            write_blind_review(path, [record], seed=1)
            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("group", row)
        self.assertNotIn("repetition", row)
        self.assertIn("Do not reward", row["confirmed_user_constraints"])

    def test_summary_reports_exact_repeat_stability(self) -> None:
        records = []
        for repetition in (1, 2):
            records.append(
                {
                    "group": "DIRECT",
                    "case_id": "c1",
                    "response": "same",
                    "error": None,
                    "latency_s": 1.0,
                    "hard_constraint_evaluation": {
                        "ok": True,
                        "missing_expected_terms": [],
                    },
                    "usage": {"prompt_tokens": 10},
                    "estimated_input_tokens": 10,
                }
            )

        summary = summarize(records, repetitions=2)

        self.assertEqual(summary["groups"]["DIRECT"]["exact_repeat_stability"], "1/1")
        self.assertEqual(summary["groups"]["DIRECT"]["hard_constraint_pass"], 2)


if __name__ == "__main__":
    unittest.main()
