"""Offline tests for the POC 6.1 typed-reference canary."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from poc.poc_agent_memory import make_dry_run_memory
from poc.poc_agent_memory_canary import (
    CANARY_CASE_IDS,
    build_typed_agent_request,
    evaluate_strict_constraints,
    normalize_translation,
    select_canary_cases,
    semantic_type,
    write_blind_review,
)
from poc.poc_reference_injection import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "poc" / "data" / "reference_poc_dataset.json"


class TypedReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset(DATASET_PATH, strict=True)
        self.entries = self.dataset.glossary_sets["200"]
        self.memory = make_dry_run_memory(self.dataset)

    def test_canary_uses_exactly_eight_difficult_cases(self) -> None:
        cases = select_canary_cases(self.dataset)

        self.assertEqual(tuple(case.id for case in cases), CANARY_CASE_IDS)
        self.assertEqual(len(cases), 8)

    def test_semantic_type_distinguishes_action_from_noun(self) -> None:
        by_id = {entry.id: entry for entry in self.entries}

        action, action_hint = semantic_type(by_id["se_deploy"])
        noun, noun_hint = semantic_type(by_id["se_database"])

        self.assertEqual(action, "action_concept")
        self.assertIn("される", action_hint)
        self.assertEqual(noun, "noun_concept")
        self.assertIn("particle", noun_hint)

    def test_typed_request_preserves_source_meaning_and_grammar_role(self) -> None:
        case = next(case for case in self.dataset.cases if case.id == "terminology_04")
        built = build_typed_agent_request(
            self.dataset,
            self.memory,
            case,
            self.entries,
        )
        current = built["messages"][-1]["content"]

        self.assertIn("source_meaning: 数据库", current)
        self.assertIn("source_meaning: 部署", current)
        self.assertIn("semantic_type: action_concept", current)
        self.assertIn("⟦REF_0⟧", current)
        self.assertNotIn("reference_glossary", current)
        self.assertLess(built["estimated_input_tokens"], 2_000)

    def test_ocr_canary_receives_only_controlled_fuzzy_candidates(self) -> None:
        case = next(case for case in self.dataset.cases if case.id == "ocr_noise_04")
        built = build_typed_agent_request(
            self.dataset,
            self.memory,
            case,
            self.entries,
        )

        self.assertEqual(
            [item["entry_id"] for item in built["fuzzy_candidates"]],
            ["se_database", "se_server"],
        )
        self.assertIn("semantic_type=noun_concept", built["messages"][-1]["content"])


class SafeFormattingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset(DATASET_PATH, strict=True)
        self.case = next(case for case in self.dataset.cases if case.id == "terminology_01")

    def test_normalizer_preserves_content_and_only_unwraps_unauthorized_terms(self) -> None:
        result = normalize_translation(
            "この [えいが]は、  ⟦REF_0⟧を ひらく。",
            {"⟦REF_0⟧": "[そふとうぇあ]"},
            allowed_bracket_targets=["[そふとうぇあ]"],
        )

        self.assertEqual(
            result["translation"],
            "この  えいがは  [そふとうぇあ]を  ひらく",
        )
        self.assertEqual(result["unauthorized_brackets_unwrapped"], ["[えいが]"])
        self.assertTrue(result["punctuation_removed"])
        self.assertTrue(result["changed"])

    def test_strict_evaluator_rejects_unauthorized_brackets(self) -> None:
        result = evaluate_strict_constraints(
            self.dataset,
            self.case,
            "[ぷろぐらむ]を  [こんぱいる]する  [えいが]",
            allowed_bracket_targets=["[ぷろぐらむ]", "[こんぱいる]"],
        )

        self.assertFalse(result["checks"]["authorized_brackets_only"])
        self.assertEqual(result["unauthorized_brackets"], ["[えいが]"])
        self.assertFalse(result["ok"])

    def test_strict_evaluator_accepts_authorized_normalized_output(self) -> None:
        result = evaluate_strict_constraints(
            self.dataset,
            self.case,
            "[そふとうぇあ]を  ひらいて  [かいはつ]を  はじめる",
            allowed_bracket_targets=["[そふとうぇあ]", "[かいはつ]"],
        )

        self.assertTrue(result["ok"], result)


class CanaryBlindReviewTests(unittest.TestCase):
    def test_blind_review_hides_group_repetition_and_normalization(self) -> None:
        record = {
            "blind_id": "blind",
            "case_id": "c1",
            "category": "semantic",
            "source": "原文",
            "final_translation": "ほんやく",
            "group": "AGENT_V61_TYPED",
            "repetition": 2,
            "normalization": {"changed": True},
            "expected_fixed_outputs": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind.jsonl"
            write_blind_review(path, [record], seed=1)
            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("group", row)
        self.assertNotIn("repetition", row)
        self.assertNotIn("normalization", row)
        self.assertEqual(row["translation"], "ほんやく")


if __name__ == "__main__":
    unittest.main()
