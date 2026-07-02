"""Offline tests for the Pro-bootstrap/Flash-session POC."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from poc_agent_bootstrap import (
    BootstrapStateError,
    _load_state,
    _write_blind_review,
    build_bootstrap_source,
    build_group_request,
    find_fuzzy_candidates,
    extract_json_object,
    make_agent_dataset,
    make_dry_run_state,
    summarize,
    validate_bootstrap_state,
)
from poc_reference_injection import evaluate_response, load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "poc_data" / "reference_poc_dataset.json"


class BootstrapStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset(DATASET_PATH, strict=True)

    def test_extracts_fenced_json_state(self) -> None:
        state = make_dry_run_state(self.dataset)
        response = "```json\n" + json.dumps(state, ensure_ascii=False) + "\n```"

        parsed = validate_bootstrap_state(extract_json_object(response))

        self.assertEqual(parsed["version"], 2)
        self.assertTrue(parsed["semantic_priorities"])

    def test_rejects_state_without_semantic_priorities(self) -> None:
        state = make_dry_run_state(self.dataset)
        state["semantic_priorities"] = []

        with self.assertRaisesRegex(BootstrapStateError, "semantic_priorities must not be empty"):
            validate_bootstrap_state(state)

    def test_bootstrap_source_does_not_leak_test_cases(self) -> None:
        source = build_bootstrap_source(self.dataset, self.dataset.glossary_sets["200"])

        self.assertIn("reference_glossary", source)
        self.assertIn(self.dataset.raw_user_constraints, source)
        self.assertNotIn(self.dataset.cases[0].source, source)

    def test_rejects_persisted_state_from_another_dataset(self) -> None:
        state = make_dry_run_state(self.dataset)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(
                json.dumps(
                    {
                        "metadata": {"dataset_sha256": "old-hash"},
                        "state": state,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(BootstrapStateError, "different dataset"):
                _load_state(path, expected_dataset_sha256="new-hash")


class GroupRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset(DATASET_PATH, strict=True)
        self.state = make_dry_run_state(self.dataset)
        self.agent_dataset = make_agent_dataset(self.dataset, self.state)
        self.entries = self.dataset.glossary_sets["200"]
        self.case = next(case for case in self.dataset.cases if case.id == "terminology_01")

    def test_three_groups_have_the_intended_context_structure(self) -> None:
        current = build_group_request(
            self.dataset, self.agent_dataset, self.case, self.entries, "CURRENT_FULL"
        )
        retrieval = build_group_request(
            self.dataset, self.agent_dataset, self.case, self.entries, "RETRIEVAL_RAW"
        )
        agent = build_group_request(
            self.dataset, self.agent_dataset, self.case, self.entries, "AGENT_BOOTSTRAP"
        )

        self.assertIn("misc_theme", current["messages"][0]["content"])
        self.assertIn("PLACEHOLDER_PROTOCOL", retrieval["messages"][1]["content"])
        self.assertNotIn("Pro-initialized persistent session", retrieval["messages"][0]["content"])
        self.assertIn("Pro-initialized persistent session", agent["messages"][0]["content"])
        self.assertIn("PLACEHOLDER_PROTOCOL", agent["messages"][1]["content"])
        self.assertNotIn("[そふとうぇあ]", retrieval["messages"][1]["content"])

    def test_fuzzy_candidates_recover_ocr_damaged_glossary_surfaces(self) -> None:
        candidates = find_fuzzy_candidates("数捗库部署在远程服器上", self.entries)
        ids = {item.entry_id for item in candidates}

        self.assertIn("se_database", ids)
        self.assertIn("se_server", ids)

    def test_clean_nonterminology_request_omits_placeholder_protocol(self) -> None:
        case = next(case for case in self.dataset.cases if case.id == "semantic_01")
        request = build_group_request(
            self.dataset,
            self.agent_dataset,
            case,
            self.entries,
            "AGENT_BOOTSTRAP",
        )

        self.assertNotIn("PLACEHOLDER_PROTOCOL", request["messages"][1]["content"])

    def test_format_check_allows_natural_word_order_with_double_spaces(self) -> None:
        case = next(case for case in self.dataset.cases if case.id == "format_02")
        natural = "[ぷろぐらむ]を  [こんぱいる]する"

        result = evaluate_response(self.dataset, case, self.entries, natural, {})

        self.assertTrue(result["machine_ok"], result)

    def test_format_check_rejects_single_space(self) -> None:
        case = next(case for case in self.dataset.cases if case.id == "format_02")
        result = evaluate_response(
            self.dataset,
            case,
            self.entries,
            "[ぷろぐらむ]を [こんぱいる]する",
            {},
        )

        self.assertFalse(result["machine_ok"])


class ReportingTests(unittest.TestCase):
    def test_summary_does_not_claim_semantic_quality(self) -> None:
        summary = summarize([])

        for group in summary["groups"].values():
            self.assertEqual(group["semantic_fidelity"], "NOT_SCORED_USE_BLIND_REVIEW")
            self.assertNotIn("quality_pass", group)

    def test_blind_export_does_not_expose_group_identity(self) -> None:
        record = {
            "blind_id": "abc123",
            "group": "AGENT_BOOTSTRAP",
            "case_id": "semantic_01",
            "category": "semantic",
            "source": "原文",
            "response": "译文",
            "evaluation": {"restored_response": "ほんやく"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind.jsonl"
            _write_blind_review(path, [record], seed=1)
            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("group", row)
        self.assertEqual(row["translation"], "ほんやく")
        self.assertIsNone(row["semantic_fidelity_0_to_5"])


if __name__ == "__main__":
    unittest.main()
