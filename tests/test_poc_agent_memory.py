"""Offline tests for POC 6 compiled persistent Agent memory."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from poc.poc_agent_memory import (
    AgentMemoryError,
    build_agent_request,
    build_memory_source,
    clone_baseline_records,
    load_agent_memory,
    load_direct_baseline,
    make_dry_run_memory,
    save_agent_memory,
    validate_agent_memory,
    write_blind_review,
)
from poc.poc_reference_injection import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "poc" / "data" / "reference_poc_dataset.json"


class AgentMemoryValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset(DATASET_PATH, strict=True)
        self.memory = make_dry_run_memory(self.dataset)

    def test_startup_source_does_not_include_unrelated_semantic_test(self) -> None:
        source = build_memory_source(
            self.dataset,
            self.dataset.glossary_sets["200"],
        )

        self.assertNotIn("请不要取消明天的预约，否则需要支付手续费", source)
        self.assertIn("reference_glossary", source)
        self.assertIn("accepted_corrections", source)

    def test_rejects_derived_demonstration_that_leaks_test_source(self) -> None:
        raw = deepcopy(self.memory)
        raw["derived_demonstrations"][0]["source"] = self.dataset.cases[0].source

        with self.assertRaisesRegex(AgentMemoryError, "leaks test source"):
            validate_agent_memory(
                raw,
                forbidden_example_sources=(case.source for case in self.dataset.cases),
            )

    def test_rejects_invalid_demonstration_target(self) -> None:
        raw = deepcopy(self.memory)
        raw["derived_demonstrations"][0]["target"] = "明日の  会議"

        with self.assertRaisesRegex(AgentMemoryError, "output contract"):
            validate_agent_memory(raw)

    def test_saved_memory_excludes_reasoning_content_from_memory_object(self) -> None:
        digest = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            save_agent_memory(
                path,
                self.memory,
                dataset_sha256=digest,
                thinking_model="pro",
                source="test",
            )
            loaded = load_agent_memory(
                path,
                dataset_sha256=digest,
                forbidden_example_sources=(case.source for case in self.dataset.cases),
            )
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, self.memory)
        self.assertFalse(raw["metadata"]["contains_reasoning_content"])
        self.assertNotIn("reasoning_content", raw["memory"])


class AgentMemoryRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset(DATASET_PATH, strict=True)
        self.entries = self.dataset.glossary_sets["200"]
        self.memory = make_dry_run_memory(self.dataset)

    def test_request_uses_visible_example_turns_and_compact_references(self) -> None:
        case = next(case for case in self.dataset.cases if case.id == "terminology_01")
        built = build_agent_request(self.dataset, self.memory, case, self.entries)
        roles = [message["role"] for message in built["messages"]]
        current_user = built["messages"][-1]["content"]

        self.assertEqual(roles[0], "system")
        self.assertEqual(roles[-1], "user")
        self.assertGreaterEqual(roles.count("assistant"), 4)
        self.assertIn("⟦REF_", current_user)
        self.assertNotIn("reference_glossary", "\n".join(
            message["content"] for message in built["messages"]
        ))
        self.assertLess(built["estimated_input_tokens"], 2_000)

    def test_fuzzy_candidates_are_limited_to_controlled_ocr_cases(self) -> None:
        clean = next(case for case in self.dataset.cases if case.id == "semantic_02")
        noisy = next(case for case in self.dataset.cases if case.id == "ocr_noise_04")

        clean_request = build_agent_request(
            self.dataset, self.memory, clean, self.entries
        )
        noisy_request = build_agent_request(
            self.dataset, self.memory, noisy, self.entries
        )

        self.assertEqual(clean_request["fuzzy_candidates"], [])
        self.assertEqual(
            [item["entry_id"] for item in noisy_request["fuzzy_candidates"]],
            ["se_database", "se_server"],
        )


class BaselineReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_dataset(DATASET_PATH, strict=True)
        self.case = self.dataset.cases[0]
        self.digest = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()

    def _write_baseline(self, path: Path, *, incomplete: bool = False) -> None:
        metadata = {
            "type": "run",
            "experiment": "visible_session_replay",
            "dataset_sha256": self.digest,
            "repetitions": 2,
            "fast_model": "flash",
        }
        rows = [metadata]
        for repetition in (1, 2):
            rows.append(
                {
                    "type": "result",
                    "group": "DIRECT",
                    "case_id": self.case.id,
                    "category": self.case.category,
                    "source": self.case.source,
                    "repetition": repetition,
                    "blind_id": f"old-{repetition}",
                    "response": "ほんやく" if not incomplete else None,
                    "error": None,
                    "hard_constraint_evaluation": {
                        "ok": True,
                        "missing_expected_terms": [],
                    },
                }
            )
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    def test_loads_complete_direct_records_and_rekeys_blind_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.jsonl"
            self._write_baseline(path)
            _, records = load_direct_baseline(
                path,
                dataset=self.dataset,
                dataset_sha256=self.digest,
                repetitions=2,
                cases=(self.case,),
            )
            cloned = clone_baseline_records(
                records,
                run_id="new-run",
                baseline_path=path,
            )

        self.assertEqual(len(cloned), 2)
        self.assertTrue(all(row["group"] == "DIRECT_BASELINE" for row in cloned))
        self.assertNotEqual(cloned[0]["blind_id"], "old-1")
        self.assertEqual(cloned[0]["baseline_original_blind_id"], "old-1")

    def test_rejects_incomplete_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.jsonl"
            self._write_baseline(path, incomplete=True)

            with self.assertRaisesRegex(AgentMemoryError, "incomplete"):
                load_direct_baseline(
                    path,
                    dataset=self.dataset,
                    dataset_sha256=self.digest,
                    repetitions=2,
                    cases=(self.case,),
                )


class AgentMemoryBlindReviewTests(unittest.TestCase):
    def test_blind_rows_hide_group_repetition_and_raw_placeholder_response(self) -> None:
        record = {
            "blind_id": "blind",
            "case_id": "c1",
            "category": "semantic",
            "source": "原文",
            "response": "⟦REF_0⟧を  ひらく",
            "evaluation": {"restored_response": "[そふとうぇあ]を  ひらく"},
            "group": "AGENT_MEMORY",
            "repetition": 2,
            "expected_fixed_outputs": ["[そふとうぇあ]"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind.jsonl"
            write_blind_review(path, [record], seed=1)
            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("group", row)
        self.assertNotIn("repetition", row)
        self.assertNotIn("⟦REF_0⟧", row["translation"])
        self.assertEqual(row["translation"], "[そふとうぇあ]を  ひらく")


if __name__ == "__main__":
    unittest.main()
