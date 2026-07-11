"""Offline tests for the final-direction feedback-memory transfer POC."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from poc.poc_agent_memory_transfer import (
    TransferPocError,
    build_common_system,
    build_compiler_source,
    build_group_messages,
    evaluate_format,
    evaluate_transfer,
    load_compiled_memory,
    load_transfer_dataset,
    make_dry_memory,
    normalize_surface,
    save_compiled_memory,
    select_memory_rules,
    validate_compiled_memory,
    write_blind_review,
)
from poc.poc_reference_injection import load_dataset


ROOT = Path(__file__).resolve().parents[1]
TRANSFER_PATH = ROOT / "poc" / "data" / "agent_memory_transfer_dataset.json"
TRANSLATION_PATH = ROOT / "poc" / "data" / "reference_poc_dataset.json"


class TransferDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_transfer_dataset(TRANSFER_PATH)

    def test_dataset_preserves_real_and_curated_provenance(self) -> None:
        real = [
            episode
            for episode in self.dataset.episodes
            if episode.provenance == "user_confirmed_feedback"
        ]
        curated = [
            episode
            for episode in self.dataset.episodes
            if episode.provenance == "project_curated_from_poc_failure"
        ]

        self.assertEqual(len(real), 2)
        self.assertEqual(len(curated), 6)

    def test_compiler_source_hides_every_transfer_query_and_expectation(self) -> None:
        source = build_compiler_source(self.dataset)

        for episode in self.dataset.episodes:
            self.assertNotIn(episode.transfer_source, source)
            for item in episode.expectations.required_all:
                if item not in episode.accepted_translation and item not in episode.feedback:
                    self.assertNotIn(item, source)


class CompiledMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_transfer_dataset(TRANSFER_PATH)
        self.memory = make_dry_memory(self.dataset)

    def test_dry_memory_has_one_rule_per_episode(self) -> None:
        self.assertEqual(
            {rule["episode_id"] for rule in self.memory["memory_rules"]},
            {episode.id for episode in self.dataset.episodes},
        )

    def test_rejects_memory_that_leaks_unseen_transfer_query(self) -> None:
        raw = deepcopy(self.memory)
        raw["memory_rules"][0]["generalized_rule"] = self.dataset.episodes[0].transfer_source

        with self.assertRaisesRegex(TransferPocError, "leaked transfer query"):
            validate_compiled_memory(raw, dataset=self.dataset)

    def test_rejects_changed_confirmed_example(self) -> None:
        raw = deepcopy(self.memory)
        raw["memory_rules"][0]["confirmed_example"]["translation"] = "changed"

        with self.assertRaisesRegex(TransferPocError, "changed accepted translation"):
            validate_compiled_memory(raw, dataset=self.dataset)

    def test_saved_memory_explicitly_excludes_reasoning(self) -> None:
        source_hash = hashlib.sha256(b"evidence").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            save_compiled_memory(
                path,
                self.memory,
                source_hash=source_hash,
                thinking_model="pro",
                source="test",
            )
            loaded = load_compiled_memory(
                path,
                source_hash=source_hash,
                dataset=self.dataset,
            )
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, self.memory)
        self.assertFalse(raw["metadata"]["contains_reasoning_content"])
        self.assertNotIn("reasoning_content", raw["memory"])


class SessionComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transfer = load_transfer_dataset(TRANSFER_PATH)
        self.translation = load_dataset(TRANSLATION_PATH, strict=True)
        self.memory = make_dry_memory(self.transfer)
        self.common = build_common_system(
            self.translation,
            self.translation.glossary_sets["20"],
        )

    def test_groups_differ_only_by_feedback_context(self) -> None:
        episode = self.transfer.episodes[0]
        stateless, stateless_ids = build_group_messages(
            self.common, episode, self.memory, "STATELESS"
        )
        raw, raw_ids = build_group_messages(
            self.common, episode, self.memory, "RAW_SESSION"
        )
        compiled, compiled_ids = build_group_messages(
            self.common, episode, self.memory, "COMPILED_MEMORY"
        )

        self.assertEqual([item["role"] for item in stateless], ["system", "user"])
        self.assertEqual(
            [item["role"] for item in raw],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )
        self.assertEqual([item["role"] for item in compiled], ["system", "user"])
        self.assertEqual(stateless[-1]["content"], compiled[-1]["content"])
        self.assertEqual(raw[-1]["content"], compiled[-1]["content"])
        self.assertEqual(stateless_ids, [])
        self.assertEqual(raw_ids, [episode.id])
        self.assertEqual(compiled_ids, [episode.id])
        self.assertIn(episode.transfer_source, compiled[-1]["content"])
        self.assertIn("PERSISTED_MEMORY", compiled[0]["content"])

    def test_trigger_selection_uses_source_text_not_episode_identity(self) -> None:
        selected = select_memory_rules(self.memory, "今天把回归测试全部跑完")

        self.assertEqual([rule["episode_id"] for rule in selected], ["user_run_tests"])


class TransferEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        dataset = load_transfer_dataset(TRANSFER_PATH)
        self.episode = next(
            episode for episode in dataset.episodes if episode.id == "user_run_tests"
        )

    def test_transfer_contract_ignores_brackets_and_spacing(self) -> None:
        result = evaluate_transfer(
            self.episode,
            "きょうの  ごご  [かいきてすと]を  じっししおえます",
        )

        self.assertTrue(result["ok"], result)

    def test_transfer_contract_rejects_original_bad_preference(self) -> None:
        result = evaluate_transfer(
            self.episode,
            "きょうの  ごご  [かいきてすと]を  うごかしおわります",
        )

        self.assertFalse(result["ok"])

    def test_surface_normalizer_changes_only_punctuation_and_whitespace(self) -> None:
        result = normalize_surface("きょう、  [てすと]を  じっしします。")

        self.assertEqual(result["translation"], "きょう  [てすと]を  じっしします")
        self.assertTrue(result["changed"])

    def test_format_evaluator_accepts_normalized_hiragana(self) -> None:
        result = evaluate_format("きょう  [てすと]を  じっしします")

        self.assertTrue(result["ok"], result)


class TransferBlindReviewTests(unittest.TestCase):
    def test_blind_rows_hide_group_repetition_and_memory_hits(self) -> None:
        record = {
            "blind_id": "blind",
            "episode_id": "e1",
            "provenance": "user_confirmed_feedback",
            "source": "新句",
            "confirmed_preference": "偏好",
            "translation": "ほんやく",
            "group": "COMPILED_MEMORY",
            "repetition": 2,
            "memory_rule_ids": ["e1"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blind.jsonl"
            write_blind_review(path, [record], seed=1)
            row = json.loads(path.read_text(encoding="utf-8"))

        self.assertNotIn("group", row)
        self.assertNotIn("repetition", row)
        self.assertNotIn("memory_rule_ids", row)
        self.assertIn("preference_transfer_0_to_5", row)


if __name__ == "__main__":
    unittest.main()
