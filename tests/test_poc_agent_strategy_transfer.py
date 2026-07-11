"""Offline tests for visible Pro strategy transfer."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from poc.poc_agent_strategy_transfer import (
    GROUPS,
    StrategyPocError,
    build_bootstrap_turn,
    build_group_messages,
    build_session_system,
    evaluate_anchors,
    load_strategy_dataset,
    make_dry_playbook,
    run,
    validate_playbook,
)


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_PATH = ROOT / "poc" / "data" / "agent_strategy_transfer_dataset.json"
TRANSLATION_PATH = ROOT / "poc" / "data" / "reference_poc_dataset.json"


class StrategyDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_strategy_dataset(STRATEGY_PATH)

    def test_has_twelve_unique_held_out_cases(self) -> None:
        self.assertEqual(len(self.dataset.cases), 12)
        self.assertEqual(len({case.source for case in self.dataset.cases}), 12)

    def test_bootstrap_request_never_contains_held_out_sources(self) -> None:
        bootstrap = build_bootstrap_turn(self.dataset)
        for case in self.dataset.cases:
            self.assertNotIn(case.source, bootstrap)

    def test_playbook_rejects_exact_test_leakage(self) -> None:
        playbook = make_dry_playbook()
        playbook["generated_examples"][0]["source"] = self.dataset.cases[0].source
        with self.assertRaisesRegex(StrategyPocError, "leaked held-out"):
            validate_playbook(playbook, self.dataset)


class StrategyMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_strategy_dataset(STRATEGY_PATH)
        self.system, _ = build_session_system(TRANSLATION_PATH)
        self.bootstrap = build_bootstrap_turn(self.dataset)
        self.playbook = make_dry_playbook()

    def test_groups_differ_only_in_assistant_startup_content(self) -> None:
        case = self.dataset.cases[0]
        raw = build_group_messages(
            system=self.system,
            bootstrap_turn=self.bootstrap,
            playbook=self.playbook,
            case=case,
            group="RAW_BRIEF",
        )
        pro = build_group_messages(
            system=self.system,
            bootstrap_turn=self.bootstrap,
            playbook=self.playbook,
            case=case,
            group="PRO_PLAYBOOK",
        )
        self.assertEqual([m["role"] for m in raw], ["system", "user", "assistant", "user"])
        self.assertEqual(raw[0:2], pro[0:2])
        self.assertNotEqual(raw[2], pro[2])
        self.assertEqual(raw[3], pro[3])

    def test_machine_anchor_is_not_labelled_quality_score(self) -> None:
        case = self.dataset.cases[0]
        result = evaluate_anchors(case, case.fixture_translation)
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["quality_claim"])


class StrategyDryRunTests(unittest.TestCase):
    def test_dry_run_builds_full_balanced_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                dataset=str(STRATEGY_PATH),
                translation_dataset=str(TRANSLATION_PATH),
                output=str(root / "results.jsonl"),
                session_output=str(root / "session.json"),
                blind_output=str(root / "blind.jsonl"),
                fast_model="deepseek-v4-flash",
                thinking_model="deepseek-v4-pro",
                repetitions=2,
                seed=20260702,
                delay=0.0,
                pro_timeout=180.0,
                flash_timeout=60.0,
                dry_run=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                output, session, blind = run(args)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            results = [row for row in rows if row.get("type") == "result"]
            summary = next(row for row in rows if row.get("type") == "summary")
            persisted = json.loads(session.read_text(encoding="utf-8"))
            blind_rows = [json.loads(line) for line in blind.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(results), 48)
        self.assertEqual(len(blind_rows), 48)
        self.assertEqual({row["group"] for row in results}, set(GROUPS))
        self.assertTrue(all(row["payload"]["thinking"] == {"type": "disabled"} for row in results))
        self.assertTrue(all(row["anchor_evaluation"]["quality_claim"] is False for row in results))
        self.assertEqual(summary["groups"]["RAW_BRIEF"]["requests"], 24)
        self.assertEqual(summary["groups"]["PRO_PLAYBOOK"]["requests"], 24)
        self.assertFalse(persisted["metadata"]["contains_reasoning_content"])
        self.assertTrue(all("group" not in row and "repetition" not in row for row in blind_rows))


if __name__ == "__main__":
    unittest.main()
