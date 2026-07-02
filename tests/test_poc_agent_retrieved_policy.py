"""Offline tests for the final retrieved-policy decision POC."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from poc_agent_retrieved_policy import (
    GROUPS,
    RetrievedPolicyError,
    build_retrieved_messages,
    clone_baseline_records,
    compile_typed_policy,
    load_baseline,
    run,
    select_policy_rules,
)
from poc_agent_strategy_transfer import (
    build_bootstrap_turn,
    build_session_system,
    load_strategy_dataset,
    make_dry_playbook,
    run as run_strategy,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "poc_data" / "agent_strategy_transfer_dataset.json"
TRANSLATION_PATH = ROOT / "poc_data" / "reference_poc_dataset.json"


class RetrievedPolicyCompilationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_strategy_dataset(DATASET_PATH)
        self.policy = compile_typed_policy(make_dry_playbook(), dataset=self.dataset)

    def test_compiler_keeps_only_source_backed_semantic_rules(self) -> None:
        triggers = {rule["trigger"] for rule in self.policy["rules"]}
        self.assertEqual(triggers, {"跑", "复现", "回滚", "合并", "返回"})
        self.assertEqual(
            self.policy["excluded_playbook_sections"],
            ["natural_patterns", "quality_checks", "generated_examples"],
        )

    def test_extra_invented_rule_is_not_kept(self) -> None:
        playbook = make_dry_playbook()
        playbook["domain_sense_decisions"].append(
            {
                "source_expression": "节点",
                "contextual_meaning": "node",
                "preferred_japanese": ["のおど"],
                "avoid": ["せってん"],
            }
        )
        policy = compile_typed_policy(playbook, dataset=self.dataset)
        self.assertNotIn("节点", {rule["trigger"] for rule in policy["rules"]})

    def test_retrieval_is_exact_and_capped(self) -> None:
        case = next(case for case in self.dataset.cases if case.id == "software_no_merge_before_tests")
        selected = select_policy_rules(self.policy, case.source)
        self.assertEqual({rule["trigger"] for rule in selected}, {"跑", "合并"})
        self.assertLessEqual(len(selected), 3)
        general = next(case for case in self.dataset.cases if case.id == "general_evening_booking")
        self.assertEqual(select_policy_rules(self.policy, general.source), [])

    def test_policy_rejects_non_hiragana_surface_guidance(self) -> None:
        playbook = make_dry_playbook()
        playbook["domain_sense_decisions"][0]["preferred_japanese"] = ["実行する"]
        with self.assertRaisesRegex(RetrievedPolicyError, "non-hiragana"):
            compile_typed_policy(playbook, dataset=self.dataset)


class RetrievedPolicyMessageTests(unittest.TestCase):
    def test_unmatched_case_uses_same_neutral_ack_as_raw_control(self) -> None:
        dataset = load_strategy_dataset(DATASET_PATH)
        policy = compile_typed_policy(make_dry_playbook(), dataset=dataset)
        system, _ = build_session_system(TRANSLATION_PATH)
        bootstrap = build_bootstrap_turn(dataset)
        case = next(case for case in dataset.cases if case.id == "general_evening_booking")
        messages, ids = build_retrieved_messages(
            system=system,
            bootstrap_turn=bootstrap,
            policy=policy,
            case=case,
        )
        self.assertEqual(ids, [])
        self.assertIn("Understood", messages[2]["content"])

    def test_matched_case_receives_only_relevant_policy_slice(self) -> None:
        dataset = load_strategy_dataset(DATASET_PATH)
        policy = compile_typed_policy(make_dry_playbook(), dataset=dataset)
        system, _ = build_session_system(TRANSLATION_PATH)
        case = next(case for case in dataset.cases if case.id == "software_run_regression")
        messages, ids = build_retrieved_messages(
            system=system,
            bootstrap_turn=build_bootstrap_turn(dataset),
            policy=policy,
            case=case,
        )
        self.assertEqual(ids, ["startup:跑"])
        self.assertIn("じっし", messages[2]["content"])
        self.assertNotIn("generated_examples", messages[2]["content"])


class RetrievedPolicyDryRunTests(unittest.TestCase):
    def _make_baseline(self, root: Path) -> Path:
        baseline = root / "agent-strategy-fixture.jsonl"
        args = argparse.Namespace(
            dataset=str(DATASET_PATH),
            translation_dataset=str(TRANSLATION_PATH),
            output=str(baseline),
            session_output=str(root / "strategy-session.json"),
            blind_output=str(root / "strategy-blind.jsonl"),
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
            run_strategy(args)
        return baseline

    def test_baseline_loader_rejects_missing_coverage(self) -> None:
        dataset = load_strategy_dataset(DATASET_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._make_baseline(root)
            rows = baseline.read_text(encoding="utf-8").splitlines()
            baseline.write_text("\n".join(rows[:-2]) + "\n", encoding="utf-8")
            with self.assertRaises(RetrievedPolicyError):
                load_baseline(baseline, dataset=dataset)

    def test_dry_run_reuses_48_and_adds_only_24_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._make_baseline(root)
            args = argparse.Namespace(
                dataset=str(DATASET_PATH),
                translation_dataset=str(TRANSLATION_PATH),
                baseline=str(baseline),
                output=str(root / "results.jsonl"),
                state_output=str(root / "state.json"),
                blind_output=str(root / "blind.jsonl"),
                fast_model="deepseek-v4-flash",
                seed=20260702,
                delay=0.0,
                flash_timeout=60.0,
                dry_run=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                output, state, blind = run(args)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            results = [row for row in rows if row.get("type") == "result"]
            summary = next(row for row in rows if row.get("type") == "summary")
            blind_rows = [json.loads(line) for line in blind.read_text(encoding="utf-8").splitlines()]
            persisted = json.loads(state.read_text(encoding="utf-8"))

        self.assertEqual(len(results), 72)
        self.assertEqual(len(blind_rows), 72)
        self.assertEqual({row["group"] for row in results}, set(GROUPS))
        self.assertEqual(sum(row["record_source"] == "reused_frozen_baseline" for row in results), 48)
        self.assertEqual(sum(row["record_source"] == "new_retrieved_request" for row in results), 24)
        self.assertEqual(summary["evidence"]["new_api_calls"], {"pro": 0, "flash": 24})
        self.assertFalse(persisted["metadata"]["contains_reasoning_content"])
        self.assertTrue(all("group" not in row and "repetition" not in row for row in blind_rows))

    def test_cloned_baseline_blind_ids_are_rekeyed(self) -> None:
        rows = [
            {
                "blind_id": "old",
                "group": "RAW_BRIEF",
                "case_id": "case",
                "repetition": 1,
            }
        ]
        cloned = clone_baseline_records(rows, run_id="new")
        self.assertNotEqual(cloned[0]["blind_id"], "old")
        self.assertEqual(cloned[0]["baseline_original_blind_id"], "old")


if __name__ == "__main__":
    unittest.main()
