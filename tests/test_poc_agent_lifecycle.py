"""Offline tests for the shared Pro/Flash Agent lifecycle POC."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from poc_agent_lifecycle import (
    LifecyclePocError,
    build_session_system,
    evaluate_preference,
    load_scenario,
    load_session,
    run,
    save_session,
    validate_session_state,
)


ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_PATH = ROOT / "poc_data" / "agent_lifecycle_dataset.json"
TRANSLATION_PATH = ROOT / "poc_data" / "reference_poc_dataset.json"


class LifecycleDataTests(unittest.TestCase):
    def test_scenario_uses_real_feedback_and_four_distinct_turns(self) -> None:
        scenario = load_scenario(LIFECYCLE_PATH)

        self.assertEqual(scenario.provenance, "user_confirmed_feedback")
        self.assertEqual(
            len(
                {
                    scenario.initial_source,
                    scenario.raw_vs_handoff_source,
                    scenario.thinking_mode_source,
                    scenario.after_thinking_source,
                }
            ),
            4,
        )

    def test_system_explicitly_supports_control_feedback_and_ocr_roles(self) -> None:
        system, prompt_hash = build_session_system(TRANSLATION_PATH)

        self.assertIn("SESSION_CONTROL", system)
        self.assertIn("USER_FEEDBACK", system)
        self.assertIn("OCR_TEXT", system)
        self.assertEqual(len(prompt_hash), 64)


class LifecyclePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "metadata": {
                "version": 1,
                "session_id": "session-a",
                "source_hash": "source",
                "prompt_hash": "prompt",
                "fast_model": "flash",
                "thinking_model": "pro",
                "handoff_version": 1,
                "contains_reasoning_content": False,
            },
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "control"},
                {"role": "assistant", "content": "handoff"},
            ],
            "events": [{"event_type": "handoff"}],
        }

    def test_session_round_trip_preserves_exact_visible_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            save_session(path, self.state)
            loaded = load_session(
                path,
                source_hash="source",
                fast_model="flash",
                thinking_model="pro",
            )
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["messages"], self.state["messages"])
        self.assertFalse(raw["metadata"]["contains_reasoning_content"])
        self.assertTrue(all("reasoning_content" not in m for m in raw["messages"]))

    def test_session_is_rejected_across_source_boundary(self) -> None:
        with self.assertRaisesRegex(LifecyclePocError, "different source evidence"):
            validate_session_state(
                self.state,
                source_hash="other",
                fast_model="flash",
                thinking_model="pro",
            )


class LifecycleEvaluationTests(unittest.TestCase):
    def test_preference_and_negation_must_both_survive_after_mode_switch(self) -> None:
        result = evaluate_preference(
            "[あんぜんてすと]は  まだ  じっししおえていません",
            required_any=("じっし", "じっこう"),
            forbidden=("うごかし", "はしり"),
            extra_required_any=("ない", "ません", "まだ"),
        )

        self.assertTrue(result["ok"], result)

    def test_missing_negation_fails_post_pro_continuity(self) -> None:
        result = evaluate_preference(
            "[あんぜんてすと]を  じっししおえました",
            required_any=("じっし", "じっこう"),
            forbidden=("うごかし", "はしり"),
            extra_required_any=("ない", "ません", "まだ"),
        )

        self.assertFalse(result["ok"])


class LifecycleDryRunTests(unittest.TestCase):
    def test_dry_run_models_full_shared_session_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            session = Path(directory) / "session.json"
            args = argparse.Namespace(
                dataset=str(LIFECYCLE_PATH),
                translation_dataset=str(TRANSLATION_PATH),
                output=str(output),
                session_output=str(session),
                fast_model="deepseek-v4-flash",
                thinking_model="deepseek-v4-pro",
                pro_timeout=180.0,
                flash_timeout=60.0,
                dry_run=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            persisted = json.loads(session.read_text(encoding="utf-8"))

        results = [row for row in rows if row.get("type") == "result"]
        summary = next(row for row in rows if row.get("type") == "summary")
        by_step = {row["step"]: row for row in results}
        self.assertEqual(
            [row["step"] for row in results],
            [
                "startup_pro",
                "initial_flash",
                "raw_control_flash",
                "refresh_pro",
                "shared_flash",
                "thinking_pro",
                "post_pro_flash",
            ],
        )
        self.assertEqual(len(results), 7)
        self.assertEqual(by_step["startup_pro"]["payload"]["thinking"], {"type": "enabled"})
        self.assertEqual(by_step["shared_flash"]["payload"]["thinking"], {"type": "disabled"})
        raw_context = "\n".join(
            message["content"] for message in by_step["raw_control_flash"]["payload"]["messages"]
        )
        shared_context = "\n".join(
            message["content"] for message in by_step["shared_flash"]["payload"]["messages"]
        )
        post_context = "\n".join(
            message["content"] for message in by_step["post_pro_flash"]["payload"]["messages"]
        )
        self.assertNotIn("Accepted correction: in software testing", raw_context)
        self.assertIn("Accepted correction: in software testing", shared_context)
        self.assertIn("[せいのうてすと]も", post_context)
        self.assertTrue(summary["lifecycle_ok"])
        self.assertEqual(summary["shared_context_checks"]["handoff_version"], 2)
        self.assertEqual(persisted["metadata"]["session_id"], summary["session_id"])
        self.assertFalse(persisted["metadata"]["contains_reasoning_content"])


if __name__ == "__main__":
    unittest.main()
