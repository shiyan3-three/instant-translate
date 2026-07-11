from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from app.settings import AppSettings
from poc.poc_agent_semantic_profile import (
    EXPECTED_CASE_SCOPES,
    EXPECTED_CASE_SOURCES,
    GROUPS,
    PROFILE_FIELDS,
    SemanticProfilePocError,
    build_flash_messages,
    build_pro_payloads,
    build_production_context,
    build_runtime_case,
    evaluate_output,
    load_dataset,
    make_dry_format_profile,
    make_dry_semantic_profile,
    render_profile,
    run,
    validate_semantic_profile,
)


DATASET_PATH = Path("poc/data/semantic_profile_dataset.json")


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=str(DATASET_PATH),
        output=str(root / "results.jsonl"),
        state_output=str(root / "state.json"),
        blind_output=str(root / "blind.jsonl"),
        fast_model="deepseek-v4-flash",
        thinking_model="deepseek-v4-pro",
        repetitions=2,
        seed=20260710,
        delay=0.0,
        pro_timeout=180.0,
        flash_timeout=60.0,
        dry_run=True,
    )


class SemanticProfileDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)

    def test_dataset_is_frozen_and_exact(self) -> None:
        self.assertEqual(self.dataset.version, 1)
        self.assertEqual(self.dataset.status, "frozen_poc")
        self.assertEqual(tuple(case.source for case in self.dataset.cases), EXPECTED_CASE_SOURCES)
        self.assertEqual(tuple(case.scope for case in self.dataset.cases), EXPECTED_CASE_SCOPES)

    def test_scope_counts_are_eight_runtime_regressions_and_four_controls(self) -> None:
        scopes = [case.scope for case in self.dataset.cases]
        self.assertEqual(scopes.count("runtime_regression"), 8)
        self.assertEqual(scopes.count("control"), 4)
        self.assertEqual(len({case.id for case in self.dataset.cases}), 12)


class SemanticProfileCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)
        cls.context = build_production_context(AppSettings.load())

    def test_two_pro_payloads_share_system_and_hide_all_evaluation_sources(self) -> None:
        payloads = build_pro_payloads(self.context, "deepseek-v4-pro")
        self.assertEqual(set(payloads), set(GROUPS))
        self.assertEqual(
            payloads["FORMAT_PROFILE"]["messages"][0],
            payloads["SEMANTIC_PROFILE"]["messages"][0],
        )
        rendered = json.dumps(payloads, ensure_ascii=False)
        for case in self.dataset.cases:
            self.assertNotIn(case.source, rendered)

    def test_dry_semantic_profile_has_exact_safe_schema(self) -> None:
        profile = validate_semantic_profile(
            make_dry_semantic_profile(),
            dataset=self.dataset,
        )
        self.assertEqual(profile["version"], 1)
        self.assertEqual(profile["role"], "translation_semantic_profile")
        self.assertEqual(set(PROFILE_FIELDS), set(profile) - {"version", "role"})

    def test_semantic_profile_rejects_extra_fields_term_mappings_and_case_leak(self) -> None:
        extra = deepcopy(make_dry_semantic_profile())
        extra["glossary"] = []
        with self.assertRaises(SemanticProfilePocError):
            validate_semantic_profile(extra, dataset=self.dataset)

        mapping = deepcopy(make_dry_semantic_profile())
        mapping["terminology_strategy"][0] = "接口 -> [いんたあふぇえす]"
        with self.assertRaises(SemanticProfilePocError):
            validate_semantic_profile(mapping, dataset=self.dataset)

        leak = deepcopy(make_dry_semantic_profile())
        leak["semantic_principles"][0] = self.dataset.cases[0].source
        with self.assertRaises(SemanticProfilePocError):
            validate_semantic_profile(leak, dataset=self.dataset)

    def test_rendered_semantic_profile_contains_all_sections_without_json_contract(self) -> None:
        profile = validate_semantic_profile(make_dry_semantic_profile(), dataset=self.dataset)
        rendered = render_profile("SEMANTIC_PROFILE", profile)
        for field in PROFILE_FIELDS:
            self.assertIn(f"[{field}]", rendered)
        self.assertNotIn('"version"', rendered)


class RuntimeIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)
        cls.context = build_production_context(AppSettings.load())

    def test_runtime_delivery_has_one_system_profile_block_and_one_current_ocr(self) -> None:
        case = self.dataset.cases[0]
        runtime = build_runtime_case(self.context, case)
        format_messages = build_flash_messages(
            context=self.context,
            runtime_case=runtime,
            rendered_profile=make_dry_format_profile(),
        )
        semantic_messages = build_flash_messages(
            context=self.context,
            runtime_case=runtime,
            rendered_profile=render_profile("SEMANTIC_PROFILE", make_dry_semantic_profile()),
        )
        self.assertEqual([message["role"] for message in format_messages], ["system", "user"])
        self.assertEqual(format_messages[1], semantic_messages[1])
        self.assertIn("<AGENT_PROFILE>", format_messages[0]["content"])
        self.assertIn(case.source, format_messages[1]["content"])
        self.assertNotIn(case.source, format_messages[0]["content"])

    def test_runtime_uses_ascii_candidate_only_for_api_case(self) -> None:
        ordinary = build_runtime_case(self.context, self.dataset.cases[0])
        api_case = build_runtime_case(self.context, self.dataset.cases[2])
        self.assertEqual(ordinary.technical_terms, ())
        self.assertIn("API", api_case.technical_terms)

    def test_all_dry_fixtures_pass_current_production_validation(self) -> None:
        for case in self.dataset.cases:
            runtime = build_runtime_case(self.context, case)
            evaluated = evaluate_output(
                case.fixture_translation,
                case=case,
                context=self.context,
                runtime_case=runtime,
            )
            self.assertTrue(evaluated["validation"]["ok"], (case.id, evaluated))
            self.assertTrue(evaluated["translation"], case.id)


class DryRunIntegrationTests(unittest.TestCase):
    def test_dry_run_writes_48_results_and_blind_rows_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "poc.poc_agent_semantic_profile._send_request",
            side_effect=AssertionError("network called"),
        ), patch("builtins.print"):
            output, state, blind = run(_args(Path(tmp)))
            output_rows = [
                json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
            ]
            state_data = json.loads(state.read_text(encoding="utf-8"))
            blind_rows = [
                json.loads(line) for line in blind.read_text(encoding="utf-8").splitlines()
            ]

        results = [row for row in output_rows if row.get("type") == "result"]
        self.assertEqual(len(results), 48)
        self.assertEqual(len(blind_rows), 48)
        self.assertEqual({row["group"] for row in results}, set(GROUPS))
        self.assertTrue(all(row["finish_reason"] == "stop" for row in results))
        self.assertTrue(all(row["production_validation"]["ok"] for row in results))
        self.assertEqual(state_data["metadata"]["profile_delivery"], "single_system_agent_profile_block_v1")
        self.assertFalse(state_data["metadata"]["pro_sees_evaluation_sources"])
        for row in blind_rows:
            self.assertNotIn("group", row)
            self.assertNotIn("repetition", row)
            self.assertIsNone(row["semantic_fidelity_0_to_5"])
            self.assertIsNone(row["major_error"])

    def test_dry_artifacts_do_not_persist_api_key_or_reasoning_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "poc.poc_agent_semantic_profile._send_request",
            side_effect=AssertionError("network called"),
        ), patch("builtins.print"):
            output, state, _ = run(_args(Path(tmp)))
            combined = output.read_text(encoding="utf-8") + state.read_text(encoding="utf-8")

        self.assertNotIn("api_key", combined)
        self.assertNotIn("reasoning_content", combined)


if __name__ == "__main__":
    unittest.main()
