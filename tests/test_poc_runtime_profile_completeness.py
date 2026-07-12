from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from app.agent.session_store import AgentSessionStore
from app.feedback.store import FeedbackStore
from app.reference_layer import ReferenceStore
from app.settings import AppSettings
from app.translation.quality import OutputNormalizer, OutputValidator
from poc.poc_runtime_profile_completeness import (
    COMPLETENESS_CHECK,
    CompletenessPocError,
    DATASET_PATH,
    EXPECTED_CASE_SCOPES,
    EXPECTED_CASE_SOURCES,
    EXPECTED_DATASET_SHA256,
    FORMAL_CALL_DESIGN,
    GROUPS,
    LEGACY_DATASET_PATHS,
    REPETITIONS,
    build_confirmed_runtime_profile,
    build_flash_messages,
    build_isolated_fixture_context,
    freeze_case_contexts,
    load_dataset,
    load_production_agent_profile,
    run,
)


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=str(DATASET_PATH),
        output=str(root / "results.jsonl"),
        state_output=str(root / "state.json"),
        blind_output=str(root / "blind.jsonl"),
        fast_model="deepseek-v4-flash",
        repetitions=REPETITIONS,
        seed=20260711,
        delay=0.0,
        flash_timeout=60.0,
        dry_run=True,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CompletenessDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)

    def test_dataset_sha_shape_and_scopes_are_frozen(self) -> None:
        self.assertEqual(hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(), EXPECTED_DATASET_SHA256)
        self.assertEqual(tuple(case.source for case in self.dataset.cases), EXPECTED_CASE_SOURCES)
        self.assertEqual(tuple(case.scope for case in self.dataset.cases), EXPECTED_CASE_SCOPES)
        self.assertEqual(len(self.dataset.cases), 16)
        self.assertEqual(len({case.id for case in self.dataset.cases}), 16)
        counts = {
            scope: sum(case.scope == scope for case in self.dataset.cases)
            for scope in set(EXPECTED_CASE_SCOPES)
        }
        self.assertEqual(
            counts,
            {"runtime_regression": 6, "control": 4, "lexical_trap": 4, "reference_memory": 2},
        )

    def test_sources_are_new_and_cover_required_categories(self) -> None:
        old_sources: set[str] = set()
        for path in LEGACY_DATASET_PATHS:
            raw = json.loads(path.read_text(encoding="utf-8"))
            old_sources.update(
                str(row.get("source", "")).strip()
                for row in raw.get("cases", [])
                if isinstance(row, dict)
            )
        self.assertFalse(set(EXPECTED_CASE_SOURCES) & old_sources)
        joined = "\n".join(EXPECTED_CASE_SOURCES)
        for marker in (
            "没有", "如果", "除非", "否则", "已经", "还没", "仍然", "一直", "被",
            "配置文件", "后台任务", "队列", "连接", "超时", "提交",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, joined)


class CompletenessProfileMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)
        cls.fixture = tempfile.TemporaryDirectory()
        cls.context, cls.fixture_metadata = build_isolated_fixture_context(
            AppSettings.load(), Path(cls.fixture.name)
        )
        cls.profile_meta, cls.production_bootstrap, cls.profile_audit = (
            load_production_agent_profile(AppSettings.load())
        )
        cls.profile = build_confirmed_runtime_profile(cls.context, cls.dataset)
        cls.frozen = freeze_case_contexts(
            cls.context,
            cls.dataset,
            rule_checklist=cls.production_bootstrap[2]["content"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.cleanup()

    def test_profile_is_exactly_one_completeness_check(self) -> None:
        self.assertTrue(self.profile.is_usable)
        self.assertEqual(self.profile.semantic_directives, ())
        self.assertEqual(self.profile.style_directives, ())
        self.assertEqual(self.profile.completeness_checks, (COMPLETENESS_CHECK,))
        self.assertIn("<RUNTIME_PROFILE>", self.profile.render())
        lowered = self.profile.render().casefold()
        for forbidden in (
            "semantic directives", "style directives", "术语", "平假名", "括号",
            "空格", "glossary", "example", "natural", "style",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered)
        for source in EXPECTED_CASE_SOURCES:
            self.assertNotIn(source, self.profile.render())

    def test_two_groups_differ_only_in_system_profile_block(self) -> None:
        for case in self.dataset.cases:
            current = build_flash_messages(
                group=GROUPS[0], context=self.context,
                frozen=self.frozen[case.id], runtime_profile=self.profile,
                production_bootstrap=self.production_bootstrap,
            )
            completeness = build_flash_messages(
                group=GROUPS[1], context=self.context,
                frozen=self.frozen[case.id], runtime_profile=self.profile,
                production_bootstrap=self.production_bootstrap,
            )
            self.assertNotIn("<RUNTIME_PROFILE>", current[0]["content"])
            self.assertIn("<RUNTIME_PROFILE>", completeness[0]["content"])
            self.assertEqual(current[:3], list(self.production_bootstrap))
            self.assertEqual(current[1:], completeness[1:])

    def test_loaded_profile_is_exact_current_three_message_bootstrap(self) -> None:
        self.assertEqual(self.profile_audit["status"], "loaded")
        self.assertEqual(self.profile_audit["key"], self.profile_meta.key())
        self.assertEqual(
            [message["role"] for message in self.production_bootstrap],
            ["system", "user", "assistant"],
        )
        self.assertEqual(len(self.profile_audit["messages"]), 3)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in self.profile_audit["messages"]))

    def test_missing_matching_profile_fails_instead_of_generating_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_store = AgentSessionStore(Path(tmp))
            with self.assertRaises(CompletenessPocError):
                load_production_agent_profile(
                    AppSettings.load(), session_store=empty_store
                )
            self.assertFalse(any(Path(tmp).rglob("*.json")))

    def test_at_least_four_cases_have_frozen_reference_or_memory_hits(self) -> None:
        matched = [
            case.id
            for case in self.dataset.cases
            if self.frozen[case.id].runtime_case.plan.matched_entries
            or self.frozen[case.id].matched_memory_ids
        ]
        self.assertGreaterEqual(len(matched), 4)
        self.assertEqual(self.fixture_metadata["kind"], "isolated_poc_fixture")
        self.assertFalse(self.fixture_metadata["settings_modified"])
        self.assertFalse(self.fixture_metadata["real_user_data_modified"])

    def test_fixture_context_does_not_mutate_settings_object(self) -> None:
        settings = AppSettings.load()
        before = asdict(settings)
        with tempfile.TemporaryDirectory() as tmp:
            _, metadata = build_isolated_fixture_context(settings, Path(tmp))
        self.assertEqual(asdict(settings), before)
        self.assertFalse(metadata["settings_modified"])


class CompletenessRunTests(unittest.TestCase):
    def test_real_builder_dry_run_is_zero_network_and_writes_64_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "poc.poc_runtime_profile_completeness._send_diagnostic_request",
                side_effect=AssertionError("dry-run attempted network"),
            ) as network:
                output, state_path, blind_path = run(_args(Path(tmp)))
            network.assert_not_called()
            rows = _read_jsonl(output)
            results = [row for row in rows if row["type"] == "result"]
            self.assertEqual(len(results), 64)
            self.assertEqual(sum(row["group"] == GROUPS[0] for row in results), 32)
            self.assertEqual(sum(row["group"] == GROUPS[1] for row in results), 32)
            self.assertTrue(all(row["injected_runtime_profile"] == (row["group"] == GROUPS[1]) for row in results))

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["dataset_sha256"], EXPECTED_DATASET_SHA256)
            self.assertEqual(state["groups"], list(GROUPS))
            self.assertEqual(state["formal_call_design"], FORMAL_CALL_DESIGN)
            self.assertTrue(state["dry_run"])
            self.assertEqual(state["pro_calls"], 0)
            self.assertEqual(state["pro_payloads"], [])
            self.assertEqual(state["attempted_flash_calls"], 0)
            self.assertEqual(len(state["frozen_request_contexts"]), 16)
            self.assertIn("token_gate_likely_failed_before_formal_run", state["summary"])
            profile_audit = state["production_agent_profile"]
            self.assertEqual(profile_audit["status"], "loaded")
            self.assertEqual(len(profile_audit["messages"]), 3)
            self.assertEqual(
                [row["role"] for row in profile_audit["messages"]],
                ["system", "user", "assistant"],
            )
            for group in GROUPS:
                self.assertEqual(
                    set(state["summary"]["groups"][group]),
                    {
                        "completed", "expected", "validation_pass", "latency_p50_s",
                        "latency_p95_s", "estimated_input_tokens_mean",
                        "provider_prompt_tokens", "injected_runtime_profile",
                    },
                )

            blind = _read_jsonl(blind_path)
            self.assertEqual(len(blind), 64)
            self.assertTrue(all("group" not in row and "repetition" not in row for row in blind))

    def test_result_has_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output, _, _ = run(_args(Path(tmp)))
            results = [row for row in _read_jsonl(output) if row["type"] == "result"]
        required = {
            "group", "case_id", "scope", "category", "source_text", "review_focus",
            "repetition", "raw_translation", "normalized_translation", "validation_result",
            "finish_reason", "latency_s", "usage", "estimated_input_tokens",
            "injected_runtime_profile", "matched_reference_ids", "matched_memory_ids",
            "source_technical_terms", "error",
        }
        self.assertTrue(all(required <= set(row) for row in results))

    def test_reference_and_memory_retrieval_happens_once_per_case(self) -> None:
        reference_calls = 0
        memory_calls = 0
        original_protect = ReferenceStore.protect
        original_match = FeedbackStore.match_memory_rules

        def counted_protect(store, *args, **kwargs):
            nonlocal reference_calls
            reference_calls += 1
            return original_protect(store, *args, **kwargs)

        def counted_match(store, *args, **kwargs):
            nonlocal memory_calls
            memory_calls += 1
            return original_match(store, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ReferenceStore, "protect", counted_protect
        ), patch.object(FeedbackStore, "match_memory_rules", counted_match):
            run(_args(Path(tmp)))
        self.assertEqual(reference_calls, 16)
        self.assertEqual(memory_calls, 16)

    def test_production_normalizer_and_validator_execute_for_every_result(self) -> None:
        normalize_calls = 0
        validate_calls = 0
        original_normalize = OutputNormalizer.normalize_with_policy
        original_validate = OutputValidator.validate

        def counted_normalize(*args, **kwargs):
            nonlocal normalize_calls
            normalize_calls += 1
            return original_normalize(*args, **kwargs)

        def counted_validate(*args, **kwargs):
            nonlocal validate_calls
            validate_calls += 1
            return original_validate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            OutputNormalizer, "normalize_with_policy", side_effect=counted_normalize
        ), patch.object(OutputValidator, "validate", side_effect=counted_validate):
            run(_args(Path(tmp)))
        self.assertGreaterEqual(normalize_calls, 64)
        self.assertEqual(validate_calls, 64)


if __name__ == "__main__":
    unittest.main()
