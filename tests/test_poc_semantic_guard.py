from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from app.agent.agent import TranslationAgent
from app.settings import AppSettings
from app.translation.service import TranslationService
from poc.poc_semantic_guard import (
    EXPECTED_CASE_SCOPES,
    EXPECTED_CASE_SOURCES,
    GROUPS,
    MAX_RULES_PER_REQUEST,
    STATIC_GUARD_RULES,
    SemanticGuardError,
    build_digest_payload,
    build_group_messages,
    build_policy_source,
    build_production_context,
    check_ascii_digit_leak,
    detect_features,
    diagnose_completeness,
    evaluate_format,
    evaluate_production_output,
    load_semantic_guard_dataset,
    make_dry_checklist,
    make_dry_typed_policy,
    run,
    select_policy_rules,
    validate_typed_policy,
)


DATASET_PATH = Path("poc/data/semantic_guard_dataset.json")


def _contains_key(value, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def _make_dry_run_args(directory: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=str(DATASET_PATH),
        output=str(directory / "results.jsonl"),
        state_output=str(directory / "state.json"),
        blind_output=str(directory / "blind.jsonl"),
        fast_model="deepseek-v4-flash",
        thinking_model="deepseek-v4-pro",
        repetitions=2,
        seed=20260703,
        delay=0.0,
        pro_timeout=180.0,
        flash_timeout=60.0,
        dry_run=True,
    )


class SemanticGuardDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_semantic_guard_dataset(DATASET_PATH)

    def test_dataset_is_exact_frozen_benchmark(self) -> None:
        self.assertEqual(self.dataset.version, 1)
        self.assertEqual(self.dataset.status, "frozen_poc")
        self.assertEqual(tuple(case.source for case in self.dataset.cases), EXPECTED_CASE_SOURCES)
        self.assertEqual(tuple(case.scope for case in self.dataset.cases), EXPECTED_CASE_SCOPES)
        self.assertEqual(len({case.id for case in self.dataset.cases}), 12)

    def test_scope_counts_are_3_seen_6_held_out_3_control(self) -> None:
        scopes = [case.scope for case in self.dataset.cases]
        self.assertEqual(scopes.count("seen_regression"), 3)
        self.assertEqual(scopes.count("held_out"), 6)
        self.assertEqual(scopes.count("control"), 3)

    def test_training_incidents_match_only_seen_regressions(self) -> None:
        incident_sources = {item.source for item in self.dataset.training_incidents}
        seen_sources = {case.source for case in self.dataset.cases if case.scope == "seen_regression"}
        other_sources = {case.source for case in self.dataset.cases if case.scope != "seen_regression"}
        self.assertEqual(incident_sources, seen_sources)
        self.assertTrue(incident_sources.isdisjoint(other_sources))

    def test_fixtures_are_valid_and_never_wrap_pure_numbers(self) -> None:
        for case in self.dataset.cases:
            self.assertNotIn("[", case.fixture_translation, case.id)
            self.assertNotIn("]", case.fixture_translation, case.id)
            self.assertTrue(evaluate_format(case.fixture_translation)["ok"], case.id)
            self.assertFalse(check_ascii_digit_leak(case.fixture_translation), case.id)

    def test_policy_source_contains_only_training_incidents(self) -> None:
        payload = build_policy_source(self.dataset)
        for incident in self.dataset.training_incidents:
            self.assertIn(incident.source, payload)
        for case in self.dataset.cases:
            if case.scope != "seen_regression":
                self.assertNotIn(case.source, payload)


class ProductionContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = AppSettings.load()
        cls.context = build_production_context(cls.settings)
        cls.dataset = load_semantic_guard_dataset(DATASET_PATH)

    def test_context_matches_current_agent_definition(self) -> None:
        service = TranslationService(self.settings)
        try:
            definition = service._resolve_agent_definition("中文", "日本語")
        finally:
            service.shutdown()
        expected_system = TranslationAgent._append_runtime_tool_rules(
            TranslationAgent._rewrite_constraints(definition.prompt),
            preserve_placeholders=not definition.policy.has_script("hiragana"),
        )
        self.assertEqual(self.context.prompt_hash, definition.runtime_meta.prompt_hash)
        self.assertEqual(self.context.prompt_length, len(definition.prompt))
        self.assertEqual(self.context.policy_digest, definition.policy.digest())
        self.assertEqual(self.context.system, expected_system)

    def test_digest_payload_contains_no_experiment_cases(self) -> None:
        rendered = json.dumps(
            build_digest_payload(self.context, "deepseek-v4-pro"),
            ensure_ascii=False,
        )
        for case in self.dataset.cases:
            self.assertNotIn(case.source, rendered)
        for incident in self.dataset.training_incidents:
            self.assertNotIn(incident.source, rendered)

    def test_all_fixtures_pass_production_validation(self) -> None:
        for case in self.dataset.cases:
            result = evaluate_production_output(
                case.fixture_translation,
                case=case,
                context=self.context,
            )
            self.assertTrue(result["validation"]["ok"], (case.id, result))

    def test_ascii_digit_output_is_normalized_before_production_validation(self) -> None:
        case = self.dataset.cases[2]
        result = evaluate_production_output(
            "しょうひん  の  ねだん  は  4.0  げん  です",
            case=case,
            context=self.context,
        )
        self.assertTrue(result["normalization_changed"])
        self.assertTrue(result["validation"]["ok"], result)
        self.assertFalse(check_ascii_digit_leak(result["translation"]))
        self.assertIn("よんてんれい", result["translation"])


class TypedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_semantic_guard_dataset(DATASET_PATH)

    def test_dry_policy_schema(self) -> None:
        policy = validate_typed_policy(make_dry_typed_policy(), dataset=self.dataset)
        self.assertEqual(policy["version"], 1)
        self.assertEqual(policy["role"], "retrieved_semantic_policy")
        self.assertEqual(policy["max_rules_per_request"], MAX_RULES_PER_REQUEST)
        self.assertEqual(len(policy["rules"]), 3)

    def test_rejects_wrong_version_duplicate_id_and_invalid_feature(self) -> None:
        raw = make_dry_typed_policy()
        raw["version"] = 2
        with self.assertRaises(SemanticGuardError):
            validate_typed_policy(raw, dataset=self.dataset)

        raw = make_dry_typed_policy()
        raw["rules"][1]["id"] = raw["rules"][0]["id"]
        with self.assertRaises(SemanticGuardError):
            validate_typed_policy(raw, dataset=self.dataset)

        raw = make_dry_typed_policy()
        raw["rules"][0]["trigger_features"] = ["contains_any_number"]
        with self.assertRaises(SemanticGuardError):
            validate_typed_policy(raw, dataset=self.dataset)

    def test_rejects_empty_trigger_and_held_out_leak(self) -> None:
        raw = make_dry_typed_policy()
        raw["rules"][0]["trigger_literals"] = []
        with self.assertRaises(SemanticGuardError):
            validate_typed_policy(raw, dataset=self.dataset)

        raw = make_dry_typed_policy()
        raw["rules"][0]["instruction"] += self.dataset.cases[3].source
        with self.assertRaises(SemanticGuardError):
            validate_typed_policy(raw, dataset=self.dataset)

    def test_feature_detection(self) -> None:
        self.assertIn("contains_decimal", detect_features("温度是4.0度"))
        self.assertIn("contains_date", detect_features("日期是2026年7月4日"))
        self.assertIn("contains_numeric_range", detect_features("成绩70到80分"))
        self.assertEqual(detect_features("没有阿拉伯数字"), [])

    def test_dry_policy_has_required_retrieval_coverage(self) -> None:
        policy = validate_typed_policy(make_dry_typed_policy(), dataset=self.dataset)
        selected = {
            case.id: [rule["id"] for rule in select_policy_rules(policy, case.source)]
            for case in self.dataset.cases
        }
        seen_hits = sum(bool(selected[case.id]) for case in self.dataset.cases if case.scope == "seen_regression")
        held_hits = sum(bool(selected[case.id]) for case in self.dataset.cases if case.scope == "held_out")
        control_hits = sum(bool(selected[case.id]) for case in self.dataset.cases if case.scope == "control")
        self.assertEqual(seen_hits, 3)
        self.assertGreaterEqual(held_hits, 5)
        self.assertEqual(control_hits, 0)
        self.assertIn("sg:complete_predicate", selected["sg_04"])
        self.assertIn("sg:no_injected_contrast", selected["sg_05"])
        self.assertIn("sg:no_injected_contrast", selected["sg_06"])
        self.assertIn("sg:no_ascii_digits", selected["sg_07"])
        self.assertIn("sg:no_ascii_digits", selected["sg_08"])

    def test_retrieval_is_capped_at_three(self) -> None:
        raw = deepcopy(make_dry_typed_policy())
        for index in range(3):
            extra = deepcopy(raw["rules"][0])
            extra["id"] = f"extra:{index}"
            raw["rules"].append(extra)
        policy = validate_typed_policy(raw, dataset=self.dataset)
        self.assertEqual(len(select_policy_rules(policy, "我准备出门")), 3)


class GroupIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_semantic_guard_dataset(DATASET_PATH)
        cls.context = build_production_context(AppSettings.load())
        cls.policy = validate_typed_policy(make_dry_typed_policy(), dataset=cls.dataset)
        cls.checklist = make_dry_checklist()

    def _build(self, group: str, case_id: str):
        case = next(item for item in self.dataset.cases if item.id == case_id)
        return build_group_messages(
            context=self.context,
            digest_user="digest",
            checklist=self.checklist,
            case=case,
            group=group,
            policy=self.policy,
        )

    def test_first_three_profile_messages_are_identical(self) -> None:
        built = [self._build(group, "sg_04")[0] for group in GROUPS]
        self.assertEqual(built[0][:3], built[1][:3])
        self.assertEqual(built[1][:3], built[2][:3])

    def test_current_has_no_semantic_memory(self) -> None:
        messages, ids = self._build("CURRENT_PROFILE", "sg_04")
        self.assertEqual(ids, [])
        self.assertNotIn("<TRANSLATION_MEMORY>", messages[3]["content"])

    def test_static_contains_only_fixed_guard(self) -> None:
        messages, ids = self._build("STATIC_SEMANTIC_GUARD", "sg_04")
        self.assertEqual(ids, [])
        for rule in STATIC_GUARD_RULES:
            self.assertIn(rule, messages[3]["content"])
        self.assertNotIn("sg:complete_predicate", messages[3]["content"])

    def test_retrieved_differs_on_matching_held_out_case(self) -> None:
        static_messages, _ = self._build("STATIC_SEMANTIC_GUARD", "sg_04")
        retrieved_messages, ids = self._build("RETRIEVED_SEMANTIC_POLICY", "sg_04")
        self.assertIn("sg:complete_predicate", ids)
        self.assertNotEqual(static_messages[3], retrieved_messages[3])

    def test_retrieved_equals_static_when_no_rule_matches(self) -> None:
        static_messages, _ = self._build("STATIC_SEMANTIC_GUARD", "sg_09")
        retrieved_messages, ids = self._build("RETRIEVED_SEMANTIC_POLICY", "sg_09")
        self.assertEqual(ids, [])
        self.assertEqual(static_messages[3], retrieved_messages[3])


class DiagnosticTests(unittest.TestCase):
    def test_completeness_detects_real_dangling_endings(self) -> None:
        self.assertFalse(diagnose_completeness("べんきょう  する  じゅんび  を")["complete"])
        self.assertFalse(diagnose_completeness("おいた  はず  なのに")["complete"])
        self.assertTrue(diagnose_completeness("べんきょうする  よてい  です")["complete"])

    def test_ascii_digit_leak(self) -> None:
        self.assertTrue(check_ascii_digit_leak("よん  4.0"))
        self.assertFalse(check_ascii_digit_leak("よんてんぜろ"))


class DryRunIntegrationTests(unittest.TestCase):
    def _run(self, root: Path):
        with patch("poc.poc_semantic_guard._send_request", side_effect=AssertionError("network called")):
            with patch("builtins.print"):
                return run(_make_dry_run_args(root))

    def test_dry_run_generates_72_production_shaped_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output, state, blind = self._run(Path(tmp))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            state_data = json.loads(state.read_text(encoding="utf-8"))
            blind_rows = [json.loads(line) for line in blind.read_text(encoding="utf-8").splitlines()]

        results = [row for row in rows if row.get("type") == "result"]
        self.assertEqual(len(results), 72)
        self.assertEqual({row["group"] for row in results}, set(GROUPS))
        self.assertTrue(all("production_validation" in row for row in results))
        self.assertTrue(all(row["production_validation"]["ok"] for row in results))
        self.assertTrue(all(row["normalized_translation"] for row in results))
        self.assertEqual(len(blind_rows), 72)
        self.assertTrue(state_data["metadata"]["prompt_hash"])
        self.assertTrue(state_data["metadata"]["policy_digest"])
        self.assertTrue(state_data["metadata"]["enhanced_system_hash"])

    def test_blind_and_persisted_files_hide_internal_or_sensitive_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output, state, blind = self._run(Path(tmp))
            output_text = output.read_text(encoding="utf-8")
            state_text = state.read_text(encoding="utf-8")
            output_rows = [json.loads(line) for line in output_text.splitlines()]
            state_data = json.loads(state_text)
            blind_rows = [json.loads(line) for line in blind.read_text(encoding="utf-8").splitlines()]

        self.assertNotIn("api_key", output_text)
        self.assertNotIn("api_key", state_text)
        self.assertFalse(_contains_key(output_rows, "reasoning_content"))
        self.assertFalse(_contains_key(state_data, "reasoning_content"))
        for row in blind_rows:
            self.assertNotIn("group", row)
            self.assertNotIn("repetition", row)
            self.assertNotIn("retrieved_rule_ids", row)
            self.assertIsNone(row["semantic_fidelity_0_to_5"])
            self.assertIsNone(row["naturalness_0_to_5"])


if __name__ == "__main__":
    unittest.main()
