from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.feedback.store import FeedbackStore
from app.prompt.runtime_profile import RuntimeProfile
from app.reference_layer import ReferenceStore
from app.settings import AppSettings
from app.translation.quality import OutputNormalizer, OutputValidator
from poc.poc_runtime_profile_ab import (
    DATASET_PATH,
    EXPECTED_CASE_SCOPES,
    EXPECTED_CASE_SOURCES,
    EXPECTED_DATASET_SHA256,
    GROUPS,
    LEGACY_DATASET_PATHS,
    REPETITIONS,
    build_confirmed_runtime_profile,
    build_flash_messages,
    build_projection_context,
    freeze_case_contexts,
    load_dataset,
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
        seed=20260710,
        delay=0.0,
        flash_timeout=60.0,
        dry_run=True,
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RuntimeProfileAbDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)

    def test_dataset_sha_and_frozen_shape(self) -> None:
        self.assertEqual(hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(), EXPECTED_DATASET_SHA256)
        self.assertEqual(self.dataset.version, 1)
        self.assertEqual(self.dataset.status, "frozen_poc")
        self.assertEqual(tuple(case.source for case in self.dataset.cases), EXPECTED_CASE_SOURCES)
        self.assertEqual(tuple(case.scope for case in self.dataset.cases), EXPECTED_CASE_SCOPES)
        self.assertEqual(len(self.dataset.cases), 12)
        self.assertEqual(len({case.id for case in self.dataset.cases}), 12)
        self.assertEqual(len({case.source for case in self.dataset.cases}), 12)

    def test_cases_cover_required_semantics_and_control_scope(self) -> None:
        joined = "\n".join(case.source for case in self.dataset.cases)
        for marker in ("已经", "还没", "仍然", "一直", "如果", "只要", "除非", "被"):
            with self.subTest(marker=marker):
                self.assertIn(marker, joined)
        for word in ("系统", "界面", "任务", "队列", "超时"):
            with self.subTest(word=word):
                self.assertIn(word, joined)
        self.assertEqual(sum(case.scope == "runtime_regression" for case in self.dataset.cases), 8)
        self.assertEqual(sum(case.scope == "control" for case in self.dataset.cases), 4)

    def test_sources_are_new_to_prior_poc_datasets(self) -> None:
        old_sources: set[str] = set()
        for path in LEGACY_DATASET_PATHS:
            raw = json.loads(path.read_text(encoding="utf-8"))
            old_sources.update(
                str(row.get("source", "")).strip()
                for row in raw.get("cases", [])
                if isinstance(row, dict)
            )
        self.assertFalse(set(EXPECTED_CASE_SOURCES) & old_sources)


class RuntimeProfileAbMessageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)
        cls.context = build_projection_context(AppSettings.load())
        cls.profile = build_confirmed_runtime_profile(cls.context, cls.dataset)
        cls.frozen = freeze_case_contexts(cls.context, cls.dataset)

    def test_fixed_profile_is_confirmed_matched_and_not_evaluation_derived(self) -> None:
        self.assertTrue(self.profile.confirmed)
        self.assertTrue(self.profile.is_usable)
        self.assertTrue(self.profile.matches_language_pair("中文", "日本語"))
        self.assertTrue(
            self.profile.matches_context(
                prompt_digest=self.context.production.prompt_hash,
                policy_digest=self.context.production.policy_digest,
                reference_digest=self.context.production.reference_digest,
            )
        )
        rendered = self.profile.render()
        self.assertIn("<RUNTIME_PROFILE>", rendered)
        for source in EXPECTED_CASE_SOURCES:
            self.assertNotIn(source, rendered)

    def test_production_runtime_profile_rejects_format_and_reference_rules(self) -> None:
        bad_directives = (
            "只输出平假名",
            "所有内容放进方括号",
            "单词之间使用两个空格",
            "软件 -> そふとうぇあ",
            "添加术语映射",
        )
        for directive in bad_directives:
            with self.subTest(directive=directive), self.assertRaises(ValueError):
                RuntimeProfile.from_dict(
                    {
                        "source_language": "中文",
                        "target_language": "日本語",
                        "semantic_directives": [directive],
                        "confirmed": True,
                    },
                    strict=True,
                )

    def test_only_system_profile_injection_differs_between_groups(self) -> None:
        for case in self.dataset.cases:
            current = build_flash_messages(
                group="CURRENT_RUNTIME",
                context=self.context,
                frozen=self.frozen[case.id],
                runtime_profile=self.profile,
            )
            profiled = build_flash_messages(
                group="RUNTIME_PROFILE_RUNTIME",
                context=self.context,
                frozen=self.frozen[case.id],
                runtime_profile=self.profile,
            )
            self.assertNotIn("<RUNTIME_PROFILE>", current[0]["content"])
            self.assertIn("<RUNTIME_PROFILE>", profiled[0]["content"])
            self.assertEqual(current[1:], profiled[1:])
            self.assertIn("<RULE_CHECKLIST>", current[-1]["content"])

    def test_each_case_has_one_shared_frozen_reference_memory_and_user_context(self) -> None:
        self.assertEqual(set(self.frozen), {case.id for case in self.dataset.cases})
        for case in self.dataset.cases:
            frozen = self.frozen[case.id]
            self.assertTrue(frozen.runtime_case.protected_text)
            self.assertIsInstance(frozen.matched_memory_ids, tuple)
            self.assertIn("<OCR_TEXT>", frozen.current_user_content)
            self.assertIn("<RULE_CHECKLIST>", frozen.current_user_content)


class RuntimeProfileAbRunTests(unittest.TestCase):
    def test_dry_run_is_zero_network_and_writes_complete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "poc.poc_runtime_profile_ab._send_diagnostic_request",
                side_effect=AssertionError("dry-run attempted network"),
            ) as network:
                output, state_path, blind_path = run(_args(root))
            network.assert_not_called()

            rows = _read_jsonl(output)
            results = [row for row in rows if row["type"] == "result"]
            self.assertEqual(len(results), 48)
            self.assertEqual(sum(row["group"] == GROUPS[0] for row in results), 24)
            self.assertEqual(sum(row["group"] == GROUPS[1] for row in results), 24)
            self.assertEqual(rows[0]["pro_calls"], 0)
            self.assertEqual(rows[0]["pro_payloads"], [])
            self.assertEqual(rows[-1]["pro_calls"], 0)

            required = {
                "group", "case_id", "source_text", "raw_translation",
                "normalized_translation", "validation_result", "finish_reason",
                "latency_s", "usage", "estimated_input_tokens",
                "injected_runtime_profile",
            }
            for row in results:
                self.assertTrue(required <= set(row))
                self.assertEqual(
                    row["injected_runtime_profile"],
                    row["group"] == "RUNTIME_PROFILE_RUNTIME",
                )

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["pro_calls"], 0)
            self.assertEqual(state["pro_payloads"], [])
            self.assertEqual(state["attempted_flash_calls"], 0)
            self.assertEqual(
                state["metadata"]["formal_call_design"],
                {"pro": 0, "flash": 48, "pro_judge": 0, "retry": 0},
            )
            self.assertEqual(len(state["frozen_request_contexts"]), 12)
            self.assertFalse(state["system_prompts"][GROUPS[0]]["injected_runtime_profile"])
            self.assertTrue(state["system_prompts"][GROUPS[1]]["injected_runtime_profile"])

            blind = _read_jsonl(blind_path)
            self.assertEqual(len(blind), 48)
            self.assertTrue(all("group" not in row and "repetition" not in row for row in blind))
            self.assertTrue(all("blind_id" in row and "translation" in row for row in blind))

    def test_freezing_reads_reference_and_memory_once_per_case(self) -> None:
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
        self.assertEqual(reference_calls, 12)
        self.assertEqual(memory_calls, 12)

    def test_dry_run_executes_production_normalizer_and_validator_for_all_results(self) -> None:
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
        self.assertEqual(normalize_calls, 48)
        self.assertEqual(validate_calls, 48)

    def test_source_texts_cannot_enter_a_pro_payload_that_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, state_path, _ = run(_args(Path(tmp)))
            state = json.loads(state_path.read_text(encoding="utf-8"))
        serialized = json.dumps(state["pro_payloads"], ensure_ascii=False)
        self.assertEqual(state["pro_payloads"], [])
        self.assertTrue(all(source not in serialized for source in EXPECTED_CASE_SOURCES))


if __name__ == "__main__":
    unittest.main()
