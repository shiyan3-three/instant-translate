from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.feedback.store import FeedbackStore
from app.prompt.policy import ConstraintPolicy
from app.reference_layer import ReferenceStore
from app.settings import AppSettings
from app.translation.client import TranslationError
from app.translation.quality import OutputNormalizer, OutputValidator
from poc.poc_agent_runtime_projection import (
    DATASET_PATH,
    EXPECTED_CASE_SCOPES,
    EXPECTED_CASE_SOURCES,
    EXPECTED_DATASET_SHA256,
    GROUPS,
    LEGACY_DATASET_PATHS,
    PROFILE_DIRECTIVE_FIELDS,
    RuntimeProjectionPocError,
    _send_diagnostic_request,
    build_flash_messages,
    build_pro_payloads,
    build_projection_context,
    build_projection_system,
    freeze_runtime_case,
    load_reused_current_profile_state,
    load_dataset,
    make_dry_format_profile,
    minimal_policy_dict,
    projection_evidence_audit,
    render_projection_profile,
    run,
    validate_projection_profile,
)


REUSE_STATE_PATH = Path("poc/results/runtime-projection-state-20260710-190955.json")


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=str(DATASET_PATH),
        output=str(root / "results.jsonl"),
        state_output=str(root / "state.json"),
        blind_output=str(root / "blind.jsonl"),
        reuse_current_profile_state=str(REUSE_STATE_PATH),
        fast_model="deepseek-v4-flash",
        thinking_model="deepseek-v4-pro",
        repetitions=2,
        seed=20260710,
        delay=0.0,
        pro_timeout=180.0,
        flash_timeout=60.0,
        dry_run=True,
    )


def _grounded_context():
    context = build_projection_context(AppSettings.load())
    return replace(
        context,
        user_constraint_layer="只能遵守用户确认的输出约束。",
        ai_optimization_layer="保留完整含义和自然语序。不得遗漏否定关系。",
    )


def _valid_profile() -> dict:
    return {
        "version": 1,
        "role": "runtime_profile_projection",
        "language_pair": {"source": "中文", "target": "日本語"},
        "semantic_directives": [
            {
                "instruction": "翻译时保留完整含义。",
                "evidence_layer": "ai_optimization",
                "evidence_quote": "保留完整含义",
            }
        ],
        "style_directives": [],
        "completeness_directives": [],
    }


def _diagnostic(
    content: str,
    *,
    reasoning_len: int = 0,
    finish_reason: str = "stop",
) -> dict:
    return {
        "http_status": 200,
        "message_keys": ["content", "reasoning_content"],
        "content_len": len(content),
        "reasoning_len": reasoning_len,
        "finish_reason": finish_reason,
        "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        "latency_s": 0.25,
        "content": content,
    }


def _formal_settings() -> AppSettings:
    settings = AppSettings.load()
    settings.ai.base_url = "https://example.invalid/v1"
    settings.ai.api_key = "secret-that-must-not-be-persisted"
    settings.ai.fast_model = "deepseek-v4-flash"
    settings.ai.thinking_model = "deepseek-v4-pro"
    return settings


class RuntimeProjectionDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)

    def test_dataset_sha256_and_frozen_shape(self) -> None:
        actual = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        self.assertEqual(actual, EXPECTED_DATASET_SHA256)
        self.assertEqual(self.dataset.version, 1)
        self.assertEqual(self.dataset.status, "frozen_poc")
        self.assertEqual(tuple(case.source for case in self.dataset.cases), EXPECTED_CASE_SOURCES)
        self.assertEqual(tuple(case.scope for case in self.dataset.cases), EXPECTED_CASE_SCOPES)
        self.assertEqual(len(self.dataset.cases), 12)

    def test_all_sources_and_ids_are_unique_with_expected_scope_counts(self) -> None:
        self.assertEqual(len({case.id for case in self.dataset.cases}), 12)
        self.assertEqual(len({case.source for case in self.dataset.cases}), 12)
        scopes = [case.scope for case in self.dataset.cases]
        self.assertEqual(scopes.count("runtime_regression"), 8)
        self.assertEqual(scopes.count("control"), 4)

    def test_sources_do_not_duplicate_any_of_three_old_datasets(self) -> None:
        old_sources: set[str] = set()
        for path in LEGACY_DATASET_PATHS:
            raw = json.loads(path.read_text(encoding="utf-8"))
            old_sources.update(
                str(row.get("source", "")).strip()
                for row in raw.get("cases", [])
                if isinstance(row, dict)
            )
        self.assertFalse(set(EXPECTED_CASE_SOURCES) & old_sources)


class ProjectionSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)
        cls.context = _grounded_context()

    def test_valid_profile_has_exact_schema_and_grounded_audit(self) -> None:
        profile = validate_projection_profile(
            _valid_profile(), context=self.context, dataset=self.dataset
        )
        self.assertEqual(
            set(profile),
            {"version", "role", "language_pair", *PROFILE_DIRECTIVE_FIELDS},
        )
        audit = projection_evidence_audit(profile, self.context)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["directive_count"], 1)
        self.assertFalse(audit["forbidden_generated_knowledge"])

    def test_rejects_extra_top_level_and_directive_keys(self) -> None:
        extra = deepcopy(_valid_profile())
        extra["glossary"] = []
        with self.assertRaises(RuntimeProjectionPocError):
            validate_projection_profile(extra, context=self.context, dataset=self.dataset)
        nested = deepcopy(_valid_profile())
        nested["semantic_directives"][0]["confidence"] = 1
        with self.assertRaises(RuntimeProjectionPocError):
            validate_projection_profile(nested, context=self.context, dataset=self.dataset)

    def test_rejects_wrong_language_pair(self) -> None:
        profile = deepcopy(_valid_profile())
        profile["language_pair"]["target"] = "English"
        with self.assertRaises(RuntimeProjectionPocError):
            validate_projection_profile(profile, context=self.context, dataset=self.dataset)

    def test_rejects_invalid_evidence_layer(self) -> None:
        profile = deepcopy(_valid_profile())
        profile["semantic_directives"][0]["evidence_layer"] = "model_guess"
        with self.assertRaises(RuntimeProjectionPocError):
            validate_projection_profile(profile, context=self.context, dataset=self.dataset)

    def test_rejects_evidence_quote_missing_from_named_layer(self) -> None:
        profile = deepcopy(_valid_profile())
        profile["semantic_directives"][0]["evidence_quote"] = "输入中不存在的要求"
        with self.assertRaises(RuntimeProjectionPocError):
            validate_projection_profile(profile, context=self.context, dataset=self.dataset)

    def test_rejects_glossary_mapping_and_example_content(self) -> None:
        for bad_instruction in (
            "Create a glossary for software words.",
            "接口 -> いんたあふぇえす",
            "例如可以输出某个测试译文。",
        ):
            with self.subTest(bad_instruction=bad_instruction):
                profile = deepcopy(_valid_profile())
                profile["semantic_directives"][0]["instruction"] = bad_instruction
                with self.assertRaises(RuntimeProjectionPocError):
                    validate_projection_profile(
                        profile, context=self.context, dataset=self.dataset
                    )

    def test_rejects_every_format_concept_reserved_for_active_policy(self) -> None:
        forbidden = (
            "Use hiragana only",
            "Do not use katakana",
            "Remove kanji",
            "Avoid romaji",
            "只能输出平假名",
            "使用括号标记",
            "Wrap it in brackets",
            "Insert two spaces",
            "Remove punctuation",
            "Allow U+3042",
            "Tokenize the output",
            "Respect each word boundary",
            "Separate every lexical unit",
            "输出 [标记]",
        )
        for instruction in forbidden:
            with self.subTest(instruction=instruction):
                profile = deepcopy(_valid_profile())
                profile["semantic_directives"][0]["instruction"] = instruction
                with self.assertRaises(RuntimeProjectionPocError):
                    validate_projection_profile(
                        profile, context=self.context, dataset=self.dataset
                    )

    def test_rejects_profile_with_all_directive_lists_empty(self) -> None:
        profile = deepcopy(_valid_profile())
        for field in PROFILE_DIRECTIVE_FIELDS:
            profile[field] = []
        with self.assertRaises(RuntimeProjectionPocError):
            validate_projection_profile(profile, context=self.context, dataset=self.dataset)

    def test_rejects_evaluation_source_leak(self) -> None:
        leaked = replace(
            self.context,
            ai_optimization_layer=(
                self.context.ai_optimization_layer + "\n" + self.dataset.cases[0].source
            ),
        )
        profile = deepcopy(_valid_profile())
        profile["semantic_directives"][0] = {
            "instruction": self.dataset.cases[0].source,
            "evidence_layer": "ai_optimization",
            "evidence_quote": self.dataset.cases[0].source,
        }
        with self.assertRaises(RuntimeProjectionPocError):
            validate_projection_profile(profile, context=leaked, dataset=self.dataset)

    def test_rejects_rendered_profile_over_1800_characters(self) -> None:
        profile = deepcopy(_valid_profile())
        long_instruction = "保持完整语义和自然表达" * 14
        directive = {
            "instruction": long_instruction,
            "evidence_layer": "ai_optimization",
            "evidence_quote": "保留完整含义",
        }
        for field in PROFILE_DIRECTIVE_FIELDS:
            profile[field] = [deepcopy(directive) for _ in range(6)]
        with self.assertRaises(RuntimeProjectionPocError):
            validate_projection_profile(profile, context=self.context, dataset=self.dataset)

    def test_runtime_renderer_omits_evidence_metadata_and_quote(self) -> None:
        profile = validate_projection_profile(
            _valid_profile(), context=self.context, dataset=self.dataset
        )
        rendered = render_projection_profile(profile)
        self.assertIn("翻译时保留完整含义。", rendered)
        self.assertNotIn("evidence_layer", rendered)
        self.assertNotIn("ai_optimization", rendered)
        self.assertNotIn('"evidence_quote"', rendered)


class RuntimeIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)
        cls.context = build_projection_context(AppSettings.load())
        cls.grounded_context = _grounded_context()
        cls.profile = validate_projection_profile(
            _valid_profile(), context=cls.grounded_context, dataset=cls.dataset
        )
        cls.rendered = render_projection_profile(cls.profile)

    def test_exactly_two_groups_and_fixed_formal_call_scale(self) -> None:
        self.assertEqual(GROUPS, ("CURRENT_RUNTIME", "PRO_RUNTIME_PROJECTION"))
        self.assertEqual(12 * len(GROUPS) * 2, 48)
        self.assertEqual(2, len(GROUPS))

    def test_both_pro_payloads_hide_all_evaluation_sources(self) -> None:
        payloads = build_pro_payloads(self.context, "deepseek-v4-pro")
        self.assertEqual(set(payloads), set(GROUPS))
        rendered = json.dumps(payloads, ensure_ascii=False)
        for case in self.dataset.cases:
            self.assertNotIn(case.source, rendered)
        projection_input = payloads["PRO_RUNTIME_PROJECTION"]["messages"][1]["content"]
        self.assertNotIn("review_focus", projection_input)
        self.assertNotIn("fixture_translation", projection_input)
        self.assertNotIn("Knowledge Reference Layer", projection_input)
        self.assertNotIn("Fixed Template Layer", projection_input)
        compiler_prompt = payloads["PRO_RUNTIME_PROJECTION"]["messages"][0]["content"]
        for marker in ("hiragana", "brackets", "whitespace", "punctuation", "ACTIVE_POLICY"):
            self.assertIn(marker, compiler_prompt)

    def test_compiler_prompt_preserves_protocol_length_checkpoints(self) -> None:
        payloads = build_pro_payloads(self.context, "deepseek-v4-pro")
        compiler = payloads["PRO_RUNTIME_PROJECTION"]["messages"][0]["content"]
        for marker in (
            "at least 1 item in total",
            "non-empty",
            "240 characters",
            "160 characters",
            "shortest continuous quote",
            "Do not copy a whole paragraph",
            "If no legal short evidence quote",
            "verify the exact Schema",
        ):
            self.assertIn(marker, compiler)

    def test_projection_system_replaces_full_compiled_prompt_and_markdown_layers(self) -> None:
        system = build_projection_system(self.grounded_context, self.rendered)
        self.assertNotEqual(system, self.grounded_context.production.system)
        self.assertNotIn("# Instant Translate Compiled Prompt", system)
        self.assertNotIn("## User Constraint Layer", system)
        self.assertNotIn("## AI Optimization Layer", system)
        if self.grounded_context.compiled_prompt:
            self.assertNotIn(self.grounded_context.compiled_prompt, system)

    def test_projection_system_omits_current_format_checklist(self) -> None:
        system = build_projection_system(self.grounded_context, self.rendered)
        self.assertNotIn(make_dry_format_profile(), system)
        self.assertNotIn(_format_user_prompt(), system)
        self.assertNotIn("涉及到软件工程相关的词应该有[]包裹起来", system)

    def test_projection_system_contains_only_minimal_policy_shape(self) -> None:
        system = build_projection_system(self.grounded_context, self.rendered)
        policy = minimal_policy_dict(self.grounded_context.production.policy)
        serialized = json.dumps(
            policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self.assertIn(f"<ACTIVE_POLICY>\n{serialized}\n</ACTIVE_POLICY>", system)
        self.assertIn("ACTIVE_POLICY is the user-confirmed authoritative output contract", system)
        self.assertIn("AGENT_PROFILE", system)
        self.assertIn("must never override ACTIVE_POLICY", system)
        self.assertNotIn('"source_text"', serialized)
        self.assertNotIn("model_instruction", serialized)
        for rule in policy["rules"]:
            self.assertEqual(set(rule), {"type", "params", "enforcement", "scope"})

    def test_minimal_policy_filters_model_instruction_rules(self) -> None:
        policy = ConstraintPolicy.from_dict(
            {
                "version": 1,
                "rules": [
                    {
                        "type": "model_instruction",
                        "params": {"text": "unstructured prose"},
                        "enforcement": "model",
                        "scope": {},
                    },
                    {
                        "type": "allowed_characters",
                        "params": {
                            "scripts": ["hiragana"],
                            "literals": ["[", "]"],
                            "allow_whitespace": True,
                        },
                        "enforcement": "both",
                        "scope": {"source": "中文", "target": "日本語"},
                    },
                ],
            },
            strict=True,
        )
        minimal = minimal_policy_dict(policy)
        self.assertEqual([rule["type"] for rule in minimal["rules"]], ["allowed_characters"])

    def test_groups_share_frozen_context_except_rule_checklist_delivery(self) -> None:
        case = self.dataset.cases[0]
        frozen = freeze_runtime_case(
            self.grounded_context,
            case,
            current_format_profile=make_dry_format_profile(),
        )
        current = build_flash_messages(
            group="CURRENT_RUNTIME",
            context=self.grounded_context,
            frozen_request=frozen,
            current_format_profile=make_dry_format_profile(),
            rendered_projection=self.rendered,
        )
        projected = build_flash_messages(
            group="PRO_RUNTIME_PROJECTION",
            context=self.grounded_context,
            frozen_request=frozen,
            current_format_profile=make_dry_format_profile(),
            rendered_projection=self.rendered,
        )
        self.assertIn("<RULE_CHECKLIST>", current[-1]["content"])
        self.assertNotIn("<RULE_CHECKLIST>", projected[-1]["content"])
        self.assertEqual(
            _remove_rule_checklist(current[-1]["content"]),
            projected[-1]["content"],
        )
        self.assertIn(case.source, current[-1]["content"])
        self.assertEqual(
            self.grounded_context.production.policy_digest,
            self.context.production.policy_digest,
        )
        self.assertEqual(
            self.grounded_context.production.reference_digest,
            self.context.production.reference_digest,
        )

    def test_current_runtime_has_production_three_message_bootstrap(self) -> None:
        frozen = freeze_runtime_case(
            self.context,
            self.dataset.cases[0],
            current_format_profile=make_dry_format_profile(),
        )
        messages = build_flash_messages(
            group="CURRENT_RUNTIME",
            context=self.context,
            frozen_request=frozen,
            current_format_profile=make_dry_format_profile(),
            rendered_projection=self.rendered,
        )
        self.assertEqual([row["role"] for row in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[0]["content"], self.context.production.system)
        self.assertEqual(messages[2]["content"], make_dry_format_profile())
        self.assertIn("<RULE_CHECKLIST>", messages[-1]["content"])
        self.assertIn(make_dry_format_profile(), messages[-1]["content"])

    def test_projection_has_no_rule_checklist_message(self) -> None:
        frozen = freeze_runtime_case(
            self.grounded_context,
            self.dataset.cases[0],
            current_format_profile=make_dry_format_profile(),
        )
        messages = build_flash_messages(
            group="PRO_RUNTIME_PROJECTION",
            context=self.grounded_context,
            frozen_request=frozen,
            current_format_profile=make_dry_format_profile(),
            rendered_projection=self.rendered,
        )
        self.assertEqual([row["role"] for row in messages], ["system", "user"])
        self.assertNotIn(make_dry_format_profile(), json.dumps(messages, ensure_ascii=False))
        self.assertNotIn("<RULE_CHECKLIST>", messages[-1]["content"])


def _format_user_prompt() -> str:
    return "请逐条列出上述翻译规则中最关键的格式要求"


def _remove_rule_checklist(text: str) -> str:
    return re.sub(
        r"<RULE_CHECKLIST>\n.*?\n</RULE_CHECKLIST>\n"
        r"翻译完成后，对照上述清单逐条检查你的输出。"
        r"如有违规请修正，最终结果用 <final> 标签包裹输出。\n",
        "",
        text,
        count=1,
        flags=re.S,
    )


class ReusedCurrentProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset(DATASET_PATH)
        cls.context = build_projection_context(AppSettings.load())

    def test_valid_v2_current_checkpoint_is_revalidated_and_reusable(self) -> None:
        reused = load_reused_current_profile_state(
            REUSE_STATE_PATH,
            context=self.context,
            dataset=self.dataset,
            fast_model="deepseek-v4-flash",
            thinking_model="deepseek-v4-pro",
        )
        self.assertEqual(reused["validation"], "passed")
        self.assertTrue(Path(reused["absolute_path"]).is_absolute())
        self.assertEqual(len(reused["state_sha256"]), 64)
        self.assertEqual(len(reused["profile_hash"]), 64)
        self.assertTrue(reused["profile_record"]["reused"])

    def test_all_context_and_model_mismatches_fail_before_network(self) -> None:
        original = json.loads(REUSE_STATE_PATH.read_text(encoding="utf-8"))
        mismatches = {
            "prompt_hash": "wrong-prompt",
            "policy_digest": "wrong-policy",
            "reference_digest": "wrong-reference",
            "fast_model": "wrong-fast-model",
            "thinking_model": "wrong-thinking-model",
        }
        for key, value in mismatches.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                mutated = deepcopy(original)
                mutated["metadata"][key] = value
                reuse_path = Path(tmp) / "reuse.json"
                reuse_path.write_text(json.dumps(mutated, ensure_ascii=False), encoding="utf-8")
                args = _args(Path(tmp))
                args.reuse_current_profile_state = str(reuse_path)
                with patch(
                    "poc.poc_agent_runtime_projection._send_diagnostic_request",
                    side_effect=AssertionError("network must not be called"),
                ) as request:
                    with self.assertRaises(RuntimeProjectionPocError):
                        run(args)
                request.assert_not_called()

    def test_current_raw_profile_is_revalidated_before_network(self) -> None:
        original = json.loads(REUSE_STATE_PATH.read_text(encoding="utf-8"))
        original["profiles"]["CURRENT_RUNTIME"]["raw_response"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            reuse_path = Path(tmp) / "reuse.json"
            reuse_path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
            args = _args(Path(tmp))
            args.reuse_current_profile_state = str(reuse_path)
            with patch(
                "poc.poc_agent_runtime_projection._send_diagnostic_request",
                side_effect=AssertionError("network must not be called"),
            ) as request:
                with self.assertRaises(RuntimeProjectionPocError):
                    run(args)
            request.assert_not_called()


class ProgressiveStateFailureTests(unittest.TestCase):
    def test_diagnostic_request_returns_empty_2xx_content_without_reasoning_body(self) -> None:
        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning_content": "actual hidden reasoning",
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 9},
                }

        with patch(
            "poc.poc_agent_runtime_projection.httpx.post",
            return_value=FakeResponse(),
        ):
            diagnostic = _send_diagnostic_request(
                base_url="https://example.invalid/v1",
                api_key="secret",
                payload={"model": "test", "messages": []},
                timeout_seconds=1,
            )
        self.assertEqual(diagnostic["content"], "")
        self.assertEqual(diagnostic["content_len"], 0)
        self.assertEqual(diagnostic["reasoning_len"], len("actual hidden reasoning"))
        self.assertEqual(diagnostic["finish_reason"], "length")
        self.assertNotIn("reasoning_content", diagnostic)
        self.assertNotIn("actual hidden reasoning", json.dumps(diagnostic))

    def test_empty_projection_content_preserves_http_diagnostics_before_error(self) -> None:
        context = _grounded_context()
        settings = _formal_settings()
        empty_diagnostic = _diagnostic(
            "",
            reasoning_len=777,
            finish_reason="length",
        )
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            args.dry_run = False
            with patch(
                "poc.poc_agent_runtime_projection.AppSettings.load",
                return_value=settings,
            ), patch(
                "poc.poc_agent_runtime_projection.build_projection_context",
                return_value=context,
            ), patch(
                "poc.poc_agent_runtime_projection._send_diagnostic_request",
                return_value=empty_diagnostic,
            ) as request, patch("builtins.print"):
                with self.assertRaises(TranslationError) as raised:
                    run(args)
            self.assertIn("empty content", str(raised.exception))
            self.assertEqual(request.call_count, 1)
            state = json.loads(Path(args.state_output).read_text(encoding="utf-8"))
            serialized = json.dumps(state, ensure_ascii=False)

        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["failure_stage"], "projection_pro_empty_content")
        self.assertEqual(state["attempted_pro_calls"], 1)
        self.assertEqual(state["usable_pro_responses"], 0)
        self.assertEqual(state["attempted_flash_calls"], 0)
        self.assertEqual(state["usable_flash_responses"], 0)
        projection = state["profiles"]["PRO_RUNTIME_PROJECTION"]
        self.assertEqual(projection["http_status"], 200)
        self.assertEqual(projection["message_keys"], ["content", "reasoning_content"])
        self.assertEqual(projection["content_len"], 0)
        self.assertEqual(projection["reasoning_len"], 777)
        self.assertEqual(projection["finish_reason"], "length")
        self.assertTrue(projection["usage"])
        self.assertEqual(projection["raw_response"], "")
        self.assertNotIn("secret-that-must-not-be-persisted", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("actual hidden reasoning", serialized)

    def test_successful_v3_calls_only_one_pro_and_original_48_flash_requests(self) -> None:
        context = _grounded_context()
        settings = _formal_settings()
        projection_content = json.dumps(_valid_profile(), ensure_ascii=False)
        diagnostics = [_diagnostic(projection_content)] + [
            _diagnostic("てすと") for _ in range(48)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(Path(tmp))
            args.dry_run = False
            with patch(
                "poc.poc_agent_runtime_projection.AppSettings.load",
                return_value=settings,
            ), patch(
                "poc.poc_agent_runtime_projection.build_projection_context",
                return_value=context,
            ), patch(
                "poc.poc_agent_runtime_projection._send_diagnostic_request",
                side_effect=diagnostics,
            ) as request, patch("builtins.print"):
                output, state_path, _ = run(args)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(request.call_count, 49)
        self.assertTrue(state["profiles"]["CURRENT_RUNTIME"]["reused"])
        self.assertEqual(state["attempted_pro_calls"], 1)
        self.assertEqual(state["usable_pro_responses"], 1)
        self.assertEqual(state["attempted_flash_calls"], 48)
        self.assertEqual(state["usable_flash_responses"], 48)
        self.assertEqual(len([row for row in rows if row.get("type") == "result"]), 48)


class DryRunIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        original_memory_match = FeedbackStore.match_memory_rules
        original_reference_protect = ReferenceStore.protect
        with patch(
            "poc.poc_agent_runtime_projection._send_diagnostic_request",
            side_effect=AssertionError("network called"),
        ), patch("builtins.print"), patch(
            "poc.poc_agent_semantic_profile.OutputNormalizer.normalize_with_policy",
            wraps=OutputNormalizer.normalize_with_policy,
        ) as normalizer, patch(
            "poc.poc_agent_semantic_profile.OutputValidator.validate",
            wraps=OutputValidator.validate,
        ) as validator, patch.object(
            FeedbackStore,
            "match_memory_rules",
            autospec=True,
            side_effect=lambda store, *args, **kwargs: original_memory_match(
                store, *args, **kwargs
            ),
        ) as memory_match, patch.object(
            ReferenceStore,
            "protect",
            autospec=True,
            side_effect=lambda store, *args, **kwargs: original_reference_protect(
                store, *args, **kwargs
            ),
        ) as reference_protect:
            output, state, blind = run(_args(root))
            cls.normalizer_calls = normalizer.call_count
            cls.validator_calls = validator.call_count
            cls.memory_match_calls = memory_match.call_count
            cls.reference_protect_calls = reference_protect.call_count
        cls.output_rows = [
            json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
        ]
        cls.state = json.loads(state.read_text(encoding="utf-8"))
        cls.blind_rows = [
            json.loads(line) for line in blind.read_text(encoding="utf-8").splitlines()
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_dry_run_never_calls_network_and_writes_exactly_48_results(self) -> None:
        results = [row for row in self.output_rows if row.get("type") == "result"]
        self.assertEqual(len(results), 48)
        self.assertEqual({row["group"] for row in results}, set(GROUPS))
        self.assertTrue(all(row["finish_reason"] == "stop" for row in results))
        self.assertTrue(all(not row["error"] for row in results))

    def test_both_groups_use_production_normalizer_and_validator(self) -> None:
        self.assertGreaterEqual(self.normalizer_calls, 48)
        self.assertEqual(self.validator_calls, 48)
        results = [row for row in self.output_rows if row.get("type") == "result"]
        self.assertTrue(all(row["production_validation"]["ok"] for row in results))

    def test_reference_and_feedback_are_read_once_per_case_before_jobs(self) -> None:
        self.assertEqual(self.memory_match_calls, 12)
        self.assertEqual(self.reference_protect_calls, 12)

    def test_groups_and_repetitions_share_frozen_memory_and_reference_results(self) -> None:
        results = [row for row in self.output_rows if row.get("type") == "result"]
        by_case: dict[str, list[dict]] = {}
        for row in results:
            by_case.setdefault(row["case_id"], []).append(row)
        self.assertEqual(len(by_case), 12)
        for case_id, rows in by_case.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(len(rows), 4)
                self.assertEqual(
                    {tuple(row["matched_reference_ids"]) for row in rows},
                    {tuple(rows[0]["matched_reference_ids"])},
                )
                self.assertEqual(
                    {tuple(row["matched_memory_ids"]) for row in rows},
                    {tuple(rows[0]["matched_memory_ids"])},
                )
                self.assertEqual(
                    {tuple(row["source_technical_terms"]) for row in rows},
                    {tuple(rows[0]["source_technical_terms"])},
                )

    def test_blind_file_has_exact_non_leaking_schema(self) -> None:
        self.assertEqual(len(self.blind_rows), 48)
        expected = {
            "blind_id",
            "source",
            "review_focus",
            "translation",
            "semantic_fidelity_0_to_5",
            "naturalness_0_to_5",
            "meaning_error",
            "review_notes",
        }
        for row in self.blind_rows:
            self.assertEqual(set(row), expected)
            self.assertNotIn("group", row)
            self.assertNotIn("repetition", row)

    def test_summary_contains_complete_predeclared_decision_gate(self) -> None:
        summary = next(row for row in self.output_rows if row.get("type") == "summary")
        gate = json.dumps(summary["decision_gate"], ensure_ascii=False)
        for marker in (">=0.4", ">=0.3", ">=30%", "<=15%", ">=20%"):
            self.assertIn(marker, gate)
        self.assertIn("zero ungrounded directives", gate)

    def test_dry_state_records_reuse_protocol_and_completed_status(self) -> None:
        self.assertEqual(self.state["status"], "completed")
        self.assertEqual(self.state["projection_protocol_version"], 3)
        self.assertIn("Reuse the already validated CURRENT_RUNTIME profile", self.state["projection_protocol_change"])
        self.assertEqual(self.state["reused_current_profile_validation"], "passed")
        self.assertEqual(len(self.state["reused_current_profile_state_sha256"]), 64)
        self.assertEqual(len(self.state["reused_current_profile_hash"]), 64)
        self.assertTrue(Path(self.state["reused_current_profile_state"]).is_absolute())
        self.assertEqual(self.state["attempted_pro_calls"], 0)
        self.assertEqual(self.state["usable_pro_responses"], 0)
        self.assertEqual(self.state["attempted_flash_calls"], 0)
        self.assertEqual(self.state["usable_flash_responses"], 0)
        self.assertEqual(self.state["executed_pro_calls"], 0)
        self.assertEqual(self.state["executed_flash_calls"], 0)
        self.assertEqual(set(self.state["profiles"]), set(GROUPS))
        self.assertEqual(set(self.state["system_prompts"]), set(GROUPS))
        metadata = self.state["metadata"]
        self.assertTrue(metadata["prompt_hash"])
        self.assertTrue(metadata["policy_digest"])
        self.assertTrue(metadata["reference_digest"])
        self.assertEqual(
            metadata["formal_call_design"],
            {
                "reused_current_pro_profiles": 1,
                "new_projection_pro_calls": 1,
                "flash": 48,
                "pro_judge": 0,
                "retry": 0,
            },
        )
        self.assertEqual(metadata["projection_protocol_version"], 3)
        self.assertEqual(
            {row["validation_status"] for row in self.state["profiles"].values()},
            {"passed"},
        )
        self.assertTrue(self.state["profiles"]["CURRENT_RUNTIME"]["reused"])
        audit = self.state["profiles"]["PRO_RUNTIME_PROJECTION"]["evidence_validation"]
        self.assertTrue(audit["valid"])
        self.assertFalse(audit["forbidden_format_concepts"])
        self.assertEqual(len(self.state["frozen_request_contexts"]), 12)
        for frozen in self.state["frozen_request_contexts"].values():
            self.assertIn("matched_memory_ids", frozen)
            self.assertIn("matched_reference_ids", frozen)

    def test_metadata_proves_no_judge_retry_or_executed_api_calls(self) -> None:
        metadata = next(row for row in self.output_rows if row.get("type") == "run")
        self.assertEqual(metadata["projection_protocol_version"], 3)
        self.assertIn("only one new Projection Pro call", metadata["projection_protocol_change"])
        self.assertEqual(metadata["pro_judge_calls"], 0)
        self.assertEqual(metadata["automatic_retries"], 0)
        self.assertEqual(
            metadata["formal_call_design"],
            {
                "reused_current_pro_profiles": 1,
                "new_projection_pro_calls": 1,
                "flash": 48,
                "pro_judge": 0,
                "retry": 0,
            },
        )
        self.assertEqual(metadata["attempted_api_calls"], {"pro": 0, "flash": 0})
        self.assertEqual(metadata["usable_api_responses"], {"pro": 0, "flash": 0})
        self.assertEqual(metadata["executed_api_calls"], {"pro": 0, "flash": 0})

    def test_artifacts_do_not_persist_api_key_or_reasoning_content(self) -> None:
        combined = json.dumps(self.output_rows, ensure_ascii=False) + json.dumps(
            self.state, ensure_ascii=False
        )
        self.assertNotIn("api_key", combined)
        self.assertNotIn("reasoning_content", combined)


if __name__ == "__main__":
    unittest.main()
