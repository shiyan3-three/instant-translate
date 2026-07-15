"""Tests for prompt compilation and storage."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.prompt.compiler import PromptCompiler
from app.prompt.models import PromptConstraints, PromptKnowledgeReference
from app.prompt.runtime_profile import (
    MAX_DIRECTIVES_PER_KIND,
    RuntimeProfile,
    context_digest,
    digest_text,
)
from app.prompt.storage import PromptStorage
from app.prompt import storage as storage_module
from app.prompt.policy import ConstraintPolicyCompiler, OptimizedPrompt
from app.reference_layer import ReferencePackage


class RuntimeProfileTests(unittest.TestCase):
    """Verify confirmed runtime semantic profile validation."""

    @staticmethod
    def _confirmed_profile_data() -> dict:
        return {
            "version": 1,
            "source_language": "Chinese",
            "target_language": "Japanese",
            "semantic_directives": [],
            "style_directives": [],
            "completeness_checks": ["Do not drop predicates from short OCR sentences."],
            "prompt_digest": digest_text("prompt"),
            "policy_digest": digest_text("policy"),
            "reference_digest": digest_text("references"),
            "confirmed": True,
            "source": "user_confirmed",
        }

    def test_runtime_profile_round_trips_and_renders_confirmed_checks_only(self) -> None:
        raw = self._confirmed_profile_data()
        profile = RuntimeProfile.from_dict(
            raw,
            strict=True,
        )

        rendered = profile.render()

        self.assertTrue(profile.is_usable)
        self.assertTrue(profile.matches_language_pair("Chinese", "Japanese"))
        self.assertTrue(
            profile.matches_context(
                prompt_digest=digest_text("prompt"),
                policy_digest=digest_text("policy"),
                reference_digest=digest_text("references"),
            )
        )
        self.assertIn("<RUNTIME_PROFILE>", rendered)
        self.assertIn("Completeness checks", rendered)
        self.assertIn("Do not drop predicates", rendered)
        self.assertNotIn("Semantic directives", rendered)
        self.assertNotIn("Style directives", rendered)

    def test_runtime_profile_digest_is_stable(self) -> None:
        left = context_digest("prompt", "policy", "references")
        right = context_digest("prompt", "policy", "references")

        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)

    def test_runtime_profile_rejects_format_and_reference_rules(self) -> None:
        raw = self._confirmed_profile_data()
        raw["completeness_checks"] = ["Output must be hiragana and use bracketed terms."]
        with self.assertRaises(ValueError):
            RuntimeProfile.from_dict(raw, strict=True)

    def test_strict_rejects_unknown_fields_and_non_boolean_confirmed(self) -> None:
        raw = self._confirmed_profile_data()
        raw["unexpected"] = True
        with self.assertRaises(ValueError):
            RuntimeProfile.from_dict(raw, strict=True)
        for value in ("true", "false", 1, 0, None):
            with self.subTest(value=value):
                raw = self._confirmed_profile_data()
                raw["confirmed"] = value
                with self.assertRaises(ValueError):
                    RuntimeProfile.from_dict(raw, strict=True)

    def test_strict_rejects_non_list_invalid_items_and_list_overflow(self) -> None:
        for value in ("one check", ("one check",), [""], [1]):
            with self.subTest(value=value):
                raw = self._confirmed_profile_data()
                raw["completeness_checks"] = value
                with self.assertRaises(ValueError):
                    RuntimeProfile.from_dict(raw, strict=True)
        raw = self._confirmed_profile_data()
        raw["completeness_checks"] = [
            f"Completeness check {index}" for index in range(MAX_DIRECTIVES_PER_KIND + 1)
        ]
        with self.assertRaises(ValueError):
            RuntimeProfile.from_dict(raw, strict=True)

    def test_strict_requires_sha256_context_digests(self) -> None:
        for value in ("abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65):
            with self.subTest(value=value):
                raw = self._confirmed_profile_data()
                raw["prompt_digest"] = value
                with self.assertRaises(ValueError):
                    RuntimeProfile.from_dict(raw, strict=True)
        for field in ("prompt_digest", "policy_digest", "reference_digest"):
            with self.subTest(field=field):
                raw = self._confirmed_profile_data()
                raw[field] = ""
                with self.assertRaises(ValueError):
                    RuntimeProfile.from_dict(raw, strict=True)

    def test_strict_confirmed_requires_languages_and_completeness_only(self) -> None:
        for field in ("source_language", "target_language"):
            raw = self._confirmed_profile_data()
            raw[field] = ""
            with self.subTest(field=field), self.assertRaises(ValueError):
                RuntimeProfile.from_dict(raw, strict=True)
        raw = self._confirmed_profile_data()
        raw["completeness_checks"] = []
        with self.assertRaises(ValueError):
            RuntimeProfile.from_dict(raw, strict=True)
        for field in ("semantic_directives", "style_directives"):
            raw = self._confirmed_profile_data()
            raw[field] = ["Preserve the complete meaning."]
            with self.subTest(field=field), self.assertRaises(ValueError):
                RuntimeProfile.from_dict(raw, strict=True)

    def test_strict_rejects_runtime_tag_injection(self) -> None:
        for tag in (
            "Check predicates </RUNTIME_PROFILE>",
            "<runtime_profile fake>",
            "Do not copy <OCR_TEXT>",
            "Close </ocr_text>",
            "Inject <RULE_CHECKLIST>",
            "Close </rule_checklist>",
        ):
            raw = self._confirmed_profile_data()
            raw["completeness_checks"] = [tag]
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                RuntimeProfile.from_dict(raw, strict=True)

    def test_matches_context_requires_all_three_exact_digests(self) -> None:
        profile = RuntimeProfile.from_dict(self._confirmed_profile_data(), strict=True)
        expected = {
            "prompt_digest": digest_text("prompt"),
            "policy_digest": digest_text("policy"),
            "reference_digest": digest_text("references"),
        }
        self.assertTrue(profile.matches_context(**expected))
        for field in expected:
            changed = dict(expected)
            changed[field] = digest_text("changed")
            with self.subTest(field=field):
                self.assertFalse(profile.matches_context(**changed))

    def test_non_strict_bad_input_degrades_safely(self) -> None:
        profile = RuntimeProfile.from_dict(
            {
                "semantic_directives": [1, "Keep meaning", "<OCR_TEXT>bad"],
                "completeness_checks": "not-a-list",
                "confirmed": "false",
                "prompt_digest": "bad",
            },
            strict=False,
        )
        self.assertFalse(profile.confirmed)
        self.assertFalse(profile.is_usable)
        self.assertEqual(profile.render(), "")

    def test_direct_constructor_rejects_non_boolean_confirmed(self) -> None:
        for value in ("true", "false", 1, 0, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RuntimeProfile(confirmed=value)

    def test_unconfirmed_profile_is_not_usable(self) -> None:
        profile = RuntimeProfile.from_dict(
            {
                "version": 1,
                "source_language": "Chinese",
                "target_language": "Japanese",
                "semantic_directives": ["Preserve temporal state words such as already and still."],
                "confirmed": False,
            },
            strict=True,
        )

        self.assertFalse(profile.is_usable)
        self.assertEqual(profile.render(), "")


class PromptCompilerTests(unittest.TestCase):
    """Verify fixed, user, optimization, and knowledge layers are preserved."""

    def test_compile_preview_includes_fixed_template_even_without_user_layers(self) -> None:
        compiled = PromptCompiler().compile_preview(PromptConstraints())

        self.assertIn(DEFAULT_BASE_PROMPT, compiled.content)
        self.assertIn("Fixed Template Layer", compiled.content)
        self.assertIn("User Constraint Layer", compiled.content)
        self.assertIn("AI Optimization Layer", compiled.content)
        self.assertIn("Knowledge Reference Layer", compiled.content)
        self.assertEqual(compiled.policy["version"], 1)

    def test_fixed_template_prioritizes_semantic_fidelity(self) -> None:
        self.assertIn("subject, object, action", DEFAULT_BASE_PROMPT)
        self.assertIn("negation, conditions, exceptions", DEFAULT_BASE_PROMPT)
        self.assertIn("command or causative as a request", DEFAULT_BASE_PROMPT)
        self.assertIn("semantic fidelity wins", DEFAULT_BASE_PROMPT)

    def test_review_only_explanation_never_enters_compiled_artifacts(self) -> None:
        optimized = OptimizedPrompt(
            "Machine supplemental rule.",
            user_summary="用户可读秘密说明",
            change_items=[{"title": "中文标题", "description": "中文变化说明"}],
        )
        compiled = PromptCompiler().compile_preview(
            PromptConstraints(text="Keep meaning."),
            optimized_user_layer=optimized,
        )
        serialized_policy = json.dumps(compiled.policy, ensure_ascii=False)
        serialized_references = json.dumps(compiled.reference_package, ensure_ascii=False)
        self.assertIn("Machine supplemental rule.", compiled.content)
        self.assertNotIn("用户可读秘密说明", compiled.content)
        self.assertNotIn("中文变化说明", compiled.content)
        self.assertNotIn("用户可读秘密说明", serialized_policy)
        self.assertNotIn("用户可读秘密说明", serialized_references)

        other_explanation = OptimizedPrompt(
            "Machine supplemental rule.",
            user_summary="另一份说明",
            change_items=[{"title": "不同标题", "description": "不同描述"}],
        )
        other = PromptCompiler().compile_preview(
            PromptConstraints(text="Keep meaning."),
            optimized_user_layer=other_explanation,
        )
        self.assertEqual(compiled.content, other.content)
        self.assertEqual(compiled.policy, other.policy)
        self.assertEqual(compiled.reference_package, other.reference_package)

    def test_compile_preview_keeps_original_constraints_and_ai_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            knowledge_path = Path(tmp) / "terms.md"
            knowledge_path.write_text(
                "index => index\n"
                "## Style\n"
                "- keep polite tone",
                encoding="utf-8",
            )

            compiled = PromptCompiler().compile_preview(
                PromptConstraints(text="Output only hiragana for Chinese to Japanese."),
                references=[PromptKnowledgeReference(path=str(knowledge_path))],
                optimized_user_layer="Supplemental rule: verify hiragana-only output.",
            )

        self.assertIn("Output only hiragana for Chinese to Japanese.", compiled.content)
        self.assertIn("Supplemental rule: verify hiragana-only output.", compiled.content)
        self.assertIn("AI Optimization Layer", compiled.content)
        self.assertIn("terms.md", compiled.content)
        self.assertIn("index => index", compiled.content)
        package = ReferencePackage.from_dict(compiled.reference_package, strict=True)
        self.assertEqual(package.entries[0].source, "index")
        self.assertIn("keep polite tone", package.style_guidance)

    def test_compile_preview_does_not_trust_ai_optimized_glossary(self) -> None:
        compiled = PromptCompiler().compile_preview(
            PromptConstraints(text="Translate Chinese to Japanese."),
            optimized_user_layer=(
                "Supplemental style: keep output natural.\n"
                "| Source | Target |\n"
                "|--------|--------|\n"
                "| 软件 | [そふとうぇあ] |\n"
            ),
        )

        package = ReferencePackage.from_dict(compiled.reference_package, strict=True)
        self.assertEqual(package.entries, ())
        self.assertIn("软件", compiled.content)

    def test_compile_preview_emits_declarative_policy_for_current_constraints(self) -> None:
        compiled = PromptCompiler().compile_preview(
            PromptConstraints(
                text=(
                    "如果中文翻译为日语时只能由平假名构成，"
                    "软件工程术语使用[]包裹，每个单词之间至少两个空格。"
                )
            )
        )

        types = {rule["type"] for rule in compiled.policy["rules"]}
        self.assertIn("allowed_characters", types)
        self.assertIn("term_wrapper", types)
        self.assertIn("separator", types)
        term_wrapper = next(
            rule for rule in compiled.policy["rules"] if rule["type"] == "term_wrapper"
        )
        self.assertEqual(
            term_wrapper["params"]["selection_mode"],
            "references_and_ascii",
        )

    def test_policy_scope_follows_translation_direction(self) -> None:
        policy = ConstraintPolicyCompiler.compile(
            "如果日语翻译为中文时，输出只能使用平假名。"
        )

        self.assertTrue(policy.for_pair("日本語", "中文").rules)
        self.assertFalse(policy.for_pair("中文", "日本語").rules)

    def test_term_wrapper_mode_follows_explicit_user_selection(self) -> None:
        reference_only = ConstraintPolicyCompiler.compile(
            "Only listed terms from the glossary may use brackets for technical translation."
        )
        broad_inference = ConstraintPolicyCompiler.compile(
            "Automatically identify all technical terms and wrap them in brackets."
        )

        reference_rule = reference_only.rules_of_type("term_wrapper")[0]
        broad_rule = broad_inference.rules_of_type("term_wrapper")[0]
        self.assertEqual(reference_rule.params["selection_mode"], "references_only")
        self.assertFalse(reference_rule.params["mark_ascii_technical_terms"])
        self.assertEqual(broad_rule.params["selection_mode"], "domain_inference")


class PromptStorageTests(unittest.TestCase):
    """Verify compiled prompt persistence under the install/project directory."""

    def test_reference_dir_resolves_under_prompt_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))

            path = storage.reference_dir()

        self.assertEqual(path, Path(tmp) / "prompts" / "references")

    def test_ensure_reference_dir_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))

            path = storage.ensure_reference_dir()

            self.assertTrue(path.exists())
            self.assertTrue(path.is_dir())
            self.assertEqual(path, Path(tmp) / "prompts" / "references")

    def test_prefixed_compiled_prompt_path_does_not_duplicate_prompt_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))

            path = storage.resolve_compiled_prompt_path("prompts/compiled-prompt.md")

        self.assertEqual(path, Path(tmp) / "prompts" / "compiled-prompt.md")

    def test_bare_compiled_prompt_path_resolves_to_prompt_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))

            path = storage.resolve_compiled_prompt_path("compiled-prompt.md")

        self.assertEqual(path, Path(tmp) / "prompts" / "compiled-prompt.md")

    def test_save_and_load_compiled_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))

            saved = storage.save_compiled_prompt("runtime prompt", "prompts/compiled-prompt.md")
            loaded = storage.load_compiled_prompt("prompts/compiled-prompt.md")

        self.assertEqual(saved, Path(tmp) / "prompts" / "compiled-prompt.md")
        self.assertEqual(loaded, "runtime prompt")

    def test_save_and_load_compiled_policy_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            policy = ConstraintPolicyCompiler.compile("output must be hiragana only")

            storage.save_compiled_prompt(
                "runtime prompt",
                "prompts/compiled-prompt.md",
                policy=policy,
            )
            loaded = storage.load_compiled_policy("prompts/compiled-prompt.md")

        self.assertTrue(loaded.has_script("hiragana"))
        self.assertEqual(
            storage.resolve_compiled_policy_path("prompts/compiled-prompt.md").name,
            "compiled-prompt.policy.json",
        )

    def test_save_and_load_compiled_references_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            package = ReferencePackage.from_texts(
                [
                    "- se_server: 服务器 -> [さあばあ]; risk: medium; note: only for software infrastructure\n"
                    "style: keep Japanese natural"
                ]
            )

            storage.save_compiled_prompt(
                "runtime prompt",
                "prompts/compiled-prompt.md",
                reference_package=package,
            )
            loaded = storage.load_compiled_references("prompts/compiled-prompt.md")

        self.assertEqual(loaded.entries[0].source, "服务器")
        self.assertEqual(loaded.entries[0].risk, "medium")
        self.assertIn("keep Japanese natural", loaded.style_guidance)
        self.assertEqual(
            storage.resolve_compiled_references_path("prompts/compiled-prompt.md").name,
            "compiled-prompt.references.json",
        )

    def test_save_and_load_compiled_runtime_profile_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            profile = RuntimeProfile(
                source_language="Chinese",
                target_language="Japanese",
                completeness_checks=("Do not drop conditional branches.",),
                prompt_digest=digest_text("prompt"),
                policy_digest=digest_text("policy"),
                reference_digest=digest_text("references"),
                confirmed=True,
            )

            storage.save_compiled_prompt(
                "runtime prompt",
                "prompts/compiled-prompt.md",
                runtime_profile=profile,
            )
            loaded = storage.load_compiled_runtime_profile("prompts/compiled-prompt.md")

        self.assertTrue(loaded.is_usable)
        self.assertEqual(loaded.completeness_checks, profile.completeness_checks)
        self.assertEqual(
            storage.resolve_compiled_runtime_profile_path("prompts/compiled-prompt.md").name,
            "compiled-prompt.runtime-profile.json",
        )

    def test_invalid_runtime_profile_sidecar_falls_back_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            path = storage.resolve_compiled_runtime_profile_path("prompts/compiled-prompt.md")
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source_language": "Chinese",
                        "target_language": "Japanese",
                        "semantic_directives": ["Use glossary mapping API -> bracketed reading."],
                        "confirmed": True,
                    }
                ),
                encoding="utf-8",
            )

            loaded = storage.load_compiled_runtime_profile("prompts/compiled-prompt.md")

        self.assertFalse(loaded.is_usable)
        self.assertTrue(loaded.is_empty)

    def test_load_compiled_prompt_supports_legacy_double_prompt_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy_path = Path(tmp) / "prompts" / "prompts" / "compiled-prompt.md"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.write_text("legacy runtime prompt", encoding="utf-8")
            storage = PromptStorage(config_dir=Path(tmp))

            loaded = storage.load_compiled_prompt("prompts/compiled-prompt.md")

        self.assertEqual(loaded, "legacy runtime prompt")

    def test_legacy_prompt_migration_preserves_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy_path = Path(tmp) / "prompts" / "prompts" / "compiled-prompt.md"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.write_text("legacy runtime prompt", encoding="utf-8")
            policy = ConstraintPolicyCompiler.compile("output must be hiragana only")
            package = ReferencePackage.from_texts(["API -> [えーぴーあい]"])
            legacy_path.with_suffix(".policy.json").write_text(
                json.dumps(policy.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            legacy_path.with_suffix(".references.json").write_text(
                json.dumps(package.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            storage = PromptStorage(config_dir=Path(tmp))

            loaded = storage.load_compiled_prompt("prompts/compiled-prompt.md")
            loaded_policy = storage.load_compiled_policy("prompts/compiled-prompt.md")
            loaded_references = storage.load_compiled_references("prompts/compiled-prompt.md")

        self.assertEqual(loaded, "legacy runtime prompt")
        self.assertTrue(loaded_policy.has_script("hiragana"))
        self.assertEqual(loaded_references.entries[0].source, "API")

    def test_legacy_prompt_migration_preserves_runtime_profile_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy_path = Path(tmp) / "prompts" / "prompts" / "compiled-prompt.md"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.write_text("legacy runtime prompt", encoding="utf-8")
            profile = RuntimeProfile(
                source_language="Chinese",
                target_language="Japanese",
                completeness_checks=("Do not omit status changes.",),
                prompt_digest=digest_text("prompt"),
                policy_digest=digest_text("policy"),
                reference_digest=digest_text("references"),
                confirmed=True,
            )
            legacy_path.with_suffix(".runtime-profile.json").write_text(
                json.dumps(profile.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            storage = PromptStorage(config_dir=Path(tmp))

            loaded = storage.load_compiled_prompt("prompts/compiled-prompt.md")
            loaded_profile = storage.load_compiled_runtime_profile("prompts/compiled-prompt.md")

        self.assertEqual(loaded, "legacy runtime prompt")
        self.assertEqual(loaded_profile.completeness_checks, profile.completeness_checks)

    def test_runtime_profile_instance_is_strictly_revalidated_before_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            invalid = RuntimeProfile(
                source_language="Chinese",
                target_language="Japanese",
                semantic_directives=("Broad semantic instruction.",),
                confirmed=True,
            )
            with self.assertRaises(ValueError):
                storage.save_compiled_prompt(
                    "runtime prompt",
                    "prompts/compiled-prompt.md",
                    runtime_profile=invalid,
                )
            self.assertFalse(
                storage.resolve_compiled_runtime_profile_path(
                    "prompts/compiled-prompt.md"
                ).exists()
            )

    def test_runtime_profile_none_preserves_and_explicit_clear_deletes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            profile = RuntimeProfile.from_dict(RuntimeProfileTests._confirmed_profile_data(), strict=True)
            storage.save_compiled_prompt(
                "first", "prompts/compiled-prompt.md", runtime_profile=profile
            )
            sidecar = storage.resolve_compiled_runtime_profile_path("prompts/compiled-prompt.md")
            before = sidecar.read_text(encoding="utf-8")
            storage.save_compiled_prompt(
                "second", "prompts/compiled-prompt.md", runtime_profile=None
            )
            self.assertEqual(sidecar.read_text(encoding="utf-8"), before)
            storage.save_compiled_prompt(
                "third", "prompts/compiled-prompt.md", clear_runtime_profile=True
            )
            self.assertFalse(sidecar.exists())

    def test_prompt_bundle_rolls_back_when_a_later_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            path = "prompts/compiled-prompt.md"
            storage.save_compiled_prompt(
                "old prompt",
                path,
                policy=ConstraintPolicyCompiler.compile("保留 API。"),
            )
            prompt_path = storage.resolve_compiled_prompt_path(path)
            policy_path = storage.resolve_compiled_policy_path(path)
            old_policy = policy_path.read_bytes()
            real_atomic_write = storage_module._atomic_write_bytes

            def fail_new_prompt(target: Path, payload: bytes) -> None:
                if target == prompt_path and payload == b"new prompt":
                    raise OSError("disk full")
                real_atomic_write(target, payload)

            with patch(
                "app.prompt.storage._atomic_write_bytes",
                side_effect=fail_new_prompt,
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    storage.save_compiled_prompt(
                        "new prompt",
                        path,
                        policy=ConstraintPolicyCompiler.compile("仅输出日语。"),
                    )

            self.assertEqual(prompt_path.read_text(encoding="utf-8"), "old prompt")
            self.assertEqual(policy_path.read_bytes(), old_policy)

    def test_runtime_profile_replace_and_clear_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            profile = RuntimeProfile.from_dict(RuntimeProfileTests._confirmed_profile_data(), strict=True)
            with self.assertRaises(ValueError):
                storage.save_compiled_prompt(
                    "runtime prompt",
                    "prompts/compiled-prompt.md",
                    runtime_profile=profile,
                    clear_runtime_profile=True,
                )

    def test_legacy_invalid_runtime_profile_is_not_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy_path = Path(tmp) / "prompts" / "prompts" / "compiled-prompt.md"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.write_text("legacy runtime prompt", encoding="utf-8")
            legacy_path.with_suffix(".runtime-profile.json").write_text(
                json.dumps(
                    {
                        **RuntimeProfileTests._confirmed_profile_data(),
                        "semantic_directives": ["Broad semantic instruction."],
                    }
                ),
                encoding="utf-8",
            )
            storage = PromptStorage(config_dir=Path(tmp))
            storage.load_compiled_prompt("prompts/compiled-prompt.md")
            migrated = storage.resolve_compiled_runtime_profile_path(
                "prompts/compiled-prompt.md"
            )
            self.assertFalse(migrated.exists())
            self.assertFalse(
                storage.load_compiled_runtime_profile("prompts/compiled-prompt.md").is_usable
            )

    def test_exports_ai_optimization_reference_candidates_for_manual_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            storage.save_compiled_prompt(
                "# Instant Translate Compiled Prompt\n\n"
                "## AI Optimization Layer\n"
                "| Source | Target |\n"
                "|--------|--------|\n"
                "| 软件 | [そふとうぇあ] |\n\n"
                "## Knowledge Reference Layer\n"
                "No knowledge references.\n",
                "prompts/compiled-prompt.md",
            )

            candidate_path = storage.export_ai_optimization_reference_candidates(
                "prompts/compiled-prompt.md"
            )
            self.assertIsNotNone(candidate_path)
            assert candidate_path is not None
            content = candidate_path.read_text(encoding="utf-8")

        self.assertEqual(candidate_path.name, "ai-optimization-candidates.md")
        self.assertIn("prompts", candidate_path.parts)
        self.assertIn("references", candidate_path.parts)
        self.assertIn("Review them manually", content)
        self.assertIn("| 软件 | [そふとうぇあ] |", content)

    def test_previews_reference_file_without_enabling_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "terms.md"
            reference_path.write_text(
                "## Glossary\n"
                "API -> [えーぴーあい]\n"
                "## Risk\n"
                "risk: triggers: API; wrap API only when it is a technical term\n",
                encoding="utf-8",
            )
            storage = PromptStorage(config_dir=Path(tmp))

            package = storage.preview_reference_file(reference_path)

        self.assertEqual(package.entries[0].source, "API")
        self.assertTrue(any("technical term" in note for note in package.risk_notes))


if __name__ == "__main__":
    unittest.main()
