"""Tests for prompt compilation and storage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.prompt.compiler import PromptCompiler
from app.prompt.models import PromptConstraints, PromptKnowledgeReference
from app.prompt.storage import PromptStorage


class PromptCompilerTests(unittest.TestCase):
    """Verify fixed, user, optimization, and knowledge layers are preserved."""

    def test_compile_preview_includes_fixed_template_even_without_user_layers(self) -> None:
        compiled = PromptCompiler().compile_preview(PromptConstraints())

        self.assertIn(DEFAULT_BASE_PROMPT, compiled.content)
        self.assertIn("Fixed Template Layer", compiled.content)
        self.assertIn("User Constraint Layer", compiled.content)
        self.assertIn("AI Optimization Layer", compiled.content)
        self.assertIn("Knowledge Reference Layer", compiled.content)

    def test_compile_preview_keeps_original_constraints_and_ai_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            knowledge_path = Path(tmp) / "terms.md"
            knowledge_path.write_text("index => index", encoding="utf-8")

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


class PromptStorageTests(unittest.TestCase):
    """Verify compiled prompt persistence under the install/project directory."""

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

    def test_load_compiled_prompt_supports_legacy_double_prompt_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            legacy_path = Path(tmp) / "prompts" / "prompts" / "compiled-prompt.md"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.write_text("legacy runtime prompt", encoding="utf-8")
            storage = PromptStorage(config_dir=Path(tmp))

            loaded = storage.load_compiled_prompt("prompts/compiled-prompt.md")

        self.assertEqual(loaded, "legacy runtime prompt")


if __name__ == "__main__":
    unittest.main()
