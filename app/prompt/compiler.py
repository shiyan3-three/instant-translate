"""Compile user prompt layers into a confirmed runtime prompt."""

from __future__ import annotations

from pathlib import Path

from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.prompt.models import (
    CompiledPrompt,
    PromptConstraints,
    PromptKnowledgeReference,
)


class PromptCompiler:
    """Compile the three prompt layers into a runtime prompt."""

    def compile_preview(
        self,
        constraints: PromptConstraints,
        references: list[PromptKnowledgeReference] | None = None,
        optimized_user_layer: str = "",
    ) -> CompiledPrompt:
        """Return a user-reviewable compiled prompt preview."""

        user_constraint_layer = constraints.text.strip()
        if not user_constraint_layer:
            user_constraint_layer = "No additional user constraints."

        optimization_layer = optimized_user_layer.strip()
        if not optimization_layer:
            optimization_layer = "No AI-optimized supplemental rules."

        knowledge_layer = self._compile_knowledge_layer(references or [])

        content = "\n\n".join(
            [
                "# Instant Translate Compiled Prompt",
                "## Fixed Template Layer\n"
                "This layer is system-owned and has the highest priority.\n\n"
                f"{DEFAULT_BASE_PROMPT}",
                "## User Constraint Layer\n"
                "This layer contains user-confirmed translation constraints. "
                "These rules are mandatory and must not be weakened by later layers.\n\n"
                f"{user_constraint_layer}",
                "## AI Optimization Layer\n"
                "This layer contains supplemental rules derived from the user layer. "
                "Use it to clarify the user's intent, never to replace or override it.\n\n"
                f"{optimization_layer}",
                "## Knowledge Reference Layer\n"
                "This layer contains glossary, style, and fixed-expression references.\n\n"
                f"{knowledge_layer}",
                "## Runtime Direction\n"
                "For every request, follow the source and target languages appended by the app.",
            ]
        )
        return CompiledPrompt(content=content, version="preview")

    def build_optimizer_messages(
        self,
        constraints: PromptConstraints,
        references: list[PromptKnowledgeReference] | None = None,
    ) -> tuple[str, str]:
        """Build messages for AI-assisted prompt optimization."""

        knowledge_layer = self._compile_knowledge_layer(references or [])
        system_prompt = (
            "You optimize translation prompt rules for an OCR-based desktop translator. "
            "Return only concise supplemental rules, glossary entries, style guidance, and fixed expressions. "
            "Preserve the user's original intent exactly. "
            "Do not weaken, replace, or override the fixed template layer or user constraints."
        )
        user_prompt = (
            "User constraint layer:\n"
            f"{constraints.text.strip() or '(empty)'}\n\n"
            "Knowledge reference layer:\n"
            f"{knowledge_layer}"
        )
        return system_prompt, user_prompt

    def _compile_knowledge_layer(self, references: list[PromptKnowledgeReference]) -> str:
        enabled_references = [ref for ref in references if ref.enabled]
        if not enabled_references:
            return "No knowledge references."

        sections: list[str] = []
        for ref in enabled_references:
            path = Path(ref.path)
            try:
                content = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                content = f"(Unable to read reference: {exc})"
            if not content:
                content = "(empty reference)"
            sections.append(f"### {path.name}\n{content}")

        return "\n\n".join(sections)
