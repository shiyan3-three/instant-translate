"""Compile user prompt layers into a confirmed runtime prompt."""

from __future__ import annotations

from pathlib import Path

from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.prompt.models import (
    CompiledPrompt,
    PromptConstraints,
    PromptKnowledgeReference,
)
from app.prompt.policy import ConstraintPolicy, ConstraintPolicyCompiler
from app.reference_layer import ReferencePackage


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

        reference_sections = self._load_knowledge_sections(references or [])
        knowledge_layer = self._format_knowledge_layer(reference_sections)
        reference_package = self._compile_reference_package(reference_sections)

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
        user_policy = ConstraintPolicyCompiler.compile(constraints.text)
        proposed_policy = getattr(optimized_user_layer, "policy", ConstraintPolicy())
        if not isinstance(proposed_policy, ConstraintPolicy):
            proposed_policy = ConstraintPolicy.from_dict(proposed_policy)
        policy = user_policy.merge_supplemental(proposed_policy)
        return CompiledPrompt(
            content=content,
            version="preview",
            policy=policy.to_dict(),
            reference_package=reference_package.to_dict(),
        )

    def build_optimizer_messages(
        self,
        constraints: PromptConstraints,
        references: list[PromptKnowledgeReference] | None = None,
    ) -> tuple[str, str]:
        """Build messages for AI-assisted prompt optimization."""

        knowledge_layer = self._compile_knowledge_layer(references or [])
        system_prompt = (
            "You optimize translation prompt rules for an OCR-based desktop translator. "
            "Return one JSON object with keys supplemental_rules, constraint_policy, user_summary, and change_items. "
            "supplemental_rules must be a concise string containing semantic/style clarifications, "
            "risk reminders, and examples only. Do not create glossary tables, source=>target mappings, "
            "fixed term readings, or terminology lists; those belong only in the user-confirmed "
            "Knowledge Reference layer. constraint_policy must be an object with version=1 and "
            "a rules array. Each rule has type, params, enforcement, scope, and source_text. "
            "Allowed locally enforceable types are allowed_characters, separator, term_wrapper, punctuation, "
            "preserve, max_length, line_breaks, case, literal_replace, and forbidden_literals. "
            "For term_wrapper, selection_mode must be one of references_only, references_and_ascii, or "
            "domain_inference. Use references_only when fixed bracketed readings must come only from "
            "user-confirmed knowledge references; use references_and_ascii when source code-like ASCII "
            "tokens may also be bracketed; use domain_inference only when the user explicitly asks the "
            "model to identify additional domain terminology. "
            "Use enforcement=model for requirements that cannot be mechanically checked. "
            "Do not output code, regexes, commands, or executable expressions. "
            "user_summary must be a short, plain Simplified Chinese explanation for a non-technical user. "
            "change_items must contain 1 to 8 objects with non-empty Simplified Chinese title and description. "
            "They must faithfully explain the effects of supplemental_rules, without chain-of-thought, API details, "
            "the complete machine prompt, invented terminology mappings, or new hard rules. "
            "Preserve the user's original intent exactly. "
            "Do not weaken, replace, or override the fixed template layer or user constraints. "
            "Always include as the first supplemental rule: produce a natural translation that faithfully conveys "
            "the original meaning with natural word order and appropriate omission of subjects/pronouns, "
            "then apply formatting constraints. Translation must not be a mechanical character-by-character substitution."
        )
        user_prompt = (
            "User constraint layer:\n"
            f"{constraints.text.strip() or '(empty)'}\n\n"
            "Knowledge reference layer:\n"
            f"{knowledge_layer}"
        )
        return system_prompt, user_prompt

    def _compile_knowledge_layer(self, references: list[PromptKnowledgeReference]) -> str:
        return self._format_knowledge_layer(self._load_knowledge_sections(references))

    def _load_knowledge_sections(
        self,
        references: list[PromptKnowledgeReference],
    ) -> list[tuple[Path, str]]:
        enabled_references = [ref for ref in references if ref.enabled]
        if not enabled_references:
            return []

        sections: list[tuple[Path, str]] = []
        for ref in enabled_references:
            path = Path(ref.path)
            try:
                content = path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                from app.logger import get_debug_logger
                get_debug_logger().warning("\u77e5\u8bc6\u5f15\u7528\u6587\u4ef6\u8bfb\u53d6\u5931\u8d25 (%s): %s", path, exc)
                content = f"(Unable to read reference: {exc})"
            if not content:
                content = "(empty reference)"
            sections.append((path, content))

        return sections

    def _format_knowledge_layer(self, sections: list[tuple[Path, str]]) -> str:
        if not sections:
            return "No knowledge references."
        return "\n\n".join(f"### {path.name}\n{content}" for path, content in sections)

    def _compile_reference_package(
        self,
        sections: list[tuple[Path, str]],
    ) -> ReferencePackage:
        texts = [content for _, content in sections]
        return ReferencePackage.from_texts(texts)
