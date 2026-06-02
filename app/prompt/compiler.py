"""Prompt compiler placeholder."""

from __future__ import annotations

from app.prompt.models import CompiledPrompt, PromptConstraints


class PromptCompiler:
    """Compile the three prompt layers into a runtime prompt."""

    def compile_preview(self, constraints: PromptConstraints) -> CompiledPrompt:
        """Return a preview object for future confirmation."""
        return CompiledPrompt(content=constraints.text)
