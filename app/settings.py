"""Settings models shared across the desktop application."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AiSettings:
    """Store OpenAI-compatible API configuration."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class PromptSettings:
    """Store prompt-layer inputs and the compiled prompt path."""

    constraints_text: str = ""
    knowledge_reference_paths: list[str] = field(default_factory=list)
    compiled_prompt_path: str = "compiled-prompt.md"


@dataclass
class AppSettings:
    """Top-level application settings."""

    ai: AiSettings = field(default_factory=AiSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)
