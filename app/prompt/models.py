"""Prompt-related data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptConstraints:
    """Store user-entered translation constraints."""

    text: str = ""


@dataclass
class PromptKnowledgeReference:
    """Store one user-supplied knowledge file."""

    path: str
    enabled: bool = True


@dataclass
class CompiledPrompt:
    """Represent the preview or active runtime prompt."""

    content: str = ""
    version: str = "draft"
    policy: dict[str, Any] | None = None
    reference_package: dict[str, Any] | None = None
    runtime_profile: dict[str, Any] | None = None
