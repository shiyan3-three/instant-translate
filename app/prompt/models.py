"""Prompt-related data models."""

from __future__ import annotations

from dataclasses import dataclass


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
