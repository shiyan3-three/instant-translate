"""Prompt storage helpers."""

from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_COMPILED_PROMPT_PATH = "prompts/compiled-prompt.md"


def _project_root() -> Path:
    """Return the project root, or the executable directory when frozen."""

    try:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).parent
        return Path(__file__).resolve().parent.parent.parent
    except Exception:
        return Path.cwd()


class PromptStorage:
    """Load and save prompt-layer files and the compiled prompt."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self._config_dir = config_dir

    def resolve_compiled_prompt_path(self, path: str | Path) -> Path:
        """Resolve a compiled prompt under the install/project directory."""

        raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
        if raw_path.is_absolute():
            return raw_path

        root = self._base_dir()
        parts = raw_path.parts
        if parts and parts[0].lower() == "prompts":
            return root / raw_path
        return root / "prompts" / raw_path

    def stored_compiled_prompt_path(self, path: str | Path) -> str:
        """Return the stable relative path stored in settings."""

        raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
        if raw_path.is_absolute():
            return DEFAULT_COMPILED_PROMPT_PATH
        parts = raw_path.parts
        if parts and parts[0].lower() == "prompts":
            return raw_path.as_posix()
        return (Path("prompts") / raw_path).as_posix()

    def save_compiled_prompt(self, content: str, path: str | Path) -> Path:
        """Write the confirmed compiled prompt and return its absolute path."""

        prompt_path = self.resolve_compiled_prompt_path(path)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(content, encoding="utf-8")
        return prompt_path

    def load_compiled_prompt(self, path: str | Path) -> str:
        """Load a confirmed compiled prompt, returning empty text when missing."""

        prompt_path = self.existing_compiled_prompt_path(path)
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def existing_compiled_prompt_path(self, path: str | Path) -> Path:
        """Return the readable prompt path, including legacy fallback."""

        prompt_path = self.resolve_compiled_prompt_path(path)
        if prompt_path.exists():
            return prompt_path

        legacy_path = self._legacy_double_prompt_path(path)
        if legacy_path is not None and legacy_path.exists():
            return legacy_path
        return prompt_path

    def _base_dir(self) -> Path:
        return self._config_dir or _project_root()

    def _legacy_double_prompt_path(self, path: str | Path) -> Path | None:
        raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
        if raw_path.is_absolute():
            return None
        parts = raw_path.parts
        if parts and parts[0].lower() == "prompts":
            return self._base_dir() / "prompts" / raw_path
        return None
