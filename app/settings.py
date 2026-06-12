"""Settings models shared across the desktop application."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _clean_prompt_path(raw: str) -> str:
    """Normalize compiled prompt settings to an install-relative path."""

    if not raw:
        return "prompts/compiled-prompt.md"
    raw_path = Path(raw)
    if raw_path.is_absolute():
        return "prompts/compiled-prompt.md"
    parts = raw_path.parts
    if len(parts) >= 2 and parts[0].lower() == "prompts" and parts[1].lower() == "prompts":
        return Path("prompts", *parts[2:]).as_posix()
    if parts and parts[0].lower() == "prompts":
        return raw_path.as_posix()
    return (Path("prompts") / raw_path).as_posix()


def _default_config_dir() -> Path:
    """Return the user-local configuration directory for this application."""

    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    path = Path(base) / "instant-translate"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    compiled_prompt_path: str = "prompts/compiled-prompt.md"


@dataclass
class AppSettings:
    """Top-level application settings with JSON persistence."""

    ai: AiSettings = field(default_factory=AiSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)
    default_source_language: str = "English"
    default_target_language: str = "中文"

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppSettings:
        """Create an AppSettings from a JSON file, falling back to defaults."""

        file_path = Path(path) if path else _default_config_dir() / "settings.json"

        if not file_path.exists():
            return cls()

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            ai_data = data.get("ai", {})
            prompt_data = data.get("prompt", {})
            return cls(
                ai=AiSettings(
                    base_url=ai_data.get("base_url", ""),
                    api_key=ai_data.get("api_key", ""),
                    model=ai_data.get("model", ""),
                ),
                prompt=PromptSettings(
                    constraints_text=prompt_data.get("constraints_text", ""),
                    knowledge_reference_paths=prompt_data.get(
                        "knowledge_reference_paths", []
                    ),
                    compiled_prompt_path=_clean_prompt_path(
                        prompt_data.get("compiled_prompt_path", "prompts/compiled-prompt.md")
                    ),
                ),
                default_source_language=data.get("default_source_language", "English"),
                default_target_language=data.get("default_target_language", "中文"),
            )
        except (json.JSONDecodeError, OSError) as exc:
            from app.logger import get_logger
            get_logger().error("\u914d\u7f6e\u6587\u4ef6\u52a0\u8f7d\u5931\u8d25\uff0c\u4f7f\u7528\u9ed8\u8ba4\u503c: %s", exc)
            return cls()

    def save(self, path: str | Path | None = None) -> None:
        """Persist current settings to a JSON file."""

        file_path = Path(path) if path else _default_config_dir() / "settings.json"
        payload = {
            "ai": asdict(self.ai),
            "prompt": asdict(self.prompt),
            "default_source_language": self.default_source_language,
            "default_target_language": self.default_target_language,
        }
        file_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def config_dir(cls) -> Path:
        """Return the application configuration directory."""

        return _default_config_dir()
