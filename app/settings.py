"""Settings models shared across the desktop application."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

from app.secret_store import protect_secret, unprotect_secret


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
    fast_model: str = ""
    thinking_model: str = ""

    @property
    def fast_model_name(self) -> str:
        """Return the model used for low-latency translation."""

        return self.fast_model.strip() or self.model.strip()

    @property
    def thinking_model_name(self) -> str:
        """Return the model used for rule digest and quality fallback."""

        return self.thinking_model.strip() or self.model.strip() or self.fast_model.strip()


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
    hotkey_create_selection: str = "Ctrl+Shift+Z"
    hotkey_toggle_edit_mode: str = "Ctrl+Shift+X"

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
            # ``utf-8-sig`` accepts both ordinary UTF-8 and the legal UTF-8
            # BOM emitted by some Windows editors.  Keep the rest of the
            # parsing/migration path unchanged so one malformed field cannot
            # discard otherwise valid settings.
            data = json.loads(file_path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError("settings root must be a JSON object")
            ai_data = data.get("ai", {})
            prompt_data = data.get("prompt", {})
            if not isinstance(ai_data, dict):
                raise ValueError("settings.ai must be a JSON object")
            if not isinstance(prompt_data, dict):
                raise ValueError("settings.prompt must be a JSON object")

            def stored_string(mapping: dict, key: str, default: str = "") -> str:
                value = mapping.get(key, default)
                return value if isinstance(value, str) else default

            raw_reference_paths = prompt_data.get("knowledge_reference_paths", [])
            reference_paths = (
                [item for item in raw_reference_paths if isinstance(item, str)]
                if isinstance(raw_reference_paths, list)
                else []
            )
            legacy_model = stored_string(ai_data, "model")
            protected_api_key = stored_string(ai_data, "api_key_protected")
            if protected_api_key:
                try:
                    api_key = unprotect_secret(protected_api_key)
                except (OSError, ValueError, UnicodeError) as exc:
                    from app.logger import get_logger
                    get_logger().error("API Key 解密失败，已清空保存的凭据: %s", exc)
                    api_key = ""
            else:
                # One-way migration path for settings written by older builds.
                # The next successful save replaces this plaintext field with
                # a current-user DPAPI value.
                api_key = stored_string(ai_data, "api_key")
            return cls(
                ai=AiSettings(
                    base_url=stored_string(ai_data, "base_url"),
                    api_key=api_key,
                    model=legacy_model,
                    fast_model=stored_string(ai_data, "fast_model", legacy_model),
                    thinking_model=stored_string(ai_data, "thinking_model", legacy_model),
                ),
                prompt=PromptSettings(
                    constraints_text=stored_string(prompt_data, "constraints_text"),
                    knowledge_reference_paths=reference_paths,
                    compiled_prompt_path=_clean_prompt_path(
                        stored_string(
                            prompt_data,
                            "compiled_prompt_path",
                            "prompts/compiled-prompt.md",
                        )
                    ),
                ),
                default_source_language=stored_string(data, "default_source_language", "English"),
                default_target_language=stored_string(data, "default_target_language", "中文"),
                hotkey_create_selection=stored_string(data, "hotkey_create_selection", "Ctrl+Shift+Z"),
                hotkey_toggle_edit_mode=stored_string(data, "hotkey_toggle_edit_mode", "Ctrl+Shift+X"),
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            from app.logger import get_logger
            get_logger().error("\u914d\u7f6e\u6587\u4ef6\u52a0\u8f7d\u5931\u8d25\uff0c\u4f7f\u7528\u9ed8\u8ba4\u503c: %s", exc)
            return cls()

    def save(self, path: str | Path | None = None) -> None:
        """Persist current settings atomically without risking the old file."""

        file_path = Path(path) if path else _default_config_dir() / "settings.json"
        ai_payload = asdict(self.ai)
        api_key = ai_payload.pop("api_key", "")
        if api_key:
            ai_payload["api_key_protected"] = protect_secret(api_key)
        payload = {
            "ai": ai_payload,
            "prompt": asdict(self.prompt),
            "default_source_language": self.default_source_language,
            "default_target_language": self.default_target_language,
            "hotkey_create_selection": self.hotkey_create_selection,
            "hotkey_toggle_edit_mode": self.hotkey_toggle_edit_mode,
        }
        encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            # The temp file must share a directory/volume with the destination
            # for os.replace to stay atomic on Windows as well as POSIX.
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                dir=file_path.parent,
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, file_path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    # A failed cleanup cannot make the canonical configuration
                    # invalid; preserve the original save exception instead.
                    pass

    @classmethod
    def config_dir(cls) -> Path:
        """Return the application configuration directory."""

        return _default_config_dir()
