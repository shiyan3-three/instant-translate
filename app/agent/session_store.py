"""Lightweight JSON persistence for TranslationAgent sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.settings import AppSettings


@dataclass(frozen=True)
class AgentSessionMeta:
    """Metadata that decides whether a saved session is reusable."""

    group_id: int
    source_language: str
    target_language: str
    prompt_hash: str
    fast_model: str
    thinking_model: str


class AgentSessionStore:
    """Persist minimal per-group Agent sessions as JSON files."""

    VERSION = 1

    def __init__(self, root_dir: Path | None = None) -> None:
        self._root_dir = root_dir or (AppSettings.config_dir() / "sessions")

    def load(self, meta: AgentSessionMeta) -> list[dict] | None:
        """Return saved messages if the metadata still matches."""

        path = self._path(meta.group_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not self._meta_matches(data, meta):
            return None

        messages = data.get("messages")
        if not self._valid_messages(messages):
            return None
        return [dict(message) for message in messages]

    def save(self, meta: AgentSessionMeta, messages: list[dict]) -> None:
        """Persist messages for a matching future run."""

        if not self._valid_messages(messages):
            return

        self._root_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "group_id": meta.group_id,
            "source_language": meta.source_language,
            "target_language": meta.target_language,
            "prompt_hash": meta.prompt_hash,
            "fast_model": meta.fast_model,
            "thinking_model": meta.thinking_model,
            "messages": messages,
        }
        self._path(meta.group_id).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def delete(self, group_id: int) -> None:
        """Remove a saved session for one group, if present."""

        try:
            self._path(group_id).unlink()
        except FileNotFoundError:
            return
        except OSError:
            return

    def _path(self, group_id: int) -> Path:
        return self._root_dir / f"group-{group_id}.json"

    @classmethod
    def _meta_matches(cls, data: dict[str, Any], meta: AgentSessionMeta) -> bool:
        return (
            data.get("version") == cls.VERSION
            and data.get("group_id") == meta.group_id
            and data.get("source_language") == meta.source_language
            and data.get("target_language") == meta.target_language
            and data.get("prompt_hash") == meta.prompt_hash
            and data.get("fast_model") == meta.fast_model
            and data.get("thinking_model") == meta.thinking_model
        )

    @staticmethod
    def _valid_messages(messages: Any) -> bool:
        if not isinstance(messages, list) or len(messages) < 3:
            return False
        for message in messages:
            if not isinstance(message, dict):
                return False
            if message.get("role") not in {"system", "user", "assistant"}:
                return False
            if not isinstance(message.get("content"), str):
                return False
        return True
