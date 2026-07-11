"""Lightweight JSON persistence for TranslationAgent sessions."""

from __future__ import annotations

import json
import hashlib
import threading
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


@dataclass(frozen=True)
class AgentProfileMeta:
    """Stable Agent bootstrap identity, independent from transient selection ids."""

    source_language: str
    target_language: str
    prompt_hash: str
    fast_model: str
    thinking_model: str

    def key(self) -> str:
        raw = json.dumps(
            {
                "source_language": self.source_language,
                "target_language": self.target_language,
                "prompt_hash": self.prompt_hash,
                "fast_model": self.fast_model,
                "thinking_model": self.thinking_model,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AgentSessionStore:
    """Persist minimal per-group Agent sessions as JSON files.

    Group sessions intentionally store only the three-message Agent Profile
    bootstrap. Fast OCR translations are untrusted until user-confirmed
    feedback memory promotes them; persisting them as reusable session context
    can replay semantic mistakes into future translations.
    """

    VERSION = 1

    def __init__(self, root_dir: Path | None = None) -> None:
        self._root_dir = root_dir or (AppSettings.config_dir() / "sessions")
        self._lock = threading.RLock()

    def load(self, meta: AgentSessionMeta) -> list[dict] | None:
        """Return saved messages if the metadata still matches."""

        path = self._path(meta.group_id)
        with self._lock:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

        if not self._meta_matches(data, meta):
            return None

        messages = data.get("messages")
        if not self._valid_messages(messages):
            return None
        return self._bootstrap_messages(messages)

    def save(self, meta: AgentSessionMeta, messages: list[dict]) -> None:
        """Persist messages for a matching future run."""

        bootstrap = self._bootstrap_messages(messages)
        if not self._valid_messages(bootstrap):
            return

        payload = {
            "version": self.VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "group_id": meta.group_id,
            "source_language": meta.source_language,
            "target_language": meta.target_language,
            "prompt_hash": meta.prompt_hash,
            "fast_model": meta.fast_model,
            "thinking_model": meta.thinking_model,
            "messages": bootstrap,
        }
        with self._lock:
            self._root_dir.mkdir(parents=True, exist_ok=True)
            path = self._path(meta.group_id)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(path)

    def delete(self, group_id: int) -> None:
        """Remove a saved session for one group, if present."""

        with self._lock:
            try:
                self._path(group_id).unlink()
            except FileNotFoundError:
                return
            except OSError:
                return

    def load_profile(self, meta: AgentProfileMeta) -> list[dict] | None:
        """Load a profile bootstrap, seeding it from any matching old group session."""

        with self._lock:
            profile_path = self._profile_path(meta)
            try:
                data = json.loads(profile_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = None
            if isinstance(data, dict) and self._profile_meta_matches(data, meta):
                messages = data.get("messages")
                if self._valid_messages(messages):
                    return [dict(message) for message in messages[:3]]

            # Migration path: the previous implementation stored the same
            # bootstrap at the front of every group session.
            for path in sorted(self._root_dir.glob("group-*.json")):
                try:
                    candidate = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not self._group_matches_profile(candidate, meta):
                    continue
                messages = candidate.get("messages")
                if self._valid_messages(messages):
                    profile_messages = [dict(message) for message in messages[:3]]
                    self.save_profile(meta, profile_messages)
                    return profile_messages
        return None

    def save_profile(self, meta: AgentProfileMeta, messages: list[dict]) -> None:
        """Persist only the visible three-message Pro bootstrap."""

        bootstrap = [dict(message) for message in messages[:3]]
        if not self._valid_messages(bootstrap):
            return
        payload = {
            "version": self.VERSION,
            "kind": "agent_profile",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source_language": meta.source_language,
            "target_language": meta.target_language,
            "prompt_hash": meta.prompt_hash,
            "fast_model": meta.fast_model,
            "thinking_model": meta.thinking_model,
            "messages": bootstrap,
        }
        with self._lock:
            path = self._profile_path(meta)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(path)

    def _path(self, group_id: int) -> Path:
        return self._root_dir / f"group-{group_id}.json"

    def _profile_path(self, meta: AgentProfileMeta) -> Path:
        return self._root_dir / "profiles" / f"profile-{meta.key()}.json"

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

    @classmethod
    def _profile_meta_matches(cls, data: dict[str, Any], meta: AgentProfileMeta) -> bool:
        return (
            data.get("version") == cls.VERSION
            and data.get("kind") == "agent_profile"
            and data.get("source_language") == meta.source_language
            and data.get("target_language") == meta.target_language
            and data.get("prompt_hash") == meta.prompt_hash
            and data.get("fast_model") == meta.fast_model
            and data.get("thinking_model") == meta.thinking_model
        )

    @classmethod
    def _group_matches_profile(cls, data: dict[str, Any], meta: AgentProfileMeta) -> bool:
        return (
            data.get("version") == cls.VERSION
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

    @staticmethod
    def _bootstrap_messages(messages: list[dict]) -> list[dict]:
        return [dict(message) for message in messages[:3]]
