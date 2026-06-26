"""Tests for lightweight Agent session persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.agent.session_store import AgentSessionMeta, AgentSessionStore


class AgentSessionStoreTests(unittest.TestCase):
    """Verify JSON-backed Agent session persistence."""

    def _meta(self, group_id: int = 1, prompt_hash: str = "hash") -> AgentSessionMeta:
        return AgentSessionMeta(
            group_id=group_id,
            source_language="中文",
            target_language="日本語",
            prompt_hash=prompt_hash,
            fast_model="deepseek-v4-flash",
            thinking_model="deepseek-v4-pro",
        )

    def test_save_and_load_matching_session(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "confirmed"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = AgentSessionStore(Path(tmp))
            meta = self._meta()

            store.save(meta, messages)
            loaded = store.load(meta)

        self.assertEqual(loaded, messages)

    def test_load_rejects_prompt_hash_mismatch(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "confirmed"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = AgentSessionStore(Path(tmp))
            store.save(self._meta(prompt_hash="old"), messages)

            loaded = store.load(self._meta(prompt_hash="new"))

        self.assertIsNone(loaded)

    def test_delete_removes_session(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "confirmed"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = AgentSessionStore(Path(tmp))
            meta = self._meta()
            store.save(meta, messages)

            store.delete(meta.group_id)
            loaded = store.load(meta)

        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
