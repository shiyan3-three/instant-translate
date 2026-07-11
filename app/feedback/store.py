"""Local feedback and confirmed translation-memory persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.settings import AppSettings


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class FeedbackRecord:
    """One user-marked translation issue waiting for review."""

    id: str
    created_at: str
    updated_at: str
    group_id: int
    source_language: str
    target_language: str
    ocr_text: str
    translation_text: str
    note: str = ""
    corrected_translation: str = ""
    status: str = "pending"
    memory_rule_id: str = ""

    @classmethod
    def create(
        cls,
        *,
        group_id: int,
        source_language: str,
        target_language: str,
        ocr_text: str,
        translation_text: str,
        note: str = "",
    ) -> FeedbackRecord:
        now = _now_iso()
        return cls(
            id=uuid4().hex,
            created_at=now,
            updated_at=now,
            group_id=group_id,
            source_language=source_language,
            target_language=target_language,
            ocr_text=ocr_text,
            translation_text=translation_text,
            note=note,
        )

    @classmethod
    def from_dict(cls, data: dict) -> FeedbackRecord:
        return cls(
            id=str(data.get("id", uuid4().hex)),
            created_at=str(data.get("created_at", _now_iso())),
            updated_at=str(data.get("updated_at", data.get("created_at", _now_iso()))),
            group_id=int(data.get("group_id", 0)),
            source_language=str(data.get("source_language", "")),
            target_language=str(data.get("target_language", "")),
            ocr_text=str(data.get("ocr_text", "")),
            translation_text=str(data.get("translation_text", "")),
            note=str(data.get("note", "")),
            corrected_translation=str(data.get("corrected_translation", "")),
            status=str(data.get("status", "pending")),
            memory_rule_id=str(data.get("memory_rule_id", "")),
        )

    def summary(self) -> str:
        text = self.ocr_text.replace("\n", " ").strip()
        if len(text) > 40:
            text = text[:40] + "..."
        return f"[{self.status}] G{self.group_id} {text or '(empty)'}"


@dataclass
class MemoryRule:
    """Confirmed local rule injected only when relevant source text appears."""

    id: str
    created_at: str
    updated_at: str
    source_language: str
    target_language: str
    trigger: str
    rule: str
    example_source: str = ""
    preferred_translation: str = ""
    source_feedback_id: str = ""
    enabled: bool = True

    @classmethod
    def create(
        cls,
        *,
        source_language: str,
        target_language: str,
        trigger: str,
        rule: str,
        example_source: str = "",
        preferred_translation: str = "",
        source_feedback_id: str = "",
    ) -> MemoryRule:
        now = _now_iso()
        return cls(
            id=uuid4().hex,
            created_at=now,
            updated_at=now,
            source_language=source_language,
            target_language=target_language,
            trigger=trigger,
            rule=rule,
            example_source=example_source,
            preferred_translation=preferred_translation,
            source_feedback_id=source_feedback_id,
        )

    @classmethod
    def from_dict(cls, data: dict) -> MemoryRule:
        return cls(
            id=str(data.get("id", uuid4().hex)),
            created_at=str(data.get("created_at", _now_iso())),
            updated_at=str(data.get("updated_at", data.get("created_at", _now_iso()))),
            source_language=str(data.get("source_language", "")),
            target_language=str(data.get("target_language", "")),
            trigger=str(data.get("trigger", "")),
            rule=str(data.get("rule", "")),
            example_source=str(data.get("example_source", "")),
            preferred_translation=str(data.get("preferred_translation", "")),
            source_feedback_id=str(data.get("source_feedback_id", "")),
            enabled=bool(data.get("enabled", True)),
        )

    def as_prompt_hint(self) -> str:
        trigger_label = " / ".join(FeedbackStore.split_triggers(self.trigger)) or self.trigger
        parts = [f"触发词：{trigger_label}", f"规则：{self.rule}"]
        if self.preferred_translation:
            parts.append(f"用户确认译法：{self.preferred_translation}")
        if self.example_source:
            parts.append(f"来源例句：{self.example_source}")
        return "；".join(parts)


class FeedbackStore:
    """Persist pending feedback and confirmed memory rules as small JSON files."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is not None:
            self._root_dir = Path(root_dir)
            self._root_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._root_dir = self._first_writable_root_dir()
        self._feedback_path = self._root_dir / "pending-feedback.json"
        self._memory_path = self._root_dir / "memory-rules.json"
        self._lock = threading.RLock()

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def add_feedback(
        self,
        *,
        group_id: int,
        source_language: str,
        target_language: str,
        ocr_text: str,
        translation_text: str,
        note: str = "",
    ) -> FeedbackRecord:
        with self._lock:
            record = FeedbackRecord.create(
                group_id=group_id,
                source_language=source_language,
                target_language=target_language,
                ocr_text=ocr_text,
                translation_text=translation_text,
                note=note,
            )
            records = self.list_feedback()
            records.append(record)
            self._write_feedback(records)
            return record

    def list_feedback(self, status: str | None = None) -> list[FeedbackRecord]:
        with self._lock:
            records = [FeedbackRecord.from_dict(item) for item in self._read_list(self._feedback_path)]
            if status is None:
                return records
            return [record for record in records if record.status == status]

    def get_feedback(self, feedback_id: str) -> FeedbackRecord | None:
        for record in self.list_feedback():
            if record.id == feedback_id:
                return record
        return None

    def update_feedback(
        self,
        feedback_id: str,
        *,
        note: str | None = None,
        corrected_translation: str | None = None,
        status: str | None = None,
    ) -> FeedbackRecord | None:
        with self._lock:
            records = self.list_feedback()
            updated: FeedbackRecord | None = None
            for record in records:
                if record.id != feedback_id:
                    continue
                if note is not None:
                    record.note = note
                if corrected_translation is not None:
                    record.corrected_translation = corrected_translation
                if status is not None:
                    record.status = status
                record.updated_at = _now_iso()
                updated = record
                break
            self._write_feedback(records)
            return updated

    def delete_feedback(self, feedback_id: str) -> bool:
        with self._lock:
            records = self.list_feedback()
            kept = [record for record in records if record.id != feedback_id]
            if len(kept) == len(records):
                return False
            self._write_feedback(kept)
            return True

    def approve_feedback(
        self,
        feedback_id: str,
        *,
        trigger: str,
        rule: str,
        preferred_translation: str = "",
    ) -> MemoryRule:
        with self._lock:
            records = self.list_feedback()
            record = next((item for item in records if item.id == feedback_id), None)
            if record is None:
                raise KeyError(f"feedback not found: {feedback_id}")
            trigger = trigger.strip()
            rule = rule.strip()
            if not trigger:
                raise ValueError("trigger is required")
            if not rule:
                raise ValueError("rule is required")

            memories = self.list_memory_rules(enabled_only=False)
            memory = next(
                (item for item in memories if item.id == record.memory_rule_id),
                None,
            )
            preferred = preferred_translation.strip() or record.corrected_translation.strip()
            if memory is None:
                memory = MemoryRule.create(
                    source_language=record.source_language,
                    target_language=record.target_language,
                    trigger=trigger,
                    rule=rule,
                    example_source=record.ocr_text,
                    preferred_translation=preferred,
                    source_feedback_id=record.id,
                )
                memories.append(memory)
            else:
                memory.trigger = trigger
                memory.rule = rule
                memory.example_source = record.ocr_text
                memory.preferred_translation = preferred
                memory.enabled = True
                memory.updated_at = _now_iso()
            self._write_memory_rules(memories)

            record.status = "confirmed"
            record.memory_rule_id = memory.id
            record.corrected_translation = memory.preferred_translation
            record.updated_at = _now_iso()
            self._write_feedback(records)
            return memory

    def list_memory_rules(self, enabled_only: bool = True) -> list[MemoryRule]:
        with self._lock:
            rules = [MemoryRule.from_dict(item) for item in self._read_list(self._memory_path)]
            if enabled_only:
                return [rule for rule in rules if rule.enabled]
            return rules

    def match_memory_rules(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        limit: int = 3,
    ) -> list[MemoryRule]:
        haystack = text.lower()
        matches: list[tuple[int, int, str, MemoryRule]] = []
        for rule in self.list_memory_rules(enabled_only=True):
            if rule.source_language and rule.source_language != source_language:
                continue
            if rule.target_language and rule.target_language != target_language:
                continue
            triggers = self.split_triggers(rule.trigger)
            if not triggers:
                continue
            matched = [trigger for trigger in triggers if trigger.lower() in haystack]
            if matched:
                matches.append(
                    (
                        max(len(trigger) for trigger in matched),
                        len(matched),
                        rule.updated_at,
                        rule,
                    )
                )
        matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in matches[:limit]]

    @staticmethod
    def join_triggers(triggers: list[str]) -> str:
        """Store multiple trigger keywords in a backward-compatible string field."""

        unique: list[str] = []
        seen: set[str] = set()
        for trigger in triggers:
            text = str(trigger).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
        return " / ".join(unique)

    @staticmethod
    def split_triggers(trigger: str) -> list[str]:
        """Return one or more trigger keywords from the stored trigger field."""

        parts = re.split(r"\s*(?:/|／|,|，|;|；|\|)\s*", trigger.strip())
        return [part for part in parts if part]

    @staticmethod
    def suggest_trigger(text: str) -> str:
        """Return a short trigger candidate from source text."""

        cleaned = re.sub(r"\s+", "", text)
        cjk_chunks = re.findall(r"[\u3400-\u9fff]{2,8}", cleaned)
        if cjk_chunks:
            return cjk_chunks[0]
        latin_chunks = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,20}", cleaned)
        return latin_chunks[0] if latin_chunks else cleaned[:12]

    def _read_list(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_feedback(self, records: list[FeedbackRecord]) -> None:
        self._write_json_list(self._feedback_path, [asdict(record) for record in records])

    def _write_memory_rules(self, rules: list[MemoryRule]) -> None:
        self._write_json_list(self._memory_path, [asdict(rule) for rule in rules])

    def _write_json_list(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def _first_writable_root_dir(cls) -> Path:
        last_error: OSError | None = None
        for root in cls._candidate_root_dirs():
            try:
                root.mkdir(parents=True, exist_ok=True)
                return root
            except OSError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("No feedback storage directory candidates configured.")

    @staticmethod
    def _candidate_root_dirs() -> list[Path]:
        roots = [AppSettings.config_dir() / "feedback"]
        if os.name == "nt":
            roots.append(Path("C:/tmp") / "instant-translate" / "feedback")
        roots.append(Path(tempfile.gettempdir()) / "instant-translate" / "feedback")
        roots.append(Path.cwd() / ".instant-translate" / "feedback")
        return roots
