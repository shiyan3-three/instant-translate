"""Local feedback and confirmed translation-memory persistence."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import tempfile
import threading
import unicodedata
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from app.logger import get_logger
from app.settings import AppSettings

__all__ = [
    "CorrectionMatch",
    "FeedbackRecord",
    "FeedbackStorageUnavailable",
    "FeedbackStore",
    "MemoryRule",
    "StaleConsolidationAborted",
]


_UNSET = object()


class StaleConsolidationAborted(RuntimeError):
    """Raised when a background rule no longer matches its source corrections."""


class FeedbackStorageUnavailable(RuntimeError):
    """Raised when feedback storage cannot be read or safely recovered."""


def _now_iso() -> str:
    # Keep microseconds.  Feedback revisions can be submitted more than once
    # in one second, and second-level timestamps made an older exact correction
    # win a stable-sort tie over the user's newer revision.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


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
    keywords: list[str] | None = None
    ai_problem_summary: str = ""
    matched_memory_rule_ids: list[str] | None = None
    matched_correction_ids: list[str] | None = None
    matched_memory_hint_snapshots: dict[str, str] | None = None
    matched_correction_hint_snapshots: dict[str, str] | None = None
    consolidation_status: str = "not_started"
    consolidation_error: str = ""
    consolidation_input_digest: str = ""
    enabled: bool = True
    consolidation_attempts: int = 0

    def __post_init__(self) -> None:
        self.keywords = list(dict.fromkeys(self._runtime_string_list(self.keywords)))
        self.matched_memory_rule_ids = list(dict.fromkeys(
            self._runtime_string_list(self.matched_memory_rule_ids)
        ))
        self.matched_correction_ids = list(dict.fromkeys(
            self._runtime_string_list(self.matched_correction_ids)
        ))
        self.matched_memory_hint_snapshots = self._runtime_hint_map(
            self.matched_memory_hint_snapshots
        )
        self.matched_correction_hint_snapshots = self._runtime_hint_map(
            self.matched_correction_hint_snapshots
        )
        self.matched_memory_hint_snapshots = {
            key: value
            for key, value in self.matched_memory_hint_snapshots.items()
            if key in self.matched_memory_rule_ids
        }
        self.matched_correction_hint_snapshots = {
            key: value
            for key, value in self.matched_correction_hint_snapshots.items()
            if key in self.matched_correction_ids
        }

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
        matched_memory_rule_ids: list[str] | None = None,
        matched_correction_ids: list[str] | None = None,
        matched_memory_hint_snapshots: dict[str, str] | None = None,
        matched_correction_hint_snapshots: dict[str, str] | None = None,
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
            matched_memory_rule_ids=list(matched_memory_rule_ids or []),
            matched_correction_ids=list(matched_correction_ids or []),
            matched_memory_hint_snapshots=dict(matched_memory_hint_snapshots or {}),
            matched_correction_hint_snapshots=dict(matched_correction_hint_snapshots or {}),
        )

    @classmethod
    def from_dict(cls, data: dict) -> FeedbackRecord:
        status = str(data.get("status", "pending"))
        if status not in {"pending", "accepted", "confirmed", "dismissed"}:
            status = "pending"
        consolidation_status = str(data.get("consolidation_status", "not_started"))
        if consolidation_status not in {"not_started", "queued", "running", "completed", "failed"}:
            consolidation_status = "not_started"
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
            status=status,
            memory_rule_id=str(data.get("memory_rule_id", "")),
            keywords=cls._string_list(data.get("keywords")),
            ai_problem_summary=str(data.get("ai_problem_summary", "")),
            matched_memory_rule_ids=cls._string_list(data.get("matched_memory_rule_ids")),
            matched_correction_ids=cls._string_list(data.get("matched_correction_ids")),
            matched_memory_hint_snapshots=cls._string_map(
                data.get("matched_memory_hint_snapshots")
            ),
            matched_correction_hint_snapshots=cls._string_map(
                data.get("matched_correction_hint_snapshots")
            ),
            consolidation_status=consolidation_status,
            consolidation_error=str(data.get("consolidation_error", "")),
            consolidation_input_digest=str(data.get("consolidation_input_digest", "")),
            enabled=FeedbackStore._stored_bool(data, "enabled", default=True),
            consolidation_attempts=FeedbackStore._stored_nonnegative_int(
                data, "consolidation_attempts", default=0
            ),
        )

    @staticmethod
    def _string_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _runtime_string_list(value) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _string_map(value) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, str] = {}
        for key, item in value.items():
            if len(result) >= 8:
                break
            if not isinstance(key, str) or not isinstance(item, str):
                continue
            clean_key = key.strip()
            clean_item = item.strip()
            if (
                clean_key
                and clean_item
                and len(clean_key) <= 128
                and len(clean_item) <= 2400
            ):
                result[clean_key] = clean_item
        return result

    @staticmethod
    def _runtime_hint_map(value) -> dict[str, str]:
        return FeedbackRecord._string_map(value)

    def summary(self) -> str:
        text = self.ocr_text.replace("\n", " ").strip()
        if len(text) > 40:
            text = text[:40] + "..."
        return f"[{self.status}] G{self.group_id} {text or '(empty)'}"


@dataclass(frozen=True)
class CorrectionMatch:
    """One correction retrieval result with an explicit identity decision."""

    score: float
    record: FeedbackRecord
    exact_match: bool


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
    source_feedback_ids: list[str] | None = None
    # Snapshot of every correction input used to derive an automatic rule.
    # Runtime re-checks these digests before the rule can become authoritative.
    source_feedback_digests: dict[str, str] | None = None
    origin: str = "legacy"
    user_locked: bool = False
    # Separate user choice to enable/disable an automatic rule from ownership
    # of its contents.  ``None`` migrates existing JSON without pretending an
    # old disabled rule was explicitly disabled by the user.
    user_enabled_override: bool | None = None

    def __post_init__(self) -> None:
        self.source_feedback_ids = list(dict.fromkeys(
            FeedbackRecord._runtime_string_list(self.source_feedback_ids)
        ))
        if self.source_feedback_id and self.source_feedback_id not in self.source_feedback_ids:
            self.source_feedback_ids.insert(0, self.source_feedback_id)
        raw_digests = self.source_feedback_digests
        if not isinstance(raw_digests, dict):
            raw_digests = {}
        self.source_feedback_digests = {
            str(feedback_id).strip(): str(digest).strip()
            for feedback_id, digest in raw_digests.items()
            if str(feedback_id).strip() and str(digest).strip()
        }

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
        origin: str = "legacy",
        user_locked: bool = False,
        user_enabled_override: bool | None = None,
    ) -> MemoryRule:
        if user_enabled_override is not None and not isinstance(user_enabled_override, bool):
            raise ValueError("user enabled override must be boolean or None")
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
            origin=origin,
            user_locked=user_locked,
            user_enabled_override=user_enabled_override,
        )

    @classmethod
    def from_dict(cls, data: dict) -> MemoryRule:
        origin = str(data.get("origin", "legacy"))
        raw_locked = data.get("user_locked")
        # Before automatic consolidation existed, persisted rules used the
        # legacy origin and sometimes explicitly stored user_locked=false.
        # Those were still user-confirmed rules and must migrate as owned.
        if origin == "legacy":
            user_locked = True
        else:
            user_locked = (
                raw_locked if isinstance(raw_locked, bool)
                else origin != "automatic"
            )
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
            enabled=FeedbackStore._stored_bool(data, "enabled", default=True),
            source_feedback_ids=FeedbackRecord._string_list(data.get("source_feedback_ids")),
            source_feedback_digests=(
                dict(data.get("source_feedback_digests"))
                if isinstance(data.get("source_feedback_digests"), dict)
                else {}
            ),
            origin=origin,
            user_locked=user_locked,
            user_enabled_override=(
                data.get("user_enabled_override")
                if isinstance(data.get("user_enabled_override"), bool)
                else None
            ),
        )

    def as_prompt_hint(self) -> str:
        trigger_label = " / ".join(FeedbackStore.split_triggers(self.trigger)) or self.trigger
        trigger_label = FeedbackStore._bounded_prompt_text(trigger_label, 160)
        rule = FeedbackStore._bounded_prompt_text(self.rule, FeedbackStore.MAX_RULE_CHARS)
        parts = [f"触发词：{trigger_label}", f"规则：{rule}"]
        if self.preferred_translation:
            parts.append(
                "用户确认译法："
                + FeedbackStore._bounded_prompt_text(self.preferred_translation, 600)
            )
        if self.example_source:
            parts.append(
                "来源例句：" + FeedbackStore._bounded_prompt_text(self.example_source, 600)
            )
        return "；".join(parts)


class FeedbackStore:
    """Persist pending feedback and confirmed memory rules as small JSON files."""

    MAX_TRIGGER_CHARS = 120
    MAX_RULE_CHARS = 600
    _TRIGGER_LIST_PREFIX = "@@feedback-triggers-v2@@"
    _ROOT_LOCKS_GUARD = threading.Lock()
    _ROOT_LOCKS: dict[str, threading.RLock] = {}
    _ROOT_REVISIONS: dict[str, int] = {}

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        allow_unavailable: bool = False,
    ) -> None:
        if root_dir is not None:
            self._root_dir = Path(root_dir)
            self._root_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._root_dir = self._first_writable_root_dir()
        self._feedback_path = self._root_dir / "pending-feedback.json"
        self._memory_path = self._root_dir / "memory-rules.json"
        self._transaction_path = self._root_dir / "feedback-state.transaction.json"
        self._lock = self._shared_root_lock(self._root_dir)
        self._revision_key = str(self._root_dir.resolve()).casefold()
        self._unparsed_rows: dict[Path, list[object]] = {}
        self._corrupt_files: set[Path] = set()
        self._reported_storage_problems: set[Path] = set()
        with self._lock:
            recovered = self._recover_cross_file_transaction()
            if not recovered and not allow_unavailable:
                raise FeedbackStorageUnavailable(
                    "feedback storage is unavailable until transaction recovery succeeds"
                )

    def _ensure_transaction_recovered(self) -> None:
        """Fail closed until an interrupted two-file commit is recovered.

        A caller must never observe or mutate the canonical files while a
        rollback journal is unresolved.  In particular, returning an empty or
        partially committed list here could let a later successful-looking
        write destroy data that the eventual rollback restores.
        """

        if not self._recover_cross_file_transaction():
            raise FeedbackStorageUnavailable(
                "feedback storage is unavailable until transaction recovery succeeds"
            )

    @classmethod
    def _shared_root_lock(cls, root_dir: Path) -> threading.RLock:
        """Serialize read/modify/write operations across Store instances."""

        key = str(root_dir.resolve()).casefold()
        with cls._ROOT_LOCKS_GUARD:
            lock = cls._ROOT_LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._ROOT_LOCKS[key] = lock
            return lock

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def knowledge_revision(self) -> int:
        """Return the process-local durable revision for runtime feedback knowledge."""

        with self._lock:
            return self._ROOT_REVISIONS.get(self._revision_key, 0)

    def _advance_knowledge_revision(self) -> None:
        """Advance the shared revision only after a durable write succeeds."""

        with self._lock:
            self._ROOT_REVISIONS[self._revision_key] = (
                self._ROOT_REVISIONS.get(self._revision_key, 0) + 1
            )

    @contextmanager
    def guard_knowledge_revision(self, expected_revision: int) -> Iterator[bool]:
        """Hold the root lock while a consumer commits revision-bound state.

        Feedback mutations for every store instance sharing this root use the
        same re-entrant lock.  A short consumer-side commit can therefore
        verify a revision and publish its local side effects without a feedback
        write landing between those two operations.  Network work must never
        run inside this guard.
        """

        with self._lock:
            yield self._ROOT_REVISIONS.get(self._revision_key, 0) == expected_revision

    def add_feedback(
        self,
        *,
        group_id: int,
        source_language: str,
        target_language: str,
        ocr_text: str,
        translation_text: str,
        note: str = "",
        matched_memory_rule_ids: list[str] | None = None,
        matched_correction_ids: list[str] | None = None,
        matched_memory_hint_snapshots: dict[str, str] | None = None,
        matched_correction_hint_snapshots: dict[str, str] | None = None,
    ) -> FeedbackRecord:
        with self._lock:
            record = FeedbackRecord.create(
                group_id=group_id,
                source_language=source_language,
                target_language=target_language,
                ocr_text=ocr_text,
                translation_text=translation_text,
                note=note,
                matched_memory_rule_ids=list(matched_memory_rule_ids or []),
                matched_correction_ids=list(matched_correction_ids or []),
                matched_memory_hint_snapshots=dict(
                    matched_memory_hint_snapshots or {}
                ),
                matched_correction_hint_snapshots=dict(
                    matched_correction_hint_snapshots or {}
                ),
            )
            records = self.list_feedback()
            records.append(record)
            self._write_feedback(records)
            return record

    def list_feedback(self, status: str | None = None) -> list[FeedbackRecord]:
        with self._lock:
            self._ensure_transaction_recovered()
            loaded = self._load_unique_rows(
                self._feedback_path,
                self._valid_feedback_row,
                FeedbackRecord.from_dict,
            )
            records = [item for item in loaded if isinstance(item, FeedbackRecord)]
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
        keywords: list[str] | None = None,
        ai_problem_summary: str | None = None,
        consolidation_status: str | None = None,
        consolidation_error: str | None = None,
        consolidation_attempts: int | None = None,
    ) -> FeedbackRecord | None:
        if status is not None and status not in {
            "pending", "accepted", "confirmed", "dismissed"
        }:
            raise ValueError(f"invalid feedback status: {status}")
        if consolidation_status is not None and consolidation_status not in {
            "not_started", "queued", "running", "completed", "failed"
        }:
            raise ValueError(f"invalid consolidation status: {consolidation_status}")
        with self._lock:
            records = self.list_feedback()
            updated: FeedbackRecord | None = None
            input_changed = False
            for record in records:
                if record.id != feedback_id:
                    continue
                previous_digest = record.consolidation_input_digest
                if note is not None:
                    record.note = note.strip()
                if corrected_translation is not None:
                    record.corrected_translation = corrected_translation.strip()
                if status is not None:
                    record.status = status
                if keywords is not None:
                    record.keywords = self._normalized_keywords(keywords)
                if ai_problem_summary is not None:
                    record.ai_problem_summary = ai_problem_summary.strip()
                if consolidation_status is not None:
                    record.consolidation_status = consolidation_status
                if consolidation_error is not None:
                    record.consolidation_error = consolidation_error
                if consolidation_attempts is not None:
                    record.consolidation_attempts = max(0, int(consolidation_attempts))
                input_fields_changed = any(
                    value is not None
                    for value in (
                        note,
                        corrected_translation,
                        keywords,
                        ai_problem_summary,
                    )
                )
                if (
                    record.status in {"accepted", "confirmed"}
                    and not record.corrected_translation.strip()
                ):
                    raise ValueError("active feedback requires a corrected translation")
                if input_fields_changed:
                    new_digest = self._consolidation_digest(
                        record.corrected_translation,
                        record.note,
                        list(record.keywords or []),
                        record.ai_problem_summary,
                    )
                    input_changed = new_digest != previous_digest
                    record.consolidation_input_digest = new_digest
                    if input_changed:
                        if record.status in {"accepted", "confirmed"}:
                            record.status = "accepted"
                            record.consolidation_status = "queued"
                        else:
                            # Inactive evidence must still receive a new
                            # identity and revoke every rule derived from the
                            # previous input.  If it is later reactivated, the
                            # old automatic rule therefore cannot ABA-revive.
                            record.consolidation_status = "not_started"
                        record.consolidation_error = ""
                        record.consolidation_attempts = 0
                record.updated_at = _now_iso()
                updated = record
                break
            if updated is not None and input_changed:
                self._write_feedback_revoking_automatic_rules(records, feedback_id)
            else:
                self._write_feedback(records)
            return updated

    def finalize_stale_consolidation(
        self,
        feedback_id: str,
        *,
        expected_input_digest: str,
        expected_attempt: int,
        message: str,
    ) -> FeedbackRecord | None:
        """End a stale job only if its original primary input is still current."""

        with self._lock:
            records = self.list_feedback()
            record = next((item for item in records if item.id == feedback_id), None)
            if (
                record is None
                or not record.enabled
                or record.status not in {"accepted", "confirmed"}
                or record.consolidation_input_digest != expected_input_digest
                or record.consolidation_status != "running"
                or record.consolidation_attempts != expected_attempt
            ):
                return None
            record.consolidation_status = "failed"
            record.consolidation_error = message.strip()
            # A stale result was discarded before commit; it is not a Pro
            # failure for this input and must not consume the retry budget.
            record.consolidation_attempts = max(0, record.consolidation_attempts - 1)
            record.updated_at = _now_iso()
            self._write_feedback(records)
            return record

    def fail_consolidation_if_current(
        self,
        feedback_id: str,
        *,
        expected_input_digest: str,
        expected_attempt: int,
        message: str,
    ) -> FeedbackRecord | None:
        """Atomically fail only the exact background run that raised.

        A digest alone is insufficient because identical input may be retried.
        The attempt counter distinguishes an older worker from a newer run.
        ``None`` means ownership changed and newer durable state was preserved.
        """

        if not expected_input_digest:
            return None
        if isinstance(expected_attempt, bool) or expected_attempt < 1:
            raise ValueError("expected consolidation attempt must be positive")
        with self._lock:
            records = self.list_feedback()
            record = next((item for item in records if item.id == feedback_id), None)
            if (
                record is None
                or not record.enabled
                or record.status not in {"accepted", "confirmed"}
                or record.consolidation_status != "running"
                or record.consolidation_input_digest != expected_input_digest
                or record.consolidation_attempts != expected_attempt
            ):
                return None
            record.consolidation_status = "failed"
            record.consolidation_error = message.strip()
            record.updated_at = _now_iso()
            self._write_feedback(records)
            return record

    def delete_feedback(self, feedback_id: str) -> bool:
        with self._lock:
            records = self.list_feedback()
            kept = [record for record in records if record.id != feedback_id]
            if len(kept) == len(records):
                return False
            rules = self.list_memory_rules(enabled_only=False)
            disabled_rule_ids = self._disable_automatic_rules_for_feedback(
                rules,
                feedback_id,
            )
            self._mark_disabled_automatic_rule_owners(
                kept,
                rules,
                disabled_rule_ids,
                changed_feedback_id=feedback_id,
            )
            if disabled_rule_ids:
                self._commit_cross_file_state(kept, rules, order="memory_first")
            else:
                self._write_feedback(kept)
            return True

    def accept_translation(
        self,
        feedback_id: str,
        corrected_translation: str,
    ) -> FeedbackRecord:
        """Backward-compatible alias for submitting an active correction."""

        record = self.get_feedback(feedback_id)
        if record is None:
            raise KeyError(f"feedback not found: {feedback_id}")
        if record.status != "pending":
            raise ValueError("only pending feedback can accept a translation")
        return self.submit_correction(
            feedback_id,
            corrected_translation=corrected_translation,
            note=record.note,
            keywords=record.keywords,
        )

    def submit_correction(
        self,
        feedback_id: str,
        *,
        corrected_translation: str,
        note: str = "",
        keywords: list[str] | None = None,
        ai_problem_summary: str | None = None,
    ) -> FeedbackRecord:
        """Persist a user-confirmed correction and activate it for retrieval."""

        corrected = corrected_translation.strip()
        if not corrected:
            raise ValueError("corrected translation is required")
        with self._lock:
            records = self.list_feedback()
            record = next((item for item in records if item.id == feedback_id), None)
            if record is None:
                raise KeyError(f"feedback not found: {feedback_id}")
            if record.status == "dismissed":
                raise ValueError("dismissed feedback cannot be submitted")
            cleaned_note = note.strip()
            # ``[]`` is an explicit user request to clear old retrieval
            # keywords.  Only ``None`` means "keep the existing value".
            cleaned_keywords = self._normalized_keywords(
                record.keywords if keywords is None else keywords
            )
            cleaned_summary = (
                ai_problem_summary.strip()
                if ai_problem_summary is not None
                else record.ai_problem_summary
            )
            input_digest = self._consolidation_digest(
                corrected,
                cleaned_note,
                cleaned_keywords,
                cleaned_summary,
            )
            input_changed = input_digest != record.consolidation_input_digest
            unchanged_in_flight_or_done = (
                input_digest == record.consolidation_input_digest
                and record.consolidation_status in {"queued", "running", "completed"}
            )
            previous_status = record.status
            record.corrected_translation = corrected
            record.note = cleaned_note
            record.keywords = cleaned_keywords
            if ai_problem_summary is not None:
                record.ai_problem_summary = cleaned_summary
            record.status = (
                "confirmed"
                if not input_changed
                and previous_status == "confirmed"
                and bool(record.memory_rule_id)
                else "accepted"
            )
            record.enabled = True
            record.consolidation_input_digest = input_digest
            if not unchanged_in_flight_or_done:
                record.consolidation_status = "queued"
                record.consolidation_error = ""
                if input_changed:
                    record.consolidation_attempts = 0
            record.updated_at = _now_iso()
            if input_changed:
                self._write_feedback_revoking_automatic_rules(records, feedback_id)
            else:
                self._write_feedback(records)
            return record

    def save_keywords(self, feedback_id: str, keywords: list[str]) -> FeedbackRecord:
        updated = self.update_feedback(feedback_id, keywords=self._normalized_keywords(keywords))
        if updated is None:
            raise KeyError(f"feedback not found: {feedback_id}")
        return updated

    def set_feedback_enabled(self, feedback_id: str, enabled: bool) -> FeedbackRecord:
        """Enable or revoke one accepted correction without deleting its audit record."""

        if not isinstance(enabled, bool):
            raise ValueError("feedback enabled state must be boolean")
        with self._lock:
            records = self.list_feedback()
            record = next((item for item in records if item.id == feedback_id), None)
            if record is None:
                raise KeyError(f"feedback not found: {feedback_id}")
            if record.status not in {"accepted", "confirmed"}:
                raise ValueError("only submitted feedback can be enabled or disabled")
            record.enabled = enabled
            if not enabled:
                # A running/completed result can no longer be considered valid
                # after revocation.  Re-enabling keeps the correction usable,
                # while an explicit resubmission can start a fresh consolidation.
                record.consolidation_status = "not_started"
                record.consolidation_error = ""
                record.consolidation_attempts = 0
            record.updated_at = _now_iso()
            if not enabled:
                rules = self.list_memory_rules(enabled_only=False)
                disabled_rule_ids = self._disable_automatic_rules_for_feedback(
                    rules,
                    feedback_id,
                )
                self._mark_disabled_automatic_rule_owners(
                    records,
                    rules,
                    disabled_rule_ids,
                    changed_feedback_id=feedback_id,
                )
                self._commit_cross_file_state(records, rules, order="memory_first")
            else:
                self._write_feedback(records)
            return record

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
            if record.status == "dismissed" or not record.enabled:
                raise ValueError("only active feedback can create a manual memory rule")
            trigger = trigger.strip()
            rule = rule.strip()
            if not trigger:
                raise ValueError("trigger is required")
            if not rule:
                raise ValueError("rule is required")
            self._validate_memory_text(trigger, rule)

            memories = self.list_memory_rules(enabled_only=False)
            memory = next(
                (item for item in memories if item.id == record.memory_rule_id),
                None,
            )
            preferred = preferred_translation.strip() or record.corrected_translation.strip()
            old_digest = record.consolidation_input_digest
            record.corrected_translation = preferred
            record.consolidation_input_digest = self._consolidation_digest(
                record.corrected_translation,
                record.note,
                list(record.keywords or []),
                record.ai_problem_summary,
            )
            evidence_changed = record.consolidation_input_digest != old_digest

            # The record may also be secondary evidence for other automatic
            # rules.  Changing its confirmed translation revokes every such
            # unlocked derivation before the new evidence is published.
            if evidence_changed:
                for candidate in memories:
                    if candidate is memory:
                        continue
                    if (
                        candidate.origin == "automatic"
                        and not candidate.user_locked
                        and record.id in (candidate.source_feedback_ids or [])
                    ):
                        candidate.enabled = False
                        candidate.updated_at = _now_iso()
            if memory is None:
                memory = MemoryRule.create(
                    source_language=record.source_language,
                    target_language=record.target_language,
                    trigger=trigger,
                    rule=rule,
                    example_source=record.ocr_text,
                    preferred_translation=preferred,
                    source_feedback_id=record.id,
                    origin="manual",
                    user_locked=True,
                )
                memories.append(memory)
            else:
                memory.trigger = trigger
                memory.rule = rule
                memory.example_source = record.ocr_text
                memory.preferred_translation = preferred
                memory.enabled = True
                memory.origin = "manual"
                memory.user_locked = True
                memory.source_feedback_id = record.id
                memory.source_feedback_ids = [record.id]
                memory.source_feedback_digests = {}
                memory.updated_at = _now_iso()

            record.status = "confirmed"
            record.memory_rule_id = memory.id
            record.corrected_translation = memory.preferred_translation
            record.consolidation_status = "completed"
            record.consolidation_error = ""
            record.updated_at = _now_iso()
            # Publish the feedback link before the manual rule.  Runtime link
            # validation and the rollback journal prevent an orphan rule from
            # ever becoming authoritative after a partial write.
            self._commit_cross_file_state(records, memories, order="feedback_first")
            return memory

    def list_memory_rules(self, enabled_only: bool = True) -> list[MemoryRule]:
        with self._lock:
            self._ensure_transaction_recovered()
            loaded = self._load_unique_rows(
                self._memory_path,
                self._valid_memory_row,
                MemoryRule.from_dict,
            )
            rules = [item for item in loaded if isinstance(item, MemoryRule)]
            if enabled_only:
                return [rule for rule in rules if rule.enabled]
            return rules

    def get_memory_rule(self, rule_id: str) -> MemoryRule | None:
        return next(
            (rule for rule in self.list_memory_rules(enabled_only=False) if rule.id == rule_id),
            None,
        )

    def update_memory_rule(
        self,
        rule_id: str,
        *,
        rule_text: str | None = None,
        enabled: bool | None = None,
        user_locked: bool | None | object = _UNSET,
    ) -> MemoryRule:
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("memory enabled state must be boolean")
        if user_locked is not _UNSET and not isinstance(user_locked, bool):
            raise ValueError("memory lock state must be boolean")
        with self._lock:
            rules = self.list_memory_rules(enabled_only=False)
            memory = next((item for item in rules if item.id == rule_id), None)
            if memory is None:
                raise KeyError(f"memory rule not found: {rule_id}")
            if rule_text is not None:
                cleaned = rule_text.strip()
                if not cleaned:
                    raise ValueError("memory rule is required")
                if len(cleaned) > self.MAX_RULE_CHARS:
                    raise ValueError(
                        f"memory rule exceeds {self.MAX_RULE_CHARS} characters"
                    )
                memory.rule = cleaned
            if enabled is not None:
                memory.enabled = enabled
                if memory.origin == "automatic":
                    memory.user_enabled_override = enabled
            # Editing text has always been an explicit ownership action.  A
            # pure enable/disable operation must not silently promote an
            # automatic rule to a user-confirmed rule.
            if user_locked is not _UNSET:
                assert isinstance(user_locked, bool)
                memory.user_locked = user_locked
            elif rule_text is not None:
                memory.user_locked = True
            memory.updated_at = _now_iso()
            self._write_memory_rules(rules)
            return memory

    def upsert_automatic_memory_rule(
        self,
        *,
        feedback_ids: list[str],
        trigger: str,
        rule: str,
        target_rule_id: str = "",
        primary_feedback_id: str = "",
        expected_input_digest: str = "",
        expected_input_digests: dict[str, str] | None = None,
    ) -> MemoryRule:
        """Create or safely refine one explicitly selected automatic rule.

        Trigger equality is deliberately not an identity key.  Words such as
        "request" or "system" can occur in unrelated failures, so callers must
        name the automatic rule they intend to refine and provide every piece
        of evidence already attached to it.
        """

        clean_ids = list(dict.fromkeys(item for item in feedback_ids if item))
        if not clean_ids:
            raise ValueError("at least one feedback id is required")
        primary_id = primary_feedback_id.strip() or clean_ids[0]
        if clean_ids[0] != primary_id:
            raise StaleConsolidationAborted(
                "the consolidation primary feedback no longer owns the proposal"
            )
        trigger = trigger.strip()
        rule = rule.strip()
        if not trigger or not rule:
            raise ValueError("automatic memory requires trigger and rule")
        self._validate_memory_text(trigger, rule)
        if not self.automatic_trigger_is_specific(trigger):
            raise ValueError("automatic memory trigger is too broad")
        frozen_digests = {
            str(feedback_id).strip(): str(digest).strip()
            for feedback_id, digest in (expected_input_digests or {}).items()
            if str(feedback_id).strip()
        }
        if len(clean_ids) > 1:
            if set(frozen_digests) != set(clean_ids) or any(
                not self._valid_input_digest(frozen_digests.get(feedback_id, ""))
                for feedback_id in clean_ids
            ):
                raise ValueError(
                    "automatic memory with multiple evidence rows requires "
                    "a complete frozen digest snapshot"
                )
        with self._lock:
            records = self.list_feedback()
            records_by_id = {item.id: item for item in records}
            missing_ids = [item_id for item_id in clean_ids if item_id not in records_by_id]
            if missing_ids:
                raise StaleConsolidationAborted(
                    "one or more consolidation evidence records were deleted"
                )
            selected = [records_by_id[item_id] for item_id in clean_ids]
            source = records_by_id[primary_id]
            if (
                source.status not in {"accepted", "confirmed"}
                or not source.enabled
                or not source.corrected_translation.strip()
            ):
                raise StaleConsolidationAborted(
                    "the primary correction is no longer active"
                )
            if (
                expected_input_digest
                and source.consolidation_input_digest != expected_input_digest
            ):
                raise StaleConsolidationAborted(
                    "the primary correction changed while consolidation was running"
                )
            if any(
                item.status not in {"accepted", "confirmed"}
                or not item.enabled
                or not item.corrected_translation.strip()
                or item.source_language != source.source_language
                or item.target_language != source.target_language
                for item in selected
            ):
                raise StaleConsolidationAborted(
                    "consolidation evidence is no longer active or language-compatible"
                )
            # Establish a durable version for legacy corrections at the exact
            # moment they are used as evidence.  New submissions already have
            # this digest; the fallback is only a deterministic migration.
            for item in selected:
                if not self._valid_input_digest(item.consolidation_input_digest):
                    item.consolidation_input_digest = self._consolidation_digest(
                        item.corrected_translation,
                        item.note,
                        list(item.keywords or []),
                        item.ai_problem_summary,
                    )
            if frozen_digests and any(
                item.consolidation_input_digest != frozen_digests.get(item.id, "")
                for item in selected
            ):
                raise StaleConsolidationAborted(
                    "one or more consolidation evidence rows changed while "
                    "the thinking model was running"
                )
            rules = self.list_memory_rules(enabled_only=False)
            implicit_source_link = not target_rule_id and bool(source.memory_rule_id)
            if implicit_source_link:
                target_rule_id = source.memory_rule_id
            memory = next((item for item in rules if item.id == target_rule_id), None)
            if target_rule_id and memory is None:
                raise StaleConsolidationAborted(
                    "the automatic rule selected for refinement no longer exists"
                )
            if memory is not None:
                existing_evidence = set(memory.source_feedback_ids or [])
                if memory.user_locked and existing_evidence.intersection(clean_ids):
                    if (
                        implicit_source_link
                        and memory.origin == "manual"
                        and not self._manual_rule_link_is_valid(
                            memory,
                            records_by_id,
                        )
                    ):
                        # The correction changed after this manual rule was
                        # confirmed.  Preserve the disabled historical rule,
                        # but do not let its stale link swallow a fresh
                        # automatic consolidation.
                        memory = None
                        target_rule_id = ""
                    else:
                        source.memory_rule_id = memory.id
                        source.consolidation_status = "completed"
                        source.consolidation_error = ""
                        source.updated_at = _now_iso()
                        self._write_feedback(records)
                        return memory
                if memory is None:
                    existing_evidence = set()
                else:
                    target_is_safe = (
                        memory.origin == "automatic"
                        and not memory.user_locked
                        and memory.source_language == source.source_language
                        and memory.target_language == source.target_language
                        and existing_evidence.issubset({item.id for item in selected})
                    )
                    if not target_is_safe:
                        # Never overwrite a user-owned rule or an automatic rule
                        # whose complete historical evidence was not re-reviewed.
                        raise StaleConsolidationAborted(
                            "the selected automatic rule is no longer safe to refine"
                        )
            if memory is None:
                memory = MemoryRule.create(
                    source_language=source.source_language,
                    target_language=source.target_language,
                    trigger=trigger,
                    rule=rule,
                    source_feedback_id=source.id,
                    origin="automatic",
                )
                rules.append(memory)
            else:
                memory.trigger = trigger
                memory.rule = rule
                # A generalized automatic rule must not carry one full-sentence
                # translation as if it applied to every related source.
                memory.example_source = ""
                memory.preferred_translation = ""
                # A user may pause automatic enforcement while leaving its
                # contents automatic.  Consolidation can refresh evidence but
                # cannot override that deliberate disabled state.
                memory.enabled = memory.user_enabled_override is not False
                memory.updated_at = _now_iso()
            memory.source_feedback_ids = list(dict.fromkeys([
                *(memory.source_feedback_ids or []),
                *clean_ids,
            ]))
            memory.source_feedback_id = memory.source_feedback_ids[0]
            memory.source_feedback_digests = {
                feedback_id: records_by_id[feedback_id].consolidation_input_digest
                for feedback_id in memory.source_feedback_ids
            }
            # feedback_ids after the first are evidence, not ownership.  Do not
            # overwrite their existing rule linkage or consolidation state.
            for record in (source,):
                record.memory_rule_id = memory.id
                record.consolidation_status = "completed"
                record.consolidation_error = ""
                record.updated_at = _now_iso()
            # The feedback ownership link is committed before publishing the
            # new/updated automatic rule.  The rollback journal restores both
            # old files if either write fails.
            self._commit_cross_file_state(records, rules, order="feedback_first")
            return memory

    def match_memory_rules(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        limit: int = 3,
    ) -> list[MemoryRule]:
        # Preserve source case.  Trigger matching decides explicitly whether a
        # token is case-sensitive; pre-lowering here defeated that boundary.
        haystack = text
        feedback_by_id = {item.id: item for item in self.list_feedback()}
        candidates: list[tuple[MemoryRule, list[str], set[str]]] = []
        exact_case_identities: set[str] = set()
        for rule in self.list_memory_rules(enabled_only=True):
            if not self._memory_rule_is_active(
                rule,
                feedback_by_id,
                source_language=source_language,
                target_language=target_language,
            ):
                continue
            triggers = self.split_triggers(rule.trigger)
            if not triggers:
                continue
            matched = [
                trigger
                for trigger in triggers
                if self._trigger_matches_source(trigger, haystack)
            ]
            if matched:
                exact = {
                    trigger
                    for trigger in matched
                    if self._trigger_matches_source_exact_case(trigger, haystack)
                }
                exact_case_identities.update(
                    self._trigger_identity(trigger) for trigger in exact
                )
                candidates.append((rule, matched, exact))
        matches: list[tuple[int, int, int, str, MemoryRule]] = []
        for rule, matched, exact in candidates:
            effective = [
                trigger
                for trigger in matched
                if trigger in exact
                or self._trigger_identity(trigger) not in exact_case_identities
            ]
            if not effective:
                continue
            matches.append(
                (
                    1 if rule.user_locked else 0,
                    max(len(trigger) for trigger in effective),
                    len(effective),
                    rule.updated_at,
                    rule,
                )
            )
        matches.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        return [item[4] for item in matches[:limit]]

    def memory_rule_is_active(
        self,
        rule: MemoryRule,
        *,
        source_language: str,
        target_language: str,
    ) -> bool:
        """Validate one canonical rule without imposing a retrieval strategy."""

        feedback_by_id = {item.id: item for item in self.list_feedback()}
        return self._memory_rule_is_active(
            rule,
            feedback_by_id,
            source_language=source_language,
            target_language=target_language,
        )

    def active_memory_rules(
        self,
        *,
        source_language: str,
        target_language: str,
    ) -> list[MemoryRule]:
        """Return canonical active rules with one feedback/rule snapshot read."""

        with self._lock:
            feedback_by_id = {item.id: item for item in self.list_feedback()}
            return [
                rule
                for rule in self.list_memory_rules(enabled_only=False)
                if self._memory_rule_is_active(
                    rule,
                    feedback_by_id,
                    source_language=source_language,
                    target_language=target_language,
                )
            ]

    @classmethod
    def _memory_rule_is_active(
        cls,
        rule: MemoryRule,
        feedback_by_id: dict[str, FeedbackRecord],
        *,
        source_language: str,
        target_language: str,
    ) -> bool:
        if not rule.enabled:
            return False
        if rule.source_language and rule.source_language != source_language:
            return False
        if rule.target_language and rule.target_language != target_language:
            return False
        if rule.origin == "automatic" and not rule.user_locked:
            return bool(
                cls.automatic_trigger_is_specific(rule.trigger)
                and cls._automatic_rule_evidence_is_active(
                    rule,
                    feedback_by_id,
                    source_language=source_language,
                    target_language=target_language,
                )
            )
        if rule.origin == "manual":
            return cls._manual_rule_link_is_valid(rule, feedback_by_id)
        return bool(rule.user_locked)

    def match_corrections(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        limit: int = 3,
        exclude_feedback_id: str = "",
        minimum_score: float = 0.42,
    ) -> list[FeedbackRecord]:
        """Retrieve confirmed examples conservatively without hardcoding output."""

        ranked = self.rank_corrections(
            text,
            source_language=source_language,
            target_language=target_language,
            limit=limit,
            exclude_feedback_id=exclude_feedback_id,
            minimum_score=minimum_score,
        )
        return [item[1] for item in ranked]

    def rank_corrections(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        limit: int = 3,
        exclude_feedback_id: str = "",
        minimum_score: float = 0.42,
    ) -> list[tuple[float, FeedbackRecord]]:
        """Compatibility projection of :meth:`rank_correction_matches`."""

        return [
            (match.score, match.record)
            for match in self.rank_correction_matches(
                text,
                source_language=source_language,
                target_language=target_language,
                limit=limit,
                exclude_feedback_id=exclude_feedback_id,
                minimum_score=minimum_score,
            )
        ]

    def rank_correction_matches(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        limit: int = 3,
        exclude_feedback_id: str = "",
        minimum_score: float = 0.42,
    ) -> list[CorrectionMatch]:
        """Return conservative matches with an explicit exact-source flag.

        Similarity normalization is deliberately separate from source
        identity.  Only the strict key can produce ``exact_match=True`` or a
        score of 1.0; punctuation and operators must not be erased into a
        false exact match.
        """

        query = self._normalized_source(text)
        exact_query = self._exact_source_key(text)
        if not query or not exact_query:
            return []
        ranked: list[tuple[float, str, int, CorrectionMatch]] = []
        for position, record in enumerate(self.list_feedback()):
            if (
                record.id == exclude_feedback_id
                or record.status not in {"accepted", "confirmed"}
                or not record.enabled
            ):
                continue
            if record.source_language != source_language or record.target_language != target_language:
                continue
            if not record.corrected_translation.strip():
                continue
            candidate = self._normalized_source(record.ocr_text)
            if not candidate:
                continue
            exact_match = exact_query == self._exact_source_key(record.ocr_text)
            keyword_hit = any(
                self._trigger_matches_source(keyword, text)
                for keyword in (record.keywords or [])
                if keyword.strip() and self._keyword_is_specific(keyword)
            )
            if exact_match:
                score = 1.0
            else:
                if not self._semantic_anchors_compatible(text, record.ocr_text):
                    continue
                sequence = SequenceMatcher(None, query, candidate).ratio()
                grams = self._ngram_jaccard(query, candidate)
                lexical_score = sequence * 0.72 + grams * 0.28
                # Keywords improve ranking only when the surrounding sentence is
                # also plausibly related.  Generic words must never force a hit.
                score = (
                    min(0.998, lexical_score + 0.15)
                    if keyword_hit and lexical_score >= 0.28
                    else min(0.998, lexical_score)
                )
            if score >= minimum_score:
                ranked.append((
                    score,
                    record.updated_at,
                    position,
                    CorrectionMatch(
                        score=score,
                        record=record,
                        exact_match=exact_match,
                    ),
                ))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected: list[CorrectionMatch] = []
        seen_exact_keys: set[str] = set()
        for _, _, _, match in ranked:
            if match.exact_match:
                identity = self._exact_source_key(match.record.ocr_text)
                if identity in seen_exact_keys:
                    continue
                seen_exact_keys.add(identity)
            selected.append(match)
            if len(selected) >= max(0, limit):
                break
        return selected

    @staticmethod
    def correction_prompt_hint(
        record: FeedbackRecord,
        *,
        score: float | None = None,
        exact_match: bool = False,
    ) -> str:
        source = FeedbackStore._bounded_prompt_text(record.ocr_text, 600)
        translation = FeedbackStore._bounded_prompt_text(record.corrected_translation, 600)
        authority = (
            "这是与当前原文完全一致的用户确认纠错，优先采用该语义修正。"
            if exact_match
            else (
                f"这是相似纠错示例（检索分数 {score:.3f}），不是当前句的固定译文。"
                "仅借鉴确实适用的语义关系；不得照搬主体、时间、数量或完成状态。"
                if score is not None
                else "这是用户确认的纠错示例；仅在当前语义确实相同时借鉴。"
            )
        )
        parts = [authority, f"用户确认纠错案例原文：{source}", f"用户确认译文：{translation}"]
        return "；".join(parts)

    @staticmethod
    def _bounded_prompt_text(value: str, limit: int) -> str:
        clean = str(value).replace("\x00", "").strip()
        return clean if len(clean) <= limit else clean[:limit] + "…"

    @staticmethod
    def _coerce_bool(value, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().casefold()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        # Missing fields are handled by _stored_bool.  A present malformed
        # value must fail closed rather than silently enabling knowledge.
        return False

    @staticmethod
    def _stored_bool(data: dict, key: str, *, default: bool) -> bool:
        if key not in data:
            return default
        return FeedbackStore._coerce_bool(data.get(key), default=False)

    @staticmethod
    def _stored_nonnegative_int(data: dict, key: str, *, default: int) -> int:
        if key not in data:
            return default
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return default
        return max(0, value)

    @staticmethod
    def _consolidation_digest(
        corrected_translation: str,
        note: str,
        keywords: list[str],
        ai_problem_summary: str,
    ) -> str:
        payload = json.dumps(
            {
                "corrected_translation": corrected_translation,
                "note": note,
                "keywords": keywords,
                "ai_problem_summary": ai_problem_summary,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _valid_input_digest(value: str) -> bool:
        return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))

    def feedback_input_digest_snapshot(self) -> dict[str, str]:
        """Freeze every feedback input version before a remote consolidation.

        The snapshot is intentionally broader than the eventual evidence set:
        the thinking model chooses related rows during the call, so the caller
        cannot know those IDs in advance.  Commit code later selects the IDs in
        the proposal and atomically verifies their pre-call versions.
        """

        with self._lock:
            snapshot: dict[str, str] = {}
            for record in self.list_feedback():
                digest = record.consolidation_input_digest
                if not self._valid_input_digest(digest):
                    digest = self._consolidation_digest(
                        record.corrected_translation,
                        record.note,
                        list(record.keywords or []),
                        record.ai_problem_summary,
                    )
                snapshot[record.id] = digest
            return snapshot

    @staticmethod
    def join_triggers(triggers: list[str]) -> str:
        """Store multiple trigger keywords in a backward-compatible string field."""

        unique: list[str] = []
        seen: set[str] = set()
        for trigger in triggers:
            text = str(trigger).strip()
            if not text:
                continue
            # Upper/mixed-case ASCII tokens can be semantic identities
            # (May/may, Polish/polish, API/api).  Do not collapse them into a
            # case-insensitive duplicate here.
            key = text if re.search(r"[A-Z]", text) else text.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
        if any(
            text.startswith(FeedbackStore._TRIGGER_LIST_PREFIX)
            or re.search(r"(?:\s+[/／]\s+|[,，;；|])", text)
            for text in unique
        ):
            # The historical ``" / "`` delimiter is ambiguous when it is
            # part of one real phrase (for example "input / output").  Encode
            # only ambiguous lists; ordinary legacy values remain readable.
            return FeedbackStore._TRIGGER_LIST_PREFIX + json.dumps(
                unique,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return " / ".join(unique)

    @staticmethod
    def _normalized_keywords(keywords: list[str]) -> list[str]:
        normalized = FeedbackStore.split_triggers(FeedbackStore.join_triggers(keywords))
        return [item for item in normalized if 2 <= len(item) <= 64][:8]

    @staticmethod
    def _normalized_source(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(text)).casefold()
        return re.sub(r"[^\w\u3400-\u9fff]+", "", normalized)

    @staticmethod
    def _exact_source_key(text: str) -> str:
        """Normalize presentation only, preserving semantic punctuation."""

        # Case is semantic for source code, identifiers and ordinary words
        # such as US/us or Polish/polish.  Exact authority therefore does not
        # case-fold; case-insensitive similarity remains available below.
        normalized = unicodedata.normalize("NFKC", str(text)).strip()
        return re.sub(r"\s+", " ", normalized)

    @staticmethod
    def exact_source_identity_key(text: str) -> str:
        """Public identity primitive for authoritative retrieval adapters."""

        return FeedbackStore._exact_source_key(text)

    @staticmethod
    def _trigger_matches_source(trigger: str, source: str) -> bool:
        """Match CJK phrases literally and ASCII tokens on identifier boundaries."""

        needle = unicodedata.normalize("NFKC", trigger).strip()
        haystack = unicodedata.normalize("NFKC", source)
        if not needle:
            return False
        prefix = r"(?<![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", needle) else ""
        suffix = r"(?![A-Za-z0-9_])" if re.search(r"[A-Za-z0-9_]$", needle) else ""
        flags = 0 if re.search(r"[A-Z]", needle) else re.I
        return bool(re.search(prefix + re.escape(needle) + suffix, haystack, flags=flags))

    @staticmethod
    def _trigger_matches_source_exact_case(trigger: str, source: str) -> bool:
        needle = unicodedata.normalize("NFKC", trigger).strip()
        haystack = unicodedata.normalize("NFKC", source)
        if not needle:
            return False
        prefix = r"(?<![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", needle) else ""
        suffix = r"(?![A-Za-z0-9_])" if re.search(r"[A-Za-z0-9_]$", needle) else ""
        return bool(re.search(prefix + re.escape(needle) + suffix, haystack))

    @staticmethod
    def _trigger_identity(trigger: str) -> str:
        return unicodedata.normalize("NFKC", trigger).strip().casefold()

    @classmethod
    def trigger_field_matches_source(cls, trigger: str, source: str) -> bool:
        """Match any decoded trigger without exposing storage encoding details."""

        return any(
            cls._trigger_matches_source(item, source)
            for item in cls.split_triggers(trigger)
        )

    @classmethod
    def _keyword_is_specific(cls, keyword: str) -> bool:
        generic = {
            "system", "software", "service", "data", "user", "app",
            "issue", "problem", "task", "status", "request", "result",
            "系统", "软件", "服务", "数据", "用户", "程序", "功能", "问题",
            "设置", "信息", "任务", "接口", "界面", "按钮", "文件", "请求",
            "申请", "处理", "结果", "内容", "状态", "连接", "完成",
            "工作", "操作", "流程", "模块", "页面", "网络",
        }
        normalized_keyword = unicodedata.normalize("NFKC", keyword).strip().casefold()
        compact = re.sub(r"[^\w\u3400-\u9fff]+", "", normalized_keyword)
        if (
            not compact
            or compact in generic
            or not re.search(r"[A-Za-z\u3400-\u9fff]", compact)
        ):
            return False
        # Reject phrases made only by concatenating broad concepts, without
        # maintaining an ever-growing blacklist of every possible combination.
        generic_compact = {
            re.sub(r"[^\w\u3400-\u9fff]+", "", item.casefold())
            for item in generic
        }
        # Closed-class grammar/connective tokens are normalized separately
        # from domain concepts.  This prevents phrases such as “系统的状态”
        # from bypassing the broad-concept check without growing a blacklist
        # of every possible complete phrase.
        generic_compact.update({
            "的", "与", "和", "及", "或", "相关", "有关", "关于", "方面", "类",
            "the", "a", "an", "of", "for", "and", "or", "related", "about",
        })
        reachable = {0}
        for start in range(len(compact)):
            if start not in reachable:
                continue
            for term in generic_compact:
                if compact.startswith(term, start):
                    reachable.add(start + len(term))
        return len(compact) not in reachable

    @classmethod
    def automatic_trigger_is_specific(cls, trigger: str) -> bool:
        """Reject automatic triggers that would match a broad unrelated domain."""

        parts = cls.split_triggers(trigger)
        if not parts:
            return False
        for part in parts:
            cleaned = part.strip()
            if not cls._keyword_is_specific(cleaned):
                return False
            compact = re.sub(r"[^\w\u3400-\u9fff]+", "", cleaned.casefold())
            # Automatic rules are more dangerous than manually confirmed ones:
            # a two-character CJK token is often a broad noun/verb (状态、完成、
            # 连接).  Keep those as correction keywords, but do not let them
            # activate a generated long-term rule without richer context.
            if len(compact) < 3:
                return False
        return True

    @staticmethod
    def _semantic_anchors_compatible(left: str, right: str) -> bool:
        """Conservatively reject lexical neighbors with conflicting logic anchors."""

        categories = (
            (
                None,
                "negation",
            ),
            (
                r"(?:如果|若(?:是|果|有|要|能|需|在|非|无|無|不|未|则|則)|"
                r"若(?=[，,\s])|只要|倘若|除非|\b(?:if|unless|provided that)\b|"
                r"もし|なら|場合)",
                "condition",
            ),
            (
                r"(?:除了|除外|例外|但不包括|\b(?:except|excluding)\b|除く|以外)",
                "exception",
            ),
        )
        left_folded = left.casefold()
        right_folded = right.casefold()
        left_ascii_words = re.findall(r"[A-Za-z]+", unicodedata.normalize("NFKC", left))
        right_ascii_words = re.findall(r"[A-Za-z]+", unicodedata.normalize("NFKC", right))
        if (
            left_ascii_words
            and [item.casefold() for item in left_ascii_words]
            == [item.casefold() for item in right_ascii_words]
            and left_ascii_words != right_ascii_words
        ):
            # If every ASCII word is otherwise identical, case is the only
            # available semantic distinction (US/us, Polish/polish, May/may).
            # Treat it conservatively instead of awarding a near-exact score.
            return False
        for pattern, _name in categories:
            left_match = (
                FeedbackStore._has_negation_anchor(left_folded)
                if pattern is None
                else bool(re.search(pattern, left_folded, flags=re.I))
            )
            right_match = (
                FeedbackStore._has_negation_anchor(right_folded)
                if pattern is None
                else bool(re.search(pattern, right_folded, flags=re.I))
            )
            if left_match != right_match:
                return False
        if FeedbackStore._symbolic_anchors(left) != FeedbackStore._symbolic_anchors(right):
            return False
        if (
            FeedbackStore._typed_semantic_anchors(left_folded)
            != FeedbackStore._typed_semantic_anchors(right_folded)
        ):
            return False
        if FeedbackStore._sentence_modality(left) != FeedbackStore._sentence_modality(right):
            return False

        left_states = FeedbackStore._semantic_states(left_folded)
        right_states = FeedbackStore._semantic_states(right_folded)
        # State-bearing corrections are high-risk evidence.  If only one side
        # carries an explicit state, do not infer that the missing state is
        # equivalent; exact-source matches are handled before this path.
        if left_states != right_states:
            return False

        left_quantities = FeedbackStore._quantity_anchors(left_folded)
        right_quantities = FeedbackStore._quantity_anchors(right_folded)
        if left_quantities != right_quantities:
            return False
        left_numbers = re.findall(r"\d+(?:\.\d+)?", left_folded)
        right_numbers = re.findall(r"\d+(?:\.\d+)?", right_folded)
        if left_numbers != right_numbers:
            return False
        return True

    @staticmethod
    def semantic_sources_compatible(left: str, right: str) -> bool:
        """Public safety gate for lexical, vector, and hybrid retrieval."""

        return FeedbackStore._semantic_anchors_compatible(left, right)

    @staticmethod
    def _has_negation_anchor(text: str) -> bool:
        """Detect explicit negation without matching place/domain words.

        Bare ``无``/``非`` and every occurrence of ``不`` previously treated
        无锡、非常 and comparison phrases as negation.  Closed phrases and
        verb-bearing ``不`` forms are safer retrieval boundaries.
        """

        pattern = (
            r"(?:没有|没能|没在|没做|没完成|没通过|没提交|没处理|尚未|"
            r"无法|无需|无须|无权|不能|不可|不要|不得|不允许|不允許|"
            r"不可以|不再|并不|並不|并没有|並沒有|绝不|絕不|从未|從未|"
            r"而非|并非|並非|勿|别(?:再|让|讓|把|做|用|提交|处理|處理)|"
            r"不(?:是|会|會|要|应|應|该|該|需|接受|支持|包含|包括|执行|執行|"
            r"处理|處理|提交|保存|启动|啟動|运行|運行|更新|完成|通过|通過|"
            r"显示|顯示|返回|使用|存在|工作|继续|繼續)|"
            r"\b(?:not|no|never|without|cannot|can't|isn't|aren't|didn't|doesn't|"
            r"won't|hasn't|haven't)\b|ない|ません|ず|ぬ|なし)"
        )
        return bool(re.search(pattern, text, flags=re.I))

    @staticmethod
    def _symbolic_anchors(text: str) -> tuple[str, ...]:
        """Preserve operators and punctuation-bearing technical identifiers."""

        normalized = unicodedata.normalize("NFKC", str(text))
        operators = re.findall(r"===|!==|==|!=|>=|<=|->|=>|\+\+|--|&&|\|\||>|<", normalized)
        signed_numbers = re.findall(
            r"(?<![A-Za-z0-9_.])[+-]\s*\d+(?:\.\d+)?",
            normalized,
        )
        technical = re.findall(
            r"(?<![A-Za-z0-9_])"
            r"[A-Za-z][A-Za-z0-9_]*"
            r"(?:[+#]+|(?:[/._-][A-Za-z0-9_+#-]+)+)"
            r"(?![A-Za-z0-9_])",
            normalized,
        )
        infix_operators = re.findall(
            r"(?<=[A-Za-z0-9_)])(?:\+|\*|=|::|%|\^|~|@|&|\||#)(?=[A-Za-z0-9_(])",
            normalized,
        )
        return tuple(sorted([*operators, *signed_numbers, *technical, *infix_operators]))

    @staticmethod
    def _typed_semantic_anchors(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Extract high-risk closed-class relations for conservative matching.

        These categories express logical relations rather than domain words.
        Keeping them in one structured signature avoids an expanding series of
        unrelated one-off guards in the ranking loop.
        """

        folded = unicodedata.normalize("NFKC", text).casefold()
        event = (
            r"(?:更新|升级|提交|部署|启动|运行|执行|处理|安装|保存|重启|"
            r"关闭|完成|审核|发送|加载)"
        )
        groups: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
            (
                "deontic",
                (
                    (
                        "prohibition",
                        r"(?:禁止|严禁|不得|不准|不许|不允许|不允許|不可以|不能|勿|"
                        r"\b(?:must not|may not|not allowed|prohibited|forbidden)\b|"
                        r"禁止|してはいけない)",
                    ),
                    (
                        "obligation",
                        r"(?:必须|必須|务必|務必|应当|應當|需要|须要|須要|"
                        r"\b(?:must|required|have to|has to|need to|needs to)\b|"
                        r"なければならない|必要)",
                    ),
                    (
                        "permission",
                        r"(?:(?<!不)允许|(?<!不)允許|准许|准許|许可|許可|"
                        r"(?<!不)可以|\b(?:may|permitted)\b|(?<!not )\ballowed\b|"
                        r"してもよい)",
                    ),
                    (
                        "possibility",
                        r"(?:可能|(?<!不)能够|(?<!不)能夠|"
                        r"(?<![不功性智体體产產效])能(?=(?:否|运行|運行|执行|執行|"
                        r"处理|處理|完成|使用|访问|訪問|连接|連接|启动|啟動|保存|"
                        r"提交|通过|通過|实现|實現|支持|显示|顯示|返回|读取|讀取|"
                        r"写入|寫入|翻译|翻譯))|"
                        r"\b(?:can|could|possible)\b|可能|できる)",
                    ),
                ),
            ),
            (
                "relative_day",
                (
                    ("yesterday", r"(?:昨天|昨日|\byesterday\b)"),
                    ("today", r"(?:今天|今日|\btoday\b)"),
                    ("tomorrow", r"(?:明天|明日|\btomorrow\b)"),
                ),
            ),
            (
                "event_phase",
                (
                    (
                        "before",
                        rf"(?:{event}(?:之前|以前|前)|\bbefore\b|前に)",
                    ),
                    (
                        "during",
                        rf"(?:{event}(?:时|時|期间|期間|过程中|過程中|的时候)|"
                        r"\b(?:during|while)\b|中に|際に)",
                    ),
                    (
                        "after",
                        rf"(?:{event}(?:之后|之後|以后|以後|后(?!台|端|续|續|门|門)|"
                        r"後(?!台|端|続|門))|\bafter\b|後に)",
                    ),
                ),
            ),
            (
                "comparison",
                (
                    (
                        "at_most",
                        r"(?:最多|至多|不超过|不超過|不高于|不高於|<=|≤|"
                        r"\b(?:at most|no more than)\b)",
                    ),
                    (
                        "at_least",
                        r"(?:至少|最少|不少于|不少於|不低于|不低於|>=|≥|"
                        r"\b(?:at least|no less than)\b)",
                    ),
                    (
                        "greater",
                        r"(?:大于|大於|(?<!不)超过|(?<!不)超過|(?<!不)高于|(?<!不)高於|"
                        r"(?<![<>=-])>(?![=])|"
                        r"\b(?:greater than|more than)\b)",
                    ),
                    (
                        "less",
                        r"(?:小于|小於|(?<!不)少于|(?<!不)少於|(?<!不)低于|(?<!不)低於|"
                        r"(?<![<>=])<(?![=])|"
                        r"\b(?:less than|fewer than)\b)",
                    ),
                ),
            ),
        )
        signature: list[tuple[str, tuple[str, ...]]] = []
        for category, patterns in groups:
            matched = tuple(
                name
                for name, pattern in patterns
                if re.search(pattern, folded, flags=re.I)
            )
            signature.append((category, matched))
        return tuple(signature)

    @staticmethod
    def _sentence_modality(text: str) -> str:
        stripped = re.sub(r"[\s。.!！]+$", "", text.strip())
        if re.search(r"[?？]", text):
            return "question"
        if re.search(r"(?:吗|嗎|么|麼|呢)$", stripped):
            return "question"
        if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]か$", stripped):
            return "question"
        if re.match(
            r"\s*(?:谁|誰|什么|什麼|为什么|為什麼|为何|為何|怎么|怎麼|"
            r"如何|哪(?:个|個|些|里|裡)?|是否|能否|可否|なぜ|どう|どこ|誰|何)",
            stripped,
        ):
            return "question"
        if re.match(
            r"\s*(?:who|what|when|where|why|how|do|does|did|is|are|was|were|"
            r"can|could|will|would|should|has|have|had)\b",
            stripped,
            flags=re.I,
        ):
            return "question"
        return "statement"

    @staticmethod
    def _semantic_states(text: str) -> set[str]:
        state_patterns = {
            "future": (
                r"(?:即将|将要|马上要|准备要|计划|預定|预定|稍后会|随后会|"
                r"\b(?:will|shall|going to|soon|planned|scheduled)\b|"
                r"予定|つもり|まもなく)"
            ),
            "ongoing": (
                r"(?:正在|仍在|还在|持續|持续|进行中|处理中|"
                r"\b(?:currently|still|ongoing|in progress)\b|"
                r"\b(?:has|have|had)\s+been\s+[a-z]+ing\b|"
                r"ている|ています|継続中|進行中)"
            ),
            "completed": (
                r"(?:(?:已经|已經|业已|業已|已).{0,6}(?:完成|结束|結束|提交|处理|處理)|"
                r"(?:完成|结束|結束|提交|处理|處理)(?:了|完毕|完畢)|"
                r"\b(?:already|completed|finished)\b|"
                r"完了した|完了済み|済み|終わった)"
            ),
            "pending": (
                r"(?:等待|待处理|待處理|待完成|尚未|还没|還沒|未完成|"
                r"\b(?:pending|awaiting|not yet)\b|未完了|保留中|待機中)"
            ),
        }
        states = {
            name
            for name, pattern in state_patterns.items()
            if re.search(pattern, text, flags=re.I)
        }
        if states.intersection({"future", "ongoing", "pending"}):
            states.discard("completed")
        if not states and not re.search(r"(?:请|請|必须|必須|需要|应当|應當|不要|勿)", text):
            terminal = re.sub(r"[\s。！？!?.,，]+$", "", text)
            if re.search(r"(?:任务|任務|工作|操作|流程|申请|申請|审核|審核|部署|同步|处理|處理).{0,8}(?:完成|结束|結束|通过|通過|关闭|關閉|停止)$", terminal):
                states.add("completed")
        return states

    @staticmethod
    def _quantity_anchors(text: str) -> tuple[str, ...]:
        number = r"(?:\d+(?:\.\d+)?|[零〇一二两兩三四五六七八九十百千万萬亿億]+)"
        unit = r"(?:个|個|次|天|日|小时|小時|分钟|分鐘|秒|年|月|周|週|台|项|項|条|條|份|人|%)"
        return tuple(re.findall(number + unit, text, flags=re.I))

    @staticmethod
    def _automatic_rule_evidence_is_active(
        rule: MemoryRule,
        feedback_by_id: dict[str, FeedbackRecord],
        *,
        source_language: str,
        target_language: str,
    ) -> bool:
        evidence_ids = list(rule.source_feedback_ids or [])
        if not evidence_ids:
            return False
        evidence_digests = rule.source_feedback_digests or {}
        if set(evidence_digests) != set(evidence_ids) or any(
            not FeedbackStore._valid_input_digest(evidence_digests.get(item, ""))
            for item in evidence_ids
        ):
            # Legacy automatic rules without an evidence-version snapshot are
            # retained for audit/UI review but are not injected at runtime.
            return False
        owner_id = rule.source_feedback_id or evidence_ids[0]
        owner = feedback_by_id.get(owner_id)
        if (
            owner is None
            or owner.memory_rule_id != rule.id
            or owner.consolidation_status != "completed"
        ):
            return False
        for feedback_id in evidence_ids:
            record = feedback_by_id.get(feedback_id)
            if (
                record is None
                or not record.enabled
                or record.status not in {"accepted", "confirmed"}
                or not record.corrected_translation.strip()
                or record.source_language != source_language
                or record.target_language != target_language
                or record.consolidation_input_digest != evidence_digests[feedback_id]
            ):
                return False
        return True

    @staticmethod
    def _manual_rule_link_is_valid(
        rule: MemoryRule,
        feedback_by_id: dict[str, FeedbackRecord],
    ) -> bool:
        """Reject manual rules that were never durably linked to confirmation."""

        record = feedback_by_id.get(rule.source_feedback_id)
        return bool(
            record is not None
            and record.status == "confirmed"
            and record.memory_rule_id == rule.id
        )

    @classmethod
    def _validate_memory_text(cls, trigger: str, rule: str) -> None:
        if len(trigger) > cls.MAX_TRIGGER_CHARS:
            raise ValueError(
                f"memory trigger exceeds {cls.MAX_TRIGGER_CHARS} characters"
            )
        if len(rule) > cls.MAX_RULE_CHARS:
            raise ValueError(f"memory rule exceeds {cls.MAX_RULE_CHARS} characters")

    @staticmethod
    def _ngram_jaccard(left: str, right: str, size: int = 2) -> float:
        def grams(value: str) -> set[str]:
            if len(value) <= size:
                return {value} if value else set()
            return {value[index:index + size] for index in range(len(value) - size + 1)}

        a = grams(left)
        b = grams(right)
        return len(a & b) / len(a | b) if a and b else 0.0

    @staticmethod
    def split_triggers(trigger: str) -> list[str]:
        """Return one or more trigger keywords from the stored trigger field."""

        raw = trigger.strip()
        if raw.startswith(FeedbackStore._TRIGGER_LIST_PREFIX):
            try:
                loaded = json.loads(raw[len(FeedbackStore._TRIGGER_LIST_PREFIX):])
            except json.JSONDecodeError:
                return []
            if not isinstance(loaded, list) or any(
                not isinstance(item, str) for item in loaded
            ):
                return []
            return list(dict.fromkeys(
                item.strip() for item in loaded if item.strip()
            ))
        # ``join_triggers`` historically emitted ``" / "``.  A bare slash is
        # also part of common technical identifiers (HTTP/2, CI/CD, TCP/IP),
        # so only the whitespace-delimited legacy form is a separator.
        parts = re.split(
            r"(?:\s+[/／]\s+|\s*(?:,|，|;|；|\|)\s*)",
            raw,
        )
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def suggest_trigger(text: str) -> str:
        """Return a short trigger candidate from source text."""

        cleaned = re.sub(r"\s+", "", text)
        cjk_chunks = re.findall(r"[\u3400-\u9fff]{2,8}", cleaned)
        if cjk_chunks:
            return cjk_chunks[0]
        latin_chunks = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,20}", cleaned)
        return latin_chunks[0] if latin_chunks else cleaned[:12]

    def _load_unique_rows(self, path: Path, validator, factory) -> list[object]:
        """Parse one canonical file and fail closed on every duplicate ID.

        Choosing the first or last duplicate would make file order determine
        runtime knowledge.  The complete conflicting group is therefore kept
        for quarantine and excluded until a human or migration resolves it.
        """

        parsed: list[tuple[object, object]] = []
        unparsed: list[object] = []
        for item in self._read_list(path):
            if not validator(item):
                unparsed.append(item)
                continue
            try:
                parsed.append((item, factory(item)))
            except (AttributeError, TypeError, ValueError):
                unparsed.append(item)
        counts = Counter(str(getattr(record, "id", "")) for _, record in parsed)
        duplicate_ids = {identifier for identifier, count in counts.items() if count > 1}
        accepted: list[object] = []
        for raw, record in parsed:
            if str(getattr(record, "id", "")) in duplicate_ids:
                unparsed.append(raw)
            else:
                accepted.append(record)
        self._unparsed_rows[path] = unparsed
        self._report_invalid_rows(path, len(unparsed))
        return accepted

    def _read_list(self, path: Path) -> list[object]:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except UnicodeDecodeError:
            self._corrupt_files.add(path)
            self._report_storage_problem(
                path,
                "存储文件无法解析；下一次写入前会保留按内容哈希命名的备份。",
            )
            return []
        except OSError as exc:
            # An access/share/permission failure says nothing about the file
            # contents.  Treating it as an empty corrupt file lets the next
            # write replace healthy canonical data.  Fail closed instead.
            raise FeedbackStorageUnavailable(
                f"feedback storage could not be read: {path}"
            ) from exc
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                self._corrupt_files.add(path)
                self._report_storage_problem(
                    path,
                    "存储顶层不是列表；下一次写入前会保留按内容哈希命名的备份。",
                )
                return []
            self._corrupt_files.discard(path)
            return data
        except json.JSONDecodeError:
            self._corrupt_files.add(path)
            self._report_storage_problem(
                path,
                "存储文件无法解析；下一次写入前会保留按内容哈希命名的备份。",
            )
            return []

    @staticmethod
    def _file_snapshot(path: Path) -> dict[str, object]:
        if not path.exists():
            return {"exists": False, "content_b64": ""}
        return {
            "exists": True,
            "content_b64": base64.b64encode(path.read_bytes()).decode("ascii"),
        }

    @staticmethod
    def _transaction_checksum(payload: dict) -> str:
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _commit_cross_file_state(
        self,
        records: list[FeedbackRecord],
        rules: list[MemoryRule],
        *,
        order: str,
    ) -> None:
        """Commit both canonical files or durably roll back their old bytes."""

        if order not in {"feedback_first", "memory_first"}:
            raise ValueError(f"unsupported feedback transaction order: {order}")
        with self._lock:
            # A previous interrupted operation is rolled back before capturing
            # a new baseline.  Never stack transactions on ambiguous state.
            self._ensure_transaction_recovered()
            body = {
                "version": 1,
                "recovery_action": "rollback",
                "feedback": self._file_snapshot(self._feedback_path),
                "memory": self._file_snapshot(self._memory_path),
            }
            journal = {**body, "checksum": self._transaction_checksum(body)}
            self._write_bytes_atomic(
                self._transaction_path,
                json.dumps(journal, ensure_ascii=True, indent=2).encode("utf-8"),
            )
            try:
                if order == "feedback_first":
                    self._write_feedback(records, bump_revision=False)
                    self._write_memory_rules(rules, bump_revision=False)
                else:
                    self._write_memory_rules(rules, bump_revision=False)
                    self._write_feedback(records, bump_revision=False)
                self._transaction_path.unlink()
                self._advance_knowledge_revision()
            except Exception:
                # Recovery restores exact pre-operation bytes.  If the device
                # is still unavailable, the journal remains for the next read
                # or process start rather than exposing a silent half-commit.
                self._recover_cross_file_transaction()
                raise

    def _recover_cross_file_transaction(self) -> bool:
        """Roll back an interrupted cross-file commit, idempotently."""

        if not self._transaction_path.exists():
            return True
        try:
            raw = self._transaction_path.read_bytes()
            journal = json.loads(raw.decode("utf-8"))
            if not isinstance(journal, dict):
                raise ValueError("transaction journal is not an object")
            checksum = journal.get("checksum")
            body = {key: value for key, value in journal.items() if key != "checksum"}
            if (
                journal.get("version") != 1
                or journal.get("recovery_action") != "rollback"
                or not isinstance(checksum, str)
                or checksum != self._transaction_checksum(body)
            ):
                raise ValueError("transaction journal checksum or schema is invalid")
            feedback_snapshot = journal.get("feedback")
            memory_snapshot = journal.get("memory")
            if not isinstance(feedback_snapshot, dict) or not isinstance(memory_snapshot, dict):
                raise ValueError("transaction journal snapshots are invalid")
            # Restore knowledge first.  Together with runtime evidence/link
            # checks, this keeps every partially restored state fail-closed.
            self._restore_file_snapshot(self._memory_path, memory_snapshot)
            self._restore_file_snapshot(self._feedback_path, feedback_snapshot)
            self._transaction_path.unlink()
            get_logger().warning(
                "Recovered interrupted feedback storage transaction | root=%s",
                self._root_dir,
            )
            return True
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
            try:
                raw = self._transaction_path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()[:12]
                backup = self._transaction_path.with_name(
                    self._transaction_path.name + f".corrupt-{digest}.backup"
                )
                if not backup.exists():
                    self._write_bytes_atomic(backup, raw)
            except OSError:
                self._report_storage_problem(
                    self._transaction_path,
                    "事务日志损坏且无法备份；反馈存储保持锁定。",
                )
                return False
            self._report_storage_problem(
                self._transaction_path,
                f"事务日志损坏，已备份但无法安全恢复；反馈存储保持锁定：{exc}",
            )
            # Keep the original journal as a persistent blocked marker.  Its
            # snapshots are not trustworthy, so neither deleting it nor
            # treating the backup as a successful rollback is safe.  An
            # operator or a future migration must explicitly resolve it.
            return False
        except OSError as exc:
            self._report_storage_problem(
                self._transaction_path,
                f"事务回滚尚未完成，将在后续读取时重试：{exc}",
            )
            return False

    @staticmethod
    def _restore_file_snapshot(path: Path, snapshot: dict) -> None:
        exists = snapshot.get("exists")
        encoded = snapshot.get("content_b64")
        if not isinstance(exists, bool) or not isinstance(encoded, str):
            raise ValueError("invalid transaction file snapshot")
        try:
            content = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
            raise ValueError("invalid transaction snapshot encoding") from exc
        if exists:
            FeedbackStore._write_bytes_atomic(path, content)
        else:
            path.unlink(missing_ok=True)

    def _write_feedback(
        self,
        records: list[FeedbackRecord],
        *,
        bump_revision: bool = True,
    ) -> None:
        self._quarantine_unparsed_rows(self._feedback_path)
        self._write_json_list(
            self._feedback_path,
            [asdict(record) for record in records],
        )
        if bump_revision:
            self._advance_knowledge_revision()

    def _write_memory_rules(
        self,
        rules: list[MemoryRule],
        *,
        bump_revision: bool = True,
    ) -> None:
        self._quarantine_unparsed_rows(self._memory_path)
        self._write_json_list(
            self._memory_path,
            [asdict(rule) for rule in rules],
        )
        if bump_revision:
            self._advance_knowledge_revision()

    @staticmethod
    def _valid_feedback_row(item: object) -> bool:
        if not isinstance(item, dict):
            return False
        required_strings = (
            "id", "source_language", "target_language", "ocr_text", "translation_text"
        )
        if any(not isinstance(item.get(key), str) for key in required_strings):
            return False
        optional_strings = (
            "created_at", "updated_at", "note", "corrected_translation",
            "status", "memory_rule_id", "ai_problem_summary",
            "consolidation_status", "consolidation_error", "consolidation_input_digest",
        )
        if any(key in item and not isinstance(item[key], str) for key in optional_strings):
            return False
        identifier = item.get("id", "")
        if (
            not identifier.strip()
            or identifier != identifier.strip()
            or len(identifier) > 128
        ):
            return False
        group_id = item.get("group_id")
        if isinstance(group_id, bool) or not isinstance(group_id, int):
            return False
        if item.get("status", "pending") not in {
            "pending", "accepted", "confirmed", "dismissed"
        }:
            return False
        if item.get("consolidation_status", "not_started") not in {
            "not_started", "queued", "running", "completed", "failed"
        }:
            return False
        if "enabled" in item and not isinstance(item["enabled"], bool):
            return False
        if "consolidation_attempts" in item and (
            isinstance(item["consolidation_attempts"], bool)
            or not isinstance(item["consolidation_attempts"], int)
            or item["consolidation_attempts"] < 0
        ):
            return False
        for key in ("keywords", "matched_memory_rule_ids", "matched_correction_ids"):
            if key in item:
                if not isinstance(item[key], list) or any(
                    not isinstance(value, str) for value in item[key]
                ):
                    return False
        for map_key, ids_key in (
            ("matched_memory_hint_snapshots", "matched_memory_rule_ids"),
            ("matched_correction_hint_snapshots", "matched_correction_ids"),
        ):
            if map_key not in item:
                continue
            snapshots = item[map_key]
            ids = set(item.get(ids_key, []))
            if (
                not isinstance(snapshots, dict)
                or len(snapshots) > 8
                or any(
                    not isinstance(key, str)
                    or not isinstance(value, str)
                    or not key.strip()
                    or key != key.strip()
                    or len(key) > 128
                    or not value.strip()
                    or len(value.strip()) > 2400
                    or key not in ids
                    for key, value in snapshots.items()
                )
            ):
                return False
        return True

    @staticmethod
    def _valid_memory_row(item: object) -> bool:
        if not isinstance(item, dict):
            return False
        for key in ("id", "source_language", "target_language", "trigger", "rule"):
            if not isinstance(item.get(key), str):
                return False
        for key in (
            "created_at", "updated_at", "example_source", "preferred_translation",
            "source_feedback_id", "origin",
        ):
            if key in item and not isinstance(item[key], str):
                return False
        identifier = item.get("id", "")
        if (
            not identifier.strip()
            or identifier != identifier.strip()
            or len(identifier) > 128
            or not item.get("trigger", "").strip()
        ):
            return False
        if not item.get("rule", "").strip():
            return False
        if "enabled" in item and not isinstance(item["enabled"], bool):
            return False
        if "user_locked" in item and not isinstance(item["user_locked"], bool):
            return False
        if "user_enabled_override" in item and item["user_enabled_override"] is not None and not isinstance(item["user_enabled_override"], bool):
            return False
        if item.get("origin", "legacy") not in {"legacy", "manual", "automatic"}:
            return False
        if "source_feedback_ids" in item and not isinstance(
            item["source_feedback_ids"], list
        ):
            return False
        if "source_feedback_ids" in item and any(
            not isinstance(value, str) for value in item["source_feedback_ids"]
        ):
            return False
        if "source_feedback_digests" in item:
            digests = item["source_feedback_digests"]
            if not isinstance(digests, dict) or any(
                not isinstance(feedback_id, str)
                or not feedback_id.strip()
                or not isinstance(digest, str)
                or not FeedbackStore._valid_input_digest(digest)
                for feedback_id, digest in digests.items()
            ):
                return False
        if item.get("origin", "legacy") == "manual":
            owner_id = item.get("source_feedback_id", "")
            evidence_ids = item.get("source_feedback_ids")
            if (
                not isinstance(owner_id, str)
                or not owner_id.strip()
                or (
                    evidence_ids is not None
                    and (
                        not isinstance(evidence_ids, list)
                        or owner_id not in evidence_ids
                    )
                )
            ):
                return False
        return True

    def _report_invalid_rows(self, path: Path, count: int) -> None:
        if count:
            self._report_storage_problem(
                path,
                f"检测到 {count} 条无效记录；将在下次写入时移至隔离文件。",
            )

    def _report_storage_problem(self, path: Path, message: str) -> None:
        if path in self._reported_storage_problems:
            return
        self._reported_storage_problems.add(path)
        get_logger().warning("Feedback storage warning | path=%s detail=%s", path, message)

    def _quarantine_unparsed_rows(self, path: Path) -> None:
        rows = list(self._unparsed_rows.get(path, []))
        if not rows:
            return
        quarantine = path.with_name(path.stem + ".invalid-rows.json")
        existing: list[object] = []
        if quarantine.exists():
            original = quarantine.read_bytes()
            try:
                loaded = json.loads(original.decode("utf-8"))
                if isinstance(loaded, list):
                    existing = loaded
                else:
                    raise ValueError("quarantine top level is not a list")
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                digest = hashlib.sha256(original).hexdigest()[:12]
                backup = quarantine.with_name(quarantine.name + f".corrupt-{digest}.backup")
                if not backup.exists():
                    # The canonical quarantine must not be replaced unless its
                    # original bytes have first been durably preserved.
                    self._write_bytes_atomic(backup, original)
        merged: list[object] = []
        seen: set[str] = set()
        for item in [*existing, *rows]:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        self._write_json_list(quarantine, merged)
        self._unparsed_rows[path] = []

    @staticmethod
    def _disable_automatic_rules_for_feedback(
        rules: list[MemoryRule],
        feedback_id: str,
    ) -> set[str]:
        disabled_rule_ids: set[str] = set()
        for memory in rules:
            if (
                memory.origin == "automatic"
                and not memory.user_locked
                and feedback_id in (memory.source_feedback_ids or [])
                and memory.enabled
            ):
                memory.enabled = False
                memory.updated_at = _now_iso()
                disabled_rule_ids.add(memory.id)
        return disabled_rule_ids

    @staticmethod
    def _mark_disabled_automatic_rule_owners(
        records: list[FeedbackRecord],
        rules: list[MemoryRule],
        disabled_rule_ids: set[str],
        *,
        changed_feedback_id: str,
    ) -> None:
        """Invalidate completed owner state when any evidence version changes."""

        if not disabled_rule_ids:
            return
        owners = {
            rule.source_feedback_id: rule.id
            for rule in rules
            if rule.id in disabled_rule_ids and rule.source_feedback_id
        }
        for record in records:
            rule_id = owners.get(record.id)
            if (
                not rule_id
                or record.id == changed_feedback_id
                or record.memory_rule_id != rule_id
            ):
                continue
            record.consolidation_status = "not_started"
            record.consolidation_error = (
                "该长期规则的一条来源纠错已变化，需要重新归纳。"
            )
            record.consolidation_attempts = 0
            record.updated_at = _now_iso()

    def _write_feedback_revoking_automatic_rules(
        self,
        records: list[FeedbackRecord],
        feedback_id: str,
    ) -> None:
        """Persist changed evidence only after every derived rule is revoked."""

        rules = self.list_memory_rules(enabled_only=False)
        disabled_rule_ids = self._disable_automatic_rules_for_feedback(
            rules,
            feedback_id,
        )
        self._mark_disabled_automatic_rule_owners(
            records,
            rules,
            disabled_rule_ids,
            changed_feedback_id=feedback_id,
        )
        changed = bool(disabled_rule_ids)
        record = next((item for item in records if item.id == feedback_id), None)
        linked_manual = next(
            (
                item for item in rules
                if record is not None
                and item.id == record.memory_rule_id
                and item.origin == "manual"
                and item.enabled
            ),
            None,
        )
        if linked_manual is not None:
            # A manual rule may include the previous preferred translation and
            # example.  Keep it for audit/editing, but fail closed until the
            # user reconfirms it against the changed correction.
            linked_manual.enabled = False
            linked_manual.updated_at = _now_iso()
            changed = True
        if changed:
            self._commit_cross_file_state(records, rules, order="memory_first")
        else:
            self._write_feedback(records)

    def _write_json_list(self, path: Path, rows: list[object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path in self._corrupt_files and path.exists():
            original = path.read_bytes()
            digest = hashlib.sha256(original).hexdigest()[:12]
            backup = path.with_name(path.name + f".corrupt-{digest}.backup")
            if not backup.exists():
                self._write_bytes_atomic(backup, original)
            self._corrupt_files.discard(path)
        self._write_bytes_atomic(
            path,
            json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    @staticmethod
    def _write_bytes_atomic(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{uuid4().hex}.tmp")
        try:
            tmp.write_bytes(content)
            tmp.replace(path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                # The canonical replace result is authoritative; a leftover
                # uniquely named temp file must not mask the original error.
                pass

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
        roots.append(Path(tempfile.gettempdir()) / "instant-translate" / "feedback")
        roots.append(Path.cwd() / ".instant-translate" / "feedback")
        return roots
