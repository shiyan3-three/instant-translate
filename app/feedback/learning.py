"""Business service for submitted corrections and automatic memory consolidation."""

from __future__ import annotations

import threading
import queue
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from app.feedback.optimizer import FeedbackOptimizer
from app.feedback.store import (
    FeedbackRecord,
    FeedbackStorageUnavailable,
    FeedbackStore,
    MemoryRule,
    StaleConsolidationAborted,
)
from app.logger import get_debug_logger, get_logger
from app.settings import AppSettings


@dataclass(frozen=True)
class _ConsolidationJob:
    feedback_id: str
    generation: int
    on_complete: Callable[[dict], None] | None = None


class FeedbackLearningService:
    """Commit corrections synchronously, then run Pro consolidation in a daemon hook."""

    QUEUE_CAPACITY = 64
    MAX_ATTEMPTS_PER_INPUT = 2

    def __init__(
        self,
        store: FeedbackStore,
        settings: AppSettings,
        optimizer: FeedbackOptimizer | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._optimizer = optimizer or FeedbackOptimizer(feedback_store=store)
        self._lock = threading.RLock()
        self._generations: dict[str, int] = {}
        self._owned_jobs: set[tuple[str, int]] = set()
        self._stopping = False
        self._queue: queue.Queue[_ConsolidationJob | None] = queue.Queue(
            maxsize=self.QUEUE_CAPACITY
        )
        self._recovery_jobs: deque[_ConsolidationJob] = deque()
        self._last_job_from_recovery = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="feedback-memory-worker",
            daemon=True,
        )
        # Recover durable work before starting the daemon.  If recovery itself
        # fails during construction, no orphan worker is left behind.
        try:
            recovered_jobs = self._recover_pending()
            self._recovery_jobs.extend(recovered_jobs)
            self._owned_jobs.update(
                (job.feedback_id, job.generation) for job in recovered_jobs
            )
        except FeedbackStorageUnavailable as exc:
            # The desktop may run in translation-only mode while an ambiguous
            # rollback journal awaits repair.  Store methods keep failing
            # closed; starting the worker still lets a later repaired store
            # accept work without rebuilding the page.
            get_logger().warning(
                "Feedback recovery deferred because storage is unavailable: %s",
                exc,
            )
        self._worker.start()

    def submit_correction(
        self,
        feedback_id: str,
        *,
        corrected_translation: str,
        note: str = "",
        keywords: list[str] | None = None,
        ai_problem_summary: str = "",
        on_complete: Callable[[dict], None] | None = None,
    ) -> FeedbackRecord:
        # Keep the durable input update and generation bump in one service
        # critical section.  Otherwise an older queued job can observe the new
        # record before its own generation has been invalidated.
        with self._lock:
            record, notification = self._submit_correction_locked(
                feedback_id,
                corrected_translation=corrected_translation,
                note=note,
                keywords=keywords,
                ai_problem_summary=ai_problem_summary,
                on_complete=on_complete,
            )
        if notification is not None:
            self._safe_notify(on_complete, notification)
        return record

    def _submit_correction_locked(
        self,
        feedback_id: str,
        *,
        corrected_translation: str,
        note: str,
        keywords: list[str] | None,
        ai_problem_summary: str,
        on_complete: Callable[[dict], None] | None,
    ) -> tuple[FeedbackRecord, dict | None]:
        if self._stopping:
            raise RuntimeError("feedback learning service is shutting down")
        previous = self._store.get_feedback(feedback_id)
        record = self._store.submit_correction(
            feedback_id,
            corrected_translation=corrected_translation,
            note=note,
            keywords=keywords,
            ai_problem_summary=ai_problem_summary,
        )
        unchanged = (
            previous is not None
            and previous.consolidation_input_digest
            and previous.consolidation_input_digest == record.consolidation_input_digest
            and (
                previous.consolidation_status == "completed"
                or (
                    previous.consolidation_status in {"queued", "running"}
                    and (
                        feedback_id,
                        self._generations.get(feedback_id, -1),
                    ) in self._owned_jobs
                )
            )
        )
        if unchanged:
            return (
                record,
                {"feedback_id": feedback_id, "skipped": "unchanged"},
            )
        if record.consolidation_attempts >= self.MAX_ATTEMPTS_PER_INPUT:
            error = (
                "同一份纠错内容的后台归纳已连续失败两次；"
                "请修改译文、备注或关键词后再提交。"
            )
            record = self._store.update_feedback(
                feedback_id,
                consolidation_status="failed",
                consolidation_error=error,
            ) or record
            return (
                record,
                {"feedback_id": feedback_id, "error": error},
            )
        unavailable = self._configuration_error()
        if unavailable:
            record = self._store.update_feedback(
                feedback_id,
                consolidation_status="failed",
                consolidation_error=unavailable,
            ) or record
            return (
                record,
                {"feedback_id": feedback_id, "error": unavailable},
            )
        try:
            self._schedule(feedback_id, on_complete)
        except RuntimeError as exc:
            # _schedule persists the durable failed state before raising.  The
            # caller must see that same state, rather than the earlier queued
            # snapshot returned by submit_correction().
            record = self._store.get_feedback(feedback_id) or record
            return (
                record,
                {"feedback_id": feedback_id, "error": str(exc)},
            )
        return record, None

    def shutdown(
        self,
        wait: bool = False,
        timeout: float | None = None,
    ) -> bool:
        """Stop accepting work and report whether the worker has exited.

        ``wait=True`` without a timeout genuinely waits for an in-flight
        optimizer call.  UI shutdown deliberately uses ``wait=False`` so a
        remote request can never freeze application exit.
        """

        if timeout is not None and timeout < 0:
            raise ValueError("shutdown timeout must be non-negative")

        with self._lock:
            first_stop_request = not self._stopping
            self._stopping = True
        if first_stop_request and self._worker.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if wait:
            # Completion callbacks run on the worker.  A callback is allowed
            # to request shutdown, but a thread cannot join itself.
            if threading.current_thread() is self._worker:
                return False
            self._worker.join(timeout=timeout)
        return not self._worker.is_alive()

    def set_correction_enabled(self, feedback_id: str, enabled: bool) -> FeedbackRecord:
        """Persist a toggle, then invalidate in-flight work if it was revoked."""

        if not isinstance(enabled, bool):
            raise ValueError("feedback enabled state must be boolean")
        # Serialize the durable state change with generation checks.  A failed
        # disk write must not cancel the only valid in-flight consolidation.
        with self._lock:
            record = self._store.set_feedback_enabled(feedback_id, enabled)
            if not enabled:
                self._generations[feedback_id] = self._generations.get(feedback_id, 0) + 1
            return record

    def _schedule(
        self,
        feedback_id: str,
        on_complete: Callable[[dict], None] | None = None,
    ) -> None:
        with self._lock:
            if self._stopping:
                raise RuntimeError("feedback learning service is shutting down")
            generation = self._generations.get(feedback_id, 0) + 1
            self._generations[feedback_id] = generation
            self._owned_jobs.add((feedback_id, generation))
        job = _ConsolidationJob(feedback_id, generation, on_complete)
        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            try:
                self._store.update_feedback(
                    feedback_id,
                    consolidation_status="failed",
                    consolidation_error="后台归纳队列已满，请稍后重新提交。",
                )
            except Exception:
                # If even the failed-state write is unavailable, do not leave
                # a durable ``queued`` row without an in-process owner.  The
                # same serial worker consumes this emergency durable queue.
                with self._lock:
                    if self._stopping:
                        raise
                    self._recovery_jobs.append(job)
                get_logger().exception(
                    "Could not persist queue-full state; retained durable job | feedback=%s",
                    feedback_id[:8],
                )
                return
            with self._lock:
                self._owned_jobs.discard((feedback_id, generation))
            raise RuntimeError("feedback consolidation queue is full") from exc

    def _recover_pending(self) -> list[_ConsolidationJob]:
        """Prepare all durable recovery jobs before the worker starts.

        Recovery work is kept in a separate in-memory deque consumed by the
        same serial worker.  This avoids both overflowing the bounded live
        submission queue and starting an auxiliary feeder thread that could be
        orphaned if construction fails.
        """

        jobs: list[_ConsolidationJob] = []
        unavailable = self._configuration_error()
        if unavailable:
            for record in self._store.list_feedback():
                if (
                    record.status in {"accepted", "confirmed"}
                    and record.enabled
                    and record.consolidation_status in {"queued", "running"}
                ):
                    self._store.update_feedback(
                        record.id,
                        consolidation_status="failed",
                        consolidation_error=unavailable,
                    )
            return jobs
        for record in self._store.list_feedback():
            if (
                record.status in {"accepted", "confirmed"}
                and record.enabled
                and record.corrected_translation.strip()
                and record.consolidation_status in {"queued", "running"}
            ):
                if record.consolidation_attempts >= self.MAX_ATTEMPTS_PER_INPUT:
                    self._store.update_feedback(
                        record.id,
                        consolidation_status="failed",
                        consolidation_error="后台归纳已达到本次纠错内容的尝试上限。",
                    )
                    continue
                if record.consolidation_status == "running":
                    self._store.update_feedback(
                        record.id,
                        consolidation_status="queued",
                        consolidation_error="",
                    )
                generation = self._generations.get(record.id, 0) + 1
                self._generations[record.id] = generation
                jobs.append(_ConsolidationJob(record.id, generation))
        return jobs

    def _configuration_error(self) -> str:
        if not callable(getattr(self._optimizer, "consolidate", None)):
            return "当前优化器不支持长期规则归纳。"
        if isinstance(self._optimizer, FeedbackOptimizer):
            ai = self._settings.ai
            if not ai.base_url or not ai.api_key or not ai.thinking_model_name:
                return "Thinking Model 尚未配置，纠错案例已生效但未归纳长期规则。"
        return ""

    def _worker_loop(self) -> None:
        while True:
            job, from_live_queue = self._next_job()
            try:
                if job is None:
                    return
                with self._lock:
                    if self._stopping:
                        return
                if not self._is_current(job.feedback_id, job.generation):
                    get_debug_logger().debug(
                        "Skipped obsolete queued feedback job | feedback=%s generation=%d",
                        job.feedback_id[:8],
                        job.generation,
                    )
                    continue
                try:
                    payload = self._consolidate(job.feedback_id, job.generation)
                except Exception as exc:
                    # A programming or persistence error outside the normal
                    # consolidation boundary must not kill the only worker.
                    payload = {
                        "feedback_id": job.feedback_id,
                        "generation": job.generation,
                        "error": str(exc),
                    }
                    get_logger().exception(
                        "Unexpected feedback worker failure | feedback=%s",
                        job.feedback_id[:8],
                    )
                    # No digest/attempt ownership token is available here.
                    # An unconditional write could overwrite a newer submit;
                    # leave the durable running/queued row for startup recovery.
                if self._is_current(job.feedback_id, job.generation):
                    self._safe_notify(job.on_complete, payload)
            finally:
                if job is not None:
                    with self._lock:
                        self._owned_jobs.discard((job.feedback_id, job.generation))
                if from_live_queue:
                    self._queue.task_done()

    def _next_job(self) -> tuple[_ConsolidationJob | None, bool]:
        """Interleave durable recovery work with newly submitted corrections.

        Recovery remains guaranteed progress, but a large startup backlog can
        no longer force every new correction to wait behind many 120-second
        Pro calls.  The live queue is polled without blocking whenever the
        preceding job came from recovery; otherwise one recovery job runs.
        """

        if self._recovery_jobs:
            if self._last_job_from_recovery:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    job = self._recovery_jobs.popleft()
                    self._last_job_from_recovery = True
                    return job, False
                self._last_job_from_recovery = False
                return job, True
            job = self._recovery_jobs.popleft()
            self._last_job_from_recovery = True
            return job, False

        job = self._queue.get()
        self._last_job_from_recovery = False
        return job, True

    def _consolidate(
        self,
        feedback_id: str,
        generation: int,
    ) -> dict:
        payload: dict = {"feedback_id": feedback_id, "generation": generation}
        expected_input_digest = ""
        expected_attempt = 0
        pre_call_digests: dict[str, str] = {}
        try:
            with self._lock:
                if not self._is_current(feedback_id, generation):
                    return payload
                current = self._store.get_feedback(feedback_id)
                if (
                    current is None
                    or current.status not in {"accepted", "confirmed"}
                    or not current.enabled
                    or not current.corrected_translation.strip()
                ):
                    raise StaleConsolidationAborted(
                        "the correction is no longer eligible for consolidation"
                    )
                expected_input_digest = current.consolidation_input_digest
                attempts = current.consolidation_attempts + 1
                expected_attempt = attempts
                self._store.update_feedback(
                    feedback_id,
                    consolidation_status="running",
                    consolidation_error="",
                    consolidation_attempts=attempts,
                )
                # Freeze all possible evidence before Pro starts.  The
                # optimizer chooses related IDs during the remote call, so a
                # primary-only digest cannot protect secondary/history rows.
                pre_call_digests = self._store.feedback_input_digest_snapshot()
            proposal = self._optimizer.consolidate(self._settings, feedback_id)
            if not self._is_current(feedback_id, generation):
                get_debug_logger().debug(
                    "Feedback consolidation discarded as stale | id=%s generation=%d",
                    feedback_id[:8],
                    generation,
                )
                return payload
            proposal_ids = list(dict.fromkeys(proposal.source_feedback_ids))
            if not proposal_ids or proposal_ids[0] != feedback_id:
                raise StaleConsolidationAborted(
                    "the consolidation proposal no longer belongs to its primary feedback"
                )
            missing_snapshot_ids = [
                item for item in proposal_ids if item not in pre_call_digests
            ]
            if missing_snapshot_ids:
                raise StaleConsolidationAborted(
                    "one or more consolidation evidence rows were created after "
                    "the thinking model started"
                )
            memory = self._store.upsert_automatic_memory_rule(
                feedback_ids=proposal_ids,
                trigger=proposal.trigger,
                rule=proposal.rule,
                target_rule_id=proposal.target_rule_id,
                primary_feedback_id=feedback_id,
                expected_input_digest=expected_input_digest,
                expected_input_digests={
                    item: pre_call_digests[item] for item in proposal_ids
                },
            )
            payload["memory"] = memory
            get_logger().info(
                "Feedback memory consolidated | feedback=%s memory=%s evidence=%d locked=%s",
                feedback_id[:8],
                memory.id[:8],
                len(memory.source_feedback_ids or []),
                memory.user_locked,
            )
        except StaleConsolidationAborted as exc:
            payload["skipped"] = "stale"
            message = "纠错证据在后台归纳期间发生变化，本次长期规则未保存。"
            payload["message"] = message
            if expected_input_digest:
                try:
                    self._store.finalize_stale_consolidation(
                        feedback_id,
                        expected_input_digest=expected_input_digest,
                        expected_attempt=expected_attempt,
                        message=message,
                    )
                except Exception as persist_exc:
                    payload["persistence_error"] = str(persist_exc)
                    get_logger().exception(
                        "Could not persist stale feedback consolidation | feedback=%s",
                        feedback_id[:8],
                    )
            get_debug_logger().debug(
                "Feedback consolidation discarded after source changed | id=%s reason=%s",
                feedback_id[:8],
                exc,
            )
        except Exception as exc:
            payload["error"] = str(exc)
            if expected_input_digest and expected_attempt:
                try:
                    # Keep the generation check and conditional durable update
                    # in one service critical section.  A newer submit cannot
                    # slip between ownership validation and the failed write.
                    with self._lock:
                        if (
                            not self._stopping
                            and self._generations.get(feedback_id) == generation
                        ):
                            self._store.fail_consolidation_if_current(
                                feedback_id,
                                expected_input_digest=expected_input_digest,
                                expected_attempt=expected_attempt,
                                message=str(exc),
                            )
                except Exception as persist_exc:
                    payload["persistence_error"] = str(persist_exc)
                    get_logger().exception(
                        "Could not persist feedback consolidation failure | feedback=%s",
                        feedback_id[:8],
                    )
            get_logger().warning(
                "Feedback memory consolidation failed | feedback=%s error=%s",
                feedback_id[:8],
                exc,
            )
        return payload

    @staticmethod
    def _safe_notify(
        callback: Callable[[dict], None] | None,
        payload: dict,
    ) -> None:
        """Notify UI/client code without allowing it to break learning state."""

        if callback is None:
            return
        try:
            callback(dict(payload))
        except Exception:
            # Qt objects may have been deleted, and third-party callbacks can
            # raise arbitrary exceptions.  Both are outside worker ownership.
            get_logger().exception(
                "Feedback completion callback failed | feedback=%s",
                str(payload.get("feedback_id", ""))[:8],
            )

    def _is_current(self, feedback_id: str, generation: int) -> bool:
        with self._lock:
            return not self._stopping and self._generations.get(feedback_id) == generation


__all__ = ["FeedbackLearningService", "FeedbackRecord", "MemoryRule"]
