from __future__ import annotations

import tempfile
import threading
import time
import unittest
import queue
from collections import deque
from unittest.mock import patch

from app.feedback.learning import FeedbackLearningService
from app.feedback.optimizer import MemoryConsolidation
from app.feedback.store import FeedbackStore
from app.settings import AppSettings


class FakeConsolidator:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def consolidate(self, settings, feedback_id):
        self.calls.append(feedback_id)
        if self.error is not None:
            raise self.error
        return self.result


class FeedbackLearningServiceTests(unittest.TestCase):
    def test_recovery_jobs_are_interleaved_with_live_submissions(self) -> None:
        service = object.__new__(FeedbackLearningService)
        service._queue = queue.Queue()
        service._recovery_jobs = deque(["recovery-1", "recovery-2"])
        service._last_job_from_recovery = False
        service._queue.put("live-1")

        first = service._next_job()
        second = service._next_job()
        third = service._next_job()

        self.assertEqual(first, ("recovery-1", False))
        self.assertEqual(second, ("live-1", True))
        self.assertEqual(third, ("recovery-2", False))

    class _BlockingConsolidator:
        def __init__(self, result: MemoryConsolidation) -> None:
            self.result = result
            self.started = threading.Event()
            self.release = threading.Event()

        def consolidate(self, settings, feedback_id):
            self.started.set()
            self.release.wait(2.0)
            return self.result

    @staticmethod
    def _proposal_for(feedback_id: str) -> MemoryConsolidation:
        return MemoryConsolidation(
            trigger="连接超时",
            rule="必须保留连接失败和超时含义。",
            source_feedback_ids=(feedback_id,),
        )

    def test_submit_activates_correction_before_background_rule_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="连接超时了", translation_text="wrong",
            )
            completed = threading.Event()
            optimizer = FakeConsolidator(
                MemoryConsolidation(
                    trigger="连接超时",
                    rule="必须保留连接失败和超时含义。",
                    source_feedback_ids=(record.id,),
                )
            )
            service = FeedbackLearningService(store, AppSettings(), optimizer)  # type: ignore[arg-type]

            submitted = service.submit_correction(
                record.id,
                corrected_translation="correct",
                note="超时不能翻成延迟",
                keywords=["连接超时"],
                on_complete=lambda payload: completed.set(),
            )

            immediate = store.match_corrections(
                "接口连接超时了",
                source_language="中文",
                target_language="日本語",
            )
            self.assertEqual(submitted.status, "accepted")
            self.assertEqual([item.id for item in immediate], [record.id])
            self.assertTrue(completed.wait(2.0))
            rules = store.match_memory_rules(
                "接口连接超时",
                source_language="中文",
                target_language="日本語",
            )
            self.assertEqual(len(rules), 1)
            self.assertEqual(store.get_feedback(record.id).consolidation_status, "completed")

    def test_pro_failure_does_not_disable_submitted_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="请求失败", translation_text="wrong",
            )
            completed = threading.Event()
            service = FeedbackLearningService(
                store,
                AppSettings(),
                FakeConsolidator(error=RuntimeError("offline")),  # type: ignore[arg-type]
            )
            service.submit_correction(
                record.id,
                corrected_translation="correct",
                on_complete=lambda payload: completed.set(),
            )

            self.assertTrue(completed.wait(2.0))
            saved = store.get_feedback(record.id)
            self.assertEqual(saved.status, "accepted")
            self.assertEqual(saved.consolidation_status, "failed")
            self.assertIn("offline", saved.consolidation_error)
            self.assertEqual(
                [item.id for item in store.match_corrections(
                    "请求失败",
                    source_language="中文",
                    target_language="日本語",
                )],
                [record.id],
            )

    def test_callback_exception_does_not_kill_worker_or_block_next_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            first = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口一连接超时", translation_text="wrong-1",
            )
            second = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="接口二连接超时", translation_text="wrong-2",
            )

            class DynamicConsolidator:
                def consolidate(self, settings, feedback_id):
                    return FeedbackLearningServiceTests._proposal_for(feedback_id)

            service = FeedbackLearningService(
                store, AppSettings(), DynamicConsolidator()  # type: ignore[arg-type]
            )
            first_notified = threading.Event()
            second_notified = threading.Event()

            def broken_callback(payload):
                first_notified.set()
                raise ValueError("callback bug")

            service.submit_correction(
                first.id, corrected_translation="correct-1", on_complete=broken_callback,
            )
            self.assertTrue(first_notified.wait(2.0))
            service.submit_correction(
                second.id,
                corrected_translation="correct-2",
                on_complete=lambda payload: second_notified.set(),
            )

            self.assertTrue(second_notified.wait(2.0))
            self.assertTrue(service._worker.is_alive())
            self.assertEqual(store.get_feedback(second.id).consolidation_status, "completed")
            service.shutdown(wait=True)

    def test_immediate_unchanged_callback_exception_does_not_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong",
            )
            service = FeedbackLearningService(
                store,
                AppSettings(),
                FakeConsolidator(self._proposal_for(record.id)),  # type: ignore[arg-type]
            )
            completed = threading.Event()
            service.submit_correction(
                record.id,
                corrected_translation="correct",
                on_complete=lambda payload: completed.set(),
            )
            self.assertTrue(completed.wait(2.0))

            service.submit_correction(
                record.id,
                corrected_translation="correct",
                on_complete=lambda payload: (_ for _ in ()).throw(ValueError("bad callback")),
            )

            self.assertTrue(service._worker.is_alive())
            service.shutdown(wait=True)

    def test_failed_disable_does_not_cancel_valid_inflight_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong",
            )
            optimizer = self._BlockingConsolidator(self._proposal_for(record.id))
            service = FeedbackLearningService(store, AppSettings(), optimizer)  # type: ignore[arg-type]
            service.submit_correction(record.id, corrected_translation="correct")
            self.assertTrue(optimizer.started.wait(1.0))
            original_generation = service._generations[record.id]

            with patch.object(
                store,
                "set_feedback_enabled",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(OSError):
                    service.set_correction_enabled(record.id, False)

            self.assertEqual(service._generations[record.id], original_generation)
            optimizer.release.set()
            service._queue.join()
            saved = store.get_feedback(record.id)
            self.assertTrue(saved.enabled)
            self.assertEqual(saved.consolidation_status, "completed")
            self.assertEqual(len(store.list_memory_rules()), 1)
            service.shutdown(wait=True)

    def test_obsolete_queued_generation_never_calls_pro_or_spends_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            blocker = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="队列阻塞故障", translation_text="wrong-blocker",
            )
            target = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong-target",
            )

            class QueueBlockingConsolidator:
                def __init__(self) -> None:
                    self.calls: list[str] = []
                    self.started = threading.Event()
                    self.release = threading.Event()

                def consolidate(self, settings, feedback_id):
                    self.calls.append(feedback_id)
                    if feedback_id == blocker.id:
                        self.started.set()
                        self.release.wait(2.0)
                        trigger = "队列阻塞"
                    else:
                        trigger = "连接超时"
                    return MemoryConsolidation(
                        trigger=trigger,
                        rule="保留原文中的故障语义。",
                        source_feedback_ids=(feedback_id,),
                    )

            optimizer = QueueBlockingConsolidator()
            service = FeedbackLearningService(store, AppSettings(), optimizer)  # type: ignore[arg-type]
            service.submit_correction(blocker.id, corrected_translation="correct-blocker")
            self.assertTrue(optimizer.started.wait(1.0))
            obsolete_callbacks: list[dict] = []
            completed = threading.Event()
            service.submit_correction(
                target.id,
                corrected_translation="correct-v1",
                on_complete=obsolete_callbacks.append,
            )
            service.submit_correction(
                target.id,
                corrected_translation="correct-v2",
                note="new evidence",
                on_complete=lambda payload: completed.set(),
            )
            optimizer.release.set()

            self.assertTrue(completed.wait(2.0))
            self.assertEqual(optimizer.calls.count(target.id), 1)
            self.assertEqual(obsolete_callbacks, [])
            saved = store.get_feedback(target.id)
            self.assertEqual(saved.corrected_translation, "correct-v2")
            self.assertEqual(saved.consolidation_attempts, 1)
            self.assertEqual(saved.consolidation_status, "completed")
            service.shutdown(wait=True)

    def test_unchanged_completed_correction_does_not_call_pro_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="连接超时", translation_text="wrong",
            )
            optimizer = FakeConsolidator(
                MemoryConsolidation(
                    trigger="连接超时",
                    rule="必须保留连接超时含义。",
                    source_feedback_ids=(record.id,),
                )
            )
            service = FeedbackLearningService(store, AppSettings(), optimizer)  # type: ignore[arg-type]
            first_done = threading.Event()
            service.submit_correction(
                record.id,
                corrected_translation="correct",
                note="保留超时",
                keywords=["连接超时"],
                on_complete=lambda payload: first_done.set(),
            )
            self.assertTrue(first_done.wait(2.0))

            second_done = threading.Event()
            service.submit_correction(
                record.id,
                corrected_translation="correct",
                note="保留超时",
                keywords=["连接超时"],
                on_complete=lambda payload: second_done.set(),
            )

            self.assertTrue(second_done.wait(0.2))
            self.assertEqual(optimizer.calls, [record.id])
            service.shutdown(wait=True)

    def test_background_consolidation_is_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            first = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="请求一失败", translation_text="wrong",
            )
            second = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="请求二失败", translation_text="wrong",
            )

            class BlockingConsolidator:
                def __init__(self) -> None:
                    self.calls: list[str] = []
                    self.first_started = threading.Event()
                    self.release = threading.Event()

                def consolidate(self, settings, feedback_id):
                    self.calls.append(feedback_id)
                    if len(self.calls) == 1:
                        self.first_started.set()
                        self.release.wait(2.0)
                    return MemoryConsolidation(
                        trigger="请求",
                        rule="必须保留请求失败含义。",
                        source_feedback_ids=(feedback_id,),
                    )

            optimizer = BlockingConsolidator()
            service = FeedbackLearningService(store, AppSettings(), optimizer)  # type: ignore[arg-type]
            finished = threading.Event()
            completions: list[str] = []

            def done(payload):
                completions.append(payload["feedback_id"])
                if len(completions) == 2:
                    finished.set()

            service.submit_correction(first.id, corrected_translation="correct-1", on_complete=done)
            service.submit_correction(second.id, corrected_translation="correct-2", on_complete=done)
            self.assertTrue(optimizer.first_started.wait(1.0))
            self.assertEqual(optimizer.calls, [first.id])
            optimizer.release.set()
            self.assertTrue(finished.wait(2.0))
            self.assertEqual(optimizer.calls, [first.id, second.id])
            service.shutdown(wait=True)

    def test_same_failed_input_is_bounded_until_user_changes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="连接超时", translation_text="wrong",
            )
            optimizer = FakeConsolidator(error=RuntimeError("offline"))
            service = FeedbackLearningService(store, AppSettings(), optimizer)  # type: ignore[arg-type]

            for _ in range(3):
                completed = threading.Event()
                service.submit_correction(
                    record.id,
                    corrected_translation="correct",
                    on_complete=lambda payload, event=completed: event.set(),
                )
                self.assertTrue(completed.wait(2.0))

            self.assertEqual(optimizer.calls, [record.id, record.id])
            self.assertEqual(store.get_feedback(record.id).consolidation_attempts, 2)

            changed = threading.Event()
            service.submit_correction(
                record.id,
                corrected_translation="correct",
                note="new evidence",
                on_complete=lambda payload: changed.set(),
            )
            self.assertTrue(changed.wait(2.0))
            self.assertEqual(optimizer.calls, [record.id, record.id, record.id])
            service.shutdown(wait=True)

    def test_disabling_correction_while_pro_runs_discards_stale_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong",
            )
            optimizer = self._BlockingConsolidator(MemoryConsolidation(
                trigger="连接超时",
                rule="必须保留连接失败和超时含义。",
                source_feedback_ids=(record.id,),
            ))
            service = FeedbackLearningService(store, AppSettings(), optimizer)  # type: ignore[arg-type]
            service.submit_correction(record.id, corrected_translation="correct")
            self.assertTrue(optimizer.started.wait(1.0))

            service.set_correction_enabled(record.id, False)
            optimizer.release.set()
            service._queue.join()

            saved = store.get_feedback(record.id)
            self.assertFalse(saved.enabled)
            self.assertNotEqual(saved.consolidation_status, "completed")
            self.assertEqual(store.list_memory_rules(enabled_only=False), [])
            service.shutdown(wait=True)

    def test_deleting_primary_while_pro_runs_cannot_reassign_rule_to_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            primary = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong-1",
            )
            related = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="数据库连接超时", translation_text="wrong-2",
            )
            store.submit_correction(related.id, corrected_translation="correct-2")
            store.update_feedback(related.id, consolidation_status="completed")
            optimizer = self._BlockingConsolidator(MemoryConsolidation(
                trigger="连接超时",
                rule="必须保留连接失败和超时含义。",
                source_feedback_ids=(primary.id, related.id),
            ))
            service = FeedbackLearningService(store, AppSettings(), optimizer)  # type: ignore[arg-type]
            service.submit_correction(primary.id, corrected_translation="correct-1")
            self.assertTrue(optimizer.started.wait(1.0))

            self.assertTrue(store.delete_feedback(primary.id))
            optimizer.release.set()
            service._queue.join()

            self.assertIsNone(store.get_feedback(primary.id))
            self.assertEqual(store.get_feedback(related.id).memory_rule_id, "")
            self.assertEqual(store.list_memory_rules(enabled_only=False), [])
            service.shutdown(wait=True)

    def test_changing_secondary_evidence_finishes_primary_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            primary = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong-1",
            )
            related = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="数据库连接超时", translation_text="wrong-2",
            )
            store.submit_correction(related.id, corrected_translation="correct-2")
            store.update_feedback(related.id, consolidation_status="completed")
            optimizer = self._BlockingConsolidator(MemoryConsolidation(
                trigger="连接超时",
                rule="必须保留连接失败和超时含义。",
                source_feedback_ids=(primary.id, related.id),
            ))
            callbacks: list[dict] = []
            service = FeedbackLearningService(store, AppSettings(), optimizer)  # type: ignore[arg-type]
            service.submit_correction(
                primary.id,
                corrected_translation="correct-1",
                on_complete=callbacks.append,
            )
            self.assertTrue(optimizer.started.wait(1.0))

            service.set_correction_enabled(related.id, False)
            optimizer.release.set()
            service._queue.join()

            saved = store.get_feedback(primary.id)
            self.assertEqual(saved.consolidation_status, "failed")
            self.assertIn("证据", saved.consolidation_error)
            self.assertEqual(saved.consolidation_attempts, 0)
            self.assertEqual(callbacks[0].get("skipped"), "stale")
            self.assertEqual(store.list_memory_rules(enabled_only=False), [])
            service.shutdown(wait=True)

    def test_editing_secondary_evidence_while_pro_runs_discards_stale_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            primary = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong-1",
            )
            secondary = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="数据库连接超时", translation_text="wrong-2",
            )
            store.submit_correction(secondary.id, corrected_translation="old evidence")
            store.update_feedback(secondary.id, consolidation_status="completed")
            optimizer = self._BlockingConsolidator(MemoryConsolidation(
                trigger="连接超时",
                rule="必须保留连接失败和超时含义。",
                source_feedback_ids=(primary.id, secondary.id),
            ))
            callbacks: list[dict] = []
            service = FeedbackLearningService(
                store, AppSettings(), optimizer  # type: ignore[arg-type]
            )
            service.submit_correction(
                primary.id,
                corrected_translation="correct-1",
                on_complete=callbacks.append,
            )
            self.assertTrue(optimizer.started.wait(1.0))

            store.submit_correction(
                secondary.id,
                corrected_translation="new evidence",
            )
            optimizer.release.set()
            service._queue.join()

            saved = store.get_feedback(primary.id)
            self.assertEqual(saved.consolidation_status, "failed")
            self.assertEqual(saved.consolidation_attempts, 0)
            self.assertEqual(callbacks[0].get("skipped"), "stale")
            self.assertEqual(store.list_memory_rules(enabled_only=False), [])
            service.shutdown(wait=True)

    def test_recovery_marks_unschedulable_queued_job_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="source", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")

            service = FeedbackLearningService(store, AppSettings())
            recovered = store.get_feedback(record.id)

            self.assertEqual(recovered.consolidation_status, "failed")
            self.assertIn("Thinking Model", recovered.consolidation_error)
            service.shutdown(wait=True)

    def test_old_failure_cannot_overwrite_a_newer_submit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="connection timeout", translation_text="wrong",
            )

            class FailThenSucceed:
                def __init__(self) -> None:
                    self.calls = 0

                def consolidate(self, settings, feedback_id):
                    self.calls += 1
                    if self.calls == 1:
                        raise RuntimeError("old run failed")
                    return FeedbackLearningServiceTests._proposal_for(feedback_id)

            optimizer = FailThenSucceed()
            service = FeedbackLearningService(
                store, AppSettings(), optimizer  # type: ignore[arg-type]
            )
            failure_write_entered = threading.Event()
            release_failure_write = threading.Event()
            original_fail = store.fail_consolidation_if_current

            def blocked_fail(*args, **kwargs):
                failure_write_entered.set()
                release_failure_write.wait(2.0)
                return original_fail(*args, **kwargs)

            second_done = threading.Event()
            submit_errors: list[Exception] = []
            with patch.object(store, "fail_consolidation_if_current", side_effect=blocked_fail):
                service.submit_correction(record.id, corrected_translation="correct-v1")
                self.assertTrue(failure_write_entered.wait(1.0))

                def submit_new() -> None:
                    try:
                        service.submit_correction(
                            record.id,
                            corrected_translation="correct-v2",
                            note="new generation",
                            on_complete=lambda payload: second_done.set(),
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic capture
                        submit_errors.append(exc)

                submit_thread = threading.Thread(target=submit_new)
                submit_thread.start()
                time.sleep(0.03)
                self.assertTrue(submit_thread.is_alive())
                release_failure_write.set()
                submit_thread.join(1.0)
                self.assertFalse(submit_thread.is_alive())
                self.assertEqual(submit_errors, [])
                self.assertTrue(second_done.wait(2.0))

            saved = store.get_feedback(record.id)
            self.assertEqual(saved.corrected_translation, "correct-v2")
            self.assertEqual(saved.note, "new generation")
            self.assertEqual(saved.consolidation_status, "completed")
            self.assertNotIn("old run failed", saved.consolidation_error)
            service.shutdown(wait=True)

    def test_unowned_durable_queued_row_is_scheduled_on_resubmit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            optimizer = FakeConsolidator()
            service = FeedbackLearningService(
                store, AppSettings(), optimizer  # type: ignore[arg-type]
            )
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="connection timeout", translation_text="wrong",
            )
            record = store.submit_correction(record.id, corrected_translation="correct")
            optimizer.result = self._proposal_for(record.id)
            completed = threading.Event()

            service.submit_correction(
                record.id,
                corrected_translation="correct",
                on_complete=lambda payload: completed.set(),
            )

            self.assertTrue(completed.wait(2.0))
            self.assertEqual(optimizer.calls, [record.id])
            self.assertEqual(store.get_feedback(record.id).consolidation_status, "completed")
            service.shutdown(wait=True)

    def test_queue_full_and_failed_status_write_keeps_durable_job_owned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            FeedbackLearningService, "QUEUE_CAPACITY", 1
        ):
            store = FeedbackStore(tmp)
            records = [
                store.add_feedback(
                    group_id=index,
                    source_language="src",
                    target_language="tgt",
                    ocr_text=f"接口{index}连接超时",
                    translation_text="wrong",
                )
                for index in range(3)
            ]

            class BlockFirst:
                def __init__(self) -> None:
                    self.started = threading.Event()
                    self.release = threading.Event()

                def consolidate(self, settings, feedback_id):
                    if feedback_id == records[0].id:
                        self.started.set()
                        self.release.wait(2.0)
                    return FeedbackLearningServiceTests._proposal_for(feedback_id)

            optimizer = BlockFirst()
            service = FeedbackLearningService(
                store, AppSettings(), optimizer  # type: ignore[arg-type]
            )
            service.submit_correction(records[0].id, corrected_translation="correct-0")
            self.assertTrue(optimizer.started.wait(1.0))
            service.submit_correction(records[1].id, corrected_translation="correct-1")

            original_update = store.update_feedback

            def fail_only_queue_full_state(feedback_id, **kwargs):
                if kwargs.get("consolidation_status") == "failed":
                    raise OSError("disk temporarily unavailable")
                return original_update(feedback_id, **kwargs)

            with patch.object(
                store,
                "update_feedback",
                side_effect=fail_only_queue_full_state,
            ):
                third = service.submit_correction(
                    records[2].id,
                    corrected_translation="correct-2",
                )
            self.assertEqual(third.consolidation_status, "queued")
            self.assertTrue(any(item[0] == records[2].id for item in service._owned_jobs))

            optimizer.release.set()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if all(
                    store.get_feedback(item.id).consolidation_status == "completed"
                    for item in records
                ):
                    break
                time.sleep(0.01)
            self.assertTrue(all(
                store.get_feedback(item.id).consolidation_status == "completed"
                for item in records
            ))
            self.assertEqual(service._owned_jobs, set())
            service.shutdown(wait=True)

    def test_immediate_callback_runs_after_service_lock_is_released(self) -> None:
        """A callback may synchronously wait for another service caller."""

        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1,
                source_language="src",
                target_language="tgt",
                ocr_text="source",
                translation_text="wrong",
            )
            # An object without consolidate() makes submission finish through
            # the immediate configuration-error notification path.
            service = FeedbackLearningService(
                store, AppSettings(), object()  # type: ignore[arg-type]
            )
            nested_finished = threading.Event()
            callback_observed: list[bool] = []

            def callback(payload) -> None:
                thread = threading.Thread(
                    target=lambda: (
                        service.set_correction_enabled(record.id, True),
                        nested_finished.set(),
                    ),
                    daemon=True,
                )
                thread.start()
                callback_observed.append(nested_finished.wait(0.5))
                thread.join(timeout=1.0)

            service.submit_correction(
                record.id,
                corrected_translation="correct",
                on_complete=callback,
            )

            self.assertEqual(callback_observed, [True])
            self.assertTrue(nested_finished.is_set())
            service.shutdown(wait=True)

    def test_queue_full_returns_the_durable_failed_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            FeedbackLearningService, "QUEUE_CAPACITY", 1
        ):
            store = FeedbackStore(tmp)
            records = [
                store.add_feedback(
                    group_id=index,
                    source_language="src",
                    target_language="tgt",
                    ocr_text=f"queue issue {index}",
                    translation_text="wrong",
                )
                for index in range(3)
            ]

            class BlockingConsolidator:
                def __init__(self) -> None:
                    self.started = threading.Event()
                    self.release = threading.Event()

                def consolidate(self, settings, feedback_id):
                    self.started.set()
                    self.release.wait(2.0)
                    source = store.get_feedback(feedback_id).ocr_text
                    return MemoryConsolidation(
                        trigger=source,
                        rule="保留原文中的故障含义。",
                        source_feedback_ids=(feedback_id,),
                    )

            optimizer = BlockingConsolidator()
            service = FeedbackLearningService(
                store, AppSettings(), optimizer  # type: ignore[arg-type]
            )
            service.submit_correction(records[0].id, corrected_translation="correct-0")
            self.assertTrue(optimizer.started.wait(1.0))
            service.submit_correction(records[1].id, corrected_translation="correct-1")
            callbacks: list[dict] = []

            returned = service.submit_correction(
                records[2].id,
                corrected_translation="correct-2",
                on_complete=callbacks.append,
            )
            durable = store.get_feedback(records[2].id)

            self.assertEqual(returned, durable)
            self.assertEqual(returned.consolidation_status, "failed")
            self.assertIn("queue is full", callbacks[0]["error"])
            optimizer.release.set()
            service._queue.join()
            service.shutdown(wait=True)

    def test_recovery_drains_all_jobs_even_when_live_queue_is_smaller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            FeedbackLearningService, "QUEUE_CAPACITY", 1
        ):
            store = FeedbackStore(tmp)
            records = []
            for index in range(5):
                record = store.add_feedback(
                    group_id=index,
                    source_language="src",
                    target_language="tgt",
                    ocr_text=f"恢复任务{index}",
                    translation_text="wrong",
                )
                store.submit_correction(record.id, corrected_translation=f"correct-{index}")
                records.append(record)

            class RecoveryConsolidator:
                def __init__(self) -> None:
                    self.calls: list[str] = []

                def consolidate(self, settings, feedback_id):
                    self.calls.append(feedback_id)
                    source = store.get_feedback(feedback_id).ocr_text
                    return MemoryConsolidation(
                        trigger=source,
                        rule="保留原文中的故障含义。",
                        source_feedback_ids=(feedback_id,),
                    )

            optimizer = RecoveryConsolidator()
            service = FeedbackLearningService(
                store, AppSettings(), optimizer  # type: ignore[arg-type]
            )
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                statuses = [
                    store.get_feedback(record.id).consolidation_status
                    for record in records
                ]
                if all(status not in {"queued", "running"} for status in statuses):
                    break
                time.sleep(0.01)

            self.assertCountEqual(optimizer.calls, [record.id for record in records])
            self.assertTrue(all(status == "completed" for status in statuses))
            service.shutdown(wait=True)

    def test_shutdown_timeout_truthfully_reports_blocked_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1,
                source_language="src",
                target_language="tgt",
                ocr_text="blocked job",
                translation_text="wrong",
            )
            optimizer = self._BlockingConsolidator(self._proposal_for(record.id))
            service = FeedbackLearningService(
                store, AppSettings(), optimizer  # type: ignore[arg-type]
            )
            service.submit_correction(record.id, corrected_translation="correct")
            self.assertTrue(optimizer.started.wait(1.0))

            started = time.monotonic()
            stopped = service.shutdown(wait=True, timeout=0.05)
            elapsed = time.monotonic() - started

            self.assertFalse(stopped)
            self.assertTrue(service._worker.is_alive())
            self.assertLess(elapsed, 0.5)
            release = threading.Timer(0.05, optimizer.release.set)
            release.daemon = True
            release.start()
            wait_started = time.monotonic()
            self.assertTrue(service.shutdown(wait=True))
            self.assertGreaterEqual(time.monotonic() - wait_started, 0.04)

    def test_recovery_failure_happens_before_worker_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            FeedbackLearningService,
            "_recover_pending",
            side_effect=OSError("broken store"),
        ), patch.object(threading.Thread, "start") as start:
            with self.assertRaises(OSError):
                FeedbackLearningService(
                    FeedbackStore(tmp),
                    AppSettings(),
                    FakeConsolidator(),  # type: ignore[arg-type]
                )
            start.assert_not_called()

    def test_submission_after_shutdown_does_not_leave_queued_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1,
                source_language="src",
                target_language="tgt",
                ocr_text="source",
                translation_text="wrong",
            )
            service = FeedbackLearningService(
                store,
                AppSettings(),
                FakeConsolidator(),  # type: ignore[arg-type]
            )
            service.shutdown(wait=True)

            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                service.submit_correction(
                    record.id,
                    corrected_translation="correct",
                )

            durable = store.get_feedback(record.id)
            self.assertEqual(durable.status, "pending")
            self.assertEqual(
                durable.consolidation_status,
                record.consolidation_status,
            )

    def test_worker_callback_can_request_waiting_shutdown_without_self_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1,
                source_language="src",
                target_language="tgt",
                ocr_text="callback shutdown",
                translation_text="wrong",
            )

            class Consolidator:
                def consolidate(self, settings, feedback_id):
                    return MemoryConsolidation(
                        trigger="callback shutdown",
                        rule="保留原文中的故障含义。",
                        source_feedback_ids=(feedback_id,),
                    )

            service = FeedbackLearningService(
                store,
                AppSettings(),
                Consolidator(),  # type: ignore[arg-type]
            )
            callback_result: list[bool] = []
            callback_done = threading.Event()

            def callback(payload) -> None:
                callback_result.append(service.shutdown(wait=True))
                callback_done.set()

            service.submit_correction(
                record.id,
                corrected_translation="correct",
                on_complete=callback,
            )

            self.assertTrue(callback_done.wait(2.0))
            self.assertEqual(callback_result, [False])
            self.assertTrue(service.shutdown(wait=True, timeout=1.0))


if __name__ == "__main__":
    unittest.main()
