"""Tests for local feedback and translation-memory storage."""

from __future__ import annotations

import tempfile
import threading
import unittest
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from app.feedback.store import (
    CorrectionMatch,
    FeedbackRecord,
    FeedbackStorageUnavailable,
    FeedbackStore,
    MemoryRule,
)


class FeedbackStoreTests(unittest.TestCase):
    def _upsert_automatic(
        self,
        store: FeedbackStore,
        **kwargs: object,
    ) -> MemoryRule:
        """Call the production CAS API with a snapshot frozen before the write."""
        feedback_ids = list(dict.fromkeys(kwargs.get("feedback_ids", [])))
        snapshot = store.feedback_input_digest_snapshot()
        kwargs.setdefault(
            "expected_input_digests",
            {feedback_id: snapshot[feedback_id] for feedback_id in feedback_ids},
        )
        return store.upsert_automatic_memory_rule(**kwargs)

    def test_store_instances_for_same_root_serialize_read_modify_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = FeedbackStore(tmp)
            second = FeedbackStore(tmp)
            self.assertIs(first._lock, second._lock)

            def add_many(store: FeedbackStore, prefix: str) -> None:
                for index in range(10):
                    store.add_feedback(
                        group_id=index,
                        source_language="src",
                        target_language="tgt",
                        ocr_text=f"{prefix}-{index}",
                        translation_text="wrong",
                    )

            threads = (
                threading.Thread(target=add_many, args=(first, "a")),
                threading.Thread(target=add_many, args=(second, "b")),
            )
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(2.0)
                self.assertFalse(thread.is_alive())
            self.assertEqual(len(FeedbackStore(tmp).list_feedback()), 20)

    def test_revision_guard_serializes_external_feedback_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            revision = store.knowledge_revision
            started = threading.Event()
            finished = threading.Event()

            def mutate() -> None:
                started.set()
                store.add_feedback(
                    group_id=1,
                    source_language="src",
                    target_language="tgt",
                    ocr_text="guarded",
                    translation_text="wrong",
                )
                finished.set()

            with store.guard_knowledge_revision(revision) as current:
                self.assertTrue(current)
                writer = threading.Thread(target=mutate)
                writer.start()
                self.assertTrue(started.wait(timeout=1.0))
                self.assertFalse(finished.wait(timeout=0.1))

            writer.join(timeout=2.0)
            self.assertFalse(writer.is_alive())
            self.assertTrue(finished.is_set())
            self.assertGreater(store.knowledge_revision, revision)

    def test_knowledge_revision_advances_for_durable_feedback_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)

            record = store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="反馈句子",
                translation_text="wrong",
            )
            revision = store.knowledge_revision

            store.update_feedback(record.id, note="updated note")
            self.assertGreater(store.knowledge_revision, revision)
            revision = store.knowledge_revision

            store.submit_correction(record.id, corrected_translation="correct")
            self.assertGreater(store.knowledge_revision, revision)
            revision = store.knowledge_revision

            store.set_feedback_enabled(record.id, False)
            self.assertGreater(store.knowledge_revision, revision)
            revision = store.knowledge_revision

            store.set_feedback_enabled(record.id, True)
            self.assertGreater(store.knowledge_revision, revision)
            revision = store.knowledge_revision

            store.update_feedback(record.id, status="dismissed")
            self.assertGreater(store.knowledge_revision, revision)

            second = store.add_feedback(
                group_id=2,
                source_language="中文",
                target_language="日本語",
                ocr_text="需要规则",
                translation_text="wrong",
            )
            revision = store.knowledge_revision
            memory = store.approve_feedback(
                second.id,
                trigger="规则",
                rule="保留领域含义。",
                preferred_translation="correct",
            )
            self.assertGreater(store.knowledge_revision, revision)
            revision = store.knowledge_revision

            store.update_memory_rule(memory.id, rule_text="更新后的领域规则。")
            self.assertGreater(store.knowledge_revision, revision)

    def test_failed_feedback_write_does_not_advance_knowledge_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="规则句子",
                translation_text="wrong",
            )
            memory = store.approve_feedback(
                record.id,
                trigger="规则",
                rule="旧规则。",
                preferred_translation="correct",
            )
            revision = store.knowledge_revision

            with patch.object(store, "_write_json_list", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    store.update_memory_rule(memory.id, rule_text="新规则。")

            self.assertEqual(store.knowledge_revision, revision)
            self.assertEqual(store.get_memory_rule(memory.id).rule, "旧规则。")

    def test_duplicate_feedback_ids_are_quarantined_as_one_conflict_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pending-feedback.json"
            conflicted = FeedbackRecord.create(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="conflict", translation_text="wrong-a",
            )
            duplicate = asdict(conflicted)
            duplicate["translation_text"] = "wrong-b"
            unique = FeedbackRecord.create(
                group_id=2, source_language="src", target_language="tgt",
                ocr_text="unique", translation_text="wrong-c",
            )
            rows = [asdict(conflicted), duplicate, asdict(unique)]
            path.write_text(json.dumps(rows), encoding="utf-8")
            store = FeedbackStore(tmp)

            self.assertEqual([item.id for item in store.list_feedback()], [unique.id])
            store.add_feedback(
                group_id=3, source_language="src", target_language="tgt",
                ocr_text="new", translation_text="wrong-d",
            )
            quarantined = json.loads(
                (Path(tmp) / "pending-feedback.invalid-rows.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["translation_text"] for item in quarantined],
                ["wrong-a", "wrong-b"],
            )
            self.assertNotIn(conflicted.id, {
                item.id for item in FeedbackStore(tmp).list_feedback()
            })

    def test_duplicate_memory_ids_never_enter_runtime_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory-rules.json"
            conflicted = MemoryRule.create(
                source_language="src", target_language="tgt",
                trigger="timeout", rule="rule-a", origin="legacy",
            )
            duplicate = asdict(conflicted)
            duplicate["rule"] = "rule-b"
            unique = MemoryRule.create(
                source_language="src", target_language="tgt",
                trigger="connection", rule="rule-c", origin="legacy",
            )
            path.write_text(
                json.dumps([asdict(conflicted), duplicate, asdict(unique)]),
                encoding="utf-8",
            )
            store = FeedbackStore(tmp)

            self.assertEqual([item.id for item in store.list_memory_rules()], [unique.id])
            self.assertEqual(store.match_memory_rules(
                "timeout", source_language="src", target_language="tgt",
            ), [])
            store.update_memory_rule(unique.id, rule_text="rule-c-updated")
            quarantined = json.loads(
                (Path(tmp) / "memory-rules.invalid-rows.json").read_text(encoding="utf-8")
            )
            self.assertEqual({item["rule"] for item in quarantined}, {"rule-a", "rule-b"})

    def test_malformed_row_is_preserved_when_valid_feedback_is_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pending-feedback.json"
            malformed = {"id": "broken", "group_id": "not-an-int"}
            path.write_text(json.dumps([malformed]), encoding="utf-8")
            store = FeedbackStore(tmp)

            store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="句子", translation_text="wrong",
            )

            rows = json.loads(path.read_text(encoding="utf-8"))
            quarantine = Path(tmp) / "pending-feedback.invalid-rows.json"
            quarantined = json.loads(quarantine.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(quarantined, [malformed])

    def test_corrupt_feedback_file_is_backed_up_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pending-feedback.json"
            path.write_text("{broken json", encoding="utf-8")
            store = FeedbackStore(tmp)

            store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="句子", translation_text="wrong",
            )

            backups = list(Path(tmp).glob("pending-feedback.json.corrupt-*.backup"))
            self.assertEqual(len(backups), 1)
            backup = backups[0]
            self.assertEqual(backup.read_text(encoding="utf-8"), "{broken json")
            self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))), 1)

    def test_accept_translation_marks_accepted_without_creating_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="一次性句子", translation_text="wrong",
            )
            accepted = store.accept_translation(record.id, "correct")

            self.assertEqual(accepted.status, "accepted")
            self.assertEqual(accepted.corrected_translation, "correct")
            self.assertEqual(accepted.memory_rule_id, "")
            self.assertEqual(store.list_memory_rules(), [])
            self.assertEqual(store.list_feedback(status="pending"), [])
            self.assertEqual(store.get_feedback(record.id).status, "accepted")

    def test_accept_translation_requires_nonempty_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="句子", translation_text="wrong",
            )
            with self.assertRaises(ValueError):
                store.accept_translation(record.id, "  ")
            self.assertEqual(store.get_feedback(record.id).status, "pending")

    def test_confirmed_feedback_cannot_be_changed_to_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="高考开始", translation_text="wrong",
            )
            memory = store.approve_feedback(
                record.id,
                trigger="高考",
                rule="按大学入学考试语境翻译。",
                preferred_translation="confirmed translation",
            )

            with self.assertRaises(ValueError):
                store.accept_translation(record.id, "replacement")

            unchanged = store.get_feedback(record.id)
            self.assertEqual(unchanged.status, "confirmed")
            self.assertEqual(unchanged.memory_rule_id, memory.id)
            self.assertEqual(unchanged.corrected_translation, "confirmed translation")
            memories = store.list_memory_rules()
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].id, memory.id)

    def test_dismissed_and_accepted_feedback_cannot_be_accepted_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            dismissed = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="dismissed", translation_text="wrong",
            )
            store.update_feedback(dismissed.id, status="dismissed")
            with self.assertRaises(ValueError):
                store.accept_translation(dismissed.id, "replacement")
            self.assertEqual(store.get_feedback(dismissed.id).status, "dismissed")

            accepted = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="accepted", translation_text="wrong",
            )
            store.accept_translation(accepted.id, "first")
            with self.assertRaises(ValueError):
                store.accept_translation(accepted.id, "second")
            unchanged = store.get_feedback(accepted.id)
            self.assertEqual(unchanged.status, "accepted")
            self.assertEqual(unchanged.corrected_translation, "first")
    """Verify pending feedback can become reusable local memory."""

    def test_feedback_can_be_approved_and_matched_after_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="高考马上开始",
                translation_text="高校の試験がまもなく始まる",
                note="不是高校考试，是大学入学考试语境。",
            )

            store.update_feedback(record.id, corrected_translation="大学入学共通テストがまもなく始まる")
            memory = store.approve_feedback(
                record.id,
                trigger="高考",
                rule="出现“高考”时，按中国大学入学考试语境翻译，不要误作高校内部考试。",
            )

            reloaded = FeedbackStore(tmp)
            confirmed = reloaded.get_feedback(record.id)
            matches = reloaded.match_memory_rules(
                "今年高考报名人数很多",
                source_language="中文",
                target_language="日本語",
            )

            self.assertIsNotNone(confirmed)
            assert confirmed is not None
            self.assertEqual(confirmed.status, "confirmed")
            self.assertEqual(confirmed.memory_rule_id, memory.id)
            self.assertEqual(len(matches), 1)
            self.assertIn("大学入学考试", matches[0].rule)
            self.assertIn("大学入学共通テスト", matches[0].as_prompt_hint())

    def test_approve_feedback_requires_explicit_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="明天跑完所有的单元测试用例。",
                translation_text="あした、すべてのテストを実行し終える。",
            )

            with self.assertRaises(ValueError):
                store.approve_feedback(
                    record.id,
                    trigger="",
                    rule="出现“单元测试用例”时，按软件测试语境翻译。",
                )

    def test_memory_rule_matches_any_joined_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="明天跑完所有的单元测试用例。",
                translation_text="あした、すべてのテストを実行し終える。",
            )
            store.approve_feedback(
                record.id,
                trigger=FeedbackStore.join_triggers(["跑完", "单元测试用例", "测试用例"]),
                rule="这些词出现时，按软件测试语境翻译。",
            )

            matches = store.match_memory_rules(
                "测试用例还没写完",
                source_language="中文",
                target_language="日本語",
            )

            self.assertEqual(len(matches), 1)
            self.assertIn("跑完 / 单元测试用例 / 测试用例", matches[0].as_prompt_hint())


    def test_reapproving_feedback_updates_existing_memory_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="把测试全部跑完",
                translation_text="wrong",
            )
            first = store.approve_feedback(
                record.id,
                trigger="跑完",
                rule="按执行测试的语境翻译",
            )
            second = store.approve_feedback(
                record.id,
                trigger="测试跑完",
                rule="优先表达执行完成",
            )

            self.assertEqual(first.id, second.id)
            self.assertEqual(len(store.list_memory_rules()), 1)
            self.assertEqual(store.list_memory_rules()[0].trigger, "测试跑完")

    def test_more_specific_memory_rule_is_returned_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            broad = store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="测试",
                translation_text="wrong",
            )
            specific = store.add_feedback(
                group_id=1,
                source_language="中文",
                target_language="日本語",
                ocr_text="回归测试跑完",
                translation_text="wrong",
            )
            store.approve_feedback(broad.id, trigger="测试", rule="broad")
            store.approve_feedback(specific.id, trigger="回归测试", rule="specific")

            matches = store.match_memory_rules(
                "今天把回归测试跑完",
                source_language="中文",
                target_language="日本語",
            )

            self.assertEqual([item.rule for item in matches[:2]], ["specific", "broad"])

    def test_submitted_correction_is_immediately_retrievable_without_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="邻居闹到很晚，小李被吵到了", translation_text="wrong",
                matched_memory_rule_ids=["rule-used"],
                matched_correction_ids=["correction-used"],
            )
            submitted = store.submit_correction(
                record.id,
                corrected_translation="correct",
                note="保留被动受害关系",
                keywords=["被吵到"],
            )

            matches = store.match_corrections(
                "隔壁吵到很晚，他也被吵到了",
                source_language="中文",
                target_language="日本語",
            )

            self.assertEqual([item.id for item in matches], [record.id])
            self.assertEqual(submitted.consolidation_status, "queued")
            self.assertEqual(submitted.matched_memory_rule_ids, ["rule-used"])
            self.assertIn("用户确认译文：correct", store.correction_prompt_hint(submitted))

    def test_automatic_rule_is_active_and_user_edit_prevents_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="连接超时", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            automatic = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id],
                trigger="连接超时",
                rule="保留连接失败含义。",
            )
            locked = store.update_memory_rule(
                automatic.id,
                rule_text="必须明确表达连接超时，不得写成普通延迟。",
            )
            repeated = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id],
                trigger="连接超时",
                rule="Pro 的新建议不应覆盖。",
            )

            self.assertTrue(locked.user_locked)
            self.assertEqual(repeated.id, locked.id)
            self.assertEqual(repeated.rule, "必须明确表达连接超时，不得写成普通延迟。")
            self.assertEqual(store.get_feedback(record.id).memory_rule_id, automatic.id)

    def test_toggling_automatic_rule_preserves_automatic_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="连接超时", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            automatic = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id], trigger="连接超时", rule="保留超时含义。",
            )

            disabled = store.update_memory_rule(automatic.id, enabled=False)
            self.assertFalse(disabled.user_locked)
            self.assertFalse(disabled.enabled)
            self.assertFalse(disabled.user_enabled_override)

            refreshed = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id], trigger="连接超时", rule="更新后的自动规则。",
            )
            self.assertFalse(refreshed.enabled)
            self.assertFalse(refreshed.user_locked)

            enabled = store.update_memory_rule(automatic.id, enabled=True)
            self.assertTrue(enabled.enabled)
            self.assertTrue(enabled.user_enabled_override)
            self.assertFalse(enabled.user_locked)
            self.assertTrue(store.memory_rule_is_active(
                enabled, source_language="中文", target_language="日本語",
            ))

    def test_old_memory_rule_json_loads_without_user_toggle_override(self) -> None:
        rule = MemoryRule.from_dict({
            "origin": "automatic",
            "enabled": False,
            "user_locked": False,
        })
        self.assertIsNone(rule.user_enabled_override)

    def test_correction_retrieval_rejects_conflicting_completion_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="任务已经完成", translation_text="wrong",
            )
            store.submit_correction(
                record.id,
                corrected_translation="task completed",
                keywords=["完成"],
            )

            matches = store.rank_corrections(
                "任务即将完成",
                source_language="中文",
                target_language="日本語",
                minimum_score=0.0,
            )

            self.assertEqual(matches, [])

    def test_correction_retrieval_rejects_bare_completed_vs_future(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="任务完成", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            self.assertEqual(store.rank_corrections(
                "任务即将完成",
                source_language="中文",
                target_language="日本語",
                minimum_score=0.0,
            ), [])

    def test_correction_retrieval_treats_has_been_running_as_ongoing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="English", target_language="日本語",
                ocr_text="The worker has been running", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            self.assertEqual(store.rank_corrections(
                "The worker completed running",
                source_language="English",
                target_language="日本語",
                minimum_score=0.0,
            ), [])
            self.assertEqual(store.rank_corrections(
                "The worker runs",
                source_language="English",
                target_language="日本語",
                minimum_score=0.0,
            ), [])

    def test_correction_retrieval_preserves_question_modality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="你已经提交了申请。", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            self.assertEqual(store.rank_corrections(
                "你已经提交了申请？",
                source_language="中文",
                target_language="日本語",
                minimum_score=0.0,
            ), [])

    def test_correction_retrieval_detects_question_particle_without_punctuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="你已经提交申请", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            self.assertEqual(store.rank_corrections(
                "你已经提交申请吗",
                source_language="中文",
                target_language="日本語",
                minimum_score=0.0,
            ), [])

    def test_correction_retrieval_detects_leading_question_word_without_punctuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="提交申请失败", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            self.assertEqual(store.rank_corrections(
                "为什么提交申请失败",
                source_language="中文",
                target_language="日本語",
                minimum_score=0.0,
            ), [])

    def test_correction_retrieval_preserves_chinese_quantities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="机器连续运行三天", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            self.assertEqual(store.rank_corrections(
                "机器连续运行五天",
                source_language="中文",
                target_language="日本語",
                minimum_score=0.0,
            ), [])

    def test_similarity_hint_is_explicitly_non_authoritative(self) -> None:
        record = FeedbackRecord.create(
            group_id=1, source_language="中文", target_language="日本語",
            ocr_text="任务已经完成", translation_text="wrong",
        )
        record.corrected_translation = "correct"
        hint = FeedbackStore.correction_prompt_hint(record, score=0.72)
        self.assertIn("相似纠错示例", hint)
        self.assertIn("不得照搬主体、时间、数量或完成状态", hint)

    def test_new_feedback_with_same_trigger_does_not_merge_into_locked_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            first = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong",
            )
            store.submit_correction(first.id, corrected_translation="correct-1")
            automatic = store.upsert_automatic_memory_rule(
                feedback_ids=[first.id], trigger="连接超时", rule="保留超时含义。",
            )
            locked = store.update_memory_rule(automatic.id, rule_text="用户确认规则。")
            second = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="数据库连接超时", translation_text="wrong",
            )
            store.submit_correction(second.id, corrected_translation="correct-2")
            store.update_feedback(second.id, consolidation_status="running")

            result = store.upsert_automatic_memory_rule(
                feedback_ids=[second.id], trigger="连接超时", rule="不得覆盖。",
            )
            saved = store.get_feedback(second.id)

            self.assertNotEqual(result.id, locked.id)
            self.assertEqual(saved.memory_rule_id, result.id)
            self.assertEqual(saved.consolidation_status, "completed")
            self.assertNotIn(second.id, store.get_memory_rule(locked.id).source_feedback_ids)
            self.assertEqual(len(store.list_memory_rules(enabled_only=False)), 2)

    def test_changed_manual_rule_evidence_is_preserved_but_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="高考报名", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            manual = store.approve_feedback(record.id, trigger="高考", rule="人工规则。")
            store.submit_correction(record.id, corrected_translation="correct-2")

            replacement = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id], trigger="高考报名", rule="自动建议。",
            )

            self.assertNotEqual(replacement.id, manual.id)
            self.assertFalse(store.get_memory_rule(manual.id).enabled)
            self.assertTrue(store.get_memory_rule(manual.id).user_locked)
            self.assertEqual(replacement.origin, "automatic")
            self.assertEqual(len(store.list_memory_rules(enabled_only=False)), 2)

    def test_explicit_false_on_legacy_disk_rule_migrates_as_user_owned(self) -> None:
        memory = MemoryRule.from_dict({
            "origin": "legacy",
            "user_locked": False,
            "enabled": True,
        })
        self.assertTrue(memory.user_locked)

    def test_string_false_does_not_enable_memory_rule(self) -> None:
        rule = MemoryRule.from_dict({"enabled": "false", "origin": "automatic"})
        self.assertFalse(rule.enabled)

    def test_malformed_present_boolean_fails_closed(self) -> None:
        rule = MemoryRule.from_dict({"enabled": 0, "origin": "automatic"})
        self.assertFalse(rule.enabled)

    def test_malformed_json_row_does_not_break_other_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            store._feedback_path.write_text('["bad-row"]', encoding="utf-8")
            self.assertEqual(store.list_feedback(), [])

    def test_generic_keyword_does_not_force_unrelated_correction_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="system cache did not refresh", translation_text="wrong",
            )
            store.submit_correction(
                record.id, corrected_translation="correct", keywords=["system"],
            )
            matches = store.match_corrections(
                "the system launches a new game",
                source_language="src",
                target_language="tgt",
            )
            self.assertEqual(matches, [])

    def test_automatic_rule_does_not_embed_one_full_sentence_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="连接超时", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="整句认可译文")
            memory = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id], trigger="连接超时", rule="保留超时含义。",
            )
            self.assertEqual(memory.preferred_translation, "")
            self.assertEqual(memory.example_source, "")
            self.assertNotIn("整句认可译文", memory.as_prompt_hint())

    def test_opposite_negation_is_not_retrieved_as_similar_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="系统没有提交申请，仍显示待审核", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")

            matches = store.match_corrections(
                "系统已经提交申请，显示审核通过",
                source_language="中文",
                target_language="日本語",
            )

            self.assertEqual(matches, [])

    def test_disabling_correction_removes_it_and_its_automatic_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            memory = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id],
                trigger="连接超时",
                rule="必须保留连接失败和超时含义。",
            )

            disabled = store.set_feedback_enabled(record.id, False)

            self.assertFalse(disabled.enabled)
            self.assertEqual(store.match_corrections(
                "接口连接超时",
                source_language="中文",
                target_language="日本語",
            ), [])
            self.assertFalse(store.get_memory_rule(memory.id).enabled)

    def test_automatic_rule_fails_closed_when_evidence_is_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            memory = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id], trigger="连接超时", rule="保留超时含义。",
            )
            records = store.list_feedback()
            records[0].enabled = False
            store._write_feedback(records)

            self.assertTrue(store.get_memory_rule(memory.id).enabled)
            self.assertEqual(store.match_memory_rules(
                "接口连接超时",
                source_language="中文",
                target_language="日本語",
            ), [])

    def test_disable_cross_file_failure_rolls_back_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="connection timeout", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            memory = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id], trigger="timeout", rule="keep timeout",
            )
            with patch.object(store, "_write_feedback", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    store.set_feedback_enabled(record.id, False)

            reloaded = FeedbackStore(tmp)
            self.assertTrue(reloaded.get_feedback(record.id).enabled)
            self.assertTrue(reloaded.get_memory_rule(memory.id).enabled)
            self.assertFalse((Path(tmp) / "feedback-state.transaction.json").exists())

    def test_startup_rolls_back_interrupted_cross_file_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="connection timeout", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            memory = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id], trigger="timeout", rule="keep timeout",
            )
            rules = store.list_memory_rules(enabled_only=False)
            body = {
                "version": 1,
                "recovery_action": "rollback",
                "feedback": store._file_snapshot(store._feedback_path),
                "memory": store._file_snapshot(store._memory_path),
            }
            journal = {**body, "checksum": store._transaction_checksum(body)}
            store._write_bytes_atomic(
                store._transaction_path,
                json.dumps(journal).encode("utf-8"),
            )
            rules[0].enabled = False
            store._write_memory_rules(rules)

            reloaded = FeedbackStore(tmp)
            self.assertTrue(reloaded.get_feedback(record.id).enabled)
            self.assertTrue(reloaded.get_memory_rule(memory.id).enabled)
            self.assertFalse(store._transaction_path.exists())

    def test_unlinked_manual_and_automatic_rules_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="API timeout", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            manual = MemoryRule.create(
                source_language="src", target_language="tgt",
                trigger="API", rule="manual", source_feedback_id=record.id,
                origin="manual", user_locked=True,
            )
            automatic = MemoryRule.create(
                source_language="src", target_language="tgt",
                trigger="timeout", rule="automatic", source_feedback_id=record.id,
                origin="automatic", user_locked=False,
            )
            store._write_memory_rules([manual, automatic])

            self.assertEqual(store.match_memory_rules(
                "API timeout", source_language="src", target_language="tgt",
            ), [])

    def test_manual_rule_write_failure_cannot_leave_active_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="高考开始", translation_text="wrong",
            )
            with patch.object(
                store, "_write_memory_rules", side_effect=OSError("disk full")
            ):
                with self.assertRaises(OSError):
                    store.approve_feedback(
                        record.id,
                        trigger="高考",
                        rule="按大学入学考试语境翻译。",
                    )

            reloaded = FeedbackStore(tmp)
            self.assertEqual(reloaded.get_feedback(record.id).status, "pending")
            self.assertEqual(reloaded.list_memory_rules(enabled_only=False), [])
            self.assertEqual(reloaded.match_memory_rules(
                "高考报名", source_language="中文", target_language="日本語",
            ), [])

    def test_failed_manual_rule_update_restores_previous_rule_and_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="高考开始", translation_text="wrong",
            )
            original = store.approve_feedback(
                record.id, trigger="高考", rule="旧规则。",
            )
            with patch.object(
                store, "_write_memory_rules", side_effect=OSError("disk full")
            ):
                with self.assertRaises(OSError):
                    store.approve_feedback(
                        record.id, trigger="高考报名", rule="新规则。",
                    )

            reloaded = FeedbackStore(tmp)
            saved = reloaded.get_feedback(record.id)
            memory = reloaded.get_memory_rule(original.id)
            self.assertEqual(saved.memory_rule_id, original.id)
            self.assertEqual(memory.trigger, "高考")
            self.assertEqual(memory.rule, "旧规则。")
            self.assertEqual([item.id for item in reloaded.match_memory_rules(
                "高考开始", source_language="中文", target_language="日本語",
            )], [original.id])

    def test_automatic_rule_write_failure_rolls_back_completion_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            with patch.object(
                store, "_write_memory_rules", side_effect=OSError("disk full")
            ):
                with self.assertRaises(OSError):
                    store.upsert_automatic_memory_rule(
                        feedback_ids=[record.id],
                        trigger="连接超时",
                        rule="保留超时含义。",
                    )

            reloaded = FeedbackStore(tmp)
            saved = reloaded.get_feedback(record.id)
            self.assertEqual(saved.consolidation_status, "queued")
            self.assertEqual(saved.memory_rule_id, "")
            self.assertEqual(reloaded.list_memory_rules(enabled_only=False), [])

    def test_failed_automatic_rule_update_restores_previous_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            first = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong-1",
            )
            second = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="数据库连接超时", translation_text="wrong-2",
            )
            store.submit_correction(first.id, corrected_translation="correct-1")
            store.submit_correction(second.id, corrected_translation="correct-2")
            original = store.upsert_automatic_memory_rule(
                feedback_ids=[first.id], trigger="连接超时", rule="旧规则。",
            )
            with patch.object(
                store, "_write_memory_rules", side_effect=OSError("disk full")
            ):
                with self.assertRaises(OSError):
                    self._upsert_automatic(
                        store,
                        feedback_ids=[second.id, first.id],
                        trigger="连接超时",
                        rule="新规则。",
                        target_rule_id=original.id,
                    )

            reloaded = FeedbackStore(tmp)
            memory = reloaded.get_memory_rule(original.id)
            self.assertEqual(memory.rule, "旧规则。")
            self.assertEqual(memory.source_feedback_ids, [first.id])
            self.assertEqual(reloaded.get_feedback(second.id).memory_rule_id, "")
            self.assertEqual(
                reloaded.get_feedback(second.id).consolidation_status,
                "queued",
            )

    def test_automatic_rule_update_requires_explicit_target_and_full_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            first = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong-1",
            )
            second = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="数据库连接超时", translation_text="wrong-2",
            )
            store.submit_correction(first.id, corrected_translation="correct-1")
            store.submit_correction(second.id, corrected_translation="correct-2")
            memory = store.upsert_automatic_memory_rule(
                feedback_ids=[first.id], trigger="连接超时", rule="rule-1",
            )

            updated = self._upsert_automatic(
                store,
                feedback_ids=[first.id, second.id],
                trigger="连接超时",
                rule="rule-2",
                target_rule_id=memory.id,
            )

            self.assertEqual(updated.id, memory.id)
            self.assertEqual(updated.rule, "rule-2")
            self.assertEqual(set(updated.source_feedback_ids), {first.id, second.id})

    def test_changing_secondary_evidence_atomically_disables_old_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            first = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong-1",
            )
            second = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="数据库连接超时", translation_text="wrong-2",
            )
            store.submit_correction(first.id, corrected_translation="correct-1")
            store.submit_correction(second.id, corrected_translation="correct-2")
            memory = self._upsert_automatic(
                store,
                feedback_ids=[first.id, second.id],
                trigger="连接超时",
                rule="保留连接超时含义。",
                primary_feedback_id=first.id,
            )
            self.assertEqual(len(store.match_memory_rules(
                "连接超时", source_language="中文", target_language="日本語",
            )), 1)

            store.submit_correction(
                second.id,
                corrected_translation="completely changed correction",
            )

            self.assertFalse(store.get_memory_rule(memory.id).enabled)
            self.assertEqual(store.match_memory_rules(
                "连接超时", source_language="中文", target_language="日本語",
            ), [])
            self.assertEqual(
                store.get_feedback(second.id).consolidation_status,
                "queued",
            )

    def test_failed_secondary_evidence_change_restores_rule_and_old_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            first = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="接口连接超时", translation_text="wrong-1",
            )
            second = store.add_feedback(
                group_id=2, source_language="中文", target_language="日本語",
                ocr_text="数据库连接超时", translation_text="wrong-2",
            )
            store.submit_correction(first.id, corrected_translation="correct-1")
            store.submit_correction(second.id, corrected_translation="correct-2")
            memory = self._upsert_automatic(
                store,
                feedback_ids=[first.id, second.id],
                trigger="连接超时",
                rule="保留连接超时含义。",
                primary_feedback_id=first.id,
            )
            with patch.object(
                store, "_write_feedback", side_effect=OSError("disk full")
            ):
                with self.assertRaises(OSError):
                    store.submit_correction(
                        second.id,
                        corrected_translation="changed correction",
                    )

            reloaded = FeedbackStore(tmp)
            self.assertEqual(
                reloaded.get_feedback(second.id).corrected_translation,
                "correct-2",
            )
            self.assertTrue(reloaded.get_memory_rule(memory.id).enabled)
            self.assertEqual([item.id for item in reloaded.match_memory_rules(
                "连接超时", source_language="中文", target_language="日本語",
            )], [memory.id])

    def test_automatic_rule_rejects_generic_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="中文", target_language="日本語",
                ocr_text="请求连接失败", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            with self.assertRaises(ValueError):
                store.upsert_automatic_memory_rule(
                    feedback_ids=[record.id], trigger="请求", rule="保留失败含义。",
                )

    def test_automatic_rule_rejects_short_context_free_cjk_trigger(self) -> None:
        for trigger in ("连接", "完成", "状态"):
            with self.subTest(trigger=trigger):
                self.assertFalse(FeedbackStore.automatic_trigger_is_specific(trigger))

    def test_automatic_rule_rejects_combinations_of_generic_terms(self) -> None:
        for trigger in (
            "系统状态", "连接问题", "任务完成", "系统的状态", "系统与状态",
            "连接相关问题", "the system status", "___", "123",
        ):
            with self.subTest(trigger=trigger):
                self.assertFalse(FeedbackStore.automatic_trigger_is_specific(trigger))

    def test_automatic_trigger_keeps_informative_residual_terms(self) -> None:
        for trigger in ("连接超时", "系统崩溃", "配置文件加载失败"):
            with self.subTest(trigger=trigger):
                self.assertTrue(FeedbackStore.automatic_trigger_is_specific(trigger))

    def test_ascii_trigger_uses_identifier_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="API failed", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            memory = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id], trigger="API", rule="keep API semantics",
            )

            self.assertEqual(store.match_memory_rules(
                "rapid growth", source_language="src", target_language="tgt",
            ), [])
            self.assertEqual([item.id for item in store.match_memory_rules(
                "API-client failed", source_language="src", target_language="tgt",
            )], [memory.id])
            self.assertEqual(store.match_memory_rules(
                "myAPIClient failed", source_language="src", target_language="tgt",
            ), [])

    def test_mixed_ascii_cjk_trigger_preserves_ascii_edge_boundaries(self) -> None:
        self.assertFalse(FeedbackStore._trigger_matches_source(
            "API错误", "GraphAPI错误需要修复",
        ))
        self.assertTrue(FeedbackStore._trigger_matches_source(
            "API错误", "调用API错误需要修复",
        ))
        self.assertFalse(FeedbackStore._trigger_matches_source(
            "错误API", "出现错误APIClient",
        ))
        self.assertTrue(FeedbackStore._trigger_matches_source(
            "错误API", "出现错误API失败",
        ))

    def test_empty_feedback_row_is_quarantined_instead_of_becoming_ghost_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pending-feedback.json"
            path.write_text("[{}]", encoding="utf-8")
            store = FeedbackStore(tmp)
            self.assertEqual(store.list_feedback(), [])
            store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="source", translation_text="wrong",
            )
            quarantine = Path(tmp) / "pending-feedback.invalid-rows.json"
            self.assertEqual(json.loads(quarantine.read_text(encoding="utf-8")), [{}])

    def test_unrecoverable_transaction_blocks_reads_and_writes_until_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            original = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="original", translation_text="wrong",
            )
            body = {
                "version": 1,
                "recovery_action": "rollback",
                "feedback": store._file_snapshot(store._feedback_path),
                "memory": store._file_snapshot(store._memory_path),
            }
            journal = {**body, "checksum": store._transaction_checksum(body)}
            store._write_bytes_atomic(
                store._transaction_path,
                json.dumps(journal).encode("utf-8"),
            )
            partial = FeedbackRecord.create(
                group_id=2, source_language="src", target_language="tgt",
                ocr_text="partial", translation_text="wrong",
            )
            store._write_feedback([original, partial])

            with patch.object(
                store, "_restore_file_snapshot", side_effect=OSError("device unavailable")
            ):
                with self.assertRaises(FeedbackStorageUnavailable):
                    store.list_feedback()
                with self.assertRaises(FeedbackStorageUnavailable):
                    store.add_feedback(
                        group_id=3, source_language="src", target_language="tgt",
                        ocr_text="must-not-be-saved", translation_text="wrong",
                    )

            self.assertEqual(
                [item.id for item in store.list_feedback()],
                [original.id],
            )
            self.assertFalse(store._transaction_path.exists())

    def test_corrupt_transaction_is_backed_up_but_remains_persistently_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            FeedbackStore(tmp).add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="safe", translation_text="wrong",
            )
            journal = root / "feedback-state.transaction.json"
            journal.write_text("{broken", encoding="utf-8")

            with self.assertRaises(FeedbackStorageUnavailable):
                FeedbackStore(tmp)

            self.assertTrue(journal.exists())
            backups = list(root.glob("feedback-state.transaction.json.corrupt-*.backup"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "{broken")
            with self.assertRaises(FeedbackStorageUnavailable):
                FeedbackStore(tmp)

    def test_non_list_quarantine_is_backed_up_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "pending-feedback.json"
            malformed = {"id": "broken", "group_id": "wrong"}
            canonical.write_text(json.dumps([malformed]), encoding="utf-8")
            quarantine = root / "pending-feedback.invalid-rows.json"
            quarantine.write_text('{"legacy": true}', encoding="utf-8")
            store = FeedbackStore(tmp)
            self.assertEqual(store.list_feedback(), [])

            store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="new", translation_text="wrong",
            )

            backups = list(root.glob(
                "pending-feedback.invalid-rows.json.corrupt-*.backup"
            ))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), '{"legacy": true}')
            self.assertEqual(json.loads(quarantine.read_text(encoding="utf-8")), [malformed])

    def test_quarantine_backup_failure_does_not_overwrite_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "pending-feedback.json"
            canonical.write_text(
                json.dumps([{"id": "broken", "group_id": "wrong"}]),
                encoding="utf-8",
            )
            quarantine = root / "pending-feedback.invalid-rows.json"
            original = '{"legacy": true}'
            quarantine.write_text(original, encoding="utf-8")
            store = FeedbackStore(tmp)
            self.assertEqual(store.list_feedback(), [])

            with patch.object(
                store, "_write_bytes_atomic", side_effect=OSError("backup failed")
            ):
                with self.assertRaises(OSError):
                    store.add_feedback(
                        group_id=1, source_language="src", target_language="tgt",
                        ocr_text="new", translation_text="wrong",
                    )

            self.assertEqual(quarantine.read_text(encoding="utf-8"), original)

    def test_update_memory_rule_requires_real_booleans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="API timeout", translation_text="wrong",
            )
            memory = store.approve_feedback(
                record.id, trigger="API", rule="keep API semantics",
            )
            for value in ("false", 0, 1, None):
                if value is None:
                    continue
                with self.subTest(enabled=value):
                    with self.assertRaises(ValueError):
                        store.update_memory_rule(memory.id, enabled=value)  # type: ignore[arg-type]
            for value in ("true", 0, 1, None):
                with self.subTest(user_locked=value):
                    with self.assertRaises(ValueError):
                        store.update_memory_rule(
                            memory.id, user_locked=value  # type: ignore[arg-type]
                        )
            saved = store.get_memory_rule(memory.id)
            self.assertTrue(saved.enabled)
            self.assertTrue(saved.user_locked)

    def test_dismissed_input_change_cannot_aba_reactivate_old_automatic_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            owner = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="API timeout", translation_text="wrong",
            )
            evidence = store.add_feedback(
                group_id=2, source_language="src", target_language="tgt",
                ocr_text="database timeout", translation_text="wrong",
            )
            store.submit_correction(owner.id, corrected_translation="owner-correct")
            before = store.submit_correction(
                evidence.id, corrected_translation="evidence-correct"
            )
            memory = self._upsert_automatic(
                store,
                feedback_ids=[owner.id, evidence.id],
                primary_feedback_id=owner.id,
                trigger="API timeout",
                rule="keep timeout semantics",
            )

            dismissed = store.update_feedback(
                evidence.id,
                corrected_translation="changed evidence",
                status="dismissed",
            )
            self.assertNotEqual(
                dismissed.consolidation_input_digest,
                before.consolidation_input_digest,
            )
            self.assertFalse(store.get_memory_rule(memory.id).enabled)

            reactivated = store.update_feedback(evidence.id, status="accepted")
            self.assertEqual(reactivated.corrected_translation, "changed evidence")
            self.assertFalse(store.get_memory_rule(memory.id).enabled)
            self.assertEqual(store.match_memory_rules(
                "API timeout", source_language="src", target_language="tgt",
            ), [])

    def test_update_feedback_rejects_active_state_without_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="source", translation_text="wrong",
            )
            with self.assertRaises(ValueError):
                store.update_feedback(record.id, status="accepted")
            self.assertEqual(store.get_feedback(record.id).status, "pending")

    def test_technical_slashes_are_not_trigger_separators(self) -> None:
        for trigger in ("HTTP/2", "CI/CD", "TCP/IP"):
            with self.subTest(trigger=trigger):
                self.assertEqual(FeedbackStore.split_triggers(trigger), [trigger])
        self.assertEqual(
            FeedbackStore.split_triggers("HTTP/2 / CI/CD / TCP/IP"),
            ["HTTP/2", "CI/CD", "TCP/IP"],
        )
        self.assertEqual(
            FeedbackStore.join_triggers(["HTTP/2", "CI/CD"]),
            "HTTP/2 / CI/CD",
        )

    def test_http_version_trigger_does_not_degrade_into_http_or_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="HTTP/2 failed", translation_text="wrong",
            )
            memory = store.approve_feedback(
                record.id, trigger="HTTP/2", rule="keep protocol version",
            )
            self.assertEqual(store.match_memory_rules(
                "HTTP failed with status 2", source_language="src", target_language="tgt",
            ), [])
            self.assertEqual([item.id for item in store.match_memory_rules(
                "HTTP/2 failed", source_language="src", target_language="tgt",
            )], [memory.id])

    def test_exact_correction_identity_preserves_semantic_punctuation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="Deploy API now.", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            matches = store.rank_correction_matches(
                "Deploy API now!",
                source_language="src",
                target_language="tgt",
                minimum_score=0.0,
            )
            self.assertEqual(len(matches), 1)
            self.assertIsInstance(matches[0], CorrectionMatch)
            self.assertFalse(matches[0].exact_match)
            self.assertLess(matches[0].score, 1.0)
            self.assertEqual(
                store.rank_corrections(
                    "Deploy API now!", source_language="src", target_language="tgt",
                    minimum_score=0.0,
                ),
                [(matches[0].score, matches[0].record)],
            )

    def test_exact_correction_identity_normalizes_whitespace_but_preserves_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="  API   FAILED  ", translation_text="wrong",
            )
            store.submit_correction(record.id, corrected_translation="correct")
            match = store.rank_correction_matches(
                "API FAILED", source_language="src", target_language="tgt",
            )[0]
            self.assertTrue(match.exact_match)
            self.assertEqual(match.score, 1.0)
            self.assertEqual(
                store.rank_correction_matches(
                    "api failed",
                    source_language="src",
                    target_language="tgt",
                    minimum_score=0.0,
                ),
                [],
            )

    def test_numeric_and_operator_variants_never_become_exact(self) -> None:
        pairs = (
            ("版本 3.0", "版本 30"),
            ("数量 >=3", "数量 <=3"),
            ("范围 70-80", "范围 7080"),
        )
        for source, query in pairs:
            with self.subTest(source=source, query=query), tempfile.TemporaryDirectory() as tmp:
                store = FeedbackStore(tmp)
                record = store.add_feedback(
                    group_id=1, source_language="src", target_language="tgt",
                    ocr_text=source, translation_text="wrong",
                )
                store.submit_correction(record.id, corrected_translation="correct")
                matches = store.rank_correction_matches(
                    query, source_language="src", target_language="tgt",
                    minimum_score=0.0,
                )
                self.assertTrue(not matches or not matches[0].exact_match)
                self.assertTrue(not matches or matches[0].score < 1.0)

    def test_prompt_hint_never_infers_exactness_from_score(self) -> None:
        record = FeedbackRecord.create(
            group_id=1, source_language="src", target_language="tgt",
            ocr_text="source", translation_text="wrong",
        )
        record.corrected_translation = "correct"
        similar = FeedbackStore.correction_prompt_hint(
            record, score=1.0, exact_match=False,
        )
        exact = FeedbackStore.correction_prompt_hint(
            record, score=0.1, exact_match=True,
        )
        self.assertIn("相似纠错示例", similar)
        self.assertNotIn("完全一致", similar)
        self.assertIn("完全一致", exact)

    def test_typed_semantic_anchors_reject_conflicting_relations(self) -> None:
        pairs = (
            ("系统允许提交", "系统禁止提交"),
            ("用户必须更新", "用户可以更新"),
            ("任务昨天完成", "任务明天完成"),
            ("更新时保存配置", "更新后保存配置"),
            ("最多处理 3 个", "至少处理 3 个"),
            ("数量 >= 3", "数量 <= 3"),
        )
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertFalse(FeedbackStore._semantic_anchors_compatible(left, right))

    def test_submit_correction_can_explicitly_clear_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="connection timeout", translation_text="wrong",
            )
            first = store.submit_correction(
                record.id,
                corrected_translation="correct",
                keywords=["connection timeout"],
            )
            cleared = store.submit_correction(
                record.id,
                corrected_translation="correct",
                keywords=[],
            )
            self.assertEqual(cleared.keywords, [])
            self.assertNotEqual(
                first.consolidation_input_digest,
                cleared.consolidation_input_digest,
            )

    def test_changing_secondary_evidence_revokes_automatic_rule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            owner = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="API connection timeout", translation_text="wrong-1",
            )
            secondary = store.add_feedback(
                group_id=2, source_language="src", target_language="tgt",
                ocr_text="database connection timeout", translation_text="wrong-2",
            )
            store.submit_correction(owner.id, corrected_translation="correct-1")
            store.submit_correction(secondary.id, corrected_translation="correct-2")
            automatic = self._upsert_automatic(
                store,
                feedback_ids=[owner.id, secondary.id],
                primary_feedback_id=owner.id,
                trigger="connection timeout",
                rule="preserve timeout semantics",
            )
            self.assertEqual(
                set(automatic.source_feedback_digests),
                {owner.id, secondary.id},
            )

            store.approve_feedback(
                secondary.id,
                trigger="database connection",
                rule="user confirmed database rule",
                preferred_translation="correct-2-revised",
            )

            self.assertFalse(store.get_memory_rule(automatic.id).enabled)
            self.assertNotIn(
                automatic.id,
                [item.id for item in store.match_memory_rules(
                    "API connection timeout",
                    source_language="src",
                    target_language="tgt",
                )],
            )

    def test_resubmitting_manual_rule_owner_preserves_only_unchanged_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="API connection timeout", translation_text="wrong",
            )
            manual = store.approve_feedback(
                record.id,
                trigger="API connection timeout",
                rule="user confirmed timeout rule",
                preferred_translation="correct-v1",
            )

            unchanged = store.submit_correction(
                record.id,
                corrected_translation="correct-v1",
            )
            self.assertEqual(unchanged.status, "confirmed")
            self.assertTrue(store.get_memory_rule(manual.id).enabled)

            changed = store.submit_correction(
                record.id,
                corrected_translation="correct-v2",
            )
            self.assertEqual(changed.status, "accepted")
            self.assertFalse(store.get_memory_rule(manual.id).enabled)
            self.assertEqual(store.match_memory_rules(
                "API connection timeout",
                source_language="src",
                target_language="tgt",
            ), [])

            automatic = store.upsert_automatic_memory_rule(
                feedback_ids=[record.id],
                trigger="API connection timeout",
                rule="fresh automatic rule",
                primary_feedback_id=record.id,
                expected_input_digest=changed.consolidation_input_digest,
            )
            self.assertNotEqual(automatic.id, manual.id)
            self.assertEqual(automatic.origin, "automatic")
            self.assertEqual(store.get_feedback(record.id).memory_rule_id, automatic.id)

    def test_automatic_rule_without_evidence_versions_is_inert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="connection timeout", translation_text="wrong",
            )
            record = store.submit_correction(record.id, corrected_translation="correct")
            rule = MemoryRule.create(
                source_language="src",
                target_language="tgt",
                trigger="connection timeout",
                rule="legacy automatic text",
                source_feedback_id=record.id,
                origin="automatic",
            )
            record.memory_rule_id = rule.id
            record.consolidation_status = "completed"
            (Path(tmp) / "pending-feedback.json").write_text(
                json.dumps([asdict(record)], ensure_ascii=False), encoding="utf-8",
            )
            (Path(tmp) / "memory-rules.json").write_text(
                json.dumps([asdict(rule)], ensure_ascii=False), encoding="utf-8",
            )
            self.assertEqual(store.match_memory_rules(
                "connection timeout", source_language="src", target_language="tgt",
            ), [])

    def test_failure_write_requires_digest_and_attempt_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="connection timeout", translation_text="wrong",
            )
            record = store.submit_correction(record.id, corrected_translation="correct")
            store.update_feedback(
                record.id,
                consolidation_status="running",
                consolidation_attempts=2,
            )
            self.assertIsNone(store.fail_consolidation_if_current(
                record.id,
                expected_input_digest=record.consolidation_input_digest,
                expected_attempt=1,
                message="old failure",
            ))
            self.assertEqual(store.get_feedback(record.id).consolidation_status, "running")
            self.assertIsNotNone(store.fail_consolidation_if_current(
                record.id,
                expected_input_digest=record.consolidation_input_digest,
                expected_attempt=2,
                message="current failure",
            ))
            self.assertEqual(store.get_feedback(record.id).consolidation_error, "current failure")

    def test_technical_symbols_do_not_form_false_similar_matches(self) -> None:
        pairs = (
            ("offset +5", "offset -5"),
            ("x == 3", "x != 3"),
            ("compile C++ code", "compile C code"),
            ("use A/B routing", "use AB routing"),
        )
        for source, query in pairs:
            with self.subTest(source=source, query=query), tempfile.TemporaryDirectory() as tmp:
                store = FeedbackStore(tmp)
                record = store.add_feedback(
                    group_id=1, source_language="src", target_language="tgt",
                    ocr_text=source, translation_text="wrong",
                )
                store.submit_correction(record.id, corrected_translation="correct")
                self.assertEqual(store.rank_correction_matches(
                    query,
                    source_language="src",
                    target_language="tgt",
                    minimum_score=0.0,
                ), [])

    def test_semantic_anchor_parser_avoids_common_substring_false_positives(self) -> None:
        compatible_pairs = (
            ("系统性能优化", "系统优化"),
            ("提交后台任务", "提交任务"),
            ("无锡部署完成", "苏州部署完成"),
            ("非常稳定", "系统稳定"),
            ("不超过 3 次", "最多 3 次"),
        )
        for left, right in compatible_pairs:
            with self.subTest(left=left, right=right):
                self.assertTrue(FeedbackStore._semantic_anchors_compatible(left, right))

    def test_trigger_matching_normalizes_width_and_ambiguous_lists_round_trip(self) -> None:
        encoded = FeedbackStore.join_triggers(["input / output"])
        self.assertEqual(FeedbackStore.split_triggers(encoded), ["input / output"])
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="API failed", translation_text="wrong",
            )
            memory = store.approve_feedback(
                record.id, trigger="ＡＰＩ", rule="preserve API semantics",
            )
            self.assertEqual([item.id for item in store.match_memory_rules(
                "API failed", source_language="src", target_language="tgt",
            )], [memory.id])

    def test_manual_orphan_rule_is_quarantined_and_never_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = asdict(MemoryRule.create(
                source_language="src", target_language="tgt",
                trigger="unsafe trigger", rule="orphan",
                origin="manual", user_locked=True,
            ))
            (Path(tmp) / "memory-rules.json").write_text(
                json.dumps([raw], ensure_ascii=False), encoding="utf-8",
            )
            store = FeedbackStore(tmp)
            self.assertEqual(store.list_memory_rules(enabled_only=False), [])

    def test_allow_unavailable_constructs_but_every_store_access_stays_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "feedback-state.transaction.json").write_text(
                "not valid json", encoding="utf-8",
            )
            store = FeedbackStore(root, allow_unavailable=True)
            with self.assertRaises(FeedbackStorageUnavailable):
                store.list_feedback()
            with self.assertRaises(FeedbackStorageUnavailable):
                store.add_feedback(
                    group_id=1, source_language="src", target_language="tgt",
                    ocr_text="source", translation_text="wrong",
                )

    def test_trigger_codec_round_trips_ambiguous_punctuation_and_case(self) -> None:
        triggers = ["May", "may", "alpha,beta", "left;right", "input|output"]
        encoded = FeedbackStore.join_triggers(triggers)
        self.assertEqual(FeedbackStore.split_triggers(encoded), triggers)

    def test_uppercase_trigger_does_not_collapse_into_lowercase_word(self) -> None:
        self.assertTrue(FeedbackStore._trigger_matches_source("May", "May release"))
        self.assertFalse(FeedbackStore._trigger_matches_source("May", "may release"))

    def test_exact_case_trigger_suppresses_casefold_competitor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            month = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="May release", translation_text="wrong",
            )
            permission = store.add_feedback(
                group_id=2, source_language="src", target_language="tgt",
                ocr_text="may proceed", translation_text="wrong",
            )
            month_rule = store.approve_feedback(
                month.id, trigger="May", rule="month meaning",
            )
            permission_rule = store.approve_feedback(
                permission.id, trigger="may", rule="permission meaning",
            )

            self.assertEqual(
                [item.id for item in store.match_memory_rules(
                    "May release", source_language="src", target_language="tgt",
                )],
                [month_rule.id],
            )
            self.assertEqual(
                [item.id for item in store.match_memory_rules(
                    "may proceed", source_language="src", target_language="tgt",
                )],
                [permission_rule.id],
            )

    def test_infix_technical_symbols_block_false_similarity(self) -> None:
        for source, query in (
            ("use A+B", "use AB"),
            ("compute A*B", "compute AB"),
            ("call foo::bar", "call foobar"),
            ("set a=b", "set ab"),
        ):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                store = FeedbackStore(tmp)
                record = store.add_feedback(
                    group_id=1, source_language="src", target_language="tgt",
                    ocr_text=source, translation_text="wrong",
                )
                store.submit_correction(record.id, corrected_translation="correct")
                self.assertEqual(store.rank_correction_matches(
                    query,
                    source_language="src",
                    target_language="tgt",
                    minimum_score=0.0,
                ), [])

    def test_duplicate_exact_corrections_with_equal_timestamps_prefer_latest_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "app.feedback.store._now_iso",
            return_value="2026-07-15T00:00:00.000000+00:00",
        ):
            store = FeedbackStore(tmp)
            first = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="same source", translation_text="wrong-1",
            )
            store.submit_correction(first.id, corrected_translation="old correction")
            second = store.add_feedback(
                group_id=2, source_language="src", target_language="tgt",
                ocr_text="same source", translation_text="wrong-2",
            )
            store.submit_correction(second.id, corrected_translation="new correction")

            matches = store.rank_correction_matches(
                "same source", source_language="src", target_language="tgt",
            )
            self.assertEqual([item.record.id for item in matches], [second.id])

    def test_invalid_utf8_store_degrades_and_preserves_backup_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pending-feedback.json"
            original = b"\xff\xfe\x00broken"
            path.write_bytes(original)
            store = FeedbackStore(tmp)
            self.assertEqual(store.list_feedback(), [])

            store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="source", translation_text="wrong",
            )

            backups = list(Path(tmp).glob("pending-feedback.json.corrupt-*.backup"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))), 1)

    def test_transient_read_error_fails_closed_without_replacing_canonical_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            original = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="original", translation_text="wrong",
            )
            path = Path(tmp) / "pending-feedback.json"
            real_read_text = Path.read_text
            failed = False

            def transient_read(target: Path, *args, **kwargs):
                nonlocal failed
                if target == path and not failed:
                    failed = True
                    raise PermissionError("temporary file lock")
                return real_read_text(target, *args, **kwargs)

            with patch.object(Path, "read_text", new=transient_read):
                with self.assertRaises(FeedbackStorageUnavailable):
                    store.add_feedback(
                        group_id=2, source_language="src", target_language="tgt",
                        ocr_text="new", translation_text="wrong",
                    )

            self.assertEqual([item.id for item in store.list_feedback()], [original.id])
            self.assertEqual(
                list(Path(tmp).glob("pending-feedback.json.corrupt-*.backup")),
                [],
            )

    def test_failure_hint_snapshots_round_trip_only_for_recorded_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="source", translation_text="wrong",
                matched_memory_rule_ids=["rule-1"],
                matched_correction_ids=["correction-1"],
                matched_memory_hint_snapshots={
                    "rule-1": "rule snapshot",
                    "unrecorded": "must be removed",
                },
                matched_correction_hint_snapshots={
                    "correction-1": "correction snapshot",
                },
            )

            loaded = FeedbackStore(tmp).get_feedback(record.id)
            self.assertEqual(
                loaded.matched_memory_hint_snapshots,
                {"rule-1": "rule snapshot"},
            )
            self.assertEqual(
                loaded.matched_correction_hint_snapshots,
                {"correction-1": "correction snapshot"},
            )


if __name__ == "__main__":
    unittest.main()
