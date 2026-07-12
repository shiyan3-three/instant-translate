"""Tests for local feedback and translation-memory storage."""

from __future__ import annotations

import tempfile
import unittest

from app.feedback.store import FeedbackStore


class FeedbackStoreTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
