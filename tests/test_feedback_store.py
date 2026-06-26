"""Tests for local feedback and translation-memory storage."""

from __future__ import annotations

import tempfile
import unittest

from app.feedback.store import FeedbackStore


class FeedbackStoreTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
