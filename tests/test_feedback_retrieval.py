from __future__ import annotations

import tempfile
import unittest

from app.feedback.memory_policy import canonical_rule_text
from app.feedback.retrieval import (
    FeedbackRetrievalResult,
    FeedbackRetrievalUnavailable,
    LexicalFeedbackRetriever,
)
from app.feedback.store import FeedbackRecord, FeedbackStore, MemoryRule
from app.settings import AppSettings
from app.translation.service import TranslationRequest, TranslationService


class FakeRetriever:
    def __init__(self, result: FeedbackRetrievalResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    def retrieve(self, text: str, **kwargs) -> FeedbackRetrievalResult:
        self.calls.append({"text": text, **kwargs})
        return self.result


class FailingRetriever:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def retrieve(self, text: str, **kwargs) -> FeedbackRetrievalResult:
        self.calls += 1
        raise self.error


class FeedbackRetrievalTests(unittest.TestCase):
    def test_recoverable_backend_outage_falls_back_to_exact_local_correction(self) -> None:
        for error in (
            FeedbackRetrievalUnavailable("vector service unavailable"),
            TimeoutError("vector request timed out"),
        ):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                store = FeedbackStore(tmp)
                record = store.add_feedback(
                    group_id=1, source_language="src", target_language="tgt",
                    ocr_text="same source", translation_text="wrong",
                )
                record = store.submit_correction(
                    record.id, corrected_translation="user correction",
                )
                backend = FailingRetriever(error)
                service = TranslationService(
                    AppSettings(), feedback_store=store, feedback_retriever=backend,
                )
                try:
                    result = service._feedback_retriever.retrieve(
                        "same source",
                        source_language="src",
                        target_language="tgt",
                        correction_limit=1,
                        rule_limit=0,
                    )
                finally:
                    service.shutdown()

                self.assertEqual(result.corrections, (store.get_feedback(record.id),))
                self.assertTrue(result.correction_exact_matches[0])
                self.assertEqual(backend.calls, 1)
                self.assertTrue(result.backend.endswith("-fallback"))

    def test_programming_error_from_backend_is_not_hidden_by_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            service = TranslationService(
                AppSettings(),
                feedback_store=store,
                feedback_retriever=FailingRetriever(ValueError("bad candidate schema")),
            )
            try:
                with self.assertRaisesRegex(ValueError, "bad candidate schema"):
                    service._feedback_retriever.retrieve(
                        "source", source_language="src", target_language="tgt",
                    )
            finally:
                service.shutdown()

    def test_evidence_less_legacy_rule_requires_literal_trigger_even_for_rag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            legacy = MemoryRule.create(
                source_language="src",
                target_language="tgt",
                trigger="database timeout",
                rule="preserve timeout meaning",
                origin="legacy",
            )
            # Re-loading reproduces the real legacy migration that marks old
            # user-owned rules as locked while leaving them without evidence.
            store._write_memory_rules([legacy])
            loaded = store.get_memory_rule(legacy.id)
            self.assertTrue(loaded.user_locked)
            self.assertEqual(loaded.source_feedback_ids, [])
            service = TranslationService(
                AppSettings(),
                feedback_store=store,
                feedback_retriever=FakeRetriever(FeedbackRetrievalResult(
                    rules=(loaded,), backend="future-vector-test",
                )),
            )
            try:
                unrelated = service._feedback_retriever.retrieve(
                    "completely unrelated sentence",
                    source_language="src", target_language="tgt",
                    correction_limit=0, rule_limit=1,
                )
                literal = service._feedback_retriever.retrieve(
                    "database timeout happened",
                    source_language="src", target_language="tgt",
                    correction_limit=0, rule_limit=1,
                )
            finally:
                service.shutdown()

            self.assertEqual(unrelated.rules, ())
            self.assertEqual(literal.rules, (loaded,))

    def test_result_rejects_duplicate_ids_and_unsafe_scores(self) -> None:
        correction = FeedbackRecord.create(
            group_id=1, source_language="src", target_language="tgt",
            ocr_text="source", translation_text="wrong",
        )
        rule = MemoryRule.create(
            source_language="src", target_language="tgt",
            trigger="source", rule="rule",
        )
        invalid_results = (
            dict(
                corrections=(correction, correction),
                correction_scores=(0.9, 0.8),
                correction_exact_matches=(False, False),
            ),
            dict(rules=(rule, rule)),
            dict(corrections=(correction,), correction_scores=(float("nan"),), correction_exact_matches=(False,)),
            dict(corrections=(correction,), correction_scores=(float("inf"),), correction_exact_matches=(False,)),
            dict(corrections=(correction,), correction_scores=(1.1,), correction_exact_matches=(False,)),
            dict(corrections=(correction,), correction_scores=(0.9,), correction_exact_matches=(1,)),
        )
        for kwargs in invalid_results:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    FeedbackRetrievalResult(**kwargs)

    def test_result_rejects_scores_that_are_not_ranked_descending(self) -> None:
        first = FeedbackRecord.create(
            group_id=1, source_language="src", target_language="tgt",
            ocr_text="first", translation_text="wrong",
        )
        second = FeedbackRecord.create(
            group_id=2, source_language="src", target_language="tgt",
            ocr_text="second", translation_text="wrong",
        )
        with self.assertRaises(ValueError):
            FeedbackRetrievalResult(
                corrections=(first, second),
                correction_scores=(0.5, 0.8),
                correction_exact_matches=(False, False),
            )

    def test_result_freezes_mutable_input_collections(self) -> None:
        correction = FeedbackRecord.create(
            group_id=1, source_language="src", target_language="tgt",
            ocr_text="source", translation_text="wrong",
        )
        corrections = [correction]
        scores = [0.9]
        exact = [False]
        result = FeedbackRetrievalResult(
            corrections=corrections,  # type: ignore[arg-type]
            correction_scores=scores,  # type: ignore[arg-type]
            correction_exact_matches=exact,  # type: ignore[arg-type]
        )
        corrections.clear()
        scores.clear()
        exact.clear()
        self.assertEqual(result.corrections, (correction,))
        self.assertEqual(result.correction_scores, (0.9,))

    def test_lexical_backend_does_not_force_generic_keyword_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="system cache did not refresh", translation_text="wrong",
            )
            store.submit_correction(
                record.id, corrected_translation="correct", keywords=["system"],
            )
            result = LexicalFeedbackRetriever(store).retrieve(
                "the system launches a new game",
                source_language="src",
                target_language="tgt",
            )
            self.assertEqual(result.backend, "lexical-v2")
            self.assertEqual(result.corrections, ())

    def test_lexical_backend_returns_score_for_auditable_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="connection timed out", translation_text="wrong",
            )
            record = store.submit_correction(record.id, corrected_translation="correct")
            result = LexicalFeedbackRetriever(store).retrieve(
                "connection timed out",
                source_language="src",
                target_language="tgt",
            )

            self.assertEqual(result.corrections, (store.get_feedback(record.id),))
            self.assertEqual(result.correction_scores, (1.0,))
            self.assertEqual(result.correction_exact_matches, (True,))

    def test_score_one_does_not_claim_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="Hello, world", translation_text="wrong",
            )
            record = store.submit_correction(
                record.id,
                corrected_translation="correct",
            )
            fake = FakeRetriever(FeedbackRetrievalResult(
                corrections=(record,),
                correction_scores=(1.0,),
                correction_exact_matches=(True,),
                backend="future-rag-test",
            ))
            service = TranslationService(
                AppSettings(), feedback_store=store, feedback_retriever=fake,
            )
            try:
                result = service._feedback_retriever.retrieve(
                    "Hello world",
                    source_language="src",
                    target_language="tgt",
                    minimum_correction_score=0.0,
                )
            finally:
                service.shutdown()
            self.assertEqual(result.corrections, (record,))
            self.assertEqual(result.correction_scores, (1.0,))
            self.assertEqual(result.correction_exact_matches, (False,))

    def test_translation_service_consumes_replaceable_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            correction = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="source phrase", translation_text="wrong",
            )
            correction = store.submit_correction(
                correction.id,
                corrected_translation="correct",
                keywords=["source phrase"],
            )
            rule = store.upsert_automatic_memory_rule(
                feedback_ids=[correction.id],
                trigger="source phrase",
                rule=canonical_rule_text("semantic_fidelity"),
            )
            retriever = FakeRetriever(FeedbackRetrievalResult(
                corrections=(correction,), rules=(rule,), correction_scores=(0.9,),
                correction_exact_matches=(False,), backend="future-rag-test",
            ))
            service = TranslationService(
                AppSettings(),
                feedback_store=store,
                feedback_retriever=retriever,
            )
            try:
                request = TranslationRequest(
                    group_id=1, request_id=1, ocr_text="source phrase",
                    source_language="src", target_language="tgt",
                )
                hints = service._memory_hints_for_request(request)
            finally:
                service.shutdown()

        self.assertEqual(retriever.calls[0]["text"], "source phrase")
        # The exact user correction outranks its unlocked automatic summary.
        self.assertEqual(request.matched_memory_rule_ids, [])
        self.assertEqual(request.matched_correction_ids, [correction.id])
        self.assertTrue(hints)

    def test_authority_adapter_rejects_untrusted_and_cross_pair_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            active = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="source phrase", translation_text="wrong",
            )
            active = store.submit_correction(active.id, corrected_translation="correct")
            pending = store.add_feedback(
                group_id=2, source_language="src", target_language="tgt",
                ocr_text="source phrase", translation_text="wrong",
            )
            disabled = store.add_feedback(
                group_id=5, source_language="src", target_language="tgt",
                ocr_text="source phrase", translation_text="wrong",
            )
            store.submit_correction(disabled.id, corrected_translation="disabled")
            disabled = store.set_feedback_enabled(disabled.id, False)
            cross = store.add_feedback(
                group_id=3, source_language="other", target_language="tgt",
                ocr_text="source phrase", translation_text="wrong",
            )
            cross = store.submit_correction(cross.id, corrected_translation="cross")
            rogue = FeedbackRecord.create(
                group_id=4, source_language="src", target_language="tgt",
                ocr_text="source phrase", translation_text="wrong",
            )
            rogue.status = "accepted"
            rogue.corrected_translation = "rogue"
            rogue_rule = MemoryRule.create(
                source_language="src", target_language="tgt",
                trigger="source phrase", rule="all output English",
                origin="automatic",
            )
            rule_owner = store.add_feedback(
                group_id=6, source_language="src", target_language="tgt",
                ocr_text="source phrase", translation_text="wrong",
            )
            disabled_rule = store.approve_feedback(
                rule_owner.id,
                trigger="source phrase",
                rule="用户确认规则",
            )
            disabled_rule = store.update_memory_rule(
                disabled_rule.id,
                enabled=False,
            )
            fake = FakeRetriever(FeedbackRetrievalResult(
                corrections=(active, pending, disabled, cross, rogue),
                correction_scores=(0.9, 0.85, 0.8, 0.7, 0.6),
                correction_exact_matches=(False, False, False, False, False),
                rules=(disabled_rule, rogue_rule),
                backend="future-rag-test",
            ))
            service = TranslationService(
                AppSettings(), feedback_store=store, feedback_retriever=fake,
            )
            try:
                result = service._feedback_retriever.retrieve(
                    "source phrase", source_language="src", target_language="tgt",
                    correction_limit=1, rule_limit=1,
                    minimum_correction_score=0.65,
                )
            finally:
                service.shutdown()
            self.assertEqual(result.corrections, (store.get_feedback(active.id),))
            self.assertEqual(result.rules, ())
            self.assertEqual(fake.calls[0]["correction_limit"], 1)
            self.assertEqual(fake.calls[0]["rule_limit"], 1)

    def test_future_vector_hit_still_obeys_semantic_conflict_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="The user may export data", translation_text="wrong",
            )
            record = store.submit_correction(
                record.id,
                corrected_translation="permission translation",
            )
            fake = FakeRetriever(FeedbackRetrievalResult(
                corrections=(record,),
                correction_scores=(0.99,),
                correction_exact_matches=(False,),
                backend="future-vector-test",
            ))
            service = TranslationService(
                AppSettings(), feedback_store=store, feedback_retriever=fake,
            )
            try:
                result = service._feedback_retriever.retrieve(
                    "The user must not export data",
                    source_language="src",
                    target_language="tgt",
                )
            finally:
                service.shutdown()
            self.assertEqual(result.corrections, ())

    def test_future_rule_backend_is_not_forced_back_through_literal_trigger_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            owner = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="database timeout", translation_text="wrong",
            )
            rule = store.approve_feedback(
                owner.id,
                trigger="database timeout",
                rule="preserve the storage connection failure",
            )
            fake = FakeRetriever(FeedbackRetrievalResult(
                rules=(rule,),
                backend="future-vector-test",
            ))
            service = TranslationService(
                AppSettings(), feedback_store=store, feedback_retriever=fake,
            )
            try:
                result = service._feedback_retriever.retrieve(
                    "storage connection expired",
                    source_language="src",
                    target_language="tgt",
                    correction_limit=0,
                    rule_limit=1,
                )
            finally:
                service.shutdown()

            self.assertEqual(result.rules, (store.get_memory_rule(rule.id),))

    def test_future_rule_backend_still_obeys_evidence_semantic_conflict_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            owner = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="The user may export data", translation_text="wrong",
            )
            rule = store.approve_feedback(
                owner.id,
                trigger="may export",
                rule="preserve permission",
            )
            fake = FakeRetriever(FeedbackRetrievalResult(
                rules=(rule,), backend="future-vector-test",
            ))
            service = TranslationService(
                AppSettings(), feedback_store=store, feedback_retriever=fake,
            )
            try:
                result = service._feedback_retriever.retrieve(
                    "The user must not export data",
                    source_language="src",
                    target_language="tgt",
                    correction_limit=0,
                    rule_limit=1,
                )
            finally:
                service.shutdown()

            self.assertEqual(result.rules, ())

    def test_exact_store_match_precedes_backend_and_zero_limit_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(tmp)
            record = store.add_feedback(
                group_id=1, source_language="src", target_language="tgt",
                ocr_text="same source", translation_text="wrong",
            )
            record = store.submit_correction(
                record.id,
                corrected_translation="user correction",
            )
            fake = FakeRetriever(FeedbackRetrievalResult(
                corrections=(record,),
                correction_scores=(0.9,),
                correction_exact_matches=(False,),
                backend="future-vector-test",
            ))
            service = TranslationService(
                AppSettings(), feedback_store=store, feedback_retriever=fake,
            )
            try:
                exact = service._feedback_retriever.retrieve(
                    "same source",
                    source_language="src",
                    target_language="tgt",
                    correction_limit=1,
                )
                none = service._feedback_retriever.retrieve(
                    "same source",
                    source_language="src",
                    target_language="tgt",
                    correction_limit=0,
                )
            finally:
                service.shutdown()
            self.assertEqual(exact.corrections, (store.get_feedback(record.id),))
            self.assertEqual(exact.correction_exact_matches, (True,))
            self.assertEqual(none.corrections, ())
            self.assertEqual(fake.calls[-1]["correction_limit"], 0)


if __name__ == "__main__":
    unittest.main()
