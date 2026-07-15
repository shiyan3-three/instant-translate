"""Replaceable retrieval boundary for correction examples and memory rules.

The production default remains deterministic lexical retrieval.  A future
vector or hybrid RAG backend can implement :class:`FeedbackRetriever` without
changing TranslationService, FeedbackOptimizer, or the GUI/storage schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from app.feedback.memory_policy import is_canonical_automatic_rule
from app.feedback.store import FeedbackRecord, FeedbackStore, MemoryRule
from app.logger import get_logger


class FeedbackRetrievalUnavailable(RuntimeError):
    """A recoverable retrieval-backend outage, not a schema/programming error."""


@dataclass(frozen=True)
class FeedbackRetrievalResult:
    """Ranked feedback knowledge plus the backend that produced it."""

    corrections: tuple[FeedbackRecord, ...] = ()
    rules: tuple[MemoryRule, ...] = ()
    correction_scores: tuple[float, ...] = ()
    correction_exact_matches: tuple[bool, ...] = ()
    backend: str = "lexical"

    def __post_init__(self) -> None:
        object.__setattr__(self, "corrections", tuple(self.corrections))
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "correction_scores", tuple(self.correction_scores))
        object.__setattr__(
            self,
            "correction_exact_matches",
            tuple(self.correction_exact_matches),
        )
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("feedback retrieval backend must be a nonempty string")
        if any(not isinstance(item, FeedbackRecord) for item in self.corrections):
            raise ValueError("corrections must contain FeedbackRecord values")
        if any(not isinstance(item, MemoryRule) for item in self.rules):
            raise ValueError("rules must contain MemoryRule values")
        if len(self.corrections) != len(self.correction_scores):
            raise ValueError(
                "correction_scores must align one-to-one with corrections"
            )
        if len(self.corrections) != len(self.correction_exact_matches):
            raise ValueError(
                "correction_exact_matches must align one-to-one with corrections"
            )
        if any(type(item) is not bool for item in self.correction_exact_matches):
            raise ValueError("correction_exact_matches must contain real bool values")
        correction_ids = [item.id for item in self.corrections]
        rule_ids = [item.id for item in self.rules]
        if any(not identifier.strip() for identifier in [*correction_ids, *rule_ids]):
            raise ValueError("feedback retrieval IDs must be nonempty")
        if len(correction_ids) != len(set(correction_ids)):
            raise ValueError("feedback retrieval returned duplicate correction IDs")
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("feedback retrieval returned duplicate rule IDs")
        normalized_scores: list[float] = []
        for score in self.correction_scores:
            if (
                isinstance(score, bool)
                or not isinstance(score, Real)
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise ValueError("correction scores must be finite values in [0, 1]")
            normalized_scores.append(float(score))
        if any(
            left < right
            for left, right in zip(normalized_scores, normalized_scores[1:])
        ):
            raise ValueError("correction scores must be ordered from highest to lowest")
        object.__setattr__(self, "correction_scores", tuple(normalized_scores))


class FeedbackRetriever(Protocol):
    """Stable seam for lexical, vector, or hybrid feedback retrieval.

    Implementations that perform external I/O must enforce their own bounded
    timeout and raise :class:`FeedbackRetrievalUnavailable` for recoverable
    outages.  Invalid result schemas and programming errors must remain hard
    failures so the authority boundary cannot silently accept bad candidates.
    """

    def retrieve(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        correction_limit: int = 3,
        rule_limit: int = 3,
        exclude_feedback_id: str = "",
        minimum_correction_score: float = 0.42,
    ) -> FeedbackRetrievalResult:
        ...


class LexicalFeedbackRetriever:
    """Current local retrieval implementation; no embedding dependency."""

    backend_name = "lexical-v2"

    def __init__(self, store: FeedbackStore) -> None:
        self._store = store

    def retrieve(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        correction_limit: int = 3,
        rule_limit: int = 3,
        exclude_feedback_id: str = "",
        minimum_correction_score: float = 0.42,
    ) -> FeedbackRetrievalResult:
        correction_hits = []
        if correction_limit > 0:
            correction_hits = self._store.rank_correction_matches(
                text,
                source_language=source_language,
                target_language=target_language,
                limit=correction_limit,
                exclude_feedback_id=exclude_feedback_id,
                minimum_score=minimum_correction_score,
            )
        rules = []
        if rule_limit > 0:
            rules = self._store.match_memory_rules(
                text,
                source_language=source_language,
                target_language=target_language,
                limit=rule_limit,
            )
        return FeedbackRetrievalResult(
            corrections=tuple(item.record for item in correction_hits),
            rules=tuple(rules),
            correction_scores=tuple(item.score for item in correction_hits),
            correction_exact_matches=tuple(item.exact_match for item in correction_hits),
            backend=self.backend_name,
        )


class AuthoritativeFeedbackRetriever:
    """Rehydrate untrusted backend hits from the canonical local store.

    A vector or plugin backend is allowed to propose IDs and an ordering.  It
    is never allowed to supply the object injected into a model request.  This
    decorator rehydrates proposed IDs from current active store rows, enforces
    the caller's thresholds/limits and typed semantic-conflict gates, and
    applies the closed automatic-memory protocol.  Backend score/order remains
    usable by a future vector implementation.
    """

    def __init__(self, backend: FeedbackRetriever, store: FeedbackStore) -> None:
        if isinstance(backend, AuthoritativeFeedbackRetriever):
            # Avoid nested adapters while preserving the canonical store chosen
            # by the consumer that owns this boundary.
            backend = backend._backend
        self._backend = backend
        self._store = store

    def retrieve(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
        correction_limit: int = 3,
        rule_limit: int = 3,
        exclude_feedback_id: str = "",
        minimum_correction_score: float = 0.42,
    ) -> FeedbackRetrievalResult:
        correction_limit = max(0, int(correction_limit))
        rule_limit = max(0, int(rule_limit))
        retrieval_kwargs = {
            "source_language": source_language,
            "target_language": target_language,
            "correction_limit": correction_limit,
            "rule_limit": rule_limit,
            "exclude_feedback_id": exclude_feedback_id,
            "minimum_correction_score": minimum_correction_score,
        }
        try:
            raw = self._backend.retrieve(text, **retrieval_kwargs)
        except (FeedbackRetrievalUnavailable, TimeoutError, ConnectionError) as exc:
            # Semantic/vector retrieval is an optional quality layer.  A
            # recoverable outage must not suppress exact local corrections or
            # prevent the base translation from running.  Programming/schema
            # errors deliberately remain visible instead of being swallowed.
            get_logger().warning(
                "Feedback retrieval backend unavailable; using lexical fallback | "
                "backend=%s error=%s",
                type(self._backend).__name__,
                exc,
            )
            fallback = LexicalFeedbackRetriever(self._store).retrieve(
                text,
                **retrieval_kwargs,
            )
            raw = FeedbackRetrievalResult(
                corrections=fallback.corrections,
                rules=fallback.rules,
                correction_scores=fallback.correction_scores,
                correction_exact_matches=fallback.correction_exact_matches,
                backend=f"{fallback.backend}-fallback",
            )
        if not isinstance(raw, FeedbackRetrievalResult):
            raise ValueError("feedback retriever must return FeedbackRetrievalResult")

        # Backend score/order is retrieval evidence (and lets a future vector
        # backend find non-lexical paraphrases).  Object state and identity are
        # canonical-store authority.  The typed semantic compatibility gate
        # rejects known contradictions without imposing a lexical threshold.
        canonical_corrections = {
            item.id: item for item in self._store.list_feedback()
        }

        corrections: list[FeedbackRecord] = []
        scores: list[float] = []
        exact_matches: list[bool] = []
        seen_corrections: set[str] = set()
        if correction_limit:
            # Exact identity is deterministic authority, not a retrieval
            # opinion.  Prepend it even when a future backend omits it or fills
            # its own top-k with semantic neighbors.
            exact_hits = self._store.rank_correction_matches(
                text,
                source_language=source_language,
                target_language=target_language,
                limit=correction_limit,
                exclude_feedback_id=exclude_feedback_id,
                minimum_score=max(1.0, minimum_correction_score),
            ) if minimum_correction_score <= 1.0 else []
            for hit in exact_hits:
                seen_corrections.add(hit.record.id)
                corrections.append(hit.record)
                scores.append(1.0)
                exact_matches.append(True)

            for proposed, backend_score in zip(
                raw.corrections,
                raw.correction_scores,
                strict=True,
            ):
                if len(corrections) >= correction_limit:
                    break
                record = canonical_corrections.get(proposed.id)
                if record is None or proposed.id in seen_corrections:
                    continue
                if (
                    record.id == exclude_feedback_id
                    or record.status not in {"accepted", "confirmed"}
                    or not record.enabled
                    or not record.corrected_translation.strip()
                    or record.source_language != source_language
                    or record.target_language != target_language
                    or backend_score < minimum_correction_score
                    or not self._store.semantic_sources_compatible(
                        text,
                        record.ocr_text,
                    )
                ):
                    continue
                seen_corrections.add(proposed.id)
                corrections.append(record)
                scores.append(backend_score)
                exact_matches.append(False)

        # Retrieval decides relevance; the local store decides authority.
        # Requiring another literal trigger match here would silently turn a
        # future vector/hybrid backend back into lexical retrieval.
        canonical_rules = {
            item.id: item
            for item in (
                self._store.active_memory_rules(
                    source_language=source_language,
                    target_language=target_language,
                )
                if rule_limit
                else []
            )
        }
        rules: list[MemoryRule] = []
        seen_rules: set[str] = set()
        for proposed in raw.rules:
            rule = canonical_rules.get(proposed.id)
            if rule is None or rule.id in seen_rules:
                continue
            evidence_ids = list(rule.source_feedback_ids or []) or [
                rule.source_feedback_id
            ]
            evidence = [
                canonical_corrections[feedback_id]
                for feedback_id in evidence_ids
                if feedback_id in canonical_corrections
            ]
            literal_match = self._store.trigger_field_matches_source(
                rule.trigger, text
            )
            if not literal_match:
                # A semantic backend may broaden retrieval only when the
                # canonical rule has durable correction evidence that can be
                # checked against the current source.  Evidence-less legacy
                # rules remain available through their explicit trigger, but
                # can never become free-floating vector instructions.
                if not evidence or not any(
                    self._store.semantic_sources_compatible(text, item.ocr_text)
                    for item in evidence
                ):
                    continue
            # Manual memory must have a durable owner link.  Unlocked automatic
            # memory must use the local typed protocol; legacy free text remains
            # on disk but is intentionally inert until a user reviews/locks it.
            if rule.origin == "manual" and not rule.source_feedback_id.strip():
                continue
            if rule.origin == "automatic" and not rule.user_locked:
                if not is_canonical_automatic_rule(rule):
                    continue
            elif not rule.user_locked and rule.origin != "manual":
                continue
            seen_rules.add(rule.id)
            rules.append(rule)
            if len(rules) >= rule_limit:
                break

        return FeedbackRetrievalResult(
            corrections=tuple(corrections),
            rules=tuple(rules),
            correction_scores=tuple(scores),
            correction_exact_matches=tuple(exact_matches),
            backend=raw.backend,
        )


__all__ = [
    "FeedbackRetrievalUnavailable",
    "FeedbackRetrievalResult",
    "FeedbackRetriever",
    "AuthoritativeFeedbackRetriever",
    "LexicalFeedbackRetriever",
]
