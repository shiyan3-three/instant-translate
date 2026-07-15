"""Feedback and local translation-memory helpers.

The package intentionally avoids eager imports.  Translation modules import
``app.feedback.memory_policy`` during their own bootstrap, so importing the
learning/optimizer stack here would recreate a runtime import cycle.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.feedback.learning import FeedbackLearningService
    from app.feedback.retrieval import (
        AuthoritativeFeedbackRetriever,
        FeedbackRetrievalResult,
        FeedbackRetriever,
        FeedbackRetrievalUnavailable,
        LexicalFeedbackRetriever,
    )
    from app.feedback.store import (
        CorrectionMatch,
        FeedbackRecord,
        FeedbackStorageUnavailable,
        FeedbackStore,
        MemoryRule,
        StaleConsolidationAborted,
    )

__all__ = [
    "AuthoritativeFeedbackRetriever",
    "CorrectionMatch",
    "FeedbackLearningService",
    "FeedbackRecord",
    "FeedbackRetrievalUnavailable",
    "FeedbackStorageUnavailable",
    "FeedbackRetrievalResult",
    "FeedbackRetriever",
    "FeedbackStore",
    "LexicalFeedbackRetriever",
    "MemoryRule",
    "StaleConsolidationAborted",
]


_EXPORT_MODULES = {
    "AuthoritativeFeedbackRetriever": "app.feedback.retrieval",
    "CorrectionMatch": "app.feedback.store",
    "FeedbackLearningService": "app.feedback.learning",
    "FeedbackRecord": "app.feedback.store",
    "FeedbackRetrievalUnavailable": "app.feedback.retrieval",
    "FeedbackStorageUnavailable": "app.feedback.store",
    "FeedbackRetrievalResult": "app.feedback.retrieval",
    "FeedbackRetriever": "app.feedback.retrieval",
    "FeedbackStore": "app.feedback.store",
    "LexicalFeedbackRetriever": "app.feedback.retrieval",
    "MemoryRule": "app.feedback.store",
    "StaleConsolidationAborted": "app.feedback.store",
}


def __getattr__(name: str):
    """Load public helpers on demand without reintroducing import cycles."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
