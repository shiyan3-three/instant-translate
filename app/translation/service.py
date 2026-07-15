"""Translation service — bridge OCR, compiled prompts, and API calls."""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.agent import StaleRequestAborted, TranslationAgent
from app.agent.session_store import AgentProfileMeta, AgentSessionMeta, AgentSessionStore
from app.feedback.memory_policy import memory_rule_prompt_hint
from app.feedback.store import FeedbackStorageUnavailable, FeedbackStore
from app.feedback.retrieval import (
    AuthoritativeFeedbackRetriever,
    FeedbackRetriever,
    LexicalFeedbackRetriever,
)
from app.logger import get_debug_logger, get_logger
from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.prompt.policy import ConstraintPolicy, ConstraintPolicyCompiler
from app.prompt.storage import PromptStorage
from app.reference_layer import ReferenceEntry, ReferencePackage, ReferencePlan, ReferenceStore
from app.settings import AppSettings
from app.translation.client import ClientConfig, OpenAICompatibleClient, TranslationError


@dataclass
class TranslationRequest:
    """Captured state at the moment a translation is requested."""

    group_id: int
    request_id: int
    ocr_text: str
    # ``-1`` is retained as a compatibility sentinel for tests/integrations
    # that construct a request directly.  Requests created by
    # ``request_translation`` always capture the real revision.
    feedback_revision: int = -1
    revision_retry_count: int = 0
    source_language: str = "English"
    target_language: str = "中文"
    matched_memory_rule_ids: list[str] = field(default_factory=list)
    matched_correction_ids: list[str] = field(default_factory=list)
    matched_memory_hint_snapshots: dict[str, str] = field(default_factory=dict)
    matched_correction_hint_snapshots: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CachedTranslation:
    """One exact OCR variant cached for a short period within one group."""

    translation_text: str
    source_language: str
    target_language: str
    correction_ids: tuple[str, ...]
    memory_rule_ids: tuple[str, ...]
    correction_hint_snapshots: tuple[tuple[str, str], ...]
    memory_hint_snapshots: tuple[tuple[str, str], ...]
    feedback_revision: int
    stored_at: float


@dataclass
class TranslationResult:
    """Response from a completed translation request."""

    group_id: int
    request_id: int
    text: str | None
    error: str | None
    source: str = "api"
    feedback_revision: int = -1

    @property
    def from_cache(self) -> bool:
        """Compatibility flag for callers that only need cache provenance."""

        return self.source == "cache"

    @property
    def is_stale(self) -> bool:
        """Placeholder — staleness is checked against the service's version tracker."""

        return False


@dataclass
class GroupContext:
    """Mutable per-group state managed by TranslationService."""

    last_committed_text: str = ""
    last_translation: str = ""
    current_request_id: int = 0
    pending_text: str = ""
    last_memory_rule_ids: list[str] = field(default_factory=list)
    last_correction_ids: list[str] = field(default_factory=list)
    last_memory_hint_snapshots: dict[str, str] = field(default_factory=dict)
    last_correction_hint_snapshots: dict[str, str] = field(default_factory=dict)
    last_source_language: str = ""
    last_target_language: str = ""
    last_request_id: int = 0
    last_feedback_revision: int = 0
    translation_variants: OrderedDict[str, CachedTranslation] = field(
        default_factory=OrderedDict
    )
    cache_scope: tuple[object, ...] = ()


@dataclass(frozen=True)
class TranslationMemorySnapshot:
    """Committed translation and the feedback knowledge injected for it."""

    ocr_text: str = ""
    translation_text: str = ""
    correction_ids: tuple[str, ...] = ()
    memory_rule_ids: tuple[str, ...] = ()
    correction_hint_snapshots: tuple[tuple[str, str], ...] = ()
    memory_hint_snapshots: tuple[tuple[str, str], ...] = ()
    source_language: str = ""
    target_language: str = ""
    request_id: int = 0


@dataclass
class AgentRuntimeMeta:
    """Runtime metadata for persisting one group's Agent session."""

    source_language: str
    target_language: str
    prompt_hash: str
    fast_model: str
    thinking_model: str


@dataclass(frozen=True)
class AgentDefinition:
    """Resolved prompt, local policy, and stable profile identity."""

    prompt: str
    policy: ConstraintPolicy
    reference_package: ReferencePackage
    reference_entries: tuple[ReferenceEntry, ...]
    runtime_meta: AgentRuntimeMeta
    profile_meta: AgentProfileMeta


@dataclass(frozen=True)
class ReferenceRequestContext:
    """Per-request reference match shared by hints and placeholder protection."""

    hints: list[str]
    plan: ReferencePlan


class TranslationService:
    """Bridge OCR text, compiled prompts, and API calls.

    Responsibilities
    ----------------
    * Text deduplication — skip OCR results that are identical or
      near-identical to the last submitted text.
    * Versioned requests — every translation is tagged with a
      monotonically increasing request id so stale responses can be
      discarded.
    * Background execution — API calls run on a thread-pool, leaving
      the Qt main loop free.

    Thread-safety
    -------------
    Group state is protected by a re-entrant lock.  The callback
    passed to ``request_translation`` is always invoked from a
    background thread — the caller is responsible for marshalling the
    result back to the Qt main thread (e.g. via ``QMetaObject.invokeMethod``
    or a queued signal connection).
    """

    API_BACKOFF_FAILURE_THRESHOLD = 3
    API_BACKOFF_COOLDOWN_SECONDS = 30.0
    MAX_MEMORY_HINT_CHARS = 2400
    OCR_VARIANT_CACHE_TTL_SECONDS = 8.0
    OCR_VARIANT_CACHE_CAPACITY = 8
    FEEDBACK_REVISION_RETRY_LIMIT = 1

    def __init__(
        self,
        settings: AppSettings,
        max_workers: int = 3,
        max_concurrent_api_calls: int = 2,
        session_store: AgentSessionStore | None = None,
        feedback_store: FeedbackStore | None = None,
        feedback_retriever: FeedbackRetriever | None = None,
    ) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        self._closing = False
        self._groups: dict[int, GroupContext] = {}
        self._agents: dict[int, TranslationAgent] = {}
        self._agent_meta: dict[int, AgentRuntimeMeta] = {}
        # Registry operations are short and global; network work is guarded by
        # one lock per group so independent selection groups can translate in
        # parallel without mutating the same Agent session concurrently.
        self._agent_lock = threading.RLock()
        self._group_agent_locks: dict[int, threading.RLock] = {}
        self._profile_lock = threading.RLock()
        self._profile_futures: dict[str, Future] = {}
        self._prepared_profiles: dict[str, list[dict]] = {}
        self._api_slots = threading.BoundedSemaphore(
            max(1, min(max_workers, max_concurrent_api_calls))
        )
        self._api_failure_count = 0
        self._api_backoff_until = 0.0
        self._time_fn = time.monotonic
        self._session_store = session_store or AgentSessionStore()
        self._feedback_store = feedback_store or FeedbackStore()
        feedback_backend = feedback_retriever or LexicalFeedbackRetriever(
            self._feedback_store,
        )
        self._feedback_retriever = AuthoritativeFeedbackRetriever(
            feedback_backend,
            self._feedback_store,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def request_translation(
        self,
        group_id: int,
        ocr_text: str,
        on_result,
        source_language: str = "English",
        target_language: str = "中文",
    ) -> TranslationRequest | None:
        """Submit a translation to the background pool if the text is new enough.

        Returns the ``TranslationRequest`` when a job was actually queued,
        or ``None`` when the text was skipped (duplicate / empty).

        ``on_result(TranslationResult)`` is called from a **background
        thread** once the API call completes (or fails).
        """

        clean = ocr_text.strip()
        if not clean:
            return None

        feedback_revision = self._current_feedback_revision()
        cached_request: TranslationRequest | None = None
        cached_result: TranslationResult | None = None
        future: Future | None = None
        with self._lock:
            if self._closing:
                return None
            ctx = self._ensure_group(group_id)
            self._refresh_translation_cache_scope(
                ctx,
                source_language=source_language,
                target_language=target_language,
                feedback_revision=feedback_revision,
            )

            if self._is_similar(clean, ctx.last_committed_text) or self._is_similar(
                clean, ctx.pending_text
            ):
                get_debug_logger().debug(
                    "[G%d] Translation request skipped as duplicate: len=%d sha256=%s",
                    group_id,
                    len(clean),
                    hashlib.sha256(clean.encode("utf-8")).hexdigest()[:12],
                )
                return None

            self._expire_translation_variants(ctx)
            cached = ctx.translation_variants.get(clean)
            if cached is not None and cached.feedback_revision != feedback_revision:
                # The scope and the revision are read independently.  If a
                # feedback write lands between those reads, refuse the old
                # variant rather than restoring it for the new token.
                cached = None
            if cached is not None:
                # A cached variant is a new logical generation.  This makes a
                # still-running B request stale before A is restored, while
                # keeping the callback/result path identical to a real call.
                ctx.translation_variants.move_to_end(clean)
                ctx.current_request_id += 1
                ctx.pending_text = ""
                ctx.last_committed_text = clean
                ctx.last_translation = cached.translation_text
                ctx.last_memory_rule_ids = list(cached.memory_rule_ids)
                ctx.last_correction_ids = list(cached.correction_ids)
                ctx.last_memory_hint_snapshots = dict(cached.memory_hint_snapshots)
                ctx.last_correction_hint_snapshots = dict(cached.correction_hint_snapshots)
                ctx.last_source_language = cached.source_language
                ctx.last_target_language = cached.target_language
                ctx.last_request_id = ctx.current_request_id
                ctx.last_feedback_revision = cached.feedback_revision
                cached_request = TranslationRequest(
                    group_id=group_id,
                    request_id=ctx.current_request_id,
                    ocr_text=clean,
                    feedback_revision=feedback_revision,
                    source_language=source_language,
                    target_language=target_language,
                )
                cached_result = TranslationResult(
                    group_id=group_id,
                    request_id=ctx.current_request_id,
                    text=cached.translation_text,
                    error=None,
                    source="cache",
                    feedback_revision=feedback_revision,
                )
            else:
                ctx.current_request_id += 1
                ctx.pending_text = clean
                request = TranslationRequest(
                    group_id=group_id,
                    request_id=ctx.current_request_id,
                    ocr_text=clean,
                    feedback_revision=feedback_revision,
                    source_language=source_language,
                    target_language=target_language,
                )

                try:
                    future = self._executor.submit(
                        self._execute, request, on_result
                    )
                except RuntimeError:
                    # shutdown() may race a final OCR callback.  The request was
                    # never queued, so clear its pending marker without surfacing
                    # an exception through a Qt/background boundary.
                    ctx.pending_text = ""
                    return None
        if cached_result is not None:
            with self._feedback_revision_guard(
                cached_request.feedback_revision
            ) as revision_current:
                if revision_current:
                    self._invoke_callback(on_result, cached_result)
                    return cached_request
            self._discard_revision_state(cached_request)
            self._requeue_after_feedback_revision(cached_request, on_result)
            return cached_request
        # Avoids "future unused" warnings while still letting the pool
        # own lifecycle.
        assert future is not None
        future.add_done_callback(self._log_worker_failure)

        return request

    def reset_group(self, group_id: int) -> None:
        """Clear cached text signatures for one group and invalidate in-flight requests.

        The request id is monotonically incremented so it never decreases
        or reuses a value within the service lifetime, preventing ABA
        races where a stale worker would mistake its id for the current one.
        """

        with self._lock:
            ctx = self._ensure_group(group_id)
            ctx.current_request_id += 1
            ctx.last_committed_text = ""
            ctx.last_translation = ""
            ctx.pending_text = ""
            ctx.last_memory_rule_ids = []
            ctx.last_correction_ids = []
            ctx.last_memory_hint_snapshots = {}
            ctx.last_correction_hint_snapshots = {}
            ctx.last_source_language = ""
            ctx.last_target_language = ""
            ctx.last_request_id = 0
            ctx.last_feedback_revision = 0
            ctx.translation_variants.clear()
            ctx.cache_scope = ()

    def memory_provenance(self, group_id: int) -> tuple[list[str], list[str]]:
        """Return the correction/rule ids actually injected for the last committed request."""

        feedback_revision = self._current_feedback_revision()
        with self._lock:
            ctx = self._ensure_group(group_id)
            if ctx.last_feedback_revision != feedback_revision:
                return [], []
            return list(ctx.last_correction_ids), list(ctx.last_memory_rule_ids)

    def memory_provenance_snapshot(self, group_id: int) -> TranslationMemorySnapshot:
        """Return one atomic snapshot for feedback marking and audit."""

        feedback_revision = self._current_feedback_revision()
        with self._lock:
            ctx = self._ensure_group(group_id)
            provenance_is_current = ctx.last_feedback_revision == feedback_revision
            return TranslationMemorySnapshot(
                ocr_text=ctx.last_committed_text,
                translation_text=ctx.last_translation,
                correction_ids=(
                    tuple(ctx.last_correction_ids) if provenance_is_current else ()
                ),
                memory_rule_ids=(
                    tuple(ctx.last_memory_rule_ids) if provenance_is_current else ()
                ),
                correction_hint_snapshots=tuple(
                    ctx.last_correction_hint_snapshots.items()
                    if provenance_is_current
                    else ()
                ),
                memory_hint_snapshots=(
                    tuple(ctx.last_memory_hint_snapshots.items())
                    if provenance_is_current
                    else ()
                ),
                source_language=ctx.last_source_language,
                target_language=ctx.last_target_language,
                request_id=ctx.last_request_id,
            )

    def invalidate_group_requests(self, group_id: int, reason: str = "") -> None:
        """Mark current in-flight work stale without destroying the Agent session."""

        with self._lock:
            ctx = self._ensure_group(group_id)
            ctx.current_request_id += 1
            ctx.pending_text = ""
        get_debug_logger().debug(
            "[G%d] Translation requests invalidated: %s",
            group_id,
            reason or "unspecified",
        )

    def reset_agent(self, group_id: int, *, delete_persisted: bool = False) -> None:
        """Destroy the Agent session for one group so a new one is created on next translation."""

        # Callers invalidate/reset request state before this method.  Do not
        # wait for a slow in-flight API call on the Qt thread; the stale worker
        # will be prevented from committing when it returns.
        with self._agent_lock:
            self._agents.pop(group_id, None)
            self._agent_meta.pop(group_id, None)
        with self._lock:
            ctx = self._groups.get(group_id)
            if ctx is not None:
                ctx.translation_variants.clear()
                # reset_agent is also a translation-context reset: the next
                # request must not be suppressed by the previous committed
                # text or by a cache scope from the old Agent session.
                ctx.cache_scope = ()
        if delete_persisted:
            self._session_store.delete(group_id)

    def shutdown(self) -> None:
        """Invalidate all work and cancel futures that have not started."""

        with self._lock:
            if self._closing:
                return
            self._closing = True
            for ctx in self._groups.values():
                ctx.current_request_id += 1
                ctx.pending_text = ""
        with self._profile_lock:
            profile_futures = list(self._profile_futures.values())
            self._profile_futures.clear()
        for future in profile_futures:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def prepare_agent_profile(
        self,
        source_language: str,
        target_language: str,
    ) -> Future | None:
        """Prepare the visible Pro bootstrap in the background before OCR exists."""

        with self._lock:
            if self._closing:
                return None
        ai = self._settings.ai
        if not (
            ai.base_url.strip()
            and ai.api_key.strip()
            and ai.fast_model_name.strip()
            and ai.thinking_model_name.strip()
        ):
            get_debug_logger().debug("Agent profile warmup skipped: incomplete AI configuration")
            return None

        definition = self._resolve_agent_definition(source_language, target_language)
        key = definition.profile_meta.key()
        with self._profile_lock:
            if key in self._prepared_profiles:
                return None
            existing = self._profile_futures.get(key)
            if existing is not None and not existing.cancelled():
                return existing
            get_logger().info(
                "Agent profile prepare scheduled | %s→%s prompt=%s fast=%s thinking=%s",
                source_language,
                target_language,
                definition.runtime_meta.prompt_hash[:12],
                definition.runtime_meta.fast_model,
                definition.runtime_meta.thinking_model,
            )
            future = self._executor.submit(self._prepare_profile, definition)
            self._profile_futures[key] = future
            future.add_done_callback(
                lambda completed, profile_key=key: self._on_profile_prepare_done(
                    profile_key,
                    completed,
                )
            )
            return future

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _translation_cache_scope(
        self,
        source_language: str,
        target_language: str,
        *,
        feedback_revision: int | None = None,
    ) -> tuple[object, ...]:
        """Return local configuration that changes translation meaning."""

        ai = self._settings.ai
        prompt = self._settings.prompt
        if feedback_revision is None:
            feedback_revision = self._current_feedback_revision()
        return (
            source_language,
            target_language,
            ai.base_url.strip(),
            ai.fast_model_name.strip(),
            ai.thinking_model_name.strip(),
            prompt.constraints_text,
            prompt.compiled_prompt_path,
            tuple(prompt.knowledge_reference_paths),
            feedback_revision,
        )

    def _refresh_translation_cache_scope(
        self,
        ctx: GroupContext,
        *,
        source_language: str,
        target_language: str,
        feedback_revision: int | None = None,
    ) -> None:
        scope = self._translation_cache_scope(
            source_language,
            target_language,
            feedback_revision=feedback_revision,
        )
        if ctx.cache_scope != scope and (
            ctx.cache_scope
            or ctx.translation_variants
            or ctx.last_committed_text
            or ctx.pending_text
        ):
            # Configuration changes must not let the previous model/prompt
            # suppress the first request under the new configuration.
            ctx.current_request_id += 1
            ctx.last_committed_text = ""
            ctx.last_translation = ""
            ctx.pending_text = ""
            ctx.last_memory_rule_ids = []
            ctx.last_correction_ids = []
            ctx.last_memory_hint_snapshots = {}
            ctx.last_correction_hint_snapshots = {}
            ctx.last_source_language = ""
            ctx.last_target_language = ""
            ctx.last_request_id = 0
            ctx.last_feedback_revision = 0
            ctx.translation_variants.clear()
        ctx.cache_scope = scope

    def _expire_translation_variants(self, ctx: GroupContext) -> None:
        cutoff = self._time_fn() - self.OCR_VARIANT_CACHE_TTL_SECONDS
        for text, cached in list(ctx.translation_variants.items()):
            if cached.stored_at <= cutoff:
                ctx.translation_variants.pop(text, None)

    def _cache_translation_variant(
        self,
        ctx: GroupContext,
        request: TranslationRequest,
        translated_text: str,
    ) -> None:
        if not translated_text.strip():
            return
        self._expire_translation_variants(ctx)
        ctx.translation_variants[request.ocr_text] = CachedTranslation(
            translation_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            correction_ids=tuple(request.matched_correction_ids),
            memory_rule_ids=tuple(request.matched_memory_rule_ids),
            correction_hint_snapshots=tuple(request.matched_correction_hint_snapshots.items()),
            memory_hint_snapshots=tuple(request.matched_memory_hint_snapshots.items()),
            feedback_revision=request.feedback_revision,
            stored_at=self._time_fn(),
        )
        ctx.translation_variants.move_to_end(request.ocr_text)
        while len(ctx.translation_variants) > self.OCR_VARIANT_CACHE_CAPACITY:
            ctx.translation_variants.popitem(last=False)

    def _execute(self, request: TranslationRequest, on_result) -> None:
        if request.feedback_revision < 0:
            # Keep direct callers/tests compatible while all public requests
            # still carry an explicit captured revision.
            request.feedback_revision = self._current_feedback_revision()
        stale, feedback_changed = self._request_stage_status(request)
        if stale:
            if feedback_changed:
                self._requeue_after_feedback_revision(request, on_result)
            return
        try:
            group_lock = self._agent_lock_for(request.group_id)
            with group_lock:
                stale, feedback_changed = self._request_stage_status(request)
                if stale:
                    if feedback_changed:
                        self._requeue_after_feedback_revision(request, on_result)
                    return

                get_debug_logger().debug("[G%d] Waiting for API slot", request.group_id)
                with self._api_slots:
                    get_debug_logger().debug("[G%d] API slot acquired", request.group_id)
                    try:
                        agent = self._ensure_agent(
                            request.group_id,
                            request.source_language,
                            request.target_language,
                        )
                        # Agent preparation may include a slow Pro bootstrap.
                        # Re-check after it finishes so an obsolete OCR request
                        # never spends another Flash call or reaches the UI.
                        stale, feedback_changed = self._request_stage_status(request)
                        if stale:
                            if feedback_changed:
                                self._requeue_after_feedback_revision(request, on_result)
                            return
                        try:
                            memory_hints = self._memory_hints_for_request(request)
                        except FeedbackStorageUnavailable as exc:
                            # Translation remains available, but partially
                            # recovered feedback must never enter the model.
                            get_logger().warning(
                                "[G%d] Feedback memory unavailable; translating without hints: %s",
                                request.group_id,
                                exc,
                            )
                            request.matched_memory_rule_ids = []
                            request.matched_correction_ids = []
                            request.matched_memory_hint_snapshots = {}
                            request.matched_correction_hint_snapshots = {}
                            memory_hints = []
                        # A future RAG backend may be slow.  Do not spend a
                        # Flash request after retrieval if OCR already moved on.
                        stale, feedback_changed = self._request_stage_status(request)
                        if stale:
                            if feedback_changed:
                                self._requeue_after_feedback_revision(request, on_result)
                            return
                        reference_context = self._reference_context_for_request(request)
                        stale, feedback_changed = self._request_stage_status(request)
                        if stale:
                            if feedback_changed:
                                self._requeue_after_feedback_revision(request, on_result)
                            return
                        translate_kwargs = {
                            "memory_hints": memory_hints,
                            "reference_hints": reference_context.hints,
                            "record": False,
                            "cancellation_check": lambda: self._is_request_cancelled(request),
                        }
                        if isinstance(agent, TranslationAgent):
                            translate_kwargs["reference_plan"] = reference_context.plan
                        text = agent.translate(
                            request.ocr_text,
                            **translate_kwargs,
                        )
                    finally:
                        get_debug_logger().debug("[G%d] API slot released", request.group_id)

                # This is the commit gate.  FeedbackStore holds its shared root
                # lock across the short local commit, session write, and success
                # callback so a durable feedback mutation cannot land between
                # a revision check and any of those side effects.
                stale = False
                feedback_changed = False
                agent_messages = self._snapshot_agent_messages(agent)
                with self._feedback_revision_guard(
                    request.feedback_revision
                ) as revision_current:
                    if not revision_current:
                        feedback_changed = True
                    else:
                        with self._lock:
                            ctx = self._ensure_group(request.group_id)
                            if self._is_stale_request(request, ctx):
                                stale = True
                            else:
                                agent.record_translation(request.ocr_text, text)
                                # A re-entrant test double can mutate feedback
                                # from record_translation even though external
                                # writers are blocked by the revision guard.
                                if not self._feedback_revision_is_current(request):
                                    self._restore_agent_messages(agent, agent_messages)
                                    feedback_changed = True
                                else:
                                    ctx.last_committed_text = request.ocr_text
                                    ctx.pending_text = ""
                                    ctx.last_translation = text
                                    ctx.last_memory_rule_ids = list(
                                        request.matched_memory_rule_ids
                                    )
                                    ctx.last_correction_ids = list(
                                        request.matched_correction_ids
                                    )
                                    ctx.last_memory_hint_snapshots = dict(
                                        request.matched_memory_hint_snapshots
                                    )
                                    ctx.last_correction_hint_snapshots = dict(
                                        request.matched_correction_hint_snapshots
                                    )
                                    ctx.last_source_language = request.source_language
                                    ctx.last_target_language = request.target_language
                                    ctx.last_request_id = request.request_id
                                    ctx.last_feedback_revision = request.feedback_revision
                                    self._cache_translation_variant(ctx, request, text)

                        if not stale and not feedback_changed:
                            self._save_agent_session(request.group_id, agent)
                            if not self._feedback_revision_is_current(request):
                                # Defensive support for re-entrant/custom
                                # session stores.  Production feedback writers
                                # cannot enter while the revision guard is held.
                                self._restore_agent_messages(agent, agent_messages)
                                self._session_store.delete(request.group_id)
                                feedback_changed = True
                            else:
                                self._invoke_callback(
                                    on_result,
                                    TranslationResult(
                                        group_id=request.group_id,
                                        request_id=request.request_id,
                                        text=text,
                                        error=None,
                                        source="api",
                                        feedback_revision=request.feedback_revision,
                                    ),
                                )
                if stale:
                    get_debug_logger().debug(
                        "[G%d] Translation completed but was not committed: stale request=%d",
                        request.group_id,
                        request.request_id,
                    )
                    return
                if feedback_changed:
                    self._discard_revision_state(request)
                    self._requeue_after_feedback_revision(request, on_result)
                    return
                return
        except StaleRequestAborted:
            if not self._feedback_revision_is_current(request):
                self._requeue_after_feedback_revision(request, on_result)
                return
            get_debug_logger().debug(
                "[G%d] Translation aborted before thinking retry: stale request=%d",
                request.group_id,
                request.request_id,
            )
            return
        except TranslationError as exc:
            if not self._feedback_revision_is_current(request):
                self._requeue_after_feedback_revision(request, on_result)
                return
            with self._lock:
                ctx = self._ensure_group(request.group_id)
                if self._is_stale_request(request, ctx):
                    return
                ctx.pending_text = ""
            self._invoke_callback(
                on_result,
                TranslationResult(
                    group_id=request.group_id,
                    request_id=request.request_id,
                    text=None,
                    error=str(exc),
                    source="api",
                    feedback_revision=request.feedback_revision,
                )
            )
        except Exception as exc:
            if not self._feedback_revision_is_current(request):
                self._requeue_after_feedback_revision(request, on_result)
                return
            get_debug_logger().exception(
                "[G%d] Unexpected translation worker failure", request.group_id
            )
            with self._lock:
                ctx = self._ensure_group(request.group_id)
                if self._is_stale_request(request, ctx):
                    return
                ctx.pending_text = ""
            self._invoke_callback(
                on_result,
                TranslationResult(
                    group_id=request.group_id,
                    request_id=request.request_id,
                    text=None,
                    error=f"Unexpected translation failure: {exc}",
                    source="api",
                    feedback_revision=request.feedback_revision,
                )
            )

    def _ensure_agent(self, group_id: int, source: str = "English", target: str = "中文") -> TranslationAgent:
        group_lock = self._agent_lock_for(group_id)
        with group_lock:
            definition = self._resolve_agent_definition(source, target)
            prompt = definition.prompt
            policy = definition.policy
            runtime_meta = definition.runtime_meta
            prompt_hash = runtime_meta.prompt_hash
            fast_model = runtime_meta.fast_model
            thinking_model = runtime_meta.thinking_model
            with self._agent_lock:
                existing = self._agents.get(group_id)
                existing_meta = self._agent_meta.get(group_id)
            if existing is not None and existing_meta == runtime_meta:
                existing.set_reference_entries(definition.reference_entries)
                return existing
            if existing is not None:
                get_logger().info(
                    "[G%d] Agent configuration changed; rebuilding session", group_id
                )
                with self._agent_lock:
                    self._agents.pop(group_id, None)
                    self._agent_meta.pop(group_id, None)
            session_meta = AgentSessionMeta(
                group_id=group_id,
                source_language=source,
                target_language=target,
                prompt_hash=prompt_hash,
                fast_model=fast_model,
                thinking_model=thinking_model,
            )
            saved_messages = self._session_store.load(session_meta)
            if saved_messages:
                get_logger().info(
                    "[G%d] Agent session hit | persisted_messages=%d prompt=%s fast=%s thinking=%s",
                    group_id,
                    len(saved_messages),
                    prompt_hash[:12],
                    fast_model,
                    thinking_model,
                )
                get_debug_logger().debug(
                    "[G%d] Restored Agent session metadata: source=%s target=%s prompt_hash=%s",
                    group_id,
                    source,
                    target,
                    prompt_hash,
                )
                agent = self._new_agent(runtime_meta)
                agent.restore_messages(saved_messages, policy=policy)
                agent.set_reference_entries(definition.reference_entries)
                self._save_profile_if_supported(definition.profile_meta, saved_messages)
            else:
                profile_messages = self._profile_messages_for(definition)
                agent = self._new_agent(runtime_meta)
                if profile_messages:
                    get_logger().info(
                        "[G%d] Agent session seeded from prepared profile | messages=%d prompt=%s",
                        group_id,
                        len(profile_messages),
                        prompt_hash[:12],
                    )
                    agent.restore_messages(profile_messages, policy=policy)
                    agent.set_reference_entries(definition.reference_entries)
                else:
                    get_logger().info(
                        "[G%d] Agent session miss | digesting rules prompt=%s fast=%s thinking=%s",
                        group_id,
                        prompt_hash[:12],
                        fast_model,
                        thinking_model,
                    )
                    agent.digest_rules(prompt, policy=policy)
                    agent.set_reference_entries(definition.reference_entries)
                    self._save_profile_if_supported(
                        definition.profile_meta,
                        getattr(agent, "messages", []),
                    )
                messages_to_persist = self._persistable_agent_messages(agent)
                self._session_store.save(session_meta, messages_to_persist)
                get_logger().info(
                    "[G%d] Agent session saved | persisted_messages=%d prompt=%s",
                    group_id,
                    len(messages_to_persist),
                    prompt_hash[:12],
                )
            with self._agent_lock:
                self._agents[group_id] = agent
                self._agent_meta[group_id] = runtime_meta
            return agent

    def _resolve_agent_definition(self, source: str, target: str) -> AgentDefinition:
        ai = self._settings.ai
        fast_model = ai.fast_model_name
        thinking_model = ai.thinking_model_name or fast_model
        prompt = self._current_prompt(source, target)
        policy = self._current_policy(source, target)
        reference_package = self._current_reference_package()
        reference_entries = reference_package.entries
        reference_digest = reference_package.digest()
        prompt_hash = self._prompt_hash(
            prompt + "\n" + policy.canonical_json() + "\n" + reference_digest
        )
        runtime_meta = AgentRuntimeMeta(
            source_language=source,
            target_language=target,
            prompt_hash=prompt_hash,
            fast_model=fast_model,
            thinking_model=thinking_model,
        )
        return AgentDefinition(
            prompt=prompt,
            policy=policy,
            reference_package=reference_package,
            reference_entries=reference_entries,
            runtime_meta=runtime_meta,
            profile_meta=AgentProfileMeta(
                source_language=source,
                target_language=target,
                prompt_hash=prompt_hash,
                fast_model=fast_model,
                thinking_model=thinking_model,
            ),
        )

    def _current_reference_entries(self) -> tuple[ReferenceEntry, ...]:
        """Load structured reference/glossary entries for per-request retrieval."""

        return self._current_reference_package().entries

    def _current_reference_package(self) -> ReferencePackage:
        """Load the compiled reference package, with legacy Markdown fallback."""

        prompt_storage = PromptStorage()
        configured_prompt_path = self._settings.prompt.compiled_prompt_path.strip()
        if configured_prompt_path:
            compiled_package = prompt_storage.load_compiled_references(configured_prompt_path)
            if not compiled_package.is_empty:
                get_debug_logger().debug(
                    "Reference package loaded entries=%d style=%d risk=%d digest=%s",
                    len(compiled_package.entries),
                    len(compiled_package.style_guidance),
                    len(compiled_package.risk_notes),
                    compiled_package.digest()[:12],
                )
                return compiled_package

        texts: list[str] = []
        for raw_path in self._settings.prompt.knowledge_reference_paths:
            try:
                path = Path(raw_path)
                texts.append(path.read_text(encoding="utf-8"))
            except OSError:
                get_debug_logger().warning("Knowledge reference load failed for reference layer: %s", raw_path)

        package = ReferencePackage.from_texts(texts)
        if not package.is_empty:
            get_debug_logger().debug(
                "Reference layer loaded from knowledge references entries=%d style=%d risk=%d digest=%s",
                len(package.entries),
                len(package.style_guidance),
                len(package.risk_notes),
                package.digest()[:12],
            )
        return package

    def _reference_hints_for_request(self, request: TranslationRequest) -> list[str]:
        return self._reference_context_for_request(request).hints

    def _reference_context_for_request(self, request: TranslationRequest) -> ReferenceRequestContext:
        package = self._current_reference_package()
        plan = ReferenceStore(package.entries).protect(request.ocr_text)
        hints = package.runtime_hints(
            request.ocr_text,
            matched_entries=plan.matched_entries,
        )
        return ReferenceRequestContext(hints=hints, plan=plan)

    def _new_agent(self, meta: AgentRuntimeMeta) -> TranslationAgent:
        ai = self._settings.ai
        agent = TranslationAgent(
            ClientConfig(
                base_url=ai.base_url,
                api_key=ai.api_key,
                model=meta.fast_model,
            ),
            ClientConfig(
                base_url=ai.base_url,
                api_key=ai.api_key,
                model=meta.thinking_model,
                timeout_seconds=120.0,
                max_tokens=8192,
            ),
        )
        agent.set_api_backoff_hooks(
            check=self._raise_if_api_backoff_active,
            outcome=self._record_api_outcome,
        )
        return agent

    def _raise_if_api_backoff_active(self) -> None:
        """Fail fast while the API endpoint is cooling down after repeated failures."""

        with self._lock:
            now = self._time_fn()
            if self._api_backoff_until and now < self._api_backoff_until:
                remaining = max(1, math.ceil(self._api_backoff_until - now))
                raise TranslationError(
                    f"API temporarily unavailable after repeated failures; retry in {remaining}s."
                )
            if self._api_backoff_until and now >= self._api_backoff_until:
                self._api_backoff_until = 0.0
                self._api_failure_count = 0

    def _record_api_outcome(self, ok: bool, reason: str = "") -> None:
        if ok:
            self._record_api_success()
        else:
            self._record_api_failure(reason)

    def _record_api_success(self) -> None:
        with self._lock:
            if self._api_failure_count or self._api_backoff_until:
                get_debug_logger().debug("API backoff reset after successful response")
            self._api_failure_count = 0
            self._api_backoff_until = 0.0

    def _record_api_failure(self, reason: str) -> None:
        if not self._is_transient_api_error(reason):
            return
        with self._lock:
            now = self._time_fn()
            if self._api_backoff_until and now < self._api_backoff_until:
                return
            self._api_failure_count += 1
            get_debug_logger().warning(
                "Transient API failure recorded count=%d reason=%s",
                self._api_failure_count,
                reason,
            )
            if self._api_failure_count >= self.API_BACKOFF_FAILURE_THRESHOLD:
                self._api_backoff_until = now + self.API_BACKOFF_COOLDOWN_SECONDS
                get_logger().warning(
                    "API 暂停 %.0fs：连续 %d 次临时失败。",
                    self.API_BACKOFF_COOLDOWN_SECONDS,
                    self._api_failure_count,
                )

    @staticmethod
    def _is_transient_api_error(reason: str) -> bool:
        lowered = str(reason or "").casefold()
        if not lowered:
            return False
        transient_markers = (
            "timed out",
            "timeout",
            "empty translation",
            "network error",
            "connection",
            "ssl",
            "temporarily unavailable",
        )
        if any(marker in lowered for marker in transient_markers):
            return True
        match = re.search(r"api error\s+(\d{3})", lowered)
        if not match:
            return False
        status = int(match.group(1))
        return status in {408, 409, 425, 429} or status >= 500

    def _prepare_profile(self, definition: AgentDefinition) -> list[dict]:
        meta = definition.profile_meta
        load_profile = getattr(self._session_store, "load_profile", None)
        messages = load_profile(meta) if callable(load_profile) else None
        if messages:
            get_logger().info(
                "Agent profile hit | %s→%s messages=%d prompt=%s",
                meta.source_language,
                meta.target_language,
                len(messages),
                meta.prompt_hash[:12],
            )
        else:
            get_logger().info(
                "Agent profile miss | digesting at application startup prompt=%s",
                meta.prompt_hash[:12],
            )
            agent = self._new_agent(definition.runtime_meta)
            agent.digest_rules(definition.prompt, policy=definition.policy)
            messages = [dict(message) for message in agent.messages[:3]]
            self._save_profile_if_supported(meta, messages)
            get_logger().info(
                "Agent profile ready | %s→%s messages=%d prompt=%s",
                meta.source_language,
                meta.target_language,
                len(messages),
                meta.prompt_hash[:12],
            )
        with self._profile_lock:
            self._prepared_profiles[meta.key()] = [dict(message) for message in messages]
        return [dict(message) for message in messages]

    def _profile_messages_for(self, definition: AgentDefinition) -> list[dict] | None:
        key = definition.profile_meta.key()
        with self._profile_lock:
            cached = self._prepared_profiles.get(key)
            future = self._profile_futures.get(key)
        if cached:
            return [dict(message) for message in cached]
        if future is not None:
            try:
                return [dict(message) for message in future.result()]
            except Exception:
                get_debug_logger().exception("Agent startup profile preparation failed; retrying in request")
                with self._profile_lock:
                    self._profile_futures.pop(key, None)
        try:
            return self._prepare_profile(definition)
        except Exception:
            get_debug_logger().exception("Agent profile preparation failed during translation")
            return None

    def _save_profile_if_supported(
        self,
        meta: AgentProfileMeta,
        messages: list[dict],
    ) -> None:
        save_profile = getattr(self._session_store, "save_profile", None)
        if not callable(save_profile):
            return
        try:
            save_profile(meta, messages)
        except OSError:
            get_debug_logger().exception("Agent profile persistence failed")

    def _on_profile_prepare_done(self, key: str, future: Future) -> None:
        try:
            future.result()
        except Exception:
            get_debug_logger().exception("Background Agent profile preparation failed")
        finally:
            with self._profile_lock:
                if self._profile_futures.get(key) is future and future.done():
                    # Prepared messages remain in _prepared_profiles; failed
                    # futures are removed so the next request can retry.
                    self._profile_futures.pop(key, None)

    def _save_agent_session(self, group_id: int, agent: TranslationAgent) -> None:
        with self._agent_lock:
            meta = self._agent_meta.get(group_id)
        if meta is None:
            return
        try:
            messages_to_persist = self._persistable_agent_messages(agent)
            self._session_store.save(
                AgentSessionMeta(
                    group_id=group_id,
                    source_language=meta.source_language,
                    target_language=meta.target_language,
                    prompt_hash=meta.prompt_hash,
                    fast_model=meta.fast_model,
                    thinking_model=meta.thinking_model,
                ),
                messages_to_persist,
            )
        except OSError:
            get_debug_logger().exception(
                "[G%d] Agent session persistence failed", group_id
            )
            return
        get_logger().info(
            "[G%d] Agent session saved after translation | persisted_messages=%d runtime_messages=%d prompt=%s",
            group_id,
            len(messages_to_persist),
            self._agent_runtime_message_count(agent),
            meta.prompt_hash[:12],
        )
        get_debug_logger().debug(
            "[G%d] Agent session saved after translation | persisted_messages=%d runtime_messages=%d prompt=%s",
            group_id,
            len(messages_to_persist),
            self._agent_runtime_message_count(agent),
            meta.prompt_hash[:12],
        )

    @staticmethod
    def _persistable_agent_messages(agent: TranslationAgent) -> list[dict]:
        persistable = getattr(agent, "persistable_messages", None)
        messages = getattr(agent, "messages", [])
        if callable(persistable) and hasattr(agent, "messages") and isinstance(messages, list):
            return persistable()
        return [dict(message) for message in messages[:3]] if isinstance(messages, list) else []

    @staticmethod
    def _agent_runtime_message_count(agent: TranslationAgent) -> int:
        messages = getattr(agent, "messages", [])
        return len(messages) if isinstance(messages, list) else 0

    @staticmethod
    def _snapshot_agent_messages(agent: TranslationAgent) -> list[object] | None:
        """Copy mutable Agent history before a revision-bound commit.

        Production feedback writers are serialized by the store guard.  The
        snapshot is a defensive rollback for custom/re-entrant implementations
        that mutate feedback from ``record_translation`` itself.
        """

        messages = getattr(agent, "messages", None)
        if not isinstance(messages, list):
            return None
        return [dict(message) if isinstance(message, dict) else message for message in messages]

    @staticmethod
    def _restore_agent_messages(
        agent: TranslationAgent,
        snapshot: list[object] | None,
    ) -> None:
        if snapshot is None:
            return
        messages = getattr(agent, "messages", None)
        if isinstance(messages, list):
            messages[:] = [
                dict(message) if isinstance(message, dict) else message
                for message in snapshot
            ]

    def _agent_lock_for(self, group_id: int) -> threading.RLock:
        """Return the stable per-group lock used for Agent session mutation."""

        with self._agent_lock:
            lock = self._group_agent_locks.get(group_id)
            if lock is None:
                lock = threading.RLock()
                self._group_agent_locks[group_id] = lock
            return lock

    @staticmethod
    def _log_worker_failure(future: Future) -> None:
        """Log executor failures that escaped normal result handling."""

        try:
            exc = future.exception()
        except Exception:
            get_debug_logger().exception("Unable to inspect translation future")
            return
        if exc is not None:
            get_debug_logger().error(
                "Unhandled translation future exception",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    @staticmethod
    def _invoke_callback(on_result, result: TranslationResult) -> None:
        """Deliver one result without letting UI callback errors rerun work."""

        try:
            on_result(result)
        except Exception:
            get_debug_logger().exception(
                "Translation result callback failed for group %d request %d",
                result.group_id,
                result.request_id,
            )

    @staticmethod
    def _is_stale_request(request: TranslationRequest, ctx: GroupContext) -> bool:
        """Return whether a completed request has been superseded."""

        return request.request_id != ctx.current_request_id

    def _current_feedback_revision(self) -> int:
        """Read the feedback knowledge token without holding service state."""

        revision = getattr(self._feedback_store, "knowledge_revision", 0)
        if callable(revision):
            revision = revision()
        if not isinstance(revision, int) or isinstance(revision, bool):
            return 0
        return revision

    def _feedback_revision_guard(self, expected_revision: int):
        """Return a context manager serializing a revision-bound local commit."""

        guard = getattr(self._feedback_store, "guard_knowledge_revision", None)
        if callable(guard):
            return guard(expected_revision)
        # Compatibility for lightweight test doubles and optional stores.  They
        # still get a final equality check, but cannot provide serialization.
        return nullcontext(self._current_feedback_revision() == expected_revision)

    def _feedback_revision_is_current(self, request: TranslationRequest) -> bool:
        """Return whether a request still describes the current feedback knowledge."""

        return request.feedback_revision == self._current_feedback_revision()

    def _request_stage_status(
        self,
        request: TranslationRequest,
    ) -> tuple[bool, bool]:
        """Return ``(stale, feedback_changed)`` at a safe stage boundary.

        The store revision is read before taking ``_lock``.  This keeps the
        service/store lock order one-way and makes every network-free stage
        check short.
        """

        feedback_revision = self._current_feedback_revision()
        with self._lock:
            ctx = self._ensure_group(request.group_id)
            if self._closing or self._is_stale_request(request, ctx):
                return True, False
        if request.feedback_revision != feedback_revision:
            return True, True
        return False, False

    def _clear_feedback_state_locked(self, ctx: GroupContext) -> None:
        """Clear current translation/provenance/cache state after knowledge changed."""

        ctx.last_committed_text = ""
        ctx.last_translation = ""
        ctx.pending_text = ""
        ctx.last_memory_rule_ids = []
        ctx.last_correction_ids = []
        ctx.last_memory_hint_snapshots = {}
        ctx.last_correction_hint_snapshots = {}
        ctx.last_source_language = ""
        ctx.last_target_language = ""
        ctx.last_request_id = 0
        ctx.last_feedback_revision = 0
        ctx.translation_variants.clear()
        ctx.cache_scope = ()

    def _discard_revision_state(self, request: TranslationRequest) -> None:
        """Remove a just-committed result whose feedback token became stale."""

        with self._lock:
            ctx = self._ensure_group(request.group_id)
            if self._is_stale_request(request, ctx):
                return
            self._clear_feedback_state_locked(ctx)

    def _requeue_after_feedback_revision(
        self,
        request: TranslationRequest,
        on_result,
    ) -> bool:
        """Re-submit one stale OCR request under the newest feedback token.

        This submits ``_execute`` directly instead of recursively calling the
        public API from a worker.  The retry count is carried by the logical
        request and is capped so repeated knowledge churn cannot loop forever.
        """

        feedback_revision = self._current_feedback_revision()
        if request.feedback_revision == feedback_revision:
            return False
        scope = self._translation_cache_scope(
            request.source_language,
            request.target_language,
            feedback_revision=feedback_revision,
        )
        future: Future | None = None
        churn_result: TranslationResult | None = None
        with self._lock:
            if self._closing:
                return False
            ctx = self._ensure_group(request.group_id)
            if self._is_stale_request(request, ctx):
                # reset_group, shutdown, or a newer OCR request owns the group.
                return False
            self._clear_feedback_state_locked(ctx)
            if request.revision_retry_count >= self.FEEDBACK_REVISION_RETRY_LIMIT:
                ctx.current_request_id += 1
                churn_result = TranslationResult(
                    group_id=request.group_id,
                    request_id=ctx.current_request_id,
                    text=None,
                    error="Feedback knowledge changed repeatedly; OCR refresh required",
                    source="feedback_revision_churn",
                    feedback_revision=feedback_revision,
                )
                get_debug_logger().warning(
                    "[G%d] Translation paused after feedback revision churn; requesting OCR refresh | retries=%d len=%d sha256=%s",
                    request.group_id,
                    request.revision_retry_count,
                    len(request.ocr_text),
                    hashlib.sha256(request.ocr_text.encode("utf-8")).hexdigest()[:12],
                )
            else:
                ctx.current_request_id += 1
                retry = TranslationRequest(
                    group_id=request.group_id,
                    request_id=ctx.current_request_id,
                    ocr_text=request.ocr_text,
                    feedback_revision=feedback_revision,
                    revision_retry_count=request.revision_retry_count + 1,
                    source_language=request.source_language,
                    target_language=request.target_language,
                )
                ctx.pending_text = retry.ocr_text
                ctx.cache_scope = scope
                try:
                    future = self._executor.submit(self._execute, retry, on_result)
                except RuntimeError:
                    ctx.pending_text = ""
                    return False
        if churn_result is not None:
            self._invoke_callback(on_result, churn_result)
            return False
        get_debug_logger().warning(
            "[G%d] Translation requeued after feedback revision change | retry=%d/%d len=%d sha256=%s",
            request.group_id,
            retry.revision_retry_count,
            self.FEEDBACK_REVISION_RETRY_LIMIT,
            len(request.ocr_text),
            hashlib.sha256(request.ocr_text.encode("utf-8")).hexdigest()[:12],
        )
        assert future is not None
        future.add_done_callback(self._log_worker_failure)
        return True

    def _is_request_cancelled(self, request: TranslationRequest) -> bool:
        """Check whether *request* has been superseded (thread-safe).

        Used as a cancellation callback for ``TranslationAgent.translate``
        so a stale request can abort before a slow thinking retry.
        """

        feedback_revision = self._current_feedback_revision()
        with self._lock:
            ctx = self._ensure_group(request.group_id)
            return (
                self._closing
                or self._is_stale_request(request, ctx)
                or request.feedback_revision != feedback_revision
            )

    def _build_client(self) -> OpenAICompatibleClient:
        ai = self._settings.ai
        return OpenAICompatibleClient(
            ClientConfig(
                base_url=ai.base_url,
                api_key=ai.api_key,
                model=ai.fast_model_name,
            )
        )

    def _memory_hints_for_request(self, request: TranslationRequest) -> list[str]:
        retrieval = self._feedback_retriever.retrieve(
            request.ocr_text,
            source_language=request.source_language,
            target_language=request.target_language,
            correction_limit=2,
            rule_limit=2,
            minimum_correction_score=0.62,
        )
        rules = list(retrieval.rules)
        correction_rows = list(zip(
            retrieval.corrections,
            retrieval.correction_scores,
            retrieval.correction_exact_matches,
            strict=True,
        ))
        score_by_correction_id = {
            correction.id: score for correction, score, _ in correction_rows
        }
        exact_by_correction_id = {
            correction.id: exact for correction, _, exact in correction_rows
        }
        exact_corrections = [
            correction for correction, _, exact in correction_rows if exact
        ][:1]
        exact_ids = {item.id for item in exact_corrections}
        # An exact user correction is stronger than an unlocked automatic rule
        # derived from that correction.  Keep the exact example and discard the
        # lower-authority summary, not the other way around.
        rules = [
            rule for rule in rules
            if not (
                rule.origin == "automatic"
                and not rule.user_locked
                and exact_ids.intersection(rule.source_feedback_ids or [])
            )
        ]
        rule_feedback_ids = {
            feedback_id
            for rule in rules
            for feedback_id in (rule.source_feedback_ids or [])
        }
        similar_corrections = [
            correction for correction, _, exact in correction_rows
            if not exact and correction.id not in rule_feedback_ids
        ]
        corrections = [*exact_corrections, *similar_corrections]
        if not rules and not corrections:
            return []
        triggers = ", ".join(rule.trigger for rule in rules) or "(correction examples only)"
        get_logger().info(
            "[G%d] Translation memory hit | rules=%s corrections=%d",
            request.group_id,
            triggers,
            len(corrections),
        )
        get_debug_logger().debug(
            "[G%d] Feedback retrieval backend=%s correction_scores=%s",
            request.group_id,
            retrieval.backend,
            ",".join(
                f"{item.id[:8]}:{score_by_correction_id[item.id]:.3f}"
                for item in corrections
            ),
        )
        get_debug_logger().debug(
            "[G%d] Translation memory hit | rule_ids=%s correction_ids=%s",
            request.group_id,
            ",".join(rule.id for rule in rules),
            ",".join(item.id for item in corrections),
        )
        candidates = [
            *((
                "correction",
                correction,
                self._feedback_store.correction_prompt_hint(
                    correction,
                    score=score_by_correction_id[correction.id],
                    exact_match=exact_by_correction_id[correction.id],
                ),
            ) for correction in exact_corrections),
            *(("rule", rule, memory_rule_prompt_hint(rule)) for rule in rules),
            *((
                "correction",
                correction,
                self._feedback_store.correction_prompt_hint(
                    correction,
                    score=score_by_correction_id[correction.id],
                    exact_match=False,
                ),
            ) for correction in similar_corrections),
        ]
        hints: list[str] = []
        used_rules = []
        used_corrections = []
        remaining = self.MAX_MEMORY_HINT_CHARS
        for kind, item, hint in candidates:
            if len(hints) >= 3:
                break
            clean = hint.strip()
            if not clean or remaining <= 0:
                continue
            if len(clean) > remaining:
                # Never turn one trusted statement into a misleading half-rule.
                # Individual fields are bounded by their renderers; if the
                # complete hint does not fit, omit it atomically.
                continue
            hints.append(clean)
            remaining -= len(clean)
            if kind == "rule":
                used_rules.append(item)
            else:
                used_corrections.append(item)
        request.matched_memory_rule_ids = [item.id for item in used_rules]
        request.matched_correction_ids = [item.id for item in used_corrections]
        used_rule_ids = set(request.matched_memory_rule_ids)
        used_correction_ids = set(request.matched_correction_ids)
        request.matched_memory_hint_snapshots = {
            item.id: clean
            for kind, item, hint in candidates
            if kind == "rule"
            and item.id in used_rule_ids
            and (clean := hint.strip())
        }
        request.matched_correction_hint_snapshots = {
            item.id: clean
            for kind, item, hint in candidates
            if kind == "correction"
            and item.id in used_correction_ids
            and (clean := hint.strip())
        }
        return hints

    def _current_prompt(self, source: str, target: str) -> str:
        """Return the prompt to use for translation requests.

        Prefer the user-confirmed compiled prompt on disk.  Fall back
        to the system-owned fixed template so translation remains
        usable even before prompt configuration.
        """

        prompt_storage = PromptStorage()
        configured_prompt_path = self._settings.prompt.compiled_prompt_path.strip()
        compiled_path = (
            prompt_storage.existing_compiled_prompt_path(configured_prompt_path)
            if configured_prompt_path
            else None
        )
        compiled = (
            prompt_storage.load_compiled_prompt(configured_prompt_path)
            if configured_prompt_path
            else ""
        )
        prompt_source = "compiled" if compiled else "default"
        if compiled and self._compiled_prompt_incompatible_with_pair(compiled, source, target):
            get_debug_logger().warning(
                "Compiled prompt ignored for incompatible language pair source=%s target=%s path=%s",
                source,
                target,
                compiled_path or "",
            )
            compiled = ""
            prompt_source = "default-incompatible-compiled"
        if compiled:
            base_prompt = self._compact_compiled_prompt(compiled)
        else:
            base_prompt = DEFAULT_BASE_PROMPT
        base_prompt = self._add_runtime_quality_clarifications(base_prompt, source, target)

        get_debug_logger().debug(
            "Prompt source=%s length=%d path=%s",
            prompt_source,
            len(base_prompt),
            compiled_path or "",
        )
        return (
            f"{base_prompt}\n\n"
            f"Translate from {source} to {target}."
        )

    @staticmethod
    def _compiled_prompt_incompatible_with_pair(raw: str, source: str, target: str) -> bool:
        """Return True when a language-specific compiled prompt would pollute this pair."""

        if TranslationService._target_is_japanese(target):
            return False
        lowered = raw.lower()
        japanese_specific_markers = (
            "hiragana",
            "katakana",
            "kanji",
            "japanese",
            "日本語",
            "日语",
            "日語",
            "平假名",
            "ひらがな",
            "カタカナ",
            "漢字",
        )
        return any(marker in lowered or marker in raw for marker in japanese_specific_markers)

    @staticmethod
    def _target_is_japanese(target: str) -> bool:
        normalized = target.strip().lower()
        return any(
            marker in normalized
            for marker in ("japanese", "日本語", "日语", "日語", "ja", "jp")
        )

    def _current_policy(self, source: str, target: str) -> ConstraintPolicy:
        """Load the safe sidecar, or rebuild it from confirmed user constraints."""

        configured_path = self._settings.prompt.compiled_prompt_path.strip()
        policy = ConstraintPolicy()
        policy_source = "empty"
        if configured_path:
            policy = PromptStorage().load_compiled_policy(configured_path)
            if policy.rules:
                policy_source = "sidecar"
        if not policy.rules:
            policy = ConstraintPolicyCompiler.compile(self._settings.prompt.constraints_text)
            if policy.rules:
                policy_source = "confirmed-user-constraints"
        active = policy.for_pair(source, target)
        get_debug_logger().debug(
            "Constraint policy source=%s pair=%s->%s rules=%s digest=%s",
            policy_source,
            source,
            target,
            [rule.type for rule in active.rules],
            active.digest()[:12],
        )
        return active

    @staticmethod
    def _compact_compiled_prompt(raw: str) -> str:
        """Extract essential translation rules from the compiled prompt markdown.

        Strips the verbose human-readable structure and keeps only the
        directives that matter to the translator model.
        """

        lines: list[str] = []
        in_ai_optimization_layer = False
        skipping_ai_glossary = False
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip markdown structure, keep content
            if stripped.startswith("## Runtime Direction"):
                break
            if stripped.startswith("## AI Optimization Layer"):
                in_ai_optimization_layer = True
                skipping_ai_glossary = False
                continue
            if stripped.startswith("## ") or stripped.startswith("#"):
                in_ai_optimization_layer = False
                skipping_ai_glossary = False
                continue
            if in_ai_optimization_layer and TranslationService._is_ai_optimizer_glossary_heading(stripped):
                skipping_ai_glossary = True
                continue
            if in_ai_optimization_layer and skipping_ai_glossary:
                if TranslationService._is_ai_optimizer_glossary_end(stripped):
                    skipping_ai_glossary = False
                else:
                    continue
            if TranslationService._is_harmful_hiragana_optimizer_line(stripped):
                continue
            if in_ai_optimization_layer and TranslationService._is_harmful_optimizer_example_line(stripped):
                continue
            if in_ai_optimization_layer and TranslationService._is_harmful_optimizer_term_line(stripped):
                continue
            if stripped.startswith("This layer"):
                continue
            lines.append(stripped)

        compact = " ".join(lines).strip()
        return compact if compact else DEFAULT_BASE_PROMPT

    @staticmethod
    def _is_harmful_hiragana_optimizer_line(line: str) -> bool:
        """Drop optimizer wording that can turn translation into source-character reading.

        The user constraint "日语句子只能由平假名构成" means the final Japanese
        output should be hiragana-only. It does not mean reading Chinese source
        characters as Japanese kanji. Older optimized prompts used wording such
        as "Convert all Chinese characters to hiragana equivalents", which
        causes outputs like 高考 -> こうこう instead of a semantic translation.
        """

        lowered = line.lower()
        harmful_markers = (
            "convert all chinese characters to hiragana",
            "chinese characters to hiragana equivalents",
            "all kanji must be converted to their corresponding hiragana readings",
        )
        return any(marker in lowered for marker in harmful_markers)

    @staticmethod
    def _is_ai_optimizer_glossary_heading(line: str) -> bool:
        lowered = line.lower()
        return (
            lowered.startswith("glossary entries")
            or lowered.startswith("terminology")
            or lowered.startswith("term list")
            or "术语表" in line
            or "用语表" in line
        )

    @staticmethod
    def _is_ai_optimizer_glossary_end(line: str) -> bool:
        lowered = line.lower()
        return (
            lowered.startswith("style guidance")
            or lowered.startswith("fixed expressions")
            or lowered.startswith("supplemental rules")
            or lowered.startswith("risk")
            or lowered.startswith("examples")
            or "风格" in line
            or "固定表达" in line
        )

    @staticmethod
    def _is_harmful_optimizer_term_line(line: str) -> bool:
        lowered = line.lower()
        if line.startswith("|"):
            return True
        harmful_markers = (
            "wrap every software engineering term",
            "bracketed content must also be in hiragana",
            "mandatory replacements",
            "for any other software engineering term",
            "generate its hiragana reading",
            "glossary table",
            "fixed term readings",
            "source=>target",
            "source -> target",
        )
        return any(marker in lowered for marker in harmful_markers)

    @staticmethod
    def _is_harmful_optimizer_example_line(line: str) -> bool:
        """Drop old optimizer examples that demonstrate reading conversion only."""

        if line.startswith("**Examples**"):
            return True
        return "→" in line and "(" in line and ")" in line

    @staticmethod
    def _add_runtime_quality_clarifications(base_prompt: str, source: str, target: str) -> str:
        """Clarify semantic and naturalness requirements for runtime translation."""

        if "日本" not in target and "japanese" not in target.lower():
            return base_prompt

        if "自然日语质量规则" in base_prompt:
            return base_prompt

        lowered = base_prompt.lower()
        hiragana_line = ""
        if "平假名" in base_prompt or "hiragana" in lowered:
            hiragana_line = (
                "平假名输出规则的含义：先把源文本按语义翻译成自然日语，再把最终日语译文写成平假名。"
                "不要把中文源文本的汉字逐字按日语音读/训读转换；例如“高考”应按“中国高考/大学入试”语义翻译，"
                "不能仅因字符是“高”“考”就输出“こうこう”。"
                "遇到技术词、产品名、网络社区词时，先确定日语里的自然说法；如果最终必须全平假名，"
                "再把自然说法写成平假名，例如“单元测试用例”可按“たんたいてすとけーす”处理。"
            )

        clarification = (
            "自然日语质量规则：优先保留原文含义和语气，使用日本人自然会说的表达；"
            "不要逐词硬译中文量词、结构词或网络短句。"
            "例如“一名女子”可译为“あるじょせい/ひとりのじょせい”，不要译成“いちにんのおんな”；"
            "“查分”应按“せいせきをかくにんする”之类语义翻译；"
            "中文网络语境里的“家”常指服务商、店铺、账号车队或团队，不要机械翻译成“いえ”；"
            "如果 OCR 文本残缺，只翻译可见内容，不要擅自补全。"
        )
        return f"{base_prompt}\n{hiragana_line}{clarification}"

    def _ensure_group(self, group_id: int) -> GroupContext:
        if group_id not in self._groups:
            self._groups[group_id] = GroupContext()
        return self._groups[group_id]

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_similar(new_text: str, previous_text: str) -> bool:
        """Return whether *new_text* is too close to *previous_text* to re-translate."""

        # OCR has already been normalised before reaching this service.  Only
        # exact repeats are safe to skip: a single negation, digit, or tense
        # marker can reverse the meaning while retaining almost every trigram.
        return bool(previous_text) and new_text == previous_text
