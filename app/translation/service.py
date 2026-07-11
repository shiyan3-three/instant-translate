"""Translation service — bridge OCR, compiled prompts, and API calls."""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.agent import StaleRequestAborted, TranslationAgent
from app.agent.session_store import AgentProfileMeta, AgentSessionMeta, AgentSessionStore
from app.feedback.store import FeedbackStore
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
    source_language: str = "English"
    target_language: str = "中文"


@dataclass
class TranslationResult:
    """Response from a completed translation request."""

    group_id: int
    request_id: int
    text: str | None
    error: str | None

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

    def __init__(
        self,
        settings: AppSettings,
        max_workers: int = 3,
        max_concurrent_api_calls: int = 2,
        session_store: AgentSessionStore | None = None,
        feedback_store: FeedbackStore | None = None,
    ) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
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

        with self._lock:
            ctx = self._ensure_group(group_id)

            if self._is_similar(clean, ctx.last_committed_text) or self._is_similar(
                clean, ctx.pending_text
            ):
                get_debug_logger().debug(
                    "[G%d] Translation request skipped as duplicate: %r",
                    group_id,
                    clean[:80],
                )
                return None

            ctx.current_request_id += 1
            ctx.pending_text = clean
            request = TranslationRequest(
                group_id=group_id,
                request_id=ctx.current_request_id,
                ocr_text=clean,
                source_language=source_language,
                target_language=target_language,
            )

        future: Future = self._executor.submit(
            self._execute, request, on_result
        )
        # Avoids "future unused" warnings while still letting the pool
        # own lifecycle.
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
        if delete_persisted:
            self._session_store.delete(group_id)

    def shutdown(self) -> None:
        """Shut down the background thread pool (best-effort, no wait)."""

        self._executor.shutdown(wait=False)

    def prepare_agent_profile(
        self,
        source_language: str,
        target_language: str,
    ) -> Future | None:
        """Prepare the visible Pro bootstrap in the background before OCR exists."""

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

    def _execute(self, request: TranslationRequest, on_result) -> None:
        with self._lock:
            ctx = self._ensure_group(request.group_id)
            if self._is_stale_request(request, ctx):
                return

        try:
            group_lock = self._agent_lock_for(request.group_id)
            with group_lock:
                with self._lock:
                    ctx = self._ensure_group(request.group_id)
                    if self._is_stale_request(request, ctx):
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
                        with self._lock:
                            ctx = self._ensure_group(request.group_id)
                            if self._is_stale_request(request, ctx):
                                get_debug_logger().debug(
                                    "[G%d] Translation skipped after Agent prepare: stale request=%d current=%d",
                                    request.group_id,
                                    request.request_id,
                                    ctx.current_request_id,
                                )
                                return
                        memory_hints = self._memory_hints_for_request(request)
                        reference_context = self._reference_context_for_request(request)
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

                with self._lock:
                    ctx = self._ensure_group(request.group_id)
                    if self._is_stale_request(request, ctx):
                        get_debug_logger().debug(
                            "[G%d] Translation completed but was not committed: stale request=%d current=%d",
                            request.group_id,
                            request.request_id,
                            ctx.current_request_id,
                        )
                        return
                    agent.record_translation(request.ocr_text, text)
                    ctx.last_committed_text = request.ocr_text
                    ctx.pending_text = ""
                    ctx.last_translation = text
                self._save_agent_session(request.group_id, agent)

            self._invoke_callback(
                on_result,
                TranslationResult(
                    group_id=request.group_id,
                    request_id=request.request_id,
                    text=text,
                    error=None,
                )
            )
        except StaleRequestAborted:
            get_debug_logger().debug(
                "[G%d] Translation aborted before thinking retry: stale request=%d",
                request.group_id,
                request.request_id,
            )
            return
        except TranslationError as exc:
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
                )
            )
        except Exception as exc:
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

    def _is_request_cancelled(self, request: TranslationRequest) -> bool:
        """Check whether *request* has been superseded (thread-safe).

        Used as a cancellation callback for ``TranslationAgent.translate``
        so a stale request can abort before a slow thinking retry.
        """

        with self._lock:
            ctx = self._ensure_group(request.group_id)
            return self._is_stale_request(request, ctx)

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
        rules = self._feedback_store.match_memory_rules(
            request.ocr_text,
            source_language=request.source_language,
            target_language=request.target_language,
            limit=3,
        )
        if not rules:
            return []
        triggers = ", ".join(rule.trigger for rule in rules)
        get_logger().info(
            "[G%d] Translation memory hit | %s",
            request.group_id,
            triggers,
        )
        get_debug_logger().debug(
            "[G%d] Translation memory hit | ids=%s",
            request.group_id,
            ",".join(rule.id for rule in rules),
        )
        return [rule.as_prompt_hint() for rule in rules]

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

        try:
            get_debug_logger().debug(
                "Prompt source=%s length=%d path=%s",
                prompt_source,
                len(base_prompt),
                compiled_path or "",
            )
        except Exception:
            pass
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
