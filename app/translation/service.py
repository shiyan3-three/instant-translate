"""Translation service — bridge OCR, compiled prompts, and API calls."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from app.agent.agent import TranslationAgent
from app.agent.session_store import AgentSessionMeta, AgentSessionStore
from app.feedback.store import FeedbackStore
from app.logger import get_debug_logger, get_logger
from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.prompt.storage import PromptStorage
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

    last_submitted_text: str = ""
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

    _SIMILARITY_THRESHOLD = 0.92

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
        self._api_slots = threading.BoundedSemaphore(
            max(1, min(max_workers, max_concurrent_api_calls))
        )
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

            if self._is_similar(clean, ctx.last_submitted_text):
                get_debug_logger().debug(
                    "[G%d] Translation request skipped as duplicate: %r",
                    group_id,
                    clean[:80],
                )
                return None

            ctx.current_request_id += 1
            ctx.last_submitted_text = clean
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
        future.add_done_callback(lambda f: f.exception() if f.exception() else None)

        return request

    def reset_group(self, group_id: int) -> None:
        """Clear cached text signatures for one group."""

        with self._lock:
            self._groups.pop(group_id, None)

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

    def reset_agent(self, group_id: int) -> None:
        """Destroy the Agent session for one group so a new one is created on next translation."""

        group_lock = self._agent_lock_for(group_id)
        with group_lock:
            with self._agent_lock:
                self._agents.pop(group_id, None)
                self._agent_meta.pop(group_id, None)

    def shutdown(self) -> None:
        """Shut down the background thread pool (best-effort, no wait)."""

        self._executor.shutdown(wait=False)

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
                        memory_hints = self._memory_hints_for_request(request)
                        text = agent.translate(request.ocr_text, memory_hints=memory_hints)
                    finally:
                        get_debug_logger().debug("[G%d] API slot released", request.group_id)
                self._save_agent_session(request.group_id, agent)

            with self._lock:
                ctx = self._ensure_group(request.group_id)
                if self._is_stale_request(request, ctx):
                    return
                ctx.last_translation = text

            on_result(
                TranslationResult(
                    group_id=request.group_id,
                    request_id=request.request_id,
                    text=text,
                    error=None,
                )
            )
        except TranslationError as exc:
            on_result(
                TranslationResult(
                    group_id=request.group_id,
                    request_id=request.request_id,
                    text=None,
                    error=str(exc),
                )
            )

    def _ensure_agent(self, group_id: int, source: str = "English", target: str = "中文") -> TranslationAgent:
        group_lock = self._agent_lock_for(group_id)
        with group_lock:
            with self._agent_lock:
                existing = self._agents.get(group_id)
            if existing is not None:
                return existing
            ai = self._settings.ai
            fast_model = ai.fast_model_name
            thinking_model = ai.thinking_model_name or fast_model
            prompt = self._current_prompt(source, target)
            prompt_hash = self._prompt_hash(prompt)
            session_meta = AgentSessionMeta(
                group_id=group_id,
                source_language=source,
                target_language=target,
                prompt_hash=prompt_hash,
                fast_model=fast_model,
                thinking_model=thinking_model,
            )
            agent = TranslationAgent(
                ClientConfig(
                    base_url=ai.base_url,
                    api_key=ai.api_key,
                    model=fast_model,
                ),
                ClientConfig(
                    base_url=ai.base_url,
                    api_key=ai.api_key,
                    model=thinking_model,
                )
            )
            saved_messages = self._session_store.load(session_meta)
            if saved_messages:
                get_logger().info(
                    "[G%d] Agent session hit | messages=%d prompt=%s fast=%s thinking=%s",
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
                agent.restore_messages(saved_messages)
            else:
                get_logger().info(
                    "[G%d] Agent session miss | digesting rules prompt=%s fast=%s thinking=%s",
                    group_id,
                    prompt_hash[:12],
                    fast_model,
                    thinking_model,
                )
                agent.digest_rules(prompt)
                self._session_store.save(session_meta, getattr(agent, "messages", []))
                get_logger().info(
                    "[G%d] Agent session saved | messages=%d prompt=%s",
                    group_id,
                    len(getattr(agent, "messages", [])),
                    prompt_hash[:12],
                )
            with self._agent_lock:
                self._agents[group_id] = agent
                self._agent_meta[group_id] = AgentRuntimeMeta(
                    source_language=source,
                    target_language=target,
                    prompt_hash=prompt_hash,
                    fast_model=fast_model,
                    thinking_model=thinking_model,
                )
            return agent

    def _save_agent_session(self, group_id: int, agent: TranslationAgent) -> None:
        with self._agent_lock:
            meta = self._agent_meta.get(group_id)
        if meta is None:
            return
        self._session_store.save(
            AgentSessionMeta(
                group_id=group_id,
                source_language=meta.source_language,
                target_language=meta.target_language,
                prompt_hash=meta.prompt_hash,
                fast_model=meta.fast_model,
                thinking_model=meta.thinking_model,
            ),
            agent.messages,
        )
        get_logger().info(
            "[G%d] Agent session saved after translation | messages=%d prompt=%s",
            group_id,
            len(agent.messages),
            meta.prompt_hash[:12],
        )
        get_debug_logger().debug(
            "[G%d] Agent session saved after translation | messages=%d prompt=%s",
            group_id,
            len(agent.messages),
            meta.prompt_hash[:12],
        )

    def _agent_lock_for(self, group_id: int) -> threading.RLock:
        """Return the stable per-group lock used for Agent session mutation."""

        with self._agent_lock:
            lock = self._group_agent_locks.get(group_id)
            if lock is None:
                lock = threading.RLock()
                self._group_agent_locks[group_id] = lock
            return lock

    @staticmethod
    def _is_stale_request(request: TranslationRequest, ctx: GroupContext) -> bool:
        """Return whether a completed request has been superseded."""

        return request.request_id != ctx.current_request_id

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
        if compiled:
            base_prompt = self._compact_compiled_prompt(compiled)
        else:
            base_prompt = DEFAULT_BASE_PROMPT
        base_prompt = self._add_runtime_quality_clarifications(base_prompt, source, target)

        try:
            get_debug_logger().debug(
                "Prompt source=%s length=%d path=%s",
                "compiled" if compiled else "default",
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
    def _compact_compiled_prompt(raw: str) -> str:
        """Extract essential translation rules from the compiled prompt markdown.

        Strips the verbose human-readable structure and keeps only the
        directives that matter to the translator model.
        """

        lines: list[str] = []
        in_ai_optimization_layer = False
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip markdown structure, keep content
            if stripped.startswith("## Runtime Direction"):
                break
            if stripped.startswith("## AI Optimization Layer"):
                in_ai_optimization_layer = True
                continue
            if stripped.startswith("## ") or stripped.startswith("#"):
                in_ai_optimization_layer = False
                continue
            if TranslationService._is_harmful_hiragana_optimizer_line(stripped):
                continue
            if in_ai_optimization_layer and TranslationService._is_harmful_optimizer_example_line(stripped):
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

        if not previous_text:
            return False
        if new_text == previous_text:
            return True

        # Simple Jaccard-like trigram similarity for short strings.
        def trigrams(s: str) -> set[str]:
            s = s.lower()
            if len(s) <= 3:
                return {s}
            return {s[i:i + 3] for i in range(len(s) - 2)}

        a = trigrams(new_text)
        b = trigrams(previous_text)
        if not a or not b:
            return False

        overlap = len(a & b) / min(len(a), len(b))
        return overlap >= TranslationService._SIMILARITY_THRESHOLD
