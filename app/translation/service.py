"""Translation service — bridge OCR, compiled prompts, and API calls."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from app.agent.agent import TranslationAgent
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
    ) -> None:
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        self._groups: dict[int, GroupContext] = {}
        self._agent: TranslationAgent | None = None
        self._agent_lock = threading.Lock()

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

    def reset_agent(self) -> None:
        """Destroy the Agent session so a new one is created on next translation."""

        with self._agent_lock:
            self._agent = None

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
            agent = self._ensure_agent(request.source_language, request.target_language)
            text = agent.translate(request.ocr_text)

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

    def _ensure_agent(self, source: str = "English", target: str = "中文") -> TranslationAgent:
        with self._agent_lock:
            if self._agent is not None:
                return self._agent
            ai = self._settings.ai
            self._agent = TranslationAgent(
                ClientConfig(
                    base_url=ai.base_url,
                    api_key=ai.api_key,
                    model=ai.model,
                )
            )
            prompt = self._current_prompt(source, target)
            self._agent.digest_rules(prompt)
            return self._agent

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
                model=ai.model,
            )
        )

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

        try:
            from app.logger import get_debug_logger

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
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip markdown structure, keep content
            if stripped.startswith("## Runtime Direction"):
                break
            if stripped.startswith("This layer"):
                continue
            if stripped.startswith("#") or stripped.startswith("##"):
                continue
            lines.append(stripped)

        compact = " ".join(lines).strip()
        return compact if compact else DEFAULT_BASE_PROMPT

    def _ensure_group(self, group_id: int) -> GroupContext:
        if group_id not in self._groups:
            self._groups[group_id] = GroupContext()
        return self._groups[group_id]

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
