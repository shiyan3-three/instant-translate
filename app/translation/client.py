"""OpenAI-compatible HTTP client for translation requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import httpx

ThinkingMode = Union[Literal["enabled", "disabled"], bool, None]


@dataclass
class ClientConfig:
    """Connection parameters for an OpenAI-compatible endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 30.0
    max_tokens: int = 4096

    @property
    def chat_url(self) -> str:
        """Return the full chat completions endpoint."""

        base = self.base_url.rstrip("/")
        return f"{base}/chat/completions"

    @property
    def models_url(self) -> str:
        """Return the full models list endpoint."""

        base = self.base_url.rstrip("/")
        return f"{base}/models"


class TranslationError(RuntimeError):
    """Raised when a translation request fails for any reason."""


class OpenAICompatibleClient:
    """Send translation requests to an OpenAI-compatible endpoint.

    Uses httpx for HTTP transport with configurable timeouts.
    Designed for short, single-message translation requests —
    not for streaming or multi-turn conversations.
    """

    def __init__(self, config: ClientConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], thinking: ThinkingMode = None) -> str:
        """Send a full messages array and return the response text.

        Raises TranslationError on timeout, HTTP error, or empty response.
        """

        import time as _time
        from app.logger import get_debug_logger

        if not messages:
            raise TranslationError("Cannot send empty messages.")

        payload = {
            "model": self._config.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self._config.max_tokens,
        }
        self._apply_thinking(payload, thinking)

        headers = self._build_headers()

        t0 = _time.perf_counter()
        try:
            response = httpx.post(
                url=self._config.chat_url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            get_debug_logger().warning("API timeout after %.0fs", self._config.timeout_seconds)
            raise TranslationError(f"Translation request timed out after {self._config.timeout_seconds:.0f}s.")
        except httpx.HTTPStatusError as exc:
            get_debug_logger().warning("API HTTP %d", exc.response.status_code)
            raise TranslationError(f"API error {exc.response.status_code}.") from exc
        except httpx.RequestError as exc:
            get_debug_logger().warning("API network error: %s", exc)
            raise TranslationError(f"Network error reaching {self._config.base_url}: {exc}")

        http_ms = (_time.perf_counter() - t0) * 1000
        get_debug_logger().debug("API HTTP round-trip: %.0fms", http_ms)

        return self._parse_completion_response(response)

    def complete(self, system_prompt: str, user_prompt: str, thinking: ThinkingMode = None) -> str:
        """Send one chat completion request and return the response text.

        Raises TranslationError on timeout, HTTP error, or empty response.
        """

        import time as _time
        from app.logger import get_logger, get_debug_logger

        if not user_prompt.strip():
            raise TranslationError("Cannot send an empty prompt.")

        payload = self._build_payload(system_prompt, user_prompt)
        self._apply_thinking(payload, thinking)
        headers = self._build_headers()

        t0 = _time.perf_counter()
        try:
            response = httpx.post(
                url=self._config.chat_url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            get_logger().warning("API 超时 (%.0fs)", self._config.timeout_seconds)
            get_debug_logger().warning("API timeout after %.0fs", self._config.timeout_seconds)
            raise TranslationError(
                f"Translation request timed out after "
                f"{self._config.timeout_seconds:.0f}s."
            )
        except httpx.HTTPStatusError as exc:
            get_logger().warning("API HTTP 错误 %d", exc.response.status_code)
            get_debug_logger().warning("API HTTP %d", exc.response.status_code)
            raise TranslationError(f"API error {exc.response.status_code}.") from exc
        except httpx.RequestError as exc:
            get_logger().warning("API 网络错误: %s", exc)
            get_debug_logger().warning("API network error: %s", exc)
            raise TranslationError(
                f"Network error reaching {self._config.base_url}: {exc}"
            )

        http_ms = (_time.perf_counter() - t0) * 1000
        get_debug_logger().debug("API HTTP round-trip: %.0fms", http_ms)

        return self._parse_completion_response(response)

    def translate(self, system_prompt: str, source_text: str) -> str:
        """Send one translation request and return the response text."""

        if not source_text.strip():
            raise TranslationError("Cannot translate empty text.")
        return self.complete(system_prompt, source_text)

    def list_models(self) -> list[str]:
        """Fetch available model IDs from the OpenAI-compatible ``/models`` endpoint.

        Returns a sorted, de-duplicated list of model id strings.
        Raises TranslationError on timeout, HTTP error, or unexpected payload.
        """

        from app.logger import get_debug_logger

        headers = self._build_headers()
        try:
            response = httpx.get(
                url=self._config.models_url,
                headers=headers,
                timeout=min(self._config.timeout_seconds, 30.0),
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            get_debug_logger().warning("List models timeout")
            raise TranslationError(
                f"List models timed out after {min(self._config.timeout_seconds, 30.0):.0f}s."
            )
        except httpx.HTTPStatusError as exc:
            get_debug_logger().warning("List models HTTP %d", exc.response.status_code)
            raise TranslationError(f"API error {exc.response.status_code}.") from exc
        except httpx.RequestError as exc:
            get_debug_logger().warning("List models network error: %s", exc)
            raise TranslationError(
                f"Network error reaching {self._config.base_url}: {exc}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise TranslationError(f"Models response is not JSON: {exc}") from exc

        raw_items = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(raw_items, list):
            raise TranslationError("Unexpected models response format — missing data list.")

        models: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            if isinstance(item, str):
                model_id = item.strip()
            elif isinstance(item, dict):
                model_id = str(item.get("id") or item.get("name") or "").strip()
            else:
                continue
            if not model_id:
                continue
            key = model_id.casefold()
            if key in seen:
                continue
            seen.add(key)
            models.append(model_id)

        models.sort(key=str.casefold)
        get_debug_logger().debug("Listed %d models from %s", len(models), self._config.models_url)
        return models

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _build_payload(self, system_prompt: str, source_text: str) -> dict:
        model = self._config.model
        payload: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source_text},
            ],
            "temperature": 0.0,
            "max_tokens": self._config.max_tokens,
        }

        return payload

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _apply_thinking(payload: dict, thinking: ThinkingMode) -> None:
        """Apply DeepSeek-style thinking controls to a request payload.

        ``None`` omits the provider-specific field for compatibility.
        ``True``/``False`` are accepted for older callers, but new code
        should pass the explicit string modes.
        """

        if thinking is None:
            return
        if thinking is True:
            mode = "enabled"
        elif thinking is False:
            mode = "disabled"
        elif thinking in ("enabled", "disabled"):
            mode = thinking
        else:
            raise ValueError("thinking must be 'enabled', 'disabled', True, False, or None")
        payload["thinking"] = {"type": mode}

    def _parse_completion_response(self, response: httpx.Response) -> str:
        """Validate an OpenAI completion response before recording diagnostics.

        Completion providers vary in how they report malformed replies.  This
        boundary keeps every structural failure in the public
        :class:`TranslationError` vocabulary and avoids echoing a provider's
        response body into a user-facing message or log.
        """

        try:
            response_json = response.json()
        except (ValueError, TypeError) as exc:
            raise TranslationError("API response is not valid JSON.") from exc

        if not isinstance(response_json, dict):
            raise TranslationError("Unexpected API response format: expected a JSON object.")
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TranslationError("Unexpected API response format: missing non-empty choices.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise TranslationError("Unexpected API response format: invalid choices entry.")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise TranslationError("Unexpected API response format: missing message.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise TranslationError("API returned an empty translation.")

        # The payload is fully validated before diagnostics are emitted.  Log
        # only schema metadata and lengths, never text content.
        from app.logger import get_debug_logger

        finish_reason = choice.get("finish_reason")
        get_debug_logger().debug(
            "API response structure: message_keys=%s content_len=%d reasoning_len=%d finish_reason=%r",
            sorted(str(key) for key in message.keys()),
            len(content),
            len(message.get("reasoning_content") or "")
            if isinstance(message.get("reasoning_content"), str)
            else 0,
            finish_reason,
        )
        if finish_reason == "length":
            raise TranslationError(
                "API response was truncated before the translation completed; please retry with a larger token limit."
            )
        return content.strip()

    @staticmethod
    def _extract_content(response_json: dict) -> str:
        """Compatibility helper for callers that already own a JSON object."""

        choices = response_json.get("choices") if isinstance(response_json, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise TranslationError("Unexpected API response format: missing non-empty choices.")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise TranslationError("API returned an empty translation.")
        if choices[0].get("finish_reason") == "length":
            raise TranslationError(
                "API response was truncated before the translation completed; please retry with a larger token limit."
            )
        return content.strip()

    @staticmethod
    def _truncate(text: str, max_len: int = 200) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."
