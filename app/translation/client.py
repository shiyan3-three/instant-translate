"""OpenAI-compatible HTTP client for translation requests."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class ClientConfig:
    """Connection parameters for an OpenAI-compatible endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 30.0

    @property
    def chat_url(self) -> str:
        """Return the full chat completions endpoint."""

        base = self.base_url.rstrip("/")
        return f"{base}/chat/completions"


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

    def chat(self, messages: list[dict], thinking: bool = False) -> str:
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
            "max_tokens": 1024,
        }
        if thinking:
            payload["thinking"] = {"type": "enabled"}

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
            raise TranslationError(f"API error {exc.response.status_code}: {self._truncate(str(exc.response.text))}")
        except httpx.RequestError as exc:
            get_debug_logger().warning("API network error: %s", exc)
            raise TranslationError(f"Network error reaching {self._config.base_url}: {exc}")

        http_ms = (_time.perf_counter() - t0) * 1000
        get_debug_logger().debug("API HTTP round-trip: %.0fms", http_ms)

        response_json = response.json()
        
        # Detailed diagnostic: dump full response structure (first call only)
        import json
        msg = response_json.get("choices", [{}])[0].get("message", {})
        get_debug_logger().debug(
            "API response structure: message_keys=%s",
            list(msg.keys())
        )
        if "reasoning_content" in msg:
            get_debug_logger().debug(
                "reasoning_content preview: %s",
                (msg["reasoning_content"][:200] if msg["reasoning_content"] else "(empty)")
            )
        if "content" in msg:
            get_debug_logger().debug(
                "content preview: %s",
                (msg["content"][:200] if msg["content"] else "(empty)")
            )
        
        result = self._extract_content(response_json)
        
        if not result.strip():
            raise TranslationError("API returned an empty translation.")
        return result.strip()

    def complete(self, system_prompt: str, user_prompt: str, thinking: bool = False) -> str:
        """Send one chat completion request and return the response text.

        Raises TranslationError on timeout, HTTP error, or empty response.
        """

        import time as _time
        from app.logger import get_logger, get_debug_logger

        if not user_prompt.strip():
            raise TranslationError("Cannot send an empty prompt.")

        payload = self._build_payload(system_prompt, user_prompt)
        if thinking:
            payload["thinking"] = {"type": "enabled"}
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
            raise TranslationError(
                f"API error {exc.response.status_code}: "
                f"{self._truncate(str(exc.response.text))}"
            )
        except httpx.RequestError as exc:
            get_logger().warning("API 网络错误: %s", exc)
            get_debug_logger().warning("API network error: %s", exc)
            raise TranslationError(
                f"Network error reaching {self._config.base_url}: {exc}"
            )

        http_ms = (_time.perf_counter() - t0) * 1000
        get_debug_logger().debug("API HTTP round-trip: %.0fms", http_ms)

        result = self._extract_content(response.json())
        if not result.strip():
            raise TranslationError("API returned an empty translation.")

        return result.strip()

    def translate(self, system_prompt: str, source_text: str) -> str:
        """Send one translation request and return the response text."""

        if not source_text.strip():
            raise TranslationError("Cannot translate empty text.")
        return self.complete(system_prompt, source_text)

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
            "max_tokens": 4096,
        }

        # DeepSeek v4 reasoning models consume max_tokens for both
        # thinking and output.  "low" keeps reasoning short so more
        # budget is left for the actual translation content.
        # Only add this for DeepSeek reasoning models to avoid breaking
        # other providers that don't recognise this parameter.
        is_deepseek_reasoning = (
            "deepseek" in self._config.base_url.lower()
            and "v4" in model.lower()
        )
        if is_deepseek_reasoning:
            payload["reasoning_effort"] = "low"

        return payload

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_content(response_json: dict) -> str:
        try:
            msg = response_json["choices"][0]["message"]
            content = msg.get("content") or ""
            
            # If content is empty but reasoning_content exists (DeepSeek thinking mode),
            # the actual translation might be in reasoning_content
            if not content.strip() and "reasoning_content" in msg:
                content = msg.get("reasoning_content") or ""
            
            return content
        except (KeyError, IndexError, TypeError):
            raise TranslationError(
                "Unexpected API response format — missing choices[0].message.content."
            )

    @staticmethod
    def _truncate(text: str, max_len: int = 200) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."
