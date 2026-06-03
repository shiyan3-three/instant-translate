"""AI-assisted prompt rule optimization."""

from __future__ import annotations

from collections.abc import Callable

from app.prompt.compiler import PromptCompiler
from app.prompt.models import PromptConstraints, PromptKnowledgeReference
from app.settings import AppSettings
from app.translation.client import ClientConfig, OpenAICompatibleClient, TranslationError


class PromptOptimizationError(RuntimeError):
    """Raised when the prompt optimizer cannot produce rules."""


class PromptOptimizer:
    """Use an OpenAI-compatible endpoint to optimize user prompt layers."""

    def __init__(
        self,
        compiler: PromptCompiler | None = None,
        client_factory: Callable[[ClientConfig], OpenAICompatibleClient] | None = None,
    ) -> None:
        self._compiler = compiler or PromptCompiler()
        self._client_factory = client_factory or OpenAICompatibleClient

    def optimize(
        self,
        settings: AppSettings,
        constraints: PromptConstraints,
        references: list[PromptKnowledgeReference] | None = None,
    ) -> str:
        """Return optimized user-layer rules using the configured AI endpoint."""

        ai = settings.ai
        if not ai.base_url.strip() or not ai.api_key.strip() or not ai.model.strip():
            raise PromptOptimizationError(
                "请先填写 Base URL、API Key 和 Model，再生成 compiled prompt。"
            )

        system_prompt, user_prompt = self._compiler.build_optimizer_messages(
            constraints,
            references or [],
        )
        client = self._client_factory(
            ClientConfig(
                base_url=ai.base_url,
                api_key=ai.api_key,
                model=ai.model,
            )
        )

        try:
            optimized = client.complete(system_prompt, user_prompt).strip()
        except TranslationError as exc:
            raise PromptOptimizationError(str(exc)) from exc

        if not optimized:
            raise PromptOptimizationError("AI 未返回可用的 Prompt 优化结果。")
        return optimized
