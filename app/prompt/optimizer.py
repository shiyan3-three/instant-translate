"""AI-assisted prompt rule optimization."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from app.prompt.compiler import PromptCompiler
from app.prompt.models import PromptConstraints, PromptKnowledgeReference
from app.prompt.policy import ConstraintPolicy, OptimizedPrompt
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
    ) -> OptimizedPrompt:
        """Return optimized user-layer rules using the configured AI endpoint."""

        ai = settings.ai
        model = ai.thinking_model_name
        if not ai.base_url.strip() or not ai.api_key.strip() or not model.strip():
            raise PromptOptimizationError(
                "请先填写 Base URL、API Key 和 Thinking Model，再生成 compiled prompt。"
            )

        system_prompt, user_prompt = self._compiler.build_optimizer_messages(
            constraints,
            references or [],
        )
        client = self._client_factory(
            ClientConfig(
                base_url=ai.base_url,
                api_key=ai.api_key,
                model=model,
                timeout_seconds=600.0,
                max_tokens=16384,
            )
        )

        try:
            optimized = client.complete(
                system_prompt,
                user_prompt,
                thinking="enabled",
            ).strip()
        except TranslationError as exc:
            raise PromptOptimizationError(str(exc)) from exc

        if not optimized:
            raise PromptOptimizationError("AI 未返回可用的 Prompt 优化结果。")
        return self._parse_result(optimized)

    @staticmethod
    def _parse_result(raw: str) -> OptimizedPrompt:
        """Accept the new JSON contract while remaining compatible with old providers."""

        candidate = raw.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return OptimizedPrompt(raw)
        if not isinstance(payload, dict):
            return OptimizedPrompt(raw)
        supplemental = payload.get("supplemental_rules", "")
        if isinstance(supplemental, list):
            supplemental = "\n".join(str(item).strip() for item in supplemental if str(item).strip())
        elif not isinstance(supplemental, str):
            supplemental = ""
        supplemental = supplemental.strip()
        policy = ConstraintPolicy.from_dict(payload.get("constraint_policy", {}), strict=False)
        user_summary = payload.get("user_summary", "")
        if not isinstance(user_summary, str):
            user_summary = ""
        user_summary = user_summary.strip()[:800]
        if user_summary and not re.search(r"[\u3400-\u9fff]", user_summary):
            user_summary = ""
        raw_items = payload.get("change_items", [])
        change_items: list[dict[str, str]] = []
        if isinstance(raw_items, list):
            for item in raw_items[:8]:
                if not isinstance(item, dict):
                    continue
                title = item.get("title", "")
                description = item.get("description", "")
                if not isinstance(title, str) or not isinstance(description, str):
                    continue
                title = title.strip()[:40]
                description = description.strip()[:240]
                if (
                    title and description
                    and re.search(r"[\u3400-\u9fff]", title)
                    and re.search(r"[\u3400-\u9fff]", description)
                ):
                    change_items.append({"title": title, "description": description})
        return OptimizedPrompt(supplemental, policy, user_summary, change_items)
