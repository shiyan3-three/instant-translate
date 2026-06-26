"""AI-assisted conversion from user feedback to reusable memory rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.feedback.store import FeedbackRecord
from app.settings import AppSettings
from app.translation.client import ClientConfig, OpenAICompatibleClient, TranslationError


@dataclass
class FeedbackOptimization:
    """Proposed improvement for one feedback item."""

    trigger: str
    rule: str
    improved_translation: str = ""
    trigger_options: list[str] = field(default_factory=list)


class FeedbackOptimizer:
    """Ask the thinking model to draft a reusable local memory rule."""

    def optimize(self, settings: AppSettings, record: FeedbackRecord) -> FeedbackOptimization:
        ai = settings.ai
        model = ai.thinking_model_name
        if not ai.base_url or not ai.api_key or not model:
            raise TranslationError("请先配置 Base URL、API Key 和 Thinking Model。")

        client = OpenAICompatibleClient(
            ClientConfig(
                base_url=ai.base_url,
                api_key=ai.api_key,
                model=model,
                timeout_seconds=30,
            )
        )
        system_prompt = (
            "You improve a desktop OCR translation system. "
            "Return JSON only, no markdown. "
            "Fields: trigger, trigger_options, rule, improved_translation. "
            "trigger_options must contain 1 to 3 concise candidate keywords or phrases "
            "that are semantically meaningful and reusable, not arbitrary sentence truncations. "
            "trigger must be the best option from trigger_options. "
            "The rule must be short, reusable, and written in Chinese. "
            "improved_translation may be empty if the user did not provide enough information."
        )
        user_prompt = (
            f"Source language: {record.source_language}\n"
            f"Target language: {record.target_language}\n"
            f"OCR source text:\n{record.ocr_text}\n\n"
            f"Current translation:\n{record.translation_text}\n\n"
            f"User note:\n{record.note or '(none)'}\n\n"
            f"User corrected translation:\n{record.corrected_translation or '(none)'}\n\n"
            "Draft one reusable memory rule that should improve similar future translations. "
            "Think about which source terms should trigger the rule, then output only the JSON object."
        )
        raw = client.complete(system_prompt, user_prompt, thinking="enabled").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise TranslationError("AI 未返回可解析的关键词 JSON。")
        trigger_options = self._parse_trigger_options(data.get("trigger_options"))
        trigger = str(data.get("trigger") or "").strip()
        if not trigger and trigger_options:
            trigger = trigger_options[0]
        if trigger and trigger not in trigger_options:
            trigger_options.insert(0, trigger)
        trigger_options = trigger_options[:3]
        if not trigger_options:
            raise TranslationError("AI 未返回关键词候选。")
        return FeedbackOptimization(
            trigger=trigger,
            trigger_options=trigger_options,
            rule=str(data.get("rule") or ""),
            improved_translation=str(data.get("improved_translation") or record.corrected_translation),
        )

    @staticmethod
    def _parse_trigger_options(value) -> list[str]:
        """Normalize model-proposed trigger candidates."""

        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            raw_items = [part.strip() for part in value.replace("，", ",").split(",")]
        else:
            raw_items = []

        options: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            text = str(item).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            options.append(text)
            if len(options) >= 3:
                break
        return options
