"""AI-assisted conversion from user feedback to reusable memory rules."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
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

    def __init__(
        self,
        client_factory: Callable[[ClientConfig], OpenAICompatibleClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or OpenAICompatibleClient

    def optimize(self, settings: AppSettings, record: FeedbackRecord) -> FeedbackOptimization:
        ai = settings.ai
        model = ai.thinking_model_name
        if not ai.base_url or not ai.api_key or not model:
            raise TranslationError("请先配置 Base URL、API Key 和 Thinking Model。")

        client = self._client_factory(
            ClientConfig(
                base_url=ai.base_url,
                api_key=ai.api_key,
                model=model,
                timeout_seconds=120,
                max_tokens=8192,
            )
        )
        system_prompt = (
            "You improve a desktop OCR translation system. "
            "Return JSON only, no markdown. "
            "Fields: trigger, trigger_options, rule, improved_translation. "
            "improved_translation should be a complete corrected target-language translation "
            "for this specific source sentence, obeying the user's output constraints. "
            "Only leave improved_translation empty if there is truly not enough information. "
            "Keyword memory is optional. Do not invent a keyword just to satisfy the schema. "
            "If the problem is only a sentence-level mistranslation, return trigger as an empty string "
            "and trigger_options as an empty list. "
            "trigger_options may be empty when the issue is sentence-specific and no reusable keyword is needed; "
            "otherwise include 1 to 3 concise candidate keywords or phrases that are semantically meaningful "
            "and reusable, not arbitrary sentence truncations. "
            "trigger must be the best option from trigger_options, or empty when trigger_options is empty. "
            "The rule must be short, reusable when possible, and written in Chinese. "
        )
        user_prompt = (
            f"Source language: {record.source_language}\n"
            f"Target language: {record.target_language}\n"
            f"OCR source text:\n{record.ocr_text}\n\n"
            f"Current translation:\n{record.translation_text}\n\n"
            f"User note:\n{record.note or '(none)'}\n\n"
            f"User corrected translation:\n{record.corrected_translation or '(none)'}\n\n"
            "First draft a corrected translation for the current sentence. "
            "Then decide whether a reusable keyword memory is useful. "
            "If no keyword is useful, leave trigger and trigger_options empty; the app can save it as an exact example. "
            "Output only the JSON object."
        )
        raw = client.complete(system_prompt, user_prompt, thinking="enabled").strip()
        try:
            data = self._parse_json_object(raw)
        except json.JSONDecodeError:
            raise TranslationError("AI 未返回可解析的优化 JSON。")
        if not isinstance(data, dict):
            raise TranslationError("AI 返回的优化 JSON 不是对象。")
        trigger_options = self._parse_trigger_options(data.get("trigger_options"))
        trigger = str(data.get("trigger") or "").strip()
        if not trigger and trigger_options:
            trigger = trigger_options[0]
        if trigger and trigger not in trigger_options:
            trigger_options.insert(0, trigger)
        trigger_options = trigger_options[:3]
        improved_translation = str(
            data.get("improved_translation") or record.corrected_translation
        ).strip()
        rule = str(data.get("rule") or "").strip()
        if not trigger_options and not improved_translation and not rule:
            raise TranslationError("AI 未返回优化译文、关键词或记忆规则。")
        return FeedbackOptimization(
            trigger=trigger,
            trigger_options=trigger_options,
            rule=rule,
            improved_translation=improved_translation,
        )

    @staticmethod
    def _parse_json_object(raw: str):
        candidate = raw.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            candidate = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                candidate,
                flags=re.I | re.S,
            )
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                return json.loads(candidate[start:end + 1])
            raise

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
