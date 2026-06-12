"""Translation Agent — persistent messages session with thinking enabled."""

from __future__ import annotations

from app.translation.client import ClientConfig, OpenAICompatibleClient


class TranslationAgent:
    """Maintain a persistent chat session so rules are digested once."""

    def __init__(self, config: ClientConfig) -> None:
        self._client = OpenAICompatibleClient(config)
        self.messages: list[dict] = []

    def digest_rules(self, system_prompt: str) -> None:
        """Feed rules once at session start."""

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请确认你理解了以上翻译规则。"},
        ]
        resp = self._client.chat(self.messages, thinking=True)
        self.messages.append({"role": "assistant", "content": resp})

    def translate(self, text: str) -> str:
        """Translate one piece of text using the established session."""

        self.messages.append({"role": "user", "content": text})
        resp = self._client.chat(self.messages, thinking=True)
        self.messages.append({"role": "assistant", "content": resp})
        return resp
