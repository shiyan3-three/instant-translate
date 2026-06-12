"""Translation Agent — persistent messages session with thinking enabled."""

from __future__ import annotations

from app.translation.client import ClientConfig, OpenAICompatibleClient


class TranslationAgent:
    """Maintain a persistent chat session so rules are digested once.
    
    Messages are automatically compacted to prevent context overflow:
    - System message (index 0) is always preserved
    - Rule confirmation pair (index 1-2) is always preserved
    - Most recent 10 translation pairs are kept
    - Older messages are discarded
    """
    
    MAX_HISTORY_PAIRS = 10  # Keep last 10 user/assistant pairs

    def __init__(self, config: ClientConfig) -> None:
        self._client = OpenAICompatibleClient(config)
        self.messages: list[dict] = []

    def digest_rules(self, system_prompt: str) -> None:
        """Feed rules once at session start.
        
        Adds conflict resolution instruction to handle cases where
        user constraints conflict with actual input (e.g., hiragana-only
        constraint with English/number input).
        """
        
        # Add conflict handling instruction to system prompt
        enhanced_prompt = f"""{system_prompt}

当约束与实际输入冲突时，优先保证翻译可用性。
例如：约束要求"只输出平假名"但输入包含英文/数字/符号时，
这些字符原样保留，仅对中文和日文部分应用平假名约束。"""

        self.messages = [
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": "请确认你理解了以上翻译规则。"},
        ]
        resp = self._client.chat(self.messages, thinking=True)
        self.messages.append({"role": "assistant", "content": resp})

    def translate(self, text: str) -> str:
        """Translate one piece of text using the established session.
        
        Automatically compacts messages when history grows too long,
        keeping system + rule confirmation + last N pairs.
        """

        self.messages.append({"role": "user", "content": text})
        
        # Compact messages if needed
        # Structure: [system, user_confirm, assistant_confirm, ...history...]
        # Keep: system (0) + confirmation pair (1-2) + recent N pairs
        max_messages = 3 + (self.MAX_HISTORY_PAIRS * 2)
        if len(self.messages) > max_messages:
            # Keep first 3 (system + confirmation) + last N pairs
            keep_recent = self.MAX_HISTORY_PAIRS * 2
            self.messages = self.messages[:3] + self.messages[-keep_recent:]
        
        resp = self._client.chat(self.messages, thinking=True)
        self.messages.append({"role": "assistant", "content": resp})
        return resp
