from __future__ import annotations

import json
import time

import httpx

from app.agent.agent import TranslationAgent
from app.settings import AppSettings
from app.translation.client import ClientConfig
from app.translation.service import TranslationService


def call(config: ClientConfig, messages: list[dict], mode: str) -> dict:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    if mode == "enabled":
        payload["thinking"] = {"type": "enabled"}
    elif mode == "disabled":
        payload["thinking"] = {"type": "disabled"}

    started = time.perf_counter()
    try:
        response = httpx.post(
            config.chat_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        elapsed = time.perf_counter() - started
        body = response.json()
        if response.status_code >= 400:
            return {"mode": mode, "status": response.status_code, "elapsed_s": round(elapsed, 3), "error": body}
        message = body.get("choices", [{}])[0].get("message", {})
        return {
            "mode": mode,
            "status": response.status_code,
            "elapsed_s": round(elapsed, 3),
            "content": (message.get("content") or "").strip(),
            "reasoning_chars": len(message.get("reasoning_content") or ""),
            "message_keys": list(message.keys()),
            "usage": body.get("usage"),
        }
    except Exception as exc:
        return {"mode": mode, "status": None, "elapsed_s": round(time.perf_counter() - started, 3), "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    settings = AppSettings.load()
    config = ClientConfig(settings.ai.base_url, settings.ai.api_key, settings.ai.model, 60.0)
    service = TranslationService(settings)
    try:
        system = service._current_prompt("中文", "日本語")
    finally:
        service.shutdown()
    system = TranslationAgent._rewrite_constraints(system)

    digest_messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "请总结你已经理解的翻译规则，形成后续翻译时直接执行的简短工作准则。",
        },
    ]
    digest = call(config, digest_messages, "enabled")
    session = digest_messages + [{"role": "assistant", "content": digest.get("content", "")}]
    source = "最近从Windows换成了Mac，请推荐一些好用的软件。"
    test_messages = session + [{"role": "user", "content": source}]
    results = [call(config, test_messages, mode) for mode in ("enabled", "omitted", "disabled")]
    print(json.dumps({"digest": digest, "translation_results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
