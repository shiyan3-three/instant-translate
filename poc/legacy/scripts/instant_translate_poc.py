from __future__ import annotations

import json
import statistics
import time

import httpx

from app.agent.agent import TranslationAgent
from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.settings import AppSettings
from app.translation.client import ClientConfig
from app.translation.service import TranslationService


SAMPLES = [
    "请把上面的扫帚拿到这里！",
    "最近从Windows换成了Mac，请推荐一些好用的软件。",
    "SpaceX今天什么时候开盘？价格大约70-80元。",
    "AI图生视频公益站全员免费用24小时，登录后输入Python代码即可使用。",
]


def send(config: ClientConfig, messages: list[dict], thinking: bool) -> dict:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    if thinking:
        payload["thinking"] = {"type": "enabled"}

    started = time.perf_counter()
    try:
        response = httpx.post(
            config.chat_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45.0,
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        message = body.get("choices", [{}])[0].get("message", {})
        usage = body.get("usage", {}) or {}
        return {
            "ok": bool((message.get("content") or "").strip()),
            "elapsed_s": round(elapsed, 3),
            "content": (message.get("content") or "").strip(),
            "reasoning_chars": len(message.get("reasoning_content") or ""),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "content": "",
            "reasoning_chars": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_persistent(name: str, config: ClientConfig, system: str, translate_thinking: bool) -> dict:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "请确认你理解了以上翻译规则。"},
    ]
    digest = send(config, messages, thinking=True)
    if digest["content"]:
        messages.append({"role": "assistant", "content": digest["content"]})

    rows = []
    for sample in SAMPLES:
        messages.append({"role": "user", "content": sample})
        result = send(config, messages, thinking=translate_thinking)
        if result["content"]:
            messages.append({"role": "assistant", "content": result["content"]})
        rows.append({"input": sample, **result})
    return {"name": name, "digest": digest, "rows": rows}


def run_stateless(name: str, config: ClientConfig, system: str, thinking: bool) -> dict:
    rows = []
    for sample in SAMPLES:
        result = send(
            config,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": sample},
            ],
            thinking=thinking,
        )
        rows.append({"input": sample, **result})
    return {"name": name, "digest": None, "rows": rows}


def summarize(group: dict) -> dict:
    times = [row["elapsed_s"] for row in group["rows"]]
    return {
        "name": group["name"],
        "success": sum(1 for row in group["rows"] if row["ok"]),
        "total": len(group["rows"]),
        "mean_s": round(statistics.mean(times), 3),
        "median_s": round(statistics.median(times), 3),
        "max_s": max(times),
        "reasoning_chars": sum(row["reasoning_chars"] for row in group["rows"]),
        "digest_s": group["digest"]["elapsed_s"] if group["digest"] else None,
    }


def main() -> None:
    settings = AppSettings.load()
    config = ClientConfig(
        base_url=settings.ai.base_url,
        api_key=settings.ai.api_key,
        model=settings.ai.model,
        timeout_seconds=45.0,
    )

    service = TranslationService(settings)
    try:
        current_system = service._current_prompt("中文", "日本語")
    finally:
        service.shutdown()
    current_system = TranslationAgent._rewrite_constraints(current_system)

    split_system = (
        DEFAULT_BASE_PROMPT
        + "\n\nTranslate from Chinese to natural Japanese. "
        + "Preserve Latin letters, numbers, and symbols when they carry meaning."
    )

    groups = [
        run_persistent("A_current_thinking", config, current_system, True),
        run_persistent("B_split_thinking", config, split_system, True),
        run_persistent("C_split_nonthinking", config, split_system, False),
        run_stateless("D_split_stateless_nonthinking", config, split_system, False),
    ]

    print(json.dumps({"summaries": [summarize(g) for g in groups], "groups": groups}, ensure_ascii=False))


if __name__ == "__main__":
    main()
