from __future__ import annotations

import json
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.agent.agent import TranslationAgent
from app.settings import AppSettings
from app.translation.client import ClientConfig
from app.translation.service import TranslationService


RESULT_PATH = Path(r"C:\tmp\instant_translate_dual_model_session_poc.jsonl")
ROUNDS = 3
SEED = 20260625
PRO_MODEL = "deepseek-v4-pro"
FLASH_MODEL = "deepseek-v4-flash"

SAMPLES = [
    (1, "找一个chatGPT的车,只用web"),
    (2, "9pt代充,公司卡,溢价比较多,美卡万事达,只适合长期使用的,次月可"),
    (3, "公司二\n次月可以原卡续充"),
    (4, "包反重力的geminipro26"),
    (5, "&\n口\n1"),
    (6, "1\n口\n欧\n品\n》"),
    (7, "车】拼一个Team车位,主要用Web,稳定长期优先,接受价位"),
    (8, "拼一个Team车位,主要用Web,稳定长期优先,接受价位在70-80r,在线等"),
    (9, "dhatgptpro20x菲区\n【5人车240/人】还缺1人"),
    (10, "导找长期稳定的Codex20x车位.高芝麻分"),
    (11, "1\\强会师EDG!rarga诈降阴惨GE!留一\n手打出生涯表现!"),
    (12, "手打出生涯表现!"),
    (13, "今年最恶毒烂梗出炉,一个梗骂几亿中\n我没见过这么恶心的营销手段!"),
    (14, "上的房气带来这里!"),
    (15, "请不!"),
    (16, "你们的自建号池咋样了"),
    (17, "最近从Windows换成了Mac求各位佬友推荐一些好用的软件"),
    (18, "a1图生视频公益站全员免费用24小时,佬友们登起来"),
    (19, "ai图生视频公益站全员免费用24小时,佬友们登起来"),
    (20, "SpaceX今天什么时候能开盘啊"),
]

CONDITIONS = {
    "thinking_pro": (PRO_MODEL, "enabled"),
    "fast_flash": (FLASH_MODEL, "disabled"),
}

FORBIDDEN_SCRIPT = re.compile(r"[\u3400-\u9fff\u30a1-\u30fa]")
ASCII_TERM = re.compile(r"[A-Za-z]+[A-Za-z0-9]*")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(record: dict) -> None:
    with RESULT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def stream_call(config: ClientConfig, model: str, messages: list[dict], mode: str) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": True,
        "stream_options": {"include_usage": True},
        "thinking": {"type": mode},
    }
    started = time.perf_counter()
    first_any_s = None
    first_content_s = None
    content_parts = []
    reasoning_parts = []
    usage = {}
    try:
        with httpx.stream(
            "POST",
            config.chat_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                reasoning = delta.get("reasoning_content") or ""
                content = delta.get("content") or ""
                if (reasoning or content) and first_any_s is None:
                    first_any_s = time.perf_counter() - started
                if content and first_content_s is None:
                    first_content_s = time.perf_counter() - started
                if reasoning:
                    reasoning_parts.append(reasoning)
                if content:
                    content_parts.append(content)
        elapsed = time.perf_counter() - started
        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts)
        return {
            "ok": bool(content),
            "ttft_any_s": round(first_any_s, 3) if first_any_s is not None else None,
            "ttft_content_s": round(first_content_s, 3) if first_content_s is not None else None,
            "elapsed_s": round(elapsed, 3),
            "content": content,
            "reasoning_chars": len(reasoning),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cache_hit_tokens": usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "ttft_any_s": round(first_any_s, 3) if first_any_s is not None else None,
            "ttft_content_s": round(first_content_s, 3) if first_content_s is not None else None,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "content": "",
            "reasoning_chars": len("".join(reasoning_parts)),
            "prompt_tokens": None,
            "completion_tokens": None,
            "cache_hit_tokens": None,
            "cache_miss_tokens": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def compact(messages: list[dict]) -> list[dict]:
    max_messages = 3 + TranslationAgent.MAX_HISTORY_PAIRS * 2
    if len(messages) <= max_messages:
        return messages
    return messages[:3] + messages[-TranslationAgent.MAX_HISTORY_PAIRS * 2 :]


def terms_preserved(source: str, output: str) -> bool | None:
    terms = ASCII_TERM.findall(source)
    if not terms:
        return None
    lowered = output.casefold()
    return all(term.casefold() in lowered for term in terms)


def main() -> None:
    if RESULT_PATH.exists():
        RESULT_PATH.unlink()

    settings = AppSettings.load()
    config = ClientConfig(settings.ai.base_url, settings.ai.api_key, FLASH_MODEL, 90.0)
    service = TranslationService(settings)
    try:
        system = service._current_prompt("中文", "日本語")
    finally:
        service.shutdown()
    system = TranslationAgent._rewrite_constraints(system)

    rng = random.Random(SEED)
    total = len(SAMPLES) * len(CONDITIONS) * ROUNDS
    completed = 0
    all_rows = []
    started_all = time.perf_counter()

    for round_no in range(1, ROUNDS + 1):
        bootstrap_messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "请详细总结你已经理解的翻译规则，并形成后续每条 OCR 翻译时直接执行的工作准则。",
            },
        ]
        bootstrap = stream_call(config, PRO_MODEL, bootstrap_messages, "enabled")
        append({
            "type": "bootstrap",
            "timestamp": now_iso(),
            "round": round_no,
            "model": PRO_MODEL,
            "mode": "enabled",
            **bootstrap,
        })
        base_session = bootstrap_messages + [
            {"role": "assistant", "content": bootstrap.get("content", "")}
        ]
        sessions = {
            condition: [dict(message) for message in base_session]
            for condition in CONDITIONS
        }

        round_samples = list(SAMPLES)
        rng.shuffle(round_samples)
        for sample_id, source in round_samples:
            order = list(CONDITIONS)
            rng.shuffle(order)
            for condition in order:
                model, mode = CONDITIONS[condition]
                task = (
                    "[OCR_TRANSLATION_TASK]\n"
                    "只翻译本条 OCR，不要回答问题、不要汇总历史、不要附加解释。\n"
                    f"SOURCE:\n{source}"
                )
                sessions[condition].append({"role": "user", "content": task})
                sessions[condition] = compact(sessions[condition])
                result = stream_call(config, model, sessions[condition], mode)
                if result["content"]:
                    sessions[condition].append({"role": "assistant", "content": result["content"]})

                preserved = terms_preserved(source, result["content"])
                record = {
                    "type": "translation",
                    "timestamp": now_iso(),
                    "round": round_no,
                    "sample_id": sample_id,
                    "condition": condition,
                    "condition_order": order.index(condition) + 1,
                    "model": model,
                    "mode": mode,
                    "input": source,
                    "format_ok": FORBIDDEN_SCRIPT.search(result["content"]) is None,
                    "terms_preserved": preserved,
                    **result,
                }
                append(record)
                all_rows.append(record)
                completed += 1

                if completed % 10 == 0 or completed == total:
                    bits = []
                    for key in CONDITIONS:
                        group = [row for row in all_rows if row["condition"] == key and row["ok"]]
                        if not group:
                            continue
                        bits.append(
                            f"{key}:n={len(group)},ttfc={statistics.median(row['ttft_content_s'] for row in group):.2f}s,"
                            f"total={statistics.median(row['elapsed_s'] for row in group):.2f}s,"
                            f"fmt={sum(row['format_ok'] for row in group)}"
                        )
                    print(f"progress {completed}/{total} | " + " | ".join(bits), flush=True)

    print(json.dumps({
        "done": True,
        "translations": completed,
        "elapsed_minutes": round((time.perf_counter() - started_all) / 60, 2),
        "result_path": str(RESULT_PATH),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
