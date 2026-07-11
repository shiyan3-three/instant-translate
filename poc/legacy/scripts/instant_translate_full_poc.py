from __future__ import annotations

import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.agent.agent import TranslationAgent
from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.settings import AppSettings
from app.translation.client import ClientConfig
from app.translation.service import TranslationService


RESULT_PATH = Path(r"C:\tmp\instant_translate_full_poc.jsonl")
SEED = 20260623
ROUNDS = 3

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

CONDITIONS = (
    "A_current_thinking",
    "B_split_thinking",
    "C_split_requested_nonthinking",
    "D_split_stateless_requested_nonthinking",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_record(record: dict) -> None:
    with RESULT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
            timeout=60.0,
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        message = body.get("choices", [{}])[0].get("message", {})
        usage = body.get("usage", {}) or {}
        content = (message.get("content") or "").strip()
        return {
            "ok": bool(content),
            "elapsed_s": round(elapsed, 3),
            "content": content,
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


def compact(messages: list[dict]) -> list[dict]:
    max_messages = 3 + TranslationAgent.MAX_HISTORY_PAIRS * 2
    if len(messages) <= max_messages:
        return messages
    return messages[:3] + messages[-TranslationAgent.MAX_HISTORY_PAIRS * 2 :]


def digest(config: ClientConfig, condition: str, system: str, round_no: int) -> list[dict]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "请确认你理解了以上翻译规则。"},
    ]
    result = send(config, messages, thinking=True)
    append_record({
        "type": "digest",
        "timestamp": now_iso(),
        "round": round_no,
        "condition": condition,
        **result,
    })
    if result["content"]:
        messages.append({"role": "assistant", "content": result["content"]})
    return messages


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def print_running_summary(records: list[dict], completed: int, total: int) -> None:
    parts = []
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        if not rows:
            continue
        times = [r["elapsed_s"] for r in rows]
        parts.append(
            f"{condition[0]}:n={len(rows)},med={statistics.median(times):.2f}s,"
            f"p95={percentile(times, 0.95):.2f}s,ok={sum(r['ok'] for r in rows)}"
        )
    print(f"progress {completed}/{total} | " + " | ".join(parts), flush=True)


def main() -> None:
    if RESULT_PATH.exists():
        RESULT_PATH.unlink()

    settings = AppSettings.load()
    config = ClientConfig(
        base_url=settings.ai.base_url,
        api_key=settings.ai.api_key,
        model=settings.ai.model,
        timeout_seconds=60.0,
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
    systems = {
        "A_current_thinking": current_system,
        "B_split_thinking": split_system,
        "C_split_requested_nonthinking": split_system,
        "D_split_stateless_requested_nonthinking": split_system,
    }

    rng = random.Random(SEED)
    total = len(SAMPLES) * len(CONDITIONS) * ROUNDS
    completed = 0
    translation_records = []
    started_all = time.perf_counter()

    for round_no in range(1, ROUNDS + 1):
        sessions = {
            condition: digest(config, condition, systems[condition], round_no)
            for condition in CONDITIONS
            if not condition.startswith("D_")
        }
        round_samples = list(SAMPLES)
        rng.shuffle(round_samples)

        for sample_id, source in round_samples:
            condition_order = list(CONDITIONS)
            rng.shuffle(condition_order)
            for condition in condition_order:
                if condition.startswith("D_"):
                    messages = [
                        {"role": "system", "content": systems[condition]},
                        {"role": "user", "content": source},
                    ]
                else:
                    sessions[condition].append({"role": "user", "content": source})
                    sessions[condition] = compact(sessions[condition])
                    messages = sessions[condition]

                thinking = condition in {"A_current_thinking", "B_split_thinking"}
                result = send(config, messages, thinking=thinking)
                if not condition.startswith("D_") and result["content"]:
                    sessions[condition].append({"role": "assistant", "content": result["content"]})

                record = {
                    "type": "translation",
                    "timestamp": now_iso(),
                    "round": round_no,
                    "sample_id": sample_id,
                    "input": source,
                    "condition": condition,
                    "condition_order": condition_order.index(condition) + 1,
                    **result,
                }
                append_record(record)
                translation_records.append(record)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print_running_summary(translation_records, completed, total)

    print(
        json.dumps({
            "done": True,
            "result_path": str(RESULT_PATH),
            "elapsed_minutes": round((time.perf_counter() - started_all) / 60, 2),
            "translations": completed,
        }, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
