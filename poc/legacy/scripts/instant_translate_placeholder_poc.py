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


RESULT_PATH = Path(r"C:\tmp\instant_translate_placeholder_poc.jsonl")
SEED = 20260624
ROUNDS = 3

SAMPLES = [
    (1, "找一个ChatGPT共享账号，只使用Web版。", ["ChatGPT", "Web"]),
    (2, "最近从Windows换成了Mac。", ["Windows", "Mac"]),
    (3, "SpaceX今天什么时候开盘？", ["SpaceX"]),
    (4, "使用Python和NumPy处理数据。", ["Python", "NumPy"]),
    (5, "请在GitHub提交Pull Request。", ["GitHub", "Pull Request"]),
    (6, "用Docker Compose部署PostgreSQL。", ["Docker Compose", "PostgreSQL"]),
    (7, "在VS Code里安装Python插件。", ["VS Code", "Python"]),
    (8, "登录OpenAI API控制台。", ["OpenAI API"]),
    (9, "在游戏中找到NPC Alice。", ["NPC", "Alice"]),
    (10, "Team车位价格70-80r，长期稳定优先。", ["Team", "70-80r"]),
    (11, "使用GPT-4o和Claude 4进行对比。", ["GPT-4o", "Claude 4"]),
    (12, "打开Node.js项目并运行npm install。", ["Node.js", "npm install"]),
    (13, "这个C++程序依赖.NET 8。", ["C++", ".NET 8"]),
    (14, "Codex20x车位还缺1人。", ["Codex20x"]),
    (15, "AI图生视频工具免费24小时。", ["AI", "24"]),
]

CONDITIONS = ("plain", "placeholder")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(record: dict) -> None:
    with RESULT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def send(config: ClientConfig, messages: list[dict]) -> dict:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
        "thinking": {"type": "enabled"},
    }
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


def protect(source: str, terms: list[str]) -> tuple[str, dict[str, str]]:
    protected = source
    mapping = {}
    for index, term in enumerate(sorted(terms, key=len, reverse=True)):
        token = f"__IT_TERM_{index}__"
        protected = protected.replace(term, token)
        mapping[token] = term
    return protected, mapping


def restore(content: str, mapping: dict[str, str]) -> tuple[str, list[str]]:
    missing = [token for token in mapping if token not in content]
    restored = content
    for token, term in mapping.items():
        restored = restored.replace(token, term)
    return restored, missing


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
    result = send(config, messages)
    append({"type": "digest", "round": round_no, "condition": condition, **result})
    if result["content"]:
        messages.append({"role": "assistant", "content": result["content"]})
    return messages


def main() -> None:
    if RESULT_PATH.exists():
        RESULT_PATH.unlink()

    settings = AppSettings.load()
    config = ClientConfig(settings.ai.base_url, settings.ai.api_key, settings.ai.model, 60.0)
    base_system = (
        DEFAULT_BASE_PROMPT
        + "\n\nTranslate from Chinese to natural Japanese. "
        + "Preserve Latin product names, technical terms, numbers, and symbols exactly when requested."
    )
    systems = {
        "plain": base_system,
        "placeholder": base_system + (
            "\nTokens matching __IT_TERM_N__ are protected placeholders. "
            "Copy every such token into the translation exactly, without translating, editing, or deleting it."
        ),
    }

    rng = random.Random(SEED)
    completed = 0
    total = len(SAMPLES) * len(CONDITIONS) * ROUNDS
    rows = []
    started_all = time.perf_counter()

    for round_no in range(1, ROUNDS + 1):
        sessions = {
            condition: digest(config, condition, systems[condition], round_no)
            for condition in CONDITIONS
        }
        round_samples = list(SAMPLES)
        rng.shuffle(round_samples)

        for sample_id, source, terms in round_samples:
            condition_order = list(CONDITIONS)
            rng.shuffle(condition_order)
            for condition in condition_order:
                mapping = {}
                submitted = source
                if condition == "placeholder":
                    submitted, mapping = protect(source, terms)

                sessions[condition].append({"role": "user", "content": submitted})
                sessions[condition] = compact(sessions[condition])
                result = send(config, sessions[condition])
                if result["content"]:
                    sessions[condition].append({"role": "assistant", "content": result["content"]})

                restored, missing_placeholders = restore(result["content"], mapping)
                missing_terms = [term for term in terms if term not in restored]
                record = {
                    "type": "translation",
                    "timestamp": now_iso(),
                    "round": round_no,
                    "sample_id": sample_id,
                    "condition": condition,
                    "condition_order": condition_order.index(condition) + 1,
                    "input": source,
                    "submitted": submitted,
                    "terms": terms,
                    "restored": restored,
                    "missing_placeholders": missing_placeholders,
                    "missing_terms": missing_terms,
                    **result,
                }
                append(record)
                rows.append(record)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    bits = []
                    for key in CONDITIONS:
                        group = [row for row in rows if row["condition"] == key]
                        bits.append(
                            f"{key}:n={len(group)},terms_ok={sum(not row['missing_terms'] for row in group)},"
                            f"med={statistics.median(row['elapsed_s'] for row in group):.2f}s"
                        )
                    print(f"progress {completed}/{total} | " + " | ".join(bits), flush=True)

    print(json.dumps({
        "done": True,
        "result_path": str(RESULT_PATH),
        "translations": completed,
        "elapsed_minutes": round((time.perf_counter() - started_all) / 60, 2),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
