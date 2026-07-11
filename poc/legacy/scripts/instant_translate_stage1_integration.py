from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from app.agent.agent import TranslationAgent
from app.settings import AppSettings
from app.translation.client import ClientConfig
from app.translation.service import TranslationService


SAMPLES = [
    "今天的视频我们继续讲即时翻译工具的使用方法。",
    "如果识别结果重复，就不要再次请求翻译。",
    "把这个窗口拖到字幕附近，然后按快捷键开始监控。",
    "Team70r 的价格区间大概是 70-80r。",
    "这个 Pull Request 主要修复了 API 超时的问题。",
    "请保持 Claude Code、OpenCode 和 Codex 这几个术语不变。",
    "用户想要的是启动时理解一次规则，后面快速翻译。",
    "如果输出里还有片假名，就说明快速模式没有完全遵守规则。",
    "我们先验证 thinking disabled 是否真的能降低延迟。",
    "这个阶段不要急着引入 SQLite 和复杂的 PolicyCompiler。",
    "OCR 偶尔会把公益站识别成公盖站，需要模型按语义处理。",
    "快速模式失败时，应该回退到思考模式重新翻译。",
    "请只输出译文，不要解释，不要添加额外说明。",
    "这个工具最终目标是让用户看视频时不被翻译过程打断。",
    "如果原文包含 24 小时、AI、API 这些内容，需要按规则处理。",
    "下一步我们要做术语占位符和格式校验。",
    "本地 Session 不是 API 自动记忆，而是每次请求重新组装上下文。",
    "今天先把最小验证跑通，再决定是否继续扩展架构。",
    "翻译结果应该稳定、简短、直接。",
    "如果规则和输入冲突，优先保证结果可用。",
]


def main() -> None:
    settings = AppSettings.load()
    ai = settings.ai
    if not ai.base_url.strip() or not ai.api_key.strip() or not ai.model.strip():
        raise SystemExit("Missing base_url/api_key/model in AppSettings.")

    service = TranslationService(settings, max_workers=1)
    prompt = service._current_prompt(
        settings.default_source_language,
        settings.default_target_language,
    )

    agent = TranslationAgent(
        ClientConfig(
            base_url=ai.base_url,
            api_key=ai.api_key,
            model=ai.fast_model_name,
        ),
        ClientConfig(
            base_url=ai.base_url,
            api_key=ai.api_key,
            model=ai.thinking_model_name,
        )
    )

    out_path = Path("C:/tmp/instant_translate_stage1_integration_results.jsonl")
    if out_path.exists():
        out_path.unlink()

    started = time.perf_counter()
    digest_started = time.perf_counter()
    agent.digest_rules(prompt)
    digest_elapsed = time.perf_counter() - digest_started

    rows: list[dict] = [
        {
            "type": "digest",
            "fast_model": ai.fast_model_name,
            "thinking_model": ai.thinking_model_name,
            "elapsed_s": round(digest_elapsed, 3),
            "messages_after": len(agent.messages),
        }
    ]
    out_path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")

    elapsed_values: list[float] = []
    ok_count = 0
    for index, sample in enumerate(SAMPLES, start=1):
        t0 = time.perf_counter()
        error = None
        content = None
        try:
            content = agent.translate(sample)
            ok_count += 1
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            error = str(exc)
        elapsed = time.perf_counter() - t0
        elapsed_values.append(elapsed)
        row = {
            "type": "translation",
            "index": index,
            "fast_model": ai.fast_model_name,
            "thinking_model": ai.thinking_model_name,
            "elapsed_s": round(elapsed, 3),
            "input": sample,
            "content": content,
            "error": error,
            "messages_after": len(agent.messages),
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"{index:02d}/20 elapsed={elapsed:.3f}s "
            f"ok={error is None} chars={len(content or '')}"
        )

    total_elapsed = time.perf_counter() - started
    summary = {
        "type": "summary",
        "fast_model": ai.fast_model_name,
        "thinking_model": ai.thinking_model_name,
        "ok_count": ok_count,
        "n": len(SAMPLES),
        "digest_elapsed_s": round(digest_elapsed, 3),
        "median_translate_s": round(statistics.median(elapsed_values), 3),
        "mean_translate_s": round(statistics.mean(elapsed_values), 3),
        "max_translate_s": round(max(elapsed_values), 3),
        "total_elapsed_s": round(total_elapsed, 3),
        "output_path": str(out_path),
    }
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
