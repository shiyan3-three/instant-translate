"""
POC: Reference Layer Split + Keyword Matching vs Full System Prompt

Tests whether splitting the glossary out of the system prompt and injecting
matched entries per-request produces translation quality equal to or better
than the current all-in-system-prompt approach.

Hypotheses:
  H1: Keyword matching accurately catches glossary terms in OCR text
  H2: Split approach (B) produces translation quality equal to approach A
  H3: Split approach reduces system prompt size (KV cache benefit)
"""

from __future__ import annotations

import os
import re
import sys
import time

# ensure app imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.settings import AppSettings
from app.translation.client import ClientConfig, OpenAICompatibleClient, TranslationError

# =========================================================================
# Glossary (20 software engineering terms from compiled-prompt.md)
# =========================================================================

GLOSSARY: dict[str, str] = {
    "软件": "[そふとうぇあ]",
    "硬件": "[はーどうぇあ]",
    "程序": "[ぷろぐらむ]",
    "编译": "[こんぱいる]",
    "调试": "[でばっぐ]",
    "算法": "[あるごりずむ]",
    "数据库": "[でーたべーす]",
    "接口": "[いんたーふぇーす]",
    "服务器": "[さーばー]",
    "客户端": "[くらいあんと]",
    "代码": "[こーど]",
    "变量": "[へんすう]",
    "函数": "[かんすう]",
    "对象": "[おぶじぇくと]",
    "开发": "[かいはつ]",
    "部署": "[でぷろい]",
    "版本": "[ばーじょん]",
    "错误": "[ばぐ]",
    "补丁": "[ぱっち]",
    "测试": "[てすと]",
}

# =========================================================================
# System prompts
# =========================================================================

# Shared base: compiled prompt WITHOUT the glossary table
_BASE_PROMPT = """# Instant Translate Compiled Prompt

## Fixed Template Layer
This layer is system-owned and has the highest priority.

You are an instant translation assistant.
Return only the translated text.
Do not explain, annotate, or expand the result.
User constraints are hard requirements.
If a user constraint specifies an output format, script, glossary, or style, obey it exactly.
Preserve the source meaning as accurately as possible while satisfying those constraints.
Before returning, verify the output against all user constraints and rewrite it if needed.
OCR input may contain recognition errors; translate only the given text and do not invent extra context.

## User Constraint Layer
This layer contains user-confirmed translation constraints. These rules are mandatory and must not be weakened by later layers.

如果中文翻译为日语时，翻译的日语句子只能由平假名构成，涉及到软件工程相关的词应该有[]包裹起来，每个单词之间至少要隔两个空格

## AI Optimization Layer
This layer contains supplemental rules derived from the user layer. Use it to clarify the user's intent, never to replace or override it.

Supplemental Rules
- First produce a **natural Japanese translation** that faithfully conveys the original meaning. Restructure into natural Japanese word order, omit subjects/pronouns when context is clear, and translate by meaning rather than character-by-character substitution. Then apply the formatting rules below.
- Translate Chinese to Japanese using **only hiragana** characters (あ-ん). No kanji, katakana, romaji, punctuation, or symbols except `[ ]` for flagged terms.
- Wrap every software engineering term in `[ ]` (brackets). The bracketed content must also be in hiragana.
- Insert exactly **two space characters** (U+0020) between every word (lexical unit). No spaces at the start or end of the sentence.

"""

# The glossary table block (used in Approach A, omitted in Approach B)
_GLOSSARY_TABLE = """Glossary Entries (Software Engineering Terms)
Input the Chinese term; output the bracketed hiragana equivalent. These are mandatory replacements if the source contains them.

| Chinese Term | Output (hiragana in brackets) |
|--------------|-------------------------------|
| 软件         | [そふとうぇあ]                 |
| 硬件         | [はーどうぇあ]                 |
| 程序         | [ぷろぐらむ]                   |
| 编译         | [こんぱいる]                   |
| 调试         | [でばっぐ]                     |
| 算法         | [あるごりずむ]                 |
| 数据库       | [でーたべーす]                 |
| 接口         | [いんたーふぇーす]             |
| 服务器       | [さーばー]                     |
| 客户端       | [くらいあんと]                 |
| 代码         | [こーど]                       |
| 变量         | [へんすう]                     |
| 函数         | [かんすう]                     |
| 对象         | [おぶじぇくと]                 |
| 开发         | [かいはつ]                     |
| 部署         | [でぷろい]                     |
| 版本         | [ばーじょん]                   |
| 错误         | [ばぐ]                         |
| 补丁         | [ぱっち]                       |
| 测试         | [てすと]                       |

For any other software engineering term not listed, generate its hiragana reading and enclose in brackets.
"""

# The replacement note for Approach B (instead of the glossary table)
_GLOSSARY_NOTE = """Glossary Entries (Software Engineering Terms)
Glossary entries for terms present in the source text will be provided per-request in <RELEVANT_GLOSSARY> tags in the user message. For any software engineering term not provided there, generate its hiragana reading and enclose in brackets.
"""

# Style guidance + fixed expressions (shared, comes after glossary section)
_STYLE_AND_FIXED = """Style Guidance
- Tokenize the Japanese text into words (dictionary-based segmentation, morphological analysis). Use the word boundaries as spacing points.
- No particles are treated as separate words; retain them within the word they attach to in hiragana flow, but ensure double spaces between distinct lexical items.
- If a term is in brackets, treat the whole bracket as a single word unit; double-space before and after it.
- Example: "打开软件" → ひらく  [そふとうぇあ]  (double spaces shown between ひらく and the bracket, and after the bracket if sentence continues).

Fixed Expressions
- <No additional fixed expressions beyond the software terms; all output must follow hiragana-only constraint.>

## Knowledge Reference Layer
This layer contains glossary, style, and fixed-expression references.

No knowledge references.

## Runtime Direction
For every request, follow the source and target languages appended by the app.
Source: Chinese
Target: Japanese
"""

# Approach A: full system prompt with glossary table
SYSTEM_PROMPT_A = _BASE_PROMPT + _GLOSSARY_TABLE + _STYLE_AND_FIXED

# Approach B: system prompt with glossary note (no table)
SYSTEM_PROMPT_B = _BASE_PROMPT + _GLOSSARY_NOTE + _STYLE_AND_FIXED

# =========================================================================
# Test cases
# =========================================================================

TEST_CASES: list[str] = [
    # 5 glossary terms: 软件, 开发, 调试, 测试, 程序
    "软件开发过程中，调试和测试是程序必不可少的环节",
    # 3 glossary terms: 数据库, 服务器, 接口
    "数据库连接失败，检查服务器的接口配置",
    # 2 glossary terms: 函数, 变量
    "这个函数的变量名不符合命名规范",
    # 4 glossary terms: 部署, 版本, 代码, 补丁
    "部署新版本之前，先备份代码并打好补丁",
    # 0 glossary terms
    "今天的会议推迟到下午三点，请通知所有参会人员",
    # 4 glossary terms: 编译, 算法, 错误, 对象
    "编译时算法报错，错误指向了一个空对象",
    # 3 glossary terms: 客户端, 硬件, 服务器
    "客户端的硬件配置太低，连不上服务器",
    # mixed: contains a substring that's a glossary term but different meaning
    # "测试" appears in "测试" (real term) vs no false positive scenario
    "迭代版本的测试覆盖了核心算法和数据库",
]

# =========================================================================
# Keyword matching
# =========================================================================

def match_glossary(text: str, glossary: dict[str, str]) -> dict[str, str]:
    """Return glossary entries whose key appears as a substring in text."""
    matched: dict[str, str] = {}
    for term, reading in glossary.items():
        if term in text:
            matched[term] = reading
    return matched


def build_glossary_block(matched: dict[str, str]) -> str:
    """Build the <RELEVANT_GLOSSARY> injection block for the user message."""
    if not matched:
        return ""
    lines = [f"{term} → {reading}" for term, reading in matched.items()]
    return "<RELEVANT_GLOSSARY>\n" + "\n".join(lines) + "\n</RELEVANT_GLOSSARY>\n"


# =========================================================================
# Output validation
# =========================================================================

def validate_output(text: str) -> dict[str, bool | str]:
    """Check whether translation output follows the core rules."""
    result: dict[str, bool | str] = {}

    # Rule 1: only hiragana, brackets, and spaces
    hiragana_pattern = re.compile(r"^[あ-ん\s\[\]]+$")
    result["hiragana_only"] = bool(hiragana_pattern.match(text))

    # Rule 2: brackets present for content
    result["has_brackets"] = "[" in text and "]" in text

    # Rule 3: double spaces present (at least one occurrence)
    result["has_double_spaces"] = "  " in text

    # Rule 4: no leading/trailing spaces
    result["no_edge_spaces"] = text == text.strip()

    # Overall
    result["all_pass"] = all(
        v for k, v in result.items() if k != "all_pass"
    )
    return result


def check_bracket_terms(text: str, expected_terms: dict[str, str]) -> list[str]:
    """Return list of expected bracket readings that are missing from text."""
    missing = []
    for term, reading in expected_terms.items():
        if reading not in text:
            missing.append(f"{term}→{reading}")
    return missing


# =========================================================================
# Token estimation
# =========================================================================

def estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK chars ≈ 1 token, others ≈ 4 chars/token."""
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u309f")
    other_count = len(text) - cjk_count
    return cjk_count + max(1, other_count // 4)


# =========================================================================
# Main POC
# =========================================================================

def main() -> None:
    print("=" * 70)
    print("POC: Reference Layer Split + Keyword Matching vs Full System Prompt")
    print("=" * 70)

    # Load settings
    settings = AppSettings.load()
    ai = settings.ai
    model = ai.fast_model_name
    if not model:
        print("ERROR: No model configured. Set fast_model in settings.")
        return
    print(f"Model: {model}")
    print(f"Base URL: {ai.base_url}")

    client = OpenAICompatibleClient(ClientConfig(
        base_url=ai.base_url,
        api_key=ai.api_key,
        model=model,
        timeout_seconds=60.0,
    ))

    # Print system prompt sizes
    tokens_a = estimate_tokens(SYSTEM_PROMPT_A)
    tokens_b = estimate_tokens(SYSTEM_PROMPT_B)
    print(f"\nSystem Prompt A (full):     {len(SYSTEM_PROMPT_A):>5} chars, ~{tokens_a} tokens")
    print(f"System Prompt B (split):    {len(SYSTEM_PROMPT_B):>5} chars, ~{tokens_b} tokens")
    print(f"Savings per request:        {len(SYSTEM_PROMPT_A) - len(SYSTEM_PROMPT_B):>5} chars, ~{tokens_a - tokens_b} tokens")

    results: list[dict] = []

    for i, ocr_text in enumerate(TEST_CASES):
        print(f"\n{'─' * 70}")
        print(f"Test {i + 1}/{len(TEST_CASES)}: {ocr_text}")

        matched = match_glossary(ocr_text, GLOSSARY)
        print(f"Matched glossary ({len(matched)}): {', '.join(f'{k}→{v}' for k, v in matched.items()) or '(none)'}")

        # --- Approach A: full system prompt ---
        user_msg_a = f"<OCR_TEXT>\n{ocr_text}\n</OCR_TEXT>"
        t0 = time.perf_counter()
        try:
            result_a = client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT_A},
                 {"role": "user", "content": user_msg_a}],
                thinking="disabled",
            )
            time_a = time.perf_counter() - t0
            print(f"\n  [A] ({time_a:.1f}s): {result_a}")
        except TranslationError as exc:
            result_a = f"(ERROR: {exc})"
            time_a = time.perf_counter() - t0
            print(f"\n  [A] ({time_a:.1f}s): {result_a}")

        # --- Approach B: split system prompt + glossary injection ---
        glossary_block = build_glossary_block(matched)
        user_msg_b = f"{glossary_block}<OCR_TEXT>\n{ocr_text}\n</OCR_TEXT>"
        t0 = time.perf_counter()
        try:
            result_b = client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT_B},
                 {"role": "user", "content": user_msg_b}],
                thinking="disabled",
            )
            time_b = time.perf_counter() - t0
            print(f"  [B] ({time_b:.1f}s): {result_b}")
        except TranslationError as exc:
            result_b = f"(ERROR: {exc})"
            time_b = time.perf_counter() - t0
            print(f"  [B] ({time_b:.1f}s): {result_b}")

        # --- Validation ---
        val_a = validate_output(result_a)
        val_b = validate_output(result_b)
        missing_a = check_bracket_terms(result_a, matched) if matched else []
        missing_b = check_bracket_terms(result_b, matched) if matched else []

        identical = result_a == result_b
        print(f"\n  Validation A: hiragana={val_a['hiragana_only']} brackets={val_a['has_brackets']} "
              f"dbl_space={val_a['has_double_spaces']} edges={val_a['no_edge_spaces']}")
        print(f"  Validation B: hiragana={val_b['hiragana_only']} brackets={val_b['has_brackets']} "
              f"dbl_space={val_b['has_double_spaces']} edges={val_b['no_edge_spaces']}")

        if missing_a:
            print(f"  [A] Missing bracket terms: {', '.join(missing_a)}")
        if missing_b:
            print(f"  [B] Missing bracket terms: {', '.join(missing_b)}")

        print(f"  Identical: {'Y' if identical else 'N'}")
        if not identical:
            print(f"  [A] len={len(result_a)}, [B] len={len(result_b)}")

        results.append({
            "test": i + 1,
            "ocr": ocr_text,
            "matched": matched,
            "result_a": result_a,
            "result_b": result_b,
            "time_a": time_a,
            "time_b": time_b,
            "val_a": val_a,
            "val_b": val_b,
            "missing_a": missing_a,
            "missing_b": missing_b,
            "identical": identical,
        })

        # small delay to avoid rate limiting
        time.sleep(0.5)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    identical_count = sum(1 for r in results if r["identical"])
    a_all_pass = sum(1 for r in results if r["val_a"]["all_pass"])
    b_all_pass = sum(1 for r in results if r["val_b"]["all_pass"])
    a_missing_total = sum(len(r["missing_a"]) for r in results)
    b_missing_total = sum(len(r["missing_b"]) for r in results)
    avg_time_a = sum(r["time_a"] for r in results) / len(results)
    avg_time_b = sum(r["time_b"] for r in results) / len(results)

    print(f"Test cases:                {len(results)}")
    print(f"Identical results:         {identical_count}/{len(results)}")
    print(f"A all rules pass:          {a_all_pass}/{len(results)}")
    print(f"B all rules pass:          {b_all_pass}/{len(results)}")
    print(f"A missing bracket terms:   {a_missing_total}")
    print(f"B missing bracket terms:   {b_missing_total}")
    print(f"Avg time A:                {avg_time_a:.2f}s")
    print(f"Avg time B:                {avg_time_b:.2f}s")
    print(f"System prompt savings:     ~{tokens_a - tokens_b} tokens/request")

    # Detailed per-test summary
    print(f"Per-test detail:")
    print(f"{'Test':<6} {'Matched':<10} {'A_pass':<8} {'B_pass':<8} {'A_miss':<8} {'B_miss':<8} {'Same':<6} {'A_time':<8} {'B_time':<8}")
    for r in results:
        print(f"{r['test']:<6} {len(r['matched']):<10} "
              f"{'Y' if r['val_a']['all_pass'] else 'N':<8} "
              f"{'Y' if r['val_b']['all_pass'] else 'N':<8} "
              f"{len(r['missing_a']):<8} {len(r['missing_b']):<8} "
              f"{'Y' if r['identical'] else 'N':<6} "
              f"{r['time_a']:<8.1f} {r['time_b']:<8.1f}")

    # Write full results to file
    output_path = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")),
        "instant-translate", "poc_results.txt"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("POC Results: Reference Layer Split + Keyword Matching\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"System Prompt A: {len(SYSTEM_PROMPT_A)} chars, ~{tokens_a} tokens\n")
        f.write(f"System Prompt B: {len(SYSTEM_PROMPT_B)} chars, ~{tokens_b} tokens\n")
        f.write(f"Savings: ~{tokens_a - tokens_b} tokens/request\n\n")

        for r in results:
            f.write(f"{'─' * 70}\n")
            f.write(f"Test {r['test']}: {r['ocr']}\n")
            f.write(f"Matched ({len(r['matched'])}): {', '.join(f'{k}→{v}' for k, v in r['matched'].items()) or '(none)'}\n\n")
            f.write(f"[A] ({r['time_a']:.1f}s): {r['result_a']}\n")
            f.write(f"[B] ({r['time_b']:.1f}s): {r['result_b']}\n")
            f.write(f"Validation A: {r['val_a']}\n")
            f.write(f"Validation B: {r['val_b']}\n")
            f.write(f"Missing A: {r['missing_a']}\n")
            f.write(f"Missing B: {r['missing_b']}\n")
            f.write(f"Identical: {r['identical']}\n\n")

        f.write(f"{'=' * 70}\n")
        f.write(f"SUMMARY\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"Identical: {identical_count}/{len(results)}\n")
        f.write(f"A all pass: {a_all_pass}/{len(results)}\n")
        f.write(f"B all pass: {b_all_pass}/{len(results)}\n")
        f.write(f"A missing terms: {a_missing_total}\n")
        f.write(f"B missing terms: {b_missing_total}\n")
        f.write(f"Avg time A: {avg_time_a:.2f}s\n")
        f.write(f"Avg time B: {avg_time_b:.2f}s\n")

    print(f"\nFull results written to: {output_path}")


if __name__ == "__main__":
    main()
