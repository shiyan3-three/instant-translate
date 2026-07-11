"""
POC: Multi-Check Mechanism (Programmatic + Parallel Flash + Fix)
Combined with Reference Layer Split (Approach B) from previous POC.

Three approaches compared:
  A:   Full system prompt (glossary in system prompt), no checks
  B:   Split system prompt + glossary injection, no checks
  B+C: Split system prompt + glossary injection + multi-check + fix

Multi-check flow for B+C:
  1. Flash translates (approach B)
  2. Programmatic checks (instant): hiragana-only, bracket terms, edge spaces
  3. Parallel flash checks (~2s): completeness, segmentation/double-spaces
  4. Fix pass (~2s, only if issues found): flash corrects all issues
"""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.settings import AppSettings
from app.translation.client import ClientConfig, OpenAICompatibleClient, TranslationError

# =========================================================================
# Glossary (same as previous POC)
# =========================================================================

GLOSSARY: dict[str, str] = {
    "软件": "[そふとうぇあ]", "硬件": "[はーどうぇあ]", "程序": "[ぷろぐらむ]",
    "编译": "[こんぱいる]", "调试": "[でばっぐ]", "算法": "[あるごりずむ]",
    "数据库": "[でーたべーす]", "接口": "[いんたーふぇーす]", "服务器": "[さーばー]",
    "客户端": "[くらいあんと]", "代码": "[こーど]", "变量": "[へんすう]",
    "函数": "[かんすう]", "对象": "[おぶじぇくと]", "开发": "[かいはつ]",
    "部署": "[でぷろい]", "版本": "[ばーじょん]", "错误": "[ばぐ]",
    "补丁": "[ぱっち]", "测试": "[てすと]",
}

# =========================================================================
# System prompts (same as previous POC)
# =========================================================================

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

_GLOSSARY_NOTE = """Glossary Entries (Software Engineering Terms)
Relevant glossary entries for terms present in the source text are provided per-request in <RELEVANT_GLOSSARY> tags in the user message. For any software engineering term not provided there, generate its hiragana reading and enclose in brackets. You must still translate the complete source sentence, not just output the glossary terms.
"""

_STYLE_AND_FIXED = """Style Guidance
- Tokenize the Japanese text into words (dictionary-based segmentation, morphological analysis). Use the word boundaries as spacing points.
- No particles are treated as separate words; retain them within the word they attach to in hiragana flow, but ensure double spaces between distinct lexical items.
- If a term is in brackets, treat the whole bracket as a single word unit; double-space before and after it.
- Example: "打开软件" -> ひらく  [そふとうぇあ]  (double spaces shown between ひらく and the bracket, and after the bracket if sentence continues).

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

SYSTEM_PROMPT_A = _BASE_PROMPT + _GLOSSARY_TABLE + _STYLE_AND_FIXED
SYSTEM_PROMPT_B = _BASE_PROMPT + _GLOSSARY_NOTE + _STYLE_AND_FIXED

# =========================================================================
# Test cases (same as previous POC)
# =========================================================================

TEST_CASES: list[str] = [
    "软件开发过程中，调试和测试是程序必不可少的环节",
    "数据库连接失败，检查服务器的接口配置",
    "这个函数的变量名不符合命名规范",
    "部署新版本之前，先备份代码并打好补丁",
    "今天的会议推迟到下午三点，请通知所有参会人员",
    "编译时算法报错，错误指向了一个空对象",
    "客户端的硬件配置太低，连不上服务器",
    "迭代版本的测试覆盖了核心算法和数据库",
]

# =========================================================================
# Keyword matching (same as previous POC)
# =========================================================================

def match_glossary(text: str, glossary: dict[str, str]) -> dict[str, str]:
    matched: dict[str, str] = {}
    for term, reading in glossary.items():
        if term in text:
            matched[term] = reading
    return matched

def build_glossary_block(matched: dict[str, str]) -> str:
    if not matched:
        return ""
    lines = [f"{term} -> {reading}" for term, reading in matched.items()]
    return "<RELEVANT_GLOSSARY>\n" + "\n".join(lines) + "\n</RELEVANT_GLOSSARY>\n"

# =========================================================================
# Programmatic checks (instant, no API)
# =========================================================================

def check_hiragana_only(text: str) -> tuple[bool, str]:
    """Check: only hiragana, prolonged mark, brackets, spaces."""
    pattern = re.compile(r"^[あ-んー\s\[\]]+$")
    if pattern.match(text):
        return True, ""
    bad_chars = [c for c in text if not re.match(r"[あ-んー\s\[\]]", c)]
    return False, f"Non-hiragana chars: {bad_chars}"

def check_bracket_terms(text: str, matched: dict[str, str]) -> tuple[bool, str]:
    """Check: all matched glossary readings appear in text."""
    missing = []
    for term, reading in matched.items():
        if reading not in text:
            missing.append(f"{term}->{reading}")
    if missing:
        return False, f"Missing brackets: {', '.join(missing)}"
    return True, ""

def check_edge_spaces(text: str) -> tuple[bool, str]:
    """Check: no leading/trailing spaces."""
    if text != text.strip():
        return False, "Has leading/trailing spaces"
    return True, ""

def run_programmatic_checks(translation: str, matched: dict[str, str]) -> list[str]:
    """Run all programmatic checks, return list of issue strings."""
    issues = []
    for check_fn, label in [
        (lambda t: check_hiragana_only(t), "hiragana"),
        (lambda t: check_bracket_terms(t, matched) if matched else (True, ""), "brackets"),
        (lambda t: check_edge_spaces(t), "edges"),
    ]:
        passed, msg = check_fn(translation)
        if not passed:
            issues.append(f"[{label}] {msg}")
    return issues

# =========================================================================
# Parallel flash LLM checks
# =========================================================================

def llm_check_completeness(client: OpenAICompatibleClient, model: str, source: str, translation: str) -> tuple[bool, str]:
    """Check: does translation cover the full source meaning?"""
    system = "You are a translation quality checker. Check if the Japanese translation completely covers the meaning of the Chinese source text. If content is missing or incomplete, describe what is missing. If complete, output only PASS."
    user = f"Source: {source}\nTranslation: {translation}\n\nIs the translation complete? Output PASS or describe the issue."
    try:
        result = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            thinking="disabled",
        )
        passed = result.strip().upper().startswith("PASS")
        return passed, "" if passed else result.strip()
    except TranslationError as exc:
        return False, f"(check error: {exc})"

def llm_check_segmentation(client: OpenAICompatibleClient, model: str, translation: str) -> tuple[bool, str]:
    """Check: are there single spaces that should be double?"""
    system = "You are a Japanese text formatter. The rule is: exactly two spaces (U+0020) between every word. Check if the translation has any single spaces between words that should be double spaces. If all spaces are correct, output only PASS. If there are single-space issues, output the corrected text."
    user = f"Translation: {translation}\n\nCheck spacing. Output PASS or the corrected text."
    try:
        result = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            thinking="disabled",
        )
        passed = result.strip().upper().startswith("PASS")
        return passed, "" if passed else result.strip()
    except TranslationError as exc:
        return False, f"(check error: {exc})"

def run_parallel_llm_checks(client: OpenAICompatibleClient, model: str, source: str, translation: str) -> tuple[list[str], float]:
    """Run LLM checks in parallel. Returns (issues, elapsed_seconds)."""
    issues: list[str] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(llm_check_completeness, client, model, source, translation): "completeness",
            executor.submit(llm_check_segmentation, client, model, translation): "segmentation",
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                passed, msg = future.result()
                if not passed:
                    issues.append(f"[{label}] {msg}")
            except Exception as exc:
                issues.append(f"[{label}] (exception: {exc})")

    elapsed = time.perf_counter() - t0
    return issues, elapsed

# =========================================================================
# Fix pass
# =========================================================================

def fix_translation(client: OpenAICompatibleClient, model: str, translation: str, issues: list[str]) -> str:
    """Send translation + issues to flash for correction."""
    system = "You are a translation fixer. Fix the listed issues in the Japanese translation. Output ONLY the corrected Japanese translation, nothing else. The translation must use only hiragana, brackets for software terms, and exactly two spaces between words."
    issues_text = "\n".join(f"- {issue}" for issue in issues)
    user = f"Current translation:\n{translation}\n\nIssues to fix:\n{issues_text}\n\nOutput the corrected translation only:"
    try:
        result = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            thinking="disabled",
        )
        return result.strip()
    except TranslationError:
        return translation  # return original if fix fails

# =========================================================================
# Validation (with fixed hiragana check)
# =========================================================================

def validate_output(text: str, matched: dict[str, str]) -> dict:
    result = {}
    hiragana_pattern = re.compile(r"^[あ-んー\s\[\]]+$")
    result["hiragana"] = bool(hiragana_pattern.match(text))
    result["brackets"] = ("[" in text and "]" in text) if matched else True
    result["dbl_space"] = "  " in text if text.strip() else True
    result["edges"] = text == text.strip()
    result["all_pass"] = all(v for v in result.values())
    return result

def check_missing_terms(text: str, matched: dict[str, str]) -> list[str]:
    missing = []
    for term, reading in matched.items():
        if reading not in text:
            missing.append(f"{term}->{reading}")
    return missing

# =========================================================================
# Token estimation
# =========================================================================

def estimate_tokens(text: str) -> int:
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u309f")
    return cjk + max(1, (len(text) - cjk) // 4)

# =========================================================================
# Main
# =========================================================================

def main() -> None:
    print("=" * 70)
    print("POC: Multi-Check Mechanism + Reference Layer Split")
    print("=" * 70)

    settings = AppSettings.load()
    ai = settings.ai
    model = ai.fast_model_name
    if not model:
        print("ERROR: No model configured.")
        return
    print(f"Model: {model}")

    client = OpenAICompatibleClient(ClientConfig(
        base_url=ai.base_url, api_key=ai.api_key, model=model, timeout_seconds=60.0,
    ))

    tokens_a = estimate_tokens(SYSTEM_PROMPT_A)
    tokens_b = estimate_tokens(SYSTEM_PROMPT_B)
    print(f"System Prompt A: ~{tokens_a} tokens")
    print(f"System Prompt B: ~{tokens_b} tokens")
    print(f"Savings (B vs A): ~{tokens_a - tokens_b} tokens/request")

    results: list[dict] = []

    for i, ocr_text in enumerate(TEST_CASES):
        print(f"\n{'=' * 70}")
        print(f"Test {i+1}/{len(TEST_CASES)}: {ocr_text}")

        matched = match_glossary(ocr_text, GLOSSARY)
        print(f"Matched ({len(matched)}): {', '.join(f'{k}' for k in matched) or '(none)'}")

        user_msg_a = f"<OCR_TEXT>\n{ocr_text}\n</OCR_TEXT>"
        glossary_block = build_glossary_block(matched)
        user_msg_b = f"{glossary_block}<OCR_TEXT>\n{ocr_text}\n</OCR_TEXT>"

        # --- Approach A: full system prompt ---
        t0 = time.perf_counter()
        try:
            result_a = client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT_A},
                 {"role": "user", "content": user_msg_a}],
                thinking="disabled",
            )
        except TranslationError as exc:
            result_a = f"(ERROR: {exc})"
        time_a = time.perf_counter() - t0
        val_a = validate_output(result_a, matched)
        miss_a = check_missing_terms(result_a, matched)
        print(f"\n  [A] ({time_a:.1f}s): {result_a}")
        print(f"      pass={val_a['all_pass']} miss={len(miss_a)}")

        time.sleep(0.3)

        # --- Approach B: split + injection ---
        t0 = time.perf_counter()
        try:
            result_b = client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT_B},
                 {"role": "user", "content": user_msg_b}],
                thinking="disabled",
            )
        except TranslationError as exc:
            result_b = f"(ERROR: {exc})"
        time_b = time.perf_counter() - t0
        val_b = validate_output(result_b, matched)
        miss_b = check_missing_terms(result_b, matched)
        print(f"  [B] ({time_b:.1f}s): {result_b}")
        print(f"      pass={val_b['all_pass']} miss={len(miss_b)}")

        time.sleep(0.3)

        # --- Approach B+C: split + multi-check + fix ---
        t0_total = time.perf_counter()

        # Step 1: translate (same as B)
        t0 = time.perf_counter()
        try:
            result_bc = client.chat(
                [{"role": "system", "content": SYSTEM_PROMPT_B},
                 {"role": "user", "content": user_msg_b}],
                thinking="disabled",
            )
        except TranslationError as exc:
            result_bc = f"(ERROR: {exc})"
        time_translate = time.perf_counter() - t0

        # Step 2: programmatic checks (instant)
        prog_issues = run_programmatic_checks(result_bc, matched)
        print(f"\n  [B+C] translate ({time_translate:.1f}s): {result_bc}")
        print(f"        prog checks: {len(prog_issues)} issues")

        # Step 3: parallel LLM checks
        llm_issues, time_llm = run_parallel_llm_checks(client, model, ocr_text, result_bc)
        print(f"        LLM checks ({time_llm:.1f}s): {len(llm_issues)} issues")

        # Step 4: fix if needed
        all_issues = prog_issues + llm_issues
        if all_issues:
            time.sleep(0.2)
            t0 = time.perf_counter()
            result_bc_fixed = fix_translation(client, model, result_bc, all_issues)
            time_fix = time.perf_counter() - t0
            print(f"        fix ({time_fix:.1f}s): {result_bc_fixed}")
        else:
            result_bc_fixed = result_bc
            time_fix = 0.0
            print(f"        fix: (skipped, no issues)")

        time_bc = time.perf_counter() - t0_total
        val_bc = validate_output(result_bc_fixed, matched)
        miss_bc = check_missing_terms(result_bc_fixed, matched)
        print(f"        final pass={val_bc['all_pass']} miss={len(miss_bc)} total_time={time_bc:.1f}s")

        results.append({
            "test": i + 1,
            "ocr": ocr_text,
            "matched": matched,
            "result_a": result_a, "time_a": time_a, "val_a": val_a, "miss_a": miss_a,
            "result_b": result_b, "time_b": time_b, "val_b": val_b, "miss_b": miss_b,
            "result_bc": result_bc_fixed, "time_bc": time_bc, "val_bc": val_bc, "miss_bc": miss_bc,
            "issues_found": all_issues,
            "was_fixed": len(all_issues) > 0,
        })

        time.sleep(0.5)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    a_pass = sum(1 for r in results if r["val_a"]["all_pass"])
    b_pass = sum(1 for r in results if r["val_b"]["all_pass"])
    bc_pass = sum(1 for r in results if r["val_bc"]["all_pass"])
    a_miss = sum(len(r["miss_a"]) for r in results)
    b_miss = sum(len(r["miss_b"]) for r in results)
    bc_miss = sum(len(r["miss_bc"]) for r in results)
    avg_a = sum(r["time_a"] for r in results) / len(results)
    avg_b = sum(r["time_b"] for r in results) / len(results)
    avg_bc = sum(r["time_bc"] for r in results) / len(results)
    fixed_count = sum(1 for r in results if r["was_fixed"])
    bc_improved = sum(1 for r in results if not r["val_b"]["all_pass"] and r["val_bc"]["all_pass"])

    print(f"Test cases:              {len(results)}")
    print(f"A  all pass:             {a_pass}/{len(results)}")
    print(f"B  all pass:             {b_pass}/{len(results)}")
    print(f"B+C all pass:            {bc_pass}/{len(results)}")
    print(f"A  missing terms:        {a_miss}")
    print(f"B  missing terms:        {b_miss}")
    print(f"B+C missing terms:       {bc_miss}")
    print(f"Avg time A:              {avg_a:.2f}s")
    print(f"Avg time B:              {avg_b:.2f}s")
    print(f"Avg time B+C:            {avg_bc:.2f}s")
    print(f"B+C fixes triggered:     {fixed_count}/{len(results)}")
    print(f"B+C improved over B:     {bc_improved}/{len(results)}")

    print(f"\n{'Test':<6} {'A_pass':<8} {'B_pass':<8} {'BC_pass':<8} {'A_miss':<8} {'B_miss':<8} {'BC_miss':<8} {'A_t':<6} {'B_t':<6} {'BC_t':<6} {'Fixed':<6}")
    for r in results:
        print(f"{r['test']:<6} "
              f"{'Y' if r['val_a']['all_pass'] else 'N':<8} "
              f"{'Y' if r['val_b']['all_pass'] else 'N':<8} "
              f"{'Y' if r['val_bc']['all_pass'] else 'N':<8} "
              f"{len(r['miss_a']):<8} {len(r['miss_b']):<8} {len(r['miss_bc']):<8} "
              f"{r['time_a']:<6.1f} {r['time_b']:<6.1f} {r['time_bc']:<6.1f} "
              f"{'Y' if r['was_fixed'] else 'N':<6}")

    # Write full results
    output_path = os.path.join("poc", "results", "multi-check-results.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("POC: Multi-Check + Reference Layer Split\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"System A: ~{tokens_a} tokens | System B: ~{tokens_b} tokens | Savings: ~{tokens_a-tokens_b}\n\n")

        for r in results:
            f.write(f"{'=' * 70}\n")
            f.write(f"Test {r['test']}: {r['ocr']}\n")
            f.write(f"Matched: {', '.join(r['matched'].keys()) or '(none)'}\n\n")
            f.write(f"[A] ({r['time_a']:.1f}s): {r['result_a']}\n")
            f.write(f"    pass={r['val_a']} miss={r['miss_a']}\n\n")
            f.write(f"[B] ({r['time_b']:.1f}s): {r['result_b']}\n")
            f.write(f"    pass={r['val_b']} miss={r['miss_b']}\n\n")
            f.write(f"[B+C] ({r['time_bc']:.1f}s): {r['result_bc']}\n")
            f.write(f"     pass={r['val_bc']} miss={r['miss_bc']}\n")
            if r["was_fixed"]:
                f.write(f"     issues: {r['issues_found']}\n")
            f.write("\n")

        f.write(f"{'=' * 70}\nSUMMARY\n{'=' * 70}\n")
        f.write(f"A pass: {a_pass}/{len(results)} | B pass: {b_pass}/{len(results)} | B+C pass: {bc_pass}/{len(results)}\n")
        f.write(f"A miss: {a_miss} | B miss: {b_miss} | B+C miss: {bc_miss}\n")
        f.write(f"Avg time: A={avg_a:.2f}s B={avg_b:.2f}s B+C={avg_bc:.2f}s\n")
        f.write(f"Fixes: {fixed_count} | Improved: {bc_improved}\n")

    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
