from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PATH = Path(r"C:\tmp\instant_translate_full_poc.jsonl")
CONDITIONS = (
    "A_current_thinking",
    "B_split_thinking",
    "C_split_requested_nonthinking",
    "D_split_stateless_requested_nonthinking",
)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    return cov / (statistics.pstdev(xs) * statistics.pstdev(ys))


def ascii_terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+[A-Za-z0-9]*", text)


def terms_preserved(row: dict) -> bool | None:
    terms = ascii_terms(row["input"])
    if not terms:
        return None
    output = row["content"].casefold()
    return all(term.casefold() in output for term in terms)


def hiragana_constraint_ok(text: str) -> bool:
    return not re.search(r"[\u3400-\u9fff\u30a1-\u30fa]", text)


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = max(wins, losses)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


records = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
rows = [row for row in records if row["type"] == "translation"]
digests = [row for row in records if row["type"] == "digest"]

by_condition = defaultdict(list)
for row in rows:
    by_condition[row["condition"]].append(row)

summary = {}
for condition in CONDITIONS:
    group = by_condition[condition]
    times = [row["elapsed_s"] for row in group]
    reasoning = [row["reasoning_chars"] for row in group]
    prompt_tokens = [row["prompt_tokens"] for row in group if row["prompt_tokens"] is not None]
    completion_tokens = [row["completion_tokens"] for row in group if row["completion_tokens"] is not None]
    term_rows = [terms_preserved(row) for row in group]
    term_rows = [value for value in term_rows if value is not None]
    summary[condition] = {
        "n": len(group),
        "success": sum(row["ok"] for row in group),
        "min_s": round(min(times), 3),
        "mean_s": round(statistics.mean(times), 3),
        "median_s": round(statistics.median(times), 3),
        "p90_s": round(quantile(times, 0.90), 3),
        "p95_s": round(quantile(times, 0.95), 3),
        "max_s": round(max(times), 3),
        "reasoning_chars_median": round(statistics.median(reasoning), 1),
        "reasoning_chars_total": sum(reasoning),
        "prompt_tokens_total": sum(prompt_tokens),
        "completion_tokens_total": sum(completion_tokens),
        "latency_reasoning_corr": round(corr(times, reasoning), 3),
        "term_preserved": sum(term_rows),
        "term_total": len(term_rows),
        "hiragana_ok": sum(hiragana_constraint_ok(row["content"]) for row in group) if condition.startswith("A_") else None,
    }

rounds = defaultdict(dict)
for round_no in (1, 2, 3):
    for condition in CONDITIONS:
        times = [row["elapsed_s"] for row in by_condition[condition] if row["round"] == round_no]
        rounds[round_no][condition] = {
            "median_s": round(statistics.median(times), 3),
            "p95_s": round(quantile(times, 0.95), 3),
        }

indexed = {(row["round"], row["sample_id"], row["condition"]): row for row in rows}
comparisons = {}
for other in CONDITIONS[1:]:
    pairs = []
    for round_no in (1, 2, 3):
        for sample_id in range(1, 21):
            a = indexed[(round_no, sample_id, "A_current_thinking")]["elapsed_s"]
            b = indexed[(round_no, sample_id, other)]["elapsed_s"]
            pairs.append((a, b))
    wins = sum(a > b for a, b in pairs)
    losses = sum(a < b for a, b in pairs)
    ties = len(pairs) - wins - losses
    ratios = [a / b for a, b in pairs if b]
    deltas = [a - b for a, b in pairs]
    comparisons[f"A_vs_{other[0]}"] = {
        "A_slower": wins,
        "A_faster": losses,
        "ties": ties,
        "median_A_over_other": round(statistics.median(ratios), 3),
        "median_saved_s": round(statistics.median(deltas), 3),
        "sign_test_p": sign_test_two_sided(wins, losses),
    }

sample_stats = []
for sample_id in range(1, 21):
    source = indexed[(1, sample_id, "A_current_thinking")]["input"]
    medians = {
        condition[0]: round(statistics.median([
            indexed[(round_no, sample_id, condition)]["elapsed_s"]
            for round_no in (1, 2, 3)
        ]), 3)
        for condition in CONDITIONS
    }
    sample_stats.append({
        "sample_id": sample_id,
        "input": source,
        "medians": medians,
        "A_minus_B": round(medians["A"] - medians["B"], 3),
    })
sample_stats.sort(key=lambda item: item["A_minus_B"], reverse=True)

consistency = {}
for condition in CONDITIONS:
    exact_same = 0
    unique_counts = []
    for sample_id in range(1, 21):
        outputs = {
            indexed[(round_no, sample_id, condition)]["content"]
            for round_no in (1, 2, 3)
        }
        unique_counts.append(len(outputs))
        if len(outputs) == 1:
            exact_same += 1
    consistency[condition] = {
        "samples_identical_all_3_rounds": exact_same,
        "mean_unique_outputs": round(statistics.mean(unique_counts), 2),
    }

digest_summary = defaultdict(list)
for row in digests:
    digest_summary[row["condition"]].append(row["elapsed_s"])
digest_summary = {
    condition: {
        "n": len(times),
        "median_s": round(statistics.median(times), 3),
        "times_s": times,
    }
    for condition, times in digest_summary.items()
}

print(json.dumps({
    "summary": summary,
    "rounds": rounds,
    "comparisons": comparisons,
    "top_sample_differences": sample_stats[:10],
    "consistency": consistency,
    "digests": digest_summary,
}, ensure_ascii=False, indent=2))
