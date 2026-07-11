from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


path = Path(r"C:\tmp\instant_translate_dual_model_session_poc.jsonl")
records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
rows = [row for row in records if row["type"] == "translation"]
bootstraps = [row for row in records if row["type"] == "bootstrap"]


def q(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


summary = {}
for condition in ("thinking_pro", "fast_flash"):
    group = [row for row in rows if row["condition"] == condition]
    ttfc = [row["ttft_content_s"] for row in group if row["ttft_content_s"] is not None]
    totals = [row["elapsed_s"] for row in group]
    hits = [row["cache_hit_tokens"] or 0 for row in group]
    misses = [row["cache_miss_tokens"] or 0 for row in group]
    term_rows = [row for row in group if row["terms_preserved"] is not None]
    summary[condition] = {
        "n": len(group),
        "success": sum(row["ok"] for row in group),
        "ttfc_median_s": round(statistics.median(ttfc), 3),
        "ttfc_p95_s": round(q(ttfc, 0.95), 3),
        "total_median_s": round(statistics.median(totals), 3),
        "total_mean_s": round(statistics.mean(totals), 3),
        "total_p95_s": round(q(totals, 0.95), 3),
        "total_max_s": round(max(totals), 3),
        "format_ok": sum(row["format_ok"] for row in group),
        "terms_ok": sum(row["terms_preserved"] for row in term_rows),
        "terms_total": len(term_rows),
        "reasoning_chars_total": sum(row["reasoning_chars"] for row in group),
        "prompt_tokens_total": sum(row["prompt_tokens"] or 0 for row in group),
        "completion_tokens_total": sum(row["completion_tokens"] or 0 for row in group),
        "cache_hit_total": sum(hits),
        "cache_miss_total": sum(misses),
        "cache_hit_ratio": round(sum(hits) / max(1, sum(hits) + sum(misses)), 4),
    }

rounds = defaultdict(dict)
for round_no in (1, 2, 3):
    for condition in ("thinking_pro", "fast_flash"):
        group = [row for row in rows if row["round"] == round_no and row["condition"] == condition]
        rounds[round_no][condition] = {
            "ttfc_median_s": round(statistics.median(row["ttft_content_s"] for row in group), 3),
            "total_median_s": round(statistics.median(row["elapsed_s"] for row in group), 3),
            "format_ok": sum(row["format_ok"] for row in group),
            "terms_ok": sum(row["terms_preserved"] for row in group if row["terms_preserved"] is not None),
            "terms_total": sum(row["terms_preserved"] is not None for row in group),
        }

format_failures = [
    {
        "round": row["round"],
        "sample_id": row["sample_id"],
        "input": row["input"],
        "output": row["content"],
        "ttfc_s": row["ttft_content_s"],
        "total_s": row["elapsed_s"],
    }
    for row in rows
    if not row["format_ok"]
]

term_failures = [
    {
        "round": row["round"],
        "condition": row["condition"],
        "sample_id": row["sample_id"],
        "input": row["input"],
        "output": row["content"],
    }
    for row in rows
    if row["terms_preserved"] is False
]

first_vs_later = {}
for condition in ("thinking_pro", "fast_flash"):
    first_rows = []
    later_rows = []
    for round_no in (1, 2, 3):
        group = [row for row in rows if row["round"] == round_no and row["condition"] == condition]
        group.sort(key=lambda row: row["timestamp"])
        first_rows.append(group[0])
        later_rows.extend(group[1:])
    first_vs_later[condition] = {
        "first_total_median_s": round(statistics.median(row["elapsed_s"] for row in first_rows), 3),
        "later_total_median_s": round(statistics.median(row["elapsed_s"] for row in later_rows), 3),
        "first_cache_hit_median": statistics.median((row["cache_hit_tokens"] or 0) for row in first_rows),
        "later_cache_hit_median": statistics.median((row["cache_hit_tokens"] or 0) for row in later_rows),
    }

bootstrap_summary = {
    "n": len(bootstraps),
    "ttfc_s": [row["ttft_content_s"] for row in bootstraps],
    "total_s": [row["elapsed_s"] for row in bootstraps],
    "cache_hits": [row["cache_hit_tokens"] for row in bootstraps],
    "contents": [row["content"] for row in bootstraps],
}

long_outputs = [
    {
        "round": row["round"],
        "condition": row["condition"],
        "sample_id": row["sample_id"],
        "ratio": round(len(row["content"]) / max(1, len(row["input"])), 2),
        "input": row["input"],
        "output": row["content"],
    }
    for row in sorted(rows, key=lambda row: len(row["content"]) / max(1, len(row["input"])), reverse=True)[:8]
]

print(json.dumps({
    "summary": summary,
    "rounds": rounds,
    "first_vs_later": first_vs_later,
    "bootstrap": bootstrap_summary,
    "format_failures": format_failures,
    "term_failures": term_failures,
    "long_outputs": long_outputs,
}, ensure_ascii=False, indent=2))
