from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


path = Path(r"C:\tmp\instant_translate_placeholder_poc.jsonl")
records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
rows = [row for row in records if row["type"] == "translation"]


def q(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


summary = {}
for condition in ("plain", "placeholder"):
    group = [row for row in rows if row["condition"] == condition]
    times = [row["elapsed_s"] for row in group]
    summary[condition] = {
        "n": len(group),
        "success": sum(row["ok"] for row in group),
        "terms_ok": sum(not row["missing_terms"] for row in group),
        "placeholder_ok": sum(not row["missing_placeholders"] for row in group),
        "mean_s": round(statistics.mean(times), 3),
        "median_s": round(statistics.median(times), 3),
        "p95_s": round(q(times, 0.95), 3),
        "max_s": round(max(times), 3),
        "reasoning_total": sum(row["reasoning_chars"] for row in group),
        "completion_tokens_total": sum(row["completion_tokens"] or 0 for row in group),
    }

indexed = {(row["round"], row["sample_id"], row["condition"]): row for row in rows}
plain_slower = 0
placeholder_slower = 0
deltas = []
for round_no in (1, 2, 3):
    for sample_id in range(1, 16):
        plain = indexed[(round_no, sample_id, "plain")]["elapsed_s"]
        protected = indexed[(round_no, sample_id, "placeholder")]["elapsed_s"]
        plain_slower += plain > protected
        placeholder_slower += protected > plain
        deltas.append(protected - plain)

failures = [
    {
        "round": row["round"],
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "input": row["input"],
        "missing_terms": row["missing_terms"],
        "missing_placeholders": row["missing_placeholders"],
        "output": row["restored"],
    }
    for row in rows
    if row["missing_terms"] or row["missing_placeholders"]
]

round_one_outputs = [
    {
        "sample_id": sample_id,
        "input": indexed[(1, sample_id, "placeholder")]["input"],
        "output": indexed[(1, sample_id, "placeholder")]["restored"],
    }
    for sample_id in range(1, 16)
]

print(json.dumps({
    "summary": summary,
    "paired": {
        "plain_slower": plain_slower,
        "placeholder_slower": placeholder_slower,
        "median_placeholder_minus_plain_s": round(statistics.median(deltas), 3),
    },
    "failures": failures,
    "round_one_placeholder_outputs": round_one_outputs,
}, ensure_ascii=False, indent=2))
