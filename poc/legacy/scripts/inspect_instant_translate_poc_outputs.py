from __future__ import annotations

import json
import re
from pathlib import Path


path = Path(r"C:\tmp\instant_translate_full_poc.jsonl")
rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and json.loads(line).get("type") == "translation"
]


def terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+[A-Za-z0-9]*", text)


failures = []
for row in rows:
    expected = terms(row["input"])
    missing = [term for term in expected if term.casefold() not in row["content"].casefold()]
    if missing:
        failures.append({
            "round": row["round"],
            "sample_id": row["sample_id"],
            "condition": row["condition"],
            "missing": missing,
            "input": row["input"],
            "output": row["content"],
        })

length_outliers = sorted(
    rows,
    key=lambda row: len(row["content"]) / max(1, len(row["input"])),
    reverse=True,
)[:12]
length_outliers = [
    {
        "round": row["round"],
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "ratio": round(len(row["content"]) / max(1, len(row["input"])), 2),
        "input": row["input"],
        "output": row["content"],
    }
    for row in length_outliers
]

slowest = sorted(rows, key=lambda row: row["elapsed_s"], reverse=True)[:10]
slowest = [
    {
        "round": row["round"],
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "elapsed_s": row["elapsed_s"],
        "reasoning_chars": row["reasoning_chars"],
        "input": row["input"],
        "output": row["content"],
    }
    for row in slowest
]

print(json.dumps({"term_failures": failures, "length_outliers": length_outliers, "slowest": slowest}, ensure_ascii=False, indent=2))
