from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\tmp\instant_translate_poc_deps")

import fugashi


SOURCE = Path(r"C:\tmp\instant_translate_full_poc.jsonl")
ASCII_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+.#/_-]*")
ASCII_ONLY = re.compile(r"^[\x00-\x7f]+$")
SPACE_SPLIT = re.compile(r"(\s+)")
FORBIDDEN = re.compile(r"[\u3400-\u9fff\u30a1-\u30fa]")


def kata_to_hira(text: str) -> str:
    result = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            result.append(chr(code - 0x60))
        else:
            result.append(char)
    return "".join(result)


def convert_segment(tagger, segment: str) -> tuple[str, list[str]]:
    converted = []
    unknown = []
    for word in tagger(segment):
        if ASCII_ONLY.fullmatch(word.surface):
            converted.append(word.surface)
            continue
        feature = word.feature
        reading = None
        for name in ("kana", "pron", "pronBase"):
            value = getattr(feature, name, None)
            if value and value != "*":
                reading = value
                break
        if reading is None:
            converted.append(kata_to_hira(word.surface))
            if FORBIDDEN.search(word.surface):
                unknown.append(word.surface)
        else:
            converted.append(kata_to_hira(reading))
    return "".join(converted), unknown


def convert(tagger, text: str) -> tuple[str, list[str]]:
    parts = []
    unknown = []
    for part in SPACE_SPLIT.split(text):
        if not part:
            continue
        if part.isspace():
            parts.append(part)
            continue
        converted, missed = convert_segment(tagger, part)
        parts.append(converted)
        unknown.extend(missed)
    return "".join(parts), unknown


def ascii_preserved(source: str, converted: str) -> bool:
    expected = ASCII_RUN.findall(source)
    return all(token in converted for token in expected)


def main() -> None:
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [
        row["content"]
        for row in records
        if row.get("type") == "translation" and row.get("condition") == "B_split_thinking"
    ][:50]

    t0 = time.perf_counter()
    tagger = fugashi.Tagger()
    init_ms = (time.perf_counter() - t0) * 1000
    rows = []
    for index, text in enumerate(texts, 1):
        started = time.perf_counter()
        converted, unknown = convert(tagger, text)
        elapsed_ms = (time.perf_counter() - started) * 1000
        rows.append({
            "id": index,
            "source": text,
            "converted": converted,
            "elapsed_ms": round(elapsed_ms, 3),
            "format_ok": FORBIDDEN.search(converted) is None,
            "ascii_preserved": ascii_preserved(text, converted),
            "whitespace_preserved": re.findall(r"\s+", text) == re.findall(r"\s+", converted),
            "unknown": unknown,
        })

    times = [row["elapsed_ms"] for row in rows]
    print(json.dumps({
        "n": len(rows),
        "init_ms": round(init_ms, 3),
        "mean_ms": round(statistics.mean(times), 3),
        "max_ms": round(max(times), 3),
        "format_ok": sum(row["format_ok"] for row in rows),
        "ascii_preserved": sum(row["ascii_preserved"] for row in rows),
        "whitespace_preserved": sum(row["whitespace_preserved"] for row in rows),
        "unknown_rows": sum(bool(row["unknown"]) for row in rows),
        "problem_rows": [row for row in rows if not row["format_ok"] or not row["ascii_preserved"] or not row["whitespace_preserved"] or row["unknown"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
