from __future__ import annotations

import json


def is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def is_katakana(ch: str) -> bool:
    return "\u30a0" <= ch <= "\u30ff"


def is_latin(ch: str) -> bool:
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


def main() -> None:
    path = r"C:\tmp\instant_translate_stage1_integration_results.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    translations = [row for row in rows if row.get("type") == "translation"]
    print("n", len(translations))
    for row in translations:
        content = row.get("content") or ""
        cjk = "".join(ch for ch in content if is_cjk(ch))
        katakana = "".join(ch for ch in content if is_katakana(ch))
        latin = "".join(ch for ch in content if is_latin(ch))
        print(
            f"{row['index']:02d} {row['elapsed_s']}s "
            f"cjk={cjk!r} kata={katakana!r} latin={latin!r} "
            f"content={content!r}"
        )


if __name__ == "__main__":
    main()
