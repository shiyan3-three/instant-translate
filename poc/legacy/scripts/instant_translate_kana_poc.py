from __future__ import annotations

import json
import re
import sys
import time

sys.path.insert(0, r"C:\tmp\instant_translate_poc_deps")

import fugashi


CASES = [
    ("上のほうきをここに持ってきてください", "うえのほうきをここにもってきてください"),
    ("最近WindowsからMacに変えました", "さいきんWindowsからMacにかえました"),
    ("価格は約70～80元です", "かかくはやく70～80げんです"),
    ("AI画像生成動画を無料で利用できます", "AIがぞうせいせいどうがをむりょうでりようできます"),
    ("今日、銀行へ行った", "きょう、ぎんこうへいった"),
    ("この作業を行う", "このさぎょうをおこなう"),
    ("生の魚を食べる", "なまのさかなをたべる"),
    ("学生が生まれた町", "がくせいがうまれたまち"),
    ("五月一日に開催する", "ごがつついたちにかいさいする"),
    ("一日中勉強した", "いちにちじゅうべんきょうした"),
    ("日本橋駅に着いた", "にほんばしえきについた"),
    ("明日また会いましょう", "あしたまたあいましょう"),
    ("大人気の商品", "だいにんきのしょうひん"),
    ("重複したデータ", "ちょうふくしたでーた"),
    ("依存関係を確認する", "いぞんかんけいをかくにんする"),
    ("GitHubでPull Requestを作成する", "GitHubでPull Requestをさくせいする"),
    ("ゲーム内のNPCに話しかける", "げーむないのNPCにはなしかける"),
    ("東京スカイツリー", "とうきょうすかいつりー"),
    ("上の房気をここに持ってきて", "うえのぼうきをここにもってきて"),
    ("1人で行く", "ひとりでいく"),
]


ASCII_ONLY = re.compile(r"^[\x00-\x7f]+$")


def katakana_to_hiragana(text: str) -> str:
    chars = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def reading_of(word) -> str:
    if ASCII_ONLY.fullmatch(word.surface):
        return word.surface
    feature = word.feature
    for name in ("kana", "pron", "pronBase"):
        value = getattr(feature, name, None)
        if value and value != "*":
            return katakana_to_hiragana(value)
    return katakana_to_hiragana(word.surface)


def main() -> None:
    started = time.perf_counter()
    tagger = fugashi.Tagger()
    init_ms = (time.perf_counter() - started) * 1000
    rows = []
    conversion_times = []
    for source, expected in CASES:
        t0 = time.perf_counter()
        actual = "".join(reading_of(word) for word in tagger(source))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        conversion_times.append(elapsed_ms)
        rows.append({
            "source": source,
            "expected": expected,
            "actual": actual,
            "exact": actual == expected,
            "elapsed_ms": round(elapsed_ms, 3),
        })
    print(json.dumps({
        "init_ms": round(init_ms, 3),
        "mean_conversion_ms": round(sum(conversion_times) / len(conversion_times), 3),
        "exact": sum(row["exact"] for row in rows),
        "total": len(rows),
        "rows": rows,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
