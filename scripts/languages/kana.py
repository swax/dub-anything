"""Japanese text as its reading in hiragana, so a transcript and a script compare by
what is said, not by how it is written (漢字 or かな, カタカナ or ひらがな, 3 or 三).

Usage: as a module, kana.readings(["今日は3人", ...]) -> ["きょうはさんにん", ...]
       scripts/languages/kana.py   a worker in its own environment (UV_RUN): a JSON
                                   list of texts per line in, a JSON list of
                                   readings per line out

The readings come from pykakasi's dictionary. It isn't in VoiceStudio's
environment, so readings() starts this file once as a worker in a throwaway uv
environment when it can't import it (uv on PATH, as for vc.py). Digits are read
as kanji numerals first ("3人" as "三人"), as Whisper writes numbers either way.
hiragana() reads a text as it is written, and needs pykakasi where it runs
(second.py's environment has it).
"""

import json
import pathlib
import re
import subprocess
import sys
import unicodedata
from decimal import Decimal, localcontext

UV_RUN = ["uv", "run", "--no-project", "--python", "3.12", "--with", "pykakasi", "python"]
DIGITS = "〇一二三四五六七八九"


def kansuji(n):
    """A number in kanji numerals, as it is read: 2026 -> 二千二十六, 10 -> 十."""
    if n == 0:
        return "〇"
    out = ""
    for unit, name in ((10**8, "億"), (10**4, "万")):
        if n >= unit:
            out += kansuji(n // unit) + name
            n %= unit
    for unit, name in ((1000, "千"), (100, "百"), (10, "十")):
        if n >= unit:
            q = n // unit
            out += ("" if q == 1 else DIGITS[q]) + name
            n %= unit
    return out + (DIGITS[n] if n else "")


UNITS = {"十": 10, "百": 100, "千": 1000}
BIG = {"万": 10**4, "億": 10**8}

# Source-reference checks retain ordinary kanji spelling: only written numeric
# forms are equivalent here, not homophones or dictionary readings.
SOURCE_DIGITS = str.maketrans("零" + DIGITS, "0" + "0123456789")
SOURCE_PLACES = {**UNITS, **BIG, "兆": 10**12}
SOURCE_NUMBER = re.compile(
    r"(?:マイナス|[-−+])?[0-9零〇一二三四五六七八九十百千万億兆]+"
    r"(?:[.点][0-9零〇一二三四五六七八九]+[万億兆]?)?"
)
COUNTERS = "円年月日時分秒人個枚本台回歳才番号冊杯匹頭軒件割倍組階度歩勝敗位"


def _source_integer(token):
    if token.isdecimal():
        return token  # Preserve leading zeroes in identifiers, e.g. 〇三号 / 03号.
    total = section = 0
    current = None
    small, big = 10000, 10**16
    for c in token:
        if c.isdecimal():
            current = (current or 0) * 10 + int(c)
        else:
            place = SOURCE_PLACES[c]
            if place < 10000:
                if place >= small or current == 0:
                    return None
                section += (current or 1) * place
                small = place
            else:
                if place >= big or not section + (current or 0):
                    return None
                total += (section + (current or 0)) * place
                section = 0
                small, big = 10000, place
            current = None
    return str(total + section + (current or 0))


def normalize_source_numbers(text):
    """Canonical source text and ordered written numbers; no kana/ASR dependency.

    Accept digit, kanji and mixed place-value forms. Preserve leading zeroes,
    signs, nonnumeric compounds and unrecognized numeral syntax for review.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    values = []

    def replace(match):
        original = match.group()
        following = text[match.end() : match.end() + 1]
        if (
            len(original) == 1
            and original in "零" + DIGITS + "十百千万億兆"
            and following
            and "\u4e00" <= following <= "\u9fff"
            and (following not in COUNTERS or original + following == "十分")
        ):
            return original  # 一緒, 千葉, and ambiguous 十分 are not inferred numbers.
        token = original.translate(SOURCE_DIGITS).replace("点", ".")
        negative = token.startswith(("-", "−", "マイナス"))
        token = re.sub(r"^(?:マイナス|[-−+])", "", token)
        whole, dot, fraction = token.partition(".")
        value = _source_integer(whole)
        if value is None:
            return original
        if dot:
            scale = SOURCE_PLACES[fraction[-1]] if fraction[-1] in SOURCE_PLACES else 1
            if scale != 1:
                fraction = fraction[:-1]
            with localcontext() as context:
                context.prec = len(value) + len(fraction) + 16
                value = format(Decimal(value + "." + fraction) * scale, "f")
            if "." in value:
                value = value.rstrip("0").rstrip(".")
        value = ("-" if negative else "") + value
        values.append(value)
        return value

    return SOURCE_NUMBER.sub(replace, text), values


def numbers(text):
    """The numbers in a text, written in digits or kanji ("三十年代" 30, "2026年" 2026,
    "二〇二六" 2026). A lone 一 is left out: it's in too many words (一家, 一緒)."""
    out = []
    for run in re.findall(r"\d+|[〇一二三四五六七八九十百千万億]+", unicodedata.normalize("NFKC", text)):
        if run == "一":
            continue
        if run.isdigit():
            out.append(int(run))
        elif not any(c in UNITS or c in BIG for c in run):
            out.append(int("".join(str(DIGITS.index(c)) for c in run)))  # read digit by digit
        else:
            total = section = digit = 0
            for c in run:
                if c in UNITS:
                    section += (digit or 1) * UNITS[c]
                    digit = 0
                elif c in BIG:
                    total += (section + digit) * BIG[c]
                    section = digit = 0
                else:
                    digit = DIGITS.index(c)
            out.append(total + section + digit)
    return out


def numerals(text):
    """Digits as kanji numerals; a long run (a phone number) digit by digit."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return re.sub(
        r"\d+",
        lambda m: kansuji(int(m.group())) if len(m.group()) <= 12 else "".join(DIGITS[int(d)] for d in m.group()),
        text,
    )


def hiragana(text):
    """pykakasi's reading of a text as it is written (digits stay digits)."""
    import pykakasi

    if not hasattr(hiragana, "kks"):
        hiragana.kks = pykakasi.kakasi()
    return "".join(x["hira"] for x in hiragana.kks.convert(text))


def _convert(texts):
    return [hiragana(numerals(t)) for t in texts]


_cache, _worker = {}, []


def readings(texts):
    todo = sorted({t for t in texts if t not in _cache})
    if todo:
        try:
            got = _convert(todo)
        except ImportError:
            if not _worker:
                _worker.append(
                    subprocess.Popen(
                        UV_RUN + [str(pathlib.Path(__file__))],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        encoding="utf-8",
                    )
                )
            w = _worker[0]
            w.stdin.write(json.dumps(todo, ensure_ascii=False) + "\n")
            w.stdin.flush()
            got = json.loads(w.stdout.readline())
        _cache.update(zip(todo, got))
    return [_cache[t] for t in texts]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    for request in sys.stdin:
        print(json.dumps(_convert(json.loads(request)), ensure_ascii=False), flush=True)
