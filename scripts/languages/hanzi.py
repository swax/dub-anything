"""What the languages written in Chinese characters share (yue.py, zh.py)."""

import re
import unicodedata

DIGITS = str.maketrans("零〇一二兩两三四五六七八九", "0012223456789")
NUMBER = re.compile(
    r"[-負负]?[\d零〇一二兩两三四五六七八九十百千萬万億亿廿卅卌]+(?:[.點点][\d零〇一二兩两三四五六七八九]+)?"
)
PLACES = {"十": 10, "百": 100, "千": 1000, "萬": 10000, "万": 10000, "億": 100000000, "亿": 100000000}


def normalize_numbers(text):
    """Canonicalize digit sequences, decimal and place-value numerals for QA.
    Colloquial implied units and idioms still need the listener's judgement.
    """
    numbers = []

    def replace(match):
        token = match.group().replace("廿", "二十").replace("卅", "三十").replace("卌", "四十")
        sign = "-" if token[0] in "-負负" else ""
        token = token.lstrip("-負负").translate(DIGITS).replace("點", ".").replace("点", ".")
        whole, dot, fraction = token.partition(".")
        if any(c in PLACES for c in whole):
            total, section, current = 0, 0, 0
            for c in whole:
                if c.isdecimal():
                    current = current * 10 + int(c)
                else:
                    place = PLACES[c]
                    if place < 10000:
                        section += (current or 1) * place
                    elif place == 10000:
                        total += (section + current) * place
                        section = 0
                    else:
                        total = (total + section + current) * place
                        section = 0
                    current = 0
            whole = str(total + section + current)
        fraction = fraction.rstrip("0")
        value = sign + whole + ("." + fraction if dot and fraction else "")
        numbers.append(value)
        return value

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    return NUMBER.sub(replace, text), numbers
