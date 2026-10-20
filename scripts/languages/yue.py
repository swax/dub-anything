"""Cantonese: spoken target for dubbing (docs/languages/yue.md).
Compare transcripts by characters: recognisers don't agree on word spacing.
Traditional Cantonese characters are preserved.
"""

import difflib
import re
import sys
import unicodedata

from .hanzi import normalize_numbers

NAME, ISO3, NATIVE = "Cantonese", "yue", "粵語"

# ---- heard: the source ----

# Stock credits and channel calls on music/silence.
HALLUCINATIONS = (
    "多謝收睇",
    "多謝觀看",
    "多谢收看",
    "多谢观看",
    "感謝觀看",
    "謝謝觀看",
    "歡迎訂閱",
    "请订阅",
    "請訂閱",
    "點讚訂閱",
    "点赞订阅",
    "字幕由",
    "字幕提供",
    "amara.org",
)
THANKS = ("多謝", "多謝大家", "唔該", "唔該晒", "謝謝", "謝謝大家")
SECOND_OPINIONS = {
    "sensevoice": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
}
OCR = ("CH", "SERVER")


def units(text):
    """Character units with width, case, spacing and punctuation normalized."""
    text = "".join(c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum())
    # Common Cantonese ASR spellings; preserve unrelated words and homophones.
    for heard, written in (("依家", "而家"), ("我地", "我哋"), ("你地", "你哋"), ("佢地", "佢哋"), ("啊", "呀")):
        text = text.replace(heard, written)
    return text


# ---- spoken: dub into Cantonese ----

VOICES = ("actor",)
JOIN = ""
MATCH = 0.8
# 咪 alone can mean "don't"; 係咪 is a yes/no question, not a negation.
NEGATION = re.compile(r"唔|冇|(?<!係)咪|未|莫|毋|無|不|非|別|没|沒")


def speakable(text):
    return text


def compare(want, got):
    """Compare desired Cantonese text with transcription from QA."""
    want_normal, want_numbers = normalize_numbers(want)
    got_normal, got_numbers = normalize_numbers(got)
    want_s, got_s = units(want_normal), units(got_normal)
    notes = []
    head = want_s[:3]
    k = got_s.find(head) if len(head) == 3 else -1
    if 1 < k <= 8:
        notes.append(f"leading extra: '{got_s[:k]}'")
    ratio = difflib.SequenceMatcher(None, want_s, got_s).ratio()
    if not got_s:
        notes.append("silent or unintelligible")
    elif ratio < MATCH:
        notes.append(f"match {ratio:.2f}")
    # Keep punctuation here: "係。咪去" contains a negative imperative, not 係咪.
    if got_s and len(NEGATION.findall(want_normal)) != len(NEGATION.findall(got_normal)):
        notes.append("negation differs")
    if got_s and want_numbers != got_numbers:
        notes.append(f"number differs: {want_numbers} -> {got_numbers}")
    return notes, ratio


# ---- written: track names and subtitles ----

TITLES = {
    "dub": {"actor": "粵語配音（演員原聲）"},
    "subs": {"actor": "粵語（演員原聲配音）"},
    "signs": "粵語（畫面文字）",
    "subtitles": "粵語（字幕）",
    "transcript": "粵語 (transcript)",
    "original": "粵語 (original)",
}
FONT = "PingFang HK" if sys.platform == "darwin" else "Microsoft JhengHei"
READING_CPS = 9  # characters a second a subtitle can be read at in comfort


def subtitle(text):
    """Wrap a long unspaced cue near its middle, preferably at punctuation."""
    if len(text) <= 22 or "\n" in text:
        return text
    mid = len(text) / 2
    stops = "，。！？；：、!?…）】」』"
    best, best_dist = None, float("inf")
    for i, c in enumerate(text):
        if c in stops:
            dist = abs((i + 1) - mid)
            if dist < best_dist:
                best, best_dist = i + 1, dist
    if best is not None and best_dist < len(text) * 0.35:
        return text[:best].rstrip() + "\n" + text[best:].lstrip()
    cut = int(round(mid))
    return text[:cut] + "\n" + text[cut:]
