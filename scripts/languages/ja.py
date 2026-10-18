"""Japanese: the source of most dubs (the directing notes are written for it), and a
target for dubs from English (docs/languages/ja.md). Written without spaces, it
is compared by its reading in kana (kana.py) and wrapped by hand in subtitles."""

import difflib
import re

from . import kana

NAME, ISO3, NATIVE = "Japanese", "jpn", "日本語"

# ---- heard: the source ----

# Whisper's stock hallucinations on hums, music and silence, told the speech is Japanese.
HALLUCINATIONS = ("ご視聴ありがとうございました", "チャンネル登録", "ご覧いただき", "字幕")
THANKS = ()
SECOND_OPINIONS = {
    "reazon": "reazon-research/reazonspeech-k2-v2",  # trained on Japanese TV
    "sensevoice": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
}
# The reader of captions and signs (subs.py, signs.py): PP-OCRv5's Chinese recogniser
# (RapidOCR's LangRec and ModelType), which reads kana too.
OCR = ("CH", "SERVER")
UNIT_PUNCT = "、。,.!?！？…・「」『』()（）ー〜~-—:：;；\"' 　"


def units(text):
    """What second.py compares two transcripts by: their reading in hiragana,
    punctuation aside, so spelling choices (kanji or kana) don't count."""
    return "".join(c for c in kana.hiragana(text) if c not in UNIT_PUNCT)


# ---- spoken: the dub ----

VOICES = ("actor",)
JOIN = ""  # between Whisper's segments of a take


def speakable(text):
    """A line's text as the voice engine gets it: as written. VoiceStudio leaves
    digits as digits in Japanese, so the script spells numbers in kana (ja.md)."""
    return text


PUNCT = re.compile(r"[\s、。,.!?！？…・「」『』()（）ー〜~\-—:：;；\"'’“”]")
NEGATION = re.compile(r"ない|なかっ|ません")
MATCH = 0.8  # kana are finer than words: one misheard mora in a short line costs more


def compare(want, got):
    """lineqa.py's check of a take, by reading (kana.py: kanji or kana, katakana or
    hiragana, digits or kanji numerals don't count): (notes, similarity). Notes:
    leading junk, the kana-level match, a flipped negation (ない, ません), a
    number heard as another, and English heard in the take (the actor's original
    leaking from the per-line reference)."""
    want_k, got_k = (PUNCT.sub("", r) for r in kana.readings([want, got]))
    notes = []
    head = want_k[:3]
    k = got_k.find(head) if len(head) == 3 else -1
    if 1 < k <= 8:
        notes.append(f"leading extra: '{got_k[:k]}'")
    ratio = difflib.SequenceMatcher(None, want_k, got_k).ratio()
    if not got_k:
        notes.append("silent or unintelligible")
    elif ratio < MATCH:
        notes.append(f"match {ratio:.2f}")
    # Negative endings are written in kana: counted in the text, as readings run words together
    # (ほな、行って reads ほないって).
    if got_k and len(NEGATION.findall(want)) != len(NEGATION.findall(got)):
        notes.append("negation differs")
    # Numbers the script writes in digits or kanji (a number spelled in kana is compared by its reading only).
    if got_k and kana.numbers(want) and sorted(kana.numbers(want)) != sorted(kana.numbers(got)):
        notes.append(f"number differs: {kana.numbers(want)} -> {kana.numbers(got)}")
    latin = re.findall(r"[A-Za-z]+", got)
    if sum(map(len, latin)) >= 3 and not re.search(r"[A-Za-z]{3}", want):
        notes.append(f"English in the take: '{' '.join(latin)}'")
    return notes, ratio


# ---- written: track names and subtitles ----

TITLES = {
    "dub": {"actor": "日本語吹替（俳優の声）"},
    "subs": {"actor": "日本語（俳優の声の吹替）"},
    "signs": "日本語（画面の文字のみ）",
    "subtitles": "日本語（字幕）",
    "transcript": "日本語 (transcript)",
    "original": "日本語 (original)",
}
# Arial has no Japanese; Yu Gothic ships with Windows (other systems fall back to a font that has it).
FONT = "Yu Gothic"
READING_CPS = 7  # characters a second a subtitle can be read at in comfort


def subtitle(text):
    """A subtitle cue's text. Japanese has no spaces to wrap at, so a long line is
    broken in two at the punctuation (else the character) nearest its middle."""
    if len(text) <= 22 or "\n" in text:
        return text
    mid = len(text) / 2
    stops = "、。！？!?…）」』"
    breaks = [i + 1 for i, c in enumerate(text[:-1]) if c in stops + " 　" and text[i + 1] not in stops]
    cut = min(breaks, key=lambda i: abs(i - mid)) if breaks else None
    if cut is None or abs(cut - mid) > len(text) / 4:
        cut = round(mid)
    return text[:cut].rstrip() + "\n" + text[cut:].lstrip()
