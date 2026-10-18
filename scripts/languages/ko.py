"""Korean: a source for dubs into English (docs/languages/ko.md). Written with
spaces between words, but spaced loosely (Whisper and other recognisers space
the same words differently), so it is compared by its syllables, spaces aside."""

import difflib
import re
import unicodedata

from .latin import two_lines as subtitle  # noqa: F401  Korean breaks at its spaces

NAME, ISO3, NATIVE = "Korean", "kor", "한국어"

# ---- heard: the source ----

# Whisper's stock hallucinations on hums, music and silence, told the speech is Korean.
# Spaced both ways, as Whisper spaces them either way.
HALLUCINATIONS = (
    "시청해주셔서",
    "시청해 주셔서",
    "시청 감사",
    "구독과 좋아요",
    "구독 좋아요",
    "좋아요와 구독",
    "mbc 뉴스",
    "자막",
    "다음 영상에서",
)
# Whole phrases it writes on a breath, a laugh or crowd noise: suspect in a recovered phrase.
THANKS = ("감사합니다", "고맙습니다")
SECOND_OPINIONS = {
    "zipformer": "k2-fsa/sherpa-onnx-zipformer-korean-2024-06-24",
    "sensevoice": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
}
# The reader of captions and signs: PP-OCRv5's Korean recogniser, which reads Latin letters
# and digits too (the Chinese one has no Hangul). It comes only as a mobile model.
OCR = ("KOREAN", "MOBILE")


def units(text):
    """What second.py compares two transcripts by: their syllables (and any Latin
    letters and digits), without spaces and punctuation, since recognisers space
    Korean differently ("할 수 있어", "할수있어")."""
    return "".join(c for c in unicodedata.normalize("NFC", text.lower()) if c.isalnum())


# ---- spoken: the dub ----
# Korean is only a source so far: nothing dubs into it (PAIRS), so compare() has
# never been run on rendered Korean takes. It is here because every module has it.

VOICES = ("actor",)
JOIN = " "  # between Whisper's segments of a take


def speakable(text):
    """A line's text as the voice engine gets it: as written (VoiceStudio leaves
    digits as digits in Korean, as in Japanese)."""
    return text


MATCH = 0.8  # syllables are finer than words, as kana are
# Negative forms: 않다/않아, 못, 없다/없어, 아니다/아니야, 말다 (하지 마). "안" alone is
# left out: it is in too many words (안녕, 미안, 안에).
NEGATION = re.compile(r"않|못|없|아니|마세요|말아|지 ?마")


def compare(want, got):
    """lineqa.py's check of a take, syllable by syllable, spaces aside: (notes,
    similarity). Notes: leading junk, the syllable-level match, a changed count
    of negative forms, a number in digits heard as another, and English heard in
    a take whose text has none."""
    want_s, got_s = units(want), units(got)
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
    if got_s and len(NEGATION.findall(want)) != len(NEGATION.findall(got)):
        notes.append("negation differs")
    digits = lambda s: re.findall(r"\d+", s.replace(",", ""))  # noqa: E731
    if got_s and digits(want) and digits(want) != digits(got):
        notes.append(f"number differs: {digits(want)} -> {digits(got)}")
    latin = re.findall(r"[A-Za-z]+", got)
    if sum(map(len, latin)) >= 3 and not re.search(r"[A-Za-z]{3}", want):
        notes.append(f"English in the take: '{' '.join(latin)}'")
    return notes, ratio


# ---- written: track names and subtitles ----

TITLES = {
    "dub": {"actor": "한국어 더빙 (배우 목소리)"},
    "subs": {"actor": "한국어 (배우 목소리 더빙)"},
    "signs": "한국어 (화면 글자만)",
    "subtitles": "한국어 (자막)",
    "transcript": "한국어 (transcript)",
    "original": "한국어 (original)",
}
# Arial has no Hangul; Malgun Gothic ships with Windows (other systems fall back to a font that has it).
FONT = "Malgun Gothic"
READING_CPS = 12  # characters a second a subtitle can be read at in comfort (spaces count)
