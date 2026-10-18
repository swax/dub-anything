"""English: the language of most dubs (the directing notes are written for it), and
a source for dubs into Japanese and Spanish."""

import re

from .latin import as_written as subtitle, compare_words, word_units as units  # noqa: F401

NAME, ISO3, NATIVE = "English", "eng", "English"

# ---- heard: the source ----

# Whisper's stock hallucinations on hums, music and silence, told the speech is English.
HALLUCINATIONS = (
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by",
    "amara.org",
    "like and subscribe",
)
# Whole phrases it writes on a breath, a laugh or crowd noise: suspect in a recovered phrase.
THANKS = ("thank you", "thanks")
SECOND_OPINIONS = {
    "parakeet": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    "sensevoice": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
}
# The reader of captions and signs: PP-OCRv5's Chinese recogniser, which reads Latin letters too.
OCR = ("CH", "SERVER")

# ---- spoken: the dub ----

# The actor's voice (OmniVoice clones the actor, accent and all), and the American
# voice (IndexTTS speaks native English, and vc.py gives each take the actor's timbre).
VOICES = ("actor", "american")
JOIN = " "  # between Whisper's segments of a take


def speakable(text):
    """A line's text as the voice engine gets it. VoiceStudio spells numbers out in
    English, but leaves one with a thousands comma ("150,000") as digits, which
    the clone then reads in the language of its reference: with a Korean actor's
    original, "150,000" came out in Korean. Without the comma it is spelled out."""
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen"
).split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
ABBREV = {
    "dr": "doctor",
    "mr": "mister",
    "mrs": "missus",
    "ms": "miss",
    "ok": "okay",
    "vs": "versus",
    "nope": "no",
    "nah": "no",
    "cm": "centimeter",
    "mm": "millimeter",
    "km": "kilometer",
    "kg": "kilogram",
    "centimeters": "centimeter",
    "millimeters": "millimeter",
}
NEGATIONS = {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor"}
# Words that say a number; "one" and "zero" are left out, as words too ("this one").
NUMBER_WORDS = set(ONES[2:] + TENS[2:] + ["hundred", "thousand", "million", "billion"])
NUMBER_WORDS |= {w[:-1] + "ies" for w in TENS[2:]} | {w + "s" for w in ("hundred", "thousand", "million", "billion")}


def spell(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    for div, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= div:
            q, r = divmod(n, div)
            return f"{spell(q)} {name}" + ("" if r == 0 else f" {spell(r)}")


ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth"}
ORDINALS["twelve"] = "twelfth"


def ordinal(n):
    """An ordinal in words, as Whisper's "8th" or "21st" is said: "eighth", "twenty first"."""
    *head, last = spell(n).split()
    last = ORDINALS.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return " ".join(head + [last])


def spell_pairs(n):
    """A four-digit number read in pairs, as years and model numbers are:
    1979 "nineteen seventy nine", 5410 "fifty four ten", 3400 "thirty four hundred"."""
    hi, lo = divmod(n, 100)
    return f"{spell(hi)} " + ("hundred" if lo == 0 else f"oh {spell(lo)}" if lo < 10 else spell(lo))


COLLOQUIAL = {
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "gimme": "give me",
    "lemme": "let me",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "do not know",
}


def words(text, reading="plain"):
    """Normalized words: numbers spelled out, contractions expanded, "Dr." as
    "doctor". Numbers read "plain" (1979 "one thousand nine hundred seventy nine"),
    in "pairs" (years, model numbers: "nineteen seventy nine") or as "digits"
    (phone numbers: "zero three three zero ...")."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"(\d+)\s*%", r"\1 percent", t)
    t = re.sub(r"\bhalf (?:a )?percent\b", "0.5 percent", t)  # Whisper writes "half a percent" as "0.5%"
    t = re.sub(r"\bno\.\s*(?=\d)", "number ", t)  # "No.1" is "number one", not a negation
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    t = re.sub(r"(\d)\.(\d)", r"\1 point \2", t)
    # Decades: "the 70s" and "the 1970s" are "the seventies" and "the nineteen seventies".
    t = re.sub(
        r"\b(1[0-9]|20)?([2-9]0)'?s\b",
        lambda m: (spell(int(m.group(1))) + " " if m.group(1) else "") + spell(int(m.group(2)))[:-1] + "ies",
        t,
    )
    t = re.sub(r"\b(\d{1,4})(?:st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t)  # "8th" is "eighth"
    t = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", t)  # "a2.5cm" -> "a 2.5 cm"
    # Letter grades: Whisper writes "a B-minus" as "a B-" (and "A-plus" as "A+").
    t = re.sub(r"\b([a-f])([-+])(?![a-z0-9])", lambda m: m.group(1) + (" minus" if m.group(2) == "-" else " plus"), t)
    t = re.sub(r"\bnevermind\b", "never mind", t)  # one word or two, same meaning
    t = re.sub(r"\b(" + "|".join(COLLOQUIAL) + r")\b", lambda m: COLLOQUIAL[m.group(1)], t)
    for a, b in (
        ("won't", "will not"),
        ("can't", "can not"),
        ("n't", " not"),
        ("'re", " are"),
        ("'ve", " have"),
        ("'ll", " will"),
        ("'m", " am"),
        ("'d", " would"),
        ("'s", " is"),
    ):
        t = t.replace(a, b)
    out = []
    for w in re.findall(r"[a-z0-9]+", t):
        if w.isdigit() and reading == "digits":
            out += ["zero" if d == "0" else spell(int(d)) for d in w]
        elif w.isdigit() and len(w) < 13:
            n = int(w)
            out += (spell_pairs(n) if reading == "pairs" and 1000 < n < 10000 and n % 1000 else spell(n)).split()
        else:
            out.append(ABBREV.get(w, w))
    return out


def compare(want, got):
    """lineqa.py's check of a take: word by word (latin.compare_words)."""
    return compare_words(want, got, words, ("plain", "pairs", "digits"), NEGATIONS, NUMBER_WORDS.__contains__)


# ---- written: track names and subtitles ----

# Our tracks' names in the delivered file: the dub's audio and subtitles per voice,
# the signs alone, the source's transcript and its own audio.
TITLES = {
    "dub": {"actor": "English dub (actor's voice)", "american": "English dub (American)"},
    "subs": {"actor": "English (actor-voice dub)", "american": "English (American dub)"},
    "signs": "English (signs only)",
    "transcript": "English (transcript)",
    "original": "English (original)",
}
FONT = "Arial"
