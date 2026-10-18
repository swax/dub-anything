"""Spanish: a source for dubs into English, and a target for dubs from English
(neutral Latin American Spanish, docs/languages/es.md)."""

import re
import unicodedata

from .latin import two_lines as subtitle, compare_words, word_units as units  # noqa: F401

NAME, ISO3, NATIVE = "Spanish", "spa", "Español"

# ---- heard: the source ----

# Whisper's stock hallucinations on hums, music and silence, told the speech is Spanish.
HALLUCINATIONS = (
    "subtítulos realizados por",
    "subtitulado por",
    "amara.org",
    "gracias por ver el",
    "suscríbete",
    "suscribete",
    "no olvides suscribirte",
)
# Whole phrases it writes on a breath, a laugh or crowd noise. Real thanks are common in
# dialogue, so only a recovered phrase that says nothing else is suspect: in one
# telenovela's party scenes, three of four were crowd or lyrics.
THANKS = ("gracias", "muchas gracias")
SECOND_OPINIONS = {
    "parakeet": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    "canary": "csukuangfj/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8",
}
# The reader of captions and signs: PP-OCRv5's Chinese recogniser, which reads Latin
# letters, not always with their accents.
OCR = ("CH", "SERVER")

# ---- spoken: the dub ----

VOICES = ("actor",)
JOIN = " "  # between Whisper's segments of a take


def speakable(text):
    """A line's text as the voice engine gets it: VoiceStudio spells numbers out in
    Spanish but leaves one with a thousands point ("150.000") as digits (en.py)."""
    return re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)


ABBREV = {
    "sr": "senor",
    "sra": "senora",
    "srta": "senorita",
    "dr": "doctor",
    "dra": "doctora",
    "ud": "usted",
    "uds": "ustedes",
    "okey": "ok",
    "okay": "ok",
    "vs": "versus",
}
NEGATIONS = {"no", "nunca", "nada", "nadie", "ni", "ningun", "ninguno", "ninguna", "jamas", "tampoco"}
# Words that say a number, accents aside; "uno", "una" and "un" are left out, as articles too.
NUMBER = re.compile(
    r"dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|"
    r"dieci[a-z]+|veinte|veinti[a-z]+|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|"
    r"noventa|cien|ciento|[a-z]+cientos|quinientos|mil|millon|millones"
)


def words(text, reading="plain"):
    """Normalized words without accents (Whisper writes "esta" or "está", "si" or
    "sí"): numbers spelled out in Spanish (num2words), read "plain" or as "digits",
    and "Sr." as "señor"."""
    from num2words import num2words

    plain = lambda s: re.sub(r"[̀-ͯ]", "", unicodedata.normalize("NFD", s))  # noqa: E731
    t = plain(text.lower())
    t = re.sub(r"(\d+)\s*%", r"\1 por ciento", t)
    t = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", t)  # 1.000 or 1,000
    t = re.sub(r"(\d)[.,](\d)", r"\1 punto \2", t)
    t = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", t)
    out = []
    for w in re.findall(r"[a-z0-9]+", t):
        if w.isdigit() and reading == "digits":
            out += [plain(num2words(int(d), lang="es")) for d in w]
        elif w.isdigit() and len(w) < 13:
            out += plain(num2words(int(w), lang="es")).split()
        else:
            out.append(ABBREV.get(w, w))
    return out


def compare(want, got):
    """lineqa.py's check of a take: word by word, like English (latin.compare_words)."""
    return compare_words(want, got, words, ("plain", "digits"), NEGATIONS, NUMBER.fullmatch)


# ---- written: track names and subtitles ----

TITLES = {
    "dub": {"actor": "Doblaje en español (voz del actor)"},
    "subs": {"actor": "Español (doblaje con la voz del actor)"},
    "signs": "Español (solo textos en pantalla)",
    "subtitles": "Español (subtítulos)",
    "transcript": "Español (transcript)",
    "original": "Español (original)",
}
FONT = "Arial"  # has the accents, ñ and ¿¡
READING_CPS = 17  # letters a second a subtitle can be read at in comfort (spaces count)
