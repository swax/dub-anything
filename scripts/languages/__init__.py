"""What the scripts know about each language they dub from or into: one module per
language, named for its ISO 639-1 code (en.py, es.py, ja.py, ko.py, zh.py), or its
ISO 639-3 code when it has no two-letter one (yue.py), and the pairs that have been
dubbed and checked end to end (PAIRS). Helpers beside them (hanzi.py, kana.py,
latin.py) have longer names, so they aren't taken for languages.

  LANGUAGES["es"].compare(want, got)   a language's module, by code
  PAIRS[("en", "es")]["line_ref"]      what was measured for a pair
  pair_of(segments)                    a part's pair, from its segments.json

Every module defines the names in REQUIRED; docs/languages/README.md says what
each is for and how to add a language. A module imports only the standard
library at its top, so every tool's environment can load it (second.py and
signs.py run in their own); anything more (num2words, pykakasi) is imported
where it's used.
"""

import importlib
import pathlib
import re

REQUIRED = (
    "NAME",
    "ISO3",
    "NATIVE",  # its names
    "HALLUCINATIONS",
    "THANKS",
    "SECOND_OPINIONS",
    "units",  # heard: draft.py, second.py
    "OCR",  # read in the picture: subs.py, signs.py
    "VOICES",
    "JOIN",
    "compare",
    "speakable",  # spoken by the dub: dub.py, lineqa.py
    "TITLES",
    "FONT",
    "subtitle",  # written: track names, subtitles
)

LANGUAGES = {}
for _path in sorted(pathlib.Path(__file__).parent.glob("*.py")):
    if not re.fullmatch(r"[a-z]{2,3}", _path.stem):
        continue
    _module = importlib.import_module(f"{__name__}.{_path.stem}")
    _missing = [name for name in REQUIRED if not hasattr(_module, name)]
    if _missing:
        raise ImportError(f"languages/{_path.name} lacks {', '.join(_missing)} (see docs/languages/README.md)")
    LANGUAGES[_path.stem] = _module

# The pairs the scripts know: dubbed and checked end to end. `dub.py prep` records
# a part's pair in its segments.json; parts prepped before it was recorded are
# Japanese -> English (PAIR). For each pair, what was measured (docs/research.md):
#   line_ref: whether its lines render with per-line references unless the script
#             says otherwise (core.line_refs: the voice's clips, then the actor's
#             own original of the line)
PAIR = ("ja", "en")
PAIRS = {
    # Findings 6 and 8: the takes follow the actor's delivery on each line.
    ("ja", "en"): {"line_ref": True},
    # Finding 17: the takes followed the actors more closely, and their English hardly
    # changed (P(en) 0.78 against 0.80).
    ("es", "en"): {"line_ref": True},
    # Finding 16: the actor's English original carried their accent into the takes
    # (Whisper's confidence that one actor's takes were Japanese fell from 0.96 to 0.53);
    # without it the takes followed the actors' delivery as closely and matched their
    # voices as well.
    ("en", "ja"): {"line_ref": False},
    # Finding 18: the takes followed the actors' pitch far more closely, and their
    # Spanish didn't change (P(es) 0.60 against 0.61).
    ("en", "es"): {"line_ref": True},
    # Finding 19: the takes followed the actors' delivery far more closely (arousal 0.40
    # against 0.01), and their English didn't change (P(en) 0.58 against 0.59).
    ("ko", "en"): {"line_ref": True},
    # Finding 20: the Mandarin sample's per-line references improved voice likeness
    # and timing without weakening its English. Only the actor voice was exercised.
    ("zh", "en"): {"line_ref": True},
    # Finding 21: the Windows A/B followed the actors' delivery more closely with
    # per-line references. Both modes still need Cantonese pronunciation review.
    ("en", "yue"): {"line_ref": True},
}
for _pair in PAIRS:
    if not all(code in LANGUAGES for code in _pair):
        raise ImportError(f"PAIRS has {_pair}, but not every language in it has a module")

# Languages a clip may also speak, left in the original and captioned (draft.py tags
# them), that have no module: their English name and ISO 639-2 code.
OTHERS = {
    "fr": ("French", "fra"),
    "de": ("German", "deu"),
    "it": ("Italian", "ita"),
    "pt": ("Portuguese", "por"),
    "ru": ("Russian", "rus"),
    "fi": ("Finnish", "fin"),
    "sv": ("Swedish", "swe"),
}
NAMES = {**{code: name for code, (name, _) in OTHERS.items()}, **{code: m.NAME for code, m in LANGUAGES.items()}}
ISO3 = {**{code: iso3 for code, (_, iso3) in OTHERS.items()}, **{code: m.ISO3 for code, m in LANGUAGES.items()}}


def pair_of(segments):
    """(source, target) of a part, from its segments.json ({} for a part not prepped yet)."""
    return segments.get("source_lang") or PAIR[0], segments.get("target_lang") or PAIR[1]
