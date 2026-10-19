"""Subtitles for reading: when each cue shows, and how fast it asks to be read.

A script's windows follow the speech, which is what a dub needs, but subtitles
cut to the speech flash: a two-word reply shows for half a second, and lines a
few frames apart blink off and on. timed() keeps each cue's start and lengthens
it for reading:
  - at least MIN_SHOW, and long enough to read its text at the language's
    comfortable rate (READING_CPS in languages/<code>.py), up to MAX_SHOW
    (a line spoken for longer keeps its own length);
  - LINGER past the speech where there is room;
  - never into the next cue (GAP before it), and a gap shorter than CHAIN closes
    up to GAP, so the text changes instead of blinking;
  - never past `until`, the video's end when it is known: a cue lengthened past
    it makes the subtitle track, and the file, longer than the picture.
Cues that already overlap (two speakers at once, a caption under a line) stay
as they are. assemble.py times every subtitle track this way, dubs' included.

rows() is the director's check for a part to subtitle (fit.py runs it for
parts prepped with --subtitles-only): each line's reading speed once timed,
flagged FAST over FAST_RATIO times the comfortable rate, LONG past two
lines, and SPLIT? when the line is spoken for longer than MAX_SHOW.
"""

import statistics

from languages import LANGUAGES
from languages.latin import LINE, width

MIN_SHOW = 0.83  # five sixths of a second: 20 frames at 24 fps
MAX_SHOW = 7.0
LINGER = 0.5
GAP = 0.084  # two frames
CHAIN = 0.6
FAST_RATIO = 1.2  # reading speed above the comfortable rate that gets flagged


def letters(text):
    """What a reader reads: every character but the line breaks."""
    return len(text.replace("\n", " ").strip())


def rate(lang):
    return getattr(LANGUAGES[lang], "READING_CPS", 17)


def timed(cues, lang, until=float("inf")):
    """[(start, end, text)] -> the same cues, sorted, each shown long enough to read
    (and lengthened no further than `until`)."""
    cues = sorted(cues, key=lambda c: (c[0], c[1]))
    out = []
    for i, (a, b, text) in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else float("inf")
        if nxt < b:
            out.append((a, b, text))
            continue
        need = min(max(MIN_SHOW, letters(text) / rate(lang)), MAX_SHOW)
        end = max(b, min(max(b + LINGER, a + need), a + MAX_SHOW))
        end = max(b, min(end, nxt - GAP))
        if nxt - end < CHAIN:
            end = max(end, nxt - GAP)
        out.append((a, round(max(b, min(end, until)), 3), text))
    return out


def rows(lines, lang):
    """Each line's reading as delivered: {i, start, end, shown, cps, flag, text},
    in the script's order."""
    order = sorted(range(len(lines)), key=lambda i: (lines[i]["start"], lines[i]["end"]))
    sub = LANGUAGES[lang].subtitle
    shown = timed([(lines[i]["start"], lines[i]["end"], sub(lines[i]["text"])) for i in order], lang)
    out = []
    for i, (a, end, text) in zip(order, shown):
        line = lines[i]
        cps = letters(text) / max(end - a, 0.01)
        flags = []
        if cps > rate(lang) * FAST_RATIO:
            flags.append("FAST")
        if len(text.split("\n")) > 2 or any(width(t) > LINE + 4 for t in text.split("\n")):
            flags.append("LONG")
        if line["end"] - line["start"] > MAX_SHOW:
            flags.append("SPLIT?")
        out.append(
            {"i": i, "start": a, "end": line["end"], "shown": end, "cps": cps, "flag": " ".join(flags), "text": text}
        )
    return sorted(out, key=lambda r: r["i"])


def summary(cues, lang):
    """One line on how fast a whole job's subtitles read."""
    shown = timed([(a, b, LANGUAGES[lang].subtitle(t)) for a, b, t in cues], lang)
    cps = [letters(t) / max(b - a, 0.01) for a, b, t in shown]
    if not cps:
        return "no subtitles"
    fast = sum(c > rate(lang) * FAST_RATIO for c in cps)
    return (
        f"{len(cps)} subtitles; median reading speed {statistics.median(cps):.1f} characters/s "
        f"(comfortable: {rate(lang)}); {fast} faster than {rate(lang) * FAST_RATIO:.0f}"
    )
