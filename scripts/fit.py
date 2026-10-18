"""Compare each generated clip's speech length with its script window,
and with how long the actor actually speaks in it.

Usage: scripts/fit.py <name> [--only-flags]   (reads work/<name>/script.json and the job's seg_<lang>_L*.wav)
Lines not rendered yet get an estimate (~): the length OmniVoice itself will give
them, computed with its own rule from the reference the render will use (the
line's own original, or the voice's clips for the lines listed as falling back).
A part in the "american" voice is estimated with IndexTTS's length instead, from
the English and how long the actor speaks in the window (core.indextts_length).
No render is needed, so it works for a new show's first part. --voice american
estimates the director's script in the American voice (each line's "text_us",
if it has one) before that voice renders it. They're flagged
"OVER?" or "risk", to fix before rendering. "SHORT" / "SHORT?" marks English that
ends well before the actor stops: write fuller text for it (slowing a take adds
little). `dub.py generate` prints the same table for the failing lines when a
line overflows.
It first warns about any voice whose reference is much faster or slower than
its speaker's lines: lines on that reference inherit its pace (OmniVoice only).

A part to subtitle (prepped with --subtitles-only), or any part with --read,
gets its subtitles' reading speed instead (subtitles.rows): each line's
characters a second over the time it shows once timed for reading, flagged
FAST (cut words, or merge it with a neighbour), LONG (more than two lines:
split it at a pause) and SPLIT? (spoken for longer than a subtitle should stay).
Its "captions" are listed after its lines (C00...).
"""

import argparse
import statistics
import sys

import subtitles
from core import WORK, _omnivoice, fit_rows, langs, load_json, said, voice_clips

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--only-flags", action="store_true", help="print only lines that don't fit or are at risk")
ap.add_argument(
    "--voice",
    choices=["actor", "american"],
    help="estimate the part in this voice instead of its own (e.g. the director's script, with each "
    "line's text_us, for the American voice)",
)
ap.add_argument("--read", action="store_true", help="check the subtitles' reading speed, not the dub's length")
args = ap.parse_args()

if args.read or load_json(WORK / args.name / "segments.json", {}).get("subtitles_only"):
    script = load_json(WORK / args.name / "script.json")
    tgt = langs(args.name)[1]
    lines, captions = script["lines"], script.get("captions", [])
    for row in subtitles.rows(lines + captions, tgt):
        if args.only_flags and not row["flag"]:
            continue
        tag = f"L{row['i']:02d}" if row["i"] < len(lines) else f"C{row['i'] - len(lines):02d}"
        text = row["text"].replace("\n", " / ")
        print(
            f"{tag} {row['start']:7.2f}-{row['end']:7.2f} shown to {row['shown']:7.2f}  "
            f"{row['cps']:4.1f}/s {row['flag']:11s} {text}"
        )
    sys.exit(0)


def pace_check(name, tolerance=0.15):
    """A voice's reference sets the pace of every line that falls back to it, and
    part of every per-line reference. A clip said faster than the speaker usually
    talks (a read-out intro, an excited burst) makes those takes run short."""
    weight = _omnivoice()[0].calculate_total_weight
    voices = load_json(WORK / "voices.json", {})
    src = langs(name)[0]  # a voice's lines are all its show's, in one language
    scripts = [load_json(p) for p in WORK.glob("*/script.json")]
    for key in sorted({ln["voice"] for ln in load_json(WORK / name / "script.json")["lines"]}):
        spec = voices.get(key)
        if not spec:
            continue
        ref = weight(spec["ref_text"]) / sum(b - a for _, a, b in voice_clips(spec))
        rates = [
            weight(said(ln, src)) / (ln["end"] - ln["start"])
            for s in scripts
            for ln in s["lines"]
            if ln.get("voice") == key and said(ln, src) and ln["end"] - ln["start"] >= 1.5
        ]
        if len(rates) < 5:
            continue
        typical = statistics.median(rates)
        if abs(ref / typical - 1) > tolerance:
            print(
                f"pace: {key}'s reference says {ref:.1f} units/s, its {len(rates)} lines {typical:.1f} "
                f"({ref / typical - 1:+.0%}); lines on the reference will run "
                f"{'short' if ref > typical else 'long'}. Pick clips nearer the lines' pace."
            )


if (args.voice or load_json(WORK / args.name / "script.json").get("render", {}).get("voice")) != "american":
    pace_check(args.name)
total_chars = total_secs = 0.0
for row in fit_rows(args.name, predict=True, short=True, voice=args.voice):
    if args.only_flags and not row["flag"]:
        continue
    actor = f"actor {row['actor']:5.2f}s" if row["actor"] else "actor   -   "
    if row["speech"] is None:
        est = f"~{row['est']:4.2f}s" if "est" in row else "(no clip)"
        print(
            f"L{row['i']:02d} {est:>6s} / window {row['window']:5.2f}s (room {row['room']:5.2f}, {actor})  "
            f"{row['flag']:6s} {row['text']}"
        )
        continue
    total_chars += len(row["text"])
    total_secs += row["speech"]
    print(
        f"L{row['i']:02d} {row['speech']:5.2f}s / window {row['window']:5.2f}s (room {row['room']:5.2f}, {actor})  "
        f"{row['flag']:6s} {row['text']}"
    )
if total_secs:
    print(f"average rate: {total_chars / total_secs:.1f} chars/s")
