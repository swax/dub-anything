"""Find window edges that cut the actor mid-sound: starts that open after the
actor has already started speaking, and ends that close before they stop.

Usage: scripts/edges.py <name> [--apply]   (after `dub.py prep`; works on script.json)

At each window start VoiceStudio crossfades from the original mix, voice
included, to the voice-free bed in 10 ms. Speech detection misses soft attacks
(a breath, a consonant) by a few tens of ms. If the actor is already sounding
when the window opens, that sliver of the original plays and is then chopped,
which is heard as a click just before the English starts. This finds every
window start where the original voice is active and walks back to where that
sound began. For a clear soft attack (see MIN_SHIFT/MAX_SHIFT/CUT_DB) it moves
the window to start 20 ms before it, never before the previous window's end;
the rest are listed as 'skip'. The separator writes zeros (or nearly) where it
hears no voice, and speech detection can miss the first 150-400 ms of a phrase
that rises out of them: a sound that begins in that silence and runs unbroken
into the line is the line's own start, so the window moves back up to 400 ms.

Window ends are the mirror image: the original mix fades back in over the
window's last 10 ms, so a voice still sounding there plays a clipped tail of
the original after the English. A short tail that clearly belongs to the line
(same rule) moves the end to 20 ms after the voice stops, never closer than
50 ms to the next window. Longer ones are usually the next speaker, a laugh or
a crowd, and are only noted, unless the voice dies out into that silence.

Where two windows touch (a turn change, or a line split without a pause), the
voice often still dips to near silence for a moment near the boundary, too briefly
for speech detection to call it a pause. When that dip (core.end_dip) isn't at the
boundary, both windows move to it: it's where one line really ends and the next
begins, and a per-line reference cut there ends cleanly.

Captions mark speech kept in the original audio. New moves cannot cross those
windows, even when their English text is empty. Existing overlaps are left for
the director to judge.

--apply writes the moves to script.json. It's timing only: the next
`dub.py generate` re-mixes without re-synthesizing. An earlier start also
starts the take a little earlier, closer to the actor's mouth.
"""

import argparse
import json
import sys

import numpy as np

from core import JOBS, WORK, end_dip, load_json
from faster_whisper.audio import decode_audio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR, HOP = 16000, 80  # 5 ms frames
GAP = 4  # frames: the voice has stopped once it stays under the threshold this long (20 ms)
SILENT_DB = -70.0  # the separated vocals between phrases: exact zeros (-180 dB) or nearly


def frames_db(audio):
    n = len(audio) // HOP
    x = audio[: n * HOP].reshape(n, HOP)
    return 20 * np.log10(np.sqrt((x**2).mean(1)) + 1e-9)


def silent(db, a, b):
    """Some of the separated vocals' silence in frames a..b: the voice fades in
    or out over a few faint frames on its way from it."""
    seg = db[max(0, a) : b]
    return len(seg) > 0 and float(seg.min()) < SILENT_DB


def voiced_onset(db, t, floor_t, level, below=20.0, lead=0.02):
    """If the voice is sounding just before time t: the time it began (minus
    `lead`, not earlier than floor_t) and whether it rose out of silence;
    else None. `level` is the line's speech level; 'sounding' means within
    `below` dB of it."""
    k = int(t * SR / HOP)
    thresh = level - below
    if k < 2 or k > len(db) or max(db[k - 2 : k]) < thresh:  # the 10 ms before the edge
        return None
    j = k - 1
    floor_k = int(floor_t * SR / HOP)
    # Walk back over the sound and any dip shorter than GAP: a soft line over low noise
    # flickers around the threshold, and stopping at the first quiet frame put each
    # --apply run's cut just before another flicker, a little earlier every time.
    while j - 1 >= floor_k and max(db[max(floor_k, j - GAP) : j]) >= thresh:
        j -= 1
    return max(floor_t, j * HOP / SR - lead), j > floor_k and silent(db, j - 4 * GAP, j)


def voiced_release(db, t, ceil_t, level, below=20.0, lag=0.02):
    """If the voice is sounding just after time t: the time it stops (plus
    `lag`, not later than ceil_t) and whether it dies into silence; else
    None. The mirror of voiced_onset."""
    k = int(t * SR / HOP)
    thresh = level - below
    if k + 2 > len(db) or min(db[k : k + 2]) < thresh:  # the 10 ms after the edge
        return None
    j = k
    ceil_k = min(len(db) - 1, int(ceil_t * SR / HOP))
    while j + 1 <= ceil_k and max(db[j + 1 : min(ceil_k, j + GAP) + 1]) >= thresh:
        j += 1
    return min(ceil_t, (j + 1) * HOP / SR + lag), j < ceil_k and silent(db, j + 1, j + 1 + 4 * GAP)


def level_at(db, t):
    """The voice's level around time t, smoothed over 20 ms as end_dip smooths it."""
    k = int(round(t * SR / HOP))
    seg = db[max(0, k - 2) : k + 2]
    return float(seg.mean()) if len(seg) else -90.0


# A shared boundary already in a dip within this many dB of the deepest one nearby
# stays put: end_dip searches around the boundary itself, so each move would find a
# slightly deeper frame further on and walk the boundary along on every re-run.
DIP_SETTLED_DB = 3.0


def line_level(db, a, b):
    seg = db[int(a * SR / HOP) : int(b * SR / HOP)]
    return float(np.percentile(seg, 90)) if len(seg) else -90.0


# Move a start only for a real, audible soft attack. The voice must be clearly
# sounding at the cut and must have begun 20-130 ms before the window (shift
# includes the 20 ms lead). Shorter fragments weren't heard on a line nobody
# flagged. Longer ones are usually someone else's sound (a tail, a laugh),
# where moving the window would start the English too early.
MIN_SHIFT, MAX_SHIFT, CUT_DB = 0.04, 0.15, -16.0
# A sound rising out of the separated vocals' silence (or dying into it) and running
# unbroken into the line is the line's own: speech detection missed 150-400 ms of such
# attacks, at the line's full level, on many lines of one film.
MAX_SHIFT_SILENT = 0.40
SEARCH = 0.45  # how far past an edge to look for where the voice begins or ends


def check(name, apply=False, min_shift=0.015):
    path = WORK / name / "script.json"
    script = load_json(path)
    lines = script["lines"]
    captions = script.get("captions", [])
    audio = decode_audio(
        str(JOBS / load_json(WORK / name / "segments.json")["job_id"] / "vocals.wav"), sampling_rate=SR
    )
    db = frames_db(audio)
    moved, tails, joins = [], [], []
    for i, line in enumerate(lines):
        level = line_level(db, line["start"], line["end"])
        prev_end = lines[i - 1]["end"] if i else 0.0
        for caption in captions:
            if caption["start"] < line["start"]:
                prev_end = max(prev_end, min(caption["end"], line["start"]))
        onset, from_silence = voiced_onset(db, line["start"], max(prev_end, line["start"] - SEARCH), level) or (
            None,
            False,
        )
        if onset is not None and line["start"] - onset >= min_shift:
            k = int(line["start"] * SR / HOP)
            cut = float(max(db[k - 2 : k])) - level  # how loud the voice is where it gets chopped
            limit = MAX_SHIFT_SILENT if from_silence else MAX_SHIFT
            fix = MIN_SHIFT - 1e-6 <= line["start"] - onset <= limit + 1e-6 and cut >= CUT_DB
            moved.append((i, line["start"], round(onset, 2), line["text"], cut, fix))
            if apply and fix:
                line["start"] = round(onset, 2)
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else line["end"] + 1
        if nxt - line["end"] <= 0.05:
            dip = end_dip(db, line["start"], line["end"])
            if (
                dip is not None
                and abs(dip - line["end"]) >= min_shift
                and dip < lines[i + 1]["end"] - 0.3
                and level_at(db, line["end"]) - level_at(db, dip) > DIP_SETTLED_DB
                and not any(c["start"] < max(dip, line["end"]) and c["end"] > min(dip, line["end"]) for c in captions)
            ):
                joins.append((i, line["end"], round(dip, 2), line["text"]))
                if apply:
                    line["end"] = lines[i + 1]["start"] = round(dip, 2)
            continue
        ceil = nxt - 0.05
        for caption in captions:
            if caption["end"] > line["end"]:
                ceil = min(ceil, max(caption["start"], line["end"]))
        if ceil <= line["end"]:
            continue
        release, to_silence = voiced_release(db, line["end"], min(ceil, line["end"] + SEARCH), level) or (None, False)
        if release is not None:
            k = int(line["end"] * SR / HOP)
            cut = float(min(db[k : k + 2])) - level
            limit = MAX_SHIFT_SILENT if to_silence else MAX_SHIFT
            fix = (
                MIN_SHIFT - 1e-6 <= release - line["end"] <= limit + 1e-6
                and cut >= CUT_DB
                and release < min(ceil, line["end"] + SEARCH)
            )  # the voice really stopped
            tails.append((i, line["end"], round(release, 2), line["text"], cut, fix))
            if apply and fix:
                line["end"] = round(release, 2)
    for i, old, new, text, cut, fix in moved:
        print(
            f"  {'FIX ' if fix else 'skip'} L{i:02d} start {old:7.2f} -> {new:7.2f}  ({(old - new) * 1000:3.0f} ms "
            f"earlier, voice {cut:+5.1f} dB at the cut)  {text[:40]}"
        )
    for i, old, new, text, cut, fix in tails:
        what = (
            f"-> {new:7.2f}  ({(new - old) * 1000:3.0f} ms later,"
            if fix
            else f"   voice sounds {(new - old) * 1000:3.0f}+ ms on,"
            if new >= old + SEARCH - 1e-6
            else f"   voice sounds {(new - old) * 1000:3.0f} ms on,"
        )
        print(
            f"  {'FIX ' if fix else 'note'} L{i:02d} end   {old:7.2f} {what} voice {cut:+5.1f} dB at the cut)  {text[:36]}"
        )
    for i, old, new, text in joins:
        print(f"  FIX  L{i:02d}/L{i + 1:02d} boundary {old:7.2f} -> {new:7.2f}  (the voice dips there)  {text[:36]}")
    fixes = sum(m[5] for m in moved)
    end_fixes = sum(t[5] for t in tails)
    print(
        f"{name}: {fixes} starts, {end_fixes} ends and {len(joins)} shared boundaries to move"
        f"{' (applied)' if apply else ''}, {len(moved) - fixes} starts and {len(tails) - end_fixes} ends left alone"
    )
    if apply and (fixes or end_fixes or joins):
        # One entry per line, keeping every other top-level key (captions, notes).
        parts = []
        for key, val in {**script, "lines": lines}.items():
            if isinstance(val, list):
                body = ",\n".join("    " + json.dumps(l, ensure_ascii=False) for l in val)
                parts.append(f"  {json.dumps(key)}: [\n{body}\n  ]" if val else f"  {json.dumps(key)}: []")
            else:
                parts.append(f"  {json.dumps(key)}: {json.dumps(val, ensure_ascii=False)}")
        path.write_bytes(("{\n" + ",\n".join(parts) + "\n}\n").encode("utf-8"))
    return moved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--apply", action="store_true", help="write the earlier starts to script.json")
    args = ap.parse_args()
    check(args.name, args.apply)
