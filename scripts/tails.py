"""Find lines whose original voice carries heavy reverb (flashbacks, echoing halls).

Usage: scripts/tails.py <name> [--all]   (after writing script.json; --all prints every line)

For each line it measures how long the original voice rings on after the actor
stops: the time from the last loud frame (within 10 dB of the line's peak) until
the level falls 25 dB below the peak, on the separated vocals. Only the stretch
before the next speech (VAD) or the next window counts, so a following speaker
isn't mistaken for echo. Decay varies with the recording and the room.
Lines at or above WET_MS are flagged for review: give them `"effect": "echo"`
(or `"pa"`) and extend their windows over the original's tail, or the
original's Japanese echo plays on after the English. When a line is
rendered, the take's own tail is shown too, for comparison.
"""

import argparse
import sys

import numpy as np
import soundfile as sf

from core import JOBS, langs, line_id, load_json, part_dir, take_wav
from edges import HOP, SR, frames_db
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WET_MS = 200


def tail_ms(db, a, b, limit):
    """ms from the last loud frame of [a, b] until 25 dB below its peak, capped at `limit` (s)."""
    ka, kb = int(a * SR / HOP), min(len(db) - 1, int(b * SR / HOP))
    if kb - ka < 10:
        return None
    peak = float(np.percentile(db[ka:kb], 95))
    k = kb
    while k > ka and db[k] < peak - 10:
        k -= 1
    stop = min(len(db) - 1, int(limit * SR / HOP))
    j = k
    while j < stop and db[j] >= peak - 25:
        j += 1
    return (j - k) * HOP / SR * 1000, j >= stop


def rows(name):
    """Each line's reverb tail: (i, line, original ms, capped, the take's ms or None),
    for lines whose tail can be measured. WET_MS or more is wet."""
    lines = load_json(part_dir(name) / "script.json")["lines"]
    job = load_json(part_dir(name) / "segments.json")["job_id"]
    audio = decode_audio(str(JOBS / job / "vocals.wav"), sampling_rate=SR)
    db = frames_db(audio)
    onsets = [
        s["start"] / SR
        for s in get_speech_timestamps(audio, VadOptions(threshold=0.5, min_silence_duration_ms=120, speech_pad_ms=0))
    ]
    out = []
    for i, line in enumerate(lines):
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else line["end"] + 2
        nxt = min([o for o in onsets if o > line["end"] - 0.05] + [nxt])
        got = tail_ms(db, line["start"], line["end"], max(nxt, line["end"]))
        if got is None:
            continue
        ms, capped = got
        take = None
        wav = take_wav(job, line_id(line), langs(name)[1])
        if wav.exists():
            x, sr = sf.read(wav, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            x = np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x)  # to 16 kHz
            tdb = frames_db(np.concatenate([x, np.zeros(SR, dtype="float32")]))
            loud = np.nonzero(tdb > tdb.max() - 30)[0]
            t = tail_ms(tdb, loud[0] * HOP / SR, (loud[-1] + 1) * HOP / SR, len(tdb) * HOP / SR)
            take = t[0] if t else None
        out.append((i, line, ms, capped, take))
    return out


def row_text(i, line, ms, capped, take):
    flag = ms >= WET_MS
    take = (
        f"   take {take:4.0f} ms{' (' + line['effect'] + ')' if line.get('effect') else ''}" if take is not None else ""
    )
    return (
        f"  {'WET ' if flag else '    '} L{i:02d} {line['start']:7.2f}  original {ms:4.0f}{'+' if capped else ' '}ms"
        f"{take}  {line['text'][:44]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--all", action="store_true", help="print every line, not just wet ones")
    args = ap.parse_args()
    found = rows(args.name)
    wet = 0
    for row in found:
        flag = row[2] >= WET_MS
        wet += flag
        if flag or args.all:
            print(row_text(*row))
    lines = load_json(part_dir(args.name) / "script.json")["lines"]
    print(f"{args.name}: {wet} of {len(lines)} lines with a reverb tail of {WET_MS} ms or more")


if __name__ == "__main__":
    main()
