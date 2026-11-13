"""Compare each script window with where the original actor's voice actually is.

Usage: scripts/onsets.py <name>   Voice activity = 20 ms frames of the separated vocals
within 25 dB of the clip's loudest speech. Reports the active span found near
each window, so windows can be snapped to real onsets.
"""

import json
import sys

import numpy as np
import soundfile as sf

from core import JOBS, part_dir

name = sys.argv[1]
job = json.loads((part_dir(name) / "segments.json").read_text(encoding="utf-8"))["job_id"]
lines = json.loads((part_dir(name) / "script.json").read_text(encoding="utf-8"))["lines"]
audio, sr = sf.read(JOBS / job / "vocals.wav", dtype="float32")
mono = audio.mean(axis=1) if audio.ndim > 1 else audio
hop = sr // 50
db = 10 * np.log10(np.array([np.mean(mono[i : i + hop] ** 2) for i in range(0, len(mono) - hop, hop)]) + 1e-12)
active = db > np.percentile(db, 99) - 25

for i, line in enumerate(lines):
    prev_end = lines[i - 1]["end"] if i else 0.0
    next_start = lines[i + 1]["start"] if i + 1 < len(lines) else len(mono) / sr
    lo, hi = max(prev_end, line["start"] - 0.6), min(next_start, line["end"] + 0.6)
    idx = [k for k in range(int(lo * 50), int(hi * 50)) if k < len(active) and active[k]]
    if not idx:
        print(f"L{i:02d} {line['start']:6.2f}-{line['end']:6.2f}  no voice found   {line['text']}")
        continue
    on, off = idx[0] / 50, (idx[-1] + 1) / 50
    note = []
    if on - line["start"] > 0.25:
        note.append(f"starts {on - line['start']:.2f}s early")
    if line["start"] - on > 0.1:
        note.append(f"voice begins {line['start'] - on:.2f}s before window")
    if off - line["end"] > 0.1:
        note.append(f"voice runs {off - line['end']:.2f}s past window")
    print(
        f"L{i:02d} {line['start']:6.2f}-{line['end']:6.2f}  voice {on:6.2f}-{off:6.2f}  "
        f"{'; '.join(note):40s} {line['text']}"
    )
