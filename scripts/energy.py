"""Print a per-second loudness bar chart of a clip's separated vocals vs background.

Usage: scripts/energy.py <name>   (after `dub.py prep`; reads <part>/segments.json)
"""

import json
import sys

import numpy as np
import soundfile as sf

from core import JOBS, part_dir

job = json.loads((part_dir(sys.argv[1]) / "segments.json").read_text(encoding="utf-8"))["job_id"]
rows = []
for stem in ("vocals", "no_vocals"):
    audio, sr = sf.read(JOBS / job / f"{stem}.wav")
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    secs = len(mono) // sr
    rows.append([20 * np.log10(np.sqrt(np.mean(mono[i * sr : (i + 1) * sr] ** 2)) + 1e-9) for i in range(secs)])
print(" sec  vocals dB            background dB")
for i, (v, b) in enumerate(zip(*rows)):
    print(f"{i:4d}  {v:6.1f} {'#' * max(0, int((v + 60) / 3)):15s}  {b:6.1f} {'#' * max(0, int((b + 60) / 3))}")
