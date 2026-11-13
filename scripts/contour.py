"""Pitch/loudness contour of a clip's separated vocals in 0.2 s steps.

Usage: scripts/contour.py <name> <start> <end>
Needs librosa, so run it in a throwaway env:
  uv run --no-project --python 3.12 --with librosa --with soundfile python scripts/contour.py <clip> 10 15
"""

import json
import sys

import librosa
import numpy as np

from core import default_data_dir, part_dir

# Same job-folder resolution as core.py / VoiceStudio's backend/core/config.py.
JOBS = default_data_dir() / "dub_jobs"

name, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
job = json.loads((part_dir(name) / "segments.json").read_text(encoding="utf-8"))["job_id"]
y, sr = librosa.load(JOBS / job / "vocals.wav", sr=16000, mono=True, offset=t0, duration=t1 - t0)
hop = int(0.02 * sr)
f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=500, sr=sr, frame_length=1024, hop_length=hop)
rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
step = 10  # frames per 0.2 s
for i in range(0, len(f0), step):
    f = f0[i : i + step][voiced[i : i + step]]
    db = 20 * np.log10(rms[i : i + step].mean() + 1e-9)
    pitch = f"{np.median(f):4.0f}" if len(f) >= 3 else "   ."
    print(f"{t0 + i * 0.02:6.1f}  {db:6.1f} dB  {pitch} Hz  {'#' * max(0, int((db + 60) / 2))}")
