"""Median voice pitch per transcribed phrase, to sanity-check speaker labels.

Usage: scripts/pitch.py <name>   (reads <part>/segments.json and that job's vocals.wav)
Needs librosa, so run it in a throwaway env:
  uv run --no-project --python 3.12 --with librosa --with soundfile python scripts/pitch.py <clip>
"""

import json
import sys

import librosa
import numpy as np

from core import default_data_dir, part_dir

# Same job-folder resolution as core.py / VoiceStudio's backend/core/config.py.
JOBS = default_data_dir() / "dub_jobs"

sys.stdout.reconfigure(encoding="utf-8")
work = part_dir(sys.argv[1])
doc = json.loads((work / "segments.json").read_text(encoding="utf-8"))
wav = JOBS / doc["job_id"] / "vocals.wav"
y, sr = librosa.load(wav, sr=16000, mono=True)

for seg in doc["segments"]:
    for w in seg.get("words") or [{"text": seg["text"], "start": seg["start"], "end": seg["end"]}]:
        clip = y[int(w["start"] * sr) : int(w["end"] * sr)]
        if len(clip) < sr // 10:
            continue
        f0, voiced, _ = librosa.pyin(clip, fmin=70, fmax=450, sr=sr, frame_length=1024)
        f0 = f0[voiced]
        med = f"{np.median(f0):5.0f} Hz" if len(f0) else "   -- Hz"
        print(f"{seg['id']:9s} {w['start']:6.2f}-{w['end']:6.2f}  {seg.get('speaker_id'):10s} {med}  {w['text']}")
