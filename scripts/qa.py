"""Round-trip QA: transcribe a dubbed MP4 in its language and line it up with the script.

Usage: scripts/qa.py <name> [tag]
"""

import json
import pathlib
import subprocess
import sys
import tempfile

from core import WORK, langs, out_dir, whisper_setup
from gpulock import hold_gpu

whisper_setup()  # VoiceStudio's downloaded Whisper + cuDNN 8
from whisper_safe import WhisperModel  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
name = sys.argv[1]
tag = f".{sys.argv[2]}" if len(sys.argv) > 2 else ""
doc = json.loads((WORK / name / "segments.json").read_text(encoding="utf-8"))
tgt = langs(name)[1]
mp4 = out_dir(doc) / f"{name}.{tgt}{tag}.mp4"
script = json.loads((WORK / name / "script.json").read_text(encoding="utf-8"))

wav = pathlib.Path(tempfile.gettempdir()) / f"qa-{name}.wav"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-ac", "1", "-ar", "16000", str(wav)], check=True)
hold_gpu()
model = WhisperModel("Systran/faster-whisper-large-v3", device="auto", compute_type="default")
heard = []
for lang in (tgt, None):
    segs, info = model.transcribe(
        str(wav), language=lang, beam_size=5, vad_filter=True, word_timestamps=True, condition_on_previous_text=False
    )
    words = [w for s in segs for w in s.words]
    if lang == tgt:
        heard = words
    else:
        print(f"auto-detected language: {info.language} ({info.language_probability:.2f})")

for line in script["lines"]:
    got = "".join(w.word for w in heard if line["start"] - 0.1 <= w.start < line["end"] + 0.3).strip()
    print(f"{line['start']:6.2f}-{line['end']:6.2f} {line['voice']:9s} want: {line['text']}\n{'':26s} got:  {got}")
stray = [w for w in heard if not any(l["start"] - 0.1 <= w.start < l["end"] + 0.3 for l in script["lines"])]
if stray:
    print("outside script windows:", " ".join(f"{w.word.strip()}@{w.start:.1f}" for w in stray))
