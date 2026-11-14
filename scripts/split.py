"""Cut a long video into ~N-minute parts at pauses in speech, to dub part by part.

Usage: scripts/split.py <video> [--minutes 3] [--whole A-B ...] [--out DIR] [--encoder auto|h264_nvenc|libx264]
Silero VAD over the full mix finds the gaps between speech; near each N-minute
mark (+-30 s) the longest gap wins and the cut goes in its middle, so no line
is split. --whole A-B (seconds, repeatable) cuts exactly at A and B and keeps
that stretch as one part: a block of songs, a commercial break, a splice
between two recordings. Parts are re-encoded so cuts are frame-accurate and dubbed parts can
be joined back without drift. Writes <out>/<stem>-NN.mp4 and <out>/parts.json
(default out: <stem>-parts beside the video).
"""

import argparse
import itertools
import json
import pathlib
import subprocess
import sys


def patch_av_open():
    try:
        import av

        orig_open = av.open

        def _compat_open(*args, **kwargs):
            if "metadata_errors" in kwargs:
                try:
                    return orig_open(*args, **kwargs)
                except TypeError:
                    kwargs.pop("metadata_errors", None)
                    return orig_open(*args, **kwargs)
            return orig_open(*args, **kwargs)

        av.open = _compat_open
    except Exception:
        pass


patch_av_open()
from faster_whisper.audio import decode_audio  # noqa: E402
from faster_whisper.vad import VadOptions, get_speech_timestamps  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR = 16000

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--minutes", type=float, default=3)
p.add_argument(
    "--whole",
    action="append",
    default=[],
    metavar="A-B",
    help="keep A-B (seconds) as one part, cut exactly at its ends (repeatable)",
)
p.add_argument("--out")
# auto: VideoToolbox on macOS, NVENC on Windows/NVIDIA when it works, else libx264.
p.add_argument("--encoder", default="auto")
args = p.parse_args()
if args.encoder == "auto":
    if sys.platform == "darwin":
        vt = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=256x256:duration=0.1",
                             "-c:v", "h264_videotoolbox", "-f", "null", "-"], capture_output=True)  # fmt: skip
        args.encoder = "h264_videotoolbox" if vt.returncode == 0 else "libx264"
    else:
        nvenc = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=256x256:duration=0.1",
                                "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)  # fmt: skip
        args.encoder = "h264_nvenc" if nvenc.returncode == 0 else "libx264"
if "nvenc" in args.encoder:
    quality = ["-preset", "p5", "-cq", "20"]
elif "videotoolbox" in args.encoder:
    quality = ["-q:v", "65"]
else:
    quality = ["-preset", "veryfast", "-crf", "18"]
video = pathlib.Path(args.video).resolve()
out = pathlib.Path(args.out) if args.out else video.parent / f"{video.stem}-parts"
# Snap cuts to whole video frames of the source (29.97, 30, 25 fps...), so parts join without drift.
rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                       "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout.strip() or "30000/1001"  # fmt: skip
num, den = (int(x) for x in rate.split("/"))
FRAME = den / num
out.mkdir(parents=True, exist_ok=True)

print(f"voice activity in {video.name}")
audio = decode_audio(str(video), sampling_rate=SR)
duration = len(audio) / SR
speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100))
gaps = [(a["end"] / SR, b["start"] / SR) for a, b in zip(speech, speech[1:])]


def snap(t):
    return round(round(t / FRAME) * FRAME, 3)


# A --whole end within a frame of the video's end is the end (no sliver of a part after it).
whole = [
    tuple(
        duration if duration - t < FRAME else t for t in (min(snap(max(float(x), 0)), duration) for x in w.split("-"))
    )
    for w in args.whole
]
bounds = sorted({0.0, duration, *(t for w in whole for t in w)})
step, cuts = args.minutes * 60, [0.0]
for lo, hi in itertools.pairwise(bounds):
    t = lo + step
    while (lo, hi) not in whole and t < hi - step / 2:
        near = [g for g in gaps if abs((g[0] + g[1]) / 2 - t) <= 30 and lo < (g[0] + g[1]) / 2 < hi]
        a, b = max(near, key=lambda g: g[1] - g[0]) if near else (t, t)
        cuts.append(snap((a + b) / 2))
        print(f"  cut {cuts[-1]:7.2f}s  (pause {b - a:4.2f}s, target {t:.0f}s)")
        t = cuts[-1] + step
    if hi < duration:
        cuts.append(hi)
        print(f"  cut {hi:7.2f}s  (--whole)")
cuts.append(duration)

parts = []
for i, (a, b) in enumerate(zip(cuts, cuts[1:]), 1):
    name = f"{video.stem}-{i:02d}"
    dst = out / f"{name}.mp4"
    print(f"{name}: {a:7.2f}-{b:7.2f}s")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(video), "-t", f"{b - a:.3f}",
                    "-map", "0:v:0", "-map", "0:a:0", "-c:v", args.encoder, *quality, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)], check=True)  # fmt: skip
    parts.append({"name": name, "file": str(dst), "start": a, "end": b})
# In one step: the runner takes the file's being there for a video that is split.
manifest = out / "parts.json.tmp"
manifest.write_text(json.dumps({"video": str(video), "parts": parts}, indent=2), encoding="utf-8")
manifest.replace(out / "parts.json")
print(f"wrote {len(parts)} parts to {out}")
