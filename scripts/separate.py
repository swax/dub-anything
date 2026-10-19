"""Re-separate prepped clips' vocals with a Roformer model instead of VoiceStudio's htdemucs.

Usage (its own env, see AGENTS.md):
  scripts/separate.py <clip> [<clip> ...] [--model vocals_mel_band_roformer.ckpt]
  scripts/separate.py <clip> --restore            put htdemucs's stems back

Residual background music in separated vocals can affect voice references,
cloned lines, drafts, register measurements and level matching. Compare the
stems produced by each separator on your source material.

This runs audio-separator on the job's original audio (audio_hq.wav) and swaps
the new stems in as vocals.wav / no_vocals.wav, keeping htdemucs's as
*.htdemucs.wav. The background stem is the mix minus the vocals, as before.
A part prepped for subtitles only (`dub.py prep --subtitles-only`) has no
htdemucs stems: its only stems are these.
Run it right after `dub.py prep`, before draft.py and before building voices
from the clip. A voice already built keeps its old reference until its entry in
voices.json changes.
"""

import argparse
import json
import pathlib
import shutil
import sys
import tempfile

from core import cache_root, default_data_dir
from gpulock import hold_gpu

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
# Same resolution as core.py / VoiceStudio's backend/core/config.py.
JOBS = default_data_dir() / "dub_jobs"
MODELS = cache_root() / "audio-separator"
STEMS = ("vocals.wav", "no_vocals.wav")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def job_dir(clip):
    seg = json.loads((WORK / clip / "segments.json").read_text(encoding="utf-8"))
    return JOBS / seg["job_id"]


def restore(clip):
    job = job_dir(clip)
    for stem in STEMS:
        old = job / stem.replace(".wav", ".htdemucs.wav")
        if old.exists():
            shutil.move(old, job / stem)
    (job / "separator.json").unlink(missing_ok=True)
    print(f"{clip}: htdemucs stems restored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument(
        "--model",
        default="vocals_mel_band_roformer.ckpt",
        help="an audio-separator vocals model (default: Kim's Mel-Band Roformer)",
    )
    ap.add_argument("--restore", action="store_true", help="put VoiceStudio's htdemucs stems back")
    args = ap.parse_args()
    if args.restore:
        for clip in args.clips:
            restore(clip)
        return

    import torch
    from audio_separator.separator import Separator

    from gpu import rocm_settings

    hold_gpu()
    rocm_settings(torch)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="separate-"))
    sep = Separator(model_file_dir=str(MODELS), output_dir=str(tmp), output_format="WAV")
    sep.load_model(model_filename=args.model)
    for clip in args.clips:
        job = job_dir(clip)
        mark = job / "separator.json"
        if mark.exists() and json.loads(mark.read_text(encoding="utf-8")).get("model") == args.model:
            print(f"{clip}: already separated with {args.model}")
            continue
        out = [tmp / f for f in sep.separate(str(job / "audio_hq.wav"))]
        vocals = next(f for f in out if "(vocals)" in f.name.lower())
        bed = next(f for f in out if f is not vocals)
        for stem in STEMS:
            keep = job / stem.replace(".wav", ".htdemucs.wav")
            if not keep.exists() and (job / stem).exists():
                shutil.copy2(job / stem, keep)
        shutil.move(vocals, job / "vocals.wav")
        shutil.move(bed, job / "no_vocals.wav")
        mark.write_text(json.dumps({"model": args.model}), encoding="utf-8")
        kept = (job / "vocals.htdemucs.wav").exists()
        print(f"{clip}: stems from {args.model}" + (" (htdemucs kept as *.htdemucs.wav)" if kept else ""))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
