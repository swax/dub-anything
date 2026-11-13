"""Re-separate prepped clips' vocals with a Roformer model instead of VoiceStudio's htdemucs.

Usage (its own env, see AGENTS.md; gpu.separate_command() builds it):
  scripts/separate.py <clip> [<clip> ...] [--model vocals_mel_band_roformer.ckpt]
  scripts/separate.py <clip> --job <VoiceStudio job id>   a part dub.py prep is still preparing
  scripts/separate.py <clip> --restore            put htdemucs's stems back

Residual background music in separated vocals can affect voice references,
cloned lines, drafts, register measurements and level matching. Compare the
stems produced by each separator on your source material.

This runs audio-separator on the job's original audio (audio_hq.wav) and swaps
the new stems in as vocals.wav / no_vocals.wav, keeping any htdemucs stems as
*.htdemucs.wav. The background stem is the mix minus the vocals, as before.
`dub.py prep` runs it itself: VoiceStudio uploads the part without separating
it (no htdemucs), and prep attaches these stems to the job before transcribing.
Parts prepped before that, and parts prepped for subtitles only (`dub.py prep
--subtitles-only`, no backend), are separated by `dub_anything.py run` or by
hand, before draft.py and before building voices from the clip. A voice already
built keeps its old reference until its entry in voices.json changes.

Stems are swapped in with a rename, never rewritten: a job forked for another
voice (`dub.py fork`) shares them by hard link.
"""

import argparse
import json
import pathlib
import shutil
import sys
import tempfile

from core import cache_root, default_data_dir, part_dir
from gpulock import hold_gpu

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Same resolution as core.py / VoiceStudio's backend/core/config.py.
JOBS = default_data_dir() / "dub_jobs"
MODELS = cache_root() / "audio-separator"
STEMS = ("vocals.wav", "no_vocals.wav")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def job_dir(clip, job_id=None):
    if job_id:
        return JOBS / job_id
    seg = json.loads((part_dir(clip) / "segments.json").read_text(encoding="utf-8"))
    return JOBS / seg["job_id"]


def swap_in(src, dst):
    """Put `src` at `dst` with a rename on dst's volume: a copy into dst would
    write through a hard link into the other job sharing it."""
    partial = dst.with_name(f"{dst.stem}.partial{dst.suffix}")
    shutil.copy2(src, partial)
    partial.replace(dst)
    pathlib.Path(src).unlink(missing_ok=True)


def restore(clip):
    job = job_dir(clip)
    kept = [job / stem.replace(".wav", ".htdemucs.wav") for stem in STEMS]
    if not all(k.exists() for k in kept):
        print(f"{clip}: no htdemucs stems kept (VoiceStudio didn't separate this part); nothing to restore")
        return
    for old, stem in zip(kept, STEMS):
        old.replace(job / stem)
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
    ap.add_argument("--job", help="the VoiceStudio job to separate, for one clip prep hasn't finished")
    args = ap.parse_args()
    if args.job and len(args.clips) != 1:
        ap.error("--job names one clip's job")
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
        job = job_dir(clip, args.job)
        mark = job / "separator.json"
        if mark.exists() and json.loads(mark.read_text(encoding="utf-8")).get("model") == args.model:
            print(f"{clip}: already separated with {args.model}")
            continue
        out = [tmp / f for f in sep.separate(str(job / "audio_hq.wav"))]
        vocals = next(f for f in out if "(vocals)" in f.name.lower())
        bed = next(f for f in out if f is not vocals)
        for stem in STEMS:
            keep = job / stem.replace(".wav", ".htdemucs.wav")
            if not keep.exists() and (job / stem).exists() and not mark.exists():
                shutil.copy2(job / stem, keep)  # htdemucs's, from VoiceStudio's own separation
        swap_in(vocals, job / "vocals.wav")
        swap_in(bed, job / "no_vocals.wav")
        mark.write_text(json.dumps({"model": args.model}), encoding="utf-8")
        kept = (job / "vocals.htdemucs.wav").exists()
        print(f"{clip}: stems from {args.model}" + (" (htdemucs kept as *.htdemucs.wav)" if kept else ""))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
