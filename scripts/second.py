"""Second opinions on a clip's transcript from two other recognisers, to catch Whisper's mishearings.

Usage (needs sherpa-onnx, so run it in its own env):
  uv run --no-project --python 3.12 --with sherpa-onnx --with soundfile --with scipy
      --with pykakasi --with huggingface_hub python scripts/second.py <clip> [<clip> ...] [--script] [--all]
  ... scripts/second.py <clip> <t0>:<t1> [<t0>:<t1> ...]      any stretches of the clip
It runs on the CPU, so it can run beside GPU work (drafting, rendering).

Runs the two recognisers the part's source language names (its segments.json's
source_lang; SECOND_OPINIONS in languages/<code>.py) over each line of
<part>/draft.json (or script.json with --script) on the separated vocals,
and compares them with the line's original (`ja`, or the field of the source
language) as the language's module says (units): Japanese as kana, so spelling
choices don't count, Chinese character by character, English and Spanish word
by word, accents aside, Korean syllable by syllable, spacing aside. Japanese gets
ReazonSpeech k2 v2 (trained on Japanese TV) and SenseVoice, English NVIDIA's Parakeet TDT 0.6B v2 and
SenseVoice, Spanish Parakeet TDT 0.6B v3 and Canary 180M Flash, Korean k2's
Korean Zipformer and SenseVoice, Chinese Zipformer CTC and SenseVoice. A line is
flagged when the two agree with each other more than either agrees with Whisper,
and differ from it by 30% or more.
Treat agreement as a review signal: re-listen with asr_words.py and decide from
context. The models (~850 MB for Japanese, ~1 GB for English, ~900 MB for
Spanish, ~320 MB for Korean) download to VoiceStudio's model cache on first use.
"""

import argparse
import json
import os
import sys

import numpy as np
import sherpa_onnx
import soundfile as sf
from huggingface_hub import snapshot_download
from scipy.signal import resample_poly

from core import default_cache_dir, default_data_dir, part_dir
from languages import LANGUAGES, pair_of

# Same resolution as core.py.
JOBS = default_data_dir() / "dub_jobs"
HF_CACHE = default_cache_dir()
SR = 16000
PAD = 0.15  # s around each line
FLAG = 0.3
sys.stdout.reconfigure(encoding="utf-8")


def recognisers(lang):
    """The two recognisers the language names (its SECOND_OPINIONS), by name."""
    return {kind: recogniser(kind, repo, lang) for kind, repo in LANGUAGES[lang].SECOND_OPINIONS.items()}


def recogniser(kind, repo, lang):
    """A sherpa-onnx recogniser of one of the families second.py knows, from its
    Hugging Face repo, listening for `lang`."""
    part = lambda d, f: os.path.join(d, f)  # noqa: E731
    if kind == "parakeet":
        pk = snapshot_download(repo, cache_dir=HF_CACHE, allow_patterns=["*.int8.onnx", "tokens.txt"])
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=part(pk, "encoder.int8.onnx"),
            decoder=part(pk, "decoder.int8.onnx"),
            joiner=part(pk, "joiner.int8.onnx"),
            tokens=part(pk, "tokens.txt"),
            num_threads=8,
            sample_rate=SR,
            feature_dim=128,
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )
    if kind == "canary":
        cn = snapshot_download(repo, cache_dir=HF_CACHE, allow_patterns=["*.int8.onnx", "tokens.txt"])
        return sherpa_onnx.OfflineRecognizer.from_nemo_canary(
            encoder=part(cn, "encoder.int8.onnx"),
            decoder=part(cn, "decoder.int8.onnx"),
            tokens=part(cn, "tokens.txt"),
            src_lang=lang,
            tgt_lang=lang,
            num_threads=8,
        )
    if kind == "sensevoice":
        sv = snapshot_download(repo, cache_dir=HF_CACHE, allow_patterns=["model.int8.onnx", "tokens.txt"])
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=part(sv, "model.int8.onnx"), tokens=part(sv, "tokens.txt"), num_threads=8, language=lang, use_itn=True
        )
    if kind == "zipformer_ctc":
        zf = snapshot_download(repo, cache_dir=HF_CACHE, allow_patterns=["model.int8.onnx", "tokens.txt"])
        return sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=part(zf, "model.int8.onnx"),
            tokens=part(zf, "tokens.txt"),
            num_threads=8,
            sample_rate=SR,
            feature_dim=80,
            decoding_method="greedy_search",
        )
    if kind in ("reazon", "zipformer"):
        # k2 Zipformer transducers: ReazonSpeech's in full precision, the others int8.
        name = "{}-epoch-99-avg-1" + (".onnx" if kind == "reazon" else ".int8.onnx")
        zf = snapshot_download(repo, cache_dir=HF_CACHE, allow_patterns=[name.format("*"), "tokens.txt"])
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=part(zf, name.format("encoder")),
            decoder=part(zf, name.format("decoder")),
            joiner=part(zf, name.format("joiner")),
            tokens=part(zf, "tokens.txt"),
            num_threads=8,
            sample_rate=SR,
            feature_dim=80,
            decoding_method="modified_beam_search",
            max_active_paths=4,
        )
    raise SystemExit(f"second.py knows no {kind!r} recogniser (SECOND_OPINIONS in languages/{lang}.py)")


def dist(a, b, lang="ja"):
    """Edit distance between two texts (as the language's units: characters or words),
    relative to the longer."""
    a, b = LANGUAGES[lang].units(a), LANGUAGES[lang].units(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "targets",
        nargs="+",
        metavar="clip",
        help="one or more clips (the models load once), or one clip followed by t0:t1 stretches",
    )
    ap.add_argument("--script", action="store_true", help="check script.json instead of draft.json")
    ap.add_argument("--all", action="store_true", help="print every line, not just flagged ones")
    args = ap.parse_args()
    clips = [t for t in args.targets if ":" not in t]
    spans = [t for t in args.targets if ":" in t]
    if spans and len(clips) != 1:
        ap.error("t0:t1 stretches need exactly one clip")
    recs = {}
    for clip in clips:
        lang = source_lang(clip)
        if lang not in LANGUAGES or not LANGUAGES[lang].SECOND_OPINIONS:
            print(f"{clip}: no second opinions for {lang}")
            continue
        if lang not in recs:
            recs[lang] = recognisers(lang)
        check(clip, spans, recs[lang], lang, args.script, args.all)


def source_lang(clip):
    """The language a part speaks, as dub.py prep recorded it (core.langs)."""
    return pair_of(json.loads((part_dir(clip) / "segments.json").read_text(encoding="utf-8")))[0]


def decode(rec, audio):
    """A recogniser's transcript of a stretch, or "" when it rejects it (an odd
    shape). Canary warns on stderr for most lines that its first token was
    end-of-text (sherpa-onnx issue #3919, harmless), so stderr is muted meanwhile."""
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), 2)
            s = rec.create_stream()
            s.accept_waveform(SR, audio)
            rec.decode_stream(s)
            return s.result.text.strip()
    except RuntimeError:  # the model rejects an odd-shaped stretch; skip that opinion
        return ""
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def check(clip, spans, recs, lang, script, show_all):
    job = json.loads((part_dir(clip) / "segments.json").read_text(encoding="utf-8"))["job_id"]
    x, sr = sf.read(JOBS / job / "vocals.wav", dtype="float32")
    x = resample_poly(x.mean(1) if x.ndim > 1 else x, SR, sr).astype(np.float32)
    if spans:
        lines = [{**dict(zip(("start", "end"), map(float, s.split(":")))), lang: ""} for s in spans]
    else:
        name = "script.json" if script else "draft.json"
        lines = json.loads((part_dir(clip) / name).read_text(encoding="utf-8"))["lines"]
    flagged = 0
    for i, line in enumerate(lines):
        a, b = max(0.0, line["start"] - PAD), line["end"] + PAD
        clip_audio = x[int(a * SR) : int(b * SR)]
        if len(clip_audio) < 0.1 * SR or line.get("lang"):
            continue  # a draft line at (or past) the clip's very end, or not in the source language
        got = {name: decode(rec, clip_audio) for name, rec in recs.items()}
        whisper = line.get(lang, "")
        one, two = list(recs)
        agree = dist(got[one], got[two], lang)
        gap = min(dist(whisper, got[one], lang), dist(whisper, got[two], lang))
        flag = bool(whisper) and gap >= FLAG and agree < gap
        flagged += flag
        if flag or show_all or spans:
            print(f"{'FLAG' if flag else '    '} L{i:02d} {line['start']:7.2f}-{line['end']:7.2f}  {whisper}")
            print(f"{'':25s}{one}: {got[one]}   {two}: {got[two]}")
    if not spans:
        print(f"{clip}: {flagged} of {len(lines)} lines where the other two agree against the transcript")


if __name__ == "__main__":
    main()
