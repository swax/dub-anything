"""Careful transcription with word timings (beam 5, no VAD filter), in the source language.

Usage: scripts/asr_words.py <wav> [offset_seconds]         a wav file (offset is added to printed times)
       scripts/asr_words.py <clip> <t0> <t1>               a stretch of a prepped clip's separated vocals,
                                                           plus where the VAD hears speech in it
Options: --model <repo>  (default Systran/faster-whisper-large-v3)
         --lang en    the language spoken, e.g. for speech already in the target language (default:
                      the clip's source language; for a voice's reference, work/voices/<key>.wav, the
                      source language of the part it was built from; else ja)
"""

import argparse
import pathlib
import sys

from core import JOBS, WORK, langs, load_json, whisper_setup
from gpulock import hold_gpu

whisper_setup()  # VoiceStudio's downloaded Whisper + cuDNN 8
from faster_whisper.audio import decode_audio  # noqa: E402
from faster_whisper.vad import VadOptions, get_speech_timestamps  # noqa: E402
from whisper_safe import WhisperModel  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("source", help="a wav file, or a prepped clip name")
ap.add_argument("times", nargs="*", type=float, help="offset for a wav; t0 t1 for a clip")
ap.add_argument("--model", default="Systran/faster-whisper-large-v3")
ap.add_argument("--lang", help="language spoken (default: the clip's source language)")
args = ap.parse_args()

seg_doc = WORK / args.source / "segments.json"
if args.lang is None:
    voice = load_json(WORK / "voices.json", {}).get(pathlib.Path(args.source).stem, {})
    args.lang = langs(args.source)[0] if seg_doc.exists() else langs(voice["from"])[0] if voice.get("from") else "ja"
if seg_doc.exists():
    if len(args.times) != 2:
        raise SystemExit("usage: scripts/asr_words.py <clip> <t0> <t1>")
    t0, t1 = args.times
    audio = decode_audio(str(JOBS / load_json(seg_doc)["job_id"] / "vocals.wav"), sampling_rate=16000)
    audio, offset = audio[int(t0 * 16000) : int(t1 * 16000)], t0
    spans = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=120, speech_pad_ms=30))
    print("speech: " + " ".join(f"{offset + s['start'] / 16000:.2f}-{offset + s['end'] / 16000:.2f}" for s in spans))
    source = audio
else:
    source, offset = args.source, (args.times[0] if args.times else 0.0)

hold_gpu()
model = WhisperModel(args.model, device="auto", compute_type="default")
segs, _ = model.transcribe(
    source, language=args.lang, beam_size=5, vad_filter=False, condition_on_previous_text=False, word_timestamps=True
)
for s in segs:
    words = " ".join(f"{w.word}[{w.start + offset:.2f}]" for w in s.words)
    print(f"{s.start + offset:6.2f}-{s.end + offset:6.2f}  {s.text}\n        {words}")
