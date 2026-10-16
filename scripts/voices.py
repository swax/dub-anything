"""Build and check the cast's voices (work/voices.json).

Usage:
  scripts/voices.py spans <clip> <a-b> [<a-b> ...]     where the VAD hears speech around candidate clips (CPU)
  scripts/voices.py add <key> <from-clip> '<clips>' '<ref_text>' [--replace]
  scripts/voices.py check <key> [<key> ...]            transcribe each clip of a voice's joined reference (GPU)
  scripts/voices.py sim <key> <key> [<key> ...]        speaker-embedding similarity between voices' references (GPU)
  scripts/voices.py score <clip> <a-b> [<a-b> ...]     who each stretch sounds like, against the show's cast (GPU)

spans: a reference clip must start and end in a pause. Each candidate a-b (seconds in
the clip's separated vocals) is shown with the speech spans around it: OK when none
crosses its edges or comes within 0.15 s of them, CROSS when one runs across an edge,
TIGHT when one starts or stops just outside it.

add: '<clips>' is JSON, [[a, b], ["<other clip>", a, b], ...], as in voices.json. The
entry is written under a lock, so directors working on a show in parallel can add
voices at the same time, and work/voices/<key>.wav is built. An existing key is refused
unless --replace: voices.json holds every show's cast, so a character name used by
another show is taken (pick another key). Replacing a voice re-renders all its lines.

check: every phrase of a reference must be complete and its ref_text must say exactly
what is said (see "Voice-reference leakage" in docs/directing.md).

sim: two references of the same person score well above unrelated voices (in one
film, 0.5 against 0.1-0.3), e.g. when two directors built the same character.

score: the speaker scores draft.py gives a line, for any stretch of a clip's vocals
(half of a line, a window you're about to draw), against the show's own voices and the
lines already cast to them in its other parts, best three first.
"""

import json
import sys

from core import JOBS, WORK, langs, load_json, update_voices, voice_clips, voice_wav, whisper_setup

sys.stdout.reconfigure(encoding="utf-8")
SR = 16000


def spans(clip, ranges):
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    seg = load_json(WORK / clip / "segments.json")
    audio = decode_audio(str(JOBS / seg["job_id"] / "vocals.wav"), sampling_rate=SR)
    found = [
        (s["start"] / SR, s["end"] / SR)
        for s in get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=120, speech_pad_ms=30))
    ]
    for rng in ranges:
        a, b = (float(x) for x in rng.split("-"))
        near = [(s, e) for s, e in found if e > a - 1.5 and s < b + 1.5]
        edge = 0.02  # VAD times are rounded; a span starting at the clip's start is fine
        cross = any(s < t - edge and e > t + edge for s, e in near for t in (a, b))
        tight = any(a - 0.15 < e <= a - edge or b + edge <= s < b + 0.15 for s, e in near)
        flag = "CROSS" if cross else "TIGHT" if tight else "OK"
        print(f"{clip} {a:.2f}-{b:.2f} {flag:5s} " + " ".join(f"{s:.2f}-{e:.2f}" for s, e in near))


def add(key, src, clips, text, replace):
    clips = json.loads(clips)

    def change(voices):
        if key in voices and not replace:
            raise SystemExit(
                f"{key} exists already (from {voices[key].get('from')}): pick another key, "
                f"or --replace a voice you made yourself"
            )
        voices[key] = {"from": src, "clips": clips, "ref_text": text}

    update_voices(change)
    used = sum(line.get("voice") == key for p in WORK.glob("*/script.json") for line in load_json(p)["lines"])
    total = sum(b - a for _, a, b in voice_clips(load_json(WORK / "voices.json")[key]))
    print(
        f"{key}: {total:.1f} s of clips -> {voice_wav(key, rebuild=True)}"
        + (f"; {used} script lines use it and will render again" if used else "")
    )


def check(keys):
    from gpulock import hold_gpu

    voices = load_json(WORK / "voices.json")
    wavs = {k: voice_wav(k) for k in keys}
    hold_gpu()
    whisper_setup()
    from faster_whisper.audio import decode_audio
    from whisper_safe import WhisperModel

    model = WhisperModel("Systran/faster-whisper-large-v3", device="auto", compute_type="default")
    for k in keys:
        audio = decode_audio(str(wavs[k]), sampling_rate=SR)
        print(f"=== {k}  {len(audio) / SR:.1f} s  ref_text: {voices[k]['ref_text']}")
        t = 0.0
        for clip, a, b in voice_clips(voices[k]):
            piece = audio[int(t * SR) : int((t + b - a) * SR)]
            segs, _ = model.transcribe(
                piece, language=langs(clip)[0], beam_size=5, vad_filter=False, condition_on_previous_text=False
            )
            print(f"   {clip} {a:.2f}-{b:.2f} ({b - a:.1f} s): {''.join(s.text for s in segs).strip()}")
            t += b - a + 0.3


def sim(keys):
    import draft
    from faster_whisper.audio import decode_audio

    embed = draft.embedder()
    if embed is None:
        raise SystemExit("no WeSpeaker model in the cache (see draft.py)")
    vecs = {k: embed(decode_audio(str(voice_wav(k)), sampling_rate=SR)) for k in keys}
    width = max(len(k) for k in keys)
    print(" " * width + "".join(f" {k:>{max(6, len(k))}s}" for k in keys))
    for a in keys:
        print(f"{a:{width}s}" + "".join(f" {float(vecs[a] @ vecs[b]):>{max(6, len(b))}.2f}" for b in keys))


def score(clip, ranges):
    import draft

    embed = draft.embedder()
    if embed is None:
        raise SystemExit("no WeSpeaker model in the cache (see draft.py)")
    cents = draft.cast_centroids(embed, draft.show_cast(clip) or None, exclude=clip)
    seg = load_json(WORK / clip / "segments.json")
    audio = draft.decode_audio(str(JOBS / seg["job_id"] / "vocals.wav"), sampling_rate=SR)
    for rng in ranges:
        a, b = (float(x) for x in rng.split("-"))
        v = embed(audio[int(a * SR) : int(b * SR)])
        ranked = sorted(((float(v @ c), k) for k, c in cents.items()), reverse=True)[:3]
        print(
            f"{clip} {a:.2f}-{b:.2f}  "
            + "  ".join(f"{k} {s:.2f}" for s, k in ranked)
            + ("  (under 0.6 s: unreliable)" if b - a < 0.6 else "")
        )


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    if cmd == "spans" and len(args) >= 3:
        spans(args[1], args[2:])
    elif cmd == "add" and len([a for a in args if a != "--replace"]) == 5:
        add(*[a for a in args[1:] if a != "--replace"], "--replace" in args)
    elif cmd == "check" and len(args) >= 2:
        check(args[1:])
    elif cmd == "sim" and len(args) >= 3:
        sim(args[1:])
    elif cmd == "score" and len(args) >= 3:
        score(args[1], args[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
