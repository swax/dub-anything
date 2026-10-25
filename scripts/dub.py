"""Drive VoiceStudio's dubbing backend (http://127.0.0.1:3900) headlessly.

prep <video> [--speakers N] [--source ja --target en]
                              upload, separate vocals, transcribe + diarize;
                              writes work/<name>/segments.json (with the language
                              pair, Japanese -> English by default) and prints the
                              transcript. A voice variant (<part>.us) is the same
                              audio as its prepped part: only uploaded, with the
                              part's transcript and pair
prep <video> --subtitles-only [--source ja --target en]
                              for subtitles alone: no backend, no voices. Writes
                              the audio into a job folder of VoiceStudio's shape
                              (local_<id>/audio_hq.wav, no transcript) for
                              separate.py to separate and draft.py to transcribe
generate <name>               reads work/<name>/script.json (hand-directed lines:
                              start/end/voice/text, optional instruct/effect;
                              voices from work/voices.json) or else <target>.json
                              ({segment_id: text}), synthesizes cloned speech in
                              the part's target language, exports MP4 + SRT. By
                              default (the script's "render" block) each line's
                              reference ends with the actor's own original of it,
                              and takes are re-timed to end where the actor stops.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import secrets
import shutil
import subprocess
import time

import numpy as np
import requests
import soundfile as sf

from core import (
    FILL_SPEED,
    FILL_TOL,
    HERE,
    HF_CACHE,
    JOBS,
    SPEED_EXPONENT,
    WORK,
    actor_spans,
    base_part,
    fit_rows,
    indextts_length,
    langs,
    line_id,
    line_refs,
    load_json,
    out_dir,
    speech_length,
    srt_time,
    take_length,
    take_wav,
    voice_sig,
    voice_wav,
)
from gpulock import hold_gpu
from languages import LANGUAGES, NAMES, PAIR, PAIRS

API = os.environ.get("VOICESTUDIO_API", "http://127.0.0.1:3900")


def flush_gpu():
    """Ask the backend to hand its cached GPU memory back (no model is unloaded).
    After many renders PyTorch's cache can hold almost all the VRAM; renders then
    crawl and long per-line references fail with "Generation failed", and tools
    run beside the backend (Whisper in fit.py, lineqa.py) have no room."""
    try:
        requests.post(f"{API}/system/flush-memory", timeout=120)
    except requests.RequestException:
        pass


def sse(resp):
    """Yield (event, data) pairs from a server-sent-events response."""
    resp.encoding = "utf-8"
    event, data = None, []
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line == "":
            if data:
                raw = "\n".join(data)
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = raw
                yield event or "message", payload
            event, data = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


def wait_task(task_id, terminal):
    """Follow /tasks/stream until a terminal event; raise on error/cancel."""
    t0 = time.time()
    with requests.get(f"{API}/tasks/stream/{task_id}", params={"after_seq": 0}, stream=True, timeout=(10, 3600)) as r:
        r.raise_for_status()
        for _, d in sse(r):
            kind = d.get("type") if isinstance(d, dict) else None
            if kind in terminal:
                print(f"  [{time.time() - t0:5.0f}s] {kind}")
                return d
            if kind in ("error", "cancelled"):
                raise RuntimeError(f"task {task_id} {kind}: {json.dumps(d, ensure_ascii=False)}")
            if kind == "progress":
                print(f"  [{time.time() - t0:5.0f}s] {d.get('current')}/{d.get('total')} {d.get('text', '')[:60]}")
            elif kind in ("warning", "routing_notice"):
                print(f"  [{time.time() - t0:5.0f}s] {kind}: {json.dumps(d, ensure_ascii=False)[:300]}")
            elif kind and kind != "demucs_progress":
                print(f"  [{time.time() - t0:5.0f}s] {kind}")
    raise RuntimeError(f"task {task_id} stream closed without {terminal}")


def ensure_diarizer(prefer="pyannote"):
    """VoiceStudio can fall back to Sortformer after installs or restarts, which
    can't handle clips over 120 s; switch to pyannote when it is installed."""
    try:
        d = requests.get(f"{API}/engines/diarisation", timeout=30).json()
    except (requests.RequestException, ValueError):
        return
    opt = next((o for o in d.get("options", []) if o.get("id") == prefer), None)
    if d.get("active") != prefer and opt and opt.get("installed"):
        requests.post(f"{API}/engines/diarisation/select", json={"engine_id": prefer}, timeout=30)
        print(f"  diarisation engine: {d.get('active')} -> {prefer}")


def use_tts_engine(engine):
    """Select VoiceStudio's TTS engine (a global setting); returns the one it replaced."""
    prev = requests.get(f"{API}/engines/tts", timeout=30).json().get("active")
    if engine and engine != prev:
        r = requests.post(f"{API}/engines/select", json={"family": "tts", "backend_id": engine}, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"select tts engine {engine}: {r.status_code} {r.text[:300]}")
        print(f"  tts engine: {prev} -> {engine}")
    return prev


def emo_refs(work, job_id, lines, ids, alpha):
    """Cut each line's original delivery from the job's separated vocals, for an
    engine that takes an emotion reference (IndexTTS 2). A line's own "emo_alpha"
    overrides the default; 0 turns it off. Returns {id: (path, alpha)}."""
    vocals = JOBS / job_id / "vocals.wav"
    info = sf.info(str(vocals))
    (work / "emo").mkdir(exist_ok=True)
    refs = {}
    for line, i in zip(lines, ids):
        a = line.get("emo_alpha", alpha)
        if not a:
            continue
        path = work / "emo" / f"{i}_{line['start']:.2f}-{line['end']:.2f}.wav"
        if not path.exists():
            audio, sr = sf.read(
                str(vocals),
                start=int(line["start"] * info.samplerate),
                stop=int(line["end"] * info.samplerate),
                dtype="float32",
            )
            sf.write(str(path), audio, sr)
        refs[i] = (str(path), a)
    return refs


def prep_subtitles(video, work, src, tgt):
    """A part to subtitle, not dub: no voices are cloned, so nothing needs VoiceStudio.
    Its job folder (named local_..., beside VoiceStudio's own) holds the audio as
    VoiceStudio would (audio_hq.wav); separate.py adds the stems and draft.py
    transcribes, and every tool that reads a part's vocals finds them there."""
    for code in (src, tgt):
        if code not in LANGUAGES:
            raise SystemExit(f"no languages/{code}.py: the scripts know {', '.join(sorted(LANGUAGES))}")
    job_id = "local_" + hashlib.sha1(str(video.resolve()).encode()).hexdigest()[:8]
    folder = JOBS / job_id
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{video.name}: audio only, for subtitles ({NAMES[src]} -> {NAMES[tgt]}); job {job_id}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:a:0", "-vn",
                    "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(folder / "audio_hq.wav")], check=True)  # fmt: skip
    doc = {
        "job_id": job_id,
        "video": str(video),
        "source_lang": src,
        "target_lang": tgt,
        "subtitles_only": True,
        "segments": [],
    }
    (work / "segments.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_prep(args):
    video = pathlib.Path(args.video)
    work = WORK / video.stem
    work.mkdir(parents=True, exist_ok=True)
    if args.subtitles_only:
        return prep_subtitles(video, work, args.source, args.target)
    base = base_part(video.stem)
    variant = base != video.stem and (WORK / base / "segments.json").exists()
    if variant:
        args.source, args.target = langs(base)
    if (args.source, args.target) not in PAIRS:
        raise SystemExit(f"{args.source} -> {args.target}: the scripts know {sorted(PAIRS)}")

    print(f"upload {video.name} ({NAMES[args.source]} -> {NAMES[args.target]})")
    with open(video, "rb") as f:
        r = requests.post(
            f"{API}/dub/upload",
            files={"video": (video.name, f, "video/mp4")},
            data={"source_lang": args.source, "input_type": "video"},
            timeout=300,
        )
    r.raise_for_status()
    job = r.json()
    job_id = job["job_id"]
    print(f"  job {job_id}")
    wait_task(job["task_id"], {"ready"})

    if variant:
        # Another voice of a prepped part: same audio, so its transcript (the renders
        # send their own lines) instead of transcribing and diarizing again.
        doc = {**load_json(WORK / base / "segments.json"), "job_id": job_id, "video": str(video)}
        (work / "segments.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        if (WORK / base / f"{args.source}.srt").exists():
            shutil.copy2(WORK / base / f"{args.source}.srt", work / f"{args.source}.srt")
        print(f"  transcript from {base} ({len(doc['segments'])} segments)")
        return

    print("transcribe + diarize")
    ensure_diarizer()
    # VoiceStudio's per-segment references serve only renders without a script (auto:
    # profiles). Cutting and transcribing them cost 7 minutes a part with Whisper on the CPU.
    params = {"per_segment_refs": "true" if args.segment_refs else "false"}
    if args.speakers:
        params["num_speakers"] = args.speakers
    final, t0 = None, time.time()
    with requests.get(f"{API}/dub/transcribe-stream/{job_id}", params=params, stream=True, timeout=(10, 3600)) as r:
        r.raise_for_status()
        for ev, d in sse(r):
            if ev == "final":
                final = d
            elif ev == "warning":
                print(f"  [{time.time() - t0:5.0f}s] warning: {json.dumps(d, ensure_ascii=False)[:300]}")
            elif ev == "error":
                raise RuntimeError(f"transcribe error: {json.dumps(d, ensure_ascii=False)}")
            elif ev == "start":
                print(f"  [{time.time() - t0:5.0f}s] start {d}")
            elif ev == "done":
                break
    if final is None:
        raise RuntimeError("transcription ended without a final event")
    print(f"  [{time.time() - t0:5.0f}s] final: {len(final['segments'])} segments")

    doc = {
        "job_id": job_id,
        "video": str(video),
        "source_lang": args.source,
        "target_lang": args.target,
        "speaker_clones": final.get("speaker_clones"),
        "segments": final["segments"],
    }
    (work / "segments.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(work / f"{args.source}.srt", "w", encoding="utf-8") as f:
        for i, s in enumerate(final["segments"], 1):
            f.write(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n[{s.get('speaker_id')}] {s['text']}\n\n")
    for s in final["segments"]:
        print(
            f"{s['id']}  {s['start']:6.2f}-{s['end']:6.2f}  {s.get('speaker_id')}  "
            f"{s.get('profile_id') or '-'}  {s['text']}"
        )


def ensure_voice(key):
    """Resolve a voice key from work/voices.json to a profile id, building a
    VoiceStudio voice profile from hand-picked clean clips when needed."""
    spec = load_json(WORK / "voices.json", {})[key]
    if "profile" in spec:
        return spec["profile"]
    sig = voice_sig(key)
    cache_path = WORK / "profiles.json"
    cache = load_json(cache_path, {})
    old = cache.get(key)
    if old and old["sig"] == sig and requests.get(f"{API}/profiles/{old['id']}", timeout=30).ok:
        return old["id"]

    ref = voice_wav(key, rebuild=True)
    with open(ref, "rb") as f:
        r = requests.post(
            f"{API}/profiles",
            files={"ref_audio": (ref.name, f, "audio/wav")},
            data={"name": f"dub-{key}", "ref_text": spec.get("ref_text", "")},
            timeout=300,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"create profile {key}: {r.status_code} {r.text[:300]}")
    pid = r.json()["id"]
    if old:
        requests.delete(f"{API}/profiles/{old['id']}", timeout=30)
    cache[key] = {"id": pid, "sig": sig}
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(f"  voice {key}: profile {pid} ({len(spec['clips'])} clips)")
    return pid


def speech_db(audio):
    """Active-speech level: RMS over 20 ms frames within 20 dB of the loudest frame."""
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    n = len(mono) // 480 * 480
    if n == 0:
        return None
    frames = 10 * np.log10((mono[:n].reshape(-1, 480) ** 2).mean(axis=1) + 1e-12)
    active = frames[frames > frames.max() - 20]
    return 10 * np.log10(np.mean(10 ** (active / 10)))


def match_gains(job_id, lines, ids, tgt):
    """Per-line gain toward the original actor's level: the clip-wide offset is
    matched fully, each line's deviation from it by half, so whispered lines stay
    quieter than shouted ones without dropping out."""
    vocals, sr = sf.read(JOBS / job_id / "vocals.wav", dtype="float32")
    diffs = []
    for line, sid in zip(lines, ids):
        clip = take_wav(job_id, sid, tgt)
        orig = speech_db(vocals[int(line["start"] * sr) : int(line["end"] * sr)])
        gen = speech_db(sf.read(clip, dtype="float32")[0]) if clip.exists() else None
        diffs.append(None if orig is None or gen is None else orig - gen)
    known = [d for d in diffs if d is not None]
    if not known:
        return [None] * len(lines)
    offset = float(np.median(known))
    gains = []
    for sid, diff in zip(ids, diffs):
        if diff is None:
            gains.append(None)
            continue
        target = offset + 0.5 * (diff - offset)
        gains.append(round(float(np.clip(10 ** (target / 20), 0.05, 2.0)), 3))
        print(f"  level {sid}: exact match {diff:+6.1f} dB, applied {target:+6.1f} dB (gain {gains[-1]:.2f})")
    return gains


# OmniVoice only accepts these instruct tags (omnivoice/utils/voice_design.py);
# anything else, including VoiceStudio's free-text `direction`, aborts the dub.
INSTRUCT_TAGS = {
    "male",
    "female",
    "child",
    "teenager",
    "young adult",
    "middle-aged",
    "elderly",
    "very low pitch",
    "low pitch",
    "moderate pitch",
    "high pitch",
    "very high pitch",
    "whisper",
    *(
        f"{a} accent"
        for a in (
            "american",
            "british",
            "australian",
            "chinese",
            "canadian",
            "indian",
            "korean",
            "portuguese",
            "russian",
            "japanese",
        )
    ),
}
EFFECTS = ["broadcast", "cinematic", "podcast", "warm", "bright", "raw", "echo", "pa"]  # echo, pa: local patch


def check_instruct(line):
    tags = [t.strip() for t in line.get("instruct", "").split(",") if t.strip()]
    bad = [t for t in tags if t not in INSTRUCT_TAGS]
    if bad:
        raise SystemExit(f"line {line['start']}: unsupported instruct {bad}; allowed: {sorted(INSTRUCT_TAGS)}")
    return ", ".join(tags)


FILL_PASSES = 2  # re-timing passes after the natural-pace render; a third changed nothing in trials
# The render's "voice": "actor" (OmniVoice clones the actor, accent and all) or "american"
# (IndexTTS speaks native English, following each line's original delivery, and vc.py
# then gives every take the actor's timbre). Which a dub into a language can have is
# its module's VOICES (languages/): the American voice is English only.
VOICES = ("actor", "american")


def file_sha(path):
    return hashlib.sha1(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def render_mode(work, script, args, pair=PAIR):
    """How a part is rendered: script.json's "render" block, {"line_ref": .., "fill": ..,
    "dips": .., "cast_lines": .., "predict_speed": .., "voice": "actor" | "american"}. A
    part without one gets the defaults (per-line references, fill, references cut at
    a dip when speech runs on past a window, and the window check's voice test against
    the characters' cast lines too: core.unclean_windows) if it has never been
    rendered (per-line references only where languages.PAIRS says so for the
    part's pair), and the earlier behaviour (all off) if it has, so approved
    takes don't change under a new default; the block is then written so later runs
    agree. The --[no-]line-ref, --[no-]fill, --[no-]predict-speed and --voice flags
    override it for one run."""
    mode = script.get("render")
    fresh = not load_json(work / "rendered.json")
    if mode is None or "dips" not in mode or ("cast_lines" not in mode and fresh):
        # "dips" and "cast_lines" came after the others: a part rendered without them keeps its
        # takes. A rendered part's block isn't rewritten for "cast_lines" (missing reads as off),
        # so its script, and dub_anything.py's record of it, stay as they were.
        mode = script["render"] = {
            "line_ref": fresh and PAIRS[pair]["line_ref"],
            "fill": fresh,
            "dips": fresh,
            **(mode or {}),
        }
        if fresh:
            mode.setdefault("cast_lines", True)
        (work / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=1), encoding="utf-8")
        print(
            f"  render mode written to script.json: {mode}"
            + ("" if fresh else " (this part was rendered before some of these became the default)")
        )
    mode = {
        "line_ref": False,
        "fill": False,
        "predict_speed": False,
        "voice": "actor",
        "dips": False,
        "cast_lines": False,
        **mode,
    }
    for key in ("line_ref", "fill", "predict_speed", "voice"):
        if getattr(args, key) is not None:
            mode[key] = getattr(args, key)
    if mode["voice"] not in VOICES:
        raise SystemExit(f"render voice {mode['voice']!r}: use one of {VOICES}")
    if mode["voice"] not in LANGUAGES[pair[1]].VOICES:
        raise SystemExit(
            f"the {mode['voice']} voice doesn't dub into {NAMES[pair[1]]}, which this part is dubbed "
            f"into (it has: {', '.join(LANGUAGES[pair[1]].VOICES)})"
        )
    if mode["voice"] == "american" and mode["line_ref"]:
        # IndexTTS takes the actor's delivery from its emotion reference instead.
        mode["line_ref"] = False
    return mode


def cmd_generate(args):
    work = WORK / args.name
    doc = json.loads((work / "segments.json").read_text(encoding="utf-8"))
    if doc.get("subtitles_only"):
        raise SystemExit(f"{args.name} was prepped for subtitles only: prep a dub job to render voices")
    job_id = doc["job_id"]
    tgt = langs(args.name)[1]
    script = load_json(work / "script.json")
    mode = (
        render_mode(work, script, args, langs(args.name))
        if script
        else {"line_ref": False, "fill": False, "predict_speed": False, "voice": "actor"}
    )
    american = mode["voice"] == "american"
    if args.engine is None:
        args.engine = "indextts2" if american else "omnivoice"
    if args.emo_alpha is None:
        args.emo_alpha = 0.9 if american and args.engine != "omnivoice" else 0.0
    # Takes whose seg_<target> file holds a voice-converted copy (id -> its hash); the
    # engine's own take is kept beside it as src_<target>_<id>.wav.
    converted_path = work / "converted.json"
    converted = load_json(converted_path, {})
    rendered_path = work / "rendered.json"  # synthesis settings each cached line was made with
    rendered = load_json(rendered_path, {})
    # Takes lineqa.py --reroll flagged: synthesized again (on the first pass) for a fresh roll,
    # or, when only the voice conversion garbled them, converted again.
    queued = load_json(work / "reroll.json", []) if script else []
    reroll = {r["id"] for r in queued if not r.get("reconvert") and not r.get("settled")}
    reconvert = {r["id"] for r in queued if r.get("reconvert")}
    spans = actor_spans(job_id, script["lines"]) if mode["fill"] else None
    ends = [span and span[1] for span in spans] if spans else None
    for line in script["lines"] if script else []:
        if re.search(r"(\.\.\.|…)\s*$", line["text"]):
            print(f"  warning {line['start']}: ends in '…'; the last word tends to get swallowed")

    def engine_take(i):
        """The engine's own take. A converted copy can start or end a little
        differently, so fill judges timing by the take it was made from."""
        seg, src = take_wav(job_id, i, tgt), take_wav(job_id, i, tgt, "src")
        if src.exists() and seg.exists() and converted.get(i) == file_sha(seg):
            return src
        return seg

    def convert(ids):
        """Give every take not converted yet the actor's timbre (vc.py, its own environment),
        and convert the engine take of each line queued for it again."""
        import vc

        todo = []
        for line, i in zip(script["lines"], ids):
            seg, src = take_wav(job_id, i, tgt), take_wav(job_id, i, tgt, "src")
            if not seg.exists():
                continue
            if converted.get(i) != file_sha(seg):
                shutil.copy2(seg, src)
            elif i not in reconvert or not src.exists():
                continue
            todo.append({"in": str(src), "target": str(voice_wav(line["voice"])), "out": str(seg), "id": i})
        if not todo:
            return False
        pairs = work / "vc-pairs.json"
        pairs.write_text(json.dumps(todo, indent=1), encoding="utf-8")
        try:
            subprocess.run(
                vc.UV_RUN + ["python", str(HERE / "vc.py"), "--pairs", str(pairs)],
                check=True,
                env={**os.environ, "HF_HUB_CACHE": HF_CACHE},
            )
        except OSError:  # not found; in WSL, whose PATH has Windows folders it can't search, permission denied
            raise SystemExit("voice conversion needs uv on PATH (see AGENTS.md)")
        for p in todo:
            converted[p["id"]] = file_sha(p["out"])
        converted_path.write_text(json.dumps(converted, indent=1), encoding="utf-8")
        return True

    def plan(first):
        """One pass: the segments to send, their ids, the settings each take is made
        with, and the ids to synthesize (None: all)."""
        segments, ids, settings = [], [], {}
        if not script:
            dubbed = json.loads((work / f"{tgt}.json").read_text(encoding="utf-8"))
            for s in doc["segments"]:
                text = dubbed.get(s["id"], "").strip()
                if text:
                    segments.append(
                        {
                            "start": s["start"],
                            "end": s["end"],
                            "text": text,
                            "profile_id": s.get("profile_id") or "",
                            "target_lang": tgt,
                        }
                    )
                    ids.append(s["id"])
            return segments, ids, settings, None
        # Hand-directed lines: own timing, speaker and text per line.
        voices = load_json(WORK / "voices.json", {})
        for line in script["lines"]:
            # The take keeps its line's id (line_id), whatever text the engine gets.
            seg = {
                "start": line["start"],
                "end": line["end"],
                "text": LANGUAGES[tgt].speakable(line["text"]),
                "profile_id": ensure_voice(line["voice"]),
                "target_lang": tgt,
                "effect_preset": line.get("effect", args.effect),
            }
            if check_instruct(line):
                seg["instruct"] = check_instruct(line)
            segments.append(seg)
            ids.append(line_id(line))
            settings[ids[-1]] = {
                "steps": args.steps,
                "effect": seg["effect_preset"],
                "instruct": seg.get("instruct", ""),
            }
            if args.engine != "omnivoice":
                settings[ids[-1]]["engine"] = args.engine
        # Emotion references: each line's own original, for engines that take one.
        emo = emo_refs(work, job_id, script["lines"], ids, args.emo_alpha) if args.engine != "omnivoice" else {}
        for seg, i in zip(segments, ids):
            if i in emo:
                seg["emo_audio"], seg["emo_alpha"] = emo[i]
                settings[i]["emo_alpha"] = emo[i][1]
        # Per-line references: the voice's clips plus the line's own original.
        refs = (
            line_refs(
                work, job_id, script["lines"], ids, verbose=first, dips=mode["dips"], cast_lines=mode["cast_lines"]
            )
            if mode["line_ref"]
            else {}
        )
        for seg, i in zip(segments, ids):
            if i in refs:
                seg["ref_audio"], seg["ref_text"] = refs[i]
                # The window, optional cut, and fingerprint of the voice audio.
                settings[i]["line_ref"] = pathlib.Path(refs[i][0]).stem.split("_", 1)[1]
        if mode["fill"]:
            # Re-time takes that stop well short of the actor or run past them:
            # render them at the speed that lands their end on the actor's last sound.
            # With predict_speed, a line with no take yet starts at the speed its
            # predicted length (take_length, or indextts_length) asks for, and every
            # correction follows at_speed rather than assuming length changes in
            # proportion to speed. IndexTTS's duration factor is exact: it does.
            power = 1 / SPEED_EXPONENT if mode["predict_speed"] and not american else 1.0
            for seg, i, end, line, span in zip(segments, ids, ends, script["lines"], spans):
                prev = rendered.get(i, {}).get("speed")
                speed, wav = prev, engine_take(i)
                length = speech_length(wav) if wav.exists() else None
                if length is None and end and mode["predict_speed"] and american:
                    length, prev, speed = indextts_length(line["text"], span[1] - span[0]), None, None
                elif (
                    length is None and end and mode["predict_speed"] and (i in refs or "clips" in voices[line["voice"]])
                ):
                    ref = refs.get(i) or (voice_wav(line["voice"]), voices[line["voice"]].get("ref_text", ""))
                    # The prediction is at natural pace, whatever speed a lost take had.
                    length, prev, speed = take_length(*ref, line["text"], lang=tgt), None, None
                if end and length and abs(length - end) > FILL_TOL:
                    speed = min(max((prev or 1.0) * (length / end) ** power, FILL_SPEED[0]), FILL_SPEED[1])
                if speed and abs(speed - 1) >= 0.01:
                    seg["speed"] = settings[i]["speed"] = round(speed, 2)
        # A take that overflows its room at fill's fastest speed can't be re-timed:
        # give it a fresh take instead (takes of one line differ by a few tenths).
        stuck = set()
        if mode["fill"] and not first:
            stuck = {
                ids[row["i"]]
                for row in fit_rows(args.name, args.overflow)
                if row["flag"] == "OVER"
                and rendered.get(ids[row["i"]], {}).get("speed")
                == settings[ids[row["i"]]].get("speed")
                == FILL_SPEED[1]
            }
        if args.engine == "indextts2":
            # Seeded takes: fill re-times a line without changing its performance
            # (IndexTTS's duration factor stretches the same take); only a re-roll
            # draws a new one.
            for seg, i in zip(segments, ids):
                seed = rendered.get(i, {}).get("seed")
                if (first and i in reroll) or i in stuck:
                    seed = secrets.randbelow(2**31)
                elif seed is None:
                    seed = int(i[1:], 16) % 2**31
                seg["seed"] = settings[i]["seed"] = seed
        if args.full and first:
            return segments, ids, settings, None
        # Missing clips, clips made with other settings, or queued re-rolls;
        # an empty list re-mixes cached speech without synthesizing anything.
        regen = [
            i
            for i in ids
            if not take_wav(job_id, i, tgt).exists()
            or (first and i in reroll)
            or i in stuck
            or rendered.get(i, {"steps": 16, "effect": "broadcast", "instruct": ""}) != settings[i]
        ]
        return segments, ids, settings, regen

    def request(segments, ids, regen):
        body = {
            "segments": segments,
            "segment_ids": ids,
            "language": NAMES[tgt],
            "language_code": tgt,
            "timing_strategy": args.timing,
            "voice_match": args.voice_match,
            "num_step": args.steps,
        }
        if args.timing == "concise":
            body["overflow_budget_s"] = args.overflow
        if regen is not None:
            body["regen_only"] = regen
        return body

    def level(segments, ids):
        """Level each line against the actor, from the takes on disk."""
        for seg, gain in zip(segments, match_gains(job_id, script["lines"], ids, tgt)):
            if gain is not None:
                seg["gain"] = gain

    def run(body):
        r = requests.post(f"{API}/dub/generate/{job_id}", json=body, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"generate {r.status_code}: {r.text}")
        return wait_task(r.json()["task_id"], {"done"})

    prev_engine = use_tts_engine(args.engine)
    done, synthesized = None, False
    try:
        for n in range(1 + (FILL_PASSES if mode["fill"] else 0)):
            segments, ids, settings, regen = plan(first=n == 0)
            if n and not regen:
                break  # every take ends on the actor, or is as fast or slow as fill allows
            if args.match_level and script:
                level(segments, ids)
            print(
                f"{'generate' if n == 0 else f'  fill pass {n}:'} {args.name}: {len(segments)} segments "
                f"({len(regen) if regen is not None else len(segments)} to synthesize), "
                f"{args.timing}, {args.voice_match}, {args.steps} steps, {args.effect}, {args.engine}"
                + (
                    f", emotion from the original at {args.emo_alpha}"
                    if args.engine != "omnivoice" and args.emo_alpha
                    else ""
                )
                + (
                    f", {sum('ref_audio' in s for s in segments)} lines with their own reference"
                    if mode["line_ref"]
                    else ""
                )
                + (f", {sum('speed' in s for s in segments)} lines re-timed to the actor" if mode["fill"] else "")
            )
            t0 = time.time()
            try:
                done = run(request(segments, ids, regen))
            except RuntimeError as e:
                if "dub_timing_overflow" not in str(e) or not script:
                    raise
                done, stopped = None, re.search(r'"segment": (\d+)', str(e))
            finally:
                # Record what each clip was made with, even when assembly fails after
                # synthesis (e.g. dub_timing_overflow), so a retry doesn't redo them.
                for i, s in settings.items():
                    wav = take_wav(job_id, i, tgt)
                    if wav.exists() and (wav.stat().st_mtime >= t0 - 1 or (regen is not None and i not in regen)):
                        rendered[i] = s
                if settings:
                    rendered_path.write_text(json.dumps(rendered, indent=1), encoding="utf-8")
            print(f"    ({time.time() - t0:.0f} s)")
            synthesized |= regen is None or bool(regen)
            if done is None:
                if not mode["fill"] or n == FILL_PASSES:
                    break
                print(
                    "  a line overflows its room; the next fill pass re-times it, or re-rolls it at the fastest speed"
                )
        if done is None:
            # VoiceStudio stops at the first overflow; report them all so one
            # pass of edits fixes every line.
            print(f"\nVoiceStudio stopped at L{int(stopped.group(1)):02d}" if stopped else "")
            print("lines that don't fit (clip / window / room with borrowed gap and overflow budget):")
            for row in fit_rows(args.name, args.overflow):
                if row["flag"] in ("OVER", "tight"):
                    print(
                        f"  L{row['i']:02d} {row['start']:7.2f}  {row['speech']:4.2f}s / {row['window']:4.2f}s "
                        f"/ {row['room']:4.2f}s  {row['flag']:5s} {row['text']}"
                    )
            raise SystemExit("shorten the text or widen the window of each OVER line, then generate again")
        # "american": now that the takes' timing is settled, give the new ones the
        # actor's timbre. VoiceStudio re-mixes the files it finds (the job has rendered).
        voiced = bool(script) and american and convert(ids)
        if script and (voiced or (args.match_level and synthesized)):
            # Levels are measured on the takes: level the new ones and re-mix (no synthesis).
            print(
                "re-mix: "
                + ", ".join(
                    ["converted takes"] * voiced
                    + ["levels for newly synthesized lines"] * bool(args.match_level and synthesized)
                )
            )
            t0 = time.time()
            segments, ids, _, _ = plan(first=False)
            if args.match_level:
                level(segments, ids)
            done = run(request(segments, ids, []))
            print(f"    ({time.time() - t0:.0f} s)")
    finally:
        if prev_engine and prev_engine != args.engine:
            use_tts_engine(prev_engine)
    (work / "generate-result.json").write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    if script and queued:
        (work / "reroll.json").unlink(missing_ok=True)

    out = out_dir(doc)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f".{args.tag}" if args.tag else ""
    print("export mp4")
    t0 = time.time()
    r = requests.get(
        f"{API}/dub/download/{job_id}",
        params={"include_tracks": tgt, "default_track": tgt, "preserve_bg": "true"},
        timeout=1800,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"download {r.status_code}: {r.text[:500]}")
    mp4 = out / f"{args.name}.{tgt}{suffix}.mp4"
    mp4.write_bytes(r.content)
    r = requests.get(f"{API}/dub/srt/{job_id}", params={"lang": tgt}, timeout=120)
    r.raise_for_status()
    (out / f"{args.name}.{tgt}{suffix}.srt").write_text(r.text, encoding="utf-8")
    print(f"  wrote {mp4} ({time.time() - t0:.0f} s)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("video")
    a.add_argument("--speakers", type=int)
    a.add_argument("--source", default="ja", help="language the video speaks (default ja)")
    a.add_argument("--target", default="en", help="language to dub it into (default en)")
    a.add_argument(
        "--subtitles-only",
        action="store_true",
        help="prepare the part to be subtitled, not dubbed: the audio alone, without the backend",
    )
    a.add_argument(
        "--segment-refs",
        action="store_true",
        help="have VoiceStudio cut and transcribe a reference per segment, for undirected renders "
        "(a script's lines bring their own: core.line_refs)",
    )
    a.set_defaults(fn=cmd_prep)
    b = sub.add_parser("generate")
    b.add_argument("name")
    b.add_argument("--timing", default="concise", choices=["concise", "stretch_video", "strict_slot", "smart_fit"])
    b.add_argument("--voice-match", default="consistent", choices=["per_line", "consistent"])
    b.add_argument(
        "--steps",
        type=int,
        default=32,
        help="OmniVoice diffusion steps (32 = its quality preset; VoiceStudio's default is 16)",
    )
    b.add_argument(
        "--effect",
        default="broadcast",
        choices=EFFECTS,
        help="VoiceStudio voice processing per line; 'raw' skips compression/EQ/normalizing",
    )
    b.add_argument("--tag", default="")
    b.add_argument("--full", action="store_true", help="re-synthesize every line")
    b.add_argument(
        "--match-level", action="store_true", help="set each line's gain to the original actor's speech level"
    )
    b.add_argument("--overflow", type=float, default=0.3, help="seconds a concise-mode line may run past its window")
    b.add_argument(
        "--engine",
        default=None,
        help="VoiceStudio TTS engine for this render (default: omnivoice, or indextts2 for the "
        "american voice); the previous one is restored after",
    )
    b.add_argument(
        "--emo-alpha",
        type=float,
        default=None,
        help="engines with emotion control: follow each line's original delivery at this "
        "strength (0-1; a line's own emo_alpha overrides; 0 = off; default 0.9 for the "
        "american voice, else 0)",
    )
    b.add_argument(
        "--voice",
        choices=VOICES,
        default=None,
        help="override the script's render voice for this run: 'actor' clones the actor (their "
        "accent too); 'american' speaks native English (IndexTTS) and converts each take to "
        "the actor's timbre (vc.py; English dubs only)",
    )
    b.add_argument(
        "--line-ref",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the script's render mode for this run: give each line a reference of its "
        "voice's clips plus its own original, so the take follows the actor's delivery on that "
        'line ("line_ref": false opts a line out)',
    )
    b.add_argument(
        "--fill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"override the script's render mode for this run: re-render takes that end more than "
        f"{FILL_TOL} s from the actor's last sound, at the speed ({FILL_SPEED[0]}-{FILL_SPEED[1]}) "
        f"that lands them on it",
    )
    b.add_argument(
        "--predict-speed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override the script's render mode for this run: with fill, render new lines at the "
        "speed their predicted length asks for, instead of at natural pace first",
    )
    b.set_defaults(fn=cmd_generate)
    args = p.parse_args()
    if args.fn is cmd_prep and args.subtitles_only:
        return cmd_prep(args)  # ffmpeg only: no model, so no GPU lock
    hold_gpu()  # both ask the backend to run models: transcription, or synthesis
    if args.fn is not cmd_generate:
        return args.fn(args)
    flush_gpu()
    try:
        args.fn(args)
    finally:
        flush_gpu()


if __name__ == "__main__":
    main()
