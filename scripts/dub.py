"""Drive VoiceStudio's dubbing backend (http://127.0.0.1:3900) headlessly.

<part>/ is a part's work folder (core.part_dir): work/jobs/<slug>/parts/NN for a
job's part <slug>-NN, work/<name> for a clip prepped by hand. Its VoiceStudio job
folder is named for it (core.new_vs_job_id).

prep <video> [--speakers N] [--source ja --target en]
                              upload (VoiceStudio extracts the audio, without
                              separating it), separate the voices with
                              separate.py (Roformer), attach those stems to the
                              job, transcribe + diarize; writes
                              <part>/segments.json (with the language pair,
                              Japanese -> English by default) and prints the
                              transcript
fork <part>.us                another voice of a prepped part: a VoiceStudio job
                              over the same media (hard-linked, nothing uploaded
                              or separated again), with the part's transcript
                              and pair. prep of a <part>.us video does the same
prep <video> --subtitles-only [--source ja --target en]
                              for subtitles alone: no backend, no voices. Writes
                              the audio into a job folder of VoiceStudio's shape
                              (<part>-local/audio_hq.wav, no transcript) for
                              separate.py to separate and draft.py to transcribe
generate <name>               reads <part>/script.json (hand-directed lines:
                              start/end/voice/text, optional instruct/effect;
                              voices from the job's voices.json) or else <target>.json
                              ({segment_id: text}), synthesizes cloned speech in
                              the part's target language, exports MP4 + SRT. By
                              default (the script's "render" block) each line's
                              reference ends with the actor's own original of it,
                              and takes are re-timed to end where the actor stops.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import secrets
import shutil
import subprocess
import time
import uuid

import numpy as np
import requests
import soundfile as sf

from core import (
    bulk_of,
    HERE,
    HF_CACHE,
    JOBS,
    ROOT,
    active_job,
    base_part,
    langs,
    load_json,
    new_vs_job_id,
    out_dir,
    part_dir,
    profiles_file,
    report_result,
    save_json,
    srt_time,
    take_wav,
    update_json,
)
import gpu
from gpulock import hold_gpu
from languages import LANGUAGES, NAMES, PAIR, PAIRS
from lengths import (
    FILL_SPEED,
    FILL_TOL,
    SPEED_EXPONENT,
    actor_spans,
    fit_rows,
    indextts_length,
    speech_length,
    take_length,
)
from references import fallback_voice, line_refs, reference_settings, reference_signature
from voicecast import line_id, load_voices, voice_home, voice_sig, voice_wav

API = os.environ.get("VOICESTUDIO_API", "http://127.0.0.1:3900")
# How VoiceStudio's transcription says it found no speech (its error event has no code for it).
NO_SPEECH = "produced no segments"


class BackendTaskError(RuntimeError):
    """An explicit terminal backend event, unlike a lost HTTP/SSE connection."""

    def __init__(self, task_id, event):
        self.task_id, self.event = task_id, event
        super().__init__(f"task {task_id} {event['type']}: {json.dumps(event, ensure_ascii=False)}")


class UnknownTaskOutcome(RuntimeError):
    pass


def outcome_save(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Windows scanners can briefly deny renaming an otherwise closed file.
    # Retry only the same durable checkpoint, never a backend request.
    for pause in (0.02, 0.05, 0.1, None):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if pause is None:
                raise
            time.sleep(pause)


def outcome_read(work, path, attempt=None):
    """Only a complete child command can authorize a parent's cache mutation."""
    try:
        doc = load_json(pathlib.Path(path))
        valid = (
            doc["version"] == 1
            and doc["work"] == str(pathlib.Path(work).resolve())
            and (attempt is None or doc["attempt"] == attempt)
            and isinstance(doc["attempt"], str)
            and bool(doc["attempt"])
            and doc["phase"] in {"launching", "running", "unknown", "finished"}
            and isinstance(doc["requests"], list)
            and all(isinstance(r, dict) for r in doc["requests"])
        )
        if not valid:
            raise ValueError("outcome identity mismatch")
        return doc
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise UnknownTaskOutcome(f"Unconfirmed generation outcome: {path}") from error


def outcome_terminal(doc):
    return doc.get("phase") == "finished" and all(
        r.get("status") in {"done", "error", "cancelled"}
        and isinstance(r.get("task_id"), str)
        and bool(r["task_id"])
        and isinstance(r.get("event"), dict)
        and r["event"].get("type") == r["status"]
        for r in doc.get("requests", [])
    )


def require_settled_generation(work):
    pointer = pathlib.Path(work) / "shorttakes" / "active-generation.json"
    if not pointer.exists():
        return
    try:
        active = load_json(pointer)
        doc = outcome_read(work, pathlib.Path(active["path"]), active["attempt"])
        if outcome_terminal(doc):
            return
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise UnknownTaskOutcome(f"Unconfirmed generation blocks this job: {pointer}") from error
    raise UnknownTaskOutcome(f"Unconfirmed generation blocks this job: {active['path']}")


class GenerationOutcome:
    """One selected-only child receipt; no backend retries, polling or cancellation."""

    @staticmethod
    def prepare(work, path=None, baseline=None):
        work = pathlib.Path(work).resolve()
        require_settled_generation(work)
        attempt = uuid.uuid4().hex
        path = pathlib.Path(path or work / "shorttakes" / "generation" / attempt / "outcome.json").resolve()
        if path.exists():
            raise UnknownTaskOutcome(f"Keep existing generation evidence: {path}")
        doc = dict(version=1, work=str(work), attempt=attempt, phase="launching", requests=[], baseline=baseline)
        outcome_save(path, doc)
        outcome_save(work / "shorttakes" / "active-generation.json", dict(path=str(path), attempt=attempt))
        return dict(path=str(path), attempt=attempt)

    def __init__(self, work, ticket):
        self.work, self.path = pathlib.Path(work).resolve(), pathlib.Path(ticket["path"]).resolve()
        active = load_json(self.work / "shorttakes" / "active-generation.json", {})
        if active != dict(path=str(self.path), attempt=ticket["attempt"]):
            raise UnknownTaskOutcome("Generation launch no longer owns the active receipt")
        self.doc = outcome_read(self.work, self.path, ticket["attempt"])
        if self.doc["phase"] != "launching":
            raise UnknownTaskOutcome(f"Do not relaunch an existing child: {self.path}")
        self.doc.update(phase="running", pid=os.getpid())
        self.save()

    def save(self):
        outcome_save(self.path, self.doc)

    def unknown(self):
        return any(r["status"] == "unknown" for r in self.doc["requests"])

    def preserve(self, paths):
        if "before" in self.doc:
            return
        rows = []
        for source in dict.fromkeys(pathlib.Path(p) for p in paths):
            if not source.exists():
                continue
            data = source.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            target = self.path.parent / "before" / f"{len(rows):04}-{source.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest or source.read_bytes() != data:
                raise RuntimeError("Before-state changed during preservation")
            rows.append(dict(source=str(source.resolve()), path=str(target.resolve()), sha256=digest))
        self.doc["before"] = rows
        self.save()

    def start_request(self, body):
        if self.unknown():
            raise UnknownTaskOutcome("Do not submit after an unconfirmed task")
        payload = json.dumps(body, sort_keys=True, ensure_ascii=False)
        request = dict(
            status="unknown",
            task_id=None,
            body=json.loads(payload),
            body_sha256=hashlib.sha256(payload.encode()).hexdigest(),
        )
        self.doc["requests"].append(request)
        self.save()  # Must be durable before the POST, including a lost POST response.
        return request

    def task(self, request, task_id):
        if not isinstance(task_id, str) or not task_id:
            raise UnknownTaskOutcome("POST returned no usable task ID")
        request["task_id"] = task_id
        self.save()

    def terminal(self, request, event):
        if not request["task_id"] or event.get("type") not in {"done", "error", "cancelled"}:
            raise UnknownTaskOutcome("Missing explicit terminal task evidence")
        request.update(status=event["type"], event=event)
        self.save()

    def finish(self, error=None):
        self.doc.update(
            phase="unknown" if self.unknown() else "finished",
            success=error is None,
            error=repr(error) if error is not None else None,
        )
        self.save()


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
                raise BackendTaskError(task_id, d)
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
    emo = bulk_of(work) / "emo"
    emo.mkdir(parents=True, exist_ok=True)
    refs = {}
    for line, i in zip(lines, ids):
        a = line.get("emo_alpha", alpha)
        if not a:
            continue
        path = emo / f"{i}_{line['start']:.2f}-{line['end']:.2f}.wav"
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
    Its job folder (<part>-local, beside VoiceStudio's own) holds the audio as
    VoiceStudio would (audio_hq.wav); separate.py adds the stems and draft.py
    transcribes, and every tool that reads a part's vocals finds them there."""
    for code in (src, tgt):
        if code not in LANGUAGES:
            raise SystemExit(f"no languages/{code}.py: the scripts know {', '.join(sorted(LANGUAGES))}")
    job_id = re.sub(r"[^A-Za-z0-9_-]", "-", video.stem)[:56] + "-local"
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
    save_json(work / "segments.json", doc, indent=2)


def cmd_prep(args):
    video = pathlib.Path(args.video)
    work = part_dir(video.stem)
    work.mkdir(parents=True, exist_ok=True)
    if args.subtitles_only:
        return prep_subtitles(video, work, args.source, args.target)
    if base_part(video.stem) != video.stem:
        return fork(video.stem)
    if (args.source, args.target) not in PAIRS:
        raise SystemExit(f"{args.source} -> {args.target}: the scripts know {sorted(PAIRS)}")
    require_shared_media()

    print(f"upload {video.name} ({NAMES[args.source]} -> {NAMES[args.target]})")
    with open(video, "rb") as f:
        r = requests.post(
            f"{API}/dub/upload",
            files={"video": (video.name, f, "video/mp4")},
            # Named for the part, so VoiceStudio's folder says whose it is. Not separated
            # there: separate.py's Roformer stems replace htdemucs's, which only cost GPU time.
            data={
                "source_lang": args.source,
                "input_type": "video",
                "job_id": new_vs_job_id(video.stem),
                "separate": "false",
            },
            timeout=300,
        )
    r.raise_for_status()
    job = r.json()
    job_id = job["job_id"]
    print(f"  job {job_id}")
    wait_task(job["task_id"], {"ready"})
    separate(video.stem, job_id)

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
                if isinstance(d, dict) and not d.get("code") and NO_SPEECH in str(d.get("detail", "")):
                    report_result(no_speech=True)  # the runner leaves the part in the original
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
    save_json(work / "segments.json", doc, indent=2)

    with open(work / f"{args.source}.srt", "w", encoding="utf-8") as f:
        for i, s in enumerate(final["segments"], 1):
            f.write(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n[{s.get('speaker_id')}] {s['text']}\n\n")
    for s in final["segments"]:
        print(
            f"{s['id']}  {s['start']:6.2f}-{s['end']:6.2f}  {s.get('speaker_id')}  "
            f"{s.get('profile_id') or '-'}  {s['text']}"
        )


def require_shared_media():
    """The pinned VoiceStudio fork uploads without separating, takes a client's stems
    and forks jobs; an older checkout would separate with htdemucs, transcribe that
    and upload every voice again."""
    try:
        paths = requests.get(f"{API}/openapi.json", timeout=30).json().get("paths", {})
    except (requests.RequestException, ValueError):
        return  # can't tell: the requests below fail on their own if it's missing
    if "/dub/stems/{job_id}" not in paths or "/dub/fork/{job_id}" not in paths:
        raise SystemExit(
            "this VoiceStudio can't attach stems or fork jobs: update vendor/VoiceStudio "
            "(python scripts/setup.py --init-only) and restart the backend"
        )


def separate(name, job_id):
    """Separate a part's voices with separate.py (its own environment; this
    process's GPU hold covers it) and make its stems the job's voices and bed,
    which VoiceStudio transcribes, diarizes and mixes over."""
    print("separate the voices (separate.py)")
    try:
        subprocess.run(gpu.separate_command() + [name, "--job", job_id], cwd=ROOT, check=True)
    except OSError:
        raise SystemExit("separating the voices needs uv on PATH (see AGENTS.md)") from None
    r = requests.post(
        f"{API}/dub/stems/{job_id}", json={"vocals": "vocals.wav", "no_vocals": "no_vocals.wav"}, timeout=120
    )
    if not r.ok:
        raise RuntimeError(f"attach stems to {job_id}: {r.status_code} {r.text[:300]}")
    print("  stems attached")


def fork(name):
    """Another voice of a prepped part (<part>.us): a VoiceStudio job over the part's
    media, hard-linked by the backend, so nothing is uploaded, extracted or separated
    again. It takes the part's transcript and pair (the renders send their own lines)."""
    base = base_part(name)
    doc = load_json(part_dir(base) / "segments.json")
    if not doc:
        raise SystemExit(f"{name}: prep {base} first")
    require_shared_media()
    job_id = new_vs_job_id(name)
    r = requests.post(f"{API}/dub/fork/{doc['job_id']}", json={"job_id": job_id}, timeout=300)
    if not r.ok:
        raise RuntimeError(f"fork {doc['job_id']} as {job_id}: {r.status_code} {r.text[:300]}")
    marker = JOBS / doc["job_id"] / "separator.json"
    if marker.exists():  # Dub Anything's own record of which separator made the stems
        shutil.copy2(marker, JOBS / job_id / "separator.json")
    work = part_dir(name)
    work.mkdir(parents=True, exist_ok=True)
    src = langs(base)[0]
    save_json(work / "segments.json", {**doc, "job_id": job_id}, indent=2)
    if (part_dir(base) / f"{src}.srt").exists():
        shutil.copy2(part_dir(base) / f"{src}.srt", work / f"{src}.srt")
    print(
        f"{name}: job {job_id}, forked from {doc['job_id']}; transcript from {base} ({len(doc['segments'])} segments)"
    )


def ensure_voice(key):
    """Resolve a voice key from the job's voices.json to a profile id, building a
    VoiceStudio voice profile from hand-picked clean clips when needed."""
    spec = load_voices()[key]
    if "profile" in spec:
        return spec["profile"]
    sig = voice_sig(key)
    cache_path = profiles_file()
    cache = load_json(cache_path, {})
    old = cache.get(key)
    if old and old["sig"] == sig and requests.get(f"{API}/profiles/{old['id']}", timeout=30).ok:
        return old["id"]
    lender = voice_home(key)
    if lender != cache_path.parent:
        # A borrowed voice: the profile its own job uploaded serves, while it is still there.
        lent = load_json(lender / "profiles.json", {}).get(key)
        if lent and lent.get("sig") == sig and requests.get(f"{API}/profiles/{lent['id']}", timeout=30).ok:
            return lent["id"]

    ref = voice_wav(key, rebuild=True)
    with open(ref, "rb") as f:
        r = requests.post(
            f"{API}/profiles",
            files={"ref_audio": (ref.name, f, "audio/wav")},
            data={"name": f"dub-{active_job() or 'work'}-{key}", "ref_text": spec.get("ref_text", "")},
            timeout=300,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"create profile {key}: {r.status_code} {r.text[:300]}")
    pid = r.json()["id"]
    if old:
        requests.delete(f"{API}/profiles/{old['id']}", timeout=30)
    cache[key] = {"id": pid, "sig": sig}
    save_json(cache_path, cache, indent=2)
    print(f"  voice {key}: profile {pid} ({len(spec['clips'])} clips)")
    return pid


def selected_indices(spec, count):
    """Explicit zero-based line indices; absence keeps the normal full-part path."""
    if spec is None:
        return None
    try:
        selected = {int(value) for value in spec.split(",")}
    except ValueError as error:
        raise ValueError("--only needs comma-separated zero-based line indices") from error
    if not selected or any(i < 0 or i >= count for i in selected):
        raise ValueError("--only line index is outside this script")
    return selected


def seeded_voice(work, key, seed):
    """Private clone profile: the dub API honors profile seeds for OmniVoice too.

    Per-segment seeds are not honored by every engine. Never mutate an existing
    profile to work around that: upload the identical reference with a stored seed.
    Old profiles and their receipts remain available for restoring earlier takes.
    """
    spec = load_voices()[key]
    if "clips" not in spec:
        raise ValueError("seeded candidates require an actor voice with explicit reference clips")
    ref = voice_wav(key)
    text = spec.get("ref_text", "")
    if not text.strip():
        raise ValueError("seeded clone needs an explicit reference transcript")
    identity = {
        "voice": voice_sig(key),
        "audio_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
        "text": text,
        "seed": seed,
    }
    sig = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = work / "shorttakes" / "profiles.json"
    cache = load_json(path, {})
    old = cache.get(sig)
    if old:
        response = requests.get(f"{API}/profiles/{old['id']}", timeout=30)
        if response.ok and response.json().get("seed") == seed and response.json().get("ref_text") == text:
            return old["id"]
    with ref.open("rb") as stream:
        response = requests.post(
            f"{API}/profiles",
            files={"ref_audio": (ref.name, stream, "audio/wav")},
            data={"name": f"dub-candidate-{key}-{seed}", "ref_text": text, "seed": seed},
            timeout=300,
        )
    response.raise_for_status()
    profile = response.json()
    if profile.get("seed") != seed or profile.get("ref_text") != text:
        raise RuntimeError("backend did not persist the requested seed/reference transcript")
    cache[sig] = {**identity, "id": profile["id"], "transport": "clone_profile_seed"}
    save_json(path, cache)
    return profile["id"]


def incremental_baseline(work, job_id, lines, target):
    """Fail closed before a backend can widen selected-only regen to every line."""
    receipt = load_json(work / "generate-result.json", {}).get("client", {})
    if receipt.get("job_id") != job_id or receipt.get("wav_kind") != "natural":
        raise ValueError("--only needs a completed natural-rate baseline from the current dub.py client")
    for line in lines:
        wav = take_wav(job_id, line_id(line), target)
        try:
            info = sf.info(wav)
            audio, _ = sf.read(wav, dtype="float32")
            intact = info.frames > 0 and len(audio) == info.frames and np.isfinite(audio).all()
        except (OSError, RuntimeError):
            intact = False
        if not intact:
            raise ValueError(f"--only cannot repair an incomplete/corrupt baseline: {wav}")


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


def _fill_speed(length, end, previous=None, *, power=1.0, measured=True):
    """Choose the next pace from measured speech or a first-take estimate."""
    # A short prediction can still produce long speech. Publish a natural-rate
    # baseline before slowing it; failed first passes expose no measured takes.
    slowest = FILL_SPEED[0] if measured else 1.0
    return min(max((previous or 1.0) * (length / end) ** power, slowest), FILL_SPEED[1])


def _override_speed(segment, settings, speed):
    """Apply a directed pace instead of fill, preserving natural-rate cache keys."""
    if (
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or not math.isfinite(speed)
        or not FILL_SPEED[0] <= speed <= FILL_SPEED[1]
    ):
        raise ValueError(f"line speed must be finite and between {FILL_SPEED[0]} and {FILL_SPEED[1]}")
    speed = round(speed, 2)
    for spec in (segment, settings):
        if speed == 1.0:
            spec.pop("speed", None)
        else:
            spec["speed"] = speed


def _fitting_published_pace(requested, published, row):
    """Recover a fitting published pace after an automatic adjustment fails.

    A rejected staged take leaves the old WAV and settings intact. That old
    length cannot tell us how long the rejected adjustment actually was.
    """
    if not published or {k: v for k, v in requested.items() if k != "speed"} != {
        k: v for k, v in published.items() if k != "speed"
    }:
        return None
    old, new = published.get("speed", 1.0), requested.get("speed", 1.0)
    if (
        any(
            isinstance(s, bool)
            or not isinstance(s, (int, float))
            or not math.isfinite(s)
            or not FILL_SPEED[0] <= s <= FILL_SPEED[1]
            for s in (old, new)
        )
        or old == new
    ):
        return None
    speech, room = row.get("speech"), row.get("room")
    if speech is None or room is None or not all(math.isfinite(x) for x in (speech, room)):
        return None
    return old if 0 < speech <= room - 0.02 else None


def _published_remix_paces(ids, settings, rendered, rows, directed=()):
    """A complete fitting publication usable after automatic fill is exhausted.

    Only pace may differ from the requested settings. A new reference, quality
    setting or directed pace still needs synthesis, and every take must fit.
    """
    by_index = {row["i"]: row for row in rows}
    paces = {}
    for index, lid in enumerate(ids):
        requested, published = settings.get(lid), rendered.get(lid)
        row = by_index.get(index, {})
        if (
            not requested
            or not published
            or {k: v for k, v in requested.items() if k != "speed"}
            != {k: v for k, v in published.items() if k != "speed"}
        ):
            return None
        old, new = published.get("speed", 1.0), requested.get("speed", 1.0)
        if any(
            isinstance(s, bool)
            or not isinstance(s, (int, float))
            or not math.isfinite(s)
            or not FILL_SPEED[0] <= s <= FILL_SPEED[1]
            for s in (old, new)
        ) or (lid in directed and old != new):
            return None
        speech, room = row.get("speech"), row.get("room")
        if (
            speech is None
            or room is None
            or not all(math.isfinite(x) for x in (speech, room))
            or not 0 < speech <= room - 0.02
        ):
            return None
        paces[lid] = old
    return paces or None


def _record_published_settings(rendered, settings, job, target, started, regen, *, published, selected=None):
    """Associate settings with takes the backend successfully published."""
    if not published:
        return
    for lid, spec in settings.items():
        if selected is not None and lid not in selected:
            continue
        wav = take_wav(job, lid, target)
        if wav.exists() and (wav.stat().st_mtime >= started - 1 or (regen is not None and lid not in regen)):
            rendered[lid] = spec


def file_sha(path):
    return hashlib.sha1(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def render_mode(work, script, args, pair=PAIR):
    """How a part is rendered: script.json's "render" block, {"line_ref": .., "fill": ..,
    "dips": .., "cast_lines": .., "predict_speed": .., "voice": "actor" | "american"}. A
    part without one gets the defaults (per-line references, fill, references cut at
    a dip when speech runs on past a window, and the window check's voice test against
    the characters' cast lines too: references.unclean_windows) if it has never been
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
        # Into the file as it is now: the director may have saved it again since it was read.
        update_json(work / "script.json", lambda current: current and {**current, "render": mode})
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
    work = part_dir(args.name)
    path = getattr(args, "task_outcome", None)
    args._generation = None
    if getattr(args, "only", None) is None:
        require_settled_generation(work)
        if path is not None:
            raise ValueError("--task-outcome requires --only")
        return _cmd_generate(args)
    if path is None:
        ticket = GenerationOutcome.prepare(work)
    else:
        active = load_json(work / "shorttakes" / "active-generation.json", {})
        if active.get("path") != str(pathlib.Path(path).resolve()):
            raise UnknownTaskOutcome("Selected child does not own the prepared outcome")
        ticket = active
    generation = args._generation = GenerationOutcome(work, ticket)
    error = None
    try:
        return _cmd_generate(args)
    except BaseException as exception:
        error = exception
        if generation.unknown():
            raise UnknownTaskOutcome(
                f"Backend task outcome unknown; retain evidence and reconcile: {generation.path}"
            ) from exception
        raise
    finally:
        generation.finish(error)


def _cmd_generate(args):
    render = PartRender(args)
    done = render.render()
    render.save_receipt(done)
    if not getattr(args, "no_export", False):
        render.export()


class PartRender:
    """One `generate` of a part. Its passes (the first render, fill's re-timing, the
    re-mix for levels and converted takes) share the part's script and render mode
    and the records of what each cached take was made with. plan() says what a pass
    sends, run() sends it, and render() takes the passes in order."""

    def __init__(self, args):
        self.args = args
        self.work = work = part_dir(args.name)
        self.doc = json.loads((work / "segments.json").read_text(encoding="utf-8"))
        if self.doc.get("subtitles_only"):
            raise SystemExit(f"{args.name} was prepped for subtitles only: prep a dub job to render voices")
        self.job_id = job_id = self.doc["job_id"]
        self.tgt = tgt = langs(args.name)[1]
        self.script = script = load_json(work / "script.json")
        self.lines = script["lines"] if script else []
        self.only = selected_indices(getattr(args, "only", None), len(self.lines))
        self.explicit_seed = getattr(args, "seed", None)
        if self.explicit_seed is not None and (self.only is None or not 0 <= self.explicit_seed < 2**31):
            raise ValueError("--seed requires --only and a seed from 0 through 2**31-1")
        if self.only is not None:
            self._require_baseline()
        # The takes a selected-only render leaves alone: run() puts back any the backend changes.
        self.protected = {
            line_id(line): (take_wav(job_id, line_id(line), tgt), take_wav(job_id, line_id(line), tgt).read_bytes())
            for index, line in enumerate(self.lines)
            if self.only is not None and index not in self.only
        }
        self.mode = mode = (
            render_mode(work, script, args, langs(args.name))
            if script
            else {"line_ref": False, "fill": False, "predict_speed": False, "voice": "actor"}
        )
        self.american = mode["voice"] == "american"
        if args.engine is None:
            args.engine = "indextts2" if self.american else "omnivoice"
        if self.explicit_seed is not None and (args.engine != "omnivoice" or self.american):
            raise ValueError("--seed currently supports actor/OmniVoice only")
        if args.emo_alpha is None:
            args.emo_alpha = 0.9 if self.american and args.engine != "omnivoice" else 0.0
        # Takes whose seg_<target> file holds a voice-converted copy (id -> its hash); the
        # engine's own take is kept beside it as src_<target>_<id>.wav.
        self.converted_path = work / "converted.json"
        self.converted = load_json(self.converted_path, {})
        self.rendered_path = work / "rendered.json"  # synthesis settings each cached line was made with
        self.rendered = load_json(self.rendered_path, {})
        # Takes lineqa.py --reroll flagged: synthesized again (on the first pass) for a fresh roll,
        # or, when only the voice conversion garbled them, converted again.
        self.queued = load_json(work / "reroll.json", []) if script else []
        self.reroll = {r["id"] for r in self.queued if not r.get("reconvert") and not r.get("settled")}
        self.forced_regeneration = args.full or self.only is not None or bool(self.reroll)
        self.reconvert = {r["id"] for r in self.queued if r.get("reconvert")}
        self.fitting_paces = {}  # rejected automatic adjustments can reuse a validated published pace
        self.spans = actor_spans(job_id, self.lines) if mode["fill"] else None
        self.ends = [span and span[1] for span in self.spans] if self.spans else None
        for line in self.lines:
            if re.search(r"(\.\.\.|…)\s*$", line["text"]):
                print(f"  warning {line['start']}: ends in '…'; the last word tends to get swallowed")

    def _require_baseline(self):
        """--only re-renders chosen lines of a part that is rendered, whole and intact."""
        args, work, job_id, tgt = self.args, self.work, self.job_id, self.tgt
        if args.full:
            raise ValueError("--only cannot be combined with --full")
        if not self.script or not load_json(work / "rendered.json"):
            raise ValueError("--only requires an already rendered full-part baseline")
        incremental_baseline(work, job_id, self.lines, tgt)
        missing = [i for i, line in enumerate(self.lines) if not take_wav(job_id, line_id(line), tgt).exists()]
        if missing:
            raise ValueError(f"--only cannot synthesize an incomplete baseline; missing lines: {missing}")
        ids = [line_id(line) for line in self.lines]
        if any(ids.count(ids[i]) != 1 for i in self.only):
            raise ValueError("--only needs unique line ids; distinguish repeated text with the line's take field")

    # ---- takes on disk ----

    def engine_take(self, i):
        """The engine's own take. A converted copy can start or end a little
        differently, so fill judges timing by the take it was made from."""
        seg, src = take_wav(self.job_id, i, self.tgt), take_wav(self.job_id, i, self.tgt, "src")
        if src.exists() and seg.exists() and self.converted.get(i) == file_sha(seg):
            return src
        return seg

    def convert(self, ids):
        """Give every take not converted yet the actor's timbre (vc.py, its own environment),
        and convert the engine take of each line queued for it again."""
        import vc

        todo = []
        for line, i in zip(self.lines, ids):
            seg, src = take_wav(self.job_id, i, self.tgt), take_wav(self.job_id, i, self.tgt, "src")
            if not seg.exists():
                continue
            if self.converted.get(i) != file_sha(seg):
                shutil.copy2(seg, src)
            elif i not in self.reconvert or not src.exists():
                continue
            todo.append({"in": str(src), "target": str(voice_wav(line["voice"])), "out": str(seg), "id": i})
        if not todo:
            return False
        pairs = self.work / "vc-pairs.json"
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
            self.converted[p["id"]] = file_sha(p["out"])
        save_json(self.converted_path, self.converted)
        return True

    # ---- what a pass sends ----

    def plan(self, first):
        """One pass: the segments to send, their ids, the settings each take is made
        with, and the ids to synthesize (None: all)."""
        if not self.script:
            return self._undirected()
        # Hand-directed lines: own timing, speaker and text per line.
        voices = load_voices()
        segments, ids, settings = self._segments(first)
        refs = self._references(first, segments, ids, settings)
        if self.mode["fill"]:
            self._fill(segments, ids, settings, refs, voices)
        for seg, lid, line in zip(segments, ids, self.lines):
            if lid in self.fitting_paces:
                _override_speed(seg, settings[lid], self.fitting_paces[lid])
            if "speed" in line:
                _override_speed(seg, settings[lid], line["speed"])
        stuck = self._stuck(first, ids, settings)
        if self.args.engine == "indextts2":
            self._seed(first, segments, ids, settings, stuck)
        if self.args.full and first:
            return segments, ids, settings, None
        return segments, ids, settings, self._to_synthesize(first, ids, settings, stuck)

    def _undirected(self):
        """A part without a script: VoiceStudio's own segments, with <target>.json's text."""
        segments, ids = [], []
        dubbed = json.loads((self.work / f"{self.tgt}.json").read_text(encoding="utf-8"))
        for s in self.doc["segments"]:
            text = dubbed.get(s["id"], "").strip()
            if text:
                segments.append(
                    {
                        "start": s["start"],
                        "end": s["end"],
                        "text": text,
                        "profile_id": s.get("profile_id") or "",
                        "target_lang": self.tgt,
                    }
                )
                ids.append(s["id"])
        return segments, ids, {}, None

    def _segments(self, first):
        """Each line as the backend takes it, its id, and the settings its take is made with."""
        args, tgt, only = self.args, self.tgt, self.only
        segments, ids, settings = [], [], {}
        for line_index, line in enumerate(self.lines):
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
            if args.engine == "omnivoice":
                seed = (
                    self.explicit_seed
                    if only is not None and line_index in only
                    else self.rendered.get(ids[-1], {}).get("seed")
                )
                if self.explicit_seed is None and first and ids[-1] in self.reroll and seed is not None:
                    seed = secrets.randbelow(2**31)
                if seed is not None:
                    seg["profile_id"] = seeded_voice(self.work, line["voice"], seed)
                    settings[ids[-1]]["seed"] = seed
                    settings[ids[-1]]["seed_profile"] = seg["profile_id"]
        return segments, ids, settings

    def _references(self, first, segments, ids, settings):
        """Give each line what its take follows: an emotion reference for an engine
        that takes one, its own original as the reference to clone, or the voice it
        falls back to. Returns the per-line references, {id: (path, transcript)}."""
        args, mode = self.args, self.mode
        # Emotion references: each line's own original, for engines that take one.
        emo = emo_refs(self.work, self.job_id, self.lines, ids, args.emo_alpha) if args.engine != "omnivoice" else {}
        for seg, i in zip(segments, ids):
            if i in emo:
                seg["emo_audio"], seg["emo_alpha"] = emo[i]
                settings[i]["emo_alpha"] = emo[i][1]
        # Per-line references: the voice's clips plus the line's own original.
        refs = (
            line_refs(
                self.work,
                self.job_id,
                self.lines,
                ids,
                verbose=first,
                dips=mode["dips"],
                cast_lines=mode["cast_lines"],
                prefix=mode.get("ref_prefix", True),
            )
            if mode["line_ref"]
            else {}
        )
        for seg, i in zip(segments, ids):
            if i in refs:
                seg["ref_audio"], seg["ref_text"] = refs[i]
                # The window, optional cut, and fingerprint of the voice audio.
                settings[i]["line_ref"] = pathlib.Path(refs[i][0]).stem.split("_", 1)[1]
        if not self.american:
            for seg, i, line in zip(segments, ids, self.lines):
                settings[i].update(reference_settings(line, mode, refs.get(i)))
                if i not in refs:
                    selected = fallback_voice(line, mode)
                    if selected != line["voice"]:
                        seed = settings[i].get("seed")
                        seg["profile_id"] = (
                            seeded_voice(self.work, selected, seed) if seed is not None else ensure_voice(selected)
                        )
                        if seed is not None:
                            settings[i]["seed_profile"] = seg["profile_id"]
        return refs

    def _fill(self, segments, ids, settings, refs, voices):
        """Re-time takes that stop well short of the actor or run past them:
        render them at the speed that lands their end on the actor's last sound.
        With predict_speed, a line with no take yet starts at the speed its
        predicted length (take_length, or indextts_length) asks for, and every
        correction follows at_speed rather than assuming length changes in
        proportion to speed. IndexTTS's duration factor is exact: it does."""
        mode, american, rendered = self.mode, self.american, self.rendered
        power = 1 / SPEED_EXPONENT if mode["predict_speed"] and not american else 1.0
        for seg, i, end, line, span in zip(segments, ids, self.ends, self.lines, self.spans):
            prev = rendered.get(i, {}).get("speed")
            speed, wav = prev, self.engine_take(i)
            compatible = reference_signature(rendered.get(i, {})) == reference_signature(settings[i])
            length = speech_length(wav) if wav.exists() and compatible else None
            measured = length is not None
            if not compatible:
                prev, speed = None, None
            if length is None and end and mode["predict_speed"] and american:
                length, prev, speed = indextts_length(line["text"], span[1] - span[0]), None, None
            elif length is None and end and mode["predict_speed"] and (i in refs or "clips" in voices[line["voice"]]):
                selected = fallback_voice(line, mode)
                ref = refs.get(i) or (voice_wav(selected), voices[selected].get("ref_text", ""))
                # The prediction is at natural pace, whatever speed a lost take had.
                length, prev, speed = take_length(*ref, line["text"], lang=self.tgt), None, None
            if end and length and abs(length - end) > FILL_TOL:
                speed = _fill_speed(length, end, prev, power=power, measured=measured)
            if speed and abs(speed - 1) >= 0.01:
                seg["speed"] = settings[i]["speed"] = round(speed, 2)

    def _stuck(self, first, ids, settings):
        """A take that overflows its room at fill's fastest speed can't be re-timed:
        give it a fresh take instead (takes of one line differ by a few tenths)."""
        if not self.mode["fill"] or first:
            return set()
        return {
            ids[row["i"]]
            for row in fit_rows(self.args.name, self.args.overflow)
            if row["flag"] == "OVER"
            and "speed" not in self.lines[row["i"]]
            and self.rendered.get(ids[row["i"]], {}).get("speed")
            == settings[ids[row["i"]]].get("speed")
            == FILL_SPEED[1]
        }

    def _seed(self, first, segments, ids, settings, stuck):
        """Seeded takes (IndexTTS): fill re-times a line without changing its performance
        (its duration factor stretches the same take); only a re-roll draws a new one."""
        for seg, i in zip(segments, ids):
            seed = self.rendered.get(i, {}).get("seed")
            if (first and i in self.reroll) or i in stuck:
                seed = secrets.randbelow(2**31)
            elif seed is None:
                seed = int(i[1:], 16) % 2**31
            seg["seed"] = settings[i]["seed"] = seed

    def _to_synthesize(self, first, ids, settings, stuck):
        """Missing clips, clips made with other settings, or queued re-rolls;
        an empty list re-mixes cached speech without synthesizing anything."""
        regen = [
            i
            for i in ids
            if not take_wav(self.job_id, i, self.tgt).exists()
            or (first and i in self.reroll)
            or i in stuck
            or self.rendered.get(i, {"steps": 16, "effect": "broadcast", "instruct": ""}) != settings[i]
        ]
        if self.only is not None:
            permitted = {ids[i] for i in self.only}
            regen = [i for i in regen if i in permitted]
            if first and self.explicit_seed is not None:
                regen = [i for i in ids if i in permitted]
        return regen

    def request(self, segments, ids, regen):
        args = self.args
        body = {
            "segments": segments,
            "segment_ids": ids,
            "language": NAMES[self.tgt],
            "language_code": self.tgt,
            "timing_strategy": args.timing,
            "voice_match": args.voice_match,
            "num_step": args.steps,
        }
        if args.timing == "concise":
            body["overflow_budget_s"] = args.overflow
        if regen is not None:
            body["regen_only"] = regen
        return body

    def level(self, segments, ids):
        """Level each line against the actor, from the takes on disk."""
        for seg, gain in zip(segments, match_gains(self.job_id, self.lines, ids, self.tgt)):
            if gain is not None:
                seg["gain"] = gain

    # ---- sending it ----

    def run(self, body):
        """Send one pass to the backend and wait for it. A selected-only render keeps
        its receipt current around the request and puts back any take it didn't select."""
        args, work, job_id, tgt = self.args, self.work, self.job_id, self.tgt
        started = time.perf_counter()
        generation = args._generation
        request_outcome = None
        if generation is not None:
            generation.preserve(
                [self.rendered_path, work / "script.json", work / "generate-result.json"]
                + [take_wav(job_id, lid, tgt, kind) for lid in body["segment_ids"] for kind in ("seg", "src")]
            )
            request_outcome = generation.start_request(body)
        try:
            r = requests.post(f"{API}/dub/generate/{job_id}", json=body, timeout=120)
            if r.status_code >= 400:
                raise RuntimeError(f"generate {r.status_code}: {r.text}")
            task_id = r.json()["task_id"]
            if generation is not None:
                generation.task(request_outcome, task_id)
            try:
                result = wait_task(task_id, {"done"})
            except BackendTaskError as error:
                if generation is not None:
                    generation.terminal(request_outcome, error.event)
                raise
            if generation is not None:
                generation.terminal(request_outcome, result)
            return result
        finally:
            metrics = getattr(args, "_metrics", None)
            if metrics is not None:
                metrics["backend_tasks"].append(
                    {
                        "seconds": time.perf_counter() - started,
                        "regenerated_ids": body.get("regen_only"),
                        "speeds": {i: s.get("speed", 1.0) for i, s in zip(body["segment_ids"], body["segments"])},
                    }
                )
            changed = []
            safe = request_outcome is None or request_outcome["status"] != "unknown"
            for lid, (wav, baseline) in self.protected.items() if safe else ():
                if not wav.exists() or wav.read_bytes() != baseline:
                    # Preserve unexpected output too, but never leave an unselected
                    # take silently changed by the backend's full-regeneration guard.
                    if wav.exists():
                        folder = work / "shorttakes" / "unexpected"
                        folder.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(wav, folder / f"{lid}-{file_sha(wav)}.wav")
                    temp = wav.with_name(wav.name + ".restore.tmp")
                    temp.write_bytes(baseline)
                    temp.replace(wav)
                    changed.append(lid)
            if changed:
                raise RuntimeError(f"backend changed unselected caches; originals restored: {changed}")

    # ---- the passes, in order ----

    def render(self):
        """Render the part with its engine (the one selected before is restored after):
        the first pass and fill's re-timing, the re-mix that keeps a fitting
        publication, then the American voice's conversion and the levels. Returns
        the backend's result for the mix that stands."""
        args = self.args
        prev_engine = use_tts_engine(args.engine)
        try:
            if args._generation is not None:
                args._generation.doc["previous_engine"] = prev_engine
                args._generation.save()
            done, ids, synthesized, recovered_adjustment, stopped = self._passes()
            if done is None and recovered_adjustment and not self.forced_regeneration:
                done, ids = self._remix_published()
            if done is None:
                self._report_overflow(stopped)
            return self._convert_and_level(done, ids, synthesized)
        finally:
            if (
                prev_engine
                and prev_engine != args.engine
                and (args._generation is None or not args._generation.unknown())
            ):
                use_tts_engine(prev_engine)

    def _passes(self):
        """The first render, then fill's passes while a take still needs re-timing.
        Returns the last result (None: a line doesn't fit), the last plan's ids,
        whether anything was synthesized, whether the last rejected re-timing fell
        back to the pace already published, and where the backend's check stopped."""
        args, mode, job_id, tgt = self.args, self.mode, self.job_id, self.tgt
        done, synthesized, stopped = None, False, None
        for n in range(1 + (FILL_PASSES if mode["fill"] else 0)):
            segments, ids, settings, regen = self.plan(first=n == 0)
            if n and not regen and (done is not None or not self.fitting_paces):
                break  # every take ends on the actor, or is as fast or slow as fill allows
            if args.match_level and self.script:
                self.level(segments, ids)
            self._announce(n, segments, regen)
            t0 = time.time()
            published = False
            recovered_adjustment = False
            try:
                done = self.run(self.request(segments, ids, regen))
                published = True
            except RuntimeError as e:
                if args._generation is not None and args._generation.unknown():
                    raise
                if "dub_timing_overflow" not in str(e) or not self.script:
                    raise
                # Failed attempts publish no takes. Keep the backend's validation
                # detail available when the director repairs the reported line.
                print(f"  timing validation: {e}")
                done, stopped = None, re.search(r'"segment": (\d+)', str(e))
                recovered_adjustment = self._keep_published_pace(e, ids, settings)
            finally:
                # Failed/cancelled attempts leave the last publication intact.
                if args._generation is None or not args._generation.unknown():
                    _record_published_settings(
                        self.rendered,
                        settings,
                        job_id,
                        tgt,
                        t0,
                        regen,
                        published=published,
                        selected={ids[k] for k in self.only} if self.only is not None else None,
                    )
                if settings and (args._generation is None or not args._generation.unknown()):
                    save_json(self.rendered_path, self.rendered)
            print(f"    ({time.time() - t0:.0f} s)")
            synthesized |= regen is None or bool(regen)
            if done is None:
                if not mode["fill"] or n == FILL_PASSES or self.forced_regeneration:
                    break
                if not any(take_wav(job_id, lid, tgt).exists() for lid in ids):
                    print("  no published takes to measure; revise the reported line before retrying")
                    break
                print(
                    "  a line overflows its room; the next fill pass re-times it, or re-rolls it at the fastest speed"
                )
        return done, ids, synthesized, recovered_adjustment, stopped

    def _announce(self, n, segments, regen):
        args, mode = self.args, self.mode
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

    def _keep_published_pace(self, e, ids, settings):
        """The backend rejected an automatic re-timing of one line, so that take was
        not published: when the take already published fits at its own pace, keep
        that pace for the passes to come. Returns whether it did."""
        args = self.args
        if not (
            self.mode["fill"]
            and args.timing == "concise"
            and args.engine == "omnivoice"
            and not self.american
            and not self.forced_regeneration
            and isinstance(e, BackendTaskError)
            and e.event.get("type") == "error"
            and e.event.get("error_code") == "dub_timing_overflow"
        ):
            return False
        index = e.event.get("segment")
        if not (type(index) is int and 0 <= index < len(ids)):
            return False
        lid = ids[index]
        if not (e.event.get("segment_id", lid) == lid and lid not in self.reroll and "speed" not in self.lines[index]):
            return False
        row = next(r for r in fit_rows(args.name, args.overflow) if r["i"] == index)
        pace = _fitting_published_pace(settings[lid], self.rendered.get(lid), row)
        if pace is None:
            return False
        self.fitting_paces[lid] = pace
        print(f"  L{index:02d}: keeping fitting published pace {pace:g}; rejected adjustment was not published")
        return True

    def _remix_published(self):
        """A different rejected slowdown can consume each fill pass. Even the last
        rejection gets one cache-only assembly, provided the entire current
        publication still matches and fits. Require a new successful backend
        result; the failed attempt is never success. Returns that result (None
        when the publication can't stand) and the plan's ids."""
        args = self.args
        segments, ids, settings, _ = self.plan(first=False)
        paces = _published_remix_paces(
            ids,
            settings,
            self.rendered,
            fit_rows(args.name, args.overflow),
            directed={lid for lid, line in zip(ids, self.lines) if "speed" in line},
        )
        if paces is None:
            return None, ids
        self.fitting_paces.update(paces)
        for segment, lid in zip(segments, ids):
            _override_speed(segment, settings[lid], paces[lid])
        if args.match_level:
            self.level(segments, ids)
        print("  fill adjustments exhausted; re-mix the complete fitting publication (no synthesis)")
        t0 = time.time()
        done = self.run(self.request(segments, ids, []))
        _record_published_settings(self.rendered, settings, self.job_id, self.tgt, t0, [], published=True)
        save_json(self.rendered_path, self.rendered)
        return done, ids

    def _report_overflow(self, stopped):
        """VoiceStudio stops at the first overflow; report them all so one
        pass of edits fixes every line."""
        args = self.args
        print(f"\nVoiceStudio stopped at L{int(stopped.group(1)):02d}" if stopped else "")
        print("lines that don't fit (clip / window / room with borrowed gap and overflow budget):")
        rows = [row for row in fit_rows(args.name, args.overflow) if row["flag"] in ("OVER", "tight")]
        for row in rows:
            print(
                f"  L{row['i']:02d} {row['start']:7.2f}  {row['speech']:4.2f}s / {row['window']:4.2f}s "
                f"/ {row['room']:4.2f}s  {row['flag']:5s} {row['text']}"
            )
        # The runner blocks the part on these until its script changes.
        report_result(
            overflow=[
                {k: row[k] for k in ("i", "start", "speech", "window", "room", "text")}
                for row in rows
                if row["flag"] == "OVER"
            ]
        )
        raise SystemExit("shorten the text or widen the window of each OVER line, then generate again")

    def _convert_and_level(self, done, ids, synthesized):
        """The American voice: now that the takes' timing is settled, give the new ones
        the actor's timbre. VoiceStudio re-mixes the files it finds (the job has rendered).
        Levels are measured on the takes: level the new ones and re-mix (no synthesis)."""
        args, script = self.args, self.script
        voiced = bool(script) and self.american and self.convert(ids)
        if script and (voiced or (args.match_level and synthesized)):
            print(
                "re-mix: "
                + ", ".join(
                    ["converted takes"] * voiced
                    + ["levels for newly synthesized lines"] * bool(args.match_level and synthesized)
                )
            )
            t0 = time.time()
            segments, ids, _, _ = self.plan(first=False)
            if args.match_level:
                self.level(segments, ids)
            done = self.run(self.request(segments, ids, []))
            print(f"    ({time.time() - t0:.0f} s)")
        return done

    # ---- after the render ----

    def save_receipt(self, done):
        """Record how the part was rendered (generate-result.json); the re-rolls
        lineqa.py queued have had their fresh takes."""
        args = self.args
        receipt = {
            **done,
            "client": {
                "job_id": self.job_id,
                "timing": args.timing,
                "wav_kind": "slotted" if args.timing == "strict_slot" else "natural",
                "policy": {
                    "timing": args.timing,
                    "overflow": args.overflow,
                    "match_level": args.match_level,
                    "voice_match": args.voice_match,
                    "engine": args.engine,
                    "render_mode": self.mode,
                },
            },
        }
        save_json(self.work / "generate-result.json", receipt, indent=2)
        if self.script and self.queued and self.only is None:
            (self.work / "reroll.json").unlink(missing_ok=True)

    def export(self):
        """Download the mixed part (MP4) and its subtitles (SRT) into the dubbed folder."""
        args, job_id, tgt = self.args, self.job_id, self.tgt
        out = out_dir(self.doc)
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
        "(a script's lines bring their own: references.line_refs)",
    )
    a.set_defaults(fn=cmd_prep)
    f = sub.add_parser("fork")
    f.add_argument("name", help="the voice's copy of a prepped part, <part>.us")
    f.set_defaults(fn=lambda args: fork(args.name))
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
        "--only", help="synthesize only these zero-based comma-separated line indices; needs a full baseline"
    )
    b.add_argument(
        "--seed", type=int, help="opt-in fixed seed for --only actor/OmniVoice lines, via private clone profiles"
    )
    b.add_argument("--no-export", action="store_true", help="keep takes and backend mix but skip MP4/SRT download")
    b.add_argument("--metrics", type=pathlib.Path, help="optional JSON costs: lock wait, held wall time, backend tasks")
    b.add_argument("--task-outcome", type=pathlib.Path, help="prepared selected-only child outcome receipt")
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
    if (args.fn is cmd_prep and args.subtitles_only) or args.cmd == "fork":
        return args.fn(args)  # ffmpeg, or links: no model, so no GPU lock
    started = time.perf_counter()
    hold_gpu()  # both ask the backend to run models: transcription, or synthesis
    if args.fn is not cmd_generate:
        return args.fn(args)
    args._metrics = (
        {
            "gpu_lock_wait_s": time.perf_counter() - started,
            "backend_tasks": [],
            "active_gpu_s": None,
            "scope": "Backend tasks include generation/fill/remix; active GPU compute is not isolated.",
        }
        if args.metrics
        else None
    )
    held = time.perf_counter()
    flush_gpu()
    try:
        args.fn(args)
        if args._metrics is not None:
            args._metrics["completed"] = True
    finally:
        flush_gpu()
        if args._metrics is not None:
            args._metrics["held_wall_s"] = time.perf_counter() - held
            args.metrics.parent.mkdir(parents=True, exist_ok=True)
            args.metrics.write_text(json.dumps(args._metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
