"""A job's cast of voices: the entries in its voices.json (and in the casts it
borrows), each voice's reference audio, the lock directors share when they add
voices at the same time, and the fingerprints that name a take (voice_sig,
line_id). Where a cast's files are is core.py's to say (cast_dir, voices_file).

core.py and the modules beside it (this one, references.py, lengths.py) call one
another through the module (core.load_json, voicecast.voice_sig), so a test
patches a name once, where it lives.
"""

import hashlib
import json
import sys
import time

import core


def load_voices():
    """The cast the active job can use: its own voices.json over those it borrows.
    Only its own is ever written (update_voices)."""
    voices = {}
    for other in core.borrowed_casts():
        voices.update(core.load_json(core.cast_dir(other) / "voices.json", {}))
    voices.update(core.load_json(core.voices_file(), {}))
    return voices


def voice_home(key):
    """The cast folder a voice's entry comes from: the job's own, or a borrowed one."""
    if key in core.load_json(core.voices_file(), {}):
        return core.cast_dir()
    for other in reversed(core.borrowed_casts()):
        if key in core.load_json(core.cast_dir(other) / "voices.json", {}):
            return core.cast_dir(other)
    return core.cast_dir()


def voice_path(key):
    """Where a voice's reference audio is (voice_wav builds it): the job's voices/<key>.wav,
    or the borrowed cast's file, when it has one."""
    home = voice_home(key) / "voices" / f"{key}.wav"
    return home if home.exists() else core.voices_dir() / f"{key}.wav"


def update_voices(change):
    """Apply change(voices) to the active job's voices.json (voices_file) under a lock
    and write it back: directors working on a show in parallel may add voices at the
    same time. A borrowed cast is read, never written."""
    path = core.voices_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a+") as lock:
        while True:
            try:
                lock.seek(0)
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                time.sleep(0.2)
        voices = core.load_json(path, {})
        result = change(voices)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(voices, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        # Closing the file releases the lock.
    return result


def voice_sig(key, voices=None):
    """Fingerprint of what a voice's takes are made from: its entry in voices.json
    (clips, text, source, register) without the remarks (REMARKS). It is part of
    every line's id (line_id), which names the take's file: an entry that hasn't
    changed must keep its fingerprint, or every take of the voice is made again.
    That is why an entry's register fields stay in it, though no take is made from
    them: the tools that build a voice write them once, and leaving them out now
    would rename the takes of every voice that has them. `voices` saves reading
    the cast again when fingerprinting several."""
    voices = load_voices() if voices is None else voices
    spec = {k: v for k, v in voices[key].items() if k not in core.REMARKS}
    return hashlib.sha1(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def voice_clips(spec):
    """A voice's reference clips as (clip, start, end). An entry is [start, end]
    in the voice's "from" clip, or ["clip", start, end] from any prepped clip,
    so a bit part can pool its clean lines from several parts."""
    return [tuple(c) if len(c) == 3 else (spec["from"], *c) for c in spec["clips"]]


def voice_wav(key, rebuild=False):
    """voices/<key>.wav in the job's cast folder: the voice's clips from voices.json
    joined with 0.3 s pauses. Built when missing, when rebuild is set, or when the
    voice's entry no longer matches the profile it was last uploaded as. A voice
    borrowed from another job's cast is that job's file, as it was built there."""
    import numpy as np
    import soundfile as sf

    spec = load_voices()[key]
    ref = voice_path(key)
    if ref.parent != core.voices_dir() and not rebuild:
        return ref
    ref = core.voices_dir() / f"{key}.wav"
    uploaded = core.load_json(core.profiles_file(), {}).get(key, {})
    if ref.exists() and not rebuild and uploaded.get("sig") == voice_sig(key):
        return ref
    sources, parts = {}, []
    for clip, a, b in voice_clips(spec):
        if clip not in sources:
            sources[clip] = sf.read(core.vocals_wav(clip), dtype="float32")
        audio, sr = sources[clip]
        gap = np.zeros((int(0.3 * sr),) + audio.shape[1:], dtype="float32")
        parts += [audio[int(a * sr) : int(b * sr)], gap]
    ref.parent.mkdir(parents=True, exist_ok=True)
    sf.write(ref, np.concatenate(parts[:-1]), sr)
    return ref


def line_id(line):
    """Stable id from voice + text, so retiming a line reuses its cached speech;
    changing the voice's clips in voices.json re-synthesizes its lines. A line
    repeated word for word needs `"take": n` on the repeats to get ids of its own."""
    key = f"{line['voice']}|{voice_sig(line['voice'])}|{line['text']}"
    if line.get("take"):
        key += f"|take {line['take']}"
    return "L" + hashlib.sha1(key.encode()).hexdigest()[:10]
