"""Shared by the Dub Anything scripts: where things live (work/, VoiceStudio's job
folders, the model cache), voices and their reference clips, line ids, takes and
their lengths, each actor's last sound in a window, per-line references and the
check that a window is clean, and the length OmniVoice will give a take
(take_length, fit_rows). dub.py drives VoiceStudio with these; the other tools
import them from here.
"""

import functools
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import unicodedata

from gpu import rocm_env
from languages import LANGUAGES, PAIR, PAIRS, pair_of

HERE = pathlib.Path(__file__).resolve().parent  # the scripts
ROOT = HERE.parent  # the repo: work/, videos/, vendor/
WORK = ROOT / "work"


def default_data_dir():
    """Default location for VoiceStudio / OmniVoice data (same as backend/core/config.py)."""
    if os.environ.get("OMNIVOICE_DATA_DIR"):
        return pathlib.Path(os.environ["OMNIVOICE_DATA_DIR"])
    if sys.platform == "win32":
        return pathlib.Path(os.environ.get("APPDATA", "")) / "OmniVoice"
    if sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Application Support" / "OmniVoice"
    return pathlib.Path.home() / ".omnivoice"


def cache_root():
    """OmniVoice's model folder: $OMNIVOICE_CACHE_DIR, else the platform's default.
    Tools' own models (separate.py's audio-separator/) go in it."""
    if os.environ.get("OMNIVOICE_CACHE_DIR"):
        return pathlib.Path(os.environ["OMNIVOICE_CACHE_DIR"])
    if sys.platform == "win32":
        return pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "OmniVoice"
    if sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Caches" / "OmniVoice"
    return pathlib.Path.home() / ".cache" / "OmniVoice"


def default_cache_dir():
    """The Hugging Face model cache: $HF_HUB_CACHE, else $OMNIVOICE_CACHE_DIR
    itself, else hf_cache in the platform's default. On Linux VoiceStudio leaves
    the cache where Hugging Face puts it, so the scripts look there too."""
    if os.environ.get("HF_HUB_CACHE") or os.environ.get("OMNIVOICE_CACHE_DIR"):
        return os.environ.get("HF_HUB_CACHE") or os.environ["OMNIVOICE_CACHE_DIR"]
    if sys.platform not in ("win32", "darwin"):
        cache = os.environ.get("XDG_CACHE_HOME") or str(pathlib.Path.home() / ".cache")
        return str(pathlib.Path(os.environ.get("HF_HOME") or pathlib.Path(cache) / "huggingface") / "hub")
    return str(cache_root() / "hf_cache")


# Same resolution as VoiceStudio's backend/core/config.py.
JOBS = default_data_dir() / "dub_jobs"
HF_CACHE = default_cache_dir()
# Here, not only in whisper_setup(): huggingface_hub reads it once, on import, and
# faster-whisper imports it (through ctranslate2) before a script calls whisper_setup().
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)
rocm_env()


def patch_av_open():
    """PyAV 19.0.0 removed the metadata_errors argument from av.open(), which
    faster-whisper's decode_audio still passes. Compatibility wrapper."""
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


def whisper_setup():
    """Prepare a standalone faster-whisper: reuse VoiceStudio's downloaded models
    and preload the cuDNN 8 libraries CTranslate2 needs on CUDA (PyTorch ships
    cuDNN 9; the backend does the same in backend/core/cudnn8.py)."""
    patch_av_open()
    os.environ.setdefault("HF_HUB_CACHE", HF_CACHE)
    if sys.platform == "darwin":
        return
    import ctypes

    if sys.platform == "win32":
        lib = pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "cudnn8_compat" / "nvidia" / "cudnn" / "bin"
        if lib.is_dir():
            whisper_setup.cookie = os.add_dll_directory(str(lib))  # must stay alive
            for dll in sorted(lib.glob("cudnn*64_8.dll")):
                ctypes.CDLL(str(dll))
        return
    # Linux: the loader finds a library CTranslate2 asks for by name once it is
    # loaded globally; each one finds its own dependencies beside it ($ORIGIN).
    for lib in pathlib.Path(sys.prefix).glob("lib/python3*/site-packages/cudnn8_compat/nvidia/cudnn/lib"):
        for so in sorted(lib.glob("libcudnn*.so.8")):
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)


def out_dir(doc):
    """Finished dubs go to a 'dubbed' folder beside the source video (or $DUB_OUT)."""
    return pathlib.Path(os.environ.get("DUB_OUT") or pathlib.Path(doc["video"]).parent / "dubbed")


def original_audio_language(lang, source, quicktime_audio=False):
    """Optional QuickTime playback preference workaround; preserve other languages."""
    return "und" if quicktime_audio and source == "en" and lang == "eng" else lang


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def srt_time(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


# The language pair: what the source speaks and what the dub speaks, as ISO 639-1 codes
# (what the scripts know of each language is in languages/). `dub.py prep` records it
# in each part's segments.json. A line keeps its original words in the field named for
# the source language ("ja": "...", or "en" or "es" when dubbing from English or
# Spanish) and the dub's words in "text".
def langs(name):
    """(source, target) of a part, from its segments.json (languages.pair_of), or
    from the part a voice's copy (<part>.us) was made from."""
    if name not in _langs:
        for n in (name, base_part(name)):
            seg = load_json(WORK / n / "segments.json")
            if seg:
                _langs[name] = pair_of(seg)
                break
        else:
            return PAIR  # not prepped (yet): not remembered
    return _langs[name]


_langs = {}


def said(line, src):
    """A line's original words: its field named for the source language."""
    return line.get(src) or ""


def take_wav(job_id, lid, tgt, kind="seg"):
    """A line's take in VoiceStudio's job folder: seg_<target>_<id>.wav (the take
    mixed in), or src_<target>_<id>.wav (the engine's own take, kept beside a
    voice-converted copy)."""
    return JOBS / job_id / f"{kind}_{tgt}_{lid}.wav"


LINE_REF_MAX = 19.5  # s; OmniVoice refuses a reference over 20 s that has a transcript
ROOM_EFFECTS = {"echo", "pa"}  # their original carries the room, which a reference would clone


# How far Whisper's hearing of a window may be from the line's `ja` (character error
# rate). Kanji written as kana counts against it, so only gross mismatches fail.
LINE_REF_CER = 0.5
KANA = {c: c - 0x60 for c in range(0x30A1, 0x30F7)}  # katakana -> hiragana, for str.translate


def bare(text):
    """Words as a window's check compares them with what Whisper hears: whatever the
    case, without spaces and punctuation, katakana as hiragana (Whisper writes a name
    or a word in either, "タロウ", "ナニ") and Latin letters without their accents
    (it writes "esta" or "está")."""
    text = re.sub(r"[\s、。，；：,.!?！？¿¡…・「」『』【】«»“”()（）ー〜~\-—'\"‘’:;]", "", text.lower())
    text = re.sub(r"[\u0300-\u036f]", "", unicodedata.normalize("NFD", text))
    return unicodedata.normalize("NFC", text).translate(KANA)


# Speech running on past a window's end still leaves a clean reference when the
# voice dips to near silence (this far below the line's speech level) within
# DIP_REACH of the end: a turn change, or the actor's tail, too quick for the
# VAD's 50 ms pauses. The line's reference is then cut at the dip.
DIP_DB = 20.0
DIP_REACH = 0.15  # s either side of the window's end


def end_dip(db, start, end):
    """The time of the deepest dip in the voice within DIP_REACH of a window's end
    (5 ms frames of edges.frames_db, smoothed over 20 ms), if it is at least DIP_DB
    below the line's speech level; else None."""
    import numpy as np

    import edges

    frame = edges.HOP / edges.SR
    level = edges.line_level(db, start, end)
    lo = max(int((end - DIP_REACH) / frame), int((start + 0.3) / frame))
    hi = min(int((end + DIP_REACH) / frame), len(db))
    if hi - lo < 4:
        return None
    smooth = np.convolve(db[lo:hi], np.ones(4) / 4, mode="same")
    k = int(np.argmin(smooth))
    return round((lo + k) * frame, 3) if smooth[k] - level <= -DIP_DB else None


def unclean_windows(work, job_id, lines, ids, dips=False, cast_lines=False):
    """Whether each line's window can be its own reference, as (why, cut) per line.
    `why` says why it can't, or is None if it can: speech running on past the
    window's end (the reference would end mid-sound, which OmniVoice finishes at
    the start of the take), Whisper hearing something else than the line's
    original words (crosstalk, another voice), or a voice nearer another of the
    clip's characters. With `dips`, speech running on doesn't count where the voice dips
    to near silence right at the end (end_dip); `cut` is then where the line's
    reference ends, instead of the window's end. With `cast_lines`, a voice is
    only nearer another character when the characters' cast centroids (their clips
    plus lines cast to them in the show's other parts, draft.cast_centroids) say so
    too: a reference alone can miss much of a character's range. Cached in
    refs/clean.json by what the check reads (window, voice and its clips, original
    words, and the clips of every voice it's compared with), not by the take: rewording the
    dub's text keeps it, and rebuilding another of the clip's voices checks its lines
    again. Lines cast in other parts aren't in the key, so a decision stays put
    while the show's other parts are directed. Whisper listens in the part's source
    language, and the line's original words are its field named for it."""
    src = langs(work.name)[0]
    cache_path = work / "refs" / "clean.json"
    cache = load_json(cache_path, {})
    rivals = hashlib.sha1(
        "|".join(f"{v}:{voice_sig(v)}" for v in sorted({line["voice"] for line in lines})).encode()
    ).hexdigest()[:8]
    keys = [
        f"{line['start']:.2f}-{line['end']:.2f}_"
        + hashlib.sha1(f"{line['voice']}|{voice_sig(line['voice'])}|{said(line, src)}".encode()).hexdigest()[:12]
        + f"_{rivals}"
        + ("_dips" if dips else "")
        + ("_cast" if cast_lines else "")
        for line in lines
    ]
    if any(k not in cache for k in keys):
        import draft
        import edges
        from register import cer

        sr = draft.SR
        audio = draft.decode_audio(str(JOBS / job_id / "vocals.wav"), sampling_rate=sr)
        db = edges.frames_db(audio)
        # 50 ms: a turn change in fast dialogue often leaves no longer pause, and a
        # longer minimum would merge the two speakers into one span.
        opts = draft.VadOptions(threshold=0.5, min_silence_duration_ms=50, speech_pad_ms=0)
        spans = [(s["start"] / sr, s["end"] / sr) for s in draft.get_speech_timestamps(audio, opts)]
        from gpulock import hold_gpu

        if draft.whisper_on_gpu():  # else Whisper and the embeddings both run on the CPU
            hold_gpu("window check")
        model = draft.WhisperModel("Systran/faster-whisper-large-v3", device="auto", compute_type="default")
        embed = draft.embedder()
        voices = load_json(WORK / "voices.json", {})
        who = lambda v: voices[v].get("register_of", v)  # noqa: E731
        cast = (
            {
                v: embed(draft.decode_audio(str(WORK / "voices" / f"{v}.wav"), sampling_rate=sr))
                for v in {line["voice"] for line in lines}
            }
            if embed
            else {}
        )
        # Exclude this part's own lines: a line mustn't vouch for itself.
        cents = draft.cast_centroids(embed, {who(v) for v in cast}, exclude=work.name) if (cast_lines and embed) else {}
        for line, key in zip(lines, keys):
            if key in cache:
                continue
            a, b = line["start"], line["end"]
            why, cut = None, None
            if any(s < b - 0.05 and e > b + 0.05 for s, e in spans):
                cut = end_dip(db, a, b) if dips else None
                if cut is None:
                    why = "speech runs on past the window's end"
            x = audio[int(a * sr) : int((cut or b) * sr)]
            if why is None:
                heard = "".join(p["said"] for p in draft.transcribe(model, x, language=src))
                err = cer(bare(said(line, src)), bare(heard))
                if err > LINE_REF_CER:
                    why = f"heard as '{heard}' ({err:.0%} off)"
                elif b - a >= 0.6:
                    e = embed(x)
                    best = max(cast, key=lambda v: float(e @ cast[v]))
                    if who(best) != who(line["voice"]) and not (
                        who(line["voice"]) in cents
                        and max(cents, key=lambda v: float(e @ cents[v])) == who(line["voice"])
                    ):
                        why = f"voice nearer {best}"
            cache[key] = {"cut": cut} if cut and not why else why
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return [(None, cache[k]["cut"]) if isinstance(cache[k], dict) else (cache[k], None) for k in keys]


def line_refs(work, job_id, lines, ids, verbose=True, dips=False, cast_lines=False):
    """Per-line references: the voice's own clips (work/voices/<key>.wav), a 0.3 s
    pause, then this line's original from the job's separated vocals, with the
    line's original words (`ja`, or the field of the part's source language) after
    the voice's transcript. OmniVoice copies its reference's
    delivery, so the take follows the actor on this line while the clips hold the
    voice steady. Lines with a room effect or "line_ref": false keep the voice's
    reference, as do lines whose window isn't clean (unclean_windows; with `dips`,
    a window whose speech runs on into a near-silent dip is clean, and its
    original is cut at the dip) and lines too long to fit OmniVoice's limit with
    the clips. Returns {id: (path, transcript)}; the path records the window,
    cut and voice-audio fingerprint, which dub.py saves with the take. Rebuilding
    a voice must not reuse a cached reference containing its old audio."""
    import numpy as np
    import soundfile as sf

    voices = load_json(WORK / "voices.json", {})
    src = langs(work.name)[0]
    vocals, sr = sf.read(JOBS / job_id / "vocals.wav", dtype="float32")
    (work / "refs").mkdir(exist_ok=True)
    refs = {}
    for line, i, (why, cut) in zip(lines, ids, unclean_windows(work, job_id, lines, ids, dips, cast_lines)):
        if not line.get("line_ref", True) or line.get("effect") in ROOM_EFFECTS:
            continue
        if why:
            if verbose:
                print(f"  line ref skipped at {line['start']}: {why}")
            continue
        base_path = WORK / "voices" / f"{line['voice']}.wav"
        base, base_sr = sf.read(base_path, dtype="float32")
        own = vocals[int(line["start"] * sr) : int((cut or line["end"]) * sr)]
        if base_sr != sr or base.shape[1:] != own.shape[1:]:
            print(f"  line ref skipped at {line['start']}: {base_path.name} isn't in the vocals' format")
            continue
        if (len(base) + len(own)) / sr + 0.3 > LINE_REF_MAX:
            if verbose:
                print(f"  line ref skipped at {line['start']}: over {LINE_REF_MAX:.0f} s with the voice's clips")
            continue
        base_sig = hashlib.sha1(base.tobytes()).hexdigest()[:12]
        path = (
            work / "refs" / f"{i}_{line['start']:.2f}-{line['end']:.2f}{f'_{cut:.2f}' if cut else ''}_v{base_sig}.wav"
        )
        if not path.exists():
            gap = np.zeros((int(0.3 * sr),) + own.shape[1:], dtype="float32")
            sf.write(path, np.concatenate([base, gap, own]), sr)
        refs[i] = (str(path), f"{voices[line['voice']]['ref_text']} {said(line, src)}")
    return refs


FILL_TOL = 0.25  # s; a take ending this close to the actor's last sound is left alone
FILL_SPEED = (0.8, 1.25)  # OmniVoice speeds allowed when re-timing a take (judge by ear)


def actor_spans(job_id, lines):
    """Seconds from each window's start to the actor's first and last sound in it
    (Silero VAD on the separated vocals, as draft.py uses), or None where it hears
    nothing."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(str(JOBS / job_id / "vocals.wav"), sampling_rate=16000)
    opts = VadOptions(threshold=0.5, min_silence_duration_ms=120, speech_pad_ms=30)
    spans = [(s["start"] / 16000, s["end"] / 16000) for s in get_speech_timestamps(audio, opts)]
    out = []
    for line in lines:
        inside = [
            (max(a, line["start"]), min(b, line["end"])) for a, b in spans if b > line["start"] and a < line["end"]
        ]
        out.append(
            (min(a for a, _ in inside) - line["start"], max(b for _, b in inside) - line["start"]) if inside else None
        )
    return out


def actor_ends(job_id, lines):
    """Seconds from each window's start to the actor's last sound in it, or None."""
    return [span and span[1] for span in actor_spans(job_id, lines)]


def load_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def update_voices(change):
    """Apply change(voices) to work/voices.json under a lock and write it back:
    directors working on a show in parallel may add voices at the same time."""
    path = WORK / "voices.json"
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
        voices = load_json(path, {})
        result = change(voices)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(voices, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        # Closing the file releases the lock.
    return result


def voice_sig(key):
    """Fingerprint of a voice's spec in work/voices.json (clips, text, source)."""
    spec = load_json(WORK / "voices.json", {})[key]
    return hashlib.sha1(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def voice_clips(spec):
    """A voice's reference clips as (clip, start, end). An entry is [start, end]
    in the voice's "from" clip, or ["clip", start, end] from any prepped clip,
    so a bit part can pool its clean lines from several parts."""
    return [tuple(c) if len(c) == 3 else (spec["from"], *c) for c in spec["clips"]]


def voice_wav(key, rebuild=False):
    """work/voices/<key>.wav: the voice's clips from voices.json joined with 0.3 s
    pauses. Built when missing, when rebuild is set, or when the voice's entry no
    longer matches the profile it was last uploaded as."""
    import numpy as np
    import soundfile as sf

    spec = load_json(WORK / "voices.json", {})[key]
    ref = WORK / "voices" / f"{key}.wav"
    uploaded = load_json(WORK / "profiles.json", {}).get(key, {})
    if ref.exists() and not rebuild and uploaded.get("sig") == voice_sig(key):
        return ref
    sources, parts = {}, []
    for clip, a, b in voice_clips(spec):
        if clip not in sources:
            job = load_json(WORK / clip / "segments.json")["job_id"]
            sources[clip] = sf.read(JOBS / job / "vocals.wav", dtype="float32")
        audio, sr = sources[clip]
        gap = np.zeros((int(0.3 * sr),) + audio.shape[1:], dtype="float32")
        parts += [audio[int(a * sr) : int(b * sr)], gap]
    ref.parent.mkdir(parents=True, exist_ok=True)
    sf.write(ref, np.concatenate(parts[:-1]), sr)
    return ref


def speech_length(wav):
    """A take's length as VoiceStudio times it: generated edge silence trimmed
    to the first/last sample above -50 dBFS, plus 50 ms either side
    (services/audio_dsp.trim_speech_padding)."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(wav, dtype="float32")
    env = np.abs(audio).max(axis=1) if audio.ndim > 1 else np.abs(audio)
    voiced = np.nonzero(env > 10 ** (-50 / 20))[0]
    if not len(voiced):
        return len(env) / sr
    margin = int(sr * 0.05)
    return (min(len(env), voiced[-1] + 1 + margin) - max(0, voiced[0] - margin)) / sr


OMNIVOICE_SR, OMNIVOICE_FPS = 24000, 25  # OmniVoice's audio tokenizer: 24 kHz, 960-sample frames
STRETCH_USED = 0.5  # share of OmniVoice's stretch of short lines that a take actually speaks
SPEED_EXPONENT = 0.7  # a take asked for at speed s changes length by 1/s**0.7, not 1/s


def at_speed(natural, speed, exponent=SPEED_EXPONENT):
    """A take's length at `speed`, from its length at natural pace. It changes
    less than the speed asks: slowed, OmniVoice often says the words at its own
    pace and leaves the extra time silent, which is trimmed; sped up, it
    shortens less too. (IndexTTS's duration factor is exact: exponent 1.)"""
    return natural / speed**exponent


@functools.lru_cache(maxsize=1)
def _omnivoice():
    """OmniVoice's length rule and reference helpers, and VoiceStudio's text
    normalization, from the VoiceStudio checkout this Python runs from."""
    import importlib.util

    import omnivoice
    from omnivoice.utils.audio import remove_silence_safe
    from omnivoice.utils.duration import RuleDurationEstimator
    from omnivoice.utils.text import add_punctuation

    path = pathlib.Path(omnivoice.__file__).parents[1] / "backend" / "services" / "text_normalization.py"
    spec = importlib.util.spec_from_file_location("voicestudio_text_normalization", path)
    norm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norm)
    return RuleDurationEstimator(), add_punctuation, remove_silence_safe, norm.normalize_text


@functools.lru_cache(maxsize=512)
def _ref_frames(path, mtime):
    """A reference's length in OmniVoice frames, prepared as OmniVoice prepares it
    (create_voice_clone_prompt): mono at 24 kHz, pauses over 0.2 s shortened and
    the edges trimmed. (OmniVoice also raises a quiet clip's level before
    trimming; leaving that out predicted rendered takes better.)"""
    import soundfile as sf
    import torch
    import torchaudio

    audio, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(audio.mean(axis=1) if audio.ndim > 1 else audio)[None]
    wav = torchaudio.functional.resample(wav, sr, OMNIVOICE_SR)
    wav = _omnivoice()[2](wav, OMNIVOICE_SR, mid_sil=200, lead_sil=100, trail_sil=200)
    return wav.size(-1) * OMNIVOICE_FPS // OMNIVOICE_SR


def take_length(ref_audio, ref_text, text, speed=1.0, lang="en"):
    """How long a take of `text` (in language `lang`) cloned from this reference
    will be, in seconds of trimmed speech (as speech_length measures it). OmniVoice
    fixes a take's length before generating it (omnivoice/utils/duration.py): the
    reference's length times the text's weight over the transcript's weight. A
    kanji weighs 3, kana 2.2, a digit 3.5, a Latin letter 1 and a space 0.2, so
    English follows how dense the actor's Japanese is and how fast they said it
    (and Japanese how many letters the actor's English took, and how fast).
    Anything under 2 s is stretched towards 2 s; takes speak about half of that
    stretch. Fitted on about 1,000 Japanese -> English takes: typically within
    0.2 s, nine in ten within 0.4 s, about as close as two takes of one line are
    to each other."""
    estimator, add_punctuation, _, normalize = _omnivoice()
    ref = pathlib.Path(ref_audio)
    frames = _ref_frames(str(ref), ref.stat().st_mtime)
    text, ref_text = normalize(LANGUAGES[lang].speakable(text), lang), add_punctuation(ref_text)
    plain = estimator.estimate_duration(text, ref_text, frames, low_threshold=None)
    stretched = estimator.estimate_duration(text, ref_text, frames)
    return at_speed((stretched - STRETCH_USED * (stretched - plain)) / OMNIVOICE_FPS, speed)


# The "american" voice's engine, IndexTTS 2.5, fixes no length before it speaks. Following
# each line's original as its emotion reference, it keeps much of the actor's pace, so
# its takes follow both the English and how long the actor speaks in the window. Fitted
# on about 780 takes from an animated series and a live-action film: typically within
# 0.2 s, nine in ten within 0.5 s, and a fit on either show predicts the other as well.
IDX_LENGTH = (0.33, 0.10, 0.44)  # s: base, per English syllable, per second of the actor's speech


def syllables(text):
    """English syllables, counted by vowel groups (a number as two per digit)."""
    n = 0
    for word in re.findall(r"[A-Za-z']+|\d+", text):
        if word.isdigit():
            n += 2 * len(word)
            continue
        word = word.lower().replace("'", "")
        count = len(re.findall(r"[aeiouy]+", word))
        if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
            count -= 1  # a silent e
        n += max(count, 1)
    return n


def indextts_length(text, actor, speed=1.0):
    """How long an IndexTTS take of `text` will be, in seconds of trimmed speech,
    when the actor speaks for `actor` seconds in the window. Its duration factor
    (VoiceStudio's speed, with the local patch) scales a take exactly."""
    base, per_syllable, per_actor = IDX_LENGTH
    return (base + per_syllable * syllables(text) + per_actor * actor) / speed


SHORT_BY = 0.5  # s of the actor's speech left uncovered that makes a line "SHORT"


def fit_rows(name, overflow=0.3, predict=False, short=False, voice=None):
    """Each rendered clip's length vs its window, and the room concise timing
    allows: VoiceStudio's own rule (dub_generate.py) of the window, plus up to
    0.25 s of the gap before the next line (less a 0.05 s buffer), plus the
    overflow budget. Takes are measured with their edge silence trimmed, as
    VoiceStudio does. With `predict`, lines not rendered yet get the length
    OmniVoice will give them at natural pace ("est", take_length), from the
    reference the render will use: the line's own original when the part renders
    with per-line references and its window is clean (fallbacks are printed, as
    generate prints them), otherwise the voice's clips. They're flagged "OVER?"
    if the estimate exceeds the room, "risk" if within 0.3 s of it. When the part
    renders with fill, a line only overflows if it would even at fill's fastest
    speed. With `short`, a line whose English ends more than SHORT_BY before the
    actor stops is flagged "SHORT" (the take as rendered) or "SHORT?" (the
    estimate at natural pace), with the actor's end in "actor": it needs fuller
    text. Estimates get no credit for fill's slowing: slowed takes often come
    out little longer, so the text has to fill the time by itself.
    A part in the "american" voice is estimated with IndexTTS's length
    (indextts_length) instead. Its duration factor scales a take exactly, so
    there fill's slowest speed is credited, and "SHORT?" means even that ends
    SHORT_BY early. `voice` estimates the part in another voice than its own
    renders (the director's script in the American voice, with each line's
    "text_us"): its takes are ignored and every line is estimated."""
    work = WORK / name
    job = load_json(work / "segments.json")["job_id"]
    pair = langs(name)
    tgt = pair[1]
    script = load_json(work / "script.json")
    lines = script["lines"]
    rendered = load_json(work / "rendered.json", {})
    mode = script.get("render", {})
    other = voice is not None and voice != mode.get("voice", "actor")
    if other:
        # Another voice's render: nothing of this one's applies, and it renders with fill.
        rendered, mode = {}, {"fill": True, "voice": voice}
        if voice == "american":
            lines = [{**line, "text": line.get("text_us", line["text"])} for line in lines]
    fill = mode.get("fill", not rendered)
    fastest = FILL_SPEED[1] if fill else 1.0
    american = mode.get("voice") == "american"
    exponent = 1.0 if american else SPEED_EXPONENT
    slowest = FILL_SPEED[0] if fill and american else 1.0
    ids = [line_id(line) for line in lines]
    own, base = {}, {}  # per-line references by line id, voices' references by key
    if predict and not american:
        voices = load_json(WORK / "voices.json", {})
        todo = [
            k for k, i in enumerate(ids) if not take_wav(job, i, tgt).exists() and "clips" in voices[lines[k]["voice"]]
        ]
        base = {key: (voice_wav(key), voices[key].get("ref_text", "")) for key in {lines[k]["voice"] for k in todo}}
        if todo and mode.get("line_ref", not rendered and PAIRS[pair]["line_ref"]):
            own = line_refs(
                work,
                job,
                [lines[k] for k in todo],
                [ids[k] for k in todo],
                dips=mode.get("dips", not rendered),
                cast_lines=mode.get("cast_lines", not rendered),
            )
    spans = actor_spans(job, lines) if short or (predict and american) else [None] * len(lines)
    rows = []
    for i, (line, sid, span) in enumerate(zip(lines, ids, spans)):
        wav = take_wav(job, sid, tgt)
        if other:
            wav = JOBS / job / "none"
        end = span and span[1] if short else None
        row = {
            "i": i,
            "start": line["start"],
            "text": line["text"],
            "window": line["end"] - line["start"],
            "actor": end,
        }
        gap = lines[i + 1]["start"] - line["end"] if i + 1 < len(lines) else 0.0
        borrow = min(gap - 0.05, 0.25) if gap > 0.05 else 0.0
        row["room"] = row["window"] + borrow + overflow
        length = None
        if wav.exists():
            length = row["speech"] = speech_length(wav)
            row["flag"] = "OVER" if row["speech"] > row["room"] else ("tight" if row["speech"] > row["window"] else "")
        else:
            row["speech"], row["flag"] = None, ""
            ref = own.get(sid) or base.get(line["voice"])
            if american and predict:
                length = row["est"] = indextts_length(line["text"], span[1] - span[0] if span else row["window"])
            elif ref:
                length = row["est"] = take_length(*ref, line["text"], lang=tgt)
            if length:
                row["flag"] = (
                    "OVER?"
                    if at_speed(length, fastest, exponent) > row["room"]
                    else ("risk" if at_speed(length, fastest, exponent) > row["room"] - 0.3 else "")
                )
        # A take as rendered; an estimate as slow as fill can make it (American: exact).
        ends_at = at_speed(length, slowest, exponent) if length and row["speech"] is None else length
        if short and end and length and not row["flag"] and ends_at < end - SHORT_BY:
            row["flag"] = "SHORT" if row["speech"] is not None else "SHORT?"
        rows.append(row)
    return rows


# A part's work folder per voice: <show>-NN for the actor's voice (the director's
# script lives there), <show>-NN.us for the American voice, whose script dub_anything.py
# derives from it (a line's "text_us" replaces its text for that voice only). Which
# voices a dub into a language can have is its module's VOICES (languages/).
VOICE_SUFFIX = {"actor": "", "american": ".us"}
VOICE_LABEL = {"actor": "actor's voice", "american": "American"}


def voice_part(name, voice):
    return name + VOICE_SUFFIX[voice]


def base_part(name):
    """The part a voice's copy (<part>.us) was made from; the name itself otherwise."""
    return next((name[: -len(s)] for s in VOICE_SUFFIX.values() if s and name.endswith(s)), name)


def line_id(line):
    """Stable id from voice + text, so retiming a line reuses its cached speech;
    changing the voice's clips in voices.json re-synthesizes its lines. A line
    repeated word for word needs `"take": n` on the repeats to get ids of its own."""
    key = f"{line['voice']}|{voice_sig(line['voice'])}|{line['text']}"
    if line.get("take"):
        key += f"|take {line['take']}"
    return "L" + hashlib.sha1(key.encode()).hexdigest()[:10]
