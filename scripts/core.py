"""Shared by the Dub Anything scripts: where things live (work/, a job's folder,
VoiceStudio's job folders, the model cache), a part's language pair and its
takes' files, and how the JSON files the tools share are read and written. The
rest of what the scripts share is beside it: the cast and its fingerprints
(voicecast.py), per-line references and the window check (references.py), take
lengths and fit (lengths.py).
"""

import json
import os
import pathlib
import re
import sys
import time

from gpu import rocm_env
from languages import PAIR, pair_of

HERE = pathlib.Path(__file__).resolve().parent  # the scripts
ROOT = HERE.parent  # the repo: work/, videos/, vendor/
WORK = ROOT / "work"


# ---- where a job's work lives ----
#
# A job (dub_anything.py new) keeps everything it makes in one folder, work/jobs/<slug>/:
#   job.json, report.md, NOTES.md      what the job is, where it stands, the show's notes
#   voices.json, voices/, profiles.json   its cast (job.json's "voices_from" borrows others')
#   logs/, scratch/                     the runner's logs; the director's own scripts and probes
#   media/<slug>.<ext>                  a link to the source video
#   media/split/                        its parts (<slug>-NN.mp4, parts.json); renders in dubbed/
#   parts/NN/, parts/NN.us/             each part's work: script, draft, state, refs, takes
# and VoiceStudio's folder for a part is named after it (<slug>-NN, <slug>-NN-us). A part
# is still named <slug>-NN on every command line; part_dir finds its folder. A clip prepped
# by hand, outside a job, keeps the older layout: work/<name> and the shared work/voices.json.
PART_NAME = re.compile(r"(?P<slug>[a-z0-9_]+)-(?P<nn>\d+)(?P<suffix>(?:\.[a-z]+)?)")
_job = {"slug": None}


def job_dir(slug):
    return WORK / "jobs" / slug


def job_of(name):
    """The job a part belongs to (its name's slug, when that job exists), else None."""
    m = PART_NAME.fullmatch(name)
    return m["slug"] if m and (job_dir(m["slug"]) / "job.json").exists() else None


def use_job(slug):
    """Name the job whose cast this process reads and writes (None: none). Without
    this, it is the job of the first part a tool looks up, or $DUB_JOB."""
    _job["slug"] = slug


def active_job():
    slug = _job["slug"] or os.environ.get("DUB_JOB")
    return slug if slug and (job_dir(slug) / "job.json").exists() else None


def adopt_job(words):
    """For a tool given voice keys rather than parts: make active the job of the first
    part named in `words`, else the one job whose cast has every key in it. Returns it."""
    if active_job():
        return active_job()
    for word in words:
        if job_of(word):
            use_job(job_of(word))
            return job_of(word)
    owners = [
        d.name
        for d in sorted((WORK / "jobs").glob("*"))
        if (d / "job.json").exists() and words and set(words) <= set(load_json(d / "voices.json", {}))
    ]
    if len(owners) == 1:
        use_job(owners[0])
        return owners[0]
    return None


def part_dir(name):
    """A part's work folder: work/jobs/<slug>/parts/NN[.us] for a job's part,
    work/<name> for a clip prepped by hand."""
    slug = job_of(name)
    if not slug:
        return WORK / name
    if not _job["slug"]:
        _job["slug"] = slug
    m = PART_NAME.fullmatch(name)
    return job_dir(slug) / "parts" / (m["nn"] + m["suffix"])


def part_name(folder):
    """The part a work folder belongs to, as part_dir names it."""
    folder = pathlib.Path(folder)
    if folder.parent.name == "parts" and (folder.parent.parent / "job.json").exists():
        return f"{folder.parent.parent.name}-{folder.name}"
    return folder.name


def part_names(slug=None):
    """The parts with a work folder, by name: the job's (the active one by default),
    or without a job every clip prepped by hand in work/."""
    slug = slug or active_job()
    if slug:
        folders = (job_dir(slug) / "parts").glob("*")
        return sorted(f"{slug}-{d.name}" for d in folders if d.is_dir())
    return sorted(d.name for d in WORK.glob("*") if (d / "segments.json").exists() or (d / "script.json").exists())


def script_files():
    """(name, script.json) of every directed part of the active job (see part_names)."""
    return [(n, part_dir(n) / "script.json") for n in part_names() if (part_dir(n) / "script.json").exists()]


def split_dir(slug):
    """Where split.py puts a video's parts: the job's media/split, or for a video
    split by hand, videos/<stem>-parts."""
    if (job_dir(slug) / "job.json").exists():
        return job_dir(slug) / "media" / "split"
    return ROOT / "videos" / f"{slug}-parts"


def cast_dir(slug=None):
    """The folder holding a cast: voices.json, voices/<key>.wav, profiles.json."""
    slug = slug or active_job()
    return job_dir(slug) if slug else WORK


def voices_file():
    return cast_dir() / "voices.json"


def voices_dir():
    return cast_dir() / "voices"


def profiles_file():
    return cast_dir() / "profiles.json"


def borrowed_casts(slug=None):
    """The jobs whose casts this one borrows (job.json "voices_from"), nearest last."""
    slug = slug or active_job()
    return load_json(job_dir(slug) / "job.json", {}).get("voices_from", []) if slug else []


def segments_of(name):
    """A part's segments.json (the VoiceStudio job id, the machine transcript), or None."""
    return load_json(part_dir(name) / "segments.json")


def vs_job(name):
    """A part's VoiceStudio job folder (stems, takes, renders)."""
    return JOBS / segments_of(name)["job_id"]


def vocals_wav(name):
    """A part's separated voices."""
    return vs_job(name) / "vocals.wav"


def new_vs_job_id(name):
    """The id to give a part's new VoiceStudio job: its name, so the folder says
    whose it is (letters, digits, - and _ only), numbered if one is there already."""
    base = re.sub(r"[^A-Za-z0-9_-]", "-", name)[:56]
    job_id, n = base, 1
    while (JOBS / job_id).exists():
        n += 1
        job_id = f"{base}-{n}"
    return job_id


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
            seg = segments_of(n)
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


def load_json(path, default=None):
    # Windows can briefly refuse a read while another process swaps the file in (save_json).
    for pause in (0.02, 0.05, 0.1, None):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        except PermissionError:
            if pause is None:
                raise
            time.sleep(pause)


def save_json(path, value, indent=1):
    """Write JSON so a reader sees the old file or the new one, never part of one:
    the runner, the director and the tools read each other's files while they work,
    and a write cut short would otherwise leave JSON that stops the next reader."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=indent)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows refuses the swap while a reader or a scanner has the file open.
        for pause in (0.02, 0.05, 0.1, 0.25, 0.5, None):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if pause is None:
                    raise
                time.sleep(pause)
    finally:
        temporary.unlink(missing_ok=True)


RESULT_ENV = "DUB_RESULT_FILE"  # where the runner asks a tool to leave what it decided


def report_result(**fields):
    """Leave the runner the facts it acts on (dub_anything.tool reads them back), so
    its decisions don't depend on how a message is worded. Without the runner, nothing."""
    path = os.environ.get(RESULT_ENV)
    if path:
        path = pathlib.Path(path)
        save_json(path, {**load_json(path, {}), **fields})


# The director's remarks: fields of a line, a caption, a sign, a script or a voice
# that people and the report read, never a render or the delivery. A fingerprint
# leaves them out, so writing one down costs no render. Any other field counts as
# part of what is rendered: one no script knows yet makes a take again rather than
# leaving a stale one.
REMARKS = ("listen", "note", "notes")


def without_remarks(value):
    """A script (or any part of one) without the remark fields, at every depth."""
    if isinstance(value, dict):
        return {k: without_remarks(v) for k, v in value.items() if k not in REMARKS}
    if isinstance(value, list):
        return [without_remarks(v) for v in value]
    return value


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
