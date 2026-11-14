"""What a take is cloned from: a line's own original as its reference (line_refs),
the check that a window is clean enough to be one (unclean_windows), the voice a
line falls back to when it isn't (fallback_voice), and what a rendered take
remembers of all that (reference_settings, reference_signature).
"""

from collections import Counter
import hashlib
import pathlib
import re
import unicodedata

import core
from languages import LANGUAGES
import voicecast

LINE_REF_MAX = 19.5  # s; OmniVoice refuses a reference over 20 s that has a transcript


ROOM_EFFECTS = {"echo", "pa"}  # their original carries the room, which a reference would clone


# How far Whisper's hearing of a window may be from the line's `ja` (character error
# rate). Kanji written as kana counts against it, so only gross mismatches fail.
LINE_REF_CER = 0.5


KANA = {c: c - 0x60 for c in range(0x30A1, 0x30F7)}  # katakana -> hiragana, for str.translate


def fallback_voice(line, mode, voices=None):
    """Resolve an optional fallback within the assigned speaker's voice family.

    This changes only the reference used when a clean per-line reference is
    unavailable; it never changes casting or the per-line source checks.
    """
    mapping = mode.get("fallback_voices", {})
    if not isinstance(mapping, dict):
        raise ValueError("render.fallback_voices must be an object")
    original = line["voice"]
    selected = line.get("fallback_voice", mapping.get(original, original))
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError("fallback_voice must name a voice in voices.json")
    if selected == original:
        return selected
    voices = voicecast.load_voices() if voices is None else voices

    def family(key):
        seen = set()
        while True:
            if key in seen or key not in voices:
                raise ValueError("fallback_voice has an unknown or cyclic speaker family")
            seen.add(key)
            parent = voices[key].get("register_of")
            if not parent:
                return key
            key = parent

    if family(selected) != family(original):
        raise ValueError("fallback_voice must belong to the assigned speaker's family")
    return selected


def reference_settings(line, mode, own=None):
    """Cache identity of the actual audio reference, including fallback changes."""
    if own:
        return {"line_ref": pathlib.Path(own[0]).stem.split("_", 1)[1]}
    selected = fallback_voice(line, mode)
    return (
        {"fallback_voice": selected, "fallback_sig": voicecast.voice_sig(selected)} if selected != line["voice"] else {}
    )


def reference_signature(settings):
    """Only the conditioning fields that make a measured take incompatible."""
    return {k: settings[k] for k in ("line_ref", "fallback_voice", "fallback_sig") if k in settings}


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


def source_check_keys(work, lines, dips=False, cast_lines=False):
    """Source-check cache keys shared by eligibility and diagnostic reports."""
    src = core.langs(core.part_name(work))[0]
    policy = getattr(LANGUAGES[src], "SOURCE_CHECK_VERSION", "")
    rivals = hashlib.sha1(
        "|".join(f"{v}:{voicecast.voice_sig(v)}" for v in sorted({line["voice"] for line in lines})).encode()
    ).hexdigest()[:8]
    return [
        f"{line['start']:.2f}-{line['end']:.2f}_"
        + hashlib.sha1(
            f"{line['voice']}|{voicecast.voice_sig(line['voice'])}|{core.said(line, src)}".encode()
        ).hexdigest()[:12]
        + f"_{rivals}"
        + ("_dips" if dips else "")
        + ("_cast" if cast_lines else "")
        + (f"_{policy}" if policy else "")
        for line in lines
    ]


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
    words, the language's source-check policy, and the clips of every voice it's
    compared with), not by the take: rewording the
    dub's text keeps it, and rebuilding another of the clip's voices checks its lines
    again. Lines cast in other parts aren't in the key, so a decision stays put
    while the show's other parts are directed. Whisper listens in the part's source
    language, and the line's original words are its field named for it."""
    src = core.langs(core.part_name(work))[0]
    normalize = getattr(LANGUAGES[src], "normalize_source", lambda text: (text, []))
    cache_path = work / "refs" / "clean.json"
    cache = core.load_json(cache_path, {})
    keys = source_check_keys(work, lines, dips, cast_lines)
    if any(k not in cache for k in keys):
        import draft
        import edges
        from register import cer
        from reference_review import speech_margins

        sr = draft.SR
        audio = draft.decode_audio(str(core.JOBS / job_id / "vocals.wav"), sampling_rate=sr)
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
        voices = voicecast.load_voices()
        who = lambda v: voices[v].get("register_of", v)  # noqa: E731
        cast = (
            {
                v: embed(draft.decode_audio(str(voicecast.voice_path(v)), sampling_rate=sr))
                for v in {line["voice"] for line in lines}
            }
            if embed
            else {}
        )
        # Exclude this part's own lines: a line mustn't vouch for itself.
        cents = (
            draft.cast_centroids(embed, {who(v) for v in cast}, exclude=core.part_name(work))
            if (cast_lines and embed)
            else {}
        )
        review_path = work / "refs" / "source-review.json"
        reviews = core.load_json(review_path, {})
        source_stat = (core.JOBS / job_id / "vocals.wav").stat()
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
                found = draft.transcribe(model, x, language=src)
                heard = "".join(p["said"] for p in found)
                reviews[key] = {
                    "start": a,
                    "end": cut or b,
                    "voice": line["voice"],
                    "heard": heard,
                    "source": {"job": job_id, "size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
                    **speech_margins(found, len(x) / sr),
                }
                expected, expected_numbers = normalize(core.said(line, src))
                observed, observed_numbers = normalize(heard)
                err = cer(bare(expected), bare(observed))
                if expected_numbers != observed_numbers:
                    why = f"number differs: {expected_numbers} -> {observed_numbers}"
                elif err > LINE_REF_CER:
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
        core.save_json(cache_path, cache)
        core.save_json(review_path, reviews)
    return [(None, cache[k]["cut"]) if isinstance(cache[k], dict) else (cache[k], None) for k in keys]


def line_refs(work, job_id, lines, ids, verbose=True, dips=False, cast_lines=False, prefix=True, notes=None):
    """Per-line references: the voice's own clips (voice_path: voices/<key>.wav), a 0.3 s
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
    a voice must not reuse a cached reference containing its old audio.
    With prefix=False (or a line's ref_prefix=False), use only that line's
    original audio/transcript. Source cleanliness checks still apply.
    With `notes` (a dict), what verbose would print goes into it instead, as
    {line index: [note, ...]}, for a caller that reports by line (check.py)."""
    import numpy as np
    import soundfile as sf

    voices = voicecast.load_voices()
    src = core.langs(core.part_name(work))[0]
    vocals, sr = sf.read(core.JOBS / job_id / "vocals.wav", dtype="float32")
    (work / "refs").mkdir(exist_ok=True)
    refs = {}
    repeated = {key for key, count in Counter(ids).items() if count > 1}
    checks = unclean_windows(work, job_id, lines, ids, dips, cast_lines)
    reviews = core.load_json(work / "refs" / "source-review.json", {})
    keys = source_check_keys(work, lines, dips, cast_lines) if reviews else [None] * len(lines)
    source_stat = (core.JOBS / job_id / "vocals.wav").stat()
    source = {"job": job_id, "size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns}

    def note(k, kind, detail):
        if notes is not None:
            notes.setdefault(k, []).append(f"{kind}: {detail}")
        elif verbose:
            print(f"  {kind} at {lines[k]['start']}: {detail}")

    for k, (line, i, key, (why, cut)) in enumerate(zip(lines, ids, keys, checks)):
        if not line.get("line_ref", True) or line.get("effect") in ROOM_EFFECTS:
            continue
        if why:
            note(k, "line ref skipped", why)
            continue
        review = reviews.get(key, {})
        own = vocals[int(line["start"] * sr) : int((cut or line["end"]) * sr)]
        if (verbose or notes is not None) and review.get("source") == source and review.get("review_tail"):
            from reference_review import audible_tail

            tail = audible_tail(own, sr, review)
        else:
            tail = {"review_tail": False}
        if tail["review_tail"]:
            note(
                k,
                "reference review",
                f"{review['trailing_seconds']:.2f} s after the last recognized word; "
                f"{tail['tail_active_seconds']:.2f} s has audible energy; "
                "check for laughter, crosstalk, noise or missed speech (reference unchanged)",
            )
        with_prefix = line.get("ref_prefix", prefix)
        if not isinstance(with_prefix, bool):
            raise ValueError("ref_prefix must be true or false")
        transcript = core.said(line, src)
        if with_prefix:
            base_path = voicecast.voice_path(line["voice"])
            base, base_sr = sf.read(base_path, dtype="float32")
            if base_sr != sr or base.shape[1:] != own.shape[1:]:
                note(k, "line ref skipped", f"{base_path.name} isn't in the vocals' format")
                continue
            duration = (len(base) + len(own)) / sr + 0.3
            reference_sig = "v" + hashlib.sha1(base.tobytes()).hexdigest()[:12]
            gap = np.zeros((int(0.3 * sr),) + own.shape[1:], dtype="float32")
            audio = np.concatenate([base, gap, own])
            transcript = f"{voices[line['voice']]['ref_text']} {transcript}"
        else:
            duration = len(own) / sr
            reference_sig = "own" + hashlib.sha1(own.tobytes() + transcript.encode()).hexdigest()[:12]
            audio = own
        if duration > LINE_REF_MAX:
            note(k, "line ref skipped", f"over {LINE_REF_MAX:.0f} s")
            continue
        if i in repeated:
            raise ValueError(
                f"Repeated line {i} would share a per-line source reference; add distinct take values to its occurrences"
            )
        path = (
            work
            / "refs"
            / f"{i}_{line['start']:.2f}-{line['end']:.2f}{f'_{cut:.2f}' if cut else ''}_{reference_sig}.wav"
        )
        if not path.exists():
            sf.write(path, audio, sr)
        refs[i] = (str(path), transcript)
    return refs
