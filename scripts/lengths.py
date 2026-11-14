"""How long a take is, how long one will be, and whether it fits: a take's speech
as VoiceStudio times it (speech_length), where the actor speaks in a window
(actor_spans), the length OmniVoice or IndexTTS will give a line before it is
rendered (take_length, indextts_length), and each line against its room (fit_rows).
"""

import functools
import pathlib
import re

import core
from languages import LANGUAGES, PAIRS
import references
import voicecast

FILL_TOL = 0.25  # s; a take ending this close to the actor's last sound is left alone


FILL_SPEED = (0.8, 1.25)  # OmniVoice speeds allowed when re-timing a take (judge by ear)


def actor_spans(job_id, lines):
    """Seconds from each window's start to the actor's first and last sound in it
    (Silero VAD on the separated vocals, as draft.py uses), or None where it hears
    nothing. A director's checked actor_end overrides the detected endpoint."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(str(core.JOBS / job_id / "vocals.wav"), sampling_rate=16000)
    opts = VadOptions(threshold=0.5, min_silence_duration_ms=120, speech_pad_ms=30)
    spans = [(s["start"] / 16000, s["end"] / 16000) for s in get_speech_timestamps(audio, opts)]
    out = []
    for line in lines:
        inside = [
            (max(a, line["start"]), min(b, line["end"])) for a, b in spans if b > line["start"] and a < line["end"]
        ]
        span = (
            (min(a for a, _ in inside) - line["start"], max(b for _, b in inside) - line["start"]) if inside else None
        )
        if "actor_end" in line:
            end = line["actor_end"]
            if isinstance(end, bool) or not isinstance(end, (int, float)) or not line["start"] < end <= line["end"]:
                raise ValueError("actor_end must be a finite time inside the line's window")
            first = span[0] if span and span[0] < end - line["start"] else 0.0
            span = first, end - line["start"]
        out.append(span)
    return out


def actor_ends(job_id, lines):
    """Seconds from each window's start to the actor's last sound in it, or None."""
    return [span and span[1] for span in actor_spans(job_id, lines)]


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


def _voicestudio_normalizer(backend):
    """Load the checkout's normalizer without replacing this module's globals."""
    import importlib.util

    # VoiceStudio's normalizer imports helpers such as core.language_variants.
    # Keep our core.py as the parent, with its backend's helper search path.
    core.__path__ = [str(pathlib.Path(backend) / "core")]
    path = pathlib.Path(backend) / "services" / "text_normalization.py"
    spec = importlib.util.spec_from_file_location("voicestudio_text_normalization", path)
    norm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norm)
    return norm.normalize_text


@functools.lru_cache(maxsize=1)
def _omnivoice():
    """OmniVoice's length rule and reference helpers, and VoiceStudio's text
    normalization, from the VoiceStudio checkout this Python runs from."""
    import omnivoice
    from omnivoice.utils.audio import remove_silence_safe
    from omnivoice.utils.duration import RuleDurationEstimator
    from omnivoice.utils.text import add_punctuation

    backend = pathlib.Path(omnivoice.__file__).parents[1] / "backend"
    return RuleDurationEstimator(), add_punctuation, remove_silence_safe, _voicestudio_normalizer(backend)


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
    (VoiceStudio's speed, with the pinned fork) scales a take exactly."""
    base, per_syllable, per_actor = IDX_LENGTH
    return (base + per_syllable * syllables(text) + per_actor * actor) / speed


SHORT_BY = 0.5  # s of the actor's speech left uncovered that makes a line "SHORT"


def fit_rows(name, overflow=0.3, predict=False, short=False, voice=None, notes=None):
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
    estimate at natural pace), with the actor's end in "actor". Review meaning,
    source pauses and quiet-reference estimation before expanding the text;
    complete translations do not need invented padding. Estimates get no credit
    for fill's slowing: slowed takes often come out little longer, so the text
    has to fill the time by itself.
    A part in the "american" voice is estimated with IndexTTS's length
    (indextts_length) instead. Its duration factor scales a take exactly, so
    there fill's slowest speed is credited, and "SHORT?" means even that ends
    SHORT_BY early. `voice` estimates the part in another voice than its own
    renders (the director's script in the American voice, with each line's
    "text_us"): its takes are ignored and every line is estimated. `notes`
    collects the per-line reference notes generate would print (references.line_refs)."""
    work = core.part_dir(name)
    job = core.load_json(work / "segments.json")["job_id"]
    pair = core.langs(name)
    tgt = pair[1]
    script = core.load_json(work / "script.json")
    lines = script["lines"]
    rendered = core.load_json(work / "rendered.json", {})
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
    ids = [voicecast.line_id(line) for line in lines]
    own, base = {}, {}  # per-line references by line id, voices' references by key
    check_references = (
        bool(mode.get("fallback_voices"))
        or any("fallback_voice" in line for line in lines)
        or any("fallback_voice" in settings for settings in rendered.values())
    )
    if predict and not american:
        voices = voicecast.load_voices()
        todo = [
            k
            for k, i in enumerate(ids)
            if (check_references or not core.take_wav(job, i, tgt).exists()) and "clips" in voices[lines[k]["voice"]]
        ]
        keys = {references.fallback_voice(lines[k], mode, voices) for k in todo}
        base = {
            key: (voicecast.voice_wav(key), voices[key].get("ref_text", "")) for key in keys if "clips" in voices[key]
        }
        if todo and mode.get("line_ref", not rendered and PAIRS[pair]["line_ref"]):
            # Generation checks the full cast, including lines with cached takes.
            # Keep those rivals and their cached source checks after a rewording.
            own = references.line_refs(
                work,
                job,
                lines,
                ids,
                dips=mode.get("dips", not rendered),
                cast_lines=mode.get("cast_lines", not rendered),
                prefix=mode.get("ref_prefix", True),
                notes=notes,
            )
    spans = actor_spans(job, lines) if short or (predict and american) else [None] * len(lines)
    rows = []
    for i, (line, sid, span) in enumerate(zip(lines, ids, spans)):
        line_fastest = line.get("speed", fastest)
        line_slowest = line.get("speed", slowest)
        wav = core.take_wav(job, sid, tgt)
        if other:
            wav = core.JOBS / job / "none"
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
        stale = (
            predict
            and not american
            and check_references
            and references.reference_signature(rendered.get(sid, {}))
            != references.reference_settings(line, mode, own.get(sid))
        )
        if stale:
            row["stale_reference"] = True
        if wav.exists() and not stale:
            length = row["speech"] = speech_length(wav)
            row["flag"] = "OVER" if row["speech"] > row["room"] else ("tight" if row["speech"] > row["window"] else "")
        else:
            row["speech"], row["flag"] = None, ""
            ref = own.get(sid) or base.get(references.fallback_voice(line, mode))
            if american and predict:
                length = row["est"] = indextts_length(line["text"], span[1] - span[0] if span else row["window"])
            elif ref:
                length = row["est"] = take_length(*ref, line["text"], lang=tgt)
            if length:
                if "speed" in line:
                    row["est"] = at_speed(length, line["speed"], exponent)
                row["flag"] = (
                    "OVER?"
                    if at_speed(length, line_fastest, exponent) > row["room"]
                    else ("risk" if at_speed(length, line_fastest, exponent) > row["room"] - 0.3 else "")
                )
        # A take as rendered; an estimate as slow as fill can make it (American: exact).
        ends_at = at_speed(length, line_slowest, exponent) if length and row["speech"] is None else length
        if short and end and length and not row["flag"] and ends_at < end - SHORT_BY:
            row["flag"] = "SHORT" if row["speech"] is not None else "SHORT?"
        rows.append(row)
    return rows
