"""Match each character's voice reference to the register of the lines it speaks.

OmniVoice copies a reference's delivery along with the voice: a character cloned
from a line spoken to a room projects every line, and one cloned from a quiet
aside flattens every outburst. This measures the register of speech on the
separated vocals: pitch (semitones), vocal effort (spectral tilt, 1-4 kHz
against 100 Hz-1 kHz, which rises as people raise their voice) and loudness.
"Arousal" combines their robust z-scores within the character's own lines, so
it is relative to how that actor usually speaks: 0 is typical, +1 is noticeably
raised, -1 noticeably quieter.

A character can have extra references, one per register: a voice key whose
spec in voices.json has "register_of": "<character>" (and "register": "calm" or
"raised"). Every tool treats them as the same character; `route` gives each line
the reference nearest the actor's delivery on that line, except lines marked
"keep_voice": true (a word one reference can't say, say).

Usage: scripts/register.py report [clip ...]            each reference against the lines it serves
       scripts/register.py build <voice> calm|raised    a reference in that register from the
                                                        character's own clean lines (-> voices.json)
       scripts/register.py route <clip ...> [--apply]   give each line the nearest-register reference
       scripts/register.py check [clip ...]             how well the takes' register follows the
                                                        original's, and whether each extra reference
                                                        delivers its register
       scripts/register.py drop <voice>                 retire an extra reference that doesn't
report and check default to every directed clip (work/*/script.json).
"""

import argparse
import json
import re
import sys

import numpy as np

import draft
from core import JOBS, WORK, bare, langs, line_id, load_json, said, take_wav, update_voices, voice_clips
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SR, FRAME, HOP = 16000, 1024, 256
FMIN, FMAX = 70, 500
MIN_LINE = 0.6  # s; shorter lines are too short to measure
APART = 1.0  # arousal gap between a line and its reference that counts as a mismatch
HYSTERESIS = 0.5  # route only when another reference is nearer by this much
_audio = {}


def vocals(clip):
    """A clip's separated vocals at 16 kHz (None if its job is gone)."""
    if clip not in _audio:
        wav = JOBS / load_json(WORK / clip / "segments.json", {}).get("job_id", "-") / "vocals.wav"
        _audio[clip] = decode_audio(str(wav), sampling_rate=SR) if wav.exists() else None
    return _audio[clip]


def yin(fr):
    """Pitch per frame by YIN (Hz; nan where the frame isn't periodic)."""
    tmin, tmax = SR // FMAX, SR // FMIN
    w = FRAME - tmax
    r = np.fft.irfft(np.conj(np.fft.rfft(fr[:, :w], 2048)) * np.fft.rfft(fr, 2048), 2048)[:, : tmax + 1]
    cs = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(fr**2, axis=1)], axis=1)
    taus = np.arange(tmax + 1)
    d = cs[:, w][:, None] + cs[:, taus + w] - cs[:, taus] - 2 * r
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(np.cumsum(d[:, 1:], axis=1), 1e-12)
    f0 = np.full(len(fr), np.nan)
    for k, c in enumerate(cmnd[:, tmin:]):
        below = np.nonzero(c < 0.15)[0]
        if len(below):
            j = below[0]
            while j + 1 < len(c) and c[j + 1] < c[j]:
                j += 1
        else:
            j = int(np.argmin(c))
            if c[j] > 0.35:
                continue
        f0[k] = SR / (tmin + j)
    return f0


def features(x):
    """Register of a stretch of 16 kHz speech: pitch (semitones re 100 Hz),
    effort (spectral tilt, dB) and level (dB), over its loud frames."""
    if len(x) < FRAME + HOP * 8:
        return None
    n = 1 + (len(x) - FRAME) // HOP
    fr = x[np.arange(FRAME)[None] + HOP * np.arange(n)[:, None]].astype(np.float64)
    db = 10 * np.log10((fr**2).mean(axis=1) + 1e-10)
    loud = db > db.max() - 25
    f0 = yin(fr[loud])
    f0 = f0[~np.isnan(f0)]
    p = np.abs(np.fft.rfft(fr[loud] * np.hanning(FRAME), axis=1)) ** 2
    hz = np.fft.rfftfreq(FRAME, 1 / SR)
    tilt = 10 * np.log10(p[:, (hz >= 1000) & (hz < 4000)].sum() / (p[:, (hz >= 100) & (hz < 1000)].sum() + 1e-12))
    return {
        "pitch": 12 * np.log2(np.median(f0) / 100) if len(f0) >= 5 else np.nan,
        "effort": float(tilt),
        "level": float(np.percentile(db, 95)),
    }


FLOOR = {"pitch": 1.0, "effort": 1.0, "level": 1.5}
WEIGHT = {"pitch": 1.0, "effort": 1.0, "level": 0.5}


def norms(feats):
    """Median and robust spread of each feature over a character's lines."""
    out = {}
    for k in FLOOR:
        v = np.array([f[k] for f in feats if not np.isnan(f[k])])
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        out[k] = (med, max((q3 - q1) / 1.349, FLOOR[k]))
    return out


def arousal(f, nm):
    parts = [(WEIGHT[k] * (f[k] - nm[k][0]) / nm[k][1], WEIGHT[k]) for k in FLOOR if not np.isnan(f[k])]
    return sum(p for p, _ in parts) / sum(w for _, w in parts)


def families(voices):
    """Character -> its voice keys (the character's own key first)."""
    fam = {}
    for k, s in voices.items():
        fam.setdefault(s.get("register_of", k), []).append(k)
    return {b: sorted(ks, key=lambda k: k != b) for b, ks in fam.items()}


def clips_of(args_clips):
    return args_clips or sorted(p.parent.name for p in WORK.glob("*/script.json"))


class Show:
    """Every directed line of the show, measured, grouped by character."""

    def __init__(self):
        self.voices = load_json(WORK / "voices.json", {})
        self.fam = families(self.voices)
        self.base = {k: b for b, ks in self.fam.items() for k in ks}
        self.lines = {}  # character -> [(clip, i, line, features)]
        seen = set()  # an A/B copy of a part repeats its lines exactly: count them once
        for clip in clips_of(None):
            if vocals(clip) is None:
                continue
            for i, line in enumerate(load_json(WORK / clip / "script.json")["lines"]):
                if line["end"] - line["start"] < MIN_LINE or line.get("voice") not in self.base:
                    continue
                twin = (line["start"], line["end"], said(line, langs(clip)[0]), line["voice"])
                if twin in seen:
                    continue
                seen.add(twin)
                f = features(vocals(clip)[int(line["start"] * SR) : int(line["end"] * SR)])
                if f:
                    self.lines.setdefault(self.base[line["voice"]], []).append((clip, i, line, f))
        self.norm = {b: norms([f for *_, f in ls]) for b, ls in self.lines.items() if len(ls) >= 8}

    def ref_arousal(self, key):
        """Duration-weighted arousal of a voice's reference clips."""
        nm = self.norm[self.base[key]]
        got = [
            (arousal(f, nm), b - a)
            for clip, a, b in voice_clips(self.voices[key])
            if vocals(clip) is not None and (f := features(vocals(clip)[int(a * SR) : int(b * SR)]))
        ]
        return sum(x * w for x, w in got) / sum(w for _, w in got)

    def refs(self, base):
        return {k: self.ref_arousal(k) for k in self.fam[base] if self.voices[k].get("clips")}


def cmd_report(args):
    show = Show()
    for base, ls in sorted(show.lines.items(), key=lambda x: -len(x[1])):
        if base not in show.norm:
            continue
        refs = show.refs(base)
        if not refs:
            continue
        nm = show.norm[base]
        ar = np.array([arousal(f, nm) for *_, f in ls])
        near = np.array([min(refs.values(), key=lambda r: abs(a - r)) for a in ar])
        calmer, louder = int((ar - near <= -APART).sum()), int((ar - near >= APART).sum())
        need = max(4, 0.15 * len(ls))
        todo = [b for b, n in (("calm", calmer), ("raised", louder)) if n >= need]
        print(
            f"{base:10s} {len(ls):3d} lines  "
            + "  ".join(f"ref {k} {r:+.1f} ({(ar < r).mean() * 100:.0f}th pct)" for k, r in refs.items())
            + f"   {calmer} lines {APART:g}+ calmer, {louder} louder than their nearest reference"
            + (f"   -> build {' and '.join(todo)}" if todo else "")
        )


def clean_lines(show, base):
    """The character's lines that can serve as reference clips: 1.2-6 s, a
    complete phrase (no trailing …) with nobody else within 0.3 s, and bounded by
    real pauses, so no word is cut off (a cut word leaks into every take)."""
    spans, out = {}, []
    for clip, i, line, f in show.lines[base]:
        d = line["end"] - line["start"]
        if not 1.2 <= d <= 6.0 or re.search(r"(…|\.\.\.)\s*$", said(line, langs(clip)[0])):
            continue
        script = load_json(WORK / clip / "script.json")
        others = [x for x in script["lines"] + script.get("captions", []) if x is not line and x != line]
        if any(x["start"] < line["end"] + 0.3 and x["end"] > line["start"] - 0.3 for x in others):
            continue
        if clip not in spans:
            opts = VadOptions(threshold=0.5, min_silence_duration_ms=100, speech_pad_ms=0)
            spans[clip] = [(s["start"] / SR, s["end"] / SR) for s in get_speech_timestamps(vocals(clip), opts)]
        if any(
            a < line["start"] - 0.05 and b > line["start"] - 0.25 or a < line["end"] + 0.25 and b > line["end"] + 0.05
            for a, b in spans[clip]
        ):
            continue
        out.append((clip, i, line, f))
    return out


def cmd_build(args):
    show = Show()
    base = show.base.get(args.voice, args.voice)
    key = f"{base}_{args.band}"
    if key in show.voices:
        raise SystemExit(f"{key} already exists in voices.json")
    nm, refs = show.norm[base], show.refs(base)
    sign = -1 if args.band == "calm" else 1
    # The lines the existing references serve worst in this direction set the target.
    gaps = [
        arousal(f, nm)
        for *_, f in show.lines[base]
        if sign * (arousal(f, nm) - min(refs.values(), key=lambda r: abs(arousal(f, nm) - r))) >= APART
    ]
    if not gaps:
        raise SystemExit(f"no {args.band} lines that the current references serve badly")
    target = float(np.median(gaps))
    embed = draft.embedder()
    centroid = draft.cast_centroids(embed, {base})[base]
    draft.hold_gpu_for_whisper()
    model = draft.WhisperModel("Systran/faster-whisper-large-v3", device="auto", compute_type="default")
    picked, total = [], 0.0
    for clip, i, line, f in sorted(clean_lines(show, base), key=lambda x: abs(arousal(x[3], nm) - target)):
        a = arousal(f, nm)
        if abs(a - target) > 1.0 or total >= 8.0 or len(picked) == 4:
            break
        x = vocals(clip)[int(line["start"] * SR) : int(line["end"] * SR)]
        sim = float(embed(x) @ centroid)
        src = langs(clip)[0]
        heard = "".join(q["said"] for q in draft.transcribe(model, x, language=src))
        ref, got = bare(said(line, src)), bare(heard)
        err = cer(ref, got)
        ok = sim >= 0.5 and err <= 0.25
        print(
            f"  {'take' if ok else 'skip'} {clip} L{i:02d} {line['start']:7.2f}-{line['end']:7.2f} arousal {a:+.1f}"
            f"  voice match {sim:.2f}  heard {err:.0%} off  {said(line, src)}"
        )
        if ok:
            picked.append([clip, line["start"], line["end"], said(line, src), a])
            total += line["end"] - line["start"]
    if total < 5.0:
        raise SystemExit(f"only {total:.1f} s of clean {args.band} speech for {base}: not enough for a reference")
    level = sum(p[4] * (p[2] - p[1]) for p in picked) / total
    nearest = min(refs.values(), key=lambda r: abs(level - r))
    if abs(level - nearest) < 0.8:
        raise SystemExit(
            f"the clean {args.band} lines average {level:+.1f}, too close to an existing reference "
            f"({nearest:+.1f}) to be a register of their own"
        )
    spec = {
        "from": picked[0][0],
        "clips": [p[:3] for p in picked],
        "ref_text": " ".join(p[3] for p in picked),
        "register_of": base,
        "register": args.band,
    }
    update_voices(lambda voices: voices.update({key: spec}))
    print(
        f"{key}: {total:.1f} s from {len(picked)} lines at arousal {level:+.1f} (target {target:+.1f}; "
        f"the character's own reference is {refs[base]:+.1f}). Written to voices.json; "
        f"check work/voices/{key}.wav with asr_words.py after the first render."
    )


def cer(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def cmd_route(args):
    show = Show()
    refs = {b: show.refs(b) for b in show.fam if len(show.fam[b]) > 1 and b in show.norm}
    for clip in args.clips:
        path = WORK / clip / "script.json"
        script = load_json(path)
        changed = 0
        measured = {i: f for c, i, _, f in (x for ls in show.lines.values() for x in ls) if c == clip}
        for i, line in enumerate(script["lines"]):
            base = show.base.get(line.get("voice"))
            if base not in refs or i not in measured or line.get("keep_voice"):
                continue
            a = arousal(measured[i], show.norm[base])
            cur = line["voice"]
            best = min(refs[base], key=lambda k: abs(a - refs[base][k]))
            if best != cur and abs(a - refs[base][cur]) - abs(a - refs[base][best]) >= HYSTERESIS:
                print(
                    f"  {clip} L{i:02d} {line['start']:7.2f}  {cur} -> {best}  (line {a:+.1f}; "
                    + ", ".join(f"{k} {r:+.1f}" for k, r in refs[base].items())
                    + f")  {line['text']}"
                )
                line["voice"] = best
                changed += 1
        if changed and args.apply:
            path.write_text(json.dumps(script, ensure_ascii=False, indent=1), encoding="utf-8")
        if changed:
            print(f"{clip}: {changed} lines {'moved' if args.apply else 'to move (run with --apply)'}")


def cmd_check(args):
    show = Show()
    wanted = set(clips_of(args.clips))
    rows = []
    for base, ls in show.lines.items():
        if base not in show.norm:
            continue
        pairs = []
        for clip, i, line, f in ls:
            if clip not in wanted:
                continue
            wav = take_wav(load_json(WORK / clip / "segments.json")["job_id"], line_id(line), langs(clip)[1])
            t = features(decode_audio(str(wav), sampling_rate=SR)) if wav.exists() else None
            if t:
                pairs.append((arousal(f, show.norm[base]), t, line["voice"]))
        if len(pairs) < 8:
            continue
        tn = norms([t for _, t, _ in pairs])
        orig = np.array([a for a, _, _ in pairs])
        take = np.array([arousal(t, tn) for _, t, _ in pairs])
        rho = np.corrcoef(np.argsort(np.argsort(orig)), np.argsort(np.argsort(take)))[0, 1]
        rows.append((base, len(pairs), rho))
        print(f"{base:10s} {len(pairs):3d} takes  register follows the original: rank correlation {rho:+.2f}")
        # Does each extra reference deliver its register? Its takes should stand
        # apart from the character's other takes the way its lines do in the original.
        keys = np.array([k for _, _, k in pairs])
        for v in show.fam[base][1:]:
            on = keys == v
            if on.sum() < 3 or (~on).sum() < 3:
                continue
            d_orig, d_take = orig[on].mean() - orig[~on].mean(), take[on].mean() - take[~on].mean()
            want = -1 if show.voices[v].get("register") == "calm" else 1
            verdict = (
                "working" if want * d_take >= 0.5 else f"not delivering its register: scripts/register.py drop {v}"
            )
            print(
                f"    {v}: its {on.sum()} lines stand {d_orig:+.2f} apart from the others in the original, "
                f"{d_take:+.2f} in the takes -> {verdict}"
            )
    if rows:
        print(f"all: {sum(n * r for _, n, r in rows) / sum(n for _, n, _ in rows):+.2f} (weighted by lines)")


def cmd_drop(args):
    """Retire an extra reference: its lines go back to the character's own voice
    (whose cached takes, where they exist, come back without synthesis)."""
    voices = load_json(WORK / "voices.json")
    base = voices.get(args.voice, {}).get("register_of")
    if not base:
        raise SystemExit(f"{args.voice} is not an extra reference (no register_of)")
    for clip in clips_of(None):
        path = WORK / clip / "script.json"
        script = load_json(path)
        n = 0
        for line in script["lines"]:
            if line.get("voice") == args.voice:
                line["voice"] = base
                n += 1
        if n:
            path.write_text(json.dumps(script, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{clip}: {n} lines back to {base}; run scripts/dub.py generate {clip}")
    update_voices(lambda voices: voices.pop(args.voice))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("clips", nargs="*")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("build")
    p.add_argument("voice")
    p.add_argument("band", choices=["calm", "raised"])
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("route")
    p.add_argument("clips", nargs="+")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(fn=cmd_route)
    p = sub.add_parser("check")
    p.add_argument("clips", nargs="*")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("drop")
    p.add_argument("voice")
    p.set_defaults(fn=cmd_drop)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
