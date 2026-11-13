"""Rank clean conversational fallback references near four seconds.

Usage: neutral_refs.py VOICE PART [...] --out REPORT.json
       ... --key NEW_VOICE [--pick PART:START:END] [--benchmark AUDIO.wav]

Source checks remain authoritative. Ranking is a speaker-relative acoustic
heuristic, not an emotion classifier. Building creates a separate voice key;
use render.fallback_voices to keep own-line conditioning and casting unchanged.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

FLOORS = {"pitch": 1.0, "effort": 1.0, "level": 1.5, "pitch_range": 1.0, "level_range": 1.5, "rate": 0.5}
WEIGHTS = {"pitch": 1.0, "effort": 1.0, "level": 0.25, "pitch_range": 0.5, "level_range": 0.25, "rate": 0.5}


def baseline(features):
    """Robust within-speaker centers and spreads, with finite-value filtering."""
    result = {}
    for key, floor in FLOORS.items():
        values = [f[key] for f in features if f.get(key) is not None and math.isfinite(f[key])]
        if not values:
            continue
        center = statistics.median(values)
        spread = max(floor, 1.4826 * statistics.median(abs(v - center) for v in values))
        result[key] = (center, spread)
    return result


def score(features, duration, norms, benchmark=None):
    """Independent feature distances prevent opposite extremes cancelling."""
    deviations = {}
    for key, (center, spread) in norms.items():
        value = features.get(key)
        if value is None or not math.isfinite(value):
            continue
        target = (benchmark or {}).get(key, center)
        if target is None or not math.isfinite(target):
            target = center
        deviations[key] = abs(value - target) / spread
    if any(key not in deviations for key in ("pitch", "effort", "level")):
        return None
    weighted = sum(WEIGHTS[k] * z for k, z in deviations.items()) / sum(WEIGHTS[k] for k in deviations)
    extreme = max(deviations.values())
    distance = weighted + max(0.0, extreme - 2.0) ** 2 + 0.2 * abs(duration - 4.0)
    return {"distance": distance, "deviations": deviations, "largest_deviation": extreme}


def measure(audio, text, language):
    """Existing register features plus robust variation and a rough text rate."""
    import numpy as np

    import register

    result = register.features(audio)
    if result is None:
        return None
    count = 1 + (len(audio) - register.FRAME) // register.HOP
    frames = audio[np.arange(register.FRAME)[None] + register.HOP * np.arange(count)[:, None]].astype(float)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    active = db > db.max() - 25
    f0 = register.yin(frames[active])
    f0 = f0[np.isfinite(f0)]
    if len(f0) < 5:
        return None
    pitch = 12 * np.log2(f0 / 100)
    result["pitch_range"] = float(np.percentile(pitch, 90) - np.percentile(pitch, 10))
    result["level_range"] = float(np.percentile(db[active], 90) - np.percentile(db[active], 10))
    # Compare source-language characters, with Japanese readings normalized by
    # the existing kana worker. This remains a relative pace proxy, not ASR.
    from languages import LANGUAGES

    if language == "ja":
        from languages.kana import readings

        text = readings([text])[0]
    units = (
        "".join(c for c in text if c not in LANGUAGES[language].UNIT_PUNCT)
        if language == "ja"
        else LANGUAGES[language].units(text)
    )
    result["rate"] = len(units) / max(float(active.sum()) * register.HOP / register.SR, 0.1)
    return {k: float(v) for k, v in result.items()}


def collect(voice, parts):
    """Get complete single-speaker windows already accepted by source checks."""
    import core
    import draft
    from languages import LANGUAGES

    voices = core.load_voices()
    family = voices[voice].get("register_of", voice)
    pool, candidates = [], []
    for name in parts:
        folder = core.part_dir(name)
        script = core.load_json(folder / "script.json")
        if not script:
            continue
        lines = script["lines"]
        job = core.load_json(folder / "segments.json")["job_id"]
        checks = core.unclean_windows(folder, job, lines, [core.line_id(l) for l in lines], dips=True, cast_lines=True)
        audio = draft.decode_audio(str(core.JOBS / job / "vocals.wav"), sampling_rate=draft.SR)
        clean = []
        for index, (line, (why, cut)) in enumerate(zip(lines, checks)):
            same = voices[line["voice"]].get("register_of", line["voice"]) == family
            if not same or why or line.get("line_ref") is False or line.get("effect") in core.ROOM_EFFECTS:
                continue
            end = cut or line["end"]
            feature = measure(
                audio[int(line["start"] * draft.SR) : int(end * draft.SR)],
                core.said(line, core.langs(name)[0]),
                core.langs(name)[0],
            )
            if feature:
                pool.append(feature)
                clean.append((index, line, end, feature))
        for position, (index, line, end, feature) in enumerate(clean):
            first_end = end
            group = [line]
            for next_index, following, following_end, _ in clean[position + 1 :]:
                if next_index != index + len(group) or following["start"] - end > 0.08:
                    break
                if following_end - line["start"] > 5.0:
                    break
                group.append(following)
                end = following_end
            # Keep individual candidates too, then a merged complete phrase.
            variants = [(first_end, [line])]
            if len(group) > 1:
                variants.append((end, group))
            for stop, utterances in variants:
                duration = stop - line["start"]
                if not 3.0 <= duration <= 5.0:
                    continue
                if any(c["start"] < stop and c["end"] > line["start"] for c in script.get("captions", [])):
                    continue
                text = LANGUAGES[core.langs(name)[0]].JOIN.join(core.said(l, core.langs(name)[0]) for l in utterances)
                if any(mark in text for mark in ("…", "...", "!", "！", "（笑", "[laughter]")):
                    continue
                sample = audio[int(line["start"] * draft.SR) : int(stop * draft.SR)]
                feature = measure(sample, text, core.langs(name)[0])
                if feature:
                    candidates.append(
                        dict(
                            part=name,
                            start=line["start"],
                            end=stop,
                            duration=duration,
                            text=text,
                            features=feature,
                            source_checks="Each constituent window passed source checks; complete adjacent same-speaker phrases only.",
                        )
                    )
    return pool, candidates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("voice")
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--benchmark", type=Path, help="listener-approved same-speaker reference used as an acoustic target"
    )
    ap.add_argument(
        "--pick", help="director-selected eligible PART:START:END (bounds within 25ms of checked candidate)"
    )
    ap.add_argument("--key", help="create this separate voice key from the selected candidate")
    args = ap.parse_args()
    import core
    import draft

    pool, candidates = collect(args.voice, args.parts)
    if len(pool) < 8 or not candidates:
        raise SystemExit("need at least eight clean source phrases and one complete 3–5s candidate")
    norms = baseline(pool)
    target = None
    if args.benchmark:
        target = measure(
            draft.decode_audio(str(args.benchmark), sampling_rate=draft.SR), "", core.langs(args.parts[0])[0]
        )
        # A reference path does not carry its transcript; do not use a zero rate.
        if target:
            target.pop("rate", None)
    for candidate in candidates:
        candidate["ranking"] = score(candidate["features"], candidate["duration"], norms, target)
    candidates = sorted((c for c in candidates if c["ranking"]), key=lambda c: c["ranking"]["distance"])
    if not candidates:
        raise SystemExit("no candidates with reliable pitch and complete feature measurements")
    picked = candidates[0]
    if args.pick:
        name, a, b = args.pick.rsplit(":", 2)
        a, b = float(a), float(b)
        matches = [
            c for c in candidates if c["part"] == name and abs(c["start"] - a) <= 0.025 and abs(c["end"] - b) <= 0.025
        ]
        if not matches:
            raise SystemExit("selected bounds must match a source-checked complete candidate")
        picked = min(matches, key=lambda c: abs(c["start"] - a) + abs(c["end"] - b))
    report = dict(
        voice=args.voice,
        target_seconds=4,
        range_seconds=[3, 5],
        baseline_phrases=len(pool),
        baseline=norms,
        benchmark=str(args.benchmark) if args.benchmark else None,
        candidates=candidates,
        selected=picked,
        method="Speaker-relative acoustic shortlist; no neutrality probability. Director/listener review remains required.",
    )
    if args.key:
        from voices import add

        voices = core.load_voices()
        family = voices[args.voice].get("register_of", args.voice)
        from register import cer

        job = core.load_json(core.part_dir(picked["part"]) / "segments.json")["job_id"]
        source = draft.decode_audio(str(core.JOBS / job / "vocals.wav"), sampling_rate=draft.SR)
        sample = source[int(picked["start"] * draft.SR) : int(picked["end"] * draft.SR)]
        draft.hold_gpu_for_whisper()
        ear = draft.WhisperModel(
            "Systran/faster-whisper-large-v3",
            device="auto",
            compute_type="default" if draft.whisper_on_gpu() else "int8",
        )
        heard = "".join(p["said"] for p in draft.transcribe(ear, sample, language=core.langs(picked["part"])[0]))
        error = cer(core.bare(picked["text"]), core.bare(heard))
        embed = draft.embedder()
        vector = embed(sample)
        identity = {}
        for key, voice in voices.items():
            if voice.get("register_of", key) == family and voice.get("clips"):
                match = embed(draft.decode_audio(str(core.voice_wav(key)), sampling_rate=draft.SR))
                identity[key] = float(vector @ match)
        report["verification"] = {"heard": heard, "cer": error, "family_reference_similarity": identity}
        if error > 0.25 or not identity or max(identity.values()) < 0.5:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise SystemExit("selected source needs director review: transcript or speaker verification failed")
        spec = {
            "from": picked["part"],
            "clips": [[picked["start"], picked["end"]]],
            "ref_text": picked["text"],
            "register_of": family,
            "register": "conversation",
            "reference_selection": "neutral4",
        }
        if args.key in voices:
            if voices[args.key] != spec:
                raise SystemExit("key already exists with a different reference; create a new key")
        else:
            add(args.key, picked["part"], json.dumps(spec["clips"]), picked["text"], False)
            core.update_voices(
                lambda values: values[args.key].update(
                    {k: v for k, v in spec.items() if k not in ("from", "clips", "ref_text")}
                )
            )
        path = core.voice_wav(args.key, rebuild=True)
        report["built"] = dict(
            key=args.key,
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            seconds=picked["duration"],
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"selected": picked, "built": report.get("built"), "report": str(args.out)}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
