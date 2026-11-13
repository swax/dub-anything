"""Opt-in actor/OmniVoice short-take trial; normal dubbing never invokes this.

Usage: shorttakes.py PART --lines 0,3,8 --candidates 3 [--apply]
       shorttakes.py PART --lines 0,3,8 --candidates 3 --restore-baseline

Run only on a directed, unready part with a complete normal first render. The
baseline counts as candidate zero. Generate all further candidates before loading
Whisper, so its GPU lock cannot deadlock generation. Measurement restores the
baseline; --apply is a separate explicit installation of the measured choice.

Eligibility is first-to-last source VAD speech, not the container/window length.
Selection is transcription validity first, then endpoint error; equal results
prefer the baseline, otherwise the lower candidate ordinal. A timing-only change
must improve endpoint error by at least 100 ms. This is no claim about acting.
VAD speech end drives endpoint quality and underfill; an effect or noise tail
cannot fill missing speech. Padded duration must fit the normal concise room,
and all audible activity must stop before the protected speaker boundary.
Profiles have identical reference bytes/text and private persisted seeds. Their
fixed seeds disable native multi-row batching; report that cost, not a speedup.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

import core
import dub
import lineqa
from languages import LANGUAGES


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    dub.outcome_save(path, value)


def save_report(folder, action, report):
    save(folder / (action + ".json"), report)
    save(folder / "reports" / f"{time.time_ns()}-{action}.json", report)


@contextmanager
def locked(path):
    """One candidate/apply process per part; the part must also remain unready."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class Policy:
    version: int = 1
    max_actor_s: float = 1.5
    overflow_s: float = 0.3
    boundary_buffer_s: float = 0.05
    short_by_s: float = 0.5
    minimum_timing_gain_s: float = 0.1
    clip_run_ms: float = 0.5


def fixed_settings(settings):
    return {key: value for key, value in settings.items() if key not in ("seed", "speed", "seed_profile")}


def take_seed(signature, lid, ordinal, base, baseline_seed=None):
    """Stable distinct trial seeds, also avoiding a known seeded baseline."""
    start = (base + int(identity([signature, lid])[:8], 16)) % 2**31
    used = {baseline_seed}
    value = start
    for _ in range(ordinal):
        value = (value + 1) % 2**31
        while value in used:
            value = (value + 1) % 2**31
        used.add(value)
    return value


def render_room(script, index, policy):
    """Exactly the concise room used by core.fit_rows, before onset protection."""
    line = script["lines"][index]
    next_dub = script["lines"][index + 1]["start"] if index + 1 < len(script["lines"]) else math.inf
    gap = next_dub - line["end"]
    borrow = min(gap - 0.05, 0.25) if 0.05 < gap < math.inf else 0
    return line["end"] - line["start"] + borrow + policy.overflow_s


def safe_end(script, index, policy):
    line = script["lines"][index]
    next_dub = script["lines"][index + 1]["start"] if index + 1 < len(script["lines"]) else math.inf
    end = min(line["start"] + render_room(script, index, policy), next_dub - policy.boundary_buffer_s)
    for caption in script.get("captions", []):
        if caption["end"] > line["start"]:
            end = min(end, max(line["start"], caption["start"] - policy.boundary_buffer_s))
    return end


def vad(wav):
    """CPU Silero only; no transcription or GPU model is loaded here."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    x = decode_audio(str(wav), sampling_rate=16000)
    spans = get_speech_timestamps(x, VadOptions(threshold=0.5, min_silence_duration_ms=120, speech_pad_ms=0))
    return [[s["start"] / 16000, s["end"] / 16000] for s in spans]


def timing(wav, line, actor, limit, policy, speech=None, room=None):
    """Assembly-equivalent trim, actual VAD, clipping and protected-onset checks."""
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    notes = []
    if not len(x) or not np.isfinite(x).all():
        return {"notes": ["empty or nonfinite audio"], "endpoint_error_s": None}
    envelope = np.abs(x).max(axis=1)
    active = np.flatnonzero(envelope > 10 ** (-50 / 20))
    spans = vad(wav) if speech is None else speech
    if not len(active) or not spans:
        return {"notes": ["no measured speech"], "endpoint_error_s": None}
    margin = int(sr * 0.05)
    first = max(0, int(active[0]) - margin)
    last = min(len(x), int(active[-1]) + 1 + margin)
    trim_start = first / sr
    length = (last - first) / sr
    room = line["end"] - line["start"] + policy.overflow_s if room is None else room
    actual_end = line["start"] + max(b for _, b in spans) - trim_start
    audible_end = max(actual_end, line["start"] + (active[-1] + 1) / sr - trim_start)
    if audible_end > limit:
        notes.append("speech crosses safe end or protected onset")
    if length > room:
        notes.append("padded take exceeds normal concise room")
    if line["start"] + actor[1] - actual_end > policy.short_by_s:
        notes.append("speech ends too far before actor")
    longest = 0
    for channel in x.T:
        hot = np.abs(channel) >= 1 - 1 / 32768
        edges = np.flatnonzero(np.diff(np.r_[False, hot, False].astype(np.int8)))
        if len(edges):
            longest = max(longest, int((edges[1::2] - edges[::2]).max()))
    if longest / sr * 1000 >= policy.clip_run_ms:
        notes.append("clipped full-scale run")
    return {
        "notes": notes,
        "trim_start_s": float(trim_start),
        "trimmed_length_s": float(length),
        "actual_end": float(actual_end),
        "audible_end": float(audible_end),
        "endpoint_method": "VAD speech end; audible activity is gated separately",
        "safe_end": limit,
        "legal_room_s": room,
        "vad": spans,
        "peak": float(envelope.max()),
        "clipped_run_ms": longest / sr * 1000,
        "endpoint_error_s": abs(actual_end - line["start"] - actor[1]),
    }


def choose(candidates, policy):
    """Stable objective choice; a valid baseline survives unsuccessful extra takes."""
    valid = [c for c in candidates if not c["notes"]]
    if not valid:
        return {"ordinal": 0, "unresolved": True, "reason": "No valid take; baseline retained for director review"}
    best = min(valid, key=lambda c: (-c["ratio"], round(c["endpoint_error_s"], 6), c["ordinal"], c["sha256"]))
    baseline = next(c for c in candidates if c["ordinal"] == 0)
    if not baseline["notes"] and best["ordinal"]:
        ratio_gain = best["ratio"] - baseline["ratio"]
        timing_gain = baseline["endpoint_error_s"] - best["endpoint_error_s"]
        if ratio_gain < 0.001 and (ratio_gain < 0 or timing_gain < policy.minimum_timing_gain_s):
            best = baseline
    return {
        "ordinal": best["ordinal"],
        "sha256": best["sha256"],
        "unresolved": False,
        "reason": "Valid transcription/timing; baseline retained unless objectively improved",
    }


class Store:
    """Immutable candidate receipts plus an idempotent audio/metadata restore journal."""

    def __init__(self, folder, wav, rendered_path, lid):
        self.folder, self.wav, self.rendered_path, self.lid = Path(folder), Path(wav), Path(rendered_path), lid
        self.folder.mkdir(parents=True, exist_ok=True)

    def receipt(self, ordinal):
        path = self.folder / str(ordinal) / "candidate.json"
        result = core.load_json(path)
        if result and identity({k: v for k, v in result.items() if k != "receipt_sha256"}) != result.get(
            "receipt_sha256"
        ):
            raise ValueError("archived candidate receipt changed")
        if result and result.get("audio"):
            if sha(self.folder / result["audio"]) != result["sha256"]:
                raise ValueError("archived candidate audio changed")
        return result

    def settled(self):
        dub.require_settled_generation(self.rendered_path.parent)
        pending = core.load_json(self.folder / "pending.json", {})
        if pending:
            ticket = pending.get("generation")
            if not ticket:
                raise dub.UnknownTaskOutcome("Legacy pending child lacks terminal outcome proof")
            outcome = dub.outcome_read(self.rendered_path.parent, Path(ticket["path"]), ticket["attempt"])
            if not dub.outcome_terminal(outcome):
                raise dub.UnknownTaskOutcome(f"Child outcome is unconfirmed: {ticket['path']}")

    def capture(self, ordinal, seed, cost=None, error=None):
        self.settled()
        old = self.receipt(ordinal)
        if old:
            return old
        settings = core.load_json(self.rendered_path, {}).get(self.lid)
        if settings is None:
            raise ValueError("missing matching render metadata")
        folder = self.folder / str(ordinal)
        folder.mkdir(exist_ok=True)
        result = {"ordinal": ordinal, "seed": seed, "settings": settings, "cost": cost or {}, "error": error}
        profiles = core.load_json(self.rendered_path.parent / "shorttakes" / "profiles.json", {})
        if settings.get("seed_profile"):
            result["profile_receipt"] = next(
                (p for p in profiles.values() if p["id"] == settings["seed_profile"]), None
            )
        if isinstance(ordinal, int) and ordinal > 0 and not error:
            receipt = result.get("profile_receipt") or {}
            if (
                settings.get("seed") != seed
                or receipt.get("seed") != seed
                or receipt.get("transport") != "clone_profile_seed"
            ):
                result["error"] = "applied clone-profile seed provenance is missing or mismatched"
        if self.wav.exists() and (ordinal == 0 or settings.get("seed") == seed):
            target = folder / "audio.wav"
            shutil.copy2(self.wav, target)
            result.update(audio=str(target.relative_to(self.folder)), sha256=sha(target))
        result["receipt_sha256"] = identity(result)
        save(folder / "candidate.json", result)
        return result

    def restore(self, ordinal):
        self.settled()
        entry = self.receipt(ordinal)
        if not entry or not entry.get("audio"):
            raise ValueError("candidate has no archived audio")
        journal = self.folder / "restore.json"
        save(journal, {"ordinal": ordinal, "sha256": entry["sha256"], "complete": False})
        source = self.folder / entry["audio"]
        temp = self.wav.with_name(self.wav.name + ".shorttakes.tmp")
        shutil.copy2(source, temp)
        temp.replace(self.wav)
        rendered = core.load_json(self.rendered_path, {})
        rendered[self.lid] = entry["settings"]
        save(self.rendered_path, rendered)
        save(journal, {"ordinal": ordinal, "sha256": entry["sha256"], "complete": True})

    def recover(self):
        self.settled()
        journal = core.load_json(self.folder / "restore.json", {})
        if journal and not journal.get("complete"):
            self.restore(journal["ordinal"])

    def generate(self, ordinal, seed, generate):
        """Never regenerate a receipted take; recover a completed child after interruption."""
        self.settled()
        old = self.receipt(ordinal)
        if old:
            return old
        pending_path = self.folder / "pending.json"
        pending = core.load_json(pending_path, {})
        metrics_path = self.folder / str(ordinal) / "generation-cost.json"
        metrics = core.load_json(metrics_path, {})
        current = core.load_json(self.rendered_path, {}).get(self.lid, {})
        if pending.get("ordinal") == ordinal:
            ticket = pending["generation"]
            outcome = dub.outcome_read(self.rendered_path.parent, Path(ticket["path"]), ticket["attempt"])
            error = outcome.get("error")
            if not outcome.get("success") or current.get("seed") != seed:
                error = error or "completed child lacks matching candidate settings"
            result = self.capture(ordinal, seed, metrics, error)
            self.restore(0)
            return result
        self.restore(0)
        metrics_path.parent.mkdir(exist_ok=True)
        ticket = dub.GenerationOutcome.prepare(
            self.rendered_path.parent,
            metrics_path.with_name("task-outcome.json"),
            baseline={
                "receipt": str((self.folder / "0/candidate.json").resolve()),
                "sha256": sha(self.folder / "0/candidate.json"),
            },
        )
        save(pending_path, {"ordinal": ordinal, "seed": seed, "generation": ticket})
        started = time.perf_counter()
        error = None
        try:
            generate(seed, metrics_path)
        except Exception as exception:
            error = repr(exception)
        # A client exiting does not stop its backend writer. Unknown outcomes
        # escape before either capture or the baseline-restoration finally.
        self.settled()
        outcome = dub.outcome_read(self.rendered_path.parent, Path(ticket["path"]), ticket["attempt"])
        if not outcome.get("success"):
            error = error or outcome.get("error") or "terminal child failure"
        try:
            cost = core.load_json(metrics_path, {})
            cost["process_wall_s"] = time.perf_counter() - started
            return self.capture(ordinal, seed, cost, error)
        finally:
            self.restore(0)


def listen(wav, target, ear, heard_path):
    heard = core.load_json(heard_path, {})
    key = f"{ear.tag}:{target}:{lineqa.file_sha(wav)}"
    if key not in heard:
        heard[key] = ear(wav, target)
        save(heard_path, heard)
    return key, heard[key]


@contextmanager
def qa_cost(folder, ear):
    """Retain original and resumed QA costs instead of replacing them with cache hits."""
    started = time.perf_counter()
    try:
        yield
    finally:
        path = folder / "qa-cost-history.json"
        history = core.load_json(path, [])
        history.append(
            {
                "time_ns": time.time_ns(),
                "model_wait_load_s": ear.load_s,
                "transcription_s": ear.hear_s,
                "wall_s": time.perf_counter() - started,
                "model": ear.tag,
            }
        )
        save(path, history)
        save(
            folder / "qa-cost.json",
            {
                "runs": len(history),
                "model_wait_load_s": sum(r["model_wait_load_s"] for r in history),
                "transcription_s": sum(r["transcription_s"] for r in history),
                "wall_s": sum(r["wall_s"] for r in history),
            },
        )


def measure(store, entry, line, actor, limit, policy, target, ear, heard_path, room=None):
    if not entry.get("audio"):
        return {**entry, "notes": [entry.get("error") or "no candidate audio"], "ratio": 0, "endpoint_error_s": None}
    wav = store.folder / entry["audio"]
    evidence_path = store.folder / str(entry["ordinal"]) / "evidence.json"
    evidence_key = identity(
        {
            "receipt": entry["receipt_sha256"],
            "line": line,
            "actor": actor,
            "limit": limit,
            "legal_room_s": room,
            "policy": asdict(policy),
            "target": target,
            "model": ear.tag,
        }
    )
    old = core.load_json(evidence_path, {})
    if old.get("evidence_key") == evidence_key:
        return old
    checked = timing(wav, line, actor, limit, policy, room=room)
    key, got = listen(wav, target, ear, heard_path)
    notes, ratio = LANGUAGES[target].compare(line["text"], got)
    failures = checked["notes"] + notes + ([entry["error"]] if entry.get("error") else [])
    if fixed_settings(entry["settings"]) != fixed_settings(store.receipt(0)["settings"]):
        failures.append("candidate was made with different fixed render settings")
    result = {
        **entry,
        "timing": checked,
        "got": got,
        "hearing_key": key,
        "ratio": ratio,
        "notes": failures,
        "endpoint_error_s": checked["endpoint_error_s"],
        "evidence_key": evidence_key,
    }
    save(evidence_path, result)
    return result


def context(name, script, rendered, policy, candidates, seed, indices=None):
    work = core.part_dir(name)
    doc = core.load_json(work / "segments.json")
    references = {}
    for key in sorted({line["voice"] for line in script["lines"]}):
        references[key] = {"spec": core.voice_sig(key), "wav_sha256": sha(core.voice_wav(key))}
    line_references = {}
    voices = core.load_voices()
    source = core.langs(name)[0]
    for line in script["lines"]:
        lid = core.line_id(line)
        reference = rendered.get(lid, {}).get("line_ref")
        if not reference:
            continue
        path = work / "refs" / f"{lid}_{reference}.wav"
        if not path.is_file():
            raise ValueError(f"declared per-line reference is missing: {path}")
        line_references[lid] = {
            "wav_sha256": sha(path),
            "text": f"{voices[line['voice']].get('ref_text', '')} {core.said(line, source)}",
        }
    language_files = sorted((Path(__file__).parent / "languages").glob("*.py"))
    return {
        "script": {k: v for k, v in script.items() if k != "ready"},
        "job": doc["job_id"],
        "language_pair": core.langs(name),
        "source_sha256": sha(core.JOBS / doc["job_id"] / "vocals.wav"),
        "references": references,
        "line_references": line_references,
        "settings": {key: fixed_settings(value) for key, value in rendered.items()},
        "generation_policy": core.load_json(work / "generate-result.json", {}).get("client", {}).get("policy", {}),
        "policy": asdict(policy),
        "candidates": candidates,
        "seed_base": seed,
        "selected_indices": sorted(indices) if indices is not None else None,
        "implementation_sha256": identity(
            {
                p: sha(Path(__file__).parent / p)
                for p in ("shorttakes.py", "dub.py", "core.py", "lineqa.py", "whisper_safe.py")
            }
        ),
        "languages_sha256": identity({p.name: sha(p) for p in language_files}),
    }


def trial(
    name,
    indices,
    candidates=3,
    seed=1729,
    apply=False,
    restore=False,
    archive=False,
    control=False,
    apply_control=False,
    control_cost=None,
):
    if candidates not in (2, 3):
        raise ValueError("choose two or three total candidates, including the baseline")
    work = core.part_dir(name)
    with locked(work / "shorttakes.lock"):
        dub.require_settled_generation(work)
        script = core.load_json(work / "script.json")
        if not script or script.get("ready"):
            raise ValueError("trial needs a directed unready part; stop its normal rendering first")
        mode = script.get("render", {})
        if mode.get("voice", "actor") != "actor":
            raise ValueError("shorttakes currently supports actor/OmniVoice only")
        rendered_path = work / "rendered.json"
        rendered = core.load_json(rendered_path, {})
        doc = core.load_json(work / "segments.json")
        producer = core.load_json(work / "generate-result.json", {}).get("client", {})
        generation_policy = producer.get("policy", {})
        if producer.get("job_id") != doc["job_id"] or generation_policy.get("timing") != "concise":
            raise ValueError("trial needs a completed concise baseline with generation-policy provenance")
        if generation_policy.get("engine") != "omnivoice":
            raise ValueError("trial needs an OmniVoice baseline generation policy")
        overflow = generation_policy["overflow"]
        if not isinstance(overflow, (float, int)) or not math.isfinite(overflow) or overflow < 0:
            raise ValueError("invalid baseline overflow policy")
        policy = Policy(overflow_s=overflow)
        target = core.langs(name)[1]
        ids = [core.line_id(line) for line in script["lines"]]
        if not rendered or any(
            not core.take_wav(doc["job_id"], lid, target).exists() or lid not in rendered for lid in ids
        ):
            raise ValueError("trial needs a complete normally rendered baseline")
        if any(ids.count(ids[index]) != 1 for index in indices):
            raise ValueError("selected repeated lines need distinct take fields")
        if any(rendered[ids[index]].get("engine", "omnivoice") != "omnivoice" for index in indices):
            raise ValueError("trial needs OmniVoice baseline takes")
        frozen = context(name, script, rendered, policy, candidates, seed, indices)
        signature = identity(frozen)
        folder = work / "shorttakes" / signature
        save(folder / "context.json", frozen)
        spans = core.actor_spans(doc["job_id"], script["lines"])
        stores, eligibility = {}, []
        for index in sorted(indices):
            actor = spans[index]
            if not actor or not 0 < actor[1] - actor[0] <= policy.max_actor_s:
                eligibility.append(
                    {"index": index, "eligible": False, "reason": "source VAD span unavailable or over limit"}
                )
                continue
            eligibility.append({"index": index, "eligible": True, "actor_span": actor})
            lid = ids[index]
            store = Store(folder / lid, core.take_wav(doc["job_id"], lid, target), rendered_path, lid)
            store.recover()
            if (control or apply or apply_control or restore) and store.receipt(0) is None:
                raise ValueError("archive the common first baseline before control/apply/restore actions")
            if store.receipt(0) is None and core.load_json(work / "takes" / "kept.json", {}).get(lid):
                raise ValueError("normal QA repair history exists; cannot relabel its result as the common first take")
            store.capture(0, rendered[lid].get("seed"))
            stores[index] = store
        save(folder / "eligibility.json", eligibility)
        if archive or control:
            control_entries = []
            if control:
                selected_ids = {ids[i] for i in stores}
                if any(
                    r["id"] in selected_ids and not r.get("settled") for r in core.load_json(work / "reroll.json", [])
                ):
                    raise ValueError("normal control still has an uncompleted fresh-take request")
                ear = lineqa.Ear()
                heard = core.load_json(work / "heard.json", {})
                for index, store in stores.items():
                    current = core.load_json(rendered_path)[ids[index]]
                    key = f"{ear.tag}:{target}:{lineqa.file_sha(store.wav)}"
                    if key not in heard:
                        raise ValueError("run normal isolated QA before capturing control; this action never loads ASR")
                    entry = store.capture("control", current.get("seed"))
                    if fixed_settings(entry["settings"]) != fixed_settings(store.receipt(0)["settings"]):
                        raise ValueError("control wording/reference/render policy changed")
                    evidence = measure(
                        store,
                        entry,
                        script["lines"][index],
                        spans[index],
                        safe_end(script, index, policy),
                        policy,
                        target,
                        ear,
                        work / "heard.json",
                        room=render_room(script, index, policy),
                    )
                    control_entries.append({"index": index, "sha256": entry["sha256"], "evidence": evidence})
                if control_cost:
                    costs = core.load_json(control_cost)
                    if costs is None:
                        raise ValueError("control cost manifest does not exist")
                    save(
                        folder / "control-cost.json",
                        {"source": str(control_cost), "sha256": sha(control_cost), "costs": costs},
                    )
            report = {
                "part": name,
                "context": signature,
                "eligibility": eligibility,
                "arm": "control" if control else "common_first_take",
                "control_entries": control_entries,
                "unresolved": any(c["evidence"]["notes"] for c in control_entries),
                "cost_provided": bool(control_cost),
            }
            save_report(folder, "control-report" if control else "baseline-report", report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return report
        if restore:
            for store in stores.values():
                store.restore(0)
        elif not apply and not apply_control:
            for store in stores.values():
                store.restore(0)
            for index, store in stores.items():
                for ordinal in range(1, candidates):
                    candidate_seed = take_seed(signature, ids[index], ordinal, seed, store.receipt(0).get("seed"))

                    def generate(candidate_seed, metrics, index=index, store=store, ordinal=ordinal):
                        args = [
                            sys.executable,
                            str(core.HERE / "dub.py"),
                            "generate",
                            name,
                            "--only",
                            str(index),
                            "--seed",
                            str(candidate_seed),
                            "--steps",
                            str(store.receipt(0)["settings"]["steps"]),
                            "--effect",
                            store.receipt(0)["settings"]["effect"],
                            "--overflow",
                            str(policy.overflow_s),
                            "--no-export",
                            "--metrics",
                            str(metrics),
                            "--task-outcome",
                            str(metrics.with_name("task-outcome.json")),
                        ]
                        if generation_policy["match_level"]:
                            args.append("--match-level")
                        args += ["--voice-match", generation_policy["voice_match"]]
                        for key in ("fill", "line_ref", "predict_speed"):
                            flag = key.replace("_", "-")
                            args.append("--" + ("" if generation_policy["render_mode"][key] else "no-") + flag)
                        with (store.folder / str(ordinal) / "generation.log").open("w", encoding="utf-8") as log:
                            subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT)

                    store.generate(ordinal, candidate_seed, generate)
        decisions = []
        if apply_control:
            for index, store in stores.items():
                entry = store.receipt("control")
                if not entry:
                    raise ValueError("capture the normal-policy control before --apply-control")
                checked = core.load_json(store.folder / "control" / "evidence.json")
                if not checked or checked.get("sha256") != entry["sha256"]:
                    raise ValueError("control QA evidence is missing or stale")
                decisions.append(
                    {
                        "index": index,
                        "context": signature,
                        "ordinal": "control",
                        "unresolved": bool(checked["notes"]),
                        "sha256": entry["sha256"],
                        "reason": "Explicit installation of archived normal-policy result",
                    }
                )
        elif apply:
            for index, store in stores.items():
                decision = core.load_json(store.folder / "selection.json")
                if not decision or decision.get("context") != signature:
                    raise ValueError("measure candidates before --apply")
                entry = store.receipt(decision["ordinal"])
                if not entry or (not decision["unresolved"] and entry["sha256"] != decision["sha256"]):
                    raise ValueError("selection no longer matches the candidate")
                decisions.append(decision)
        elif not restore:
            ear = lineqa.Ear()
            with qa_cost(folder, ear):
                for index, store in stores.items():
                    entries = [store.receipt(n) for n in range(candidates)]
                    if any(entry is None for entry in entries):
                        raise ValueError("measure candidates before --apply")
                    evidence = [
                        measure(
                            store,
                            entry,
                            script["lines"][index],
                            spans[index],
                            safe_end(script, index, policy),
                            policy,
                            target,
                            ear,
                            work / "heard.json",
                            room=render_room(script, index, policy),
                        )
                        for entry in entries
                    ]
                    decision = {"index": index, "context": signature, **choose(evidence, policy)}
                    save(store.folder / "selection.json", decision)
                    decisions.append(decision)
        # Recheck the frozen source, wording, references and fixed settings before installation.
        now = context(
            name, core.load_json(work / "script.json"), core.load_json(rendered_path), policy, candidates, seed, indices
        )
        if identity(now) != signature:
            raise ValueError("trial inputs changed; candidates kept but installation refused")
        if apply or restore or apply_control:
            state_path = work / "state.json"
            state = core.load_json(state_path, {})
            if state:
                save(state_path, {**state, "stage": "render", "fails": 0})
            if apply or apply_control:
                for decision in decisions:
                    stores[decision["index"]].restore(decision["ordinal"])
        report = {
            "part": name,
            "context": signature,
            "eligibility": eligibility,
            "selections": decisions,
            "applied": apply or apply_control,
            "arm": "control" if apply_control else "trial",
            "restored_baseline": restore,
            "native_batching": "Seeded profiles render serially; no native multi-row batching benefit claimed",
            "listener_preference": "unmeasured",
            "unresolved": any(d["unresolved"] for d in decisions),
        }
        action = (
            "control-apply"
            if apply_control
            else ("trial-apply" if apply else ("baseline-restore" if restore else "trial-report"))
        )
        save_report(folder, action, report)
        save(folder / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--lines", required=True, help="Prospectively chosen zero-based comma-separated indices")
    parser.add_argument("--candidates", type=int, choices=(2, 3), default=3)
    parser.add_argument("--seed", type=int, default=1729, help="Stable experiment seed base")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="Install only a measured choice; no candidate generation")
    action.add_argument("--restore-baseline", action="store_true")
    action.add_argument(
        "--archive-baseline",
        action="store_true",
        help="Archive common first take before normal QA repairs; no generation",
    )
    action.add_argument(
        "--capture-control", action="store_true", help="Archive completed normal repair result; no generation"
    )
    action.add_argument("--apply-control", action="store_true", help="Install the archived normal repair result")
    parser.add_argument("--control-cost", type=Path, help="JSON manifest of normal-policy generation/QA cost receipts")
    args = parser.parse_args()
    script = core.load_json(core.part_dir(args.name) / "script.json")
    if not script:
        parser.error("missing directed script")
    indices = dub.selected_indices(args.lines, len(script["lines"]))
    result = trial(
        args.name,
        indices,
        args.candidates,
        args.seed,
        args.apply,
        args.restore_baseline,
        args.archive_baseline,
        args.capture_control,
        args.apply_control,
        args.control_cost,
    )
    if result["unresolved"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
