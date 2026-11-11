"""Observe normal QA and audit a frozen sample without changing production takes.

Enable capture with lineqa.py --observations DIR (or DUB_QA_OBSERVATIONS).
Then: qa_triage.py enroll DIR --freeze [--allow-incomplete]
      qa_triage.py hear DIR --reader sensevoice|parakeet|whisper
      qa_triage.py report DIR

The first normal hearing and pre-repair audio of every occurrence are retained.
Secondary checks are unprompted, bounded and separately cached. Recommendations
never clear QA, restore audio, queue synthesis or approve delivery. All evidence
and media belong in ignored work/. See docs/plans/qa-triage-pilot.md.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import core
from languages import LANGUAGES

PROTOCOL = {
    "version": 1,
    "flags": 12,
    "passing_short": 6,
    "passing_long": 6,
    "short_source_s": 1.5,
    "passing_selection": "recording-order round robin; at most two per speaker/stratum, relax only at explicit sparse final freeze",
    "blind_selection": "first four flagged occurrences, first short pass, first longer pass; available counts only",
    "secondary_budget": "one isolated and one contextual hearing per enrolled observation, reader and pinned settings",
    "policy": "observation only; primary repair policy unchanged; semantic and physical risks require director review",
}
DECODING = {"beam_size": 5, "vad_filter": False, "condition_on_previous_text": False, "initial_prompt": None}


def identity(value):
    """Stable binding for evidence, independent of dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha(path):
    """Hash actual file contents without loading a movie into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    """Atomically replace a private evidence document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path, default=None):
    """Read a saved evidence document, preserving an explicit missing value."""
    return core.load_json(Path(path), default)


@contextmanager
def locked(folder):
    """Serialize the observation index and immutable enrollment across processes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "writer.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def archive(folder, path, kind):
    """Retain content-addressed inputs before ordinary QA can replace them."""
    path, folder = Path(path), Path(folder)
    digest = sha(path)
    relative = Path(kind) / (digest + path.suffix.lower())
    target = folder / relative
    if target.exists():
        if sha(target) != digest:
            raise ValueError("archived input changed")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f".{os.getpid()}.{time.time_ns()}.tmp")
        shutil.copy2(path, temporary)
        if sha(temporary) != digest:
            raise ValueError("input changed while archiving")
        temporary.replace(target)
    return {"path": relative.as_posix(), "sha256": digest}


def package_version(name):
    """Record runtime package identity without importing a model."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_identity(ear):
    """Describe the configured primary reader and locally resolved HF revision.

    A legacy hearing cache does not establish its original weight revision. Its
    provenance remains explicitly limited rather than being relabeled as fresh.
    """
    repo = getattr(ear, "MODEL", None) or getattr(ear, "tag", type(ear).__name__)
    revision = None
    resolved = getattr(ear, "resolved_model_path", None)
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = (
            str(resolved)
            if isinstance(resolved, (str, Path))
            else try_to_load_from_cache(str(repo), "config.json", cache_dir=os.environ.get("HF_HUB_CACHE"))
        )
        if isinstance(cached, str) and "snapshots" in Path(cached).parts:
            bits = Path(cached).parts
            revision = bits[bits.index("snapshots") + 1]
    except (ImportError, ValueError, OSError):
        pass
    return {
        "model": str(repo),
        "revision": revision,
        "cache_tag": str(getattr(ear, "tag", "unknown")),
        "compute_type": str(getattr(ear, "compute_type", "unknown")),
        "faster_whisper": package_version("faster-whisper"),
        "ctranslate2": package_version("ctranslate2"),
        "decoding": DECODING,
    }


def normalized(text, lang):
    """Expose normalized tokens, in addition to preserving the raw hearing."""
    module = LANGUAGES[lang]
    return module.words(text) if hasattr(module, "words") else module.units(text)


def comparison(want, got, lang):
    """Keep normal QA reasons and alignment differences as separate evidence."""
    notes, ratio = LANGUAGES[lang].compare(want, got)
    reading_modes = None
    if lang in ("en", "es"):
        split = LANGUAGES[lang].words
        a, b, expected, heard = max(
            (
                (a, b, split(want, a), split(got, b))
                for a in ("plain", "pairs", "digits")
                for b in ("plain", "pairs", "digits")
            ),
            key=lambda row: difflib.SequenceMatcher(None, row[2], row[3]).ratio(),
        )
        reading_modes = {"expected": a, "heard": b}
    else:
        expected, heard = normalized(want, lang), normalized(got, lang)
    changes = [
        {"operation": tag, "expected": expected[a:b], "heard": heard[c:d]}
        for tag, a, b, c, d in difflib.SequenceMatcher(None, expected, heard).get_opcodes()
        if tag != "equal"
    ]
    name_changes = []
    if lang == "en":
        names = {
            token
            for match in re.finditer(r"\b[A-Z][a-z]+\b", want)
            if match.start() and want[: match.start()].rstrip()[-1:] not in (".", "!", "?")
            for token in normalized(match.group(), lang)
        }
        for change in changes:
            name_changes += sorted(names.intersection(change["expected"]) - set(change["heard"]))
    return {
        "notes": list(notes),
        "ratio": ratio,
        "expected": expected,
        "heard": heard,
        "changes": changes,
        "name_like_changes": name_changes,
        "reading_modes": reading_modes,
    }


def semantic_risks(comp, lang="en"):
    """Keep local omissions and meaning-bearing substitutions visible at high match.

    These are review prompts, never automatic diagnoses. A second recognizer
    agreeing with the intended text cannot dismiss a primary semantic risk.
    """
    risks = [n for n in comp["notes"] if "number" in n or "negation" in n or "leading extra" in n]
    if comp.get("name_like_changes"):
        risks.append("possible name disagreement")
    for change in comp["changes"]:
        if change["operation"] == "delete":
            risks.append("possible omitted words or clause")
        if lang == "en":
            left, right = set(change["expected"]), set(change["heard"])
            if (left | right) & {"mother", "father", "sister", "brother", "husband", "wife", "he", "she", "his", "her"}:
                risks.append("possible relationship or pronoun change")
            if {word.rstrip("s") for word in left | right} & {
                "centimeter",
                "millimeter",
                "meter",
                "kilometer",
                "kilogram",
                "gram",
                "mile",
                "percent",
                "hour",
                "minute",
                "second",
            }:
                risks.append("possible unit change")
    return sorted(set(risks))


def physical(wav, script, index, actor, speech=None):
    """Use existing CPU VAD and assembly-equivalent endpoint checks as evidence."""
    from shorttakes import Policy, safe_end, timing

    if actor is None:
        return {"notes": ["source speech span unresolved"], "source_unknown": True}
    policy = Policy()
    return timing(wav, script["lines"][index], actor, safe_end(script, index, policy), policy, speech=speech)


def source_spans(folder, segments, script, source_sha256):
    """Reuse CPU source VAD for unchanged source bytes, windows and detector settings."""
    lines = [{k: v for k, v in l.items() if k != "actor_end"} for l in script["lines"]]
    binding = {
        "source_sha256": source_sha256,
        "windows": [[l["start"], l["end"]] for l in lines],
        "vad": {"threshold": 0.5, "min_silence_duration_ms": 120, "speech_pad_ms": 30},
        "runtime": package_version("faster-whisper"),
    }
    path = Path(folder) / "source-vad" / (identity(binding) + ".json")
    previous = load(path)
    if previous is not None:
        if previous["binding"] != binding or previous["receipt_sha256"] != identity(
            {k: v for k, v in previous.items() if k != "receipt_sha256"}
        ):
            raise ValueError("source VAD cache changed")
        return previous["spans"]
    value = {"binding": binding, "spans": core.actor_spans(segments["job_id"], lines)}
    value["receipt_sha256"] = identity(value)
    save(path, value)
    return value["spans"]


def take_spans(folder, wav, audio_sha256):
    """Reuse CPU take VAD across occurrences and later changes to neighboring audio."""
    from shorttakes import vad

    binding = {
        "audio_sha256": audio_sha256,
        "vad": {"threshold": 0.5, "min_silence_duration_ms": 120, "speech_pad_ms": 0},
        "runtime": package_version("faster-whisper"),
    }
    path = Path(folder) / "take-vad" / (identity(binding) + ".json")
    previous = load(path)
    if previous is not None:
        if previous["binding"] != binding or previous["receipt_sha256"] != identity(
            {k: v for k, v in previous.items() if k != "receipt_sha256"}
        ):
            raise ValueError("take VAD cache changed")
        return previous["spans"]
    value = {"binding": binding, "spans": vad(wav)}
    value["receipt_sha256"] = identity(value)
    save(path, value)
    return value["spans"]


class ObservationStore:
    """Capture evidence around the existing control, without changing its decisions."""

    def __init__(self, folder, part, work, segments, script, ear):
        started = time.perf_counter()
        self.folder, self.part, self.work = Path(folder), part, Path(work)
        self.segments, self.script = segments, script
        self.voices = load(core.WORK / "voices.json", {})
        keys = {l["voice"] for l in script["lines"]}
        keys.update(script.get("render", {}).get("fallback_voices", {}).values())
        keys.update(l["fallback_voice"] for l in script["lines"] if l.get("fallback_voice"))
        self.voice_inputs = {}
        for key in keys:
            reference = core.WORK / "voices" / (key + ".wav")
            self.voice_inputs[key] = {
                "specification": self.voices.get(key),
                "sha256": sha(reference) if reference.exists() else None,
            }
        self.source, self.target = core.langs(part)
        self.model = model_identity(ear)
        self.ear, self.resolved_model = ear, getattr(ear, "resolved_model_path", None)
        self.protocol = {**PROTOCOL, "protocol_sha256": identity(PROTOCOL)}
        with locked(self.folder):
            old = load(self.folder / "protocol.json")
            if old is not None and old != self.protocol:
                raise ValueError("observation protocol changed; use a new study folder")
            if old is None:
                save(self.folder / "protocol.json", self.protocol)
        source = core.JOBS / segments["job_id"] / "vocals.wav"
        self.source_input = {"path": str(source), "sha256": sha(source)}
        # Source-duration strata use detected first-to-last speech, not a window
        # length or a directed actor_end override.
        self.actors = source_spans(self.folder, segments, script, self.source_input["sha256"])
        self.mix = None
        mixed = core.out_dir(segments) / f"{part}.{self.target}.mp4"
        if mixed.exists():
            self.mix = archive(self.folder, mixed, "mixes")
        self.qa_round = load(self.work / "state.json", {}).get("round")
        self.setup_wall_s = time.perf_counter() - started

    def capture(self, index, wav, got, settings, cached, hearing_seconds, primary_cost=None):
        """Archive the current input before best-prior restoration or reroll queuing."""
        started = time.perf_counter()
        resolved = getattr(self.ear, "resolved_model_path", None)
        if resolved and resolved != self.resolved_model:
            self.model = model_identity(self.ear)
            self.resolved_model = resolved
        line = self.script["lines"][index]
        actor = self.actors[index]
        audio = archive(self.folder, wav, "audio")
        context = {
            "part": self.part,
            "index": index,
            "line": line,
            "render": self.script.get("render"),
            "source_lang": self.source,
            "target_lang": self.target,
            "source_audio": self.source_input,
            "source_speech": actor,
            "following_onset": self.script["lines"][index + 1]["start"]
            if index + 1 < len(self.script["lines"])
            else None,
            "captions": self.script.get("captions", []),
            "mix": self.mix,
        }
        if not settings.get("line_ref") or line.get(
            "ref_prefix", self.script.get("render", {}).get("ref_prefix", True)
        ):
            selected = (
                settings.get("fallback_voice")
                or line.get("fallback_voice")
                or self.script.get("render", {}).get("fallback_voices", {}).get(line["voice"], line["voice"])
            )
            context["shared_reference"] = {"voice": selected, **self.voice_inputs.get(selected, {})}
        inputs = {"context_sha256": identity(context), "audio": audio, "settings": settings, "model": self.model}
        inputs["comparison_code_sha256"] = identity(
            [
                (path.name, sha(path))
                for path in (Path(LANGUAGES[self.target].__file__), Path(__file__).parent / "languages/latin.py")
            ]
        )
        oid = identity(inputs)
        path = self.folder / "observations" / (oid + ".json")
        previous = load(path)
        if previous is not None:
            read_observation(self.folder, oid)
            if previous["inputs"] != inputs or previous["raw_hearing"] != got:
                raise ValueError("same pinned observation produced conflicting evidence")
            return oid
        comp = comparison(line["text"], got, self.target)
        item = {
            "observation_id": oid,
            "inputs": inputs,
            "context": context,
            "occurrence_id": identity([self.part, index]),
            "qa_round": self.qa_round,
            "mix": self.mix,
            "raw_hearing": got,
            "comparison": comp,
            "semantic_risks": semantic_risks(comp, self.target),
            "physical": physical(
                wav,
                self.script,
                index,
                actor,
                speech=take_spans(self.folder, wav, audio["sha256"]) if actor is not None else None,
            ),
            "source_speech_seconds": None if actor is None else actor[1] - actor[0],
            "hearing_cache_hit": bool(cached),
            "primary_cache_provenance": "legacy cache; original revision not independently established"
            if cached
            else "current configured reader",
            "observed_hearing_s": hearing_seconds,
            "primary_cost": primary_cost,
            "cost_note": "hearing wall includes wait/load/transcription; overlapping layers must not be added as active GPU compute",
            "observer_setup_wall_s": self.setup_wall_s,
        }
        with locked(self.folder):
            previous = load(path)
            if previous is not None:
                if previous["inputs"] != inputs or previous["raw_hearing"] != got:
                    raise ValueError("same pinned observation produced conflicting evidence")
                return oid
            item["created_utc"] = datetime.now(timezone.utc).isoformat()
            item["capture_wall_s"] = time.perf_counter() - started
            item["evidence_sha256"] = identity(item)
            save(path, item)
            index_path = self.folder / "index.json"
            ids = load(index_path, [])
            save(index_path, ids + [oid])
        return oid

    def action(self, oid, outcome, wav):
        """Append the ordinary control action; repeated identical receipts are reused."""
        outcome = {**outcome, "result_audio": archive(self.folder, wav, "audio"), "qa_round": self.qa_round}
        value = {"observation_id": oid, "control": outcome}
        path = self.folder / "actions" / oid / (identity(value) + ".json")
        old = load(path)
        if old is not None and old != value:
            raise ValueError("control action receipt changed")
        if old is None:
            save(path, value)


def read_observation(folder, oid):
    """Reject tampered evidence or archived audio before an audit uses it."""
    folder = Path(folder)
    item = load(folder / "observations" / (oid + ".json"))
    if not item or identity({k: v for k, v in item.items() if k != "evidence_sha256"}) != item.get("evidence_sha256"):
        raise ValueError("observation evidence changed")
    if item["observation_id"] != oid or identity(item["inputs"]) != oid:
        raise ValueError("observation input binding changed")
    audio = item["inputs"]["audio"]
    if sha(folder / audio["path"]) != audio["sha256"]:
        raise ValueError("archived audio changed")
    return item


def balanced_passes(items, count, relax=False):
    """Round-robin speakers with a declared cap, explicitly relaxed only if sparse."""
    groups = {}
    for item in items:
        groups.setdefault(item["context"]["line"]["voice"], []).append(item)
    selected = []
    picked = {key: 0 for key in groups}
    while len(selected) < count and any(groups.values()):
        added = False
        for key, rows in groups.items():
            if rows and len(selected) < count and (relax or picked[key] < 2):
                selected.append(rows.pop(0))
                picked[key] += 1
                added = True
        if not added:
            break
    return selected


def enroll(folder, freeze=False, allow_incomplete=False):
    """Freeze first flags and stratified passes in source order before secondary hearing."""
    folder = Path(folder)
    with locked(folder):
        existing = load(folder / "enrollment.json")
        if existing:
            if (
                identity({k: v for k, v in existing.items() if k != "enrollment_sha256"})
                != existing["enrollment_sha256"]
            ):
                raise ValueError("frozen enrollment changed")
            return existing
        protocol = load(folder / "protocol.json")
        if protocol != {**PROTOCOL, "protocol_sha256": identity(PROTOCOL)}:
            raise ValueError("missing or mismatched declared protocol")
        first = {}
        for oid in load(folder / "index.json", []):
            item = read_observation(folder, oid)
            # Later repaired takes/wording cannot replace the first enrolled
            # occurrence; amendments remain in observations and control receipts.
            first.setdefault(item["occurrence_id"], item)
        items = sorted(first.values(), key=lambda x: (x["context"]["part"], x["context"]["index"]))
        flags = [x for x in items if x["comparison"]["notes"]][: PROTOCOL["flags"]]
        passing = [x for x in items if not x["comparison"]["notes"] and x["source_speech_seconds"] is not None]
        short_pool = [x for x in passing if x["source_speech_seconds"] <= PROTOCOL["short_source_s"]]
        long_pool = [x for x in passing if x["source_speech_seconds"] > PROTOCOL["short_source_s"]]
        short = balanced_passes(short_pool, PROTOCOL["passing_short"])
        longer = balanced_passes(long_pool, PROTOCOL["passing_long"])
        relaxed = []
        if freeze and allow_incomplete:
            for name, pool, selected, target in (
                ("passing_short", short_pool, short, 6),
                ("passing_long", long_pool, longer, 6),
            ):
                if len(selected) < target:
                    expanded = balanced_passes(pool, target, relax=True)
                    if len(expanded) > len(selected):
                        selected[:] = expanded
                        relaxed.append(name)
        counts = {"flagged": len(flags), "passing_short": len(short), "passing_long": len(longer)}
        complete = counts == {"flagged": 12, "passing_short": 6, "passing_long": 6}
        if freeze and not complete and not allow_incomplete:
            raise ValueError(
                f"sample incomplete: {counts}; finish primary QA or explicitly freeze actual sparse counts"
            )
        selected = flags + short + longer
        blind = flags[:4] + short[:1] + longer[:1]
        result = {
            "protocol": protocol,
            "counts": counts,
            "complete": complete,
            "frozen": freeze,
            "observations": [x["observation_id"] for x in selected],
            "blind_observations": [x["observation_id"] for x in blind],
            "observed_occurrences": len(items),
            "speaker_cap_relaxed": relaxed,
            "missing_source_speech_spans": sum(x["source_speech_seconds"] is None for x in items),
        }
        result["enrollment_sha256"] = identity(result)
        if freeze:
            save(folder / "enrollment.json", result)
        return result


def audit_inputs(folder, item):
    """Materialize immutable isolated and contextual crops for the frozen baseline."""
    folder = Path(folder)
    result = {"isolated": item["inputs"]["audio"]}
    if not item.get("mix"):
        return result
    mix = item["mix"]
    if sha(folder / mix["path"]) != mix["sha256"]:
        raise ValueError("archived contextual mix changed")
    line = item["context"]["line"]
    crop = {
        "source_sha256": mix["sha256"],
        "start": max(0, line["start"] - 2),
        "end": line["end"] + 2,
        "sample_rate": 16000,
        "channels": 1,
    }
    path = folder / "crops" / (identity(crop) + ".wav")
    receipt_path = path.with_suffix(".json")
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(crop["start"]), "-i", str(folder / mix["path"]),
                        "-t", str(crop["end"] - crop["start"]), "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(path)], check=True)  # fmt: skip
        save(receipt_path, {"sha256": sha(path), "crop": crop})
    receipt = load(receipt_path)
    if receipt is None or receipt["crop"] != crop or receipt["sha256"] != sha(path):
        raise ValueError("contextual crop or its receipt changed")
    result["contextual"] = {"path": path.relative_to(folder).as_posix(), **receipt}
    return result


class Reader:
    """Lazy unprompted CPU secondary reader; no TTS backend is contacted."""

    def __init__(self, kind, lang):
        self.kind, self.lang, self.model = kind, lang, None
        self.load_s, self.hear_s = 0.0, 0.0
        if kind == "whisper":
            self.repo = "Systran/faster-whisper-large-v3"
            self.settings = {"device": "cpu", "compute_type": "int8", **DECODING}
        else:
            self.repo = LANGUAGES[lang].SECOND_OPINIONS[kind]
            self.settings = {
                "device": "cpu",
                "family": kind,
                "expected_text_prompt": None,
                "sample_rate": 16000,
                "threads": 8,
            }
        from huggingface_hub import snapshot_download

        # Local-only resolution does not download models just to calculate a key.
        # The normal second-opinion setup caches only runtime assets. Newer hub
        # clients reject an unfiltered partial snapshot even when those assets
        # are present, so request the same required subset when resolving it.
        assets = {
            "sensevoice": ["model.int8.onnx", "tokens.txt"],
            "parakeet": ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"],
            "whisper": ["model.bin", "config.json", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"],
        }[kind]
        self.snapshot = Path(
            snapshot_download(
                self.repo, cache_dir=core.default_cache_dir(), allow_patterns=assets, local_files_only=True
            )
        )
        self.binding = {
            "model": self.repo,
            "revision": self.snapshot.name,
            "settings": self.settings,
            "runtime_version": package_version("faster-whisper" if kind == "whisper" else "sherpa-onnx"),
        }

    def __call__(self, audio):
        """Hear only the audio; the intended text is never passed to the reader."""
        if self.model is None:
            started = time.perf_counter()
            if self.kind == "whisper":
                core.whisper_setup()
                from whisper_safe import WhisperModel

                self.model = WhisperModel(str(self.snapshot), device="cpu", compute_type="int8")
            else:
                import sherpa_onnx

                # Use the resolved immutable snapshot directly; resolving main a
                # second time could silently load other weights under this key.
                if self.kind == "sensevoice":
                    self.model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                        model=str(self.snapshot / "model.int8.onnx"),
                        tokens=str(self.snapshot / "tokens.txt"),
                        num_threads=8,
                        language=self.lang,
                        use_itn=True,
                    )
                else:
                    self.model = sherpa_onnx.OfflineRecognizer.from_transducer(
                        encoder=str(self.snapshot / "encoder.int8.onnx"),
                        decoder=str(self.snapshot / "decoder.int8.onnx"),
                        joiner=str(self.snapshot / "joiner.int8.onnx"),
                        tokens=str(self.snapshot / "tokens.txt"),
                        num_threads=8,
                        sample_rate=16000,
                        feature_dim=128,
                        model_type="nemo_transducer",
                        decoding_method="greedy_search",
                    )
            self.load_s += time.perf_counter() - started
        started = time.perf_counter()
        if self.kind == "whisper":
            rows, _ = self.model.transcribe(
                str(audio), language=self.lang, **{k: v for k, v in DECODING.items() if k != "initial_prompt"}
            )
            text = LANGUAGES[self.lang].JOIN.join(r.text.strip() for r in rows)
        else:
            import numpy as np
            import soundfile as sf
            from scipy.signal import resample_poly

            data, rate = sf.read(audio, dtype="float32", always_2d=True)
            samples = data.mean(axis=1)
            if rate != 16000:
                samples = resample_poly(samples, 16000, rate).astype(np.float32)
            stream = self.model.create_stream()
            stream.accept_waveform(16000, samples)
            self.model.decode_stream(stream)
            text = stream.result.text
        self.hear_s += time.perf_counter() - started
        return text


def hear_sample(folder, reader_factory, kind):
    """Run at most one cached hearing per frozen input/model/settings binding."""
    folder = Path(folder)
    enrolled = enroll(folder)
    if not enrolled.get("frozen"):
        raise ValueError("freeze enrollment before secondary checks")
    readers, results = {}, []
    for oid in enrolled["observations"]:
        item = read_observation(folder, oid)
        lang = item["context"]["target_lang"]
        if lang not in readers:
            readers[lang] = reader_factory(kind, lang)
        reader = readers[lang]
        for crop_kind, audio in audit_inputs(folder, item).items():
            binding = {"audio_sha256": audio["sha256"], "reader": reader.binding, "language": lang}
            hearing_key = identity(binding)
            path = folder / "hearings" / (hearing_key + ".json")
            receipt = load(path)
            cached = receipt is not None
            if receipt is None:
                got = reader(folder / audio["path"])
                receipt = {"binding": binding, "raw_hearing": got}
                receipt["receipt_sha256"] = identity(receipt)
                save(path, receipt)
            elif receipt["binding"] != binding or receipt["receipt_sha256"] != identity(
                {k: v for k, v in receipt.items() if k != "receipt_sha256"}
            ):
                raise ValueError("hearing cache binding changed")
            # A waveform hearing can be reused; a comparison/review stays bound
            # to this occurrence, expected wording and context.
            record = {
                "observation_id": oid,
                "context_sha256": item["inputs"]["context_sha256"],
                "crop_kind": crop_kind,
                "audio": audio,
                "hearing_key": hearing_key,
                "hearing": receipt,
                "cache_hit": cached,
                "comparison_scope": "isolated target"
                if crop_kind == "isolated"
                else "whole context, including neighboring speech; director localizes target",
                "comparison": comparison(item["context"]["line"]["text"], receipt["raw_hearing"], lang),
            }
            record["evidence_sha256"] = identity({k: v for k, v in record.items() if k != "cache_hit"})
            save(folder / "secondary" / oid / (identity([crop_kind, reader.binding]) + ".json"), record)
            results.append(record)
    cost = {
        "reader": kind,
        "model_load_s": sum(r.load_s for r in readers.values()),
        "hearing_s": sum(r.hear_s for r in readers.values()),
        "cache_hits": sum(x["cache_hit"] for x in results),
        "checks": len(results),
        "queue_wait_s": 0.0,
        "active_gpu_compute_s": None,
        "note": "CPU audit; action wall time is not active GPU compute",
    }
    save(folder / "costs" / f"{time.time_ns()}-{kind}.json", cost)
    return cost


def recommendation(item, readings):
    """Classify evidence conservatively without certifying speech or choosing a take."""
    physical_notes = item["physical"].get("notes", [])
    risks = item["semantic_risks"]
    isolated = [r for r in readings if r["crop_kind"] == "isolated"]
    if item["physical"].get("source_unknown"):
        category = "unresolved source speech evidence"
    elif physical_notes:
        category = "physical timing or signal defect requires review"
    elif risks:
        category = "suspected word or meaning defect requires review"
    elif not item["comparison"]["notes"]:
        category = "passing control; review for missed defects"
    elif any(not r["comparison"]["notes"] for r in isolated):
        category = "probable recognition disagreement; director confirmation required"
    else:
        category = "unresolved"
    return {
        "category": category,
        "physical_reasons": physical_notes,
        "semantic_risks": risks,
        "production_action": "none; audit only",
        "listener_preference": None,
        "director_verdict": None,
    }


def decision_binding(item, readings):
    """An occurrence review includes wording/source/window and secondary settings."""
    return identity(
        {
            "observation": item["observation_id"],
            "inputs": item["inputs"],
            "secondary": sorted(identity({k: v for k, v in r.items() if k != "cache_hit"}) for r in readings),
        }
    )


def secondary_readings(folder, oid):
    """Verify saved secondary evidence before a report or review uses it."""
    folder = Path(folder)
    item = read_observation(folder, oid)
    readings = []
    for path in sorted((folder / "secondary" / oid).glob("*.json")):
        value = load(path)
        bound = {k: v for k, v in value.items() if k not in ("cache_hit", "evidence_sha256")}
        if (
            identity(bound) != value.get("evidence_sha256")
            or value["observation_id"] != oid
            or value["context_sha256"] != item["inputs"]["context_sha256"]
        ):
            raise ValueError("secondary evidence binding changed")
        readings.append(value)
    return readings


def record_decision(folder, oid, verdict, note, readings):
    """Save a director's evidence judgment separately from listener preference.

    ``no_defect_found`` records a bounded negative review, especially for sampled
    passing controls; it does not establish acting quality or certify speech.
    This approves no production take. A changed hearing/model/context binding
    leaves the prior decision archived and marks it stale in the audit report.
    """
    if (
        verdict not in {"supported_defect", "probable_recognition_issue", "no_defect_found", "unresolved"}
        or not note.strip()
    ):
        raise ValueError("a supported verdict and explanatory note are required")
    item = read_observation(folder, oid)
    value = {"observation_id": oid, "binding": decision_binding(item, readings), "verdict": verdict, "note": note}
    save(Path(folder) / "decisions" / oid / (identity(value) + ".json"), value)
    return value


def report(folder):
    """Write a review queue; never interpret recognizer agreement as approval."""
    folder = Path(folder)
    enrolled = enroll(folder)
    if not enrolled.get("frozen"):
        raise ValueError("freeze enrollment before reporting triage")
    all_items = [read_observation(folder, oid) for oid in load(folder / "index.json", [])]
    by_occurrence = {}
    for item in all_items:
        by_occurrence.setdefault(item["occurrence_id"], []).append(item)
    rows = []
    for oid in enrolled["observations"]:
        item = read_observation(folder, oid)
        readings = secondary_readings(folder, oid)
        decisions = [load(p) for p in sorted((folder / "decisions" / oid).glob("*.json"))]
        binding = decision_binding(item, readings)
        for decision in decisions:
            decision["current_evidence"] = decision["binding"] == binding
        amendments = []
        for other in by_occurrence[item["occurrence_id"]]:
            other_id = other["observation_id"]
            if other_id != oid:
                if other["occurrence_id"] == item["occurrence_id"]:
                    amendments.append(
                        {
                            "observation_id": other_id,
                            "context_changed": other["inputs"]["context_sha256"] != item["inputs"]["context_sha256"],
                        }
                    )
        rows.append(
            {
                "observation": item,
                "secondary": readings,
                "recommendation": recommendation(item, readings),
                "director_decisions": decisions,
                "subsequent_observations": amendments,
                "control_actions": [load(p) for p in sorted((folder / "actions" / oid).glob("*.json"))],
            }
        )
    value = {
        "enrollment_sha256": enrolled["enrollment_sha256"],
        "cases": rows,
        "scope": "observational; avoided repairs are counterfactual, not measured GPU savings",
    }
    save(folder / "triage-report.json", value)
    return value


def main():
    """Small command surface for frozen enrollment and non-mutating checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("enroll", "hear", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--reader", choices=("whisper", "sensevoice", "parakeet"), default="sensevoice")
    args = parser.parse_args()
    if args.action == "enroll":
        result = enroll(args.folder, args.freeze, args.allow_incomplete)
    elif args.action == "hear":
        result = hear_sample(args.folder, Reader, args.reader)
    else:
        value = report(args.folder)
        result = {"cases": len(value["cases"]), "path": str(args.folder / "triage-report.json")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
