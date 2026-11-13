"""Frozen QA evidence observes normal repairs without becoming a repair policy."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import core
import lineqa
import qa_triage as triage


class PartialReaderCacheTests(unittest.TestCase):
    def test_runtime_only_snapshot_resolves_offline_without_loading_a_model(self):
        """An installed reader need not cache unrelated model-card files."""
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshots" / "pinned-revision"
            snapshot.mkdir(parents=True)
            for filename in ("model.int8.onnx", "tokens.txt"):
                (snapshot / filename).write_bytes(b"cached runtime asset")

            def resolve(repo, *, cache_dir, allow_patterns=None, local_files_only=False):
                self.assertTrue(local_files_only, "Reader identity must not trigger downloads")
                if allow_patterns is None or "README.md" in allow_patterns:
                    raise OSError("Incomplete snapshot: model card was not cached")
                self.assertEqual(set(allow_patterns), {"model.int8.onnx", "tokens.txt"})
                return str(snapshot)

            with patch("huggingface_hub.snapshot_download", side_effect=resolve):
                reader = triage.Reader("sensevoice", "en")
            self.assertEqual(reader.snapshot, snapshot)
            self.assertEqual(reader.binding["revision"], "pinned-revision")
            self.assertIsNone(reader.model)
            self.assertEqual((reader.load_s, reader.hear_s), (0.0, 0.0))


def wave(path, seconds=1, silence=False, clip=False):
    """A finite test signal, with explicit silence/clipping when required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.zeros(round(seconds * 16000)) if silence else np.sin(np.arange(round(seconds * 16000)) * 0.05) * 0.2
    if clip:
        signal[500:550] = 1
    sf.write(path, signal, 16000)
    return path


def seed(folder, number, *, flag=False, duration=1.0, voice="speaker_a", audio=None, part="sample-01", index=None):
    """Build immutable fixture evidence through the public evidence schema."""
    folder.mkdir(parents=True, exist_ok=True)
    if audio is None:
        audio = wave(folder / "input.wav")
    archived = triage.archive(folder, audio, "audio")
    line = {"start": number * 3, "end": number * 3 + 2, "voice": voice, "ja": "はい", "text": "The box is ready."}
    context = {"part": part, "index": number if index is None else index, "line": line, "target_lang": "en"}
    inputs = {
        "context_sha256": triage.identity(context),
        "audio": archived,
        "settings": {},
        "model": {"revision": "primary-v1"},
    }
    oid = triage.identity(inputs)
    comp = triage.comparison(line["text"], "Something else." if flag else line["text"], "en")
    item = {
        "observation_id": oid,
        "inputs": inputs,
        "context": context,
        "occurrence_id": triage.identity([part, context["index"]]),
        "qa_round": 0,
        "mix": None,
        "raw_hearing": "Something else." if flag else line["text"],
        "comparison": comp,
        "semantic_risks": triage.semantic_risks(comp),
        "physical": {"notes": []},
        "source_speech_seconds": duration,
    }
    item["evidence_sha256"] = triage.identity(item)
    triage.save(folder / "observations" / (oid + ".json"), item)
    ids = triage.load(folder / "index.json", [])
    triage.save(folder / "index.json", ids + [oid])
    triage.save(folder / "protocol.json", {**triage.PROTOCOL, "protocol_sha256": triage.identity(triage.PROTOCOL)})
    return item


class DefectEvidenceTests(unittest.TestCase):
    def test_valid_number_readings_do_not_create_false_omissions(self):
        comp = triage.comparison("This was written in 1979.", "This was written in nineteen seventy nine.", "en")
        self.assertEqual(comp["notes"], [])
        self.assertEqual(comp["changes"], [])
        self.assertEqual(triage.semantic_risks(comp), [])

    def test_known_bad_hearings_remain_flags_and_meaning_risks_are_not_cleared(self):
        cases = [
            ("The box is ready.", "", "silent or unintelligible"),
            ("The small green box is ready.", "The box is ready.", "match"),
            ("The box is ready.", "Good morning. The box is ready.", "leading extra"),
            ("We need twenty boxes.", "We need thirty boxes.", "number differs"),
            ("I can open it.", "I can't open it.", "negation differs"),
        ]
        for expected, heard, reason in cases:
            with self.subTest(reason=reason):
                comp = triage.comparison(expected, heard, "en")
                self.assertTrue(any(reason in n for n in comp["notes"]), comp)
                item = {"comparison": comp, "physical": {"notes": []}, "semantic_risks": triage.semantic_risks(comp)}
                advice = triage.recommendation(item, [{"crop_kind": "isolated", "comparison": {"notes": []}}])
                self.assertEqual(advice["production_action"], "none; audit only")
                if "number" in reason or "negation" in reason or "leading" in reason:
                    self.assertIn("requires review", advice["category"])

    def test_high_overall_match_cannot_hide_omitted_clause_relationship_or_unit(self):
        expected = "Tomorrow morning I will bring all the small green boxes to your father at the station."
        for heard in (expected.replace("Tomorrow morning ", ""), expected.replace("father", "mother")):
            comp = triage.comparison(expected, heard, "en")
            self.assertGreater(comp["ratio"], 0.85)
            self.assertTrue(triage.semantic_risks(comp))
        comp = triage.comparison("The box weighs five kilograms.", "The box weighs five grams.", "en")
        self.assertIn("possible unit change", triage.semantic_risks(comp))

    def test_name_confused_with_an_ordinary_word_is_reviewed_even_above_global_match(self):
        expected = "Please ask Mira to bring the small green box to our door."
        comp = triage.comparison(expected, expected.replace("Mira", "mirror"), "en")
        self.assertGreater(comp["ratio"], 0.85)
        self.assertIn("possible name disagreement", triage.semantic_risks(comp))

    def test_silence_clipping_and_onset_crossings_are_physical_failures(self):
        import shorttakes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = {"lines": [{"start": 0, "end": 1.1}, {"start": 1.3, "end": 2}]}
            for label, path, spans, expected in [
                ("silence", wave(root / "silent.wav", silence=True), [], "no measured speech"),
                ("clip", wave(root / "clip.wav", clip=True), [[0, 1]], "clipped full-scale run"),
                ("onset", wave(root / "long.wav", seconds=1.6), [[0, 1.6]], "speech crosses safe end"),
            ]:
                with self.subTest(label=label), patch.object(shorttakes, "vad", return_value=spans):
                    evidence = triage.physical(path, script, 0, (0, 1))
                    self.assertTrue(any(expected in n for n in evidence["notes"]), evidence)


class FrozenEnrollmentTests(unittest.TestCase):
    def test_complete_stratified_enrollment_and_first_occurrence_are_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            first = [seed(folder, i, flag=True) for i in range(12)]
            for i in range(12, 24):
                seed(folder, i, duration=1 if i < 18 else 2, voice=f"speaker_{i % 3}")
            sample = triage.enroll(folder, freeze=True)
            self.assertEqual(sample["counts"], {"flagged": 12, "passing_short": 6, "passing_long": 6})
            self.assertEqual(len(sample["blind_observations"]), 6)
            self.assertEqual(sample["observations"][:12], [x["observation_id"] for x in first])
            self.assertEqual(sample["speaker_cap_relaxed"], [])
            amendment = seed(folder, 100, part="sample-01", index=0)
            self.assertNotIn(amendment["observation_id"], triage.enroll(folder)["observations"])
            self.assertEqual(sample, triage.enroll(folder))
            row = triage.report(folder)["cases"][0]
            self.assertEqual(
                row["subsequent_observations"],
                [{"observation_id": amendment["observation_id"], "context_changed": True}],
            )

    def test_sparse_final_relaxation_is_explicit_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for i in range(12):
                seed(folder, i, flag=True)
            for i in range(12, 24):
                seed(folder, i, duration=1 if i < 18 else 2)
            preview = triage.enroll(folder)
            self.assertEqual(preview["counts"]["passing_short"], 2)
            with self.assertRaisesRegex(ValueError, "sample incomplete"):
                triage.enroll(folder, freeze=True)
            sample = triage.enroll(folder, freeze=True, allow_incomplete=True)
            self.assertEqual(sample["counts"]["passing_short"], 6)
            self.assertEqual(sample["speaker_cap_relaxed"], ["passing_short", "passing_long"])

    def test_secondary_checks_require_freeze_and_cannot_expand_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            item = seed(folder, 0, flag=True)
            with self.assertRaisesRegex(ValueError, "freeze enrollment"):
                triage.hear_sample(folder, lambda *args: None, "fake")
            sample = triage.enroll(folder, True, True)
            self.assertEqual(sample["observations"], [item["observation_id"]])
            later = seed(folder, 1)
            self.assertNotIn(later["observation_id"], triage.enroll(folder)["observations"])


class AuditCacheTests(unittest.TestCase):
    def test_take_vad_reuses_the_waveform_and_rejects_modified_receipts(self):
        import shorttakes

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            audio = wave(folder / "take.wav")
            with patch.object(shorttakes, "vad", return_value=[[0, 1]]) as vad:
                first = triage.take_spans(folder, audio, triage.sha(audio))
                self.assertEqual(first, triage.take_spans(folder, audio, triage.sha(audio)))
                self.assertEqual(vad.call_count, 1)
                receipt = next((folder / "take-vad").glob("*.json"))
                value = triage.load(receipt)
                value["spans"] = [[0, 100]]
                triage.save(receipt, value)
                with self.assertRaisesRegex(ValueError, "VAD cache"):
                    triage.take_spans(folder, audio, triage.sha(audio))

    def reader(self, revision, calls):
        class FakeReader:
            binding = {"model": "reader", "revision": revision, "settings": {"initial_prompt": None}}
            load_s, hear_s = 0.0, 0.0

            def __call__(self, path):
                calls.append(path)
                return "The box is ready."

        return lambda kind, lang: FakeReader()

    def test_hearings_resume_by_audio_model_settings_but_reviews_bind_occurrences(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            one = seed(folder, 0)
            two = seed(folder, 1)
            triage.enroll(folder, True, True)
            calls = []
            first = triage.hear_sample(folder, self.reader("r1", calls), "fake")
            self.assertEqual(len(calls), 1)
            self.assertEqual(first["checks"], 2)
            readings = triage.secondary_readings(folder, one["observation_id"])
            decision = triage.record_decision(
                folder, one["observation_id"], "unresolved", "A spelling needs review.", readings
            )
            repeat = triage.hear_sample(folder, self.reader("r1", calls), "fake")
            self.assertEqual(repeat["cache_hits"], 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                decision["binding"],
                triage.decision_binding(one, triage.secondary_readings(folder, one["observation_id"])),
            )
            self.assertNotEqual(triage.decision_binding(one, []), triage.decision_binding(two, []))
            triage.hear_sample(folder, self.reader("r2", calls), "fake")
            self.assertEqual(len(calls), 2)
            report = triage.report(folder)
            self.assertFalse(report["cases"][0]["director_decisions"][0]["current_evidence"])

    def test_tampered_audio_hearing_and_enrollment_are_rejected(self):
        for target in ("audio", "hearing", "enrollment"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                item = seed(folder, 0)
                triage.enroll(folder, True, True)
                triage.hear_sample(folder, self.reader("r1", []), "fake")
                if target == "audio":
                    (folder / item["inputs"]["audio"]["path"]).write_bytes(b"changed")
                elif target == "hearing":
                    path = next((folder / "hearings").glob("*.json"))
                    value = triage.load(path)
                    value["raw_hearing"] = "Changed."
                    triage.save(path, value)
                else:
                    value = triage.load(folder / "enrollment.json")
                    value["observations"] = []
                    triage.save(folder / "enrollment.json", value)
                with self.assertRaises(ValueError):
                    triage.hear_sample(folder, self.reader("r1", []), "fake")

    def test_context_crop_is_bound_to_source_range_and_immutable_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            item = seed(folder, 0)
            item["context"]["line"].update(start=2, end=3)
            mixed = wave(folder / "mixed.wav", seconds=6)
            item["mix"] = triage.archive(folder, mixed, "mixes")
            first = triage.audit_inputs(folder, item)
            second = triage.audit_inputs(folder, item)
            self.assertEqual(first, second)
            self.assertEqual(first["contextual"]["crop"]["start"], 0)
            self.assertAlmostEqual(sf.info(folder / first["contextual"]["path"]).duration, 5, places=2)
            (folder / first["contextual"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "crop"):
                triage.audit_inputs(folder, item)


class NormalControlIntegrationTests(unittest.TestCase):
    def test_observed_fresh_reader_records_exact_resolved_weight_snapshot(self):
        with (
            patch.object(lineqa, "whisper_on_gpu", return_value=True),
            patch.object(lineqa, "hold_gpu_for_whisper"),
            patch(
                "faster_whisper.utils.download_model", return_value=str(Path("models/snapshots/pinned-revision"))
            ) as download,
            patch.object(lineqa, "WhisperModel") as model,
            patch.object(lineqa, "hear", return_value="The box is ready."),
        ):
            ear = lineqa.Ear()
            ear.record_provenance = True
            self.assertEqual(ear("clip.wav", "en"), "The box is ready.")
            download.assert_called_once_with(ear.MODEL)
            self.assertEqual(model.call_args.args[0], str(Path("models/snapshots/pinned-revision")))
            self.assertEqual(triage.model_identity(ear)["revision"], "pinned-revision")

    def test_capture_resume_and_invalidation_include_source_wording_voice_window_model_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work/sample-01"
            work.mkdir(parents=True)
            source = wave(root / "jobs/job/vocals.wav")
            wav = wave(root / "take.wav")
            segments = {"job_id": "job", "video": str(root / "source.mp4"), "source_lang": "ja", "target_lang": "en"}
            triage.save(work / "segments.json", segments)
            original = {"start": 0, "end": 1.5, "voice": "speaker_a", "ja": "はい", "text": "The box is ready."}

            def capture(line=None, model="r1", settings=None):
                with patch.object(triage, "model_identity", return_value={"model": "reader", "revision": model}):
                    store = triage.ObservationStore(
                        root / "study", "sample-01", work, segments, {"lines": [line or original]}, object()
                    )
                    return store.capture(0, wav, "The box is ready.", settings or {}, False, 0)

            with (
                patch.object(core, "WORK", root / "work"),
                patch.object(core, "JOBS", root / "jobs"),
                patch.object(core, "actor_spans", return_value=[(0, 1)]) as source_vad,
                patch.object(triage, "physical", return_value={"notes": []}) as physical,
                patch.object(triage, "take_spans", return_value=[[0, 1]]),
            ):
                first = capture()
                self.assertEqual(first, capture())
                self.assertEqual(source_vad.call_count, 1)
                self.assertEqual(physical.call_count, 1)
                changed = [
                    capture({**original, "end": 1.8}),
                    capture({**original, "voice": "speaker_b"}),
                    capture({**original, "ja": "そうです"}),
                    capture({**original, "text": "The crate is ready."}),
                    capture(model="r2"),
                    capture(settings={"guidance_scale": 3}),
                ]
                self.assertEqual(len({first, *changed}), 7)
                reference = wave(root / "work/voices/speaker_a.wav", seconds=0.7)
                ref_one = capture()
                wave(reference, seconds=0.8)
                self.assertNotEqual(ref_one, capture())
                mixed = root / "dubbed/sample-01.en.mp4"
                mixed.parent.mkdir()
                mixed.write_bytes(b"first contextual mix")
                mix_one = capture()
                mixed.write_bytes(b"changed contextual mix")
                self.assertNotEqual(mix_one, capture())
                wave(source, seconds=1.1)
                self.assertNotEqual(first, capture())

    def test_observation_archives_rejected_take_before_normal_best_prior_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work/sample-01"
            (work / "takes").mkdir(parents=True)
            source = wave(root / "jobs/job/vocals.wav")
            wav = wave(root / "current.wav", seconds=0.9)
            current_bytes = wav.read_bytes()
            prior = wave(work / "takes/prior.wav", seconds=1.0)
            line = {"start": 0, "end": 1.5, "voice": "speaker_a", "ja": "はい", "text": "The box is ready."}
            segments = {"job_id": "job", "video": str(root / "source.mp4"), "source_lang": "ja", "target_lang": "en"}
            triage.save(work / "segments.json", segments)
            triage.save(work / "script.json", {"lines": [line]})
            triage.save(
                work / "takes/kept.json",
                {
                    "line": [
                        {
                            "notes": [],
                            "ratio": 1,
                            "got": line["text"],
                            "sha": "prior",
                            "settings": {},
                            "file": "prior.wav",
                        }
                    ]
                },
            )

            class FakeEar:
                MODEL, tag, compute_type = "fake/reader", "fake", "int8"

                def __call__(self, path, lang):
                    return "Something else."

            with (
                patch.object(core, "WORK", root / "work"),
                patch.object(core, "WORK", root / "work"),
                patch.object(core, "JOBS", root / "jobs"),
                patch.object(lineqa, "take_wav", return_value=wav),
                patch.object(lineqa, "line_id", return_value="line"),
                patch.object(core, "actor_spans", return_value=[(0, 1)]),
                patch.object(triage, "physical", return_value={"notes": []}),
                patch.object(triage, "take_spans", return_value=[[0, 1]]),
                patch.object(triage, "model_identity", return_value={"model": "fake", "revision": "r1"}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                lineqa.check("sample-01", FakeEar(), False, True, observations=root / "study")
            self.assertEqual(wav.read_bytes(), prior.read_bytes())
            oid = triage.load(root / "study/index.json")[0]
            item = triage.read_observation(root / "study", oid)
            self.assertEqual((root / "study" / item["inputs"]["audio"]["path"]).read_bytes(), current_bytes)
            self.assertEqual(item["raw_hearing"], "Something else.")
            self.assertEqual(item["context"]["source_audio"]["sha256"], triage.sha(source))
            action = triage.load(next((root / "study/actions" / oid).glob("*.json")))["control"]
            self.assertTrue(action["restored_best_prior"])
            self.assertFalse(action["repair_requested"])
            self.assertEqual((root / "study" / action["result_audio"]["path"]).read_bytes(), prior.read_bytes())
            self.assertIsNone(triage.load(work / "reroll.json"))
            before = {str(p): triage.sha(p) for p in work.rglob("*") if p.is_file()}
            triage.enroll(root / "study", True, True)
            triage.hear_sample(root / "study", AuditCacheTests().reader("r1", []), "fake")
            triage.report(root / "study")
            self.assertEqual(before, {str(p): triage.sha(p) for p in work.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
