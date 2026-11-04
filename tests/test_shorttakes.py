"""Candidate safety, objective selection, cache invalidation and crash recovery."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import shorttakes as st


def candidate(ordinal, error=0.2, notes=None, ratio=1):
    return {"ordinal": ordinal, "endpoint_error_s": error, "notes": notes or [], "ratio": ratio, "sha256": str(ordinal)}


class SelectionTests(unittest.TestCase):
    def test_seed_sequence_is_distinct_stable_and_avoids_known_baseline(self):
        first = st.take_seed("context", "L0", 1, 1729)
        second = st.take_seed("context", "L0", 2, 1729)
        self.assertNotEqual(first, second)
        self.assertEqual(first, st.take_seed("context", "L0", 1, 1729))
        self.assertNotEqual(st.take_seed("context", "L0", 1, 1729, first), first)
        self.assertNotEqual(
            st.take_seed("context", "L0", 1, 1729, first), st.take_seed("context", "L0", 2, 1729, first)
        )

    def test_invalid_timing_cannot_beat_valid_baseline(self):
        rows = [candidate(0), candidate(1, 0, ["protected onset"])]
        self.assertEqual(st.choose(rows, st.Policy())["ordinal"], 0)

    def test_no_valid_candidate_keeps_baseline_unresolved(self):
        result = st.choose([candidate(0, notes=["wrong words"]), candidate(1, notes=["silent"])], st.Policy())
        self.assertTrue(result["unresolved"])
        self.assertEqual(result["ordinal"], 0)

    def test_small_timing_gain_is_not_an_improvement(self):
        self.assertEqual(st.choose([candidate(0, 0.2), candidate(1, 0.11)], st.Policy())["ordinal"], 0)
        self.assertEqual(st.choose([candidate(0, 0.2), candidate(1, 0.05)], st.Policy())["ordinal"], 1)

    def test_ties_are_stable_and_prefer_baseline(self):
        rows = [candidate(2), candidate(1), candidate(0)]
        self.assertEqual(st.choose(rows, st.Policy())["ordinal"], 0)
        rows = [candidate(0, notes=["silent"]), candidate(2, 0.1), candidate(1, 0.1)]
        self.assertEqual(st.choose(rows, st.Policy())["ordinal"], 1)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wav = self.root / "current.wav"
        self.wav.write_bytes(b"baseline")
        self.rendered = self.root / "rendered.json"
        self.baseline = {"steps": 32, "effect": "broadcast", "line_ref": "ref-hash"}
        st.save(self.rendered, {"L0": self.baseline, "L1": {"untouched": True}})
        self.store = st.Store(self.root / "archive", self.wav, self.rendered, "L0")
        self.store.capture(0, None)

    def generate(self, seed, metrics):
        outcome = st.dub.GenerationOutcome(
            self.root, st.core.load_json(self.store.folder / "pending.json")["generation"]
        )
        self.wav.write_bytes(b"candidate")
        data = st.core.load_json(self.rendered)
        data["L0"] = {**self.baseline, "seed": seed, "seed_profile": "private", "speed": 0.92}
        st.save(self.rendered, data)
        st.save(
            self.root / "shorttakes" / "profiles.json",
            {"receipt": {"id": "private", "seed": seed, "transport": "clone_profile_seed"}},
        )
        st.save(metrics, {"completed": True})
        outcome.finish()

    def test_completed_resume_never_generates_again_and_restores_baseline(self):
        generate = Mock(side_effect=self.generate)
        first = self.store.generate(1, 123, generate)
        second = self.store.generate(1, 123, generate)
        self.assertEqual(first, second)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(self.wav.read_bytes(), b"baseline")
        self.assertEqual(st.core.load_json(self.rendered)["L0"], self.baseline)

    def test_completed_child_is_recovered_after_parent_interruption(self):
        metrics = self.store.folder / "1" / "generation-cost.json"
        ticket = st.dub.GenerationOutcome.prepare(self.root, metrics.with_name("task-outcome.json"))
        st.save(self.store.folder / "pending.json", {"ordinal": 1, "seed": 123, "generation": ticket})
        self.generate(123, metrics)
        never = Mock(side_effect=AssertionError("must reuse completed child"))
        entry = self.store.generate(1, 123, never)
        self.assertEqual(entry["seed"], 123)
        never.assert_not_called()
        self.assertEqual(self.wav.read_bytes(), b"baseline")

    def test_failed_attempt_is_bounded_and_preserves_baseline(self):
        def terminal_failure(seed, metrics):
            outcome = st.dub.GenerationOutcome(
                self.root, st.core.load_json(self.store.folder / "pending.json")["generation"]
            )
            request = outcome.start_request({})
            outcome.task(request, "failed-task")
            outcome.terminal(request, {"type": "error", "error": "generation failed"})
            error = RuntimeError("generation failed")
            outcome.finish(error)
            raise error

        fail = Mock(side_effect=terminal_failure)
        result = self.store.generate(1, 123, fail)
        self.assertIn("generation failed", result["error"])
        self.store.generate(1, 123, fail)
        self.assertEqual(fail.call_count, 1)
        self.assertEqual(self.wav.read_bytes(), b"baseline")

    def test_restore_pairs_audio_and_exact_metadata_and_preserves_other_rows(self):
        self.store.generate(1, 123, self.generate)
        self.store.restore(1)
        self.assertEqual(self.wav.read_bytes(), b"candidate")
        data = st.core.load_json(self.rendered)
        self.assertEqual(data["L0"], {**self.baseline, "seed": 123, "seed_profile": "private", "speed": 0.92})
        self.assertEqual(data["L1"], {"untouched": True})

    def test_restore_journal_repairs_crash_between_audio_and_metadata(self):
        self.store.generate(1, 123, self.generate)
        original_save = st.save

        def fail_metadata(path, value):
            if Path(path) == self.rendered:
                raise OSError("interrupted metadata write")
            original_save(path, value)

        with patch.object(st, "save", side_effect=fail_metadata):
            with self.assertRaises(OSError):
                self.store.restore(1)
        self.assertFalse(st.core.load_json(self.store.folder / "restore.json")["complete"])
        self.store.recover()
        self.assertEqual(self.wav.read_bytes(), b"candidate")
        self.assertEqual(st.core.load_json(self.rendered)["L0"]["seed"], 123)

    def test_changed_archive_or_metadata_is_not_restored(self):
        entry = self.store.generate(1, 123, self.generate)
        (self.store.folder / entry["audio"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.store.restore(1)
        self.assertEqual(self.wav.read_bytes(), b"baseline")

    def pending_child(self):
        ticket = st.core.load_json(self.store.folder / "pending.json")["generation"]
        return st.dub.GenerationOutcome(self.root, ticket)

    def test_unknown_child_never_captures_restores_recovers_or_advances(self):
        baseline = (self.store.folder / "0" / "candidate.json").read_bytes()
        original_audio = (self.store.folder / "0" / "audio.wav").read_bytes()

        def uncertain(seed, metrics):
            outcome = self.pending_child()
            request = outcome.start_request({})
            outcome.task(request, "still-running")
            self.wav.write_bytes(b"partial candidate")
            st.save(self.rendered, {"L0": {"seed": seed, "partial": True}})
            st.save(metrics, {"completed": True})  # A stale/request-level success cannot authorize restore.
            outcome.finish(RuntimeError("SSE lost"))
            raise RuntimeError("SSE lost")

        callback = Mock(side_effect=uncertain)
        with self.assertRaises(st.dub.UnknownTaskOutcome):
            self.store.generate(1, 123, callback)
        self.assertEqual(callback.call_count, 1)
        self.assertFalse((self.store.folder / "1" / "candidate.json").exists())
        self.assertEqual((self.store.folder / "0" / "candidate.json").read_bytes(), baseline)
        self.assertEqual((self.store.folder / "0" / "audio.wav").read_bytes(), original_audio)
        self.assertEqual(self.wav.read_bytes(), b"partial candidate")
        self.assertTrue(st.core.load_json(self.rendered)["L0"]["partial"])
        st.save(self.store.folder / "restore.json", {"ordinal": 0, "complete": False})
        for action in [
            lambda: self.store.capture(1, 123),
            lambda: self.store.restore(0),
            self.store.recover,
            lambda: self.store.generate(1, 123, callback),
            lambda: self.store.generate(2, 456, callback),
        ]:
            with self.subTest(action=action), self.assertRaises(st.dub.UnknownTaskOutcome):
                action()
        self.assertEqual(callback.call_count, 1)
        # All command entry points guard before context, capture, QA or apply.
        with patch.object(st.core, "WORK", self.root.parent), patch.object(st.lineqa, "Ear") as ear:
            for flags in [
                {},
                {"archive": True},
                {"control": True},
                {"apply": True},
                {"restore": True},
                {"apply_control": True},
            ]:
                with self.subTest(flags=flags), self.assertRaises(st.dub.UnknownTaskOutcome):
                    st.trial(self.root.name, {0}, **flags)
            ear.assert_not_called()
        self.assertEqual(self.wav.read_bytes(), b"partial candidate")
        pointer = st.core.load_json(self.root / "shorttakes" / "active-generation.json")
        receipt = st.dub.outcome_read(self.root, Path(pointer["path"]))
        self.assertEqual(receipt["baseline"]["sha256"], st.sha(self.store.folder / "0" / "candidate.json"))

    def test_completed_request_does_not_authorize_interrupted_command_recovery(self):
        def interrupted(seed, metrics):
            outcome = self.pending_child()
            request = outcome.start_request({})
            outcome.task(request, "first-request")
            outcome.terminal(request, {"type": "done"})
            st.save(metrics, {"completed": True})
            self.wav.write_bytes(b"current candidate")
            # Simulate process death after one request; no whole-command finish.

        with self.assertRaises(st.dub.UnknownTaskOutcome):
            self.store.generate(1, 123, interrupted)
        never = Mock()
        with self.assertRaises(st.dub.UnknownTaskOutcome):
            self.store.generate(1, 123, never)
        never.assert_not_called()
        self.assertEqual(self.wav.read_bytes(), b"current candidate")

    def test_missing_and_legacy_outcomes_are_not_successful_metrics(self):
        st.save(self.store.folder / "pending.json", {"ordinal": 1, "seed": 123})
        st.save(self.store.folder / "1" / "generation-cost.json", {"completed": True})
        with self.assertRaises(st.dub.UnknownTaskOutcome):
            self.store.restore(0)
        missing = self.root / "missing.json"
        st.save(
            self.store.folder / "pending.json",
            {"ordinal": 1, "seed": 123, "generation": {"path": str(missing), "attempt": "missing"}},
        )
        with self.assertRaises(st.dub.UnknownTaskOutcome):
            self.store.generate(1, 123, Mock())

    def test_cancel_ack_and_malformed_terminal_receipt_do_not_authorize_restore(self):
        ticket = st.dub.GenerationOutcome.prepare(self.root)
        outcome = st.dub.GenerationOutcome(self.root, ticket)
        request = outcome.start_request({})
        outcome.task(request, "task")
        request.update(event={"cancel_requested": True})
        outcome.finish(RuntimeError("cancel accepted; task has not stopped"))
        with self.assertRaises(st.dub.UnknownTaskOutcome):
            self.store.restore(0)
        request.update(status="done", event={"type": "cancelled"})
        outcome.doc.update(phase="finished")
        outcome.save()
        with self.assertRaises(st.dub.UnknownTaskOutcome):
            self.store.restore(0)
        # Only explicit terminal cancellation and the finished command permit mutation.
        outcome.terminal(request, {"type": "cancelled"})
        outcome.finish(RuntimeError("cancelled"))
        self.store.restore(0)

    def test_meaning_changing_negation_and_numbers_are_rejected(self):
        for want, got in [("I can go.", "I can't go."), ("I have two.", "I have three.")]:
            with self.subTest(want=want):
                ear = Mock(tag="fake", return_value=got)
                with patch.object(st, "timing", return_value={"notes": [], "endpoint_error_s": 0.1}):
                    result = st.measure(
                        self.store,
                        self.store.receipt(0),
                        {"text": want},
                        (0, 1),
                        1,
                        st.Policy(),
                        "en",
                        ear,
                        self.root / (str(len(want)) + "-heard.json"),
                    )
                self.assertTrue(result["notes"])

    def test_hearing_cache_is_audio_and_model_specific(self):
        path = self.root / "heard.json"
        ear = Mock(tag="model-a", return_value="Yes.")
        st.listen(self.wav, "en", ear, path)
        st.listen(self.wav, "en", ear, path)
        self.assertEqual(ear.call_count, 1)
        self.wav.write_bytes(b"different audio")
        st.listen(self.wav, "en", ear, path)
        self.assertEqual(ear.call_count, 2)
        other = Mock(tag="model-b", return_value="Yes.")
        st.listen(self.wav, "en", other, path)
        self.assertEqual(other.call_count, 1)


class TimingTests(unittest.TestCase):
    def test_reverb_tail_does_not_make_early_speech_a_better_take(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for ordinal, speech_end in enumerate((0.85, 0.55)):
                sr = 16000
                x = np.zeros(2 * sr)
                first, last, tail = round(0.15 * sr), round(speech_end * sr), round((speech_end + 0.5) * sr)
                x[first:last] = 0.2 * np.sin(2 * np.pi * 200 * np.arange(last - first) / sr)
                x[last:tail] = 0.01 * np.sin(2 * np.pi * 200 * np.arange(tail - last) / sr)
                wav = Path(tmp) / f"take-{ordinal}.wav"
                sf.write(wav, x, sr)
                measured = st.timing(
                    wav, {"start": 0, "end": 1.6}, (0.1, 0.8), 1.8, st.Policy(), [[0.15, speech_end]], room=1.9
                )
                self.assertFalse(measured["notes"])
                self.assertGreater(measured["audible_end"], measured["actual_end"] + 0.4)
                rows.append(candidate(ordinal, measured["endpoint_error_s"], measured["notes"]))
            self.assertEqual(st.choose(rows, st.Policy())["ordinal"], 0)

    def test_non_speech_tail_cannot_hide_underfill_or_cross_a_protected_onset(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            x = np.zeros(2 * sr)
            x[1600:4800] = 0.2 * np.sin(2 * np.pi * 200 * np.arange(3200) / sr)
            x[4800:19200] = 0.01 * np.sin(2 * np.pi * 200 * np.arange(14400) / sr)
            wav = Path(tmp) / "speech-and-tail.wav"
            sf.write(wav, x, sr)
            measured = st.timing(wav, {"start": 0, "end": 1.6}, (0.05, 1), 1.6, st.Policy(), [[0.1, 0.3]], room=1.9)
            self.assertIn("speech ends too far before actor", measured["notes"])
            self.assertGreater(measured["audible_end"], 1)
            self.assertLess(measured["actual_end"], 0.3)
            protected = st.timing(wav, {"start": 0, "end": 1.6}, (0.05, 1), 1, st.Policy(), [[0.1, 0.3]], room=1.9)
            self.assertIn("speech crosses safe end or protected onset", protected["notes"])

    def test_padded_overflow_is_rejected_even_when_audible_end_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "speech.wav"
            x = np.zeros(3 * 16000)
            n = round(1.22 * 16000)
            x[800 : 800 + n] = 0.2 * np.sin(2 * np.pi * 200 * np.arange(n) / 16000)
            sf.write(wav, x, 16000)
            line = {"start": 0, "end": 1}
            result = st.timing(wav, line, (0.05, 1.2), 1.3, st.Policy(), [[0.05, 1.27]], room=1.3)
            self.assertLess(result["actual_end"], 1.3)
            self.assertGreater(result["trimmed_length_s"], 1.3)
            self.assertLess(result["trimmed_length_s"], 1.4)  # old +100ms incorrectly accepted this take
            self.assertEqual(result["trimmed_length_s"], st.core.speech_length(wav))
            self.assertEqual(result["legal_room_s"], 1.3)
            self.assertIn("padded take exceeds normal concise room", result["notes"])

    def test_padding_beyond_protected_buffer_is_distinct_from_audible_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "speech.wav"
            x = np.zeros(2 * 16000)
            n = round(0.99 * 16000)
            x[800 : 800 + n] = 0.2 * np.sin(2 * np.pi * 200 * np.arange(n) / 16000)
            sf.write(wav, x, 16000)
            script = {
                "lines": [{"start": 0, "end": 1}, {"start": 2, "end": 3}],
                "captions": [{"start": 1.1, "end": 1.3}],
            }
            room = st.render_room(script, 0, st.Policy())
            end = st.safe_end(script, 0, st.Policy())
            result = st.timing(wav, script["lines"][0], (0.05, 1), end, st.Policy(), [[0.05, 1.04]], room=room)
            self.assertGreater(result["trimmed_length_s"], end)
            self.assertLessEqual(result["trimmed_length_s"], room)
            self.assertLess(result["actual_end"], end)
            self.assertFalse(result["notes"])

    def test_measures_speech_not_container_and_protects_retained_onset(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "speech.wav"
            sr = 16000
            x = np.zeros(2 * sr)
            x[800:17600] = 0.2 * np.sin(2 * np.pi * 200 * np.arange(16800) / sr)
            sf.write(wav, x, sr)
            line = {"start": 0, "end": 1.2}
            good = st.timing(wav, line, (0.05, 1.1), 1.6, st.Policy(), [[0.05, 1.1]])
            self.assertFalse(good["notes"])
            self.assertLess(good["trimmed_length_s"], 1.3)
            script = {"lines": [line, {"start": 2, "end": 3}], "captions": [{"start": 1.1, "end": 1.5}]}
            limit = st.safe_end(script, 0, st.Policy())
            self.assertAlmostEqual(limit, 1.05)
            bad = st.timing(wav, line, (0.05, 1.1), limit, st.Policy(), [[0.05, 1.1]])
            self.assertTrue(bad["notes"])

    def test_silent_clipped_and_non_speech_takes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "speech.wav"
            for values, spans in [(np.zeros(16000), [[0, 1]]), (np.ones(16000), [[0, 1]]), (np.full(16000, 0.1), [])]:
                sf.write(wav, values, 16000)
                self.assertTrue(st.timing(wav, {"start": 0, "end": 1}, (0, 1), 1.3, st.Policy(), spans)["notes"])


class ContextTests(unittest.TestCase):
    def line_reference_context(self, root):
        work = root / "sample"
        (work / "refs").mkdir(parents=True)
        (root / "jobs" / "job").mkdir(parents=True)
        st.save(work / "segments.json", {"job_id": "job", "source_lang": "ja", "target_lang": "en"})
        st.save(root / "voices.json", {"speaker_a": {"ref_text": "Base reference."}})
        (root / "jobs" / "job" / "vocals.wav").write_bytes(b"source")
        ref = root / "reference.wav"
        ref.write_bytes(b"reference")
        script = {"lines": [{"voice": "speaker_a", "text": "Hello.", "ja": "Original words.", "start": 0, "end": 1}]}
        return work, ref, script, {"L0": {"steps": 32, "line_ref": "0.00-1.00_vref"}}

    def test_existing_per_line_reference_bytes_and_exact_text_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, ref, script, settings = self.line_reference_context(root)
            own = work / "refs" / "L0_0.00-1.00_vref.wav"
            own.write_bytes(b"first per-line reference")
            with (
                patch.object(st.core, "WORK", root),
                patch.object(st.core, "JOBS", root / "jobs"),
                patch.object(st.core, "voice_sig", return_value="spec"),
                patch.object(st.core, "voice_wav", return_value=ref),
                patch.object(st.core, "line_id", return_value="L0"),
                patch.object(st.core, "line_refs") as rebuild,
            ):
                first = st.context("sample", script, settings, st.Policy(), 3, 1729)
                self.assertEqual(first["line_references"]["L0"]["text"], "Base reference. Original words.")
                own.write_bytes(b"changed bytes at the same path")
                self.assertNotEqual(
                    st.identity(first), st.identity(st.context("sample", script, settings, st.Policy(), 3, 1729))
                )
                rebuild.assert_not_called()

    def test_missing_declared_per_line_reference_fails_without_rebuilding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, ref, script, settings = self.line_reference_context(root)
            with (
                patch.object(st.core, "WORK", root),
                patch.object(st.core, "JOBS", root / "jobs"),
                patch.object(st.core, "voice_sig", return_value="spec"),
                patch.object(st.core, "voice_wav", return_value=ref),
                patch.object(st.core, "line_id", return_value="L0"),
                patch.object(st.core, "line_refs") as rebuild,
            ):
                with self.assertRaisesRegex(ValueError, "declared per-line reference is missing"):
                    st.context("sample", script, settings, st.Policy(), 3, 1729)
                settings["L0"].pop("line_ref")
                fallback = st.context("sample", script, settings, st.Policy(), 3, 1729)
                self.assertEqual(fallback["line_references"], {})
                rebuild.assert_not_called()

    def test_wording_window_settings_and_reference_changes_invalidate_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample").mkdir()
            (root / "jobs" / "job").mkdir(parents=True)
            st.save(root / "sample" / "segments.json", {"job_id": "job"})
            (root / "jobs" / "job" / "vocals.wav").write_bytes(b"source")
            ref = root / "reference.wav"
            ref.write_bytes(b"reference")
            script = {"lines": [{"voice": "speaker_a", "text": "Hello.", "start": 0, "end": 1}]}
            settings = {"L0": {"steps": 32, "seed": 10, "speed": 1}}
            with (
                patch.object(st.core, "WORK", root),
                patch.object(st.core, "JOBS", root / "jobs"),
                patch.object(st.core, "voice_sig", return_value="spec"),
                patch.object(st.core, "voice_wav", return_value=ref),
            ):

                def get():
                    return st.identity(st.context("sample", script, settings, st.Policy(), 3, 1729))

                baseline = get()
                settings["L0"]["seed"], settings["L0"]["speed"] = 20, 0.9
                self.assertEqual(get(), baseline)
                for key, value in [("text", "Goodbye."), ("end", 1.2)]:
                    old = script["lines"][0][key]
                    script["lines"][0][key] = value
                    self.assertNotEqual(get(), baseline)
                    script["lines"][0][key] = old
                settings["L0"]["steps"] = 16
                self.assertNotEqual(get(), baseline)
                settings["L0"]["steps"] = 32
                ref.write_bytes(b"new reference with same spec")
                self.assertNotEqual(get(), baseline)


class TrialIntegrationTests(unittest.TestCase):
    def test_common_baseline_control_trial_resume_and_explicit_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, job = root / "sample", root / "jobs" / "job"
            work.mkdir()
            job.mkdir(parents=True)
            source = job / "vocals.wav"
            ref = root / "reference.wav"
            source.write_bytes(b"frozen source")
            ref.write_bytes(b"frozen reference")
            wav = job / "seg_en_L0.wav"

            def write_audio(end):
                x = np.zeros(16000)
                n = round((end - 0.05) * 16000)
                x[800 : 800 + n] = 0.2 * np.sin(2 * np.pi * 200 * np.arange(n) / 16000)
                sf.write(wav, x, 16000)

            write_audio(0.95)
            baseline_audio = wav.read_bytes()
            settings = {"steps": 32, "effect": "broadcast", "instruct": ""}
            mode = {
                "voice": "actor",
                "fill": False,
                "line_ref": False,
                "predict_speed": False,
                "dips": False,
                "cast_lines": False,
            }
            st.save(
                work / "script.json",
                {
                    "ready": False,
                    "render": mode,
                    "lines": [{"start": 0, "end": 1, "voice": "speaker_a", "text": "Hello."}],
                },
            )
            st.save(work / "segments.json", {"job_id": "job", "source_lang": "ja", "target_lang": "en"})
            st.save(work / "rendered.json", {"L0": settings})
            st.save(work / "state.json", {"stage": "final", "round": 0})
            st.save(
                work / "generate-result.json",
                {
                    "client": {
                        "job_id": "job",
                        "wav_kind": "natural",
                        "policy": {
                            "timing": "concise",
                            "overflow": 0.3,
                            "engine": "omnivoice",
                            "match_level": True,
                            "voice_match": "consistent",
                            "render_mode": mode,
                        },
                    }
                },
            )
            calls = []
            ear = Mock(tag="fake", return_value="Hello.", load_s=0, hear_s=0)

            def generate(args, **kwargs):
                calls.append(args)
                path = Path(args[args.index("--task-outcome") + 1])
                pointer = st.core.load_json(work / "shorttakes" / "active-generation.json")
                self.assertEqual(path, Path(pointer["path"]))
                outcome = st.dub.GenerationOutcome(work, pointer)
                seed = int(args[args.index("--seed") + 1])
                write_audio(0.78)
                st.save(work / "rendered.json", {"L0": {**settings, "seed": seed, "seed_profile": "private"}})
                st.save(
                    work / "shorttakes" / "profiles.json",
                    {"receipt": {"id": "private", "seed": seed, "transport": "clone_profile_seed"}},
                )
                st.save(Path(args[args.index("--metrics") + 1]), {"completed": True})
                outcome.finish()

            def fixture_speech(path):
                audio, sr = sf.read(path)
                active = np.flatnonzero(np.abs(audio) > 0.001)
                return [[float(active[0] / sr), float((active[-1] + 1) / sr)]]

            with (
                patch.object(st.core, "WORK", root),
                patch.object(st.core, "JOBS", root / "jobs"),
                patch.object(st.core, "voice_sig", return_value="spec"),
                patch.object(st.core, "voice_wav", return_value=ref),
                patch.object(st.core, "line_id", return_value="L0"),
                patch.object(st.core, "actor_spans", return_value=[(0.05, 0.8)]),
                patch.object(st, "vad", side_effect=fixture_speech),
                patch.object(st.lineqa, "Ear", return_value=ear),
                patch.object(st.subprocess, "run", side_effect=generate),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                archived = st.trial("sample", {0}, archive=True)
                st.listen(wav, "en", ear, work / "heard.json")  # normal control's cached first-pass QA
                control = st.trial("sample", {0}, control=True)
                self.assertEqual(control["arm"], "control")
                self.assertFalse(control["unresolved"])
                self.assertEqual(control["context"], archived["context"])
                result = st.trial("sample", {0})
                self.assertEqual(len(calls), 2)
                self.assertEqual(result["selections"][0]["ordinal"], 1)
                self.assertEqual(wav.read_bytes(), baseline_audio)
                self.assertEqual(st.core.load_json(work / "rendered.json")["L0"], settings)
                heard_count = ear.call_count
                resumed = st.trial("sample", {0})
                self.assertEqual(len(calls), 2)
                self.assertEqual(ear.call_count, heard_count)
                self.assertEqual(result["context"], resumed["context"])
                st.trial("sample", {0}, apply=True)
                self.assertNotEqual(wav.read_bytes(), baseline_audio)
                self.assertIn("seed", st.core.load_json(work / "rendered.json")["L0"])
                self.assertEqual(st.core.load_json(work / "state.json")["stage"], "render")
                self.assertEqual(ear.call_count, heard_count)  # installation itself never loads ASR
                st.trial("sample", {0})
                self.assertEqual(wav.read_bytes(), baseline_audio)
                self.assertEqual(len(calls), 2)
                self.assertEqual(ear.call_count, heard_count)
                st.trial("sample", {0}, apply_control=True)
                self.assertEqual(wav.read_bytes(), baseline_audio)
                self.assertEqual(st.core.load_json(work / "rendered.json")["L0"], settings)
                changed = st.core.load_json(work / "script.json")
                changed["lines"][0]["text"] = "Goodbye."
                st.save(work / "script.json", changed)
                with self.assertRaises(ValueError):
                    st.trial("sample", {0}, apply=True)
                self.assertEqual(wav.read_bytes(), baseline_audio)


if __name__ == "__main__":
    unittest.main()
