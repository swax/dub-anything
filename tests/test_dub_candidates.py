"""Opt-in hooks must not widen regeneration or alter normal render defaults."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dub


class SeededProfileTests(unittest.TestCase):
    def test_seed_profiles_preserve_reference_and_never_mutate_existing_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "reference.wav"
            wav.write_bytes(b"exact reference bytes")
            (root / "voices.json").write_text(
                json.dumps({"speaker": {"clips": [[0, 1]], "ref_text": "Reference words."}}), encoding="utf-8"
            )
            created = []
            profiles = {}

            def post(url, files, data, timeout):
                created.append((files["ref_audio"][1].read(), dict(data)))
                profile = {"id": str(len(created)), "seed": data["seed"], "ref_text": data["ref_text"]}
                profiles[profile["id"]] = profile
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: profile)

            def get(url, timeout):
                return SimpleNamespace(ok=True, json=lambda: profiles[url.rsplit("/", 1)[-1]])

            with (
                patch.object(dub, "WORK", root),
                patch.object(dub, "voice_wav", return_value=wav),
                patch.object(dub, "voice_sig", return_value="same actor spec"),
                patch.object(dub.requests, "post", side_effect=post),
                patch.object(dub.requests, "get", side_effect=get),
                patch.object(dub.requests, "put", side_effect=AssertionError("no profile mutation")),
                patch.object(dub.requests, "delete", side_effect=AssertionError("no profile deletion")),
            ):
                first = dub.seeded_voice(root / "part", "speaker", 42)
                self.assertEqual(first, dub.seeded_voice(root / "part", "speaker", 42))
                self.assertNotEqual(first, dub.seeded_voice(root / "part", "speaker", 43))
                self.assertEqual(len(created), 2)
                self.assertEqual(created[0][0], b"exact reference bytes")
                self.assertEqual(created[0][1]["ref_text"], "Reference words.")
                self.assertEqual(created[0][1]["seed"], 42)
                wav.write_bytes(b"changed reference bytes")
                self.assertNotEqual(first, dub.seeded_voice(root / "part", "speaker", 42))
                self.assertEqual(len(created), 3)


class IncrementalRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work = self.root / "part"
        self.work.mkdir()
        self.lines = [
            {"start": 0, "end": 1, "voice": "a", "text": "Hello."},
            {"start": 2, "end": 3, "voice": "b", "text": "Goodbye."},
        ]
        self.settings = {"steps": 32, "effect": "broadcast", "instruct": ""}
        self.write("segments.json", {"job_id": "job"})
        self.write("script.json", {"lines": self.lines})
        self.write("rendered.json", {"L0": self.settings, "L2": self.settings})
        self.write("generate-result.json", {"client": {"job_id": "job", "wav_kind": "natural"}})
        for lid in ("L0", "L2"):
            sf.write(self.root / (lid + ".wav"), np.full(8000, 0.1), 16000)
        self.requests = []
        self.widen = False
        self.previous_engine = "omnivoice"
        self.engine_requests = []

    def write(self, name, value):
        (self.work / name).write_text(json.dumps(value), encoding="utf-8")

    def args(self, only=None, seed=None):
        return argparse.Namespace(
            name="part",
            only=only,
            seed=seed,
            full=False,
            engine="omnivoice",
            emo_alpha=0,
            timing="concise",
            voice_match="consistent",
            steps=32,
            effect="broadcast",
            overflow=0.3,
            match_level=False,
            no_export=True,
            tag="",
        )

    def run_generate(self, args, wait=None, submit=None, *, fill=False, spans=None):
        def post(url, json, timeout):
            self.requests.append(json)
            return SimpleNamespace(status_code=200, json=lambda: {"task_id": "task"})

        def done(task, terminal):
            for lid in self.requests[-1].get("regen_only", []):
                sf.write(self.root / (lid + ".wav"), np.full(8000, 0.2), 16000)
            if self.widen:
                sf.write(self.root / "L2.wav", np.full(8000, 0.3), 16000)
            return {"type": "done"}

        def engine(name):
            self.engine_requests.append(name)
            return self.previous_engine

        mode = {
            "voice": "actor",
            "fill": fill,
            "line_ref": False,
            "predict_speed": False,
            "dips": False,
            "cast_lines": False,
        }
        with (
            patch.object(dub, "WORK", self.root),
            patch.object(dub, "langs", return_value=("ja", "en")),
            patch.object(dub, "line_id", side_effect=lambda line: "L" + str(line["start"])),
            patch.object(dub, "take_wav", side_effect=lambda job, lid, target, kind="seg": self.root / (lid + ".wav")),
            patch.object(dub, "render_mode", return_value=mode),
            patch.object(dub, "actor_spans", return_value=spans or [(0, 0.9), (0, 0.5)]),
            patch.object(
                dub,
                "fit_rows",
                return_value=[
                    {"i": 0, "speech": 0.5, "room": 1.55, "flag": ""},
                    {"i": 1, "speech": 0.5, "room": 1.3, "flag": ""},
                ],
            ),
            patch.object(dub, "ensure_voice", side_effect=lambda key: "original-" + key),
            patch.object(dub, "seeded_voice", side_effect=lambda work, key, seed: "private-" + str(seed)) as seeded,
            patch.object(dub, "use_tts_engine", side_effect=engine),
            patch.object(dub, "match_gains", return_value=[1, 1]),
            patch.object(dub.requests, "post", side_effect=submit or post),
            patch.object(dub, "wait_task", side_effect=wait or done),
            patch.object(dub.requests, "get", side_effect=AssertionError("no export requested")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            dub.cmd_generate(args)
            return seeded.call_args_list

    def test_default_cached_render_uses_original_profiles_and_no_seeds(self):
        calls = self.run_generate(self.args())
        self.assertEqual(calls, [])
        self.assertEqual(self.requests[0]["regen_only"], [])
        self.assertEqual([s["profile_id"] for s in self.requests[0]["segments"]], ["original-a", "original-b"])
        self.assertFalse(any("seed" in s for s in dub.load_json(self.work / "rendered.json").values()))
        self.assertFalse((self.work / "shorttakes" / "active-generation.json").exists())

    def overflow(self, *args):
        raise dub.BackendTaskError(
            "task", {"type": "error", "segment": 0, "segment_id": "L0", "error_code": "dub_timing_overflow"}
        )

    def test_rolled_back_slow_take_remixes_the_fitting_published_take(self):
        self.write("rendered.json", {"L0": self.settings, "L2": {**self.settings, "speed": 0.8}})
        before = {lid: (self.root / (lid + ".wav")).read_bytes() for lid in ("L0", "L2")}

        def wait(*args):
            if len(self.requests) == 1:
                return self.overflow(*args)
            return {"type": "done"}

        self.run_generate(self.args(), wait=wait, fill=True)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0]["regen_only"], ["L0"])
        self.assertEqual(self.requests[0]["segments"][0]["speed"], 0.8)
        self.assertEqual(self.requests[1]["regen_only"], [])
        self.assertNotIn("speed", self.requests[1]["segments"][0])
        self.assertEqual({lid: (self.root / (lid + ".wav")).read_bytes() for lid in before}, before)
        self.assertEqual(dub.load_json(self.work / "rendered.json")["L0"], self.settings)

    def test_overflow_recovery_does_not_override_directed_pace(self):
        self.lines[0]["speed"] = 0.8
        self.write("script.json", {"lines": self.lines})
        with self.assertRaises(SystemExit):
            self.run_generate(self.args(), wait=self.overflow, fill=True)
        self.assertTrue(all(r["segments"][0]["speed"] == 0.8 for r in self.requests))

    def test_overflow_recovery_does_not_undo_requested_word_reroll(self):
        self.write("reroll.json", [{"id": "L0"}])
        with self.assertRaises(SystemExit):
            self.run_generate(self.args(), wait=self.overflow, fill=True)
        self.assertTrue(all("L0" in r["regen_only"] for r in self.requests))

    def test_failed_word_reroll_cannot_be_replaced_with_a_cached_only_success(self):
        self.write("reroll.json", [{"id": "L0"}])

        def wait(*args):
            if len(self.requests) == 1:
                return self.overflow(*args)
            return {"type": "done"}

        with self.assertRaises(SystemExit):
            self.run_generate(self.args(), wait=wait, fill=True, spans=[(0, 0.5), (0, 0.5)])
        self.assertEqual(len(self.requests), 1)

    def test_failed_full_regeneration_cannot_reuse_a_previous_publication(self):
        args = self.args()
        args.full = True
        with self.assertRaises(SystemExit):
            self.run_generate(args, wait=self.overflow, fill=True)
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("regen_only", self.requests[0])

    def test_overflow_recovery_does_not_accept_old_quality_settings(self):
        args = self.args()
        args.steps = 16
        with self.assertRaises(SystemExit):
            self.run_generate(args, wait=self.overflow, fill=True)
        self.assertTrue(all("L0" in r["regen_only"] for r in self.requests))

    def test_only_seeded_line_regenerates_and_other_metadata_survives(self):
        other = {**self.settings, "marker": "must survive"}
        self.write("rendered.json", {"L0": self.settings, "L2": other})
        original = (self.root / "L2.wav").read_bytes()
        self.run_generate(self.args("0", 42))
        self.assertEqual(self.requests[0]["regen_only"], ["L0"])
        self.assertEqual(self.requests[0]["segments"][0]["profile_id"], "private-42")
        self.assertEqual((self.root / "L2.wav").read_bytes(), original)
        self.assertEqual(dub.load_json(self.work / "rendered.json")["L2"], other)
        self.assertEqual(dub.load_json(self.work / "rendered.json")["L0"]["seed"], 42)

    def test_normal_remix_reuses_selected_seed_and_matching_metadata(self):
        self.run_generate(self.args("0", 42))
        self.requests.clear()
        self.run_generate(self.args())
        self.assertEqual(self.requests[0]["regen_only"], [])
        self.assertEqual(self.requests[0]["segments"][0]["profile_id"], "private-42")

    def test_unknown_or_slotted_baseline_fails_before_requests(self):
        for receipt in ({}, {"client": {"job_id": "job", "wav_kind": "slotted"}}):
            self.write("generate-result.json", receipt)
            with self.assertRaises(ValueError):
                self.run_generate(self.args("0", 42))
        self.assertEqual(self.requests, [])

    def test_corrupt_unselected_cache_cannot_widen_selected_regeneration(self):
        (self.root / "L2.wav").write_bytes(b"not audio")
        with self.assertRaises(ValueError):
            self.run_generate(self.args("0", 42))
        self.assertEqual(self.requests, [])

    def test_only_cannot_be_combined_with_full(self):
        args = self.args("0", 42)
        args.full = True
        with self.assertRaises(ValueError):
            self.run_generate(args)

    def test_unexpected_backend_widening_restores_and_archives_unselected_audio(self):
        original = (self.root / "L2.wav").read_bytes()
        self.widen = True
        with self.assertRaisesRegex(RuntimeError, "unselected caches"):
            self.run_generate(self.args("0", 42))
        self.assertEqual((self.root / "L2.wav").read_bytes(), original)
        unexpected = list((self.work / "shorttakes" / "unexpected").glob("L2-*.wav"))
        self.assertEqual(len(unexpected), 1)
        self.assertNotEqual(unexpected[0].read_bytes(), original)

    def outcome(self):
        active = dub.load_json(self.work / "shorttakes" / "active-generation.json")
        return dub.outcome_read(self.work, active["path"], active["attempt"])

    def test_lost_stream_does_not_restore_over_live_late_writer(self):
        original = (self.root / "L2.wav").read_bytes()
        metadata = (self.work / "rendered.json").read_bytes()
        release = threading.Event()
        wrote = threading.Event()

        def writer():
            if release.wait(3):
                (self.root / "L2.wav").write_bytes(b"late writer")
                wrote.set()

        thread = threading.Thread(target=writer)
        self.addCleanup(thread.join, 3)
        self.addCleanup(release.set)

        def disconnect(*args):
            (self.root / "L2.wav").write_bytes(b"partial writer")
            thread.start()
            raise dub.requests.ConnectionError("connection lost")

        self.previous_engine = "previous-engine"
        with self.assertRaises(dub.UnknownTaskOutcome):
            self.run_generate(self.args("0", 42), wait=disconnect)
        self.assertEqual((self.root / "L2.wav").read_bytes(), b"partial writer")
        self.assertEqual((self.work / "rendered.json").read_bytes(), metadata)
        self.assertEqual(self.engine_requests, ["omnivoice"])
        doc = self.outcome()
        self.assertEqual(doc["phase"], "unknown")
        self.assertEqual(doc["requests"][0]["task_id"], "task")
        self.assertEqual(doc["requests"][0]["body"]["segments"][0]["profile_id"], "private-42")
        saved = {row["source"]: Path(row["path"]).read_bytes() for row in doc["before"]}
        self.assertEqual(saved[str((self.root / "L2.wav").resolve())], original)
        self.assertEqual(saved[str((self.work / "rendered.json").resolve())], metadata)
        self.assertEqual(doc["previous_engine"], "previous-engine")
        release.set()
        self.assertTrue(wrote.wait(3))
        for args in [self.args(), self.args("0", 43)]:
            with self.assertRaises(dub.UnknownTaskOutcome):
                self.run_generate(args)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual((self.root / "L2.wav").read_bytes(), b"late writer")

    def test_lost_post_response_or_task_id_remains_unknown(self):
        def lost_post(*args, **kwargs):
            raise dub.requests.Timeout("POST response lost")

        for submit in [
            lost_post,
            Mock(return_value=SimpleNamespace(status_code=200, json=lambda: {})),
            Mock(return_value=SimpleNamespace(status_code=200, json=lambda: {"task_id": None})),
        ]:
            with self.subTest(submit=submit):
                # A fresh isolated job for each uncertain POST.
                self.work = self.root / ("post-" + str(len(list(self.root.glob("post-*")))))
                self.work.mkdir()
                self.write("segments.json", {"job_id": "job"})
                self.write("script.json", {"lines": self.lines})
                self.write("rendered.json", {"L0": self.settings, "L2": self.settings})
                self.write("generate-result.json", {"client": {"job_id": "job", "wav_kind": "natural"}})
                args = self.args("0", 42)
                args.name = self.work.name
                with self.assertRaises(dub.UnknownTaskOutcome):
                    self.run_generate(args, submit=submit)
                doc = self.outcome()
                self.assertIsNone(doc["requests"][0]["task_id"])
                self.assertEqual(doc["phase"], "unknown")
                self.assertTrue(doc["before"])

    def test_terminal_error_can_restore_protected_files_and_finish_command(self):
        original = (self.root / "L2.wav").read_bytes()

        def failed(*args):
            (self.root / "L2.wav").write_bytes(b"unexpected terminal output")
            raise dub.BackendTaskError("task", {"type": "error", "error": "generation failed"})

        with self.assertRaisesRegex(RuntimeError, "unselected caches"):
            self.run_generate(self.args("0", 42), wait=failed)
        self.assertEqual((self.root / "L2.wav").read_bytes(), original)
        doc = self.outcome()
        self.assertTrue(dub.outcome_terminal(doc))
        self.assertFalse(doc["success"])
        self.run_generate(self.args())  # Known quiescent failure does not block normal work.

    def test_finished_first_request_is_not_finished_remix_command(self):
        calls = []

        def wait(*args):
            calls.append(args)
            if len(calls) == 1:
                return {"type": "done"}
            raise dub.requests.ConnectionError("remix lost")

        args = self.args("0", 42)
        args.match_level = True
        with self.assertRaises(dub.UnknownTaskOutcome):
            self.run_generate(args, wait=wait)
        doc = self.outcome()
        self.assertEqual([r["status"] for r in doc["requests"]], ["done", "unknown"])
        self.assertFalse(dub.outcome_terminal(doc))

    def test_unconfirmed_overflow_message_cannot_trigger_next_fill(self):
        def failure(*args):
            raise RuntimeError("dub_timing_overflow connection lost")

        with self.assertRaises(dub.UnknownTaskOutcome):
            self.run_generate(self.args("0", 42), wait=failure)
        self.assertEqual(len(self.requests), 1)

    def test_receipt_failure_before_post_restores_previous_engine_and_keeps_files(self):
        self.previous_engine = "previous-engine"
        paths = list(self.work.glob("*.json")) + [self.root / "L0.wav", self.root / "L2.wav"]
        before = {path: path.read_bytes() for path in paths}
        original_save = dub.GenerationOutcome.save
        failed = []

        def fail_engine_checkpoint_once(outcome):
            if "previous_engine" in outcome.doc and not failed:
                failed.append(True)
                raise OSError("engine checkpoint publication failed")
            original_save(outcome)

        with patch.object(dub.GenerationOutcome, "save", fail_engine_checkpoint_once):
            with self.assertRaisesRegex(OSError, "engine checkpoint publication failed"):
                self.run_generate(self.args("0", 42))
        self.assertEqual(self.engine_requests, ["omnivoice", "previous-engine"])
        self.assertEqual(self.requests, [])
        self.assertEqual({path: path.read_bytes() for path in paths}, before)
        doc = self.outcome()
        self.assertTrue(dub.outcome_terminal(doc))
        self.assertFalse(doc["success"])
        self.assertEqual(doc["requests"], [])


class TaskEventTests(unittest.TestCase):
    def test_checkpoint_retries_only_file_publication_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            ticket = dub.GenerationOutcome.prepare(work)
            outcome = dub.GenerationOutcome(work, ticket)
            original_replace = Path.replace
            attempts = []

            def briefly_locked(path, target):
                attempts.append(target)
                if len(attempts) < 3:
                    raise PermissionError("brief sharing violation")
                return original_replace(path, target)

            with patch.object(Path, "replace", briefly_locked), patch.object(dub.time, "sleep") as sleep:
                outcome.save()
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            outcome.doc.update(phase="finished")
            with (
                patch.object(Path, "replace", side_effect=PermissionError("still locked")) as replace,
                patch.object(dub.time, "sleep"),
            ):
                with self.assertRaises(PermissionError):
                    outcome.save()
            self.assertEqual(replace.call_count, 4)
            with self.assertRaises(dub.UnknownTaskOutcome):
                dub.require_settled_generation(work)  # Old nonterminal receipt remains authoritative.

    def test_explicit_errors_are_runtime_compatible_terminal_evidence(self):
        for kind in ["error", "cancelled"]:
            event = {"type": kind, "error": "failure"}
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            with (
                patch.object(dub.requests, "get", return_value=response),
                patch.object(dub, "sse", return_value=iter([(None, event)])),
            ):
                with self.assertRaises(dub.BackendTaskError) as caught:
                    dub.wait_task("task", {"done"})
            self.assertIsInstance(caught.exception, RuntimeError)
            self.assertEqual(caught.exception.event, event)
            self.assertEqual(str(caught.exception), f"task task {kind}: {json.dumps(event)}")

    def test_stream_eof_is_not_terminal_evidence(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(dub.requests, "get", return_value=response), patch.object(dub, "sse", return_value=iter([])):
            with self.assertRaises(RuntimeError) as caught:
                dub.wait_task("task", {"done"})
        self.assertNotIsInstance(caught.exception, dub.BackendTaskError)


if __name__ == "__main__":
    unittest.main()
