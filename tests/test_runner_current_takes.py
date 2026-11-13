"""A successful remix cannot finalize speech different from the checked takes."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dub_anything as runner


class CurrentTakeGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.name = "sample-01"
        self.work = self.root / self.name
        self.work.mkdir()
        self.audio = self.root / "takes"
        self.audio.mkdir()
        self.script = {
            "lines": [
                {"id": "Lfirst", "start": 0.0, "end": 2.0, "voice": "speaker_a", "text": "The parcel is here."},
                {"id": "Lsecond", "start": 3.0, "end": 5.0, "voice": "speaker_b", "text": "Thank you very much."},
            ],
            "render": {"fill": True},
        }
        self.write("script.json", self.script)
        self.write("segments.json", {"job_id": "sample-job"})
        self.wav("Lfirst").write_bytes(b"first checked speech")
        self.wav("Lsecond").write_bytes(b"second checked speech")
        for patcher in (
            patch.object(runner, "WORK", self.root),
            patch.object(runner, "line_id", side_effect=lambda line: line["id"]),
            patch.object(runner, "langs", return_value=("ja", "en")),
            patch.object(runner, "take_wav", side_effect=lambda job, lid, target: self.wav(lid)),
            patch.object(runner, "clean_backend"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)

    def wav(self, lid):
        return self.audio / (lid + ".wav")

    def write(self, name, value):
        (self.work / name).write_text(json.dumps(value), encoding="utf-8")

    def state(self, stage="qa", **extra):
        self.write("state.json", {"script": runner.script_sha(self.script), "stage": stage, "round": 0, **extra})

    def current_state(self):
        return json.loads((self.work / "state.json").read_text(encoding="utf-8"))

    def checked(self):
        return {
            line["id"]: hashlib.sha256(self.wav(line["id"]).read_bytes()).hexdigest() for line in self.script["lines"]
        }

    def test_qa_records_restored_current_audio_then_unchanged_remix_finishes(self):
        self.state()

        def qa(*args):
            self.wav("Lfirst").write_bytes(b"restored better prior speech")
            return True, "", {"parts": {self.name: {"flagged": 0, "restored": 1}}}

        with patch.object(runner, "tool", side_effect=qa):
            self.assertEqual(runner.advance({}, self.name), "busy")
        state = self.current_state()
        self.assertEqual(state["stage"], "remix")
        self.assertEqual(state["qa_take_hashes"], self.checked())
        with patch.object(runner, "tool", return_value=(True, "done", {})):
            self.assertEqual(runner.advance({}, self.name), "busy")
        self.assertEqual(self.current_state()["stage"], "final")
        self.assertTrue(runner.finished({"name": self.name}, "actor"))

    def test_changed_remix_returns_to_normal_qa_before_final(self):
        self.state("remix", qa_take_hashes=self.checked())

        def remix(*args):
            self.wav("Lsecond").write_bytes(b"new fitting speech not yet checked")
            return True, "done", {}

        with patch.object(runner, "tool", side_effect=remix):
            self.assertEqual(runner.advance({}, self.name), "busy")
        state = self.current_state()
        self.assertEqual(state["stage"], "qa")
        self.assertEqual(state["qa_remix_rechecks"], 1)
        self.assertNotEqual(state["qa_take_hashes"], self.checked())
        self.assertFalse(runner.finished({"name": self.name}, "actor"))
        with patch.object(
            runner, "tool", return_value=(True, "", {"parts": {self.name: {"flagged": 0, "restored": 0}}})
        ) as tool:
            self.assertEqual(runner.advance({}, self.name), "busy")
        self.assertEqual(tool.call_args.args[2][1:], ["scripts/lineqa.py", self.name, "--reroll"])
        self.assertEqual(self.current_state()["qa_take_hashes"], self.checked())
        self.assertTrue(runner.finished({"name": self.name}, "actor"))

    def test_reported_overflow_blocks_with_the_lines_that_do_not_fit(self):
        self.state("render")
        over = {"i": 1, "start": 3.0, "speech": 2.9, "window": 2.0, "room": 2.55, "text": "Thank you very much."}
        with patch.object(runner, "tool", return_value=(False, "stopped", {"overflow": [over]})):
            self.assertEqual(runner.advance({}, self.name), "blocked")
        problem = self.current_state()["problem"]
        self.assertTrue(problem.startswith("lines don't fit their windows: L01 3.00"))
        self.assertIn("2.90s / 2.00s / 2.55s  Thank you very much.", problem)

    def test_failure_wording_without_a_report_is_retried_not_blocked(self):
        self.state("render")
        log = "  L01    3.00  2.90s / 2.00s / 2.55s  OVER  Thank you very much."
        with patch.object(runner, "tool", return_value=(False, log, {})), patch.object(runner.backend, "ensure"):
            self.assertEqual(runner.advance({}, self.name), "busy")
        self.assertNotIn("problem", self.current_state())
        self.assertEqual(self.current_state()["fails"], 1)

    def test_qa_without_a_report_remixes_before_final(self):
        self.state()
        with patch.object(runner, "tool", return_value=(True, "0 of 2 lines flagged", {})):
            self.assertEqual(runner.advance({}, self.name), "busy")
        self.assertEqual(self.current_state()["stage"], "remix")

    def test_missing_take_after_successful_qa_blocks(self):
        self.state()
        self.wav("Lsecond").unlink()
        with patch.object(
            runner, "tool", return_value=(True, "", {"parts": {self.name: {"flagged": 0, "restored": 0}}})
        ):
            self.assertEqual(runner.advance({}, self.name), "blocked")
        self.assertIn("missing or unreadable current take Lsecond", self.current_state()["problem"])
        self.assertNotEqual(self.current_state()["stage"], "final")

    def test_missing_take_after_successful_remix_blocks(self):
        self.state("remix", qa_take_hashes=self.checked())

        def remix(*args):
            self.wav("Lfirst").unlink()
            return True, "done", {}

        with patch.object(runner, "tool", side_effect=remix):
            self.assertEqual(runner.advance({}, self.name), "blocked")
        self.assertIn("current take Lfirst", self.current_state()["problem"])

    def test_repeated_remix_restore_changes_have_a_director_block(self):
        self.state()
        remixes = 0

        def tools(job, label, command, logfile):
            nonlocal remixes
            if command[1] == "scripts/lineqa.py":
                self.wav("Lfirst").write_bytes(b"restored prior")
                return True, "", {"parts": {self.name: {"flagged": 0, "restored": 1}}}
            remixes += 1
            self.wav("Lfirst").write_bytes(f"new remix {remixes}".encode())
            return True, "done", {}

        with patch.object(runner, "tool", side_effect=tools):
            for count in range(runner.MAX_REMIX_QA_RECHECKS + 1):
                self.assertEqual(runner.advance({}, self.name), "busy")
                expected = "busy" if count < runner.MAX_REMIX_QA_RECHECKS else "blocked"
                self.assertEqual(runner.advance({}, self.name), expected)
        state = self.current_state()
        self.assertEqual(state["stage"], "qa")
        self.assertIn("director review required", state["problem"])
        self.assertEqual(remixes, runner.MAX_REMIX_QA_RECHECKS + 1)

    def test_changed_new_final_reopens_qa_without_generating(self):
        self.state("final", qa_take_hashes=self.checked())
        self.wav("Lfirst").write_bytes(b"external change")
        self.assertFalse(runner.finished({"name": self.name}, "actor"))
        with patch.object(runner, "tool") as tool:
            self.assertEqual(runner.advance({}, self.name), "busy")
        tool.assert_not_called()
        self.assertEqual(self.current_state()["stage"], "qa")

    def test_missing_new_final_cannot_silently_pass_delivery_gate(self):
        self.state("final", qa_take_hashes=self.checked())
        self.wav("Lfirst").unlink()
        self.assertFalse(runner.finished({"name": self.name}, "actor"))
        with patch.object(runner, "tool") as tool:
            self.assertEqual(runner.advance({}, self.name), "blocked")
        tool.assert_not_called()
        self.assertIn("missing or unreadable", self.current_state()["problem"])

    def test_legacy_final_without_receipt_is_not_migrated(self):
        self.state("final")
        self.wav("Lfirst").unlink()
        before = (self.work / "state.json").read_bytes()
        with (
            patch.object(runner, "tool") as tool,
            patch.object(runner, "current_take_hashes", side_effect=AssertionError("legacy migration")),
        ):
            self.assertTrue(runner.finished({"name": self.name}, "actor"))
            self.assertEqual(runner.advance({}, self.name), "final")
        tool.assert_not_called()
        self.assertEqual((self.work / "state.json").read_bytes(), before)

    def test_resumed_legacy_remix_needs_a_current_qa_receipt(self):
        self.state("remix")
        with patch.object(runner, "tool", return_value=(True, "done", {})):
            self.assertEqual(runner.advance({}, self.name), "busy")
        self.assertEqual(self.current_state()["stage"], "qa")
        self.assertNotIn("qa_take_hashes", self.current_state())

    def test_status_does_not_label_changed_checked_audio_final(self):
        self.state("final", qa_take_hashes=self.checked())
        self.wav("Lfirst").write_bytes(b"changed after QA")
        output = io.StringIO()
        job = {"slug": "sample", "source": "sample.mp4", "langs": ["ja", "en"], "voices": ["actor"]}
        with (
            patch.object(runner, "load_job", return_value=job),
            patch.object(runner, "parts", return_value=[{"name": self.name, "start": 0}]),
            patch.object(runner, "job_dir_of", return_value=None),
            contextlib.redirect_stdout(output),
        ):
            runner.cmd_status(types.SimpleNamespace(slug="sample"))
        self.assertIn("current take QA coverage pending", output.getvalue())
        self.assertNotIn("actor: final", output.getvalue())

    def test_status_with_missing_director_script_does_not_read_take_receipt(self):
        self.state("final", qa_take_hashes=self.checked())
        (self.work / "script.json").unlink()
        output = io.StringIO()
        job = {"slug": "sample", "source": "sample.mp4", "langs": ["ja", "en"], "voices": ["actor"]}
        with (
            patch.object(runner, "load_job", return_value=job),
            patch.object(runner, "parts", return_value=[{"name": self.name, "start": 0}]),
            patch.object(runner, "job_dir_of", return_value=None),
            patch.object(runner, "current_take_hashes", side_effect=AssertionError("no directed script")),
            contextlib.redirect_stdout(output),
        ):
            runner.cmd_status(types.SimpleNamespace(slug="sample"))
        self.assertIn("to direct", output.getvalue())

    def test_changed_script_starts_fresh_render_and_resets_loop_bound(self):
        self.state(
            "qa", qa_take_hashes=self.checked(), qa_remix_rechecks=runner.MAX_REMIX_QA_RECHECKS + 1, problem="old loop"
        )
        self.script["lines"][0]["text"] = "The parcel arrived this morning."
        self.write("script.json", self.script)
        with patch.object(runner, "tool", return_value=(True, "done", {})) as tool:
            self.assertEqual(runner.advance({}, self.name), "busy")
        self.assertEqual(tool.call_args.args[2][1:3], ["scripts/dub.py", "generate"])
        self.assertEqual(self.current_state()["stage"], "qa")
        self.assertNotIn("qa_remix_rechecks", self.current_state())
        self.assertNotIn("problem", self.current_state())


if __name__ == "__main__":
    unittest.main()
