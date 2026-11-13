"""The runner acts on what a tool reports, not on how its messages are worded,
and shared JSON files are replaced in one step."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import core
import dub
import dub_anything as runner


class SaveJsonTests(unittest.TestCase):
    def test_failed_write_keeps_the_previous_file_and_leaves_no_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "state.json"
            core.save_json(path, {"stage": "qa"})
            with self.assertRaises(TypeError):
                core.save_json(path, {"stage": object()})
            self.assertEqual(core.load_json(path), {"stage": "qa"})
            self.assertEqual([p.name for p in path.parent.iterdir()], ["state.json"])

    def test_swap_is_retried_while_a_reader_holds_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "nested" / "job.json"
            replace, attempts = pathlib.Path.replace, []

            def busy_once(self, target):
                attempts.append(target)
                if len(attempts) == 1:
                    raise PermissionError("held by a reader")
                return replace(self, target)

            with patch.object(pathlib.Path, "replace", busy_once), patch.object(core.time, "sleep"):
                core.save_json(path, {"slug": "sample", "title": "día"})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"slug": "sample", "title": "día"})


class ReportResultTests(unittest.TestCase):
    def test_reports_only_when_the_runner_asks_and_merges_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "step.result.json"
            with patch.dict(core.os.environ, {core.RESULT_ENV: ""}):
                core.report_result(output="ignored")
            self.assertFalse(path.exists())
            with patch.dict(core.os.environ, {core.RESULT_ENV: str(path)}):
                core.report_result(output="sample.mp4")
                core.report_result(parts={"sample-01": {"restored": 1}})
            self.assertEqual(core.load_json(path), {"output": "sample.mp4", "parts": {"sample-01": {"restored": 1}}})


class ToolTests(unittest.TestCase):
    def run_tool(self, root, body):
        child = [sys.executable, "-c", f"import sys; sys.path.insert(0, 'scripts'); import core; {body}"]
        with patch.object(core, "WORK", root), contextlib.redirect_stdout(io.StringIO()):
            return runner.tool({"slug": "sample"}, "step", child, "step")

    def test_returns_the_report_and_drops_an_earlier_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ok, out, result = self.run_tool(root, "print('-> misleading.mp4'); core.report_result(output='real.mp4')")
            self.assertTrue(ok)
            self.assertIn("misleading", out)
            self.assertEqual(result, {"output": "real.mp4"})
            ok, _, result = self.run_tool(root, "raise SystemExit(3)")
            self.assertFalse(ok)
            self.assertEqual(result, {})


class PrepareStepTests(unittest.TestCase):
    def prepare(self, tool_result):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            job = {"slug": "sample", "voices": ["actor"]}
            part = {"name": "sample-01", "file": "sample-01.mp4"}
            with (
                patch.object(core, "WORK", work),
                patch.object(runner, "save_job"),
                patch.object(runner, "tool", return_value=tool_result),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                step = runner.prepare_step(job, [part])
            return step, core.load_json(work / "sample-01" / "script.json"), job

    def test_reported_silence_leaves_the_part_original(self):
        step, script, job = self.prepare((False, "transcribe error", {"no_speech": True}))
        self.assertEqual(step, "busy")
        self.assertEqual(script, {"lines": []})
        self.assertNotIn("problems", job)

    def test_the_wording_alone_is_a_failed_prep(self):
        step, script, job = self.prepare((False, "Transcription produced no segments", {}))
        self.assertEqual(step, "failed")
        self.assertIsNone(script)
        self.assertIn("prep sample-01 failed", job["problems"][0])


class PrepReportTests(unittest.TestCase):
    def prep(self, event):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            video = root / "sample-01.mp4"
            video.write_bytes(b"video")
            reported = root / "prep.result.json"
            requests = MagicMock()
            requests.post.return_value.json.return_value = {"job_id": "job", "task_id": "task"}
            args = types.SimpleNamespace(
                video=str(video), subtitles_only=False, source="ja", target="en", segment_refs=False, speakers=None
            )
            with (
                patch.object(core, "WORK", root / "work"),
                patch.object(dub, "requests", requests),
                patch.object(dub, "wait_task"),
                patch.object(dub, "require_shared_media"),
                patch.object(dub, "separate"),
                patch.object(dub, "ensure_diarizer"),
                patch.object(dub, "sse", return_value=iter([("error", event)])),
                patch.dict(core.os.environ, {core.RESULT_ENV: str(reported)}),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(RuntimeError),
            ):
                dub.cmd_prep(args)
            return core.load_json(reported, {})

    def test_silent_part_is_reported_and_other_failures_are_not(self):
        silent = {"detail": "Transcription produced no segments. The audio may be silent.", "retryable": True}
        self.assertEqual(self.prep(silent), {"no_speech": True})
        # A chunk failure carries a code and its own recovery message: a real failure.
        self.assertEqual(self.prep({**silent, "code": "asr_chunk_failed"}), {})
        self.assertEqual(self.prep({"detail": "model missing"}), {})


if __name__ == "__main__":
    unittest.main()
