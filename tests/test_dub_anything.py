"""Direct job creation, checkpoint pauses, and resumed watcher failures."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import dub_anything as runner


class TextEvidenceTests(unittest.TestCase):
    def test_cli_preserves_text_modes_and_routes_caption_reading(self):
        for flag, mode, signs, extra in (
            (None, False, False, None),
            ("--captions", True, False, []),
            ("--subtitles", "open", False, ["--open"]),
            ("--auto-text", "auto", True, ["--auto"]),
        ):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                source = root / "sample.mp4"
                source.touch()
                work = root / "work"
                folder = work / "sample-01"
                folder.mkdir(parents=True)
                (folder / "segments.json").write_text('{"job_id": "sample"}', encoding="utf-8")
                (root / "separator.json").write_text("{}", encoding="utf-8")
                argv = ["dub_anything.py", "new", str(source)] + ([flag] if flag else [])
                with (
                    patch.object(runner, "VIDEOS", root / "videos"),
                    patch.object(runner, "JOB_DIR", work / "jobs"),
                    patch.object(runner, "WORK", work),
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaises(SystemExit) as stopped:
                        runner.main()
                    self.assertEqual(stopped.exception.code, 0)
                    job = runner.load_job("sample")
                    self.assertEqual(job["captions"], mode)
                    self.assertEqual(job["signs"], signs)
                    with (
                        patch.object(runner, "job_dir_of", return_value=root),
                        patch.object(runner, "tool", return_value=(True, "")) as tool,
                    ):
                        self.assertEqual(runner.prepare_step(job, [{"name": "sample-01"}]), "busy")
                    command = tool.call_args.args[2]
                    if extra is None:
                        self.assertIn("scripts/draft.py", command)
                    else:
                        self.assertEqual(command, runner.SUBS + ["sample-01"] + extra)


class OriginalAudioTextTests(unittest.TestCase):
    def test_original_audio_scans_text_without_audio_prep_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            folder = work / "sample-01"
            folder.mkdir()
            (folder / "script.json").write_text('{"lines": []}', encoding="utf-8")
            part = {"name": folder.name, "file": "opening.mp4"}
            job = {"signs": True, "langs": ["en", "es"]}
            with patch.object(runner, "WORK", work), patch.object(runner, "tool", return_value=(True, "")) as tool:
                self.assertEqual(runner.prepare_step(job, [part]), "busy")
                self.assertEqual(
                    tool.call_args.args[2],
                    runner.SIGNS + [part["name"], "--video", part["file"], "--source", "en", "--target", "es"],
                )
                self.assertFalse((folder / "segments.json").exists())
                (folder / "signs.json").write_text('{"candidates": []}', encoding="utf-8")
                tool.reset_mock()
                self.assertEqual(runner.prepare_step(job, [part]), "done")
                tool.assert_not_called()

    def test_original_audio_is_not_a_text_scan_completion_exemption(self):
        for scanned in (False, True):
            with self.subTest(scanned=scanned), tempfile.TemporaryDirectory() as tmp:
                work = pathlib.Path(tmp)
                part = {"name": "sample-01", "file": "opening.mp4"}
                folder = work / part["name"]
                folder.mkdir()
                (folder / "script.json").write_text('{"lines": []}', encoding="utf-8")
                if scanned:
                    (folder / "signs.json").write_text('{"candidates": []}', encoding="utf-8")
                job = {"slug": "sample", "voices": ["actor"], "signs": True, "delivered": {"path": "out.mkv"}}
                with (
                    patch.object(runner, "WORK", work),
                    patch.object(runner, "load_job", return_value=job),
                    patch.object(runner, "save_job"),
                    patch.object(runner.backend, "ensure", return_value=True),
                    patch.object(runner, "split", return_value=True),
                    patch.object(runner, "parts", return_value=[part]),
                    patch.object(runner, "second"),
                    patch.object(runner, "variant", return_value=part["name"]),
                    patch.object(runner, "advance", return_value="final"),
                    patch.object(runner, "prepare_step", return_value="done"),
                    patch.object(runner, "deliver_job", return_value=True) as delivery,
                    patch.object(runner, "report", return_value=work / "report.md"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(runner.cmd_run(types.SimpleNamespace(slug="sample", watch=False)), 0)
                self.assertEqual(delivery.call_count, int(scanned))

    def test_text_scan_remains_optional_when_not_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            folder = work / "sample-01"
            folder.mkdir()
            (folder / "script.json").write_text('{"lines": []}', encoding="utf-8")
            with patch.object(runner, "WORK", work), patch.object(runner, "tool") as tool:
                self.assertEqual(runner.prepare_step({"signs": False}, [{"name": folder.name}]), "done")
                tool.assert_not_called()


class PauseCheckpointTests(unittest.TestCase):
    def test_pause_waits_for_background_checks_and_skips_the_next_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = pathlib.Path(tmp) / "pause"
            marker.touch()
            background = Mock()
            with (
                patch.dict(runner.os.environ, {"DUB_PAUSE_FILE": str(marker)}),
                patch.object(runner, "SECOND_RUN", {"sample": (background, [])}),
                patch.object(runner.subprocess, "run") as launch,
            ):
                with self.assertRaises(SystemExit) as stopped:
                    runner.tool({"slug": "sample"}, "next step", [], "next")
                self.assertEqual(stopped.exception.code, 75)
                background.wait.assert_called_once()
                launch.assert_not_called()

    def test_direct_dubs_have_no_pause_marker(self):
        with patch.dict(runner.os.environ, {"DUB_PAUSE_FILE": ""}):
            runner.pause_checkpoint()


class ReadyPreflightTests(unittest.TestCase):
    def test_reference_prefix_choices_must_be_boolean(self):
        for value in ("false", 0, None):
            with self.subTest(value=value):
                problems = runner.ready_problems({"lines": [], "render": {"ref_prefix": value}})
                self.assertIn("render.ref_prefix must be true or false", problems)
                problems = runner.ready_problems({"lines": [self.line(ref_prefix=value)]})
                self.assertIn("lines[0].ref_prefix must be true or false", problems)
        self.assertEqual(
            runner.ready_problems({"lines": [self.line(ref_prefix=True)], "render": {"ref_prefix": False}}), []
        )

    def line(self, **changes):
        return {"start": 0, "end": 1.2, "voice": "speaker_a", "text": "Good morning.", **changes}

    def test_checked_actor_endpoint_is_validated_before_queueing(self):
        for value in (-1, 0, 1.3, float("nan"), float("inf"), "1", True):
            with self.subTest(value=value):
                problems = runner.ready_problems({"lines": [self.line(actor_end=value)]}, {"speaker_a": {}})
                self.assertIn("actor_end", "\n".join(problems))
        self.assertEqual(runner.ready_problems({"lines": [self.line(actor_end=1.1)]}, {"speaker_a": {}}), [])

    def test_invalid_structure_times_text_and_voice_are_reported(self):
        for script, error in (
            ([], "script must be an object"),
            ({}, "lines must be a list"),
            ({"lines": {}}, "lines must be a list"),
            ({"lines": [None]}, "lines[0] must be an object"),
            ({"lines": [self.line(start=-1)]}, "0 <= start < end"),
            ({"lines": [self.line(end=0)]}, "0 <= start < end"),
            ({"lines": [self.line(end=float("nan"))]}, "end must be a finite number"),
            ({"lines": [self.line(start=float("inf"))]}, "start must be a finite number"),
            ({"lines": [self.line(start=True)]}, "start must be a finite number"),
            ({"lines": [self.line(end="2")]}, "end must be a finite number"),
            ({"lines": [self.line(end=10**400)]}, "end must be a finite number"),
            ({"lines": [self.line(text=" \n")]}, "text must be nonblank text"),
            ({"lines": [self.line(text=None)]}, "text must be text"),
            ({"lines": [self.line(voice="unknown")]}, "voice must name a key"),
            ({"lines": [self.line(voice=[])]}, "voice must name a key"),
            ({"lines": [], "captions": ["bad"]}, "captions[0] must be an object"),
            ({"lines": [], "captions": [{"start": 0, "end": 1}]}, "captions[0].text must be text"),
            ({"lines": [], "captions": [{"start": 0, "end": 1, "text": None}]}, "captions[0].text must be text"),
            ({"lines": [], "render": []}, "render must be an object"),
        ):
            with self.subTest(script=script):
                self.assertIn(error, "\n".join(runner.ready_problems(script, {"speaker_a": {}})))

    def test_valid_original_subtitle_overlap_and_shared_take_scripts(self):
        self.assertEqual(runner.ready_problems({"lines": []}, {}), [])
        # Subtitle jobs do not need a cast. Captions may overlap spoken lines,
        # and deliberate repetitions may share an existing cached take.
        line = {"start": 0, "end": 2, "text": "Good morning."}
        script = {"lines": [line], "captions": [{**line, "start": 1}]}
        self.assertEqual(runner.ready_problems(script), [])
        # A source-only caption can deliberately omit its translated cue.
        script["captions"][0]["text"] = ""
        self.assertEqual(runner.ready_problems(script), [])
        self.assertEqual(runner.ready_problems({"lines": [self.line(), self.line()]}, {"speaker_a": {}}), [])

    def test_american_override_is_checked_when_it_will_be_spoken(self):
        script = {"lines": [self.line(text_us=" ")]}
        self.assertEqual(runner.ready_problems(script, {"speaker_a": {}}), [])
        self.assertIn(
            "text_us must be nonblank text", "\n".join(runner.ready_problems(script, {"speaker_a": {}}, american=True))
        )

    def test_timed_subtitle_layout_requires_finite_bounds_and_picture_fraction(self):
        entry = {"start": 1, "end": 3, "top": 0.19}
        for value, error in (
            ({}, "subtitle_layout must be a list"),
            ([None], "subtitle_layout[0] must be an object"),
            ([{**entry, "start": -1}], "0 <= start < end"),
            ([{**entry, "end": 1}], "0 <= start < end"),
            ([{**entry, "end": float("inf")}], "end must be a finite number"),
            ([{**entry, "start": True}], "start must be a finite number"),
            ([{**entry, "top": float("nan")}], ".top must be a finite picture-height fraction"),
            ([{**entry, "top": 10**400}], ".top must be a finite picture-height fraction"),
            ([{**entry, "top": True}], ".top must be a finite picture-height fraction"),
            ([{**entry, "top": -0.1}], ".top must be a finite picture-height fraction"),
            ([{**entry, "top": 1.1}], ".top must be a finite picture-height fraction"),
        ):
            with self.subTest(value=value):
                self.assertIn(error, "\n".join(runner.ready_problems({"lines": [], "subtitle_layout": value})))
        for top in (0, 0.19, 1):
            self.assertEqual(runner.ready_problems({"lines": [], "subtitle_layout": [{**entry, "top": top}]}), [])

    def test_ready_batch_does_not_write_until_every_script_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            scripts = [{"lines": [self.line()]}, {"lines": [self.line(end=0)]}]
            parts = [{"name": "sample-01"}, {"name": "sample-02"}]
            for part, script in zip(parts, scripts):
                folder = root / part["name"]
                folder.mkdir()
                (folder / "script.json").write_text(json.dumps(script), encoding="utf-8")
            (root / "voices.json").write_text('{"speaker_a": {}}', encoding="utf-8")
            with (
                patch.object(runner, "WORK", root),
                patch.object(runner, "load_job", return_value={"voices": ["actor"]}),
                patch.object(runner, "parts", return_value=parts),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                args = types.SimpleNamespace(slug="sample", parts=[1, 2])
                with self.assertRaisesRegex(SystemExit, "sample-02: cannot mark ready"):
                    runner.cmd_ready(args)
                self.assertNotIn("ready", json.loads((root / "sample-01/script.json").read_text(encoding="utf-8")))
                scripts[1]["lines"][0]["end"] = 1.2
                (root / "sample-02/script.json").write_text(json.dumps(scripts[1]), encoding="utf-8")
                runner.cmd_ready(args)
            for part in parts:
                self.assertTrue(json.loads((root / part["name"] / "script.json").read_text(encoding="utf-8"))["ready"])

    def test_subtitle_job_needs_no_registry_and_bad_json_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            folder = root / "sample-01"
            folder.mkdir()
            path = folder / "script.json"
            path.write_text('{"lines": [', encoding="utf-8")
            with (
                patch.object(runner, "WORK", root),
                patch.object(runner, "load_job", return_value={"voices": []}),
                patch.object(runner, "parts", return_value=[{"name": "sample-01"}]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                args = types.SimpleNamespace(slug="sample", parts=[1])
                with self.assertRaisesRegex(SystemExit, "part 1: invalid script.json"):
                    runner.cmd_ready(args)
                script = {"lines": [{"start": 0, "end": 2, "text": "Good morning."}]}
                path.write_text(json.dumps(script), encoding="utf-8")
                runner.cmd_ready(args)
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["ready"])


class StatusTests(unittest.TestCase):
    def test_changed_script_does_not_report_an_old_final_or_its_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            part = {"name": "sample-01", "start": 0}
            work = root / part["name"]
            work.mkdir()
            script = {"lines": [{"start": 0, "end": 2, "text": "Good morning."}], "ready": True}
            state = {"script": runner.script_sha(script), "stage": "final", "round": 1}
            (work / "state.json").write_text(json.dumps(state), encoding="utf-8")
            (work / "reroll-left.json").write_text('[{"id": "old"}]', encoding="utf-8")

            def status():
                (work / "script.json").write_text(json.dumps(script), encoding="utf-8")
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    runner.cmd_status(types.SimpleNamespace(slug="sample"))
                return output.getvalue()

            with (
                patch.object(runner, "WORK", root),
                patch.object(runner, "parts", return_value=[part]),
                patch.object(runner, "job_dir_of", return_value=None),
                patch.object(
                    runner,
                    "load_job",
                    return_value={"slug": "sample", "source": "sample.mp4", "langs": ["ja", "en"], "voices": ["actor"]},
                ),
            ):
                self.assertIn("actor: final r1 (1 flagged)", status())
                script["signs"] = [{"start": 0, "end": 2, "text": "Station"}]
                self.assertIn("actor: final r1 (1 flagged)", status())
                script["lines"][0]["text"] = "Good morning, everyone."
                changed = status()
                self.assertIn("actor: script changed; render pending", changed)
                self.assertNotIn("final", changed)
                self.assertNotIn("flagged", changed)


class WatchBlockTests(unittest.TestCase):
    def run_changed_block(self, before, after, known=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            part = {"name": "sample-01"}
            path = root / part["name"] / "state.json"
            path.parent.mkdir()
            path.write_text(json.dumps(before), encoding="utf-8")

            def advance(job, name):
                path.write_text(json.dumps(after), encoding="utf-8")
                return "blocked"

            with (
                patch.object(runner, "WORK", root),
                patch.object(runner, "load_job", return_value={"slug": "sample", "voices": ["actor"]}),
                patch.object(runner.backend, "ensure", return_value=True),
                patch.object(runner, "split", return_value=True),
                patch.object(runner, "parts", return_value=[part]),
                patch.object(runner, "save_job"),
                patch.object(runner, "second"),
                patch.object(runner, "ready", return_value=True),
                patch.object(runner, "original_only", return_value=False),
                patch.object(runner, "drafted", return_value=True),
                patch.object(runner, "variant", return_value=part["name"]),
                patch.object(runner, "advance", side_effect=advance),
                patch.object(runner, "prepare_step", return_value="idle"),
                patch.object(runner, "finished", return_value=False),
                patch.object(runner, "report") as report,
                patch.object(runner.time, "sleep", side_effect=AssertionError("watcher waited on a new block")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                args = types.SimpleNamespace(slug="sample", watch=True)
                if known:
                    with self.assertRaisesRegex(AssertionError, "watcher waited on a new block"):
                        runner.cmd_run(args)
                    report.assert_not_called()
                else:
                    result = runner.cmd_run(args)
                    report.assert_called_once()
                    self.assertEqual(result, 2)

    def test_changed_failure_on_previously_blocked_part_stops_watch(self):
        self.run_changed_block(
            {"script": "old", "problem": "lines don't fit: L01"},
            {"script": "old", "problem": "lines don't fit: L05"},
        )

    def test_same_failure_after_script_change_stops_watch(self):
        self.run_changed_block(
            {"script": "old", "problem": "lines don't fit: L01"},
            {"script": "new", "problem": "lines don't fit: L01"},
        )

    def test_unchanged_known_block_keeps_waiting(self):
        state = {"script": "old", "problem": "lines don't fit: L01"}
        self.run_changed_block(state, state, known=True)
