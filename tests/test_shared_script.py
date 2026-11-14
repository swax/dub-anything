"""A part's script.json has several writers. A tool writes only over what it read,
and a script that can't be read makes its part wait instead of stopping the run."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import core
import dub_anything as runner

LINE = {"start": 0, "end": 1, "voice": "hero", "text": "first words"}


class UpdateJsonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "script.json"
        core.save_json(self.path, {"lines": [LINE]})

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_change_is_applied_again_to_what_another_writer_saved_meanwhile(self):
        seen = []

        def add_render(script):
            seen.append(script["lines"][0]["text"])
            if len(seen) == 1:  # the director saves an edit while the tool works
                core.save_json(self.path, {"lines": [{**LINE, "text": "second words"}]})
            return {**script, "render": {"fill": True}}

        written = core.update_json(self.path, add_render)
        self.assertEqual(seen, ["first words", "second words"])
        self.assertEqual(written, core.load_json(self.path))
        self.assertEqual(written["lines"][0]["text"], "second words")
        self.assertEqual(written["render"], {"fill": True})

    def test_nothing_to_change_writes_nothing(self):
        before = self.path.stat().st_mtime_ns
        self.assertEqual(core.update_json(self.path, lambda script: None), {"lines": [LINE]})
        self.assertEqual(self.path.stat().st_mtime_ns, before)

    def test_a_file_that_never_holds_still_is_left_alone(self):
        def restless(script):
            core.save_json(self.path, {"lines": [], "n": script.get("n", 0) + 1})
            return {"lines": ["mine"]}

        with self.assertRaises(RuntimeError):
            core.update_json(self.path, restless, tries=3)
        self.assertEqual(core.load_json(self.path), {"lines": [], "n": 3})

    def test_save_text_unchanged_refuses_a_file_saved_since_it_was_read(self):
        was = core.read_text(self.path)
        self.assertTrue(core.save_text_unchanged(self.path, '{"lines": []}', was))
        self.assertFalse(core.save_text_unchanged(self.path, '{"lines": ["late"]}', was))
        self.assertEqual(core.load_json(self.path), {"lines": []})
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["script.json"])


class RunnerScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.patches = [
            patch.object(core, "ROOT", self.root),
            patch.object(core, "WORK", self.root / "work"),
            patch.object(core, "JOBS", self.root / "vs_jobs"),
            patch.dict(runner.ENV),
            patch.dict(runner.UNREADABLE, clear=True),
        ]
        for p in self.patches:
            p.start()
        home = core.job_dir("show")
        (home / "parts").mkdir(parents=True)
        self.job = {"slug": "show", "source": "show.mp4", "voices": ["actor"]}
        core.save_json(home / "job.json", self.job)
        core.use_job("show")
        self.part = {"name": "show-01", "start": 0, "end": 9}
        self.script = core.part_dir("show-01") / "script.json"
        core.save_json(
            core.part_dir("show-01") / "segments.json",
            {"job_id": "show-01", "source_lang": "ja", "target_lang": "en"},
        )
        core.save_json(self.script, {"lines": [LINE], "ready": True})

    def tearDown(self):
        core.use_job(None)
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_render_defaults_go_into_the_script_as_the_director_last_saved_it(self):
        real, edited = core.load_json, []

        def director_saves_after_the_read(path, default=None):
            value = real(path, default)
            if pathlib.Path(path) == self.script and not edited:
                edited.append(True)
                core.save_json(self.script, {"lines": [{**LINE, "text": "second words"}], "ready": True})
            return value

        with patch.object(runner, "load_json", director_saves_after_the_read):
            runner.variant(self.job, self.part, "actor")
        written = real(self.script)
        self.assertEqual(written["lines"][0]["text"], "second words")
        self.assertEqual(written["render"]["voice"], "actor")

    def test_an_unreadable_script_is_not_ready_and_says_why(self):
        self.script.write_text('{"lines": [,]}', encoding="utf-8")
        self.assertFalse(runner.ready(self.part))
        self.assertFalse(runner.original_only(self.part))
        self.assertIn("isn't valid JSON", runner.UNREADABLE["show-01"])
        with self.assertRaises(runner.ScriptError):
            runner.finished(self.part, "actor")
        core.save_json(self.script, {"lines": [LINE], "ready": True})
        self.assertTrue(runner.ready(self.part))
        self.assertEqual(runner.UNREADABLE, {})

    def test_a_script_without_lines_is_unreadable_too(self):
        core.save_json(self.script, {"ready": True})
        self.assertFalse(runner.ready(self.part))
        self.assertIn('"lines"', runner.UNREADABLE["show-01"])

    def test_status_and_report_name_an_unreadable_script(self):
        split = core.split_dir("show")
        core.save_json(split / "parts.json", {"parts": [self.part]})
        self.script.write_text("{", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            runner.cmd_status(types.SimpleNamespace(slug="show"))
            runner.report(self.job | {"title": "Show", "default": "actor"}, [self.part])
        self.assertIn("INVALID SCRIPT", out.getvalue())
        self.assertIn("Needs attention", out.getvalue())

    def test_run_without_watch_stops_for_the_director_instead_of_crashing(self):
        split = core.split_dir("show")
        core.save_json(split / "parts.json", {"parts": [self.part]})
        core.save_json(core.part_dir("show-01") / "draft.json", {"lines": []})
        self.script.write_text('{"lines": [', encoding="utf-8")
        args = types.SimpleNamespace(slug="show", watch=False)
        with (
            patch.object(runner.backend, "ensure", return_value=True),
            patch.object(runner, "second"),
            patch.object(runner, "prepare_step", return_value="done"),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            code = runner.cmd_run(args)
        self.assertEqual(code, 2)
        self.assertIn("isn't valid JSON", out.getvalue())
        self.assertIn("isn't valid JSON", json.dumps(core.load_json(core.job_dir("show") / "job.json")))

    def test_the_ready_mark_goes_only_on_the_script_that_was_checked(self):
        checked = {"lines": [LINE]}
        core.save_json(self.script, checked)
        runner.mark_ready("show-01", checked)
        self.assertEqual(core.load_json(self.script), {"lines": [LINE], "ready": True})
        core.save_json(self.script, {"lines": [{**LINE, "text": "second words"}]})
        with self.assertRaises(SystemExit) as stopped:
            runner.mark_ready("show-01", checked)
        self.assertIn("changed while it was checked", str(stopped.exception))
        self.assertNotIn("ready", core.load_json(self.script))


if __name__ == "__main__":
    unittest.main()
