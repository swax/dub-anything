"""Everything a job makes lives in its folder, work/jobs/<slug>/; clips prepped by hand keep work/<name>."""

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
import voices as voice_tool


class JobLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.work = self.root / "work"
        self.jobs = self.root / "vs_jobs"
        self.patches = [
            patch.object(core, "ROOT", self.root),
            patch.object(core, "WORK", self.work),
            patch.object(core, "JOBS", self.jobs),
            patch.dict(runner.ENV),
        ]
        for p in self.patches:
            p.start()
        core.use_job(None)

    def tearDown(self):
        core.use_job(None)
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def new_job(self, slug, *flags):
        source = self.root / f"{slug}.mp4"
        source.touch()
        argv = ["dub_anything.py", "new", str(source), "--slug", slug, *flags]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                runner.main()
        self.assertEqual(stopped.exception.code, 0)
        core.use_job(None)
        return core.job_dir(slug)

    def test_new_job_keeps_its_media_logs_and_scratch_in_its_folder(self):
        home = self.new_job("show")
        job = runner.load_job("show")
        self.assertEqual(pathlib.Path(job["video"]), home / "media" / "show.mp4")
        self.assertTrue(pathlib.Path(job["video"]).exists())
        for folder in ("media", "parts", "logs", "scratch"):
            self.assertTrue((home / folder).is_dir(), folder)
        self.assertEqual(core.split_dir("show"), home / "media" / "split")
        self.assertFalse((self.root / "videos").exists())

    def test_part_names_resolve_into_the_job_and_back(self):
        home = self.new_job("show")
        self.assertEqual(core.part_dir("show-03"), home / "parts" / "03")
        self.assertEqual(core.part_dir("show-03.us"), home / "parts" / "03.us")
        self.assertEqual(core.part_name(home / "parts" / "03.us"), "show-03.us")
        self.assertEqual(core.active_job(), "show")
        # A clip prepped by hand, or a name whose job doesn't exist, keeps work/<name>.
        self.assertEqual(core.part_dir("clip"), self.work / "clip")
        self.assertEqual(core.part_dir("other-01"), self.work / "other-01")
        self.assertEqual(core.split_dir("film"), self.root / "videos" / "film-parts")

    def test_each_job_has_its_own_cast_and_lists_only_its_parts(self):
        self.new_job("first")
        self.new_job("second")
        for name in ("first-01", "second-01"):
            core.part_dir(name).mkdir(parents=True)
            (core.part_dir(name) / "script.json").write_text('{"lines": []}', encoding="utf-8")
        core.use_job("first")
        core.update_voices(lambda v: v.update({"hero": {"from": "first-01", "clips": [[0, 1]]}}))
        self.assertEqual([n for n, _ in core.script_files()], ["first-01"])
        core.use_job("second")
        self.assertEqual(core.load_voices(), {})
        self.assertEqual([n for n, _ in core.script_files()], ["second-01"])
        self.assertTrue((core.job_dir("first") / "voices.json").exists())
        self.assertFalse((self.work / "voices.json").exists())

    def test_a_later_job_borrows_an_earlier_cast_without_writing_it(self):
        self.new_job("ep01")
        core.use_job("ep01")
        core.update_voices(lambda v: v.update({"hero": {"from": "ep01-01", "clips": [[0, 1]]}}))
        (core.job_dir("ep01") / "voices").mkdir()
        (core.job_dir("ep01") / "voices" / "hero.wav").write_bytes(b"RIFF")
        self.new_job("ep02", "--voices-from", "ep01")
        core.use_job("ep02")
        self.assertIn("hero", core.load_voices())
        self.assertEqual(core.voice_path("hero"), core.job_dir("ep01") / "voices" / "hero.wav")
        self.assertEqual(core.voice_wav("hero"), core.job_dir("ep01") / "voices" / "hero.wav")
        with self.assertRaises(SystemExit):
            voice_tool.add("hero", "ep02-01", "[[0, 1]]", "words", False)
        core.update_voices(lambda v: v.update({"guest": {"from": "ep02-01", "clips": [[0, 1]]}}))
        self.assertNotIn("guest", json.loads((core.job_dir("ep01") / "voices.json").read_text(encoding="utf-8")))

    def test_voices_from_must_name_an_existing_job(self):
        source = self.root / "x.mp4"
        source.touch()
        argv = ["dub_anything.py", "new", str(source), "--slug", "x", "--voices-from", "missing"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as stopped:
            runner.main()
        self.assertIn("missing", str(stopped.exception.code))

    def test_key_only_tools_find_the_job_whose_cast_has_the_keys(self):
        self.new_job("ep01")
        self.new_job("ep02")
        core.use_job("ep02")
        core.update_voices(lambda v: v.update({"hero": {}, "rival": {}}))
        core.use_job(None)
        self.assertEqual(core.adopt_job(["hero", "rival"]), "ep02")
        core.use_job(None)
        self.assertIsNone(core.adopt_job(["hero", "stranger"]))
        self.assertEqual(core.adopt_job(["ep01-04", "hero"]), "ep01")

    def test_voicestudio_job_ids_are_named_for_their_part(self):
        self.assertEqual(core.new_vs_job_id("show-03.us"), "show-03-us")
        (self.jobs / "show-03").mkdir(parents=True)
        self.assertEqual(core.new_vs_job_id("show-03"), "show-03-2")

    def test_cleanup_stages_media_logs_takes_and_every_voicestudio_folder(self):
        home = self.new_job("show")
        split = home / "media" / "split"
        split.mkdir(parents=True)
        (split / "parts.json").write_text(json.dumps({"parts": [{"name": "show-01"}]}), encoding="utf-8")
        part = core.part_dir("show-01")
        (part / "takes").mkdir(parents=True)
        (part / "segments.json").write_text('{"job_id": "show-01"}', encoding="utf-8")
        (part / "script.json").write_text('{"lines": []}', encoding="utf-8")
        for name in ("show-01", "show-01-2", "showcase-01"):
            (self.jobs / name).mkdir(parents=True)
        job = runner.load_job("show")
        job["delivered"] = {"path": "x"}
        runner.save_job(job)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.cmd_cleanup(types.SimpleNamespace(slug="show", force=False))
        hold = self.root / "videos" / "_to-delete" / "show"
        self.assertEqual(sorted(p.name for p in (hold / "dub_jobs").iterdir()), ["show-01", "show-01-2"])
        self.assertTrue((self.jobs / "showcase-01").exists())
        self.assertTrue((hold / "media" / "split" / "parts.json").exists())
        self.assertTrue((hold / "media" / "show.mp4").exists())
        self.assertTrue((hold / "parts" / "show-01" / "takes").is_dir())
        self.assertTrue((part / "script.json").exists())
        self.assertTrue((home / "scratch").is_dir())


if __name__ == "__main__":
    unittest.main()
