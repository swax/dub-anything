"""Everything a job makes lives in its folder, work/jobs/<slug>/, what it can make again in bulk/;
clips prepped by hand keep work/<name>."""

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
import voicecast
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

    def test_new_job_keeps_what_it_can_make_again_in_bulk(self):
        home = self.new_job("show")
        job = runner.load_job("show")
        original = self.root / "show.mp4"
        self.assertEqual(pathlib.Path(job["video"]), home / "bulk" / "media" / "show.mp4")
        self.assertTrue(pathlib.Path(job["video"]).exists())
        # Delivery replaces its source: a working copy in bulk/, never the original.
        self.assertEqual(pathlib.Path(job["source"]), home / "bulk" / "media" / "source" / "show.mp4")
        self.assertTrue(pathlib.Path(job["source"]).exists())
        self.assertEqual(job["original"], str(original))
        self.assertTrue(original.exists())
        for folder in ("parts", "scratch"):
            self.assertTrue((home / folder).is_dir(), folder)
        for folder in ("media", "logs"):
            self.assertFalse((home / folder).exists(), folder)
        self.assertEqual(core.split_dir("show"), home / "bulk" / "split")
        self.assertEqual(core.bulk_of(core.part_dir("show-01")), home / "bulk" / "parts" / "01")
        self.assertEqual(core.bulk_of(core.part_dir("show-01.us")), home / "bulk" / "parts" / "01.us")
        clip = self.work / "clip"
        self.assertEqual(core.bulk_of(clip), clip)  # a clip prepped by hand keeps everything together

    def test_a_working_copy_inside_the_job_is_used_as_is(self):
        home = core.job_dir("show")
        copy = home / "bulk" / "media" / "source" / "show.mp4"
        copy.parent.mkdir(parents=True)
        copy.touch()
        argv = ["dub_anything.py", "new", str(copy), "--slug", "show"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main()
        job = runner.load_job("show")
        self.assertEqual(pathlib.Path(job["source"]), copy)
        self.assertNotIn("original", job)

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
        voicecast.update_voices(lambda v: v.update({"hero": {"from": "first-01", "clips": [[0, 1]]}}))
        self.assertEqual([n for n, _ in core.script_files()], ["first-01"])
        core.use_job("second")
        self.assertEqual(voicecast.load_voices(), {})
        self.assertEqual([n for n, _ in core.script_files()], ["second-01"])
        self.assertTrue((core.job_dir("first") / "voices.json").exists())
        self.assertFalse((self.work / "voices.json").exists())

    def test_a_later_job_borrows_an_earlier_cast_without_writing_it(self):
        self.new_job("ep01")
        core.use_job("ep01")
        voicecast.update_voices(lambda v: v.update({"hero": {"from": "ep01-01", "clips": [[0, 1]]}}))
        (core.job_dir("ep01") / "voices").mkdir()
        (core.job_dir("ep01") / "voices" / "hero.wav").write_bytes(b"RIFF")
        self.new_job("ep02", "--voices-from", "ep01")
        core.use_job("ep02")
        self.assertIn("hero", voicecast.load_voices())
        self.assertEqual(voicecast.voice_path("hero"), core.job_dir("ep01") / "voices" / "hero.wav")
        self.assertEqual(voicecast.voice_wav("hero"), core.job_dir("ep01") / "voices" / "hero.wav")
        with self.assertRaises(SystemExit):
            voice_tool.add("hero", "ep02-01", "[[0, 1]]", "words", False)
        voicecast.update_voices(lambda v: v.update({"guest": {"from": "ep02-01", "clips": [[0, 1]]}}))
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
        voicecast.update_voices(lambda v: v.update({"hero": {}, "rival": {}}))
        core.use_job(None)
        self.assertEqual(core.adopt_job(["hero", "rival"]), "ep02")
        core.use_job(None)
        self.assertIsNone(core.adopt_job(["hero", "stranger"]))
        self.assertEqual(core.adopt_job(["ep01-04", "hero"]), "ep01")

    def test_voicestudio_job_ids_are_named_for_their_part(self):
        self.assertEqual(core.new_vs_job_id("show-03.us"), "show-03-us")
        (self.jobs / "show-03").mkdir(parents=True)
        self.assertEqual(core.new_vs_job_id("show-03"), "show-03-2")

    def delivered_job(self, publish=True):
        home = self.new_job("show")
        split = home / "bulk" / "split"
        split.mkdir(parents=True)
        (split / "parts.json").write_text(json.dumps({"parts": [{"name": "show-01"}]}), encoding="utf-8")
        part = core.part_dir("show-01")
        part.mkdir(parents=True)
        (part / "script.json").write_text('{"lines": []}', encoding="utf-8")
        refs = core.bulk_of(part) / "refs"
        refs.mkdir(parents=True)
        (refs / "L01.wav").write_bytes(b"x" * 10)
        for name in ("show-01", "show-01-2", "showcase-01"):
            (self.jobs / name).mkdir(parents=True)
        job = runner.load_job("show")
        out = pathlib.Path(job["source"]).with_name("show (dubbed).mkv")
        out.write_bytes(b"dub")
        job["delivered"] = {"path": str(out)}
        if publish:
            with contextlib.redirect_stdout(io.StringIO()):
                job["delivered"]["published"] = runner.publish(job, out)
        runner.save_job(job)
        return home, part

    def cleanup(self, **flags):
        args = types.SimpleNamespace(**{"slug": "show", "delete": False, "force": False, **flags})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = runner.cmd_cleanup(args)
        return code, out.getvalue()

    def test_cleanup_lists_then_deletes_only_bulk_and_the_jobs_voicestudio_folders(self):
        home, part = self.delivered_job()
        published = self.root / "show (dubbed).mkv"
        self.assertEqual(published.read_bytes(), b"dub")
        code, listing = self.cleanup()
        self.assertEqual(code, 0)
        self.assertIn("published, SHA-256 matches", listing)
        self.assertTrue((home / "bulk").is_dir())  # listing alone deletes nothing
        code, _ = self.cleanup(delete=True)
        self.assertEqual(code, 0)
        self.assertFalse((home / "bulk").exists())
        self.assertEqual(sorted(p.name for p in self.jobs.iterdir()), ["showcase-01"])
        for kept in (part / "script.json", home / "scratch", home / "job.json", published, self.root / "show.mp4"):
            self.assertTrue(kept.exists(), kept)
        self.assertGreater(runner.load_job("show")["cleaned"]["freed"], 0)

    def test_cleanup_keeps_a_dub_that_exists_only_in_bulk(self):
        home, _ = self.delivered_job(publish=False)
        with self.assertRaises(SystemExit):
            self.cleanup(delete=True)
        self.assertTrue((home / "bulk").is_dir())
        code, _ = self.cleanup(delete=True, force=True)
        self.assertFalse((home / "bulk").exists())

    def test_cleanup_refuses_when_the_published_dub_changed(self):
        home, _ = self.delivered_job()
        (self.root / "show (dubbed).mkv").write_bytes(b"something else")
        with self.assertRaises(SystemExit):
            self.cleanup(delete=True)
        self.assertTrue((home / "bulk").is_dir())

    def test_publishing_replaces_only_its_own_earlier_copy(self):
        self.new_job("show")
        job = runner.load_job("show")
        out = pathlib.Path(job["source"]).with_name("show (dubbed).mkv")
        out.write_bytes(b"first")
        sidecar = out.with_name("show (dubbed).en.srt")
        sidecar.write_text("1", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            first = runner.publish(job, out)
            self.assertEqual([pathlib.Path(f["path"]).name for f in first["files"]], [out.name, sidecar.name])
            out.write_bytes(b"second")
            second = runner.publish(job, out, first)  # its own earlier delivery: replaced
            self.assertEqual((self.root / out.name).read_bytes(), b"second")
            (self.root / out.name).write_bytes(b"the user's own file")
            self.assertIsNone(runner.publish(job, out, second))  # changed since: left alone
        self.assertEqual((self.root / out.name).read_bytes(), b"the user's own file")
        self.assertFalse(list(self.root.glob(".*.partial")))

    def test_a_sidecar_conflict_is_found_before_publishing_the_video(self):
        self.new_job("show")
        job = runner.load_job("show")
        out = pathlib.Path(job["source"]).with_name("show (dubbed).mkv")
        out.write_bytes(b"dub")
        sidecar = out.with_suffix(".en.srt")
        sidecar.write_bytes(b"translation")
        unrelated = self.root / sidecar.name
        unrelated.write_bytes(b"the user's subtitles")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(runner.publish(job, out))
        self.assertFalse((self.root / out.name).exists())
        self.assertEqual(unrelated.read_bytes(), b"the user's subtitles")

    def test_publishing_resumes_after_a_copy_was_verified_but_not_recorded(self):
        self.new_job("show")
        job = runner.load_job("show")
        out = pathlib.Path(job["source"]).with_name("show (dubbed).mkv")
        out.write_bytes(b"dub")
        sidecar = out.with_suffix(".en.srt")
        sidecar.write_bytes(b"translation")
        target = self.root / out.name
        target.write_bytes(b"dub")
        before = target.stat().st_mtime_ns
        with contextlib.redirect_stdout(io.StringIO()):
            published = runner.publish(job, out)
        self.assertIsNotNone(published)
        self.assertEqual(target.stat().st_mtime_ns, before)
        self.assertEqual((self.root / sidecar.name).read_bytes(), b"translation")
        self.assertEqual(published["files"][0]["sha256"], runner.file_sha256(target))

    def test_a_destination_changed_during_copy_is_preserved(self):
        self.new_job("show")
        job = runner.load_job("show")
        out = pathlib.Path(job["source"]).with_name("show (dubbed).mkv")
        out.write_bytes(b"dub")
        target = self.root / out.name
        copy = runner.shutil.copy2

        def changed(source, destination):
            copy(source, destination)
            target.write_bytes(b"another file arrived")

        with patch.object(runner.shutil, "copy2", side_effect=changed), contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(runner.publish(job, out))
        self.assertEqual(target.read_bytes(), b"another file arrived")
        self.assertFalse(list(self.root.glob(".*.partial")))

    def test_failed_republication_keeps_the_previous_receipt_for_a_retry(self):
        self.new_job("show")
        job = runner.load_job("show")
        out = pathlib.Path(job["source"]).with_name("show (dubbed).mkv")
        out.write_bytes(b"first dub")
        sidecar = out.with_suffix(".en.srt")
        sidecar.write_bytes(b"first subtitles")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(runner.delivered(job, out, "first-state", {}))
            previous = job["delivered"]["published"]
            out.write_bytes(b"second dub")
            sidecar.write_bytes(b"second subtitles")
            target = self.root / sidecar.name
            target.write_bytes(b"the user's subtitles")
            self.assertFalse(runner.delivered(job, out, "second-state", job["delivered"]))
            saved = runner.load_job("show")
            self.assertNotIn("published", saved["delivered"])
            self.assertEqual(saved["delivered"]["publication_previous"], previous)
            self.assertEqual((self.root / out.name).read_bytes(), b"first dub")
            # The user resolves the conflict by restoring their earlier sidecar.
            target.write_bytes(b"first subtitles")
            self.assertTrue(runner.delivered(saved, out, "second-state", saved["delivered"]))
        self.assertEqual((self.root / out.name).read_bytes(), b"second dub")
        self.assertEqual(target.read_bytes(), b"second subtitles")
        self.assertNotIn("publication_previous", runner.load_job("show")["delivered"])

    def lend_hero(self, built=True):
        """ep01 with a voice cut from its first part, whose stems are gone (cleaned up); ep02 borrows it."""
        self.new_job("ep01")
        core.use_job("ep01")
        voicecast.update_voices(lambda v: v.update({"hero": {"from": "ep01-01", "clips": [[0, 1]]}}))
        core.save_json(core.part_dir("ep01-01") / "segments.json", {"job_id": "ep01-01"})
        wav = core.job_dir("ep01") / "voices" / "hero.wav"
        if built:
            wav.parent.mkdir()
            wav.write_bytes(b"RIFF")
        self.new_job("ep02", "--voices-from", "ep01")
        core.use_job("ep02")
        return wav

    def test_a_borrowed_voice_is_never_rebuilt_from_the_lenders_deleted_stems(self):
        wav = self.lend_hero()
        # What dub.ensure_voice asks for before it uploads a voice's profile.
        self.assertEqual(voicecast.voice_wav("hero", rebuild=True), wav)
        self.assertFalse((core.job_dir("ep02") / "voices").exists())

    def test_a_borrowed_voice_that_was_never_built_says_what_is_missing(self):
        self.lend_hero(built=False)
        with self.assertRaises(SystemExit) as stopped:
            voicecast.voice_wav("hero", rebuild=True)
        self.assertIn("ep01-01", str(stopped.exception))
        self.assertIn("borrowed", str(stopped.exception))

    def test_a_job_made_before_per_job_casts_lends_its_voices_from_the_shared_cast(self):
        old = core.job_dir("oldshow")
        old.mkdir(parents=True)
        core.save_json(old / "job.json", {"slug": "oldshow", "voices": ["actor"]})
        shared = {
            "hero": {"from": "oldshow-03", "clips": [[0, 1]]},
            "pooled": {"clips": [["oldshow-07", 2, 3]]},
            "stranger": {"from": "othershow-01", "clips": [[0, 1]]},
        }
        core.save_json(self.work / "voices.json", shared)
        (self.work / "voices").mkdir()
        (self.work / "voices" / "hero.wav").write_bytes(b"RIFF")
        self.assertEqual(core.job_layout("oldshow"), "flat")
        self.new_job("ep02", "--voices-from", "oldshow")
        core.use_job("ep02")
        self.assertEqual(sorted(voicecast.load_voices()), ["hero", "pooled"])
        self.assertEqual(voicecast.voice_wav("hero", rebuild=True), self.work / "voices" / "hero.wav")

    def earlier_job(self):
        """A delivered job as the version before bulk/ left it: media/, logs/ and each part's references in its folder."""
        home = self.new_job("show")
        (home / "bulk" / "media").rename(home / "media")
        (home / "bulk").rmdir()
        split = home / "media" / "split"
        split.mkdir()
        core.save_json(split / "parts.json", {"parts": [{"name": "show-01", "start": 0, "end": 9}]})
        (home / "logs").mkdir()
        (home / "logs" / "split.log").write_text("x", encoding="utf-8")
        part = home / "parts" / "01"
        (part / "refs").mkdir(parents=True)
        (part / "refs" / "L01.wav").write_bytes(b"x" * 10)
        (part / "script.json").write_text('{"lines": []}', encoding="utf-8")
        (self.jobs / "show-01").mkdir(parents=True)
        job = json.loads((home / "job.json").read_text(encoding="utf-8"))
        job["video"] = str(home / "media" / "show.mp4")
        job["delivered"] = {"path": str(self.root / "show (dubbed).mkv"), "at": "then"}
        core.save_json(home / "job.json", job)
        return home, part

    def test_a_job_made_before_bulk_still_finds_its_split_logs_and_references(self):
        home, part = self.earlier_job()
        self.assertEqual(core.job_layout("show"), "folder")
        self.assertEqual(core.split_dir("show"), home / "media" / "split")
        self.assertEqual(core.logs_dir("show"), home / "logs")
        self.assertEqual(core.bulk_of(part), part)
        self.assertEqual([p["name"] for p in runner.parts(runner.load_job("show"))], ["show-01"])

    def test_cleanup_of_a_job_made_before_bulk_deletes_what_bulk_holds_now(self):
        home, part = self.earlier_job()
        _, listing = self.cleanup()
        self.assertIn(str(home / "media"), listing)
        self.assertIn("1 folders", listing)
        self.assertTrue((part / "refs").exists())  # listing alone deletes nothing
        code, _ = self.cleanup(delete=True, force=True)
        self.assertEqual(code, 0)
        for gone in (home / "media", home / "logs", part / "refs", self.jobs / "show-01"):
            self.assertFalse(gone.exists(), gone)
        for kept in (part / "script.json", home / "job.json", home / "scratch"):
            self.assertTrue(kept.exists(), kept)

    def test_a_job_made_before_job_folders_is_reported_not_run(self):
        old = core.job_dir("oldshow")
        old.mkdir(parents=True)
        job = {"slug": "oldshow", "source": "x.mp4", "voices": ["actor"], "delivered": {"at": "then"}}
        core.save_json(old / "job.json", job)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            runner.cmd_status(types.SimpleNamespace(slug="oldshow"))
        self.assertIn("made before a job kept its work in its own folder", out.getvalue())
        self.assertNotIn("not split yet", out.getvalue())
        for command in (runner.cmd_run, runner.cmd_cleanup):
            with self.assertRaises(SystemExit) as stopped:
                command(types.SimpleNamespace(slug="oldshow", watch=False, delete=False, force=False))
            self.assertIn("made before", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
