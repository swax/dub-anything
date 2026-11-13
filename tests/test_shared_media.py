"""Parts are separated once, by separate.py, and a second voice shares them.

VoiceStudio uploads a part without separating it; prep separates it with
Roformer and attaches the stems. The American voice forks the part's job, so
its media are hard links to the actor job's: stems must only ever be replaced,
never rewritten. After prep, the runner unloads the backend's models instead of
restarting it, unless that leaves memory held.
"""

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import backend
import core
import dub
import dub_anything as runner
import separate


class StemSwapTests(unittest.TestCase):
    def test_a_new_stem_replaces_a_hard_linked_one_without_writing_through_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "actor").mkdir()
            (root / "american").mkdir()
            (root / "actor" / "vocals.wav").write_bytes(b"shared")
            os.link(root / "actor" / "vocals.wav", root / "american" / "vocals.wav")
            new = root / "separated.wav"
            new.write_bytes(b"re-separated")

            separate.swap_in(new, root / "american" / "vocals.wav")

            self.assertEqual((root / "american" / "vocals.wav").read_bytes(), b"re-separated")
            self.assertEqual((root / "actor" / "vocals.wav").read_bytes(), b"shared")
            self.assertFalse(new.exists())
            self.assertEqual(sorted(p.name for p in (root / "american").iterdir()), ["vocals.wav"])

    def test_restore_without_htdemucs_stems_keeps_the_roformer_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = pathlib.Path(tmp)
            for name in ("vocals.wav", "no_vocals.wav"):
                (job / name).write_bytes(b"roformer")
            (job / "separator.json").write_text('{"model": "m"}', encoding="utf-8")
            with patch.object(separate, "job_dir", return_value=job), contextlib.redirect_stdout(io.StringIO()):
                separate.restore("sample-01")
            self.assertTrue((job / "separator.json").exists())
            self.assertEqual((job / "vocals.wav").read_bytes(), b"roformer")


class SeparateCommandTests(unittest.TestCase):
    def test_prep_and_the_runner_launch_the_same_separator(self):
        self.assertEqual(runner.SEPARATE, runner.gpu.separate_command())
        self.assertEqual(runner.SEPARATE[-2:], ["python", "scripts/separate.py"])


class ForkTests(unittest.TestCase):
    def test_another_voice_forks_the_parts_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            work, jobs = root / "work", root / "dub_jobs"
            base = work / "probe-01"
            base.mkdir(parents=True)
            doc = {"job_id": "probe-01", "video": "p.mp4", "source_lang": "zh", "target_lang": "en", "segments": [1]}
            (base / "segments.json").write_text(json.dumps(doc), encoding="utf-8")
            (base / "zh.srt").write_text("1\n", encoding="utf-8")
            (jobs / "probe-01").mkdir(parents=True)
            (jobs / "probe-01" / "separator.json").write_text('{"model": "m"}', encoding="utf-8")
            requests = MagicMock()

            def forked(url, json, timeout):
                (jobs / json["job_id"]).mkdir()
                return MagicMock(ok=True)

            requests.post.side_effect = forked
            with (
                patch.object(core, "WORK", work),
                patch.object(dub, "JOBS", jobs),
                patch.object(core, "JOBS", jobs),
                patch.object(dub, "requests", requests),
                patch.object(dub, "require_shared_media"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                dub.cmd_prep(types.SimpleNamespace(video=str(root / "probe-01.us.mp4"), subtitles_only=False))

            url, body = requests.post.call_args.args[0], requests.post.call_args.kwargs["json"]
            self.assertTrue(url.endswith("/dub/fork/probe-01"), url)
            self.assertEqual(body, {"job_id": "probe-01-us"})
            variant = core.load_json(work / "probe-01.us" / "segments.json")
            self.assertEqual(variant, {**doc, "job_id": "probe-01-us"})
            self.assertTrue((work / "probe-01.us" / "zh.srt").exists())
            self.assertTrue((jobs / "probe-01-us" / "separator.json").exists())

    def test_the_runner_forks_rather_than_uploading_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            (work / "sample-01").mkdir()
            script = {"lines": [{"start": 0, "end": 1, "voice": "a", "text": "Hi."}]}
            (work / "sample-01" / "script.json").write_text(json.dumps(script), encoding="utf-8")
            part = {"name": "sample-01", "file": str(work / "sample-01.mp4")}
            jobs = {"sample-01": work / "vs-actor", "sample-01.us": work / "vs-american"}
            for folder in jobs.values():
                folder.mkdir()
                (folder / "separator.json").write_text("{}", encoding="utf-8")

            def tool(job, label, command, logname):
                (work / "sample-01.us" / "segments.json").write_text("{}", encoding="utf-8")
                return True, "", {}

            with (
                patch.object(core, "WORK", work),
                patch.object(runner, "tool", side_effect=tool) as ran,
                patch.object(runner, "job_dir_of", side_effect=jobs.get),
            ):
                self.assertEqual(runner.variant({"slug": "sample"}, part, "american"), "sample-01.us")
            self.assertEqual(ran.call_args.args[2][1:], ["scripts/dub.py", "fork", "sample-01.us"])
            self.assertFalse((work / "sample-01.us.mp4").exists(), "no copy of the video to upload")

    def test_a_variant_uploaded_before_forks_without_stems_waits_for_the_director(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            for name in ("sample-01", "sample-01.us"):
                (work / name).mkdir()
            script = {"lines": [{"start": 0, "end": 1, "voice": "a", "text": "Hi."}]}
            (work / "sample-01" / "script.json").write_text(json.dumps(script), encoding="utf-8")
            (work / "sample-01.us" / "segments.json").write_text("{}", encoding="utf-8")
            (work / "vs-actor").mkdir()
            (work / "vs-actor" / "separator.json").write_text("{}", encoding="utf-8")
            (work / "vs-american").mkdir()
            jobs = {"sample-01": work / "vs-actor", "sample-01.us": work / "vs-american"}
            with (
                patch.object(core, "WORK", work),
                patch.object(runner, "job_dir_of", side_effect=jobs.get),
                patch.object(runner, "problem") as problem,
                patch.object(runner, "tool") as ran,
            ):
                self.assertIsNone(
                    runner.variant({"slug": "sample"}, {"name": "sample-01", "file": "p.mp4"}, "american")
                )
            ran.assert_not_called()
            self.assertIn("fork sample-01 again", problem.call_args.args[1])


class UnloadTests(unittest.TestCase):
    def unload(self, answer):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(answer).encode()
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            result = backend.unload()
        self.assertIn("unload_model=true", urlopen.call_args.args[0].full_url)
        return result

    def test_an_unload_that_hands_the_memory_back_is_enough(self):
        # Measured: after prep, an unload left 0.78 GB reserved, the same after four preps.
        self.assertTrue(self.unload({"flushed": True, "vram_reserved": 0.78}))

    def test_memory_still_held_or_no_answer_calls_for_a_restart(self):
        self.assertFalse(self.unload({"flushed": True, "vram_reserved": 3.2}))
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertFalse(backend.unload())

    def test_the_runner_restarts_only_when_unloading_is_not_enough(self):
        for unloaded, restarts in ((True, 0), (False, 1)):
            with (
                self.subTest(unloaded=unloaded),
                patch.dict(runner.BACKEND, {"dirty": True}),
                patch.object(runner.backend, "unload", return_value=unloaded),
                patch.object(runner.backend, "restart", return_value=True) as restart,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                runner.clean_backend({"slug": "sample"})
                self.assertEqual(restart.call_count, restarts)
                self.assertFalse(runner.BACKEND["dirty"])


if __name__ == "__main__":
    unittest.main()
