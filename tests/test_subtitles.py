"""Subtitle jobs: creating them, preparing their parts without the backend, timing
cues for reading, and delivering the subtitles into the source."""

import contextlib
import io
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import deliver
import dub_anything as runner
import subtitles
from languages import LANGUAGES
from languages.latin import two_lines


def new_job(root, *flags):
    source = root / "sample.mp4"
    source.touch()
    work = root / "work"
    argv = ["dub_anything.py", "new", str(source), *flags]
    with (
        patch.object(runner, "VIDEOS", root / "videos"),
        patch.object(runner, "JOB_DIR", work / "jobs"),
        patch.object(sys, "argv", argv),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        with unittest.TestCase().assertRaises(SystemExit) as stopped:
            runner.main()
        if stopped.exception.code:
            return stopped.exception.code
        return runner.load_job("sample")


class SubtitleJobTests(unittest.TestCase):
    def test_no_voices_makes_a_subtitle_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = new_job(pathlib.Path(tmp), "--voices", "none", "--signs")
            self.assertEqual(job["voices"], [])
            self.assertIsNone(job["default"])
            self.assertTrue(runner.subtitles_only(job))

    def test_subtitles_need_only_both_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Korean into Japanese is no dub pair, but both languages have a module.
            job = new_job(pathlib.Path(tmp), "--voices", "none", "--source", "ko", "--target", "ja")
            self.assertEqual(job["langs"], ["ko", "ja"])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn(
                "subtitles from ko into xx",
                str(new_job(pathlib.Path(tmp), "--voices", "none", "--source", "ko", "--target", "xx")),
            )
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("the scripts know", str(new_job(pathlib.Path(tmp), "--source", "ko", "--target", "ja")))

    def test_parts_prep_from_their_audio_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            job = new_job(root, "--voices", "none")
            with (
                patch.object(runner, "WORK", root / "work"),
                patch.object(runner, "tool", return_value=(True, "")) as tool,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(runner.prepare_step(job, [{"name": "sample-01", "file": "p.mp4"}]), "busy")
            self.assertIn("--subtitles-only", tool.call_args.args[2])
            self.assertFalse(runner.BACKEND["dirty"])

    def test_a_ready_part_is_final_without_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "sample-01").mkdir()
            script = {"lines": [{"start": 1, "end": 2, "ja": "はい", "text": "Yes."}], "ready": True}
            (root / "sample-01" / "script.json").write_text(json.dumps(script), encoding="utf-8")
            part = {"name": "sample-01"}
            with patch.object(runner, "WORK", root):
                self.assertTrue(runner.ready(part))
                self.assertTrue(all(runner.finished(part, v) for v in []))


class TimingTests(unittest.TestCase):
    def test_short_cue_stays_up_long_enough_to_read(self):
        (cue,) = subtitles.timed([(10.0, 10.3, "Yes.")], "en")
        self.assertGreaterEqual(cue[1] - cue[0], subtitles.MIN_SHOW)

    def test_long_text_gets_reading_time_but_not_past_the_next_cue(self):
        text = "x" * 85  # five seconds at 17 a second
        first, second = subtitles.timed([(0.0, 1.0, text), (3.0, 4.0, "Next.")], "en")
        self.assertAlmostEqual(first[1], 3.0 - subtitles.GAP, places=3)
        self.assertEqual(second[0], 3.0)

    def test_small_gaps_close_instead_of_blinking(self):
        first, _ = subtitles.timed([(0.0, 2.0, "One."), (2.9, 4.0, "Two.")], "en")
        self.assertAlmostEqual(first[1], 2.9 - subtitles.GAP, places=3)

    def test_overlapping_cues_stay_as_they_are(self):
        cues = [(0.0, 3.0, "Over a long line."), (1.0, 2.0, "A caption under it.")]
        self.assertEqual(subtitles.timed(cues, "en")[0], cues[0])

    def test_long_speech_keeps_its_own_length(self):
        (cue,) = subtitles.timed([(0.0, 9.0, "A long speech.")], "en")
        self.assertEqual(cue[1], 9.0)

    def test_rows_flag_fast_and_long_lines(self):
        lines = [
            {"start": 0.0, "end": 0.9, "text": "This is far too much to read in under a second."},
            {"start": 0.95, "end": 2.0, "text": "Fine."},
            {"start": 5.0, "end": 13.0, "text": " ".join(["word"] * 30)},
        ]
        rows = subtitles.rows(lines, "en")
        self.assertIn("FAST", rows[0]["flag"])
        self.assertEqual(rows[1]["flag"], "")
        self.assertIn("LONG", rows[2]["flag"])
        self.assertIn("SPLIT?", rows[2]["flag"])


class LineBreakTests(unittest.TestCase):
    def test_short_text_stays_on_one_line(self):
        self.assertEqual(two_lines("A short line."), "A short line.")

    def test_long_text_breaks_near_the_middle(self):
        text = "This bento comes from a shop that has made it by hand since the war."
        top, bottom = two_lines(text).split("\n")
        self.assertLessEqual(abs(len(top) - len(bottom)), 12)
        self.assertEqual(f"{top} {bottom}", text)

    def test_break_prefers_punctuation(self):
        text = "Here we are, at the station where the line ends today."
        self.assertEqual(two_lines(text).split("\n")[0], "Here we are, at the station")

    def test_written_breaks_stay(self):
        self.assertEqual(LANGUAGES["en"].subtitle("One\nTwo"), "One\nTwo")


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
class DeliveryTests(unittest.TestCase):
    def test_subtitles_go_into_matroska_with_an_srt_beside_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "Show - Episode 1.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=128x72:duration=2",
                            "-f", "lavfi", "-i", "sine=duration=2", "-c:v", "libx264", "-c:a", "aac",
                            "-shortest", str(source)], check=True)  # fmt: skip
            base = root / "show.subtitled.mkv"
            cue = "1\n00:00:00,500 --> 00:00:01,500\n{}\n\n"
            base.with_suffix(".en.srt").write_text(cue.format("Hello."), encoding="utf-8")
            base.with_suffix(".ja.srt").write_text(cue.format("こんにちは"), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                out = deliver.deliver_subtitles(source, base, ("ja", "en"))
                self.assertEqual(out.name, "Show - Episode 1 (subtitled).mkv")
                self.assertFalse(source.exists())
                self.assertTrue(out.with_suffix(".en.srt").exists())
                streams = deliver.probe(out)["streams"]
                titles = [deliver.name_of(s) for s in streams if s["codec_type"] == "subtitle"]
                self.assertEqual(titles, ["English (subtitles)", "日本語 (transcript)"])
                audio = next(s for s in streams if s["codec_type"] == "audio")
                self.assertEqual(audio["tags"]["language"], "jpn")
                # Delivering again replaces its own tracks rather than adding more.
                out = deliver.deliver_subtitles(out, base, ("ja", "en"))
                subs = [s for s in deliver.probe(out)["streams"] if s["codec_type"] == "subtitle"]
                self.assertEqual(len(subs), 2)


if __name__ == "__main__":
    unittest.main()
