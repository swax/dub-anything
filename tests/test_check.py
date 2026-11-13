"""check.py: one report of a part's pre-render checks, what must be fixed, and the ready mark."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import check
import core


def row(i, flag="", window=2.0, **extra):
    return {"i": i, "start": i * 3.0, "text": f"Line {i}.", "window": window, "room": window + 0.5, "actor": 1.5,
            "speech": None, "est": 1.0, "flag": flag, **extra}  # fmt: skip


class Part:
    """A part in a temporary work folder, with the analysis tools mocked."""

    def __init__(self, test, lines, job=None, segments=None):
        self.test = test
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        self.work = self.root / "sample-01"
        self.work.mkdir()
        (self.work / "segments.json").write_text(json.dumps(segments or {"job_id": "job"}), encoding="utf-8")
        self.script = self.work / "script.json"
        self.script.write_text(json.dumps({"lines": lines}), encoding="utf-8")
        (self.root / "voices.json").write_text(json.dumps({"speaker": {"ref_text": "Identity."}}), encoding="utf-8")
        self.job = job
        self.fit_calls = []
        self.edge_calls = []

    def run(self, *args, rows=(), notes=None, reading=()):
        def fit_rows(name, predict, short, voice, notes):
            self.fit_calls.append(voice)
            notes.update(notes_by_voice.get(voice, {}))
            return [dict(r) for r in rows]

        notes_by_voice = {"actor": notes or {}}
        output = io.StringIO()
        with (
            patch.object(check, "WORK", self.root),
            patch.object(check.runner, "WORK", self.root),
            patch.object(check, "job_of", return_value=self.job),
            patch.object(check.edges, "analyse", side_effect=lambda n, a: self.edge_calls.append(a) or ([], [], [])),
            patch.object(check.tails, "rows", return_value=[]),
            patch.object(check.fit, "pace_warnings", return_value=[]),
            patch.object(check.fit, "reading_rows", return_value=list(reading)),
            patch.object(check, "fit_rows", side_effect=fit_rows),
            patch.object(sys, "argv", ["check.py", "sample-01", *args]),
            contextlib.redirect_stdout(output),
        ):
            with self.test.assertRaises(SystemExit) as exit_:
                check.main()
        return exit_.exception.code, output.getvalue()

    def ready(self):
        return bool(json.loads(self.script.read_text(encoding="utf-8")).get("ready"))


LINES = [
    {"start": i * 3.0, "end": i * 3.0 + 2, "voice": "speaker", "ja": "Source.", "text": f"Line {i}."} for i in range(3)
]


class MustFixTests(unittest.TestCase):
    def test_overflow_and_risk_on_a_short_window_must_be_fixed(self):
        self.assertTrue(check.fit_must(row(0, "OVER?")))
        self.assertTrue(check.fit_must(row(0, "OVER")))
        self.assertTrue(check.fit_must(row(0, "risk", window=0.8)))
        self.assertFalse(check.fit_must(row(0, "risk", window=1.4)))
        self.assertFalse(check.fit_must(row(0, "SHORT?")))
        self.assertFalse(check.fit_must(row(0, "tight")))


class ReportTests(unittest.TestCase):
    def test_flags_are_marked_and_unflagged_lines_left_out(self):
        part = Part(self, LINES)
        code, out = part.run(rows=[row(0, "OVER?"), row(1), row(2, "SHORT?")])
        self.assertEqual(code, 2)
        self.assertIn("actor: 1 to fix, 1 to review", out)
        self.assertIn("! L00", out)
        self.assertIn("  L02", out)
        self.assertNotIn("L01", out)
        self.assertIn("next: fix the 1 marked !", out)
        self.assertEqual(part.edge_calls, [True])

    def test_fallback_reasons_print_under_their_line_even_when_it_fits(self):
        part = Part(self, LINES)
        notes = {1: ["line ref skipped: voice nearer someone"]}
        code, out = part.run(rows=[row(0), row(1), row(2)], notes=notes)
        self.assertEqual(code, 0)
        self.assertIn("1 on the character's reference", out)
        self.assertIn("L01", out)
        self.assertIn("      line ref skipped: voice nearer someone", out)
        self.assertIn("check.py sample-01 --ready", out)

    def test_every_voice_of_the_job_is_estimated(self):
        part = Part(self, LINES, job={"voices": ["actor", "american"]})
        part.run(rows=[row(0)])
        self.assertEqual(part.fit_calls, ["actor", "american"])

    def test_dry_run_leaves_the_windows(self):
        part = Part(self, LINES)
        part.run("--dry-run", rows=[row(0)])
        self.assertEqual(part.edge_calls, [False])

    def test_structural_problems_stop_the_other_checks(self):
        part = Part(self, [{**LINES[0], "voice": "nobody"}])
        code, out = part.run("--ready", rows=[row(0)])
        self.assertEqual(code, 2)
        self.assertIn("! lines[0].voice must name a key in voices.json", out)
        self.assertEqual(part.edge_calls, [])
        self.assertEqual(part.fit_calls, [])
        self.assertFalse(part.ready())

    def test_subtitles_check_reading_speed_only(self):
        part = Part(self, LINES, segments={"job_id": "job", "subtitles_only": True})
        reading = [{"i": 0, "tag": "L00", "start": 0, "end": 2, "shown": 2, "cps": 30, "flag": "FAST", "text": "x"},
                   {"i": 1, "tag": "L01", "start": 3, "end": 5, "shown": 5, "cps": 10, "flag": "SPLIT?", "text": "y"}]  # fmt: skip
        code, out = part.run(reading=reading)
        self.assertEqual(code, 2)
        self.assertIn("reading: 2 to fix, 0 to review", out)
        self.assertEqual(part.edge_calls, [])
        self.assertEqual(part.fit_calls, [])


class ReadyTests(unittest.TestCase):
    def test_ready_marks_a_part_with_nothing_to_fix(self):
        part = Part(self, LINES)
        code, out = part.run("--ready", rows=[row(0, "SHORT?")])
        self.assertEqual(code, 0)
        self.assertTrue(part.ready())
        self.assertIn("ready:", out)

    def test_ready_waits_for_the_fixes(self):
        part = Part(self, LINES)
        code, _ = part.run("--ready", rows=[row(0, "OVER?")])
        self.assertEqual(code, 2)
        self.assertFalse(part.ready())


class LineRefNotesTests(unittest.TestCase):
    def test_notes_collect_by_line_instead_of_printing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            part = root / "part"
            (part / "refs").mkdir(parents=True)
            (root / "jobs" / "job").mkdir(parents=True)
            sf.write(root / "jobs" / "job" / "vocals.wav", np.full(8 * 16000, 0.2, dtype="float32"), 16000)
            (root / "voices.json").write_text(json.dumps({"speaker": {"ref_text": "Identity."}}), encoding="utf-8")
            lines = [{"start": 0.0, "end": 2.0, "voice": "speaker", "ja": "One."},
                     {"start": 3.0, "end": 5.0, "voice": "speaker", "ja": "Two."}]  # fmt: skip
            notes, output = {}, io.StringIO()
            with (
                patch.object(core, "WORK", root),
                patch.object(core, "JOBS", root / "jobs"),
                patch.object(core, "langs", return_value=("ja", "en")),
                patch.object(core, "unclean_windows", return_value=[(None, None), ("voice nearer other", None)]),
                contextlib.redirect_stdout(output),
            ):
                refs = core.line_refs(part, "job", lines, ["L0", "L1"], prefix=False, notes=notes)
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(notes, {1: ["line ref skipped: voice nearer other"]})
            self.assertEqual(set(refs), {"L0"})


if __name__ == "__main__":
    unittest.main()
