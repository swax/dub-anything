"""Kept takes must be scored using the current language QA rules."""

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
import lineqa


class KeptTakeTests(unittest.TestCase):
    def check_kept(self, want, current, earlier, stale_notes, stale_ratio):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            work = root / "clip"
            (work / "takes").mkdir(parents=True)
            (work / "segments.json").write_text(json.dumps({"job_id": "job"}), encoding="utf-8")
            (work / "script.json").write_text(
                json.dumps({"lines": [{"voice": "speaker_a", "text": want, "start": 0.0, "end": 1.0}]}),
                encoding="utf-8",
            )
            wav = root / "current.wav"
            wav.write_bytes(b"current")
            (work / "takes" / "earlier.wav").write_bytes(b"earlier")
            kept = {
                "line": [
                    {
                        "got": earlier,
                        "notes": stale_notes,
                        "ratio": stale_ratio,
                        "sha": "old",
                        "settings": {},
                        "file": "earlier.wav",
                    }
                ]
            }
            (work / "takes" / "kept.json").write_text(json.dumps(kept), encoding="utf-8")
            with (
                patch.object(lineqa, "WORK", root),
                patch.object(lineqa, "langs", return_value=("en", "yue")),
                patch.object(lineqa, "line_id", return_value="line"),
                patch.object(lineqa, "take_wav", return_value=wav),
                patch.object(lineqa.sf, "info", return_value=types.SimpleNamespace(duration=1.0)),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                # check() hears each new take through the ear it is given (lineqa.Ear).
                lineqa.check("clip", lambda wav, lang: current, False, False)
            return wav.read_bytes()

    def test_recognized_written_variant_can_restore_an_earlier_take(self):
        result = self.check_kept("而家我哋出發。", "而家我哋出門。", "依家我地出發。", ["match 0.10"], 0.1)
        self.assertEqual(result, b"earlier")

    def test_stale_passing_score_cannot_restore_a_wrong_number(self):
        result = self.check_kept("我哋有三本書。", "我哋有三本新書。", "我哋有四本書。", [], 1.0)
        self.assertEqual(result, b"current")
