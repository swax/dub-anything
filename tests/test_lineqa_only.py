"""Targeted experimental QA leaves unselected takes and queued repairs intact."""

import contextlib
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lineqa


class PartialCheckTests(unittest.TestCase):
    def test_seed_profiles_share_comparison_policy_without_changing_legacy_roll_keys(self):
        settings = {"steps": 32, "effect": "broadcast"}
        old = json.dumps({**settings, "speed": None, "seed": None}, sort_keys=True)
        expected = "L0:" + hashlib.sha1(old.encode()).hexdigest()[:8]
        self.assertEqual(lineqa.rolls_key("L0", settings), expected)
        self.assertEqual(lineqa.rolls_key("L0", {**settings, "seed": 42, "seed_profile": "private-a"}), expected)
        self.assertEqual(
            lineqa.comparable_settings({**settings, "seed_profile": "private-a"}),
            lineqa.comparable_settings({**settings, "seed_profile": "private-b"}),
        )

    def test_passing_selected_line_clears_only_its_queued_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "part"
            work.mkdir()
            (work / "segments.json").write_text('{"job_id":"job"}', encoding="utf-8")
            lines = [
                {"start": 0, "end": 1, "voice": "a", "text": "Hello."},
                {"start": 2, "end": 3, "voice": "b", "text": "Goodbye."},
            ]
            (work / "script.json").write_text(json.dumps({"lines": lines}), encoding="utf-8")
            untouched = {"id": "L2", "text": "Goodbye.", "got": "Wrong."}
            (work / "reroll.json").write_text(json.dumps([{"id": "L0"}, untouched]), encoding="utf-8")
            for lid in ("L0", "L2"):
                (root / (lid + ".wav")).write_bytes(lid.encode())
            ear = Mock(tag="fake", return_value="Hello.", load_s=0.0, hear_s=0.0)
            with (
                patch.object(lineqa, "WORK", root),
                patch.object(lineqa, "langs", return_value=("ja", "en")),
                patch.object(lineqa, "line_id", side_effect=lambda line: "L" + str(line["start"])),
                patch.object(
                    lineqa, "take_wav", side_effect=lambda job, lid, target, kind="seg": root / (lid + ".wav")
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                lineqa.check("part", ear, False, True, only={0})
            self.assertEqual(json.loads((work / "reroll.json").read_text()), [untouched])
            self.assertEqual(ear.call_count, 1)
            self.assertEqual(ear.call_args[0][0], root / "L0.wav")
            self.assertEqual((root / "L2.wav").read_bytes(), b"L2")


if __name__ == "__main__":
    unittest.main()
