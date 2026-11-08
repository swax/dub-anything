"""First-take predictions must not slow unmeasured speech into an overflow."""

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from dub import _fill_speed, _override_speed
from dub_anything import ready_problems
import core


class FillSpeedTests(unittest.TestCase):
    def test_unmeasured_short_prediction_stays_at_natural_pace(self):
        self.assertEqual(_fill_speed(1.2, 2.0, power=1 / 0.7, measured=False), 1.0)

    def test_published_short_take_can_still_slow_down(self):
        self.assertEqual(_fill_speed(1.2, 2.0, power=1 / 0.7), 0.8)

    def test_unmeasured_long_prediction_can_speed_up(self):
        self.assertEqual(_fill_speed(2.5, 1.0, measured=False), 1.25)

    def test_measured_correction_keeps_previous_speed(self):
        self.assertAlmostEqual(_fill_speed(1.1, 1.0, previous=1.1), 1.21)

    def test_directed_pace_overrides_automatic_slowdown(self):
        segment, settings = {"speed": 0.8}, {"speed": 0.8, "steps": 32}
        _override_speed(segment, settings, 0.95)
        self.assertEqual(segment, {"speed": 0.95})
        self.assertEqual(settings, {"speed": 0.95, "steps": 32})

    def test_natural_pace_matches_published_settings_without_speed(self):
        segment, settings = {"speed": 0.8}, {"speed": 0.8, "steps": 32}
        _override_speed(segment, settings, 1.0)
        self.assertEqual(segment, {})
        self.assertEqual(settings, {"steps": 32})

    def test_directed_pace_rejects_invalid_values(self):
        for speed in (True, None, float("nan"), float("inf"), 0.7, 1.3, "fast"):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                _override_speed({}, {}, speed)

    def test_readiness_validates_directed_pace(self):
        for speed in (True, None, float("nan"), 0.7, 1.3):
            with self.subTest(speed=speed):
                self.assertTrue(ready_problems({"lines": [{"start": 0, "end": 1, "text": "Hi.", "speed": speed}]}))
        self.assertFalse(ready_problems({"lines": [{"start": 0, "end": 1, "text": "Hi.", "speed": 1.0}]}))

    def test_fit_does_not_credit_automatic_speed_when_pace_is_directed(self):
        script = {
            "lines": [{"start": 0, "end": 1, "voice": "voice", "text": "Hello.", "speed": 1.0}],
            "render": {"fill": True, "line_ref": False},
        }
        docs = {
            "segments.json": {"job_id": "job"},
            "script.json": script,
            "rendered.json": {},
            "voices.json": {"voice": {"clips": [], "ref_text": "Reference."}},
        }
        with (
            patch.object(core, "load_json", side_effect=lambda path, default=None: docs[path.name]),
            patch.object(core, "langs", return_value=("ja", "en")),
            patch.object(core, "line_id", return_value="line"),
            patch.object(core, "voice_wav", return_value=pathlib.Path("reference.wav")),
            patch.object(core, "take_wav", return_value=SimpleNamespace(exists=lambda: False)),
            patch.object(core, "take_length", return_value=1.5),
        ):
            row = core.fit_rows("clip", predict=True)[0]
        self.assertEqual(row["flag"], "OVER?")
        self.assertEqual(row["est"], 1.5)
