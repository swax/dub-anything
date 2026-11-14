"""First-take predictions must not slow unmeasured speech into an overflow."""

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from dub import _fill_speed, _fitting_published_pace, _override_speed, _published_remix_paces
from dub_anything import ready_problems
import core


class FillSpeedTests(unittest.TestCase):
    def test_final_remix_requires_every_published_take_to_fit(self):
        settings = {"a": {"steps": 32, "speed": 0.8}, "b": {"steps": 32, "speed": 0.8}}
        published = {"a": {"steps": 32}, "b": {"steps": 32}}
        rows = [{"i": 0, "speech": 0.8, "room": 1.3}, {"i": 1, "speech": 0.8, "room": 1.3}]
        self.assertEqual(_published_remix_paces(["a", "b"], settings, published, rows), {"a": 1.0, "b": 1.0})
        for bad in (None, 0, float("nan"), 1.29, 1.4):
            with self.subTest(bad=bad):
                self.assertIsNone(
                    _published_remix_paces(["a", "b"], settings, published, [rows[0], {**rows[1], "speech": bad}])
                )
        self.assertIsNone(_published_remix_paces(["a", "b"], settings, published, rows[:1]))

    def test_final_remix_cannot_reuse_another_lines_old_conditioning_or_quality(self):
        rows = [{"i": 0, "speech": 0.8, "room": 1.3}, {"i": 1, "speech": 0.8, "room": 1.3}]
        for field in ("line_ref", "fallback_sig", "steps", "seed", "effect", "instruct"):
            with self.subTest(field=field):
                settings = {"a": {"steps": 32, "speed": 0.8}, "b": {field: "new", "speed": 0.8}}
                published = {"a": {"steps": 32}, "b": {field: "old"}}
                self.assertIsNone(_published_remix_paces(["a", "b"], settings, published, rows))
        self.assertIsNone(_published_remix_paces(["a"], {"a": {"steps": 32}}, {}, rows))

    def test_final_remix_respects_directed_pace_and_validates_all_speeds(self):
        rows = [{"i": 0, "speech": 0.8, "room": 1.3}]
        settings, published = {"a": {"speed": 0.8}}, {"a": {"speed": 1.0}}
        self.assertIsNone(_published_remix_paces(["a"], settings, published, rows, directed={"a"}))
        for bad in (True, float("nan"), 0.7, 1.3, "fast"):
            with self.subTest(bad=bad):
                self.assertIsNone(_published_remix_paces(["a"], {"a": {"speed": bad}}, published, rows))
                self.assertIsNone(_published_remix_paces(["a"], settings, {"a": {"speed": bad}}, rows))

    def test_rejected_adjustment_can_reuse_fitting_published_pace(self):
        spec = {"steps": 32, "line_ref": "own"}
        self.assertEqual(_fitting_published_pace({**spec, "speed": 0.8}, spec, {"speech": 0.8, "room": 1.3}), 1.0)

    def test_recovery_requires_current_reference_and_other_settings(self):
        for field in ("line_ref", "fallback_ref", "steps", "seed", "effect"):
            with self.subTest(field=field):
                self.assertIsNone(
                    _fitting_published_pace({field: "new", "speed": 0.8}, {field: "old"}, {"speech": 0.8, "room": 1.3})
                )

    def test_recovery_requires_a_measured_take_inside_the_current_room(self):
        for speech in (None, 0, float("nan"), 1.29, 1.4):
            with self.subTest(speech=speech):
                self.assertIsNone(
                    _fitting_published_pace({"steps": 32, "speed": 0.8}, {"steps": 32}, {"speech": speech, "room": 1.3})
                )

    def test_unchanged_or_invalid_pace_is_not_an_automatic_adjustment(self):
        for old, new in ((1, 1), (True, 0.8), (1, float("nan")), (1.3, 0.8)):
            with self.subTest(old=old, new=new):
                self.assertIsNone(
                    _fitting_published_pace(
                        {"steps": 32, "speed": new}, {"steps": 32, "speed": old}, {"speech": 0.8, "room": 1.3}
                    )
                )

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
