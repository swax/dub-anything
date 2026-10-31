"""Explicit subtitle placement and the existing automatic sign avoidance."""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import ass


class DialogueLayoutTests(unittest.TestCase):
    def events(self, dialogue, signs=(), layout=()):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "subtitles.ass"
            ass.write(path, 1000, 1000, dialogue, signs, subtitle_layout=layout)
            return [l for l in path.read_text(encoding="utf-8-sig").splitlines() if l.startswith("Dialogue: 0,")]

    def test_defaults_and_automatic_sign_avoidance_are_unchanged(self):
        (plain,) = self.events([(0, 2, "Hello.")])
        self.assertTrue(plain.endswith(",,Hello."))
        sign = {"start": 0, "end": 2, "text": "Platform", "box": [0.3, 0.85, 0.7, 0.97]}
        (moved,) = self.events([(0, 2, "Hello.")], [sign])
        self.assertTrue(moved.endswith("{\\an8}Hello."))
        top = {**sign, "box": [0.3, 0.02, 0.7, 0.15]}
        (blocked,) = self.events([(0, 2, "Hello.")], [sign, top])
        self.assertEqual(blocked, plain)

    def test_explicit_layout_overrides_signs_and_only_intersecting_cues(self):
        sign = {"start": 0, "end": 5, "text": "Platform", "box": [0.3, 0.85, 0.7, 0.97]}
        layout = [{"start": 2, "end": 4, "top": 0.19}, {"start": 3, "end": 4, "top": 0.25}]
        before, placed, last, after = self.events(
            [(0, 2, "Before."), (2, 3, "Placed."), (3, 4, "Last."), (4, 5, "After.")], [sign], layout
        )
        self.assertTrue(before.endswith("{\\an8}Before."))
        self.assertTrue(placed.endswith("{\\an8\\pos(500,190)}Placed."))
        self.assertTrue(last.endswith("{\\an8\\pos(500,250)}Last."))
        self.assertTrue(after.endswith("{\\an8}After."))


class SignMotionTests(unittest.TestCase):
    def sign(self, end_box, **extra):
        return {
            "start": 10,
            "end": 15,
            "text": "Room",
            "box": [0.3, 0.2, 0.35, 0.25],
            "box_end": end_box,
            **extra,
        }

    def test_small_pans_follow_pixel_positions_on_both_axes(self):
        for box, destination in [
            ([0.305, 0.2, 0.355, 0.25], "634,243"),
            ([0.3, 0.205, 0.35, 0.255], "624,248"),
        ]:
            with self.subTest(box=box):
                event = ass.sign_event(self.sign(box), 1920, 1080)
                self.assertIn(f"\\move(624,243,{destination})", event)

    def test_subpixel_motion_with_identical_rounded_anchors_stays_static(self):
        event = ass.sign_event(self.sign([0.30001, 0.2, 0.35001, 0.25]), 1920, 1080)
        self.assertIn("\\pos(624,243)", event)
        self.assertNotIn("\\move", event)

    def test_small_pan_keeps_explicit_motion_interval(self):
        event = ass.sign_event(self.sign([0.305, 0.2, 0.355, 0.25], seen=[11, 14]), 1920, 1080)
        self.assertIn("\\move(624,243,634,243,1000,4000)", event)


if __name__ == "__main__":
    unittest.main()
