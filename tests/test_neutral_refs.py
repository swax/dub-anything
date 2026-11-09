"""Reference selection avoids expressive outliers and preserves source routing."""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import core
import dub_anything
import neutral_refs


class NeutralRankingTests(unittest.TestCase):
    def test_opposing_extremes_cannot_cancel_to_neutral(self):
        norms = {k: (0.0, 1.0) for k in ("pitch", "effort", "level")}
        ordinary = neutral_refs.score({"pitch": 0.2, "effort": -0.2, "level": 0.0}, 4, norms)
        extremes = neutral_refs.score({"pitch": 4.0, "effort": -4.0, "level": 0.0}, 4, norms)
        self.assertLess(ordinary["distance"], extremes["distance"])
        self.assertEqual(extremes["largest_deviation"], 4)

    def test_listener_benchmark_can_define_the_conversational_target(self):
        norms = {k: (0.0, 1.0) for k in ("pitch", "effort", "level")}
        target = {"pitch": -0.5, "effort": -0.7, "level": -0.2}
        matched = neutral_refs.score(target, 4, norms, target)
        raised = neutral_refs.score({"pitch": 1.5, "effort": 1.5, "level": 1.5}, 4, norms, target)
        self.assertEqual(matched["distance"], 0)
        self.assertLess(matched["distance"], raised["distance"])

    def test_no_pitch_is_not_interpreted_as_neutral(self):
        self.assertIsNone(
            neutral_refs.score(
                {"pitch": float("nan"), "effort": 0, "level": 0},
                4,
                neutral_refs.baseline([{k: 0 for k in ("pitch", "effort", "level")}]),
            )
        )


class FallbackRoutingTests(unittest.TestCase):
    def setUp(self):
        self.voices = {"a": {}, "a_neutral": {"register_of": "a"}, "a_scene": {"register_of": "a"}, "b": {}}
        self.line = {"voice": "a"}
        self.mode = {"fallback_voices": {"a": "a_neutral"}}

    def test_fallback_stays_with_same_speaker(self):
        self.assertEqual(core.fallback_voice(self.line, self.mode, self.voices), "a_neutral")
        with self.assertRaisesRegex(ValueError, "speaker's family"):
            core.fallback_voice(self.line, {"fallback_voices": {"a": "b"}}, self.voices)

    def test_scene_override_and_legacy_default(self):
        self.assertEqual(
            core.fallback_voice({**self.line, "fallback_voice": "a_scene"}, self.mode, self.voices), "a_scene"
        )
        self.assertEqual(core.fallback_voice(self.line, {}, self.voices), "a")

    def test_own_reference_ignores_fallback_and_keeps_cache_identity(self):
        own = ("L00_1.00-5.00_own123.wav", "Source words")
        with patch.object(core, "voice_sig", side_effect=AssertionError("fallback fingerprint not needed")):
            self.assertEqual(core.reference_settings(self.line, self.mode, own), {"line_ref": "1.00-5.00_own123"})

    def test_changing_only_fallback_invalidates_its_signature(self):
        with (
            patch.object(core, "load_json", return_value=self.voices),
            patch.object(core, "voice_sig", return_value="old"),
        ):
            old = core.reference_settings(self.line, self.mode)
        with (
            patch.object(core, "load_json", return_value=self.voices),
            patch.object(core, "voice_sig", return_value="new"),
        ):
            new = core.reference_settings(self.line, self.mode)
        self.assertNotEqual(core.reference_signature(old), core.reference_signature(new))
        self.assertEqual(self.line, {"voice": "a"})

    def test_preflight_rejects_missing_and_cross_speaker_fallbacks(self):
        for selected in ("missing", "b", None, 42):
            with self.subTest(selected=selected):
                problems = dub_anything.ready_problems(
                    {"lines": [], "render": {"fallback_voices": {"a": selected}}}, self.voices
                )
                self.assertTrue(problems)
        self.assertEqual(dub_anything.ready_problems({"lines": [], "render": self.mode}, self.voices), [])

    def test_fit_repredicts_changed_fallback_but_measures_unchanged_own_take(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            work = root / "sample"
            work.mkdir()
            lines = [
                {"voice": "a", "text": "First sentence.", "start": 0, "end": 2},
                {"voice": "a", "text": "Second sentence.", "start": 3, "end": 5},
            ]
            voices = {
                "a": {"clips": [[0, 4]]},
                "a_neutral": {"clips": [[0, 4]], "register_of": "a", "ref_text": "Reference."},
            }
            (root / "voices.json").write_text(json.dumps(voices), encoding="utf-8")
            (work / "script.json").write_text(
                json.dumps({"lines": lines, "render": {**self.mode, "line_ref": True}}), encoding="utf-8"
            )
            (work / "segments.json").write_text('{"job_id":"sample"}', encoding="utf-8")
            with (
                patch.object(core, "WORK", root),
                patch.object(core, "langs", return_value=("ja", "en")),
                patch.object(core, "voice_sig", return_value="fingerprint"),
                patch.object(core, "voice_wav", return_value=root / "neutral.wav"),
                patch.object(core, "take_wav", side_effect=lambda job, sid, tgt: root / f"{sid}.wav"),
                patch.object(core, "speech_length", return_value=1.5) as measured,
                patch.object(core, "take_length", return_value=2.2) as predicted,
            ):
                ids = [core.line_id(line) for line in lines]
                own = (str(work / "L00_source.wav"), "Original.")
                (work / "rendered.json").write_text(
                    json.dumps({ids[0]: core.reference_settings(lines[0], self.mode, own), ids[1]: {}}),
                    encoding="utf-8",
                )
                for sid in ids:
                    (root / f"{sid}.wav").write_bytes(b"unchanged cached take")
                with patch.object(core, "line_refs", return_value={ids[0]: own}):
                    rows = core.fit_rows("sample", predict=True)
                measured.assert_called_once_with(root / f"{ids[0]}.wav")
                predicted.assert_called_once_with(root / "neutral.wav", "Reference.", "Second sentence.", lang="en")
                self.assertEqual(rows[0]["speech"], 1.5)
                self.assertNotIn("stale_reference", rows[0])
                self.assertTrue(rows[1]["stale_reference"])
                self.assertIsNone(rows[1]["speech"])
                self.assertEqual(rows[1]["est"], 2.2)
                self.assertEqual((root / f"{ids[0]}.wav").read_bytes(), b"unchanged cached take")


if __name__ == "__main__":
    unittest.main()
