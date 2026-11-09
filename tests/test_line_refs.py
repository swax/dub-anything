"""Cached per-line references follow the actual voice sample used for a take."""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import core


class LineReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.work = self.root / "sample"
        self.work.mkdir()
        (self.root / "voices").mkdir()
        self.jobs = self.root / "jobs"
        (self.jobs / "sample").mkdir(parents=True)
        self.sr = 16000
        sf.write(self.jobs / "sample" / "vocals.wav", np.full(self.sr, 0.2), self.sr)
        (self.root / "voices.json").write_text(
            json.dumps({"speaker1": {"ref_text": "Hello."}, "speaker2": {"ref_text": "Goodbye."}}), encoding="utf-8"
        )
        self.write_voice("speaker1", 0.1)
        self.write_voice("speaker2", 0.4)
        self.line = {"start": 0.1, "end": 0.6, "voice": "speaker1", "ja": "Example source."}

    def write_voice(self, key, level):
        sf.write(self.root / "voices" / f"{key}.wav", np.full(self.sr // 2, level), self.sr)

    def reference(self, **options):
        with (
            patch.object(core, "WORK", self.root),
            patch.object(core, "JOBS", self.jobs),
            patch.object(core, "langs", return_value=("ja", "en")),
            patch.object(core, "unclean_windows", return_value=[(None, None)]),
        ):
            path, text = core.line_refs(self.work, "sample", [self.line], ["L00"], **options)["L00"]
        return pathlib.Path(path), text

    def test_rebuilt_voice_changes_reference_audio_and_cache_identity(self):
        old, _ = self.reference()
        self.write_voice("speaker1", 0.4)
        new, text = self.reference()
        self.assertNotEqual(old, new)
        self.assertTrue(old.exists())
        audio, _ = sf.read(new)
        self.assertAlmostEqual(float(audio[: self.sr // 2].mean()), 0.4, places=4)
        self.assertEqual(text, "Hello. Example source.")

    def test_recasting_the_same_window_uses_the_new_speaker(self):
        old, _ = self.reference()
        self.line["voice"] = "speaker2"
        new, text = self.reference()
        self.assertNotEqual(old, new)
        audio, _ = sf.read(new)
        self.assertAlmostEqual(float(audio[: self.sr // 2].mean()), 0.4, places=4)
        self.assertEqual(text, "Goodbye. Example source.")

    def test_unchanged_reference_is_reused(self):
        old, _ = self.reference()
        os.utime(old, (1234567890, 1234567890))
        new, _ = self.reference()
        self.assertEqual(old, new)
        self.assertEqual(new.stat().st_mtime, 1234567890)

    def test_source_only_contains_no_shared_audio_or_transcript(self):
        prefixed, _ = self.reference()
        own, text = self.reference(prefix=False)
        self.assertNotEqual(prefixed, own)
        audio, rate = sf.read(own)
        self.assertEqual(rate, self.sr)
        self.assertEqual(len(audio), self.sr // 2)
        self.assertAlmostEqual(float(audio.mean()), 0.2, places=4)
        self.assertEqual(text, "Example source.")

    def test_source_only_does_not_need_a_shared_voice_file(self):
        (self.root / "voices" / "speaker1.wav").unlink()
        own, text = self.reference(prefix=False)
        self.assertTrue(own.is_file())
        self.assertEqual(text, "Example source.")

    def test_line_can_opt_back_into_prefix_for_a_comparison(self):
        self.line["ref_prefix"] = True
        own, text = self.reference(prefix=False)
        audio, _ = sf.read(own)
        self.assertAlmostEqual(float(audio[: self.sr // 2].mean()), 0.1, places=4)
        self.assertEqual(text, "Hello. Example source.")

    def test_source_transcript_changes_invalidate_source_only_reference(self):
        old, _ = self.reference(prefix=False)
        self.line["ja"] = "Corrected source."
        new, text = self.reference(prefix=False)
        self.assertNotEqual(old, new)
        self.assertEqual(text, "Corrected source.")

    def test_source_only_still_rejects_crosstalk(self):
        with (
            patch.object(core, "WORK", self.root),
            patch.object(core, "JOBS", self.jobs),
            patch.object(core, "langs", return_value=("ja", "en")),
            patch.object(core, "unclean_windows", return_value=[("another speaker", None)]),
        ):
            self.assertEqual(core.line_refs(self.work, "sample", [self.line], ["L00"], prefix=False), {})

    def test_repeated_ids_cannot_borrow_another_occurrences_own_reference(self):
        for disabled in (False, True):
            second = {**self.line, "start": 0.6, "end": 0.9, "line_ref": not disabled}
            with (
                self.subTest(second_disabled=disabled),
                patch.object(core, "WORK", self.root),
                patch.object(core, "JOBS", self.jobs),
                patch.object(core, "langs", return_value=("ja", "en")),
                patch.object(core, "unclean_windows", return_value=[(None, None)] * 2),
                self.assertRaisesRegex(ValueError, "distinct take values"),
            ):
                core.line_refs(self.work, "sample", [self.line, second], ["shared", "shared"], prefix=False)

    def test_distinct_takes_keep_their_own_source_audio_for_identical_english(self):
        audio = np.full(self.sr, 0.2)
        audio[int(0.6 * self.sr) :] = 0.4
        sf.write(self.jobs / "sample" / "vocals.wav", audio, self.sr)
        lines = [{**self.line, "text": "Yes."}, {**self.line, "text": "Yes.", "start": 0.6, "end": 0.9, "take": 1}]
        with (
            patch.object(core, "WORK", self.root),
            patch.object(core, "JOBS", self.jobs),
            patch.object(core, "langs", return_value=("ja", "en")),
            patch.object(core, "unclean_windows", return_value=[(None, None)] * 2),
        ):
            ids = [core.line_id(line) for line in lines]
            refs = core.line_refs(self.work, "sample", lines, ids, prefix=False)
        self.assertEqual(len(refs), 2)
        means = [sf.read(refs[key][0])[0].mean() for key in ids]
        np.testing.assert_allclose(means, [0.2, 0.4], atol=1 / 32768)

    def test_explicit_fallback_repetitions_can_still_share_a_take(self):
        lines = [{**self.line, "line_ref": False}, {**self.line, "start": 0.6, "end": 0.9, "line_ref": False}]
        with (
            patch.object(core, "WORK", self.root),
            patch.object(core, "JOBS", self.jobs),
            patch.object(core, "langs", return_value=("ja", "en")),
            patch.object(core, "unclean_windows", return_value=[(None, None)] * 2),
        ):
            self.assertEqual(core.line_refs(self.work, "sample", lines, ["shared", "shared"], prefix=False), {})

    def test_fit_predicts_with_the_same_source_only_reference_as_generation(self):
        voices = json.loads((self.root / "voices.json").read_text(encoding="utf-8"))
        voices["speaker1"].update({"from": "sample", "clips": [[0, 0.5]]})
        (self.root / "voices.json").write_text(json.dumps(voices), encoding="utf-8")
        (self.work / "segments.json").write_text('{"job_id": "sample"}', encoding="utf-8")
        line = {**self.line, "text": "A synthetic sentence."}
        script = {"lines": [line], "render": {"line_ref": True, "ref_prefix": False}}
        (self.work / "script.json").write_text(json.dumps(script), encoding="utf-8")
        with (
            patch.object(core, "WORK", self.root),
            patch.object(core, "JOBS", self.jobs),
            patch.object(core, "langs", return_value=("ja", "en")),
            patch.object(core, "voice_wav", return_value=self.root / "voices/speaker1.wav"),
            patch.object(core, "unclean_windows", return_value=[(None, None)]),
            patch.object(core, "take_length", return_value=0.4) as estimate,
        ):
            rows = core.fit_rows("sample", predict=True)
        reference, transcript, text = estimate.call_args.args
        audio, _ = sf.read(reference)
        self.assertEqual(len(audio), self.sr // 2)
        self.assertAlmostEqual(float(audio.mean()), 0.2, places=4)
        self.assertEqual(transcript, "Example source.")
        self.assertEqual(text, line["text"])
        self.assertEqual(rows[0]["est"], 0.4)


if __name__ == "__main__":
    unittest.main()
