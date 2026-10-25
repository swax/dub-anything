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

    def reference(self):
        with (
            patch.object(core, "WORK", self.root),
            patch.object(core, "JOBS", self.jobs),
            patch.object(core, "langs", return_value=("ja", "en")),
            patch.object(core, "unclean_windows", return_value=[(None, None)]),
        ):
            path, text = core.line_refs(self.work, "sample", [self.line], ["L00"])["L00"]
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


if __name__ == "__main__":
    unittest.main()
