"""Reference-word margins flag nonverbal tails without changing conditioning."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import core
from reference_review import audible_tail, speech_margins


class SpeechMarginTests(unittest.TestCase):
    def test_words_then_long_untranscribed_tail_need_review(self):
        result = speech_margins([{"words": [(0.0, 0.6, "A"), (0.6, 1.8, "complete sentence.")]}], 3.8)
        self.assertTrue(result["review_tail"])
        self.assertEqual(result["trailing_seconds"], 2.0)

    def test_normal_padding_and_long_utterance_with_small_relative_tail_pass(self):
        for duration, last in ((4.0, 3.7), (10.0, 9.0)):
            self.assertFalse(speech_margins([{"words": [(0, last, "Words.")]}], duration)["review_tail"])

    def test_missing_or_unusable_alignment_does_not_claim_a_nonverbal_tail(self):
        for phrases in ([{"said": "Words without timings."}], [{"words": [(0, float("nan"), "Bad.")]}], []):
            self.assertEqual(speech_margins(phrases, 4), {"aligned_words": 0, "review_tail": False})

    def test_later_phrase_prevents_false_tail_after_an_internal_pause(self):
        result = speech_margins([{"words": [(0, 1, "First.")]}, {"words": [(3, 3.9, "Last.")]}], 4)
        self.assertFalse(result["review_tail"])
        self.assertEqual(result["last_word"], 3.9)

    def test_model_numpy_timestamps_remain_json_serializable(self):
        result = speech_margins([{"words": [(np.float64(0), np.float64(1.8), "Sentence.")]}], np.float64(3.8))
        restored = json.loads(json.dumps(result))
        self.assertIs(restored["review_tail"], True)

    def test_quiet_padding_and_small_alignment_error_do_not_prompt_review(self):
        audio = np.zeros(3800, dtype="float32")
        audio[:2000] = 0.1
        margins = speech_margins([{"words": [(0, 1.8, "Sentence.")]}], 3.8)
        result = audible_tail(audio, 1000, margins)
        self.assertFalse(result["review_tail"])
        self.assertEqual(result["tail_active_seconds"], 0)

    def test_audible_nonword_tail_and_stereo_remain_reviewable(self):
        audio = np.full((3800, 2), 0.1, dtype="float32")
        margins = speech_margins([{"words": [(0, 1.8, "Sentence.")]}], 3.8)
        result = audible_tail(audio, 1000, margins)
        self.assertTrue(result["review_tail"])
        self.assertGreater(result["tail_active_seconds"], 1.5)
        json.dumps(result)

    def test_low_noise_without_audible_energy_does_not_prompt_review(self):
        audio = np.full(3800, 1e-6, dtype="float32")
        margins = speech_margins([{"words": [(0, 1.8, "Sentence.")]}], 3.8)
        self.assertFalse(audible_tail(audio, 1000, margins)["review_tail"])


class ReferenceWarningTests(unittest.TestCase):
    def test_warning_preserves_exact_source_and_ignores_stale_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            part = root / "part"
            (part / "refs").mkdir(parents=True)
            job = root / "jobs" / "job"
            job.mkdir(parents=True)
            audio = np.full(4 * 16000, 0.2, dtype="float32")
            sf.write(job / "vocals.wav", audio, 16000)
            (root / "voices.json").write_text(json.dumps({"speaker": {"ref_text": "Identity."}}))
            line = {"start": 0.0, "end": 4.0, "voice": "speaker", "ja": "Source."}
            stat = (job / "vocals.wav").stat()
            source = {"job": "job", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            with (
                patch.object(core, "WORK", root),
                patch.object(core, "JOBS", root / "jobs"),
                patch.object(core, "langs", return_value=("ja", "en")),
                patch.object(core, "unclean_windows", return_value=[(None, None)]),
            ):
                key = core.source_check_keys(part, [line])[0]
                report = {key: {"source": source, "review_tail": True, "last_word": 2, "trailing_seconds": 2}}
                path = part / "refs" / "source-review.json"
                path.write_text(json.dumps(report))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    own = core.line_refs(part, "job", [line], ["L00"], prefix=False)
                self.assertIn("reference review at 0.0", output.getvalue())
                self.assertIn("L00", own)
                got, _ = sf.read(own["L00"][0], dtype="float32")
                original, _ = sf.read(job / "vocals.wav", dtype="float32")
                np.testing.assert_array_equal(got, original)
                report[key]["source"]["mtime_ns"] -= 1
                path.write_text(json.dumps(report))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    core.line_refs(part, "job", [line], ["L00"], prefix=False)
                self.assertNotIn("reference review", output.getvalue())


if __name__ == "__main__":
    unittest.main()
