"""Degenerate Whisper windows must never reach native alignment unguarded."""

import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from core import whisper_setup

whisper_setup()
from whisper_safe import BaseWhisperModel, WhisperModel  # noqa: E402


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        # No model download or GPU is needed to exercise the alignment boundary.
        self.model = object.__new__(WhisperModel)
        self.tokenizer = object()
        self.encoder = object()

    def align(self, tokens, frames):
        return self.model.find_alignment(self.tokenizer, tokens, self.encoder, frames)

    def test_short_scalar_windows_never_call_native_alignment(self):
        with patch.object(BaseWhisperModel, "find_alignment") as native:
            for frames in (0, 1):
                with self.subTest(frames=frames):
                    self.assertEqual(self.align([[42]], frames), [[]])
            native.assert_not_called()

    def test_valid_window_preserves_tokens_frames_and_timings(self):
        tokens, result = [[42]], [[{"word": "Example", "start": 0.1, "end": 0.3}]]
        with patch.object(BaseWhisperModel, "find_alignment", return_value=result) as native:
            self.assertIs(self.align(tokens, 2), result)
            native.assert_called_once_with(self.tokenizer, tokens, self.encoder, 2, 7)

    def test_short_batch_preserves_result_count_without_native_call(self):
        with patch.object(BaseWhisperModel, "find_alignment") as native:
            self.assertEqual(self.align([[42], [43]], [0, 1]), [[], []])
            self.assertEqual(self.align([], []), [])
            native.assert_not_called()

    def test_mixed_batch_keeps_valid_rows_and_removes_short_rows_text(self):
        with patch.object(BaseWhisperModel, "find_alignment", return_value=[[], ["valid"]]) as native:
            self.assertEqual(self.align([[42], [43]], [1, 30]), [[], ["valid"]])
            native.assert_called_once_with(self.tokenizer, [[], [43]], self.encoder, [2, 30], 7)

    def test_empty_text_has_no_word_timings(self):
        with patch.object(BaseWhisperModel, "find_alignment") as native:
            self.assertEqual(self.align([[]], 100), [[]])
            native.assert_not_called()

    def test_mismatched_batch_fails_before_native_code(self):
        with patch.object(BaseWhisperModel, "find_alignment") as native:
            with self.assertRaisesRegex(ValueError, "frame counts"):
                self.align([[42], [43]], [100])
            native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
