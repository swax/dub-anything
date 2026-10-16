"""Whisper word alignment without CTranslate2's short-window native crash.

CTranslate2 before 4.8.1 divides by zero when an alignment window has fewer
than two mel frames. A trailing Whisper segment can reach this even in a long
recording. Keep the existing CUDA/cuDNN setup and discard only those timings.
See https://github.com/OpenNMT/CTranslate2/pull/2065.
"""

import logging
from numbers import Integral

from faster_whisper import WhisperModel as BaseWhisperModel

LOG = logging.getLogger(__name__)


class WhisperModel(BaseWhisperModel):
    def find_alignment(self, tokenizer, text_tokens, encoder_output, num_frames, median_filter_width=7):
        if not text_tokens:
            return []
        # faster-whisper uses a scalar for ordinary transcription and a list
        # for batched transcription. Preserve one result for each token list.
        frames = [num_frames] * len(text_tokens) if isinstance(num_frames, Integral) else list(num_frames)
        usable = [count >= 2 and bool(tokens) for count, tokens in zip(frames, text_tokens)]
        if len(frames) != len(text_tokens):
            raise ValueError("alignment frame counts must match the token batch")
        if not all(usable):
            LOG.warning("Skipping word alignment for %s empty or sub-two-frame window(s)", usable.count(False))
            if not any(usable):
                return [[] for _ in text_tokens]
            # Empty text produces no word timings in faster-whisper. Padding its
            # already padded encoder window to two frames also keeps older
            # CTranslate2 safe when other rows in the batch still need alignment.
            text_tokens = [tokens if valid else [] for tokens, valid in zip(text_tokens, usable)]
            num_frames = [max(2, count) for count in frames]
        return super().find_alignment(tokenizer, text_tokens, encoder_output, num_frames, median_filter_width)
