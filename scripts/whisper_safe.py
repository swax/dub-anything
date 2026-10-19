"""Whisper word alignment without CTranslate2's short-window native crash.

CTranslate2 before 4.8.1 divides by zero when an alignment window has fewer
than two mel frames. A trailing Whisper segment can reach this even in a long
recording. Keep the existing CUDA/cuDNN setup and discard only those timings.
See https://github.com/OpenNMT/CTranslate2/pull/2065. On the CPU it also uses a
thread per physical core instead of faster-whisper's four.
"""

import logging
import os
from numbers import Integral

from faster_whisper import WhisperModel as BaseWhisperModel

LOG = logging.getLogger(__name__)


def whisper_on_gpu():
    """Whether Whisper runs on the GPU here. CTranslate2 has CUDA only: with an AMD
    GPU or on a Mac it runs on the CPU."""
    import ctranslate2

    return ctranslate2.get_cuda_device_count() > 0


def hold_gpu_for_whisper():
    """Wait for the GPU lock (gpulock.py) when Whisper will run on the GPU; on the
    CPU a re-listen needn't queue behind renders and preps."""
    from gpulock import hold_gpu

    if whisper_on_gpu():
        hold_gpu()


class WhisperModel(BaseWhisperModel):
    def __init__(self, *args, cpu_threads=0, **kwargs):
        # On the CPU (no CUDA: an AMD GPU, a Mac) faster-whisper uses 4 threads by
        # default; one per physical core transcribed 1.4x as fast on 12 cores.
        if not cpu_threads and not whisper_on_gpu():
            cpu_threads = max(4, (os.cpu_count() or 8) // 2)
        super().__init__(*args, cpu_threads=cpu_threads, **kwargs)

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
