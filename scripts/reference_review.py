"""Screen recognized word boundaries without treating them as acting judgments."""

import math


def speech_margins(phrases, duration):
    """Summarize untranscribed margins; missing alignment stays unknown.

    Word timestamps can be early or omit quiet speech. A large trailing margin
    is a review prompt for silence, laughter, crosstalk or missed words, never
    an instruction to trim the reference or replace it automatically.
    """
    words = []
    for phrase in phrases:
        for start, end, text in phrase.get("words", []):
            if text.strip() and all(math.isfinite(t) for t in (start, end)) and 0 <= start < end <= duration + 0.1:
                words.append((start, min(end, duration)))
    if not words:
        return {"aligned_words": 0, "review_tail": False}
    first = float(min(a for a, _ in words))
    last = float(max(b for _, b in words))
    tail = duration - last
    return {
        "aligned_words": len(words),
        "first_word": round(first, 3),
        "last_word": round(last, 3),
        "leading_seconds": round(first, 3),
        "trailing_seconds": round(tail, 3),
        "review_tail": bool(tail >= 0.8 and tail >= 0.25 * duration),
    }


def audible_tail(audio, sample_rate, margins):
    """Whether a candidate margin has sustained audio after alignment slack.

    Quiet window padding should not create a review prompt. This energy screen
    also cannot distinguish laughter, noise and missed words; the director must.
    """
    import numpy as np

    if not margins.get("review_tail"):
        return {"review_tail": False, "tail_active_seconds": 0.0}
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    hop = max(1, round(sample_rate * 0.02))
    frames = len(mono) // hop
    if not frames:
        return {"review_tail": False, "tail_active_seconds": 0.0}
    db = 10 * np.log10(np.mean(mono[: frames * hop].reshape(-1, hop).astype(float) ** 2, axis=1) + 1e-12)
    # Word alignment often runs early. A quarter-second allowance avoids
    # counting an ordinary word's remaining tail as a separate sound.
    begin = math.ceil((margins["last_word"] + 0.25) * sample_rate / hop)
    active = float(np.count_nonzero(db[begin:] > max(float(db.max()) - 25, -60)) * hop / sample_rate)
    return {"review_tail": active >= 0.35, "tail_active_seconds": round(active, 3)}
