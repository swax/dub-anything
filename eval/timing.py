"""Timing of rendered parts against the actors: where each take ends compared
with the actor's last sound in its window, and the speeds fill chose.

Usage: eval/timing.py <clip> [<clip> ...]   (after dub.py generate)

Per clip it prints the lines with their own reference (per-line references),
how many takes fill left at each speed, and the distance d between the actor's
last sound (Silero VAD on the separated vocals, as fill sees it) and the end of
the take (its trimmed speech, as VoiceStudio times it):
  dead air  the actor still speaking after the English stopped (sum of d > 0)
  overrun   the English running on past the actor (sum of -d > 0)
  |end|     mean and median distance
  early/late  takes ending more than fill's tolerance (0.25 s) before/after
Lines with no take, or where the VAD hears nothing, are left out. Compare
renders of one part made from the same script (A/B copies, see research.md).
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from core import FILL_SPEED, FILL_TOL, actor_ends, langs, line_id, load_json, part_dir, speech_length, take_wav


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    slow, fast = FILL_SPEED
    for name in sys.argv[1:]:
        work = part_dir(name)
        lines = load_json(work / "script.json")["lines"]
        job = load_json(work / "segments.json")["job_id"]
        rendered = load_json(work / "rendered.json", {})
        ids = [line_id(line) for line in lines]
        own = sum(1 for i in ids if rendered.get(i, {}).get("line_ref"))
        speeds = [rendered.get(i, {}).get("speed", 1.0) for i in ids]
        tgt = langs(name)[1]
        d = np.array(
            [
                end - speech_length(take_wav(job, i, tgt))
                for i, end in zip(ids, actor_ends(job, lines))
                if end is not None and take_wav(job, i, tgt).exists()
            ]
        )
        if not len(d):
            print(f"{name}: no rendered takes")
            continue
        print(
            f"{name}: {len(lines)} lines, {own} with their own reference; speed <={slow} "
            f"{sum(s <= slow for s in speeds)}, <1 {sum(slow < s < 1 for s in speeds)}, "
            f"1 {sum(s == 1 for s in speeds)}, >1 {sum(1 < s < fast for s in speeds)}, "
            f">={fast} {sum(s >= fast for s in speeds)}; dead air {d.clip(0).sum():.1f} s, "
            f"overrun {(-d).clip(0).sum():.1f} s, |end| mean {np.abs(d).mean():.2f} "
            f"median {np.median(np.abs(d)):.2f} s, early {(d > FILL_TOL).sum()}, late {(d < -FILL_TOL).sum()}"
        )


if __name__ == "__main__":
    main()
