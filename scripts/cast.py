"""Group the lines of one or more clips by voice, to cast consistently across parts.

Usage: scripts/cast.py <name> [<name> ...] [--min 0.9] [--threshold 0.38] [--cast key,key,...]   (after draft.py)

Every draft line of at least --min seconds gets a speaker embedding (WeSpeaker,
as in draft.py). Average-linkage clustering by cosine similarity then puts one
voice in each cluster. Clusters are printed with their lines and the nearest
existing cast member, so a recurring character keeps one voice and a new one is
identified once (from the dialogue and frames.py), not part by part. Short
lines end up as singletons: settle them with draft.py once the cast is built.
"""

import argparse
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import draft
from core import JOBS, WORK, langs, load_json, said

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cluster(sim, threshold):
    """Average linkage: merge while two clusters' mean similarity is at least threshold."""
    if len(sim) < 2:
        return [list(range(len(sim)))]
    dist = np.clip(1 - sim, 0, 2)
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), "average"), 1 - threshold, "distance")
    groups = {}
    for k, label in enumerate(labels):
        groups.setdefault(label, []).append(k)
    return sorted(groups.values(), key=len, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--min", type=float, default=0.9, help="shortest line to embed (s)")
    ap.add_argument("--threshold", type=float, default=0.38, help="merge clusters above this similarity")
    ap.add_argument("--cast", help="comma-separated voice keys for nearest-cast hints (default: all)")
    args = ap.parse_args()

    embed = draft.embedder()
    if embed is None:
        raise SystemExit("needs the pyannote models (docs/setup.md, step 5)")
    cast = draft.cast_centroids(embed, set(args.cast.split(",")) if args.cast else None)
    items = []
    for name in args.names:
        job = load_json(WORK / name / "segments.json")["job_id"]
        audio = draft.decode_audio(str(JOBS / job / "vocals.wav"), sampling_rate=draft.SR)
        for i, line in enumerate(load_json(WORK / name / "draft.json")["lines"]):
            if line["end"] - line["start"] >= args.min and not line.get("suspect") and not line.get("song"):
                v = embed(audio[int(line["start"] * draft.SR) : int(line["end"] * draft.SR)])
                items.append((name, i, line, v))
    if not items:
        raise SystemExit("no lines to cluster (run draft.py first)")
    vecs = np.stack([it[3] for it in items])
    for c, members in enumerate(cluster(vecs @ vecs.T, args.threshold)):
        centre = vecs[members].mean(0)
        centre /= np.linalg.norm(centre)
        near = max(((float(centre @ v), k) for k, v in cast.items()), default=None)
        hint = f"nearest cast: {near[1]} {near[0]:.2f}" if near else "no cast yet"
        print(f"\n=== C{c}  {len(members)} lines  {hint}")
        for k in sorted(members, key=lambda k: (items[k][0], items[k][2]["start"])):
            name, i, line, v = items[k]
            print(
                f"  {name} L{i:02d} {line['start']:7.2f}-{line['end']:7.2f} {float(v @ centre):4.2f}  "
                f"{said(line, langs(name)[0])}"
            )


if __name__ == "__main__":
    main()
