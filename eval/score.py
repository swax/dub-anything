"""Score renders of the same part(s) against the original actors, line by line.

Usage: eval/score.py NAME=clip[,clip...] NAME=clip[,clip...] ...   (after dub.py generate)

Each NAME is a variant: one render of the listed parts, all made from the same
script (A/B copies of a part, each its own VoiceStudio job; see research.md).
The first variant's clips supply the originals, and the n-th clip of every
variant must be the same part. Only lines present in every variant are scored.
Needs a GPU (SQUIM, Whisper).

Per variant:
  arousal, pitch, effort  rank correlation between each take and the actor's
                          original line, per character, weighted by lines
                          (register.py's measures; characters with at least
                          5 measurable lines and a register norm)
  like      WeSpeaker similarity of the takes to the actor's own lines
  early/late  takes ending more than 0.5 s before/after the actor's last sound
  |end|     mean distance from it (s)
  PESQ, SDR   torchaudio SQUIM estimates of each take's quality
  floor     quietest vs loud frames inside the speech (dB): noise between words
  P(lang)   Whisper's confidence that the take is in the dub's language (P(en),
            or P(ja) dubbing into Japanese): near 1 for native speech, well
            below for a clone that keeps a non-native accent
The actor's reference for likeness comes from their lines in the show's main
parts (named <show>-NN); A/B copies (a name with a further suffix) are left out.
"""

import pathlib
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import draft
import register as R
from core import langs, line_id, load_json, part_dir, speech_length, take_wav
from gpulock import hold_gpu
from faster_whisper.audio import decode_audio

SR = 16000
MAIN_PART = re.compile(r"^[^-]+-\d+$")  # <show>-NN; A/B copies carry a further suffix


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    return np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1] if len(a) >= 4 else np.nan


def floor_db(x):
    n = len(x) // 320
    db = 10 * np.log10((x[: n * 320].reshape(n, 320) ** 2).mean(axis=1) + 1e-12)
    on = np.nonzero(db > db.max() - 40)[0]
    span = db[on[0] : on[-1] + 1] if len(on) else db
    return np.percentile(span, 10) - np.percentile(span, 95)


def main():
    variants = {k: v.split(",") for k, v in (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    if not variants:
        raise SystemExit(__doc__)
    origs = next(iter(variants.values()))
    tgt = langs(origs[0])[1]
    show = R.Show()
    base = lambda v: show.base.get(v, v)  # noqa: E731  (a register key counts as its character)

    # Every part's lines, pooled: (part index, line); the originals' features and actor ends.
    lines, vocals, spans = [], {}, {}
    for p, clip in enumerate(origs):
        vocals[p] = R.vocals(clip)
        spans[p] = draft.vad_spans(vocals[p])
        lines += [(p, line) for line in load_json(part_dir(clip) / "script.json")["lines"]]
    vmaps = {
        name: [
            {round(line["start"], 2): line for line in load_json(part_dir(c) / "script.json")["lines"]} for c in clips
        ]
        for name, clips in variants.items()
    }
    keep = [k for k, (p, line) in enumerate(lines) if all(round(line["start"], 2) in m[p] for m in vmaps.values())]
    orig = []
    for p, line in lines:
        x = vocals[p][int(line["start"] * SR) : int(line["end"] * SR)]
        inside = [min(b, line["end"]) for a, b in spans[p] if b > line["start"] and a < line["end"]]
        orig.append((base(line["voice"]), R.features(x), (max(inside) - line["start"]) if inside else None))
    embed = draft.embedder()
    cent = {}
    for b in {c for c, *_ in orig}:
        vecs = [
            embed(R.vocals(clip)[int(line["start"] * SR) : int(line["end"] * SR)])
            for clip, _, line, _ in show.lines.get(b, [])[:40]
            if line["end"] - line["start"] >= 1.0 and MAIN_PART.match(clip)
        ]
        if len(vecs) >= 3:
            c = np.mean(vecs, axis=0)
            cent[b] = c / np.linalg.norm(c)
    chars = [b for b in cent if b in show.norm and sum(1 for k in keep if orig[k][0] == b and orig[k][1]) >= 5]

    from torchaudio.pipelines import SQUIM_OBJECTIVE

    hold_gpu()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    squim = SQUIM_OBJECTIVE.get_model().to(dev).eval()
    whisper = draft.WhisperModel("Systran/faster-whisper-large-v3", device="auto", compute_type="default")

    print(
        f"{','.join(origs)}: {len(keep)} lines; "
        + ", ".join(f"{b} {sum(1 for k in keep if orig[k][0] == b)}" for b in chars)
    )
    print(
        f"{'variant':12s} {'arousal':>7s} {'pitch':>6s} {'effort':>6s} {'like':>5s} {'early':>5s} {'late':>4s} "
        f"{'|end|':>5s} {'PESQ':>4s} {'SDR':>5s} {'floor':>6s} {f'P({tgt})':>5s}   per character arousal"
    )
    for name, clips in variants.items():
        jobs = [load_json(part_dir(c) / "segments.json")["job_id"] for c in clips]
        take_lines = {k: vmaps[name][lines[k][0]][round(lines[k][1]["start"], 2)] for k in keep}
        feats, sims, ends, q = {}, [], [], []
        for k in keep:
            wav = take_wav(jobs[lines[k][0]], line_id(take_lines[k]), tgt)
            t = decode_audio(str(wav), sampling_rate=SR)
            feats[k] = R.features(t)
            b = orig[k][0]
            if b in cent and len(t) > SR:
                sims.append(float(embed(t) @ cent[b]))
            if orig[k][2] is not None:
                ends.append(speech_length(wav) - orig[k][2])
            with torch.no_grad():
                _, pesq, sdr = (float(v) for v in squim(torch.from_numpy(t)[None].to(dev)))
            _, info = whisper.transcribe(t, language=None, beam_size=1, without_timestamps=True)
            q.append((pesq, sdr, floor_db(t), dict(info.all_language_probs or []).get(tgt, 0)))
        rho, prh, erh, n, per = [], [], [], [], []
        for b in chars:
            idx = [k for k in keep if orig[k][0] == b and orig[k][1] and feats[k]]
            takes_norm = R.norms([feats[k] for k in idx])
            rho.append(
                spearman(
                    [R.arousal(orig[k][1], show.norm[b]) for k in idx], [R.arousal(feats[k], takes_norm) for k in idx]
                )
            )
            prh.append(spearman([orig[k][1]["pitch"] for k in idx], [feats[k]["pitch"] for k in idx]))
            erh.append(spearman([orig[k][1]["effort"] for k in idx], [feats[k]["effort"] for k in idx]))
            n.append(len(idx))
            per.append(f"{b} {rho[-1]:+.2f}")
        w = np.array(n) / sum(n) if n else np.array([])
        ends, q = np.array(ends), np.array(q)
        print(
            f"{name:12s} {np.dot(w, rho) if n else np.nan:+7.2f} {np.dot(w, prh) if n else np.nan:+6.2f} "
            f"{np.dot(w, erh) if n else np.nan:+6.2f} {np.mean(sims):5.2f} {(ends < -0.5).sum():5d} "
            f"{(ends > 0.5).sum():4d} {np.mean(np.abs(ends)):5.2f} {q[:, 0].mean():4.2f} {q[:, 1].mean():5.1f} "
            f"{q[:, 2].mean():6.1f} {q[:, 3].mean():5.2f}   " + ", ".join(per)
        )
    print(
        "arousal/pitch/effort: rank correlation with the original line, per character, weighted; like: WeSpeaker\n"
        "similarity to the actor's own lines; early/late: takes ending >0.5 s before/after the actor's last sound;\n"
        "|end|: mean distance from it (s); PESQ/SDR: SQUIM estimates; floor: quietest vs loud in-speech frames (dB)"
    )


if __name__ == "__main__":
    main()
