"""On-screen text: find the signs, labels and title cards in a clip's picture.

Usage: scripts/signs.py <clip> [--fps 2] [--from T0] [--to T1]   (after `dub.py prep`; subs.py's environment, see AGENTS.md)
       scripts/signs.py <clip> --preview [SIGN ...]

Candidates for the director to translate into a part's "signs" (see
docs/directing.md, "Signs"). Most text in a picture is clutter; the director
reads the sheets and keeps what the viewer needs.

1. Shot cuts come from ffmpeg's scene detection: a sign usually shows for
   exactly its shot.
2. Frames are sampled (2 a second) and text is detected and read anywhere in
   them with subs.py's engine. Closed captions (light text on a broadcaster's
   semi-transparent box) and text that matches the dialogue in script.json or
   draft.json are left out; a title card on black stays in.
3. Text close together in a frame is one block; a block followed across the
   frames of its shot is one candidate, with the shot's times when it shows
   from cut to cut, its box at its first and last sighting (a pan moves it),
   and the reading seen most often.
4. The reader misreads stylised and vertical text: each candidate gets a sheet,
   work/<clip>/signs/NN.jpg, with the frame (its box drawn) and the text
   enlarged. Read the sheets, not just the OCR.

Writes work/<clip>/signs.json ({"cuts": [...], "candidates": [{id, start, end,
box, box_end, text, frames}]}; boxes are [x0, y0, x1, y1] as fractions of the
picture) and prints the candidates.

--preview draws the script's signs and dialogue (ass.py, as delivered) into a
frame from the middle of each sign, or of the signs given by number (1 is the
first), and tiles them in work/<clip>/signs/preview-N.jpg: check that each
sits where it should and nothing collides. No GPU.
"""

import argparse
import collections
import difflib
import json
import pathlib
import re
import subprocess
import sys

import numpy as np

import ass
import subs
from languages import LANGUAGES, pair_of

ROOT = pathlib.Path(__file__).resolve().parents[1]
W, H = subs.W, subs.H
PUNCT = re.compile(r"[\s、。,.!?！？…・「」『』（）()〈〉《》≪≫～~ー—-]")


def cuts(video, t0=None, t1=None, threshold=0.3):
    """Scene cut times in the clip."""
    cmd = ["ffmpeg", "-v", "error"]
    if t0:
        cmd += ["-ss", str(t0)]
    cmd += ["-i", str(video)]
    if t1:
        cmd += ["-t", str(t1 - (t0 or 0))]
    cmd += ["-vf", f"scale=320:-1,select='gt(scene,{threshold})',metadata=print:file=-", "-an", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, encoding="utf-8").stdout
    return [round(float(m) + (t0 or 0), 3) for m in re.findall(r"pts_time:([\d.]+)", out)]


def bare(text):
    return PUNCT.sub("", text).lower()


def pair(work):
    """(source, target) languages of a part, as dub.py prep recorded them (core.langs)."""
    seg = json.loads((work / "segments.json").read_text(encoding="utf-8")) if (work / "segments.json").exists() else {}
    return pair_of(seg)


def spoken(text, t, dialogue):
    """True when the text is a line of dialogue near t: a caption, not a sign.
    A short label inside a line ("groundwater" in a sentence about it) is a
    diagram echoing the narration, not a caption: only a long piece counts."""
    b = bare(text)
    if len(b) < 2:
        return False
    for a, z, ja in dialogue:
        if a - 4 <= t <= z + 4 and (
            b in ja and len(b) >= max(6, len(ja) / 2) or difflib.SequenceMatcher(None, b, ja).ratio() >= 0.6
        ):
            return True
    return False


def caption(img, box):
    """Light text on a broadcaster's semi-transparent box. Not subs.caption_box,
    which takes any light text on a dark ground: a title card on black too."""
    xs, ys = [int(p[0]) for p in box], [int(p[1]) for p in box]
    crop = img[max(min(ys), 0) : max(ys) + 1, max(min(xs), 0) : max(xs) + 1].reshape(-1, 3).astype(np.float32)
    if len(crop) < 20:
        return False
    lum = crop @ np.array([0.299, 0.587, 0.114], np.float32)
    return np.percentile(lum, 97) >= 150 and subs.overlay_box(img, box)


def read_text(ocr, img):
    """Text pieces in a frame that aren't closed captions: [(text, score, x0, y0, x1, y1)] in W x H units."""
    r = ocr(np.ascontiguousarray(img[:, :, ::-1]))
    if r.boxes is None:
        return []
    u = img.shape[1] / W
    out = []
    for box, text, score in zip(r.boxes, r.txts, r.scores):
        text = text.strip()
        if not text or score < 0.4 or caption(img, box):
            continue
        xs, ys = [float(p[0]) / u for p in box], [float(p[1]) / u for p in box]
        out.append((text, float(score), min(xs), min(ys), max(xs), max(ys)))
    return out


def blocks(pieces, gap=12):
    """Pieces close together (lines of one sign) as one block: [(texts, x0, y0, x1, y1)]."""
    groups = [[p] for p in pieces]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = rect(groups[i]), rect(groups[j])
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    groups[i] += groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    for g in groups:
        # Reading order: vertical text right to left, else top to bottom.
        tall = sum((p[5] - p[3]) > 1.5 * (p[4] - p[2]) for p in g) > len(g) / 2
        g.sort(key=lambda p: -p[2] if tall else p[3])
        out.append((" ".join(p[0] for p in g),) + rect(g))
    return out


def rect(group):
    return (min(p[2] for p in group), min(p[3] for p in group), max(p[4] for p in group), max(p[5] for p in group))


def overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def track(readings, shot_cuts, step, t_end):
    """Per-frame blocks -> candidates: one per block, over the frames of one shot that show it."""

    def shot_of(t):
        return sum(c <= t + 1e-3 for c in shot_cuts)

    open_, done = [], []
    for t, found in readings:
        still = []
        for text, x0, y0, x1, y1 in found:
            box = (x0, y0, x1, y1)
            c = next(
                (
                    c
                    for c in open_
                    if c not in still
                    and c["shot"] == shot_of(t)
                    and (
                        overlap(c["last"], box) > 0.4
                        or difflib.SequenceMatcher(None, c["texts"][-1], text).ratio() >= 0.6
                        and abs(c["last"][0] - x0) < 120
                        and abs(c["last"][1] - y0) < 120
                    )
                ),
                None,
            )
            if c is None:
                c = {"first_t": t, "shot": shot_of(t), "texts": [], "first": box}
            c["texts"].append(text)
            c["last"], c["seen"] = box, t
            still.append(c)
        for c in open_:
            if c not in still:
                (still if t - c["seen"] <= step * 2.5 and c["shot"] == shot_of(t) else done).append(c)
        open_ = still
    done += open_
    bounds = [0.0] + list(shot_cuts) + [t_end]
    out = []
    for c in sorted(done, key=lambda c: (c["first_t"], c["first"][1])):
        if len(c["texts"]) < 2 and bounds[c["shot"] + 1] - bounds[c["shot"]] > 2 * step:
            continue  # a single glimpse in a longer shot: usually a misread piece of the scene
        a, z = bounds[c["shot"]], bounds[c["shot"] + 1]
        # Shown from cut to cut unless it appears or leaves well inside the shot.
        start = a if c["first_t"] - a <= step * 1.5 else c["first_t"] - step / 2
        end = z if z - c["seen"] <= step * 2.5 else c["seen"] + step
        text = collections.Counter(c["texts"]).most_common(1)[0][0]
        out.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "frames": len(c["texts"]),
                "box": [round(v / d, 3) for v, d in zip(c["first"], (W, H, W, H))],
                "box_end": [round(v / d, 3) for v, d in zip(c["last"], (W, H, W, H))],
                "seen": [round(c["first_t"], 2), round(c["seen"], 2)],
            }
        )
    return out


def logos(cands):
    """Candidates that are a channel's logo: the same text in the same place, near
    an edge of the picture, in three shots or more."""

    def centre(c):
        b = c["box"]
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    out = set()
    for c in cands:
        x, y = centre(c)
        if 0.15 < x < 0.85 and 0.15 < y < 0.85:
            continue
        same = [
            d
            for d in cands
            if abs(centre(d)[0] - x) < 0.03
            and abs(centre(d)[1] - y) < 0.03
            and difflib.SequenceMatcher(None, bare(c["text"]), bare(d["text"])).ratio() >= 0.6
        ]
        if len({(d["start"], d["end"]) for d in same}) >= 3:
            out |= {id(d) for d in same}
    return out


def shot_sheets(thumbs, bounds, out, per=24, cols=4):
    """One thumbnail per shot, labelled with its number and times, `per` to a sheet:
    for spotting text the reader missed (vertical brush lettering, handwriting)."""
    import cv2

    paths = []
    for s in range(0, len(thumbs), per):
        batch = thumbs[s : s + per]
        tw, th = batch[0][1].shape[1], batch[0][1].shape[0]
        rows = (len(batch) + cols - 1) // cols
        canvas = np.full((rows * th, cols * tw, 3), 32, np.uint8)
        for k, (shot, img) in enumerate(batch):
            y, x = k // cols * th, k % cols * tw
            canvas[y : y + th, x : x + tw] = img
            label = f"S{shot:02d} {bounds[shot]:.1f}-{bounds[shot + 1]:.1f}"
            cv2.rectangle(canvas, (x, y), (x + 12 * len(label) + 8, y + 24), (0, 0, 0), -1)
            cv2.putText(canvas, label, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x, y), (x + tw - 1, y + th - 1), (32, 32, 32), 2)
        path = out / f"shots-{s // per + 1}.jpg"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    return paths


def sheet(video, cand, path, w, h):
    """The frame with the candidate's box drawn, and the text enlarged beside it."""
    import cv2

    t = (cand["seen"][0] + cand["seen"][1]) / 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout  # fmt: skip
    if len(raw) < w * h * 3:
        return
    img = np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3).copy()
    bx = [cand["box"][0] * w, cand["box"][1] * h, cand["box"][2] * w, cand["box"][3] * h]
    pad = 12
    x0, y0 = max(int(bx[0]) - pad, 0), max(int(bx[1]) - pad, 0)
    x1, y1 = min(int(bx[2]) + pad, w), min(int(bx[3]) + pad, h)
    crop = img[y0:y1, x0:x1]
    scale = min(3.0, 500 / max(crop.shape[0], 1), 900 / max(crop.shape[1], 1))
    crop = cv2.resize(crop, None, fx=max(scale, 1.0), fy=max(scale, 1.0), interpolation=cv2.INTER_CUBIC)
    frame = img.copy()
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 4)
    frame = cv2.resize(frame, (960, 960 * h // w))
    canvas = np.full((frame.shape[0] + crop.shape[0] + 10, max(960, crop.shape[1]), 3), 32, np.uint8)
    canvas[: frame.shape[0], :960] = frame
    canvas[frame.shape[0] + 10 :, : crop.shape[1]] = crop
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


def preview(work, video, w, h, which=None):
    """The script's signs drawn into their frames, as delivered: preview-N.jpg sheets."""
    import cv2

    script = json.loads((work / "script.json").read_text(encoding="utf-8"))
    signs = sorted(script.get("signs", []), key=lambda s: s["start"])
    if not signs:
        raise SystemExit(f"{work.name}: no signs in script.json")
    pick = [(k, s) for k, s in enumerate(signs, 1) if not which or k in which]
    tgt = pair(work)[1]
    dialogue = [
        (l["start"], l["end"], LANGUAGES[tgt].subtitle(l["text"])) for l in script["lines"] + script.get("captions", [])
    ]
    out = work / "signs"
    out.mkdir(exist_ok=True)
    ass.write(out / "preview.ass", w, h, dialogue, signs, lang=tgt, subtitle_layout=script.get("subtitle_layout", []))
    tiles = []
    for k, s in pick:
        t = (s["start"] + s["end"]) / 2
        # -copyts keeps the frame's own time, so the subtitles line up after the seek;
        # the filter takes a bare file name, run from its folder.
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-copyts", "-i", str(video),
                              "-vf", "ass=preview.ass,scale=960:-2", "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "bgr24", "-"], capture_output=True, cwd=out).stdout  # fmt: skip
        th = len(raw) // (960 * 3)
        if not th:
            continue
        img = np.frombuffer(raw[: 960 * th * 3], np.uint8).reshape(th, 960, 3).copy()
        label = f"#{k} {t:.1f}s {s['text'][:40]}"
        cv2.rectangle(img, (0, 0), (min(960, 11 * len(label) + 10), 24), (0, 0, 0), -1)
        cv2.putText(img, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    for old in out.glob("preview-*.jpg"):
        old.unlink()
    for n in range(0, len(tiles), 6):
        batch = tiles[n : n + 6]
        rows = [
            np.hstack(batch[r : r + 2] + [np.zeros_like(batch[0])] * (2 - len(batch[r : r + 2])))
            for r in range(0, len(batch), 2)
        ]
        path = out / f"preview-{n // 6 + 1}.jpg"
        cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(path)


def dialogue_of(work):
    """(start, end, bare original) of every line, from the script, else the draft."""
    src = pair(work)[0]
    for name in ("script.json", "draft.json"):
        p = work / name
        if p.exists():
            doc = json.loads(p.read_text(encoding="utf-8"))
            lines = doc.get("lines") or []
            return [(l["start"], l["end"], bare(l.get(src) or "")) for l in lines if l.get(src)]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--fps", type=float, default=2.0, help="frames read a second (default 2)")
    ap.add_argument("--from", dest="t0", type=float)
    ap.add_argument("--to", dest="t1", type=float)
    ap.add_argument(
        "--preview",
        nargs="*",
        type=int,
        metavar="SIGN",
        help="draw the script's signs into their frames (all, or these by number)",
    )
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    work = ROOT / "work" / args.clip
    video = json.loads((work / "segments.json").read_text(encoding="utf-8"))["video"]
    video = pathlib.Path(video) if pathlib.Path(video).is_absolute() else ROOT / video
    w, h = subs.size(video)
    if args.preview is not None:
        return preview(work, video, w, h, set(args.preview))
    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    shot_cuts = cuts(video, args.t0, args.t1)
    dialogue = dialogue_of(work)
    ocr, cuda = subs.engine(pair(work)[0])
    print(
        f"reading text in {video.name} ({'GPU' if cuda else 'CPU'}, {args.fps:g} frames/s, {len(shot_cuts)} cuts)",
        file=sys.stderr,
    )
    import cv2

    t_end = args.t1 or duration
    bounds = [args.t0 or 0.0] + shot_cuts + [t_end]
    mids = [(a + b) / 2 for a, b in zip(bounds, bounds[1:])]
    readings, thumbs, last, found, n, dropped = [], [], None, [], 0, 0
    for t, img in subs.frames(video, args.fps, args.t0, args.t1, w, h):
        if subs.changed(img, last):
            pieces = read_text(ocr, img)
            keep = [p for p in pieces if not spoken(p[0], t, dialogue)]
            dropped += len(pieces) - len(keep)
            found, last, n = blocks(keep), img, n + 1
        readings.append((t, found))
        shot = sum(c <= t + 1e-3 for c in shot_cuts)
        if (not thumbs or thumbs[-1][0] != shot) and t + 0.5 / args.fps >= mids[shot]:
            thumbs.append((shot, cv2.resize(np.ascontiguousarray(img[:, :, ::-1]), (480, 480 * h // w))))
    cands = track(readings, shot_cuts, 1 / args.fps, t_end)
    logo = logos(cands)
    cands = [c for c in cands if id(c) not in logo]
    out = work / "signs"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for k, c in enumerate(cands, 1):
        c["id"] = k
        sheet(video, c, out / f"{k:02d}.jpg", w, h)
    sheets = shot_sheets(thumbs, bounds, out) if thumbs else []
    print(
        f"{len(readings)} frames, {n} read, {dropped} pieces of dialogue and {len(logo)} of a logo left out, "
        f"{len(cands)} candidates; {len(bounds) - 1} shots on {len(sheets)} sheet(s) in {out}",
        file=sys.stderr,
    )
    (work / "signs.json").write_text(
        json.dumps({"video": str(video), "cuts": shot_cuts, "candidates": cands}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    for c in cands:
        b = c["box"]
        print(
            f"#{c['id']:02d} {c['start']:7.2f}-{c['end']:7.2f}  x{(b[0] + b[2]) / 2:.2f} y{(b[1] + b[3]) / 2:.2f}"
            f"  ({c['frames']} frames)  {c['text']}"
        )


if __name__ == "__main__":
    main()
