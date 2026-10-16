"""Burned-in captions: read a clip's on-screen closed captions into timed text.

Usage: scripts/subs.py <clip> [--open] [--fps 4] [--from T0] [--to T1]   (after `dub.py prep`; own environment, see AGENTS.md)

--auto checks closed-caption and outlined-bottom-subtitle appearances together,
with one OCR inference per changed frame. Events keep their appearance kind so
the director can distinguish evidence from an assumed spoken transcript.

Some recordings carry the broadcaster's closed captions burned into the picture:
the exact dialogue, often the speaker's name in brackets, "≪" for a voice off
screen, and a colour of its own for each main character. This reads them with
PaddleOCR's PP-OCRv5 models (RapidOCR on onnxruntime, GPU when available), the
recogniser the part's source language names (OCR in languages/<code>.py: the
Chinese one, which reads kana and Latin letters too, or the Korean one):

1. Frames are sampled over the whole picture (4 a second). A frame that barely
   differs from the last one read reuses its reading.
2. Text is detected and read. Only light text on a dark box, or on a
   semi-transparent one over a bright scene (found by its edges), counts as a
   caption, so signs and other text in the scene are left out. Pieces of one
   row (captions space phrases apart) are joined.
3. Readings of one caption across frames become one event, with its start,
   end, position and colour, and the reading seen most often.
4. Song lyrics (captions with ♪) become bare music markers without their text,
   one per song: songs stay in the original audio. A song can run on past
   its last ♪ (under the credits, say): find its end by ear or VAD.

--open reads a film's own subtitles instead: the translation of speech in a
third language (Finnish in a film set in Finland), drawn as white text with a
dark outline over the bottom of the picture, without a box. Only that band is
read, 2 frames a second.

Writes work/<clip>/subs.json ({"events": [{start, end, text, color, x, y,
name, off}]}; x/y are the caption's centre as fractions of the picture) and
prints the events. `name` is a bracketed speaker label, `off` marks "≪".
draft.py shows the captions beside each phrase when subs.json exists.
Caption times follow the broadcaster's display, not the speech: take windows
from the VAD as usual.
"""

import argparse
import collections
import difflib
import json
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from gpulock import hold_gpu
from languages import LANGUAGES, pair_of

ROOT = pathlib.Path(__file__).resolve().parents[1]
W, H = 960, 540  # positions are kept in these units, whatever the video's size
OFF_MARKS = "≪≤《«<"
# A speaker label; OCR sometimes drops its opening bracket.
NAME = re.compile(r"^[(（]?([^()（）\s]{1,8})[)）]\s*")
# The recogniser has no ♪ and reads it as 月.
NOTE = re.compile(r"^[♪♫]|^月(\s|[～~]|$)")


def engine(lang):
    """RapidOCR with the recogniser that reads `lang` (its module's OCR)."""
    import onnxruntime

    onnxruntime.set_default_logger_severity(3)
    try:
        onnxruntime.preload_dlls()  # CUDA/cuDNN from the pip wheels, if installed
    except Exception:
        pass
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    hold_gpu()
    cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    rec, model = LANGUAGES[lang].OCR
    return RapidOCR(
        params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType[model],
            "Rec.lang_type": LangRec[rec],
            "Global.use_cls": False,
            "Global.log_level": "critical",
            "EngineConfig.onnxruntime.use_cuda": cuda,
        }
    ), cuda


def size(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(video)], capture_output=True, text=True)  # fmt: skip
    return tuple(int(v) for v in out.stdout.strip().split(",")[:2])


def frames(video, fps, t0, t1, w, h, top=0):
    """Full-size frames: captions read much better than from a downscaled picture.
    With `top`, only the picture below that row."""
    cmd = ["ffmpeg", "-v", "error"]
    if t0:
        cmd += ["-ss", str(t0)]
    cmd += ["-i", str(video)]
    if t1:
        cmd += ["-t", str(t1 - (t0 or 0))]
    crop = f",crop={w}:{h - top}:0:{top}" if top else ""
    h -= top
    cmd += ["-vf", f"fps={fps}{crop}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    k, n = 0, w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield (t0 or 0) + k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        k += 1
    p.wait()


def changed(a, b):
    """True unless every block of the two frames is nearly the same."""
    if b is None:
        return True
    a, b = a[::2, ::2], b[::2, ::2]
    h, w = a.shape[0] // 30 * 30, a.shape[1] // 30 * 30
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    blocks = d[:h, :w].reshape(h // 30, 30, w // 30, 30).mean(axis=(1, 3))
    return blocks.max() > 2.0


def color_of(rgb):
    r, g, b = rgb
    hi, lo = 150, 110
    if r > hi and g > hi and b < lo:
        return "yellow"
    if g > hi and b > hi and r < lo:
        return "cyan"
    if g > hi and r < lo and b < lo:
        return "green"
    if r > hi and b > hi and g < lo:
        return "magenta"
    if r > hi and g < lo and b < lo:
        return "red"
    return "white"


def caption_box(img, box):
    """(is a caption, colour): light text on a dark box, unlike text in the scene."""
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    x0, x1 = max(int(min(xs)) - 2, 0), int(max(xs)) + 3
    y0, y1 = max(int(min(ys)) - 2, 0), int(max(ys)) + 3
    crop = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if len(crop) < 20:
        return False, None
    lum = crop @ np.array([0.299, 0.587, 0.114], np.float32)
    text_lum, back_lum = np.percentile(lum, 90), np.percentile(lum, 40)
    ok = text_lum >= 150 and back_lum <= 100 and text_lum - back_lum >= 80
    color = color_of(crop[lum >= np.percentile(lum, 90)].mean(axis=0))
    # Thin strokes can fill under a tenth of the box: the brightest pixels are the text.
    return ok or (np.percentile(lum, 97) >= 150 and overlay_box(img, box)), color


def edge_step(lum, box, side):
    """The sharpest rise in brightness going out from one side of a text box
    (outside over inside), within reach of the text."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    reach = int(max(10, 0.8 * min(w, h)))
    inset = max(2, int(0.15 * h))  # the detector's box can reach past the overlay's edge
    if side in ("top", "bottom"):
        a, b = int(x0 + 0.1 * w), int(x1 - 0.1 * w)
        rows = range(int(y0) + inset, int(y0) - reach, -1) if side == "top" else range(int(y1) - inset, int(y1) + reach)
        prof = np.array([np.median(lum[y, a:b]) for y in rows if 0 <= y < lum.shape[0] and b > a])
    else:
        a, b = int(y0 + 0.2 * h), int(y1 - 0.2 * h)
        cols = (
            range(int(x0) + inset, int(x0) - reach, -1) if side == "left" else range(int(x1) - inset, int(x1) + reach)
        )
        prof = np.array([np.median(lum[a:b, x]) for x in cols if 0 <= x < lum.shape[1] and b > a])
    best = 0.0
    for k in range(2, len(prof) - 3):
        inside, outside = np.median(prof[max(0, k - 4) : k]), np.median(prof[k : k + 4])
        if inside > 5:
            best = max(best, outside / inside)
    return best


def overlay_box(img, box, ratio=1.5):
    """True when the text sits on a semi-transparent dark box: the picture is
    about half as bright inside it, so brightness steps up at its edges (top and
    bottom, or three sides). A broadcaster's box over a bright scene fails the
    dark-background test but not this one."""
    xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    m = int(max(10, 0.8 * min(x1 - x0, y1 - y0))) + 4
    wx, wy = max(int(x0) - m, 0), max(int(y0) - m, 0)
    win = img[wy : int(y1) + m, wx : int(x1) + m].astype(np.float32)
    lum = win @ np.array([0.299, 0.587, 0.114], np.float32)
    rect = (x0 - wx, y0 - wy, x1 - wx, y1 - wy)
    steps = {s: edge_step(lum, rect, s) >= ratio for s in ("top", "bottom", "left", "right")}
    return (steps["top"] and steps["bottom"]) or sum(steps.values()) >= 3


def subtitle_text(img, box):
    """(is a subtitle, colour): white text with a dark outline over the scene, as a
    film's own subtitles for foreign-language speech are drawn (no box behind)."""
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    x0, x1 = max(int(min(xs)) - 2, 0), int(max(xs)) + 3
    y0, y1 = max(int(min(ys)) - 2, 0), int(max(ys)) + 3
    crop = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if len(crop) < 20:
        return False, None
    lum = crop @ np.array([0.299, 0.587, 0.114], np.float32)
    text = crop[lum >= np.percentile(lum, 95)]
    neutral = np.ptp(text.mean(axis=0)) <= 40  # white, not a coloured sign
    return bool(np.percentile(lum, 95) >= 200 and np.percentile(lum, 5) <= 60 and neutral), "white"


def read_frame(ocr, img, top=0, test=caption_box):
    """Caption rows in one frame: [(text, colour, x0, y0, x1, y1)]; `top` is the
    row of the full picture the image starts at."""
    r = ocr(np.ascontiguousarray(img[:, :, ::-1]))
    if r.boxes is None:
        return []
    u = img.shape[1] / W
    pieces = []
    for box, text, score in zip(r.boxes, r.txts, r.scores):
        ok, color = test(img, box)
        if ok and score >= 0.5 and text.strip():
            xs, ys = [p[0] for p in box], [p[1] + top for p in box]
            pieces.append([text.strip(), color, min(xs) / u, min(ys) / u, max(xs) / u, max(ys) / u])
    if test is subtitle_text and pieces:
        # A film shows one subtitle at a time, spaced into phrases and often over two
        # rows, which the detector splits differently from frame to frame: read the
        # band as one text, row by row, so the frames of one subtitle agree.
        lines = []
        for p in sorted(pieces, key=lambda p: p[3] + p[5]):
            if lines and abs((lines[-1][-1][3] + lines[-1][-1][5]) - (p[3] + p[5])) / 2 < 10:
                lines[-1].append(p)
            else:
                lines.append([p])
        text = " ".join(p[0] for line in lines for p in sorted(line, key=lambda p: p[2]))
        return [
            (
                text,
                "white",
                min(p[2] for p in pieces),
                min(p[3] for p in pieces),
                max(p[4] for p in pieces),
                max(p[5] for p in pieces),
            )
        ]
    pieces.sort(key=lambda p: (round((p[3] + p[5]) / 2 / 12), p[2]))
    rows = []
    for p in pieces:
        last = rows[-1] if rows else None
        if last and last[1] == p[1] and abs((last[3] + last[5]) - (p[3] + p[5])) / 2 < 10 and 0 <= p[2] - last[4] < 60:
            last[0] += " " + p[0]
            last[4], last[3], last[5] = p[4], min(last[3], p[3]), max(last[5], p[5])
        else:
            rows.append(p)
    return [tuple(r) for r in rows]


def read_auto_frame(ocr, img):
    """One OCR inference, two appearance filters. The director judges their meaning."""
    result = ocr(np.ascontiguousarray(img[:, :, ::-1]))
    closed = read_frame(lambda _: result, img, test=caption_box)
    if result.boxes is None:
        return closed, []
    candidates = [
        (box, text, score)
        for box, text, score in zip(result.boxes, result.txts, result.scores)
        if min(point[1] for point in box) >= img.shape[0] * 0.68 and not caption_box(img, box)[0]
    ]
    filtered = SimpleNamespace(
        boxes=[row[0] for row in candidates],
        txts=[row[1] for row in candidates],
        scores=[row[2] for row in candidates],
    )
    return closed, read_frame(lambda _: filtered, img, test=subtitle_text)


def automatic_events(closed, subtitles, step):
    events = [{**event, "kind": "closed_caption"} for event in track(closed, step)]
    for event in track(subtitles, step, anywhere=True):
        duplicate = next(
            (
                old
                for old in events
                if similar(old["text"], event["text"])
                and min(old["end"], event["end"]) > max(old["start"], event["start"])
            ),
            None,
        )
        if duplicate:
            duplicate["start"] = min(duplicate["start"], event["start"])
            duplicate["end"] = max(duplicate["end"], event["end"])
        else:
            events.append({**event, "kind": "subtitle"})
    return sorted(events, key=lambda event: (event["start"], event["y"]))


def normalize(text):
    text = re.sub(r"\s+", " ", text.replace("~", "～").replace("(", "（").replace(")", "）")).strip()
    return "♪" if NOTE.match(text) or "♪" in text else text


def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75


def track(readings, step, anywhere=False):
    """Per-frame caption rows -> events: one per caption, over the frames that show it.
    `anywhere`: one caption at a time (subtitles), so a reading continues the
    last one by its text alone, wherever the detector placed it."""
    open_, done = [], []
    for t, rows in readings:
        still = []
        for row in rows:
            text, color, x0, y0, x1, y1 = row
            cx, cy = float(x0 + x1) / 2, float(y0 + y1) / 2
            ev = next(
                (
                    e
                    for e in open_
                    if e not in still
                    and (anywhere or abs(e["cy"] - cy) < 15 and (abs(e["cx"] - cx) < 40 or abs(e["x0"] - x0) < 20))
                    and similar(e["texts"][-1], text)
                ),
                None,
            )
            if ev is None:
                ev = {"start": t, "texts": [], "colors": [], "cx": cx, "cy": cy, "x0": float(x0)}
            ev["texts"].append(text)
            ev["colors"].append(color)
            ev["end"], ev["seen"] = t + step, t
            still.append(ev)
        for e in open_:
            if e not in still:
                # One missed reading doesn't end a caption.
                (still if t - e["seen"] <= step * 1.5 else done).append(e)
        open_ = still
    done += open_
    events = []
    for e in sorted(done, key=lambda e: (e["start"], e["cy"])):
        if len(e["texts"]) < 2 and e["end"] - e["start"] <= step:
            continue  # a single glimpse: usually a misread piece of the scene
        text = normalize(collections.Counter(e["texts"]).most_common(1)[0][0])
        off = text[:1] in OFF_MARKS
        text = text.lstrip(OFF_MARKS + " ")
        m = NAME.match(text)
        name, text = (m.group(1).strip(), text[m.end() :]) if m else (None, text)
        events.append(
            {
                "start": round(e["start"], 2),
                "end": round(e["end"], 2),
                "text": text,
                "color": collections.Counter(e["colors"]).most_common(1)[0][0],
                "x": round(e["cx"] / W, 2),
                "y": round(e["cy"] / H, 2),
                "name": name,
                "off": off,
            }
        )
    # A lyric's other rows may lack the ♪: anything shown alongside a ♪ caption is lyrics too.
    songs = [e for e in events if e["text"] == "♪"]
    for e in events:
        if any(min(e["end"], s["end"]) - max(e["start"], s["start"]) > (e["end"] - e["start"]) / 2 for s in songs):
            e.update(text="♪", name=None)
    return merge_lines(events)


def merge_lines(events):
    """Join a caption's second row to its first: same times and colour, no speaker label or ≪ of its own."""
    out = []
    for e in events:
        prev = next(
            (
                p
                for p in reversed(out[-3:])
                if abs(p["start"] - e["start"]) < 0.3
                and abs(p["end"] - e["end"]) < 0.3
                and p["color"] == e["color"]
                and 0 < e["y"] - p["y"] < 0.12
            ),
            None,
        )
        if prev and prev["text"] == e["text"] == "♪":
            prev["end"] = max(prev["end"], e["end"])
        elif prev and not e["name"] and not e["off"] and "♪" not in (prev["text"], e["text"]):
            prev["text"] += " " + e["text"]
            prev["y2"] = e["y"]
        else:
            out.append(e)
    for e in out:
        e.pop("y2", None)
    # ♪ shows only now and then during a song: markers under 15 s apart are one song.
    songs, rest = [], []
    for e in out:
        if e["text"] != "♪":
            rest.append(e)
        elif songs and e["start"] - songs[-1]["end"] <= 15:
            songs[-1]["end"] = max(songs[-1]["end"], e["end"])
        else:
            songs.append({**e, "x": 0.5, "y": 0.5, "color": "white", "off": False})
    return sorted(rest + songs, key=lambda e: (e["start"], e["y"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--fps", type=float, help="frames read a second (default 4, or 2 with --open)")
    ap.add_argument("--from", dest="t0", type=float)
    ap.add_argument("--to", dest="t1", type=float)
    ap.add_argument("--auto", action="store_true", help="detect both closed captions and outlined bottom subtitles")
    ap.add_argument(
        "--open",
        action="store_true",
        help="the film's own subtitles for foreign-language speech (white, outlined, at the "
        "bottom) instead of closed captions on a dark box",
    )
    args = ap.parse_args()
    if args.auto and args.open:
        ap.error("choose --auto or --open")
    args.fps = args.fps or (2 if args.open else 4)
    sys.stdout.reconfigure(encoding="utf-8")
    work = ROOT / "work" / args.clip
    seg = json.loads((work / "segments.json").read_text(encoding="utf-8"))
    video = seg["video"]
    video = pathlib.Path(video) if pathlib.Path(video).is_absolute() else ROOT / video
    w, h = size(video)
    # Open subtitles sit in the bottom of the picture: reading only that band is
    # faster, and keeps signs and other text in the scene out.
    top = int(h * 0.68) // 2 * 2 if args.open else 0
    test = subtitle_text if args.open else caption_box
    ocr, cuda = engine(pair_of(seg)[0])
    print(
        f"reading {'subtitles' if args.open else 'captions'} in {video.name} ({'GPU' if cuda else 'CPU'}, "
        f"{args.fps:g} frames/s)",
        file=sys.stderr,
    )
    readings, last, rows, n = [], None, [], 0
    subtitle_readings, subtitle_rows = [], []
    for t, img in frames(video, args.fps, args.t0, args.t1, w, h, top):
        if changed(img, last):
            if args.auto:
                rows, subtitle_rows = read_auto_frame(ocr, img)
            else:
                rows = read_frame(ocr, img, top, test)
            last, n = img, n + 1
        readings.append((t, rows))
        if args.auto:
            subtitle_readings.append((t, subtitle_rows))
    events = (
        automatic_events(readings, subtitle_readings, 1 / args.fps)
        if args.auto
        else track(readings, 1 / args.fps, anywhere=args.open)
    )
    print(f"{len(readings)} frames, {n} read, {len(events)} captions", file=sys.stderr)
    (work / "subs.json").write_text(
        json.dumps(
            {
                "video": str(video),
                "fps": args.fps,
                "mode": "auto" if args.auto else "open" if args.open else "closed",
                "events": events,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    for e in events:
        who = ("≪" if e["off"] else "") + (f"({e['name']})" if e["name"] else "")
        print(f"{e['start']:7.2f}-{e['end']:7.2f}  {e['color']:7s} x{e['x']:.2f} y{e['y']:.2f}  {who}{e['text']}")


if __name__ == "__main__":
    main()
