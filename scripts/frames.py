"""Labelled grid of video frames, for casting: who is on screen at each line?

Usage: scripts/frames.py <out.jpg> <video> <seconds>[:label] ... [--cols 5] [--width 400]

One frame per time, stamped with the time and label, tiled into one image.
Close-ups, visible speech and on-screen name captions help identify speakers.
"""

import argparse
import os
import pathlib
import subprocess
import tempfile

# Escaped twice: once for the drawtext option, once for the filtergraph.
if os.name == "nt":
    FONT = "C\\\\:/Windows/Fonts/arial.ttf"
elif os.path.exists("/System/Library/Fonts/Supplemental/Arial.ttf"):
    FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
elif os.path.exists("/System/Library/Fonts/Helvetica.ttc"):
    FONT = "/System/Library/Fonts/Helvetica.ttc"
else:
    FONT = None


def has_drawtext():
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-filters"], capture_output=True, text=True)
        return "drawtext" in r.stdout
    except Exception:
        return False


def stamp_pil(frame_path, text):
    try:
        from PIL import Image, ImageDraw, ImageFont

        with Image.open(frame_path) as img:
            draw = ImageDraw.Draw(img)
            font = None
            for p in (
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
            ):
                if os.path.exists(p):
                    try:
                        font = ImageFont.truetype(p, 17)
                        break
                    except Exception:
                        pass
            font = font or ImageFont.load_default()
            bbox = draw.textbbox((4, 4), text, font=font)
            draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill="black")
            draw.text((4, 4), text, fill="yellow", font=font)
            img.save(frame_path)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("video")
    ap.add_argument("times", nargs="+", help="seconds, optionally with :label")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--width", type=int, default=400)
    args = ap.parse_args()
    if args.cols < 1:
        ap.error("--cols must be at least 1")

    drawtext_ok = has_drawtext()
    tmp = pathlib.Path(tempfile.mkdtemp())
    frames = []
    for k, spec in enumerate(args.times):
        t, _, label = spec.partition(":")
        display_text = f"{float(t):.1f} {label}".strip()
        f = tmp / f"{k:03d}.jpg"
        if drawtext_ok:
            esc_text = display_text.replace(":", "\\:").replace("'", "")
            draw = (
                f"drawtext={'fontfile=' + FONT + ':' if FONT else ''}text='{esc_text}':x=4:y=4:fontsize=17"
                f":fontcolor=yellow:box=1:boxcolor=black"
            )
            filters = f"scale={args.width}:-2,{draw}"
        else:
            filters = f"scale={args.width}:-2"
        result = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", t, "-i", args.video, "-frames:v", "1",
                                 "-vf", filters, str(f)], capture_output=True, text=True,
                                encoding="utf-8", errors="replace")  # fmt: skip
        if result.returncode:
            raise SystemExit(f"ffmpeg failed extracting the frame at {float(t):.1f} s:\n{result.stderr.strip()}")
        if not drawtext_ok and f.exists():
            stamp_pil(f, display_text)
        if f.exists():
            frames.append(f)
        else:  # ffmpeg writes nothing for a time past the end of the video
            print(f"no frame at {float(t):.1f} s (past the end?)")
    if not frames:
        raise SystemExit("no frames")
    height = int(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=height", "-of", "csv=p=0", str(frames[0])],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    while len(frames) % args.cols:  # blank tiles, so no time appears twice
        f = tmp / f"pad{len(frames):03d}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=black:s={args.width}x{height}",
                        "-frames:v", "1", "-pix_fmt", "yuvj420p", str(f)], check=True)  # fmt: skip
        frames.append(f)
    layout = "|".join(f"{(i % args.cols) * args.width}_{(i // args.cols) * height}" for i in range(len(frames)))
    inputs = [a for f in frames for a in ("-i", str(f))]
    grid_filter = ["-filter_complex", f"xstack=inputs={len(frames)}:layout={layout}"] if len(frames) > 1 else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, *grid_filter,
                    "-frames:v", "1", args.out], check=True)  # fmt: skip
    print(args.out)


if __name__ == "__main__":
    main()
