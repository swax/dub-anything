"""Put the finished dubs into the source video itself, renamed "<name> (dubbed)".

Usage: scripts/deliver.py <source video> <voice>=<assembled.mp4> [<voice>=<assembled.mp4> ...] [--default VOICE]
                          [--source ja --target en]

Each <assembled.mp4> is assemble.py's output for one voice ("actor" or
"american"), with its subtitles beside it, named for their languages (.en.srt
and .ja.srt for a dub between Japanese and English, .en.srt and .ko.srt from
Korean), and its
.en.ass (.ja.ass, dubbing into Japanese) and .signs.ass when the parts
translate text in the picture. The result keeps every stream of the source
as it is (video, audio, its own subtitles) and adds, in this order:
  - audio: one dub per voice, the default voice's first and marked
    default (some players ignore the flag and play the first track), then the
    source's own audio (its first track, which split.py cut the parts from,
    tagged with the source language);
  - subtitles: the dub's per voice (the default voice's first and on by
    default), the corrected transcript of the source, then the source's own.
Track names are in each track's language (TITLES in languages/<code>.py).
With signs, the dub's tracks are the styled ones (signs placed by the text
they translate), a signs-only track follows them, marked forced for players
that show forced subtitles when subtitles are off, and the file becomes
Matroska ("<name> (dubbed).mkv"): MP4's subtitles can't be placed.
Nothing is re-encoded; the new subtitles are written in the container's format.
Every copied stream is checked against its input (MD5) before the new file
replaces the source; the source's name gets " (dubbed)" (once). Delivering
again to a "(dubbed)" file replaces the dub tracks it already has.

deliver_subtitles() does the same for a job that subtitles the video instead
(dub_anything.py's --voices none): the source's own streams, then the
translation (the styled .en.ass with the signs placed when there are signs,
else the .en.srt; on by default) and the transcript, in Matroska,
"<name> (subtitled).mkv", with the translation's .srt beside it
("<name> (subtitled).en.srt", signs at the top) for players that read
subtitles from a file.
"""

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

from core import VOICE_LABEL, original_audio_language
from languages import ISO3, LANGUAGES, PAIR, PAIRS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SUFFIX = " (dubbed)"
SUBTITLED = " (subtitled)"
LEGACY = re.compile(r"^(English dub \(.*\)|English \(.* dub\)|English \(signs only\)|日本語 \(transcript\))$")


def ours(name):
    """Whether a track is one we delivered (a dub, its subtitles, the signs, a transcript)."""
    every = [m.TITLES for m in LANGUAGES.values()]
    titles = {t for lang in every for kind in ("dub", "subs") for t in lang.get(kind, {}).values()}
    titles |= {lang[kind] for lang in every for kind in ("signs", "transcript") if kind in lang}
    return name in titles or bool(LEGACY.match(name))


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=index,codec_type,codec_name:stream_tags=language,title,handler_name"
                          ":stream_disposition=attached_pic", "-of", "json", str(path)],
                         capture_output=True, encoding="utf-8", check=True).stdout  # fmt: skip
    return json.loads(out)


GENERIC = {
    "SoundHandler",
    "VideoHandler",
    "SubtitleHandler",
    "Core Media Audio",
    "Core Media Video",
    "ISO Media file produced by Google Inc.",
    "Mainconcept MP4 Sound Media Handler",
}


def name_of(stream):
    """A track's name, or "" for the generic handler names muxers write."""
    tags = stream.get("tags") or {}
    name = tags.get("title") or tags.get("handler_name") or ""
    return "" if name in GENERIC else name


def md5(path, index):
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{index}", "-c", "copy", "-f", "md5", "-"],
        capture_output=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()


def delivered_name(source, mkv=False, suffix=SUFFIX):
    stem = source.stem[: -len(suffix)] if source.stem.endswith(suffix) else source.stem
    ext = source.suffix.lower() if source.suffix.lower() in (".mp4", ".m4v", ".mov", ".mkv") and not mkv else ".mkv"
    return source.with_name(f"{stem}{suffix}{ext}")


def deliver(source, dubs, default, pair=PAIR, quicktime_audio=False):
    """dubs: {voice: assembled mp4}; pair: (source, target) languages. Returns the
    delivered file's path."""
    src, tgt = pair
    titles = LANGUAGES[tgt].TITLES
    source = pathlib.Path(source)
    voices = sorted(dubs, key=lambda v: v != default)
    info = probe(source)
    duration = float(info["format"]["duration"])
    for v in voices:
        d = float(probe(dubs[v])["format"]["duration"])
        if abs(d - duration) > 0.1:
            raise SystemExit(f"{dubs[v]} runs {d:.2f} s, the source {duration:.2f} s: assemble it from the same video")
    keep = [s for s in info["streams"] if not ours(name_of(s))]
    dropped = len(info["streams"]) - len(keep)
    vids = [s for s in keep if s["codec_type"] == "video"]
    auds = [s for s in keep if s["codec_type"] == "audio"]
    subs = [s for s in keep if s["codec_type"] == "subtitle"]
    signs = dubs[voices[0]].with_suffix(".signs.ass")
    signs = signs if signs.exists() else None
    out = delivered_name(source, mkv=bool(signs))
    tmp = out.with_name(out.stem + ".delivering" + out.suffix)
    mkv = out.suffix == ".mkv"

    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(source)]
    for v in voices:
        cmd += ["-i", str(dubs[v])]
    srts = [
        (dubs[v].with_suffix(f".{tgt}.ass" if signs else f".{tgt}.srt"), ISO3[tgt], titles["subs"][v]) for v in voices
    ]
    if signs:
        srts.append((signs, ISO3[tgt], titles["signs"]))
    transcript = dubs[voices[0]].with_suffix(f".{src}.srt")
    if transcript.exists():
        srts.append((transcript, ISO3[src], LANGUAGES[src].TITLES["transcript"]))
    for path, _, _ in srts:
        if not path.exists():
            raise SystemExit(f"no subtitles at {path}")
        cmd += ["-i", str(path)]
    for s in vids:
        cmd += ["-map", f"0:{s['index']}"]
    for k in range(len(voices)):
        cmd += ["-map", f"{1 + k}:a:0"]
    for s in auds:
        cmd += ["-map", f"0:{s['index']}"]
    for k in range(len(srts)):
        cmd += ["-map", f"{1 + len(voices) + k}:0"]
    for s in subs:
        cmd += ["-map", f"0:{s['index']}"]
    cmd += ["-c", "copy"]
    for k, (path, _, _) in enumerate(srts):  # ours are .srt and .ass files
        cmd += [f"-c:s:{k}", "ass" if path.suffix == ".ass" else "srt" if mkv else "mov_text"]
    for k, s in enumerate(subs, len(srts)):  # the source's own stay as they are, unless Matroska can't hold them
        if mkv and s["codec_name"] == "mov_text":
            cmd += [f"-c:s:{k}", "srt"]

    def label(spec, lang, title):
        return [
            f"-metadata:s:{spec}",
            f"language={lang}",
            f"-metadata:s:{spec}",
            f"title={title}",
            f"-metadata:s:{spec}",
            f"handler_name={title}",
        ]

    for k, v in enumerate(voices):
        cmd += label(f"a:{k}", ISO3[tgt], titles["dub"][v]) + [f"-disposition:a:{k}", "default" if k == 0 else "0"]
    for k, s in enumerate(auds, len(voices)):
        tags = s.get("tags") or {}
        lang = tags.get("language") if tags.get("language") not in (None, "und") else ISO3[src]
        if k == len(voices) and lang != ISO3[src]:
            # The track the parts were cut from speaks the source language, whatever its tag
            # says (a web download can tag every track with the uploader's default).
            print(f"  the original audio was tagged {lang!r}: tagging it {ISO3[src]!r}")
            lang = ISO3[src]
        lang = original_audio_language(lang, src, quicktime_audio)
        cmd += label(f"a:{k}", lang, name_of(s) or LANGUAGES[src].TITLES["original"]) + [f"-disposition:a:{k}", "0"]
    for k, (_, lang, title) in enumerate(srts):
        cmd += label(f"s:{k}", lang, title) + [
            f"-disposition:s:{k}",
            "default" if k == 0 else "forced" if title == titles["signs"] else "0",
        ]
    for k, s in enumerate(subs, len(srts)):
        cmd += [f"-disposition:s:{k}", "0"]
    if not mkv:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(tmp))
    print(
        f"deliver {', '.join(voices)} into {source.name}"
        + (f" (replacing {dropped} earlier dub tracks)" if dropped else "")
    )
    subprocess.run(cmd, check=True)

    # Every stream copied, not converted, must match its input exactly.
    got = probe(tmp)["streams"]
    by_type = {t: [s for s in got if s["codec_type"] == t] for t in ("video", "audio", "subtitle")}
    checks = [(source, s["index"], by_type["video"][k]["index"]) for k, s in enumerate(vids)]
    checks += [(dubs[v], "a:0", by_type["audio"][k]["index"]) for k, v in enumerate(voices)]
    checks += [(source, s["index"], by_type["audio"][len(voices) + k]["index"]) for k, s in enumerate(auds)]
    for path, index, out_index in checks:
        if md5(path, index) != md5(tmp, out_index):
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"stream {index} of {path.name} changed in the copy; the source is untouched")
    if len(by_type["subtitle"]) != len(srts) + len(subs):
        tmp.unlink(missing_ok=True)
        raise SystemExit("subtitle tracks went missing; the source is untouched")
    tmp.replace(out)
    if out != source:
        source.unlink()
    for s in got:
        print(
            f"  #{s['index']} {s['codec_type']:8s} {s['codec_name']:10s} {(s.get('tags') or {}).get('language', ''):4s}"
            f" {name_of(s)}"
        )
    print(f"  wrote {out} (every copied stream verified)")
    return out


def deliver_subtitles(source, base, pair=PAIR):
    """A subtitle job's delivery: base is assemble.py --subtitles' <stem>.subtitled.mkv,
    whose subtitle files lie beside it. Returns the delivered file's path."""
    src, tgt = pair
    source = pathlib.Path(source)
    styled, srt, transcript = (base.with_suffix(f".{x}") for x in (f"{tgt}.ass", f"{tgt}.srt", f"{src}.srt"))
    if not srt.exists():
        raise SystemExit(f"no subtitles at {srt}")
    tracks = [(styled if styled.exists() else srt, ISO3[tgt], LANGUAGES[tgt].TITLES["subtitles"])]
    if transcript.exists():
        tracks.append((transcript, ISO3[src], LANGUAGES[src].TITLES["transcript"]))
    # Tracks an earlier delivery of subtitles added; a dub's tracks are another delivery's.
    mine = {m.TITLES[k] for m in LANGUAGES.values() for k in ("subtitles", "signs", "transcript") if k in m.TITLES}
    info = probe(source)
    keep = [s for s in info["streams"] if not (s["codec_type"] == "subtitle" and name_of(s) in mine)]
    vids = [s for s in keep if s["codec_type"] == "video"]
    auds = [s for s in keep if s["codec_type"] == "audio"]
    subs = [s for s in keep if s["codec_type"] == "subtitle"]
    out = delivered_name(source, mkv=True, suffix=SUBTITLED)
    tmp = out.with_name(out.stem + ".delivering" + out.suffix)

    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(source)]
    for path, _, _ in tracks:
        cmd += ["-i", str(path)]
    for s in vids + auds:
        cmd += ["-map", f"0:{s['index']}"]
    for k in range(len(tracks)):
        cmd += ["-map", f"{1 + k}:0"]
    for s in subs:
        cmd += ["-map", f"0:{s['index']}"]
    cmd += ["-c", "copy"]
    for k, (path, _, _) in enumerate(tracks):
        cmd += [f"-c:s:{k}", "ass" if path.suffix == ".ass" else "srt"]
    for k, s in enumerate(subs, len(tracks)):
        if s["codec_name"] == "mov_text":  # Matroska can't hold MP4's own
            cmd += [f"-c:s:{k}", "srt"]

    def label(spec, lang, title):
        return [f"-metadata:s:{spec}", f"language={lang}", f"-metadata:s:{spec}", f"title={title}"]

    for k, s in enumerate(auds):
        tags = s.get("tags") or {}
        lang = tags.get("language") if tags.get("language") not in (None, "und") else ISO3[src]
        if k == 0 and lang != ISO3[src]:
            print(f"  the audio was tagged {lang!r}: tagging it {ISO3[src]!r}")
            lang = ISO3[src]
        cmd += label(f"a:{k}", lang, name_of(s) or LANGUAGES[src].NATIVE)
    for k, (_, lang, title) in enumerate(tracks):
        cmd += label(f"s:{k}", lang, title) + [f"-disposition:s:{k}", "default" if k == 0 else "0"]
    for k, s in enumerate(subs, len(tracks)):
        cmd += [f"-disposition:s:{k}", "0"]
    cmd.append(str(tmp))
    dropped = len(info["streams"]) - len(keep)
    print(f"deliver subtitles into {source.name}" + (f" (replacing {dropped} earlier tracks)" if dropped else ""))
    subprocess.run(cmd, check=True)

    got = probe(tmp)["streams"]
    by_type = {t: [s for s in got if s["codec_type"] == t] for t in ("video", "audio", "subtitle")}
    checks = [(s, by_type["video"][k]) for k, s in enumerate(vids)]
    checks += [(s, by_type["audio"][k]) for k, s in enumerate(auds)]
    for s, o in checks:
        if md5(source, s["index"]) != md5(tmp, o["index"]):
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"stream {s['index']} of {source.name} changed in the copy; the source is untouched")
    if len(by_type["subtitle"]) != len(tracks) + len(subs):
        tmp.unlink(missing_ok=True)
        raise SystemExit("subtitle tracks went missing; the source is untouched")
    tmp.replace(out)
    if out != source:
        source.unlink()
    beside = out.with_suffix(f".{tgt}.srt")
    shutil.copyfile(srt, beside)
    for s in got:
        print(
            f"  #{s['index']} {s['codec_type']:8s} {s['codec_name']:10s} {(s.get('tags') or {}).get('language', ''):4s}"
            f" {name_of(s)}"
        )
    print(f"  wrote {out} (every copied stream verified) and {beside.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_video")
    ap.add_argument("dubs", nargs="*", metavar="voice=assembled.mp4")
    ap.add_argument("--default", help="the voice that plays first (default: actor if given, else the first)")
    ap.add_argument("--source", default="ja", help="language the source speaks (default ja)")
    ap.add_argument("--target", default="en", help="language of the dubs (default en)")
    ap.add_argument(
        "--quicktime-audio",
        action="store_true",
        help="tag English original audio as und to prefer the dub in QuickTime",
    )
    ap.add_argument(
        "--subtitles",
        metavar="STEM.subtitled.mkv",
        help="deliver a subtitle job instead: assemble.py --subtitles' output, its files beside it",
    )
    args = ap.parse_args()
    if args.subtitles:
        if args.dubs:
            raise SystemExit("--subtitles delivers subtitles alone: leave out the dubs")
        return deliver_subtitles(args.source_video, pathlib.Path(args.subtitles), (args.source, args.target))
    if not args.dubs:
        raise SystemExit("give the dubs: actor=FILE and/or american=FILE (or --subtitles)")
    if (args.source, args.target) not in PAIRS:
        raise SystemExit(f"{args.source} -> {args.target}: the scripts know {sorted(PAIRS)}")
    dubs = {}
    for d in args.dubs:
        voice, _, path = d.partition("=")
        if voice not in VOICE_LABEL or not path:
            raise SystemExit(f"{d}: expected actor=FILE or american=FILE")
        dubs[voice] = pathlib.Path(path)
    default = args.default or ("actor" if "actor" in dubs else next(iter(dubs)))
    deliver(args.source_video, dubs, default, (args.source, args.target), quicktime_audio=args.quicktime_audio)


if __name__ == "__main__":
    main()
