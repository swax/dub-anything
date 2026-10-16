"""Transcribe every rendered line on its own and flag the ones that don't match.

Usage: scripts/lineqa.py <name> [<name> ...] [--all] [--reroll]   (after `dub.py generate`)

Full-mix QA (qa.py) hides problems that show up clearly per line:
  - extra words before the first expected one: voice-reference leakage
    (a voice clip in voices.json that cuts a word off or holds someone else)
  - dropped or changed words (fast lists), and flipped meanings ("can" heard
    as "can't": any change in negation is flagged, and so is a number heard as
    another, which a long line otherwise hides)
  - words a voice can't say, and silent takes
Takes are heard in the part's target language and compared as its module in
languages/ says (compare). Both sides are normalized first, so digits and
spelling don't count as failures: English and Spanish word by word (numbers
spelled out, contractions expanded, "Dr." = "doctor", compounds joined; Spanish
without accents, as Whisper writes "esta" or "está"), Japanese by its reading in
kana (languages/kana.py: kanji or kana, katakana or hiragana, digits or kanji
numerals), where a take with English in it (the actor's original leaking from
the per-line reference) is flagged too.
--reroll writes the flagged lines to work/<name>/reroll.json;
the next `dub.py generate` gives them fresh takes. That fixes random glitches,
but a word a voice can't say needs rewording. A fresh take can also come out
worse, so each take a re-roll replaces is kept (work/<name>/takes/), and the
next check puts the best one back if the new take transcribes worse; the next
`generate` then re-mixes it without synthesizing anything.
In the "american" voice most flags come from the conversion, not from IndexTTS:
when IndexTTS's own take of a flagged line says it right, the line is queued to
be converted again (conversion is random too), keeping the take; after
RECONVERTS failed tries the take is re-rolled instead.
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import soundfile as sf

from core import WORK, langs, line_id, load_json, take_wav, whisper_setup
from gpulock import hold_gpu
from languages import LANGUAGES

whisper_setup()
from whisper_safe import WhisperModel  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def file_sha(path):
    return hashlib.sha1(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def better(a, b):
    """Whether take a transcribes better than take b: no flags first, then the closer match."""
    return (not a["notes"], a["ratio"]) > (not b["notes"], b["ratio"])


def check(name, model, show_all, reroll):
    work = WORK / name
    job = json.loads((work / "segments.json").read_text(encoding="utf-8"))["job_id"]
    tgt = langs(name)[1]
    lines = json.loads((work / "script.json").read_text(encoding="utf-8"))["lines"]
    # Takes re-rolled before, so a worse roll never replaces a better one: before a
    # flagged line is re-rolled its take is kept in takes/, with the settings it was
    # made with, and the best of them is put back if the fresh take is worse.
    rendered = load_json(work / "rendered.json", {})
    converted = load_json(work / "converted.json", {})  # "american": the take is a converted copy
    kept_path = work / "takes" / "kept.json"
    kept = load_json(kept_path, {})
    flagged, restored = [], []
    for i, line in enumerate(lines):
        lid = line_id(line)
        wav, src = take_wav(job, lid, tgt), take_wav(job, lid, tgt, "src")
        if not wav.exists():
            print(f"L{i:02d} {line['start']:7.2f}  (not rendered)  {line['text']}")
            continue
        got = hear(model, wav, tgt)
        notes, ratio = LANGUAGES[tgt].compare(line["text"], got)
        take = {
            "notes": notes,
            "ratio": round(ratio, 3),
            "got": got,
            "sha": file_sha(wav),
            "settings": rendered.get(lid, {}),
        }

        # Only takes made the same way count (a moved window re-renders the line anyway);
        # a seed only tells two rolls apart.
        def same(k, take=take):
            return {**k["settings"], "speed": None, "seed": None} == {**take["settings"], "speed": None, "seed": None}

        earlier = [k for k in kept.get(lid, []) if same(k) and k["sha"] != take["sha"]]
        # Stored scores predate later language-normalization fixes. Compare all
        # candidates with today's rules before choosing or restoring a take.
        for k in earlier:
            notes_now, ratio_now = LANGUAGES[tgt].compare(line["text"], k["got"])
            k.update(notes=notes_now, ratio=round(ratio_now, 3))
        best = max(earlier, key=lambda k: (not k["notes"], k["ratio"]), default=None)
        if best and better(best, take):
            shutil.copy2(work / "takes" / best["file"], wav)
            if best.get("src"):
                shutil.copy2(work / "takes" / best["src"], src)
                converted[lid] = file_sha(wav)
            rendered[lid] = best["settings"]
            restored.append(lid)
            print(
                f"L{i:02d} {line['start']:7.2f} put back an earlier take (match {best['ratio']:.2f} "
                f"over {take['ratio']:.2f}: {got!r})"
            )
            take, notes, got = best, best["notes"], best["got"]
        if notes:
            flagged.append({"id": lid, "start": line["start"], "text": line["text"], "got": got})
            seed = take["settings"].get("seed")
            tries = sum(k.get("reconverted", 0) for k in kept.get(lid, []) if k["settings"].get("seed") == seed)
            if reroll and lid in converted and src.exists() and tries < RECONVERTS:
                engine_got = hear(model, src, tgt)
                if not LANGUAGES[tgt].compare(line["text"], engine_got)[0]:
                    flagged[-1]["reconvert"] = True
                    notes = notes + [f"IndexTTS's take is fine ({engine_got!r}): converting it again"]
            if reroll and not any(k["sha"] == take["sha"] for k in kept.get(lid, [])):
                (work / "takes").mkdir(exist_ok=True)
                n = len(kept.get(lid, []))
                take["file"] = f"{lid}_{n}.wav"
                shutil.copy2(wav, work / "takes" / take["file"])
                if lid in converted and src.exists():
                    take["src"] = f"{lid}_{n}.src.wav"
                    shutil.copy2(src, work / "takes" / take["src"])
                kept.setdefault(lid, []).append(take)
            if flagged[-1].get("reconvert"):
                entry = next(k for k in kept[lid] if k["sha"] == take["sha"])
                entry["reconverted"] = entry.get("reconverted", 0) + 1
        if notes or show_all:
            dur = sf.info(wav).duration
            print(
                f"L{i:02d} {line['start']:7.2f} {line['voice']:10s} {dur:4.2f}s  {'; '.join(notes) or 'ok'}\n"
                f"      want: {line['text']}\n      got:  {got}"
            )
    print(f"{len(flagged)} of {len(lines)} lines flagged")
    if kept:
        kept_path.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    if restored:
        (work / "rendered.json").write_text(json.dumps(rendered, indent=1), encoding="utf-8")
        if converted:
            (work / "converted.json").write_text(json.dumps(converted, indent=1), encoding="utf-8")
        print(f"put back {len(restored)} earlier takes: the next `scripts/dub.py generate {name}` re-mixes them")
    if reroll and flagged:
        (work / "reroll.json").write_text(json.dumps(flagged, ensure_ascii=False, indent=1), encoding="utf-8")
        again = sum(1 for f in flagged if f.get("reconvert"))
        what = [
            f"{n} {kind}"
            for n, kind in ((len(flagged) - again, "for fresh takes"), (again, "to be converted again"))
            if n
        ]
        print(
            f"queued {' and '.join(what)} on the next `scripts/dub.py generate {name}` "
            f"(their current takes are kept, and put back if the new ones are worse)"
        )


RECONVERTS = 2  # conversions of one IndexTTS take tried again before it gets a fresh take


def hear(model, wav, lang="en"):
    segs, _ = model.transcribe(str(wav), language=lang, beam_size=5, vad_filter=False, condition_on_previous_text=False)
    return LANGUAGES[lang].JOIN.join(s.text.strip() for s in segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", metavar="name", help="one or more rendered parts (Whisper loads once)")
    ap.add_argument("--all", action="store_true", help="print every line, not just flagged ones")
    ap.add_argument("--reroll", action="store_true", help="queue flagged lines for fresh takes")
    args = ap.parse_args()
    hold_gpu()
    model = WhisperModel("Systran/faster-whisper-large-v3", device="auto", compute_type="default")
    for name in args.names:
        if len(args.names) > 1:
            print(f"=== {name}")
        check(name, model, args.all, args.reroll)


if __name__ == "__main__":
    main()
