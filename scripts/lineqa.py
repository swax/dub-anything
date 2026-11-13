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
--reroll writes the flagged lines to <part>/reroll.json;
the next `dub.py generate` gives them fresh takes. That fixes random glitches,
but a word a voice can't say needs rewording. A fresh take can also come out
worse, so each take a re-roll replaces is kept (<part>/takes/), and the
next check puts the best one back if the new take transcribes worse; the next
`generate` then re-mixes it without synthesizing anything.
What Whisper heard in each take is kept by the take's hash (<part>/heard.json),
so a check after a round of fresh takes only listens to the takes that changed;
with none changed, Whisper isn't even loaded. On the CPU it listens in int8 (Ear),
and those hearings are kept apart from float32's and the GPU's.
In the "american" voice most flags come from the conversion, not from IndexTTS:
when IndexTTS's own take of a flagged line says it right, the line is queued to
be converted again (conversion is random too), keeping the take; after
RECONVERTS failed tries the take is re-rolled instead.
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

import soundfile as sf

from core import langs, line_id, load_json, part_dir, report_result, save_json, take_wav, whisper_setup
from languages import LANGUAGES

whisper_setup()
from whisper_safe import WhisperModel, hold_gpu_for_whisper, whisper_on_gpu  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def file_sha(path):
    return hashlib.sha1(pathlib.Path(path).read_bytes()).hexdigest()[:16]


def better(a, b):
    """Whether take a transcribes better than take b: no flags first, then the closer match."""
    return (not a["notes"], a["ratio"]) > (not b["notes"], b["ratio"])


def comparable_settings(settings):
    """A seeded private profile changes performance, not reference or render policy."""
    return {**{k: v for k, v in settings.items() if k != "seed_profile"}, "speed": None, "seed": None}


class Ear:
    """Whisper, loaded (and the GPU waited for) only when a take hasn't been heard before.
    On the CPU (an AMD GPU, a Mac) in int8: CTranslate2 would turn large-v3's float16
    into float32 there, which heard a part's takes 1.4-1.6x as slowly, flagging the same."""

    MODEL = "Systran/faster-whisper-large-v3"

    def __init__(self):
        self.model = None
        self.compute_type = "default" if whisper_on_gpu() else "int8"
        # heard.json keys a hearing by this: one in int8 isn't reused for float32's, or back.
        self.tag = self.MODEL.rsplit("-", 1)[-1] + ("" if self.compute_type == "default" else f"-{self.compute_type}")
        self.load_s = self.hear_s = 0.0  # waiting for the GPU and loading; listening
        self.record_provenance = False
        self.resolved_model_path = None
        self.queue_wait_s = self.model_load_s = 0.0

    def __call__(self, wav, lang):
        if self.model is None:
            t = time.time()
            hold_gpu_for_whisper()
            after_wait = time.time() if self.record_provenance else None
            source = self.MODEL
            if self.record_provenance:
                from faster_whisper.utils import download_model

                # Resolve the same requested reader once, retaining the actual
                # local weight revision used for the observation's fresh hearing.
                self.resolved_model_path = download_model(self.MODEL)
                source = self.resolved_model_path
            self.model = WhisperModel(source, device="auto", compute_type=self.compute_type)
            finished = time.time()
            self.load_s += finished - t
            if after_wait is not None:
                self.queue_wait_s += after_wait - t
                self.model_load_s += finished - after_wait
        t = time.time()
        got = hear(self.model, wav, lang)
        self.hear_s += time.time() - t
        return got


def check(name, ear, show_all, reroll, only=None, observations=None):
    work = part_dir(name)
    segments = json.loads((work / "segments.json").read_text(encoding="utf-8"))
    job = segments["job_id"]
    tgt = langs(name)[1]
    # What Whisper heard in each take, by the take's hash: the same audio is heard the
    # same way, so only new takes need listening to (a round re-rolls a few lines).
    heard_path = work / "heard.json"
    heard = load_json(heard_path, {})
    model_tag = getattr(ear, "tag", Ear.MODEL.rsplit("-", 1)[-1])
    new = []

    def listen(wav):
        key = f"{model_tag}:{tgt}:{file_sha(wav)}"
        if key not in heard:
            heard[key] = ear(wav, tgt)
            new.append(key)
        return heard[key]

    script = json.loads((work / "script.json").read_text(encoding="utf-8"))
    lines = script["lines"]
    observer = None
    if observations is not None:
        from qa_triage import ObservationStore

        if isinstance(ear, Ear):
            ear.record_provenance = True
        observer = ObservationStore(observations, name, work, segments, script, ear)
    # Takes re-rolled before, so a worse roll never replaces a better one: before a
    # flagged line is re-rolled its take is kept in takes/, with the settings it was
    # made with, and the best of them is put back if the fresh take is worse.
    rendered = load_json(work / "rendered.json", {})
    converted = load_json(work / "converted.json", {})  # "american": the take is a converted copy
    kept_path = work / "takes" / "kept.json"
    kept = load_json(kept_path, {})
    # Fresh takes each flagged line has had, per way of making it (rolls_key).
    rolls_path = work / "takes" / "rolls.json"
    rolls = load_json(rolls_path, {})
    flagged, restored = [], []
    for i, line in enumerate(lines):
        if only is not None and i not in only:
            continue
        lid = line_id(line)
        wav, src = take_wav(job, lid, tgt), take_wav(job, lid, tgt, "src")
        if not wav.exists():
            print(f"L{i:02d} {line['start']:7.2f}  (not rendered)  {line['text']}")
            continue
        cached = observer is not None and f"{model_tag}:{tgt}:{file_sha(wav)}" in heard
        hearing_started = time.perf_counter() if observer is not None else 0
        cost_fields = ("queue_wait_s", "model_load_s", "hear_s")
        cost_before = {k: getattr(ear, k, 0.0) for k in cost_fields} if observer is not None else {}
        got = listen(wav)
        notes, ratio = LANGUAGES[tgt].compare(line["text"], got)
        take = {
            "notes": notes,
            "ratio": round(ratio, 3),
            "got": got,
            "sha": file_sha(wav),
            "settings": rendered.get(lid, {}),
        }
        observation = (
            observer.capture(
                i,
                wav,
                got,
                take["settings"],
                cached,
                time.perf_counter() - hearing_started,
                {k: getattr(ear, k, 0.0) - v for k, v in cost_before.items()},
            )
            if observer is not None
            else None
        )

        # Only takes made the same way count (a moved window re-renders the line anyway);
        # a seed only tells two rolls apart.
        def same(k, take=take):
            return comparable_settings(k["settings"]) == comparable_settings(take["settings"])

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
                f"over {take['ratio']:.2f}: {best['got']!r})"
            )
            take, notes, got = best, best["notes"], best["got"]
        if notes:
            flagged.append({"id": lid, "start": line["start"], "text": line["text"], "got": got})
            seed = take["settings"].get("seed")
            tries = sum(k.get("reconverted", 0) for k in kept.get(lid, []) if k["settings"].get("seed") == seed)
            if reroll and lid in converted and src.exists() and tries < RECONVERTS:
                engine_got = listen(src)
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
            # Failed on SETTLE_TAKES takes: a word the voice can't say or a name Whisper
            # doesn't know, which further rolls rarely change. It stays flagged for the
            # report and the director, but gets no more fresh takes.
            rkey = rolls_key(lid, take["settings"])
            failed = 1 + rolls.get(rkey, 0)
            if reroll and not flagged[-1].get("reconvert"):
                if SETTLE_TAKES and failed >= SETTLE_TAKES:
                    flagged[-1]["settled"] = True
                    notes = notes + [f"failed on {failed} takes: not re-rolled (reword it, or leave it to the ear)"]
                else:
                    rolls[rkey] = rolls.get(rkey, 0) + 1
        if observer is not None:
            status = next((f for f in flagged if f["id"] == lid), None)
            observer.action(
                observation,
                {
                    "restored_best_prior": lid in restored,
                    "result_audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                    "result_settings": take["settings"],
                    "result_hearing": got,
                    "result_flag": status,
                    "repair_requested": bool(reroll and status and not status.get("settled")),
                    "reroll_enabled": reroll,
                },
                wav,
            )
        if notes or show_all:
            dur = sf.info(wav).duration
            print(
                f"L{i:02d} {line['start']:7.2f} {line['voice']:10s} {dur:4.2f}s  {'; '.join(notes) or 'ok'}\n"
                f"      want: {line['text']}\n      got:  {got}"
            )
    timing = (
        f" in {ear.hear_s:.0f} s, after {ear.load_s:.0f} s waiting for the GPU and loading"
        if new and hasattr(ear, "hear_s")
        else ""
    )
    print(
        f"{len(flagged)} of {len(lines) if only is None else len(only)} lines flagged "
        f"({len(new)} takes heard{timing}, the rest as in earlier checks)"
    )
    save_json(heard_path, heard)
    if kept:
        save_json(kept_path, kept)
    if rolls:
        save_json(rolls_path, rolls)
    if restored:
        save_json(work / "rendered.json", rendered)
        if converted:
            save_json(work / "converted.json", converted)
        print(f"put back {len(restored)} earlier takes: the next `scripts/dub.py generate {name}` re-mixes them")
    if reroll and (flagged or only is not None):
        queued = flagged
        if only is not None:
            selected_ids = {line_id(lines[i]) for i in only}
            queued = [r for r in load_json(work / "reroll.json", []) if r["id"] not in selected_ids] + flagged
        # Settled lines are listed too (for the report) but get no fresh take.
        save_json(work / "reroll.json", queued)
        again = sum(1 for f in flagged if f.get("reconvert"))
        settled = sum(1 for f in flagged if f.get("settled"))
        what = [
            f"{n} {kind}"
            for n, kind in ((len(flagged) - again - settled, "for fresh takes"), (again, "to be converted again"))
            if n
        ]
        if what:
            print(
                f"queued {' and '.join(what)} on the next `scripts/dub.py generate {name}` "
                f"(their current takes are kept, and put back if the new ones are worse)"
            )
        if settled:
            print(f"{settled} settled (failed on {SETTLE_TAKES} takes): not re-rolled")
    # Takes put back are only mixed in by the next generate: the runner re-mixes on "restored".
    return {"flagged": len(flagged), "restored": len(restored)}


RECONVERTS = 2  # conversions of one IndexTTS take tried again before it gets a fresh take
# Takes of a flagged line that fail QA before it stops being re-rolled (0: re-roll
# every round). On two films the first fresh take cleared most real glitches and
# later rounds almost none (docs/research.md, "Settling QA flags").
SETTLE_TAKES = 2


def rolls_key(lid, settings):
    """A line and the way its takes are made (a seed or speed only tells two rolls apart)."""
    sig = json.dumps(comparable_settings(settings), sort_keys=True)
    return f"{lid}:{hashlib.sha1(sig.encode()).hexdigest()[:8]}"


def hear(model, wav, lang="en"):
    segs, _ = model.transcribe(str(wav), language=lang, beam_size=5, vad_filter=False, condition_on_previous_text=False)
    return LANGUAGES[lang].JOIN.join(s.text.strip() for s in segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", metavar="name", help="one or more rendered parts (Whisper loads once)")
    ap.add_argument("--all", action="store_true", help="print every line, not just flagged ones")
    ap.add_argument("--reroll", action="store_true", help="queue flagged lines for fresh takes")
    ap.add_argument("--only", help="check only these zero-based comma-separated indices; preserve other queued lines")
    ap.add_argument("--metrics", type=pathlib.Path, help="optional cost receipt (queue/model load and transcription)")
    ap.add_argument(
        "--observations",
        type=pathlib.Path,
        default=os.environ.get("DUB_QA_OBSERVATIONS"),
        help="archive pre-repair QA evidence without changing repair policy (or DUB_QA_OBSERVATIONS)",
    )
    args = ap.parse_args()
    ear = Ear()
    started = time.perf_counter()
    summary = {}
    for name in args.names:
        if len(args.names) > 1:
            print(f"=== {name}")
        from dub import selected_indices

        lines = load_json(part_dir(name) / "script.json")["lines"]
        summary[name] = check(
            name, ear, args.all, args.reroll, selected_indices(args.only, len(lines)), observations=args.observations
        )
    report_result(parts=summary)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(
            json.dumps(
                {
                    "model_wait_load_s": ear.load_s,
                    "transcription_s": ear.hear_s,
                    "process_check_wall_s": time.perf_counter() - started,
                    "parts": args.names,
                    "only": args.only,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
