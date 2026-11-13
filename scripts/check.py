"""Every check a directed part needs before it renders, in one pass and one short report.

Usage: scripts/check.py <part> [<part> ...] [--ready] [--all] [--dry-run]

Run it after writing a part's script.json, instead of running edges.py --apply,
tails.py and fit.py (in each of the job's voices) one at a time. For each part:

  structure  what `dub_anything.py ready` would refuse; nothing else runs until it's fixed
  edges      edges.py --apply: the windows it moved (script.json is rewritten, so
             re-read the lines it names before editing them) and the edges it left alone
  echo       tails.py's wet lines that have no "effect" yet
  <voice>    fit.py in each of the job's voices: flagged lines, each with the reason
             it falls back to the character's reference, and references whose pace is off

A part to subtitle gets its structure and reading speed only.

Lines marked "!" must be fixed before rendering: structural problems, OVER and
OVER?, "risk" on a window under 1 s (short English is stretched past what fill
can recover), and in subtitles FAST, LONG and SPLIT?. The rest are to review, not to
pad: SHORT?, risk, edges left alone, wet lines, fallbacks, reference reviews.
Only what needs a look is printed; --all prints every line of each check.

--ready marks each part ready when nothing must be fixed (as `dub_anything.py
ready` does), so the run renders it: review the rest first. --dry-run leaves
the windows as they are. Exit 0 when nothing must be fixed, 2 otherwise.
"""

import argparse
import sys
import warnings

# The models' libraries warn about deprecations and precision on every load: noise in a short report.
warnings.filterwarnings("ignore")

import dub_anything as runner  # noqa: E402
import edges  # noqa: E402
import fit  # noqa: E402
import tails  # noqa: E402
import core  # noqa: E402
from core import base_part, fit_rows, load_json, load_voices, part_dir  # noqa: E402

MUST_FIT = {"OVER", "OVER?"}
SHORT_WINDOW = 1.0  # s; "risk" on a window shorter than this is as good as OVER?
MUST_READ = {"FAST", "LONG", "SPLIT?"}  # docs/directing/subtitles.md: fix every one


def job_of(name):
    """The dub_anything job the part belongs to, or None for a part made by hand."""
    slug = core.job_of(base_part(name))
    return load_json(core.job_dir(slug) / "job.json") if slug else None


def fit_must(row):
    return row["flag"] in MUST_FIT or (row["flag"] == "risk" and row["window"] < SHORT_WINDOW)


def show(rows, show_all):
    """Print (mark, text, notes) rows that need a look (all with show_all), "!" on the ones to fix."""
    for marked, text, extra in rows:
        if marked or extra or show_all:
            print(f"{'!' if marked == 'must' else ' '} {text}")
            for note in extra:
                print(f"      {note}")


def check(name, apply=True, show_all=False):
    """Print one part's report; return how many things must be fixed, or None if it can't be checked."""
    work = part_dir(name)
    if base_part(name) != name:
        print(f"== {name}: a voice's copy, made from {base_part(name)}'s script: check that part")
        return None
    seg, script = load_json(work / "segments.json"), load_json(work / "script.json")
    if seg is None or script is None:
        print(f"== {name}: {'not prepped' if seg is None else 'no script.json yet'}")
        return None
    job = job_of(name)
    subs = bool(seg.get("subtitles_only") or (job and runner.subtitles_only(job)))
    voices = [] if subs else job["voices"] if job else [script.get("render", {}).get("voice", "actor")]
    lines = script["lines"]
    print(f"== {name}: {len(lines)} lines; " + ("subtitles" if subs else ", ".join(voices)))
    if not lines:
        print("  left original: nothing to check")
        return 0

    cast = None if subs else load_voices()
    problems = runner.ready_problems(script, cast, american="american" in voices)
    if problems:
        print(f"structure: {len(problems)} to fix")
        for problem in problems:
            print(f"! {problem}")
        return len(problems)
    must = 0

    if subs:
        rows = fit.reading_rows(name)
        flagged = [r for r in rows if r["flag"]]
        to_fix = [r for r in flagged if MUST_READ & set(r["flag"].split())]
        must += len(to_fix)
        print(f"reading: {len(to_fix)} to fix, {len(flagged) - len(to_fix)} to review")
        show(
            [("must" if r in to_fix else "review" if r["flag"] else "", fit.reading_text(r), []) for r in rows],
            show_all,
        )
    else:
        moved, ends, joins = edges.analyse(name, apply)
        messages = edges.messages(moved, ends, joins)
        fixed = [text for fix, text in messages if fix]
        left = [text for fix, text in messages if not fix]
        verb = "moved" if apply else "would move"
        print(
            f"edges: {len(fixed)} {verb}, {len(left)} left alone"
            + ("; script.json rewritten" if apply and fixed else "")
        )
        if fixed and not show_all:
            # The moves, compactly: "L03 start 12.40 -> 12.36".
            print("  " + "; ".join(" ".join(text.split()[1:6]) for text in fixed))
        for fix, text in messages:
            if not fix or show_all:
                print(text)

        found = tails.rows(name)
        wet = [r for r in found if r[2] >= tails.WET_MS and not r[1].get("effect")]
        handled = sum(r[2] >= tails.WET_MS and bool(r[1].get("effect")) for r in found)
        print(f"echo: {len(wet)} wet without an effect" + (f" ({handled} with one)" if handled else ""))
        for row in found if show_all else wet:
            print(tails.row_text(*row))

        for voice in voices:
            notes = {}
            rows = fit_rows(name, predict=True, short=True, voice=voice, notes=notes)
            to_fix = [r for r in rows if fit_must(r)]
            review = [r for r in rows if r["flag"] and not fit_must(r)]
            fallbacks = sum(any(n.startswith("line ref skipped") for n in notes.get(r["i"], [])) for r in rows)
            must += len(to_fix)
            print(
                f"{voice}: {len(to_fix)} to fix, {len(review)} to review"
                + (f", {fallbacks} on the character's reference" if fallbacks else "")
            )
            if not fit.american(name, voice):
                for warning in fit.pace_warnings(name):
                    print(f"  {warning}")
            show(
                [
                    ("must" if fit_must(r) else "review" if r["flag"] else "", fit.row_text(r), notes.get(r["i"], []))
                    for r in rows
                ],
                show_all,
            )
    return must


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--ready", action="store_true", help="mark each part ready when nothing must be fixed")
    ap.add_argument("--all", action="store_true", help="print every line of each check, not only the flags")
    ap.add_argument("--dry-run", action="store_true", help="report edge moves without writing them")
    args = ap.parse_args()
    blocked = 0
    for name in args.parts:
        must = check(name, apply=not args.dry_run, show_all=args.all)
        blocked += must != 0
        if must is None:
            continue
        script = load_json(part_dir(name) / "script.json")
        if must:
            print(f"next: fix the {must} marked !, then scripts/check.py {name}" + (" --ready" if args.ready else ""))
        elif args.ready and script is not None:
            if script["lines"] and not script.get("ready"):
                script["ready"] = True
                runner.write_json(part_dir(name) / "script.json", script)
            print("ready: it renders when dub_anything.py run gets to it")
        elif script is not None and script["lines"] and not script.get("ready"):
            print(f"next: nothing to fix; review the rest, then scripts/check.py {name} --ready")
    sys.exit(2 if blocked else 0)


if __name__ == "__main__":
    main()
