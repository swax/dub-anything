"""Run a dub from source video to the delivered file: the machine work, around the director's.

Usage:
  scripts/dub_anything.py new <video> [--source ja --target en] [--voices actor|american|both] [--slug NAME] [--minutes 3] [--whole A-B ...] [--captions|--subtitles|--auto-text] [--signs]
  scripts/dub_anything.py run <slug> [--watch]       do every step that can run now (see below)
  scripts/dub_anything.py status <slug>              where each part stands, and what blocks it
  scripts/dub_anything.py ready <slug> <NN> [...]    mark parts' scripts as directed (same as "ready": true)
  scripts/dub_anything.py report <slug>              write and print the report (work/jobs/<slug>/report.md)
  scripts/dub_anything.py cleanup <slug>             after delivery, move the bulk to videos/_to-delete/<slug>/

A job (work/jobs/<slug>/job.json) is one source video, the language pair (Japanese
-> English unless --source/--target say otherwise; the American voice is English
only) and the voices to dub it in ("both" plays the actor's voice by default). The
source is hard-linked into videos/<slug>.<ext> so the split keeps working after
delivery renames it.

run does, in order and only what isn't done yet: split; dub.py prep, then
separate.py, subs.py (with --captions, --subtitles or --auto-text) and draft.py, first over the first part to
dub (so the director can start on it) and then PREP_CHUNK parts at a time in film
order, so drafts keep coming; second.py over them in the background (CPU); with
--signs, signs.py over each part once every part is drafted. For
each part whose script the director has marked ready (or left with "lines": []),
in each voice: render, lineqa, and up to QA_ROUNDS rounds of fresh takes or
conversions for the flagged lines (a line that fails on a fresh take too is
settled and gets no more: lineqa.SETTLE_TAKES); what is still flagged goes to the
report (reroll-left.json). A ready part goes first: preparation only runs, one
step at a time (a part's prep, a chunk's separation or draft), while no ready
part needs rendering, so the first parts render while the rest still prepares. The American
voice renders in <part>.us, from a copy of the director's script (a line's
"text_us" replaces its text there) and the actor job's stems. When every part is
final in every voice: assemble.py per voice, deliver.py into the source (renamed
"<name> (dubbed)"), and the report. A part that overflows is blocked until its
script changes, and one whose render fails twice until run starts again; the
others carry on. The backend is restarted before the first render after a prep,
and when a render runs out of GPU memory, since its cache keeps what it loaded. --watch
keeps waiting for scripts until delivery, and stops early when a part is blocked,
so the agent can tell the user. Exit codes: 0 done or idle, 2 needs the director,
1 error. Every GPU step waits for the GPU lock (gpulock.py); logs are in
work/jobs/<slug>/logs/.
"""

import argparse
import filecmp
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import backend
from core import (
    JOBS,
    ROOT,
    VOICE_LABEL,
    WORK,
    actor_ends,
    langs,
    line_id,
    load_json,
    speech_length,
    take_wav,
    voice_part,
)
from languages import LANGUAGES, NAMES, PAIR, PAIRS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
JOB_DIR = WORK / "jobs"
VIDEOS = ROOT / "videos"
QA_ROUNDS = 3  # rounds of fresh takes for flagged lines after the first check
PREP_CHUNK = 4  # parts prepped, separated and drafted together, after the first part
PY = backend.vs_python()
UV = ["uv", "run", "--no-project"]
if sys.platform == "darwin":
    SEPARATE = UV + [
        "--python",
        "3.11",
        "--with",
        "torch",
        "--with",
        "audio-separator",
        "--with",
        "onnxruntime",
        "python",
        "scripts/separate.py",
    ]
else:
    SEPARATE = UV + [
        "--python",
        "3.11",
        "--index",
        "https://download.pytorch.org/whl/cu128",
        "--index-strategy",
        "unsafe-best-match",
        "--with",
        "torch==2.8.0",
        "--with",
        "audio-separator",
        "--with",
        "onnxruntime",
        "python",
        "scripts/separate.py",
    ]
SECOND = UV + [
    "--python",
    "3.12",
    "--with",
    "sherpa-onnx",
    "--with",
    "soundfile",
    "--with",
    "scipy",
    "--with",
    "pykakasi",
    "--with",
    "huggingface_hub",
    "python",
    "scripts/second.py",
]
if sys.platform == "darwin":
    SUBS = UV + [
        "--python",
        "3.12",
        "--with",
        "rapidocr",
        "--with",
        "onnxruntime",
        "python",
        "scripts/subs.py",
    ]
else:
    SUBS = UV + [
        "--python",
        "3.12",
        "--with",
        "rapidocr",
        "--with",
        "onnxruntime-gpu[cuda,cudnn]",
        "python",
        "scripts/subs.py",
    ]
SIGNS = SUBS[:-1] + ["scripts/signs.py"]
STEMS = ("vocals.wav", "no_vocals.wav")
AMERICAN_RENDER = {"line_ref": False, "fill": True, "dips": False, "cast_lines": False, "voice": "american"}
ACTOR_RENDER = {"line_ref": True, "fill": True, "dips": True, "cast_lines": True, "voice": "actor"}
ENV = {**backend.full_env(), "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
GPU_FULL = re.compile(r"out of memory|Generation failed", re.I)  # the backend's cache filled the card
RETRY = ("render failed", "lineqa failed")  # problems a new run retries; the rest wait for a script change


# ---- jobs and parts ----


def job_file(slug):
    return JOB_DIR / slug / "job.json"


def load_job(slug):
    job = load_json(job_file(slug))
    if not job:
        raise SystemExit(f"no job {slug!r} (scripts/dub_anything.py new <video>)")
    return job


def save_job(job):
    job_file(job["slug"]).write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")


def job_langs(job):
    """(source, target) of a job; jobs made before the pair was recorded are Japanese -> English."""
    return tuple(job.get("langs", PAIR))


def parts(job):
    pj = load_json(VIDEOS / f"{job['slug']}-parts" / "parts.json")
    return pj["parts"] if pj else []


def write_json(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def job_dir_of(name):
    seg = load_json(WORK / name / "segments.json")
    return JOBS / seg["job_id"] if seg else None


def film_time(seconds):
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{seconds % 60:04.1f}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pause_checkpoint():
    """An optional supervisor can request a pause between completed tool steps."""
    marker = os.environ.get("DUB_PAUSE_FILE")
    if marker and pathlib.Path(marker).is_file():
        log("Pause requested; finishing background checks before stopping at this checkpoint")
        for process, *_ in SECOND_RUN.values():
            process.wait()
        raise SystemExit(75)


def tool(job, label, cmd, logname):
    """Run one step, its output to the job's log; returns (ok, output)."""
    pause_checkpoint()
    logs = JOB_DIR / job["slug"] / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"{label} ...")
    path = logs / f"{logname}.log"
    with open(path, "w", encoding="utf-8") as f:  # written as it runs, to follow a long render
        r = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    out = path.read_text(encoding="utf-8", errors="replace")
    log(f"{label} {'ok' if r.returncode == 0 else 'FAILED'} ({time.time() - t0:.0f} s)")
    return r.returncode == 0, out


# ---- preparing ----


def split(job):
    pdir = VIDEOS / f"{job['slug']}-parts"
    if (pdir / "parts.json").exists():
        return True
    try:
        import faster_whisper  # noqa: F401

        py = [PY]
    except ImportError:
        py = UV + ["--python", "3.12", "--with", "faster-whisper", "python"]
    cmd = py + ["scripts/split.py", job["video"], "--minutes", str(job["split"]["minutes"]), "--out", str(pdir)]
    for w in job["split"]["whole"]:
        cmd += ["--whole", w]
    return tool(job, "split", cmd, "split")[0]


def original_only(p):
    script = load_json(WORK / p["name"] / "script.json")
    return script is not None and not script["lines"]


def drafted(p):
    return (WORK / p["name"] / "draft.json").exists()


def prep_chunk(ps):
    """The parts to prepare next: the first part to dub on its own, so the director
    can start on it, then PREP_CHUNK at a time in film order. Each chunk is prepped,
    separated and drafted before the next starts, so drafts keep coming while the
    models still load once per chunk."""
    todo = [p for p in ps if not original_only(p) and not drafted(p)]
    return todo[: PREP_CHUNK if any(drafted(p) for p in ps) else 1]


def prepare_step(job, ps):
    """One unit of preparation, run whenever no part is ready to render: prep one
    part of the current chunk, or separate, read the captions of or draft the
    chunk. Returns 'busy', 'done' or 'failed'."""
    chunk = prep_chunk(ps)
    if not chunk:
        # Text in the picture, for translating: after the drafts, which the director needs first.
        for p in ps if job.get("signs") else []:
            if not original_only(p) and not (WORK / p["name"] / "signs.json").exists():
                if not tool(job, f"signs {p['name']}", SIGNS + [p["name"]], f"signs-{p['name']}")[0]:
                    problem(job, f"signs.py failed on {p['name']}")
                    return "failed"
                return "busy"
        return "done"
    names = [p["name"] for p in chunk]
    label = names[0] if len(names) == 1 else f"{names[0]}..{names[-1].rsplit('-', 1)[-1]}"
    for p in chunk:
        if not (WORK / p["name"] / "segments.json").exists():
            src, tgt = job_langs(job)
            ok, out = tool(
                job,
                f"prep {p['name']}",
                [PY, "scripts/dub.py", "prep", p["file"], "--source", src, "--target", tgt],
                f"prep-{p['name']}",
            )
            if not ok and "produced no segments" in out:
                # No speech at all (a silent channel card): nothing to dub, so it stays original.
                (WORK / p["name"]).mkdir(exist_ok=True)
                write_json(WORK / p["name"] / "script.json", {"lines": []})
                log(f"{p['name']}: no speech, left original")
                return "busy"
            if not ok:
                problem(job, f"prep {p['name']} failed: {last_error(out)}")
                return "failed"
            # Prep loads transcription, diarisation and separation into the backend, and its
            # cache keeps that memory: renders beside it run out of GPU memory.
            BACKEND["dirty"] = True
            return "busy"
    todo = [n for n in names if not (job_dir_of(n) / "separator.json").exists()]
    if todo:
        if not tool(job, f"separate {label}", SEPARATE + todo, f"separate-{label}")[0]:
            problem(job, "separate.py failed (see its log)")
            return "failed"
        return "busy"
    for n in names:
        if job.get("captions") and not (WORK / n / "subs.json").exists():
            extra = {"open": ["--open"], "auto": ["--auto"]}.get(job["captions"], [])
            if not tool(job, f"subs {n}", SUBS + [n] + extra, f"subs-{n}")[0]:
                problem(job, f"subs.py failed on {n}")
                return "failed"
            return "busy"
    if not tool(job, f"draft {label}", [PY, "scripts/draft.py"] + names, f"draft-{label}")[0]:
        problem(job, "draft.py failed (see its log)")
        return "failed"
    return "busy"


BACKEND = {"dirty": False}  # prep has run since the backend last started


def clean_backend(job):
    """Restart the backend before a render if prep has used it since it started."""
    if BACKEND["dirty"]:
        log("restarting the backend to free the GPU memory prep used")
        if not backend.restart():
            raise SystemExit(problem(job, "the backend won't restart (scripts/backend.py status)") or 2)
        BACKEND["dirty"] = False


SECOND_RUN = {}


def second(job, ps):
    """second.py on the CPU, in the background: the director reads its log while directing."""
    proc = SECOND_RUN.get(job["slug"])
    if proc and proc[0].poll() is not None:
        # Done either way: a failure is reported once, not retried every pass.
        job["second"] = sorted(set(job.get("second", [])) | set(proc[1]))
        save_job(job)
        if proc[0].returncode:
            problem(job, f"second.py failed (see {proc[2].name}); run it by hand if you need its opinions")
        proc[2].close()
        SECOND_RUN.pop(job["slug"])
    elif proc:
        return
    todo = [
        p["name"] for p in ps if (WORK / p["name"] / "draft.json").exists() and p["name"] not in job.get("second", [])
    ]
    if todo:
        logs = JOB_DIR / job["slug"] / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        f = open(logs / f"second-{time.strftime('%H%M%S')}.log", "w", encoding="utf-8")
        log(f"second.py over {len(todo)} parts in the background (CPU)")
        SECOND_RUN[job["slug"]] = (
            subprocess.Popen(SECOND + todo, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT),
            todo,
            f,
        )


def problem(job, text):
    log(f"PROBLEM: {text}")
    job.setdefault("problems", [])
    if text not in job["problems"]:
        job["problems"].append(text)
        save_job(job)
    return False


def last_error(out):
    lines = [l for l in out.strip().splitlines() if l.strip()]
    errs = [l for l in lines if re.search(r"error|failed|exception|exit", l, re.I)]
    return (errs or lines or ["no output"])[-1].strip()[:300]


# ---- directing hand-off ----


def ready(p):
    script = load_json(WORK / p["name"] / "script.json")
    return script is not None and (not script["lines"] or bool(script.get("ready")))


def derived_script(script, voice):
    """The script a voice renders from: the director's, with "text_us" for the American voice."""
    if voice == "actor":
        return script
    s = {k: v for k, v in script.items() if k not in ("ready", "render")}
    s["lines"] = [
        {**{k: v for k, v in line.items() if k != "text_us"}, "text": line.get("text_us", line["text"])}
        for line in script["lines"]
    ]
    s["render"] = AMERICAN_RENDER
    return s


def variant(job, p, voice):
    """The voice's work folder for part p, set up (script, VoiceStudio job, stems); None if it isn't ready."""
    name = voice_part(p["name"], voice)
    base = WORK / p["name"]
    script = load_json(base / "script.json")
    if voice == "actor":
        want = {**ACTOR_RENDER, "line_ref": PAIRS[langs(p["name"])]["line_ref"]}
        if load_json(base / "rendered.json"):
            # Rendered before "cast_lines" existed: dub.py reads it missing as off, and
            # writing it would change the script's record and render the part again.
            want.pop("cast_lines")
        if script["lines"] and any(k not in script.get("render", {}) for k in want):
            # Complete it before the first render, so dub.py doesn't change the script under it.
            script["render"] = {**want, **script.get("render", {})}
            write_json(base / "script.json", script)
        return name
    work = WORK / name
    work.mkdir(exist_ok=True)
    write_json(work / "script.json", derived_script(script, voice))
    if not script["lines"]:
        return name
    video = pathlib.Path(p["file"]).with_name(f"{name}.mp4")
    if not video.exists():
        try:
            os.link(p["file"], video)
        except OSError:
            shutil.copy2(p["file"], video)
    if not (work / "segments.json").exists():
        ok, out = tool(job, f"prep {name}", [PY, "scripts/dub.py", "prep", str(video)], f"prep-{name}")
        if not ok:
            problem(job, f"prep {name} failed: {last_error(out)}")
            return None
    new, old = job_dir_of(name), job_dir_of(p["name"])
    if not (new / "separator.json").exists():
        # Same audio as the actor job, which is separated already: reuse its stems.
        if not (old / "separator.json").exists() or not filecmp.cmp(
            old / "audio_hq.wav", new / "audio_hq.wav", shallow=False
        ):
            problem(job, f"{name}: no separated stems to share from {p['name']}")
            return None
        for stem in STEMS:
            (new / stem).replace(new / stem.replace(".wav", ".htdemucs.wav"))
            shutil.copy2(old / stem, new / stem)
        shutil.copy2(old / "separator.json", new / "separator.json")
    return name


# ---- rendering and QA, one part in one voice ----


def script_sha(script):
    return hashlib.sha1(
        json.dumps(
            {"lines": script["lines"], "render": script.get("render")}, sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()[:12]


def finished(p, voice):
    """Part p is final in this voice for its current script: a script edited after
    its part's turn in a pass still needs that part rendered again."""
    script = load_json(WORK / p["name"] / "script.json")
    if not script["lines"]:
        return True
    st = load_json(WORK / voice_part(p["name"], voice) / "state.json", {})
    return st.get("stage") == "final" and st.get("script") == script_sha(derived_script(script, voice))


def advance(job, name):
    """One step for a part in one voice: 'busy' (did something), 'final', or 'blocked'."""
    work = WORK / name
    script = load_json(work / "script.json")
    if not script["lines"]:
        return "final"
    path = work / "state.json"
    st = load_json(path, {})
    sha = script_sha(script)
    if st.get("script") != sha:
        if st:
            log(f"{name}: script changed, rendering again")
        for f in ("reroll.json", "reroll-left.json"):
            (work / f).unlink(missing_ok=True)
        st = {"script": sha, "round": 0, "stage": "render"}
    if st.get("problem"):
        return "blocked"

    def save():
        path.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")

    stage = st["stage"]
    if stage == "final":
        return "final"
    if stage in ("render", "remix"):
        clean_backend(job)
        what = {"render": "render" if not st["round"] else f"round {st['round']}", "remix": "re-mix"}[stage]
        ok, out = tool(
            job,
            f"{what} {name}",
            [PY, "scripts/dub.py", "generate", name, "--match-level"],
            f"gen-{name}-{st['round']}{'-remix' if stage == 'remix' else ''}",
        )
        if not ok:
            over = re.findall(r"^\s+(L\d+\s+[\d.]+\s.*\sOVER\s+.*)$", out, re.M)
            st["fails"] = st.get("fails", 0) + 1
            if over:
                st["problem"] = "lines don't fit their windows: " + "; ".join(o.strip() for o in over)
            elif st["fails"] >= 2:
                st["problem"] = f"render failed twice: {last_error(out)}"
            elif GPU_FULL.search(out):
                log("the backend ran out of GPU memory: restarting it")
                backend.restart()
                BACKEND["dirty"] = False
            else:
                backend.ensure()
            save()
            return "blocked" if st.get("problem") else "busy"
        st["fails"] = 0
        st["stage"] = "qa" if stage == "render" else "final"
        save()
        return "busy"
    if stage == "qa":
        ok, out = tool(
            job,
            f"qa {name}" + (f" (round {st['round']})" if st["round"] else ""),
            [PY, "scripts/lineqa.py", name, "--reroll"],
            f"qa-{name}-{st['round']}",
        )
        if not ok:
            st["problem"] = f"lineqa failed: {last_error(out)}"
            save()
            return "blocked"
        # Lines lineqa settled (heard the same wrong way on several takes) are listed
        # but get no fresh take: they don't call for another round.
        flagged = load_json(work / "reroll.json", [])
        queued = any(not r.get("settled") for r in flagged)
        if queued and st["round"] < QA_ROUNDS:
            st["round"] += 1
            st["stage"] = "render"
        else:
            if flagged:
                (work / "reroll.json").replace(work / "reroll-left.json")
            else:
                (work / "reroll-left.json").unlink(missing_ok=True)
            # Takes lineqa put back are only mixed in by the next generate.
            st["stage"] = "remix" if "put back" in out else "final"
        save()
        return "busy"
    raise SystemExit(f"{name}: unknown stage {stage!r}")


# ---- delivery and report ----


def delivery_sha(job, ps):
    h = hashlib.sha1()
    if job.get("quicktime_audio"):
        h.update(b"quicktime-audio")
    for p in ps:
        for voice in job["voices"]:
            name = voice_part(p["name"], voice)
            h.update(
                json.dumps(
                    [load_json(WORK / name / "script.json"), load_json(WORK / name / "state.json")],
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode()
            )
    return h.hexdigest()[:12]


def deliver_job(job, ps):
    import deliver

    sha = delivery_sha(job, ps)
    done = job.get("delivered") or {}
    if done.get("state") == sha and pathlib.Path(done.get("path", "")).exists():
        return True
    dubs = {}
    for voice in job["voices"]:
        cmd = [PY, "scripts/assemble.py", job["slug"], "--voice", voice]
        if job.get("quicktime_audio"):
            cmd.append("--quicktime-audio")
        ok, out = tool(job, f"assemble {voice}", cmd, f"assemble-{voice}")
        m = re.search(r"-> (.+\.mp4)", out)
        if not ok or not m:
            return problem(job, f"assemble.py failed for the {voice} voice: {last_error(out)}")
        dubs[voice] = pathlib.Path(m.group(1).strip())
    log(f"deliver into {pathlib.Path(job['source']).name}")
    out = deliver.deliver(
        job["source"], dubs, job["default"], job_langs(job), quicktime_audio=bool(job.get("quicktime_audio"))
    )
    job["source"] = str(out)
    job["delivered"] = {"path": str(out), "at": time.strftime("%Y-%m-%d %H:%M"), "state": sha}
    job.pop("problems", None)
    save_job(job)
    return True


def report(job, ps, quiet=False):
    """work/jobs/<slug>/report.md: what to listen for, with film times."""
    md = [f"# {job['title']}", ""]
    if job.get("delivered"):
        md += [f"Delivered: `{job['delivered']['path']}` ({job['delivered']['at']})", ""]
    src, tgt = job_langs(job)
    md += [
        f"{NAMES[src]} -> {NAMES[tgt]}; voices: {', '.join(VOICE_LABEL[v] for v in job['voices'])} "
        f"(default: {VOICE_LABEL[job['default']]})",
        "",
    ]
    problems = list(job.get("problems", []))
    listen, original, flags = [], [], {v: [] for v in job["voices"]}
    timing = {v: [] for v in job["voices"]}
    for p in ps:
        script = load_json(WORK / p["name"] / "script.json")
        if script is None:
            problems.append(f"{p['name']}: not directed yet")
            continue
        if not script["lines"]:
            original.append(f"{film_time(p['start'])}-{film_time(p['end'])}")
            continue
        for line in script["lines"]:
            if line.get("listen"):
                listen.append((p["start"] + line["start"], line["text"], line["listen"]))
        for voice in job["voices"]:
            name = voice_part(p["name"], voice)
            st = load_json(WORK / name / "state.json", {})
            if st.get("problem"):
                problems.append(f"{name} ({VOICE_LABEL[voice]}): {st['problem']}")
            elif st.get("stage") != "final":
                problems.append(f"{name} ({VOICE_LABEL[voice]}): not finished ({st.get('stage', 'not rendered')})")
            for r in load_json(WORK / name / "reroll-left.json", []):
                flags[voice].append((p["start"] + r["start"], r["text"], r.get("got", "")))
            seg = load_json(WORK / name / "segments.json")
            if seg and st.get("stage") == "final":
                lines = load_json(WORK / name / "script.json")["lines"]
                for line, end in zip(lines, actor_ends(seg["job_id"], lines)):
                    wav = take_wav(seg["job_id"], line_id(line), langs(name)[1])
                    if end is not None and wav.exists():
                        timing[voice].append(end - speech_length(wav))
    if problems:
        md += ["## Needs attention", ""] + [f"- {t}" for t in problems] + [""]
    md += ["## Listen for", ""]
    if listen:
        md += (
            ["Director's notes:", ""] + [f'- {film_time(t)} "{text}": {why}' for t, text, why in sorted(listen)] + [""]
        )
    for voice in job["voices"]:
        if flags[voice]:
            md += [
                f"Still flagged by QA after a fresh take ({VOICE_LABEL[voice]}): names Whisper doesn't know "
                "and near-misses, or garbles:",
                "",
            ]
            md += [f'- {film_time(t)} "{text}" heard as "{got}"' for t, text, got in sorted(flags[voice])] + [""]
    if not listen and not any(flags.values()):
        md += ["Nothing flagged.", ""]
    md += ["## Numbers", ""]
    for voice in job["voices"]:
        d = timing[voice]
        if d:
            within = sum(abs(x) <= 0.25 for x in d) / len(d)
            med = sorted(abs(x) for x in d)[len(d) // 2]
            md.append(
                f"- {VOICE_LABEL[voice]}: {len(d)} lines; {within:.0%} end within 0.25 s of the actor "
                f"(median {med:.2f} s); {len(flags[voice])} still flagged"
            )
    if original:
        md += ["", f"Left in the original (not dubbed): {', '.join(original)}"]
    path = JOB_DIR / job["slug"] / "report.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    if not quiet:
        print("\n".join(md))
    return path


# ---- commands ----


def make_slug(text):
    words = re.findall(r"[a-z0-9]+", text.lower().split(" - ")[0])
    return "_".join(words)[:40].strip("_")


def cmd_new(args):
    src = pathlib.Path(args.video).resolve()
    if not src.exists():
        raise SystemExit(f"no such video: {src}")
    slug = args.slug or make_slug(src.stem)
    if not re.fullmatch(r"[a-z0-9_]+", slug or ""):
        raise SystemExit("give the job a --slug of lowercase letters, digits and _ (no hyphens)")
    if job_file(slug).exists():
        raise SystemExit(f"job {slug!r} exists: scripts/dub_anything.py status {slug}")
    voices = {"both": ["actor", "american"], "actor": ["actor"], "american": ["american"]}[args.voices]
    if (args.source, args.target) not in PAIRS:
        raise SystemExit(f"{args.source} -> {args.target}: the scripts know {sorted(PAIRS)}")
    unknown = [VOICE_LABEL[v] for v in voices if v not in LANGUAGES[args.target].VOICES]
    if unknown:
        raise SystemExit(
            f"a dub into {NAMES[args.target]} can't have the {' or '.join(unknown)} voice: use --voices actor"
        )
    VIDEOS.mkdir(exist_ok=True)
    link = VIDEOS / f"{slug}{src.suffix.lower()}"
    if not link.exists():
        try:
            os.link(src, link)
        except OSError:
            print(f"(can't hard-link across drives: copying the source to {link})")
            shutil.copy2(src, link)
    job = {
        "slug": slug,
        "title": src.stem,
        "source": str(src),
        "video": str(link),
        "langs": [args.source, args.target],
        "voices": voices,
        "default": voices[0],
        "split": {"minutes": args.minutes, "whole": args.whole},
        "captions": "auto" if args.auto_text else "open" if args.subtitles else args.captions,
        "signs": args.auto_text or args.signs,
        "quicktime_audio": args.quicktime_audio,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    job_file(slug).parent.mkdir(parents=True, exist_ok=True)
    save_job(job)
    print(
        f"job {slug}: {src.name}, {NAMES[args.source]} -> {NAMES[args.target]}, "
        f"{' + '.join(VOICE_LABEL[v] for v in voices)} (default {VOICE_LABEL[voices[0]]})"
    )
    print(f"next: scripts/dub_anything.py run {slug} --watch   (in the background), then direct the parts")


def cmd_ready(args):
    job = load_job(args.slug)
    names = {p["name"][-2:]: p["name"] for p in parts(job)}
    for nn in args.parts:
        name = names.get(f"{int(nn):02d}")
        script = load_json(WORK / name / "script.json") if name else None
        if not script:
            raise SystemExit(f"part {nn}: no script.json yet")
        script["ready"] = True
        write_json(WORK / name / "script.json", script)
        print(f"{name}: ready")


def cmd_status(args):
    job = load_job(args.slug)
    ps = parts(job)
    print(
        f"{job['slug']}: {pathlib.Path(job['source']).name}; {' -> '.join(job_langs(job))}; "
        f"voices {', '.join(job['voices'])}" + (f"; delivered {job['delivered']['at']}" if job.get("delivered") else "")
    )
    if not ps:
        print("  not split yet")
    for p in ps:
        name = p["name"]
        script = load_json(WORK / name / "script.json")
        prep = "prepped" if (WORK / name / "segments.json").exists() else "-"
        sep = "separated" if job_dir_of(name) and (job_dir_of(name) / "separator.json").exists() else "-"
        draft = "drafted" if (WORK / name / "draft.json").exists() else "-"
        if script is None:
            direction = "to direct"
        elif not script["lines"]:
            direction = "left original"
        else:
            direction = "ready" if script.get("ready") else "directing"
        cols = []
        for voice in job["voices"]:
            st = load_json(WORK / voice_part(name, voice) / "state.json", {})
            left = len(load_json(WORK / voice_part(name, voice) / "reroll-left.json", []))
            s = ("BLOCKED" if st.get("problem") else st.get("stage", "-")) + (
                f" r{st['round']}" if st.get("round") else ""
            )
            cols.append(f"{voice}: {s}" + (f" ({left} flagged)" if left else ""))
        print(f"  {name}  {film_time(p['start'])}  {prep:8s} {sep:9s} {draft:7s} {direction:13s} " + "  ".join(cols))
        for voice in job["voices"]:
            st = load_json(WORK / voice_part(name, voice) / "state.json", {})
            if st.get("problem"):
                print(f"      {voice}: {st['problem']}")
    for t in job.get("problems", []):
        print(f"  problem: {t}")


def cmd_run(args):
    job = load_job(args.slug)
    if not backend.ensure():
        raise SystemExit("the VoiceStudio backend won't start (scripts/backend.py status)")
    if not split(job):
        return problem(job, "split.py failed") or 2
    ps = parts(job)
    job.pop("problems", None)
    save_job(job)
    # A later failure on a previously blocked part is new, including the same
    # failure after the director has changed its script.
    known = set()
    for p in ps:  # a new run retries what failed for reasons outside the script
        for v in job["voices"]:
            path = WORK / voice_part(p["name"], v) / "state.json"
            st = load_json(path, {})
            if st.get("problem", "").startswith(RETRY):
                log(f'{voice_part(p["name"], v)}: retrying after "{st.pop("problem")[:80]}"')
                st["fails"] = 0
                write_json(path, st)
            if st.get("problem"):
                known.add((voice_part(p["name"], v), st.get("script"), st["problem"]))
    seen, new_block = set(), False
    while True:
        pause_checkpoint()
        second(job, ps)
        busy = blocked = False
        for p in ps:
            if not ready(p) or not (original_only(p) or drafted(p)):
                continue
            for voice in job["voices"]:
                name = variant(job, p, voice)
                if name is None:
                    blocked = True
                    continue
                while (state := advance(job, name)) == "busy":
                    busy = True
                if state == "blocked":
                    blocked = True
                    st = load_json(WORK / name / "state.json", {})
                    block = (name, st.get("script"), st.get("problem"))
                    if block not in seen:
                        seen.add(block)
                        log(f"BLOCKED {name}: {st.get('problem')}")
                        new_block |= block not in known
        if not busy:
            # Preparation fills the time no ready part needs: renders never wait for
            # the rest of the video to be prepped.
            step = prepare_step(job, ps)
            if step == "failed":
                return 2
            busy = step == "busy"
        final = all(ready(p) and all(finished(p, v) for v in job["voices"]) for p in ps)
        if final:
            if SECOND_RUN.get(job["slug"]):
                SECOND_RUN[job["slug"]][0].wait()
            if not deliver_job(job, ps):
                return 2
            path = report(job, ps, quiet=True)
            log(f"delivered {job['delivered']['path']}; report {path}")
            return 0
        if (new_block and args.watch) or (blocked and not busy and not args.watch):
            log("waiting on the director: see scripts/dub_anything.py status " + job["slug"])
            report(job, ps, quiet=True)
            return 2
        if not args.watch and not busy:
            left = [p["name"] for p in ps if not ready(p)]
            log(f"nothing more to do now; scripts not ready: {', '.join(left) or 'none'}")
            return 0
        if not busy:
            time.sleep(30)
            job = load_job(job["slug"])


def cmd_report(args):
    job = load_job(args.slug)
    report(job, parts(job))


def cmd_cleanup(args):
    """Move what a finished job no longer needs into one folder the user deletes: the
    VoiceStudio jobs, split parts and renders, the pre-dub source link, per-line
    references, kept takes and logs. Scripts, states, notes and the report stay."""
    job = load_job(args.slug)
    if not job.get("delivered") and not args.force:
        raise SystemExit("not delivered yet (--force to clean up anyway)")
    hold = VIDEOS / "_to-delete" / job["slug"]
    moved = 0

    def mv(src, dst):
        nonlocal moved
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
            moved += 1

    for p in parts(job):
        for voice in ("actor", "american"):
            name = voice_part(p["name"], voice)
            j = job_dir_of(name)
            if j and j.exists():
                mv(j, hold / "dub_jobs" / j.name)
            for sub in ("refs", "emo", "takes"):
                mv(WORK / name / sub, hold / "work" / name / sub)
    mv(VIDEOS / f"{job['slug']}-parts", hold / "videos" / f"{job['slug']}-parts")
    mv(pathlib.Path(job["video"]), hold / "videos" / pathlib.Path(job["video"]).name)
    mv(JOB_DIR / job["slug"] / "logs", hold / "logs")
    job["cleaned"] = {"to": str(hold), "at": time.strftime("%Y-%m-%d %H:%M")}
    save_job(job)
    print(f"moved {moved} items to {hold}; delete that folder when you're happy with the delivery")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("new")
    a.add_argument("video")
    a.add_argument("--source", default="ja", help="language the video speaks (default ja)")
    a.add_argument("--target", default="en", help="language to dub it into (default en)")
    a.add_argument("--voices", choices=["actor", "american", "both"], default="actor")
    a.add_argument(
        "--quicktime-audio",
        action="store_true",
        help="tag English original audio as und to prefer the dub in QuickTime",
    )
    a.add_argument("--slug")
    a.add_argument("--minutes", type=float, default=3)
    a.add_argument("--whole", action="append", default=[], metavar="A-B")
    a.add_argument("--captions", action="store_true", help="the picture carries captions: run subs.py")
    a.add_argument("--auto-text", action="store_true", help="detect caption/subtitle styles and signs automatically")
    a.add_argument(
        "--subtitles",
        action="store_true",
        help="the film subtitles its foreign-language speech in the picture: run subs.py --open",
    )
    a.add_argument(
        "--signs",
        action="store_true",
        help="text in the picture worth translating (signs, diagrams, title cards): run signs.py",
    )
    a.set_defaults(fn=cmd_new)
    for name, fn in (("run", cmd_run), ("status", cmd_status), ("report", cmd_report), ("cleanup", cmd_cleanup)):
        b = sub.add_parser(name)
        b.add_argument("slug")
        b.set_defaults(fn=fn)
        if name == "run":
            b.add_argument("--watch", action="store_true", help="keep waiting for scripts until delivered")
        if name == "cleanup":
            b.add_argument("--force", action="store_true")
    c = sub.add_parser("ready")
    c.add_argument("slug")
    c.add_argument("parts", nargs="+")
    c.set_defaults(fn=cmd_ready)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
