"""Resolve language requests and stage agent-added support before using it in production."""

import hashlib
import importlib
import json
import math
import pathlib
import re
import shutil

import backend
from integrations import director

ROOT = pathlib.Path(__file__).resolve().parents[2]


def catalog():
    import languages

    importlib.invalidate_caches()
    return importlib.reload(languages)


def language(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 80:
        raise ValueError("enter a language name or code")
    value = value.strip()
    if not all(char.isalpha() or char.isspace() or char in "-()" for char in value):
        raise ValueError("use a language name or code, such as French or fr")
    for code, module in catalog().LANGUAGES.items():
        if value.casefold() in {code, module.NAME.casefold(), module.NATIVE.casefold(), module.ISO3.casefold()}:
            return code
    return value


def supported(source, target):
    return (source, target) in catalog().PAIRS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def allowed(relative):
    path = pathlib.PurePosixPath(relative)
    return (
        (
            path.parent == pathlib.PurePosixPath("scripts/languages")
            and re.fullmatch(r"(?:[a-z]{2,3}|__init__)\.py", path.name)
        )
        or (
            path.parent == pathlib.PurePosixPath("docs/languages")
            and re.fullmatch(r"(?:[a-z]{2,3}|README)\.md", path.name)
        )
        or (path.parent == pathlib.PurePosixPath("tests") and re.fullmatch(r"test_language_[a-z_]+\.py", path.name))
    )


def workspace(folder, root=ROOT):
    target = folder / "language-workspace"
    baseline = folder / "language-baseline.json"
    if baseline.exists():
        return target, director.read_json(baseline)
    target.mkdir(exist_ok=True)
    for name in ("scripts", "docs", "tests", "eval"):
        shutil.copytree(
            root / name, target / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("AGENTS.md", "CLAUDE.md", "ruff.toml", "Directory.Build.props"):
        shutil.copy2(root / name, target / name)
    hashes = {
        path.relative_to(root).as_posix(): digest(path)
        for name in ("scripts", "docs", "tests")
        for path in (root / name).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    baseline.write_text(json.dumps(hashes), encoding="utf-8")
    return target, hashes


def apply_support(work, hashes, source, target, root=ROOT):
    """Publish only language files; detect concurrent edits and never import other agent changes."""
    changes = []
    for name in ("scripts", "docs", "tests"):
        for path in (work / name).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(work).as_posix()
            if digest(path) == hashes.get(relative):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(work.resolve()):
                raise ValueError("language setup cannot publish links outside its workspace")
            if not allowed(relative):
                raise ValueError(f"language setup changed {relative}; this needs a normal code review")
            if relative.startswith("scripts/languages/") and path.name != "__init__.py" and hashes.get(relative):
                raise ValueError(f"language setup changed existing module {relative}; this needs a normal code review")
            if relative.startswith("tests/") and hashes.get(relative):
                raise ValueError("language setup can add tests but cannot change existing ones")
            destination = root / relative
            if digest(destination) != hashes.get(relative):
                raise ValueError(f"{relative} changed while the agent worked; refusing to overwrite it")
            changes.append((path, destination))
    for code in (source, target):
        if not (work / "scripts" / "languages" / f"{code}.py").is_file():
            raise ValueError(f"the agent did not supply a module for {code}")
        if not (work / "docs" / "languages" / f"{code}.md").is_file():
            raise ValueError(f"the agent did not supply language notes for {code}")
    # Check all paths before making any changes. No source-media/voice/test
    # artifacts from the scratch workspace can enter the checkout.
    for path, destination in changes:
        shutil.copy2(path, destination)
    return [destination.relative_to(root).as_posix() for _, destination in changes]


def setup_prompt(job, staged):
    return f"""Add missing language support for Dub Anything in this scratch workspace.
The administrator requested source {json.dumps(job["source_lang"])}, target
{json.dumps(job["target_lang"])}. Read AGENTS.md and docs/languages/README.md.
Resolve those names to the correct language codes. Follow the Adding a language
workflow, including a module, language notes, meaningful QA normalization tests,
and a short actual dub measured with and without per-line references. Do not
claim that every language is supported or add an untested PAIRS entry.

The read-only source for the test is {json.dumps(str(staged))}.
Python: {pathlib.Path(backend.vs_python()).as_posix()}
VoiceStudio is configured through VOICESTUDIO_DIR; models and GPU are available.
Use ffmpeg to extract a short sample with at least six complete spoken lines
into this workspace's videos/. Keep all transcripts, cast, audio, measurements
and source-specific findings in this workspace's ignored work/ or videos/.
Do not read work/NOTES.md in the main checkout. Do not change the original video.

You may edit only scripts/languages/<code>.py, scripts/languages/__init__.py,
docs/languages/<code>.md, docs/languages/README.md, and new
tests/test_language_<description>.py. Do not change generic tools, existing
languages' behavior, or existing pair settings. If new engine/tool support is
needed, report that blocker rather than bypassing it. No commits or branches.
Use only standard-library top-level imports in language modules. Run the tests.

Create two independent rendered test clips with the same corrected script and
at least six spoken lines. Give each a separate VoiceStudio job. Generate one
with --line-ref and one with --no-line-ref. Run lineqa.py, eval/score.py and
eval/timing.py, inspect the measurements, and choose the pair's line_ref default.
Never invent measurements. The supervisor repeats the checks independently.
Put production-specific results in work/language-report.md, with limitations
and listening notes. Use generic examples in tracked docs and tests.

Write language-result.json at this workspace root:
{{"source":"<code>","target":"<code>","on_clip":"<clip with refs>",
"off_clip":"<clip without refs>"}}.
If it cannot be supported or tested, write {{"error":"specific blocker"}}.
Do not ask questions or process the full video. End after this short test.
"""


def ensure(worker, job, staged):
    if supported(job["source_lang"], job["target_lang"]):
        return job
    worker.set_stage(job, "language_setup", "Director is adding and testing the requested language pair")
    folder = worker.folder(job)
    work, hashes = workspace(folder)
    previous_pairs = dict(catalog().PAIRS)
    worker.run(job, [backend.vs_python(), "scripts/backend.py", "start"], "language-backend")
    worker.run(
        job,
        director.command(job["director"], job["model"], workspace=work, setup=True),
        "language-setup",
        setup_prompt(job, staged),
        timeout=2 * 3600,
        cwd=work,
    )
    result = director.read_json(work / "language-result.json", {})
    if result.get("error"):
        raise ValueError("Language setup: " + str(result["error"]))
    source, target = result.get("source", ""), result.get("target", "")
    if not all(isinstance(code, str) and re.fullmatch(r"[a-z]{2,3}", code) for code in (source, target)):
        raise ValueError("the director did not resolve the requested language codes")
    for requested, resolved in ((job["source_lang"], source), (job["target_lang"], target)):
        if re.fullmatch(r"[a-z]{2,3}", requested) and requested != resolved:
            raise ValueError("language setup changed an explicitly requested language code")
    clips = [result.get("on_clip", ""), result.get("off_clip", "")]
    if len(set(clips)) != 2 or not all(isinstance(clip, str) and re.fullmatch(r"[a-z0-9_-]+", clip) for clip in clips):
        raise ValueError("language setup must provide two independent test clips")
    for clip, refs in zip(clips, (True, False)):
        script = director.read_json(work / "work" / clip / "script.json", {})
        if len(script.get("lines", [])) < 6 or script.get("render", {}).get("line_ref") is not refs:
            raise ValueError("language setup needs six rendered lines in both reference modes")
    first, second = [director.read_json(work / "work" / clip / "script.json")["lines"] for clip in clips]
    if [{k: line.get(k) for k in ("start", "end", "voice", "text", source)} for line in first] != [
        {k: line.get(k) for k in ("start", "end", "voice", "text", source)} for line in second
    ]:
        raise ValueError("language test variants must use the same corrected script")
    python = backend.vs_python()
    worker.run(job, [python, "-m", "unittest", "discover", "-s", "tests"], "language-tests", cwd=work)
    qa = worker.run(job, [python, "scripts/lineqa.py", *clips, "--all"], "language-qa", cwd=work)
    counts = re.findall(r"(\d+) of (\d+) lines flagged", qa)
    if len(counts) != 2 or any(int(flagged) or int(total) < 6 for flagged, total in counts) or "(not rendered)" in qa:
        raise ValueError("the new language pair did not pass per-line transcription checks")
    timing = worker.run(job, [python, "eval/timing.py", *clips], "language-timing", cwd=work)
    if len(re.findall(r"\|end\| mean [0-9.]+ median [0-9.]+", timing)) != 2:
        raise ValueError("the new language pair lacks timing measurements")
    score = worker.run(
        job, [python, "eval/score.py", f"line_ref={clips[0]}", f"no_line_ref={clips[1]}"], "language-score", cwd=work
    )
    for variant in ("line_ref", "no_line_ref"):
        row = re.search(rf"^{variant}\s+(.+)$", score, re.MULTILINE)
        if not row:
            raise ValueError("the new language pair lacks reference-mode comparison measurements")
        fields = row.group(1).split()
        try:
            likeness, timing_error, probability = (float(fields[index]) for index in (3, 6, 10))
            if not all(math.isfinite(value) for value in (likeness, timing_error, probability)) or probability <= 0:
                raise ValueError()
        except (IndexError, ValueError) as exc:
            raise ValueError("the new language pair lacks finite voice/timing/target-language scores") from exc
    # Import in a fresh interpreter, since the running service's registry predates
    # these modules. A bad module or a missing PAIRS entry fails before publication.
    output = worker.run(
        job,
        [
            python,
            "-c",
            "import sys,json; sys.path.insert(0, 'scripts'); from languages import PAIRS; "
            "print(json.dumps([[a,b,options] for (a,b),options in PAIRS.items()]))",
        ],
        "language-registry",
        cwd=work,
    )
    pairs = {(a, b): options for a, b, options in json.loads(output)}
    if (source, target) not in pairs or any(pairs.get(pair) != options for pair, options in previous_pairs.items()):
        raise ValueError("language setup did not add the pair or changed an existing pair's settings")
    changed = apply_support(work, hashes, source, target)
    return worker.store.update(job["id"], source_lang=source, target_lang=target, language_support_added=changed)
