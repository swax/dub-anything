"""Read-only job progress from saved pipeline files and its current tool log."""

import json
import re

from core import JOBS, voice_part
from integrations import director

STEPS = {
    "prep": "Preparing audio",
    "separate": "Separating audio",
    "subs": "Reading captions",
    "draft": "Drafting dialogue",
    "signs": "Reading on-screen text",
    "direction": "Directing dialogue",
    "render": "Rendering and checking voices",
    "qa": "Checking rendered dialogue",
    "assemble": "Assembling audio tracks",
    "deliver": "Assembling the dubbed video",
    "split": "Splitting the video",
}


def read(path):
    # Files may be between writes while the pipeline is running. A dashboard
    # refresh must remain available during that interval.
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            return {}
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def tail(path):
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 64000))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def current_tool(log, names):
    current = None
    # Ignore earlier runs when a resumed worker has just started a new command.
    for line in log.rsplit("\n--- ", 1)[-1].splitlines():
        match = re.match(r"^\[\d{2}:\d{2}:\d{2}\] (.+?) \.\.\.$", line)
        if match:
            label = match[1]
            first = label.split()[0]
            step = {"round": "render", "re-mix": "render"}.get(first, first)
            if step not in STEPS:
                continue
            indices = [
                index
                for index, name in enumerate(names, 1)
                if re.search(r"(?<![\w-])" + re.escape(name) + r"(?:\.us)?(?![\w-])", label)
            ]
            batch = re.search(r"\.\.(\d+)(?:\s|$)", label)
            if indices and batch:
                end_name = names[indices[0] - 1].rsplit("-", 1)[0] + "-" + batch[1]
                if end_name in names and names.index(end_name) + 1 >= indices[0]:
                    indices = list(range(indices[0], names.index(end_name) + 2))
            current = {"step": step, "label": STEPS[step], "parts": indices, "status": "running", "raw": label}
            if step in ("render", "qa"):
                current["label"] += " · " + ("American English" if ".us" in label else "Actor’s voice")
        elif current:
            result = re.match(r"^\[\d{2}:\d{2}:\d{2}\] " + re.escape(current["raw"]) + r" (ok|FAILED) \(", line)
            if result:
                current["status"] = "completed" if result[1] == "ok" else "failed"
    if current:
        current.pop("raw")
    return current


def snapshot(job, data, root, jobs=JOBS):
    slug = job.get("slug", "")
    if not isinstance(slug, str) or not re.fullmatch(r"[\w-]+", slug):
        return {"total": None, "current": None, "stages": []}
    manifest = read(root / "videos" / f"{slug}-parts" / "parts.json")
    parts = manifest.get("parts", [])
    if not isinstance(parts, list) or any(
        not isinstance(part, dict) or not isinstance(part.get("name"), str) or not re.fullmatch(r"[\w-]+", part["name"])
        for part in parts
    ):
        parts = []
    names = [part["name"] for part in parts]
    if len(set(names)) != len(names):
        names = []
    pipeline = read(root / "work/jobs" / slug / "job.json")
    current = current_tool(tail(data / "jobs" / job["id"] / "pipeline.log"), names)
    counts = dict.fromkeys(("prep", "separate", "subs", "draft", "signs", "direction", "render"), 0)
    pending = {key: [] for key in counts}
    # Import pure timing-state helpers only; no transcription model is loaded.
    from dub_anything import derived_script, script_sha

    for index, name in enumerate(names, 1):
        folder = root / "work" / name
        script, draft, segments = (read(folder / f"{file}.json") for file in ("script", "draft", "segments"))
        approved = job.get("approved_parts", {}).get(name) == director.script_signature(script)
        original = script.get("lines") == [] and (not draft or approved)
        ready = original or (bool(script.get("ready")) and approved)
        job_id = segments.get("job_id", "")
        valid_job = isinstance(job_id, str) and bool(re.fullmatch(r"[\w-]+", job_id))
        separated = valid_job and bool(read(jobs / job_id / "separator.json"))
        voices = ["actor", "american"] if job.get("voices") == "both" else [job.get("voices", "actor")]
        final = original
        if not original and isinstance(script.get("lines"), list) and ready:
            final = all(
                (state := read(root / "work" / voice_part(name, voice) / "state.json")).get("stage") == "final"
                and state.get("script") == script_sha(derived_script(script, voice))
                for voice in voices
            )
        flags = {
            "prep": original or valid_job,
            "separate": original or separated,
            "subs": original or bool(read(folder / "subs.json")),
            "draft": original or isinstance(draft.get("lines"), list),
            "signs": original or bool(read(folder / "signs.json")),
            "direction": ready,
            "render": final,
        }
        for key, done in flags.items():
            counts[key] += bool(done)
            if not done:
                pending[key].append(index)
    if not names:
        return {"total": None, "current": current, "stages": []}
    work_stage = job.get("work_stage", job.get("state"))
    description = job.get("work_description", job.get("stage", ""))
    if work_stage == "directing":
        match = re.search(r"part (\d+) of", description)
        current = {
            "step": "direction",
            "label": STEPS["direction"],
            "parts": [int(match[1])] if match else [],
            "status": "running",
        }
    if current and len(current["parts"]) > 1 and current["step"] in ("draft", "separate"):
        outstanding = [index for index in current["parts"] if index in pending[current["step"]]]
        if outstanding:
            current["batch"] = current["parts"]
            current["parts"] = outstanding[:1]
    keys = ["prep", "separate"] + (["subs"] if pipeline.get("captions", job.get("auto_text")) else []) + ["draft"]
    keys += (["signs"] if pipeline.get("signs", job.get("signs")) else []) + ["direction", "render"]
    stages = [{"key": key, "label": STEPS[key], "done": counts[key], "left": len(names) - counts[key]} for key in keys]
    phase = (
        "direction"
        if work_stage == "directing"
        else "render"
        if work_stage in ("rendering", "publishing", "awaiting_review", "completed")
        else "draft"
    )
    if (
        work_stage not in ("directing", "rendering", "publishing", "awaiting_review", "completed")
        and current
        and {"qa": "render", "assemble": "render", "deliver": "render"}.get(current["step"], current["step"]) in keys
    ):
        phase = {"qa": "render", "assemble": "render", "deliver": "render"}.get(current["step"], current["step"])
    if current:
        current["active"] = (
            job.get("state") in ("preparing", "rendering", "directing") and current["status"] == "running"
        )
    return {
        "total": len(names),
        "finished": counts["render"],
        "left": len(names) - counts["render"],
        "current": current,
        "phase": next(row for row in stages if row["key"] == phase),
        "stages": stages,
    }
