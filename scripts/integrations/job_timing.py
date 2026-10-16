"""Active job clocks and deliberately conservative estimates from completed work."""

import datetime
import json
import re
import statistics
import time
from collections import defaultdict

from integrations import progress

ACTIVE = ("staging", "language_setup", "preparing", "directing", "rendering", "publishing")


def transition(job, old_state, new_state, now, recovery=False):
    """Accumulate only active intervals; recovery stops at the last worker pulse."""
    since = job.get("active_since")
    if since is not None and new_state not in ACTIVE:
        end = min(now, job.get("last_active_at", since)) if recovery else now
        job["active_seconds"] = job.get("active_seconds", 0) + max(0, end - since)
        job["active_since"] = None
    if new_state in ACTIVE:
        if old_state not in ACTIVE or since is None:
            job["active_since"] = now
        job["last_active_at"] = now


def elapsed(job, now):
    seconds = max(0, job.get("active_seconds", 0))
    if job.get("state") in ACTIVE and job.get("active_since") is not None:
        seconds += max(0, now - job["active_since"])
    return seconds


def legacy_elapsed(folder, now):
    """Recover observed run intervals, excluding gaps between resumed commands.

    Session headers and timestamped output are local time, just like the writer.
    Old logs cannot prove the exact stop time, so this is always approximate.
    """
    intervals = []
    for path in folder.glob("*.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            modified = min(now, path.stat().st_mtime)
        except OSError:
            continue
        start = end = None
        for line in text.splitlines():
            header = re.fullmatch(r"--- (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ---", line)
            if header:
                if start is not None:
                    intervals.append((start, end))
                try:
                    start = end = datetime.datetime.strptime(header[1], "%Y-%m-%d %H:%M:%S").timestamp()
                except ValueError:
                    start = end = None
            elif start is not None and (stamp := re.match(r"^\[(\d\d:\d\d:\d\d)\]", line)):
                try:
                    clock = datetime.datetime.strptime(stamp[1], "%H:%M:%S").time()
                    date = datetime.datetime.fromtimestamp(end).date()
                    candidate = datetime.datetime.combine(date, clock).timestamp()
                    if candidate < end - 12 * 3600:
                        candidate += 24 * 3600
                    end = max(end, candidate)
                except ValueError:
                    pass
        if start is not None:
            intervals.append((start, max(end, modified)))
    total, previous_end = 0, 0
    for start, end in sorted(intervals):
        end = min(now, end)
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def samples(folder, view):
    """Seconds per newly completed part, with cached batch work counted once."""
    rates, seen, render = defaultdict(list), defaultdict(set), defaultdict(float)
    names = view.get("names", [])
    try:
        # Bounded history: timing is optional and must not break job polling.
        with (folder / "pipeline.log").open("rb") as handle:
            handle.seek(max(0, (folder / "pipeline.log").stat().st_size - 4 * 1024 * 1024))
            log = handle.read().decode("utf-8", errors="replace")
    except OSError:
        log = ""
    for line in log.splitlines():
        match = re.match(r"^\[\d\d:\d\d:\d\d\] (.+?) ok \((\d+(?:\.\d+)?) s\)$", line)
        if not match or not (tool := progress.describe_tool(match[1], names)) or not tool["parts"]:
            continue
        step, seconds = tool["step"], float(match[2])
        voice = "american" if ".us" in match[1] else "actor"
        if step in ("render", "qa") or (step == "prep" and voice == "american"):
            for index in tool["parts"]:
                render[(voice, index)] += seconds / len(tool["parts"])
        else:
            fresh = set(tool["parts"]) - seen[step]
            if fresh and seconds > 0:
                rates[step].extend([seconds / len(fresh)] * len(fresh))
            seen[step].update(tool["parts"])
    for (voice, index), seconds in render.items():
        if index in view.get("voices", {}).get(voice, {}).get("finished", []) and seconds > 0:
            rates["render_" + voice].append(seconds)
    try:
        for line in (folder / "timings.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if record.get("kind") == "direction" and float(record.get("seconds", 0)) > 0:
                    rates["direction"].append(float(record["seconds"]))
            except (ValueError, TypeError, AttributeError):
                continue
    except OSError:
        pass
    return {key: statistics.median(values[-12:]) for key, values in rates.items() if values}


def snapshot(job, data, view, now=None):
    now = time.time() if now is None else now
    running = job.get("state") in ACTIVE
    folder = data / "jobs" / job["id"]
    legacy = "active_seconds" not in job
    result = {
        "elapsed_seconds": legacy_elapsed(folder, now) if legacy else elapsed(job, now),
        "running": running,
        "approximate": legacy or bool(job.get("timing_approximate")),
        "remaining_seconds": None,
        "stage_seconds": None,
        "stage_label": view.get("phase", {}).get("label"),
        "missing_stages": [],
        "status": "learning",
    }
    if job.get("state") in ("completed", "awaiting_review"):
        result["status"] = "complete" if job["state"] == "completed" else "review"
        return result
    if not view.get("total"):
        return result
    rates = samples(folder, view)
    estimates = {}
    for stage in view["stages"]:
        key = stage["key"]
        if not stage["left"]:
            estimates[key] = 0
        elif key == "render":
            voices = view.get("voices", {})
            if voices and all(not row["left"] or "render_" + voice in rates for voice, row in voices.items()):
                estimates[key] = sum(row["left"] * rates.get("render_" + voice, 0) for voice, row in voices.items())
        elif key in rates:
            estimates[key] = stage["left"] * rates[key]
        if key not in estimates:
            result["missing_stages"].append(stage["label"])
    phase = view["phase"]
    if phase["left"]:
        result["stage_seconds"] = estimates.get(phase["key"])
    if not result["missing_stages"]:
        if view["left"]:
            result.update(remaining_seconds=sum(estimates.values()), status="estimated")
        else:
            result["status"] = "finishing"
    return result
