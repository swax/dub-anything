"""One GPU job at a time.

The tools share one card with the VoiceStudio backend: a render beside another
model crawls or fails with "Generation failed". Every tool calls hold_gpu() just
before it loads a model (or asks the backend to run one), waits there while
another tool holds the GPU, and keeps it until it exits. Child processes it
starts (dub.py's voice conversion) share its hold. Waiters take turns in the
order they arrived (a ticket each in work/gpu.queue), so a tool started while
dub_anything.py goes from one job to the next gets the card after the current job,
not after the whole batch. Whisper on the CPU (CTranslate2 has no GPU backend
on AMD or Apple) doesn't wait (whisper_safe.hold_gpu_for_whisper). Standard
library only, so tools in other environments (separate.py) can import it too.
"""

import json
import os
import pathlib
import sys
import time

LOCK = pathlib.Path(__file__).resolve().parents[1] / "work" / "gpu.lock"
OWNER = LOCK.with_suffix(".owner")
QUEUE = LOCK.with_suffix(".queue")
_held = None


def _try_lock(f):
    if sys.platform == "win32":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(f):
    if sys.platform == "win32":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_UN)


def _waiting(ticket):
    """A waiter keeps its ticket locked while it waits; one nobody holds was left by a
    process that died, and is removed."""
    try:
        with open(ticket, "a+") as f:
            _try_lock(f)
            _unlock(f)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        ticket.unlink()
    except OSError:  # its owner has just opened it (Windows won't delete an open file)
        return True
    return False


def owner():
    try:
        o = json.loads(OWNER.read_text(encoding="utf-8"))
        return f"{o['what']}, pid {o['pid']}, since {o['since']}"
    except (OSError, ValueError, KeyError):
        return "another tool"


def hold_gpu(what=None):
    """Wait until no other tool holds the GPU, then hold it until this process exits."""
    global _held
    if _held is not None or os.environ.get("DUB_ANYTHING_GPU") == "held":
        return
    QUEUE.mkdir(parents=True, exist_ok=True)
    ticket = QUEUE / f"{time.time_ns():020d}-{os.getpid()}"
    t = open(ticket, "a+")
    _try_lock(t)
    f = open(LOCK, "a+")
    waited = time.time()
    try:
        while True:
            ahead = [q for q in QUEUE.iterdir() if q.name < ticket.name and _waiting(q)]
            if not ahead:
                try:
                    _try_lock(f)
                    break
                except OSError:
                    pass
            if waited:
                print(f"  waiting for the GPU (held by {owner()})", flush=True)
                waited = 0
            time.sleep(1)
    finally:
        _unlock(t)
        t.close()
        try:
            ticket.unlink(missing_ok=True)
        except OSError:  # another waiter is checking it (Windows won't delete an open file); it removes it
            pass
    _held = f
    os.environ["DUB_ANYTHING_GPU"] = "held"
    what = what or pathlib.Path(sys.argv[0]).name
    OWNER.write_text(
        json.dumps({"what": " ".join([what] + sys.argv[1:2]), "pid": os.getpid(), "since": time.strftime("%H:%M:%S")}),
        encoding="utf-8",
    )
