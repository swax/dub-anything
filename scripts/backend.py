"""Keep the VoiceStudio backend running.

Usage: scripts/backend.py status | start | stop | restart

start launches the backend detached from the shell that asks (on Windows through
the process service, so it outlives an agent's session), with OMNIVOICE_DATA_DIR,
OMNIVOICE_CACHE_DIR and the full PATH set, and waits until /health says ok. The
log goes to backend.log beside the data folder (or $VOICESTUDIO_LOG). The
checkout is $VOICESTUDIO_DIR, else (macOS) the desktop app's runtime, else
VoiceStudio beside this repo. dub_anything.py calls
ensure() before GPU work and restarts a backend that stopped answering.
"""

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

from gpu import ROCM_ENV, rocm_env

ROOT = pathlib.Path(__file__).resolve().parents[1]
API = os.environ.get("VOICESTUDIO_API", "http://127.0.0.1:3900")
PORT = int(re.search(r":(\d+)", API.split("//", 1)[-1]).group(1)) if re.search(r":\d+", API.split("//", 1)[-1]) else 80


def find_vs_dir():
    if os.environ.get("VOICESTUDIO_DIR"):
        return pathlib.Path(os.environ["VOICESTUDIO_DIR"])
    candidates = []
    if sys.platform == "darwin":
        candidates.append(
            pathlib.Path.home() / "Library" / "Application Support" / "VoiceStudio" / "runtime" / "project"
        )
        candidates.append(pathlib.Path("/Applications/VoiceStudio.app/Contents/Resources"))
    candidates.append(ROOT.parent / "VoiceStudio")
    for c in candidates:
        if (c / "backend").is_dir():
            return c
    return ROOT.parent / "VoiceStudio"


VS_DIR = find_vs_dir()


def venv_python():
    """VoiceStudio's own virtualenv Python, or None without one."""
    py = VS_DIR / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return str(py) if py.is_file() else None


def vs_python():
    """The Python that runs the scripts: VoiceStudio's, else this one."""
    return venv_python() or sys.executable


def user_env(name):
    """A variable as the user set it, also when this shell started before it was set."""
    if os.environ.get(name):
        return os.environ[name]
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                return os.path.expandvars(winreg.QueryValueEx(k, name)[0])
        except OSError:
            return ""
    return ""


def full_env():
    """This process's environment plus the user's Dub Anything variables and the machine and
    user PATH (an agent's shell may predate uv or ffmpeg being installed)."""
    env = dict(os.environ)
    for name in ("OMNIVOICE_DATA_DIR", "OMNIVOICE_CACHE_DIR"):
        if user_env(name):
            env[name] = user_env(name)
    rocm_env(env)
    if sys.platform == "win32":
        import winreg

        paths = []
        for root, key in (
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, "Environment"),
        ):
            try:
                with winreg.OpenKey(root, key) as k:
                    paths.append(os.path.expandvars(winreg.QueryValueEx(k, "Path")[0]))
            except OSError:
                pass
        seen, parts = set(), []
        for p in (env.get("PATH", "") + ";" + ";".join(paths)).split(";"):
            if p and p.lower() not in seen:
                seen.add(p.lower())
                parts.append(p)
        env["PATH"] = ";".join(parts)
    else:
        # On macOS and Linux, ensure Homebrew and standard user bin dirs are on PATH
        extra_paths = [
            "/opt/homebrew/bin",
            "/opt/homebrew/sbin",
            "/usr/local/bin",
            str(pathlib.Path.home() / ".local" / "bin"),
            str(pathlib.Path.home() / ".cargo" / "bin"),
        ]
        curr = env.get("PATH", "").split(":")
        for p in extra_paths:
            if p not in curr and os.path.isdir(p):
                curr.append(p)
        env["PATH"] = ":".join(curr)
    return env


def log_path(env=None):
    env = full_env() if env is None else env
    if os.environ.get("VOICESTUDIO_LOG"):
        return pathlib.Path(os.environ["VOICESTUDIO_LOG"])
    data = env.get("OMNIVOICE_DATA_DIR")
    if data:
        return pathlib.Path(data).parent / "backend.log"
    if sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Application Support" / "OmniVoice" / "backend.log"
    return VS_DIR / "backend.log"


def health(timeout=5):
    try:
        import requests
    except ImportError:  # a bare python3 (macOS): ask with the standard library
        import json
        import urllib.request

        try:
            with urllib.request.urlopen(f"{API}/health", timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
    try:
        return requests.get(f"{API}/health", timeout=timeout).json()
    except (requests.RequestException, ValueError):
        return None


def healthy():
    h = health()
    return bool(h) and h.get("status") == "ok"


def listening_pid():
    if sys.platform == "win32":
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            f = line.split()
            if len(f) >= 5 and f[3].upper() == "LISTENING" and f[1].endswith(f":{PORT}"):
                return int(f[4])
        return None
    out = subprocess.run(["lsof", "-ti", f"tcp:{PORT}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
    return int(out.split()[0]) if out.split() else None


def stop():
    pid = listening_pid()
    if not pid:
        print("backend: not running")
        return
    if sys.platform == "win32":
        # The listener is uv's Python; its parent (uv / the venv launcher) goes too.
        parent = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').ParentProcessId",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        for p in [pid] + ([int(parent)] if parent.isdigit() else []):
            subprocess.run(["taskkill", "/PID", str(p), "/T", "/F"], capture_output=True)
    else:
        # A GUI or another launcher may share its process group with the backend.
        # Port discovery does not establish ownership of that group: signal only
        # the listener, including when escalating below.
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
    for _ in range(30):
        if listening_pid() != pid:
            break
        time.sleep(1)
    if sys.platform != "win32" and listening_pid() == pid:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    print(f"backend: stopped (pid {pid})")


def launch_cmd(env):
    """The command that runs the backend. Windows starts it through uv, as it
    always has. Elsewhere VoiceStudio's own virtualenv runs it when there is one
    (the desktop app's runtime). Which Python runs this script, or how its path
    was typed, doesn't change that."""
    host = re.sub(r"^\w+://|:\d+.*$", "", API)
    app = ["uvicorn", "main:app", "--app-dir", "backend", "--host", host, "--port", str(PORT)]
    venv = venv_python()
    if venv and sys.platform != "win32":
        return [venv, "-m"] + app
    uv = shutil.which("uv", path=env["PATH"])
    if not uv and sys.platform == "darwin":
        bundled_uv = pathlib.Path("/Applications/VoiceStudio.app/Contents/Resources/tools/uv")
        if bundled_uv.is_file():
            uv = str(bundled_uv)
    if not uv:
        raise SystemExit("backend: uv not found on PATH (see docs/setup.md)")
    return [uv, "run", "--no-sync"] + app


def start(wait=600):
    if healthy():
        print(f"backend: already up ({API})")
        return True
    if listening_pid():
        # Up but not answering "ok": still starting, or wedged. Give it a moment first.
        for _ in range(30):
            if healthy():
                print("backend: up")
                return True
            time.sleep(2)
        print("backend: not answering; restarting it")
        stop()
    env = full_env()
    log = log_path(env)
    cmd = launch_cmd(env)
    if not (VS_DIR / "backend").is_dir():
        raise SystemExit(f"backend: no VoiceStudio checkout at {VS_DIR} (set VOICESTUDIO_DIR)")
    if sys.platform == "win32":
        # A child of this shell dies with an agent's session; one started by the
        # process service (WMI) doesn't. A batch file carries the environment.
        launcher = ROOT / "work" / "backend-start.cmd"
        launcher.parent.mkdir(exist_ok=True)
        lines = ["@echo off"] + [
            f'set "{k}={env[k].replace("%", "%%")}"'
            for k in ("PATH", "OMNIVOICE_DATA_DIR", "OMNIVOICE_CACHE_DIR", *ROCM_ENV)
            if env.get(k)
        ]
        lines += [f'cd /d "{VS_DIR}"', subprocess.list2cmdline(cmd) + f' > "{log}" 2>&1']
        launcher.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
        ps = (
            "$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
            f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{"
            f"CommandLine='cmd.exe /c \"{launcher}\"'; CurrentDirectory='{VS_DIR}'; ProcessStartupInformation=$si}}; "
            "exit $r.ReturnValue"
        )
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"backend: could not start it ({r.returncode}): {r.stderr.strip()[:300]}")
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as f:
            subprocess.Popen(cmd, cwd=VS_DIR, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"backend: starting (log {log})", flush=True)
    t0 = time.time()
    while time.time() - t0 < wait:
        if healthy():
            print(f"backend: up after {time.time() - t0:.0f} s")
            return True
        time.sleep(3)
    print(f"backend: no answer after {wait} s; see {log}")
    return False


def ensure():
    """True once the backend answers ok, starting it if needed."""
    return healthy() or start()


def restart():
    """A fresh backend: frees the GPU memory its PyTorch cache holds (docs/directing.md, gotchas)."""
    stop()
    return start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "start", "stop", "restart"])
    args = ap.parse_args()
    if args.cmd == "status":
        h, pid = health(), listening_pid()
        print(
            f"backend: {(h or {}).get('status', 'down')}"
            + (f", {h.get('device')}" if h else "")
            + (f", pid {pid}" if pid else "")
            + f"; log {log_path()}"
        )
        sys.exit(0 if h and h.get("status") == "ok" else 1)
    if args.cmd == "stop":
        stop()
    else:
        sys.exit(0 if (restart() if args.cmd == "restart" else start()) else 1)


if __name__ == "__main__":
    main()
