"""Local authenticated job service for media-server adapters.

Run with VoiceStudio's Python: scripts/integrations service serve
Settings and the persistent queue live in ignored work/service/.
"""

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import pathlib
import re
import secrets
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import backend
from integrations import director, language_setup, release_info

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "work" / "service"
ACTIVE = ("staging", "language_setup", "preparing", "directing", "rendering", "publishing")
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".ts", ".m2ts"}


def settings(data=DATA):
    data.mkdir(parents=True, exist_ok=True)
    path = data / "config.json"
    if not path.exists():
        config = {"host": "127.0.0.1", "port": 8767, "token": secrets.token_urlsafe(32)}
        # Exclusive create also prevents a second startup replacing a live token.
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
            if os.name != "nt":
                path.chmod(0o600)
        except FileExistsError:
            pass
    result = director.read_json(path)
    if result.get("host") not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("this first version binds to loopback only")
    if not isinstance(result.get("token"), str) or len(result["token"]) < 32:
        raise ValueError("the service token must contain at least 32 characters")
    return result


class Store:
    def __init__(self, data=DATA):
        self.data = data
        data.mkdir(parents=True, exist_ok=True)
        self.path = data / "queue.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, document TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS service_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @staticmethod
    def check_admission(db):
        if db.execute("SELECT 1 FROM service_state WHERE key = 'maintenance'").fetchone():
            raise Maintenance("An idle update is in progress. Try again once both components restart.")

    def state(self, key, default=None):
        with self.connect() as db:
            row = db.execute("SELECT value FROM service_state WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_state(self, key, value):
        with self.connect() as db:
            if value is None:
                db.execute("DELETE FROM service_state WHERE key = ?", (key,))
            else:
                db.execute("INSERT OR REPLACE INTO service_state VALUES (?, ?)", (key, json.dumps(value)))

    def claim(self):
        # Claim and maintenance admission share a transaction, leaving no gap
        # between observing an idle queue and a new job starting.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM service_state WHERE key = 'maintenance'").fetchone():
                return None
            for row in db.execute("SELECT id, document FROM jobs ORDER BY rowid"):
                identity, document = row
                job = json.loads(document)
                if job["state"] in ("queued", "publish_queued"):
                    job["claimed_state"] = job["state"]
                    job["state"] = "publishing" if job["state"] == "publish_queued" else "staging"
                    db.execute("UPDATE jobs SET document = ? WHERE id = ?", (json.dumps(job), identity))
                    return job
        return None

    @contextlib.contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def all(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT document FROM jobs ORDER BY rowid DESC")]

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT document FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("job not found")
        return json.loads(row[0])

    def update(self, job_id, **changes):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT document FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("job not found")
            job = json.loads(row[0])
            job.update(changes, updated=time.time())
            db.execute("UPDATE jobs SET document = ? WHERE id = ?", (json.dumps(job), job_id))
        return job

    def create(self, request):
        kind = request.get("director", "codex")
        if kind not in director.KINDS:
            raise ValueError("choose codex or claude")
        source = language_setup.language(request.get("source_lang", "ja"))
        target = language_setup.language(request.get("target_lang", "en"))
        voices = request.get("voices", "actor")
        if voices not in ("actor", "american", "both") or (
            target in language_setup.catalog().LANGUAGES and target != "en" and voices != "actor"
        ):
            raise ValueError("American and both voices are available only into English")
        path = pathlib.Path(request.get("path", "")).expanduser()
        if not path.is_absolute() or not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("the library item must be an existing local video file")
        path = path.resolve()
        if not re.fullmatch(r"[a-fA-F0-9-]{32,36}", request.get("item_id", "")):
            raise ValueError("a Jellyfin item id is required")
        model = request.get("model", "")
        if not isinstance(model, str) or len(model) > 150 or model.startswith("-"):
            raise ValueError("invalid model name")
        for option in ("review_before_publish",):
            if option in request and not isinstance(request[option], bool):
                raise ValueError(f"{option} must be a boolean")
        identity = uuid.uuid4().hex
        job = {
            "id": identity,
            "slug": "jf_" + identity[:20],
            "item_id": request["item_id"],
            "title": str(request.get("title") or path.stem)[:500],
            "path": str(path),
            "source_lang": source,
            "target_lang": target,
            "requested_source": request.get("source_lang", "ja"),
            "requested_target": request.get("target_lang", "en"),
            "voices": voices,
            "director": kind,
            "model": model,
            "state": "queued",
            "stage": "Queued",
            "created": time.time(),
            "updated": time.time(),
            "cancel_requested": False,
            "review_before_publish": request.get("review_before_publish", True),
            "auto_text": True,
            "signs": True,
        }
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self.check_admission(db)
            for row in db.execute("SELECT document FROM jobs"):
                old = json.loads(row[0])
                if (
                    old["path"] == job["path"]
                    and old["target_lang"] == target
                    and old["voices"] == voices
                    and old["state"] not in ("failed", "cancelled")
                ):
                    raise ValueError("this item, target and voice already have a job; resume or publish that job")
            db.execute("INSERT INTO jobs VALUES (?, ?)", (identity, json.dumps(job)))
        return job

    def recover(self):
        for job in self.all():
            if job["state"] in ACTIVE:
                self.update(job["id"], state="interrupted", stage="Worker stopped; resume to continue")

    def action(self, job_id, action):
        # All state transitions are serialized, including races with the worker.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT document FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("job not found")
            job = json.loads(row[0])
            if action in ("resume", "publish"):
                self.check_admission(db)
            if action == "cancel" and job["state"] in (*ACTIVE, "queued", "awaiting_review", "publish_queued"):
                job.update(cancel_requested=True)
                if job["state"] not in ACTIVE:
                    job.update(state="cancelled", stage="Cancelled; working files kept")
            elif action == "resume" and job["state"] in ("failed", "interrupted", "cancelled"):
                job.update(state="queued", stage="Queued to resume", cancel_requested=False, error="")
            elif action == "publish" and job["state"] == "awaiting_review":
                job.update(state="publish_queued", stage="Queued to publish", cancel_requested=False)
            else:
                raise ValueError(f"cannot {action} a job in state {job['state']}")
            job["updated"] = time.time()
            db.execute("UPDATE jobs SET document = ? WHERE id = ?", (json.dumps(job), job_id))
        return job


class Cancelled(Exception):
    pass


class Maintenance(Exception):
    pass


class Worker:
    def __init__(self, store):
        self.store = store
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, name="dub-worker", daemon=True)

    def cancelled(self, job):
        if self.stop.is_set() or self.store.get(job["id"])["cancel_requested"]:
            raise Cancelled()

    def set_stage(self, job, state, stage):
        self.cancelled(job)
        self.store.update(job["id"], state=state, stage=stage)

    def folder(self, job):
        path = self.store.data / "jobs" / job["id"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(self, job, args, name, prompt=None, timeout=24 * 3600, cwd=ROOT):
        self.cancelled(job)
        path = self.folder(job) / f"{name}.log"
        env = director.environment()
        with path.open("a", encoding="utf-8") as log:
            log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log.flush()
            offset = log.tell()
            kwargs = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt"
                else {"start_new_session": True}
            )
            with subprocess.Popen(
                args,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE if prompt is not None else subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **kwargs,
            ) as process:
                try:
                    if prompt is not None:
                        process.stdin.write(prompt.encode("utf-8"))
                        process.stdin.close()
                    started = time.monotonic()
                    while process.poll() is None:
                        self.cancelled(job)
                        if time.monotonic() - started > timeout:
                            raise RuntimeError(f"{name} timed out; see {path}")
                        self.stop.wait(0.5)
                finally:
                    if process.poll() is None:
                        if os.name == "nt":
                            killed = subprocess.run(
                                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=False,
                                timeout=10,
                            )
                            if killed.returncode and process.poll() is None:
                                process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                code = process.returncode
        with path.open(encoding="utf-8", errors="replace") as log:
            log.seek(offset)
            # Limit UI/error excerpts without loading a full feature-film log.
            output = ""
            for line in log:
                output = (output + line)[-24000:]
        if code:
            raise RuntimeError(f"{name} exited {code}; see {path}\n{output[-1200:]}")
        if prompt is not None:
            director.check_result(path, offset, job["director"])
        return output

    def pipeline(self, job, *args):
        return self.run(job, [backend.vs_python(), "scripts/dub_anything.py", *args], "pipeline")

    def process(self, job):
        folder = self.folder(job)
        self.set_stage(job, "staging", "Checking director and copying the source")
        director.command(job["director"], job["model"])
        source = pathlib.Path(job["path"])
        staged = folder / "source" / source.name
        staged.parent.mkdir(exist_ok=True)
        if not staged.exists():
            self.copy(job, source, staged)
        job = language_setup.ensure(self, job, staged)
        if job["target_lang"] != "en" and job["voices"] != "actor":
            raise ValueError("American and both voices are available only into English; choose Actor's voice")
        if not (ROOT / "work" / "jobs" / job["slug"] / "job.json").exists():
            options = [
                "new",
                str(staged),
                "--slug",
                job["slug"],
                "--source",
                job["source_lang"],
                "--target",
                job["target_lang"],
                "--voices",
                job["voices"],
            ]
            options += ["--auto-text"]
            self.pipeline(job, *options)
        import dub_anything as runner

        pipeline_job = runner.load_job(job["slug"])
        ps = runner.parts(pipeline_job)
        # On resume, do not give unvalidated scripts written by an interrupted
        # director to the renderer. Directing starts only after all prep is done.
        prepared = ps and all(
            (runner.original_only(part) and not runner.drafted(part))
            or (
                runner.drafted(part)
                and (not pipeline_job.get("captions") or (ROOT / "work" / part["name"] / "subs.json").exists())
                and (not pipeline_job.get("signs") or (ROOT / "work" / part["name"] / "signs.json").exists())
            )
            for part in ps
        )
        if not prepared:
            self.set_stage(job, "preparing", "Separating audio and drafting dialogue")
            self.pipeline(job, "run", job["slug"])
            pipeline_job = runner.load_job(job["slug"])
            ps = runner.parts(pipeline_job)
        if not ps:
            raise RuntimeError("the pipeline produced no parts")
        for index, part in enumerate(ps, 1):
            self.cancelled(job)
            script_path = ROOT / "work" / part["name"] / "script.json"
            script = director.read_json(script_path)
            approved = self.store.get(job["id"]).get("approved_parts", {})
            if script and (
                (script.get("ready") and approved.get(part["name"]) == director.script_signature(script))
                or (script.get("lines") == [] and not runner.drafted(part))
            ):
                continue
            self.set_stage(job, "directing", f"{job['director']}: directing part {index} of {len(ps)}")
            feedback = ""
            for attempt in range(3):
                self.run(
                    job,
                    director.command(job["director"], job["model"]),
                    f"director-{index:02d}",
                    director.prompt(job, part, folder / "notes.md", feedback),
                    timeout=3600,
                )
                try:
                    script = director.validate_script(
                        director.read_json(script_path),
                        part,
                        job,
                        director.read_json(ROOT / "work" / "voices.json", {}),
                    )
                    runner.write_json(script_path, script)
                    if script["lines"]:
                        self.run(
                            job,
                            [backend.vs_python(), "scripts/edges.py", part["name"], "--apply"],
                            f"edges-{index:02d}",
                        )
                        flags = ""
                        for voice in ["actor", "american"] if job["voices"] == "both" else [job["voices"]]:
                            output = self.run(
                                job,
                                [backend.vs_python(), "scripts/fit.py", part["name"], "--voice", voice, "--only-flags"],
                                f"fit-{index:02d}-{voice}",
                            )
                            flags += "\n".join(
                                line for line in output.splitlines() if re.search(r"\b(?:OVER|SHORT)\??", line)
                            )
                        if flags:
                            raise ValueError("Fix these timing predictions:\n" + flags)
                    # Edges may have changed the script. Validate that final version too.
                    script = director.validate_script(
                        director.read_json(script_path),
                        part,
                        job,
                        director.read_json(ROOT / "work" / "voices.json", {}),
                    )
                    script["ready"] = True
                    runner.write_json(script_path, script)
                    approved[part["name"]] = director.script_signature(script)
                    self.store.update(job["id"], approved_parts=approved)
                    break
                except ValueError as exc:
                    feedback = str(exc)
                    if attempt == 2:
                        raise RuntimeError(f"part {index} needs review: {feedback}") from exc
        self.set_stage(job, "rendering", "Rendering voices, checking lines, and assembling the dub")
        self.pipeline(job, "run", job["slug"])
        result = runner.load_job(job["slug"]).get("delivered", {}).get("path")
        if not result or not pathlib.Path(result).is_file():
            raise RuntimeError("the pipeline did not deliver a file; inspect pipeline.log")
        self.store.update(job["id"], rendered_path=result, report_path=str(runner.JOB_DIR / job["slug"] / "report.md"))
        if job["review_before_publish"]:
            self.set_stage(job, "awaiting_review", "Dub ready: review the report and preview, then publish")
        else:
            self.publish(self.store.get(job["id"]))

    def copy(self, job, source, destination):
        temporary = destination.with_name(destination.name + f".{job['id']}.partial")
        before = source.stat()
        checksum = hashlib.sha256()
        # Restart an interrupted copy, keeping the source untouched.
        with source.open("rb") as src, temporary.open("wb") as dst:
            while chunk := src.read(4 * 1024 * 1024):
                self.cancelled(job)
                dst.write(chunk)
                checksum.update(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("source changed during copy")
        if checksum.hexdigest() != self.checksum(job, temporary):
            raise RuntimeError("copy verification failed; the original is intact")
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        temporary.rename(destination)

    def checksum(self, job, path):
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(4 * 1024 * 1024):
                self.cancelled(job)
                checksum.update(chunk)
        return checksum.hexdigest()

    def publish(self, job):
        self.set_stage(job, "publishing", "Publishing a separate dubbed version")
        rendered = pathlib.Path(job["rendered_path"])
        original = pathlib.Path(job["path"])
        destination = original.with_name(
            f"{original.stem} - Dub Anything {job['target_lang']} {job['voices']} {job['id'][:8]}{rendered.suffix}"
        )
        if destination.exists():
            if self.checksum(job, rendered) != self.checksum(job, destination):
                raise FileExistsError(f"refusing to overwrite {destination}")
        else:
            self.copy(job, rendered, destination)
        self.store.update(
            job["id"], published_path=str(destination), state="completed", stage="Published to the library"
        )

    def loop(self):
        while not self.stop.is_set():
            job = self.store.claim()
            if not job:
                self.stop.wait(1)
                continue
            try:
                if job["claimed_state"] == "publish_queued":
                    self.publish(job)
                elif job.get("rendered_path") and pathlib.Path(job["rendered_path"]).is_file():
                    self.set_stage(job, "awaiting_review", "Dub ready: review and publish")
                else:
                    self.process(job)
            except Cancelled:
                self.store.update(
                    job["id"],
                    state="interrupted" if self.stop.is_set() else "cancelled",
                    stage="Worker stopped" if self.stop.is_set() else "Cancelled; working files kept",
                )
            except Exception as exc:
                self.store.update(job["id"], state="failed", stage="Needs attention", error=str(exc))


def capabilities():
    env = director.environment()
    registry = language_setup.catalog()
    return {
        "directors": [{"id": kind, "available": bool(director.executable(kind, env))} for kind in director.KINDS],
        "pairs": [
            {"source": a, "target": b, "label": f"{registry.NAMES[a]} → {registry.NAMES[b]}"} for a, b in registry.PAIRS
        ],
        "languages": [
            {"code": code, "name": module.NAME, "native": module.NATIVE} for code, module in registry.LANGUAGES.items()
        ],
        "voices": {key: list(value.VOICES) for key, value in registry.LANGUAGES.items()},
        "python": backend.vs_python(),
        "ffmpeg": bool(shutil.which("ffmpeg", path=env.get("PATH"))),
    }


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config, store):
        self.config, self.store = config, store
        super().__init__((config["host"], config["port"]), Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Do not log authorization headers or POST bodies.
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def reply(self, status, data, content_type="application/json"):
        body = data.encode("utf-8") if isinstance(data, str) else json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def dispatch(self):
        path = urlsplit(self.path).path.rstrip("/")
        if self.command == "GET" and path == "/health":
            return self.reply(
                200,
                {
                    "status": "ok",
                    "service": "Dub Anything",
                    "version": release_info.metadata()["version"],
                    "protocol": release_info.PROTOCOL,
                    "plugin": self.server.store.state("plugin"),
                },
            )
        token = self.headers.get("Authorization", "")
        if not hmac.compare_digest(token, "Bearer " + self.server.config["token"]):
            return self.reply(401, {"error": "service authentication required"})
        body = {}
        if self.command == "POST":
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 32768:
                return self.reply(413, {"error": "request too large"})
            body = json.loads(self.rfile.read(length) or "{}")
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
        if path == "/capabilities" and self.command == "GET":
            return self.reply(200, capabilities())
        if path == "/plugin" and self.command == "POST":
            if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", body.get("version", "")):
                raise ValueError("invalid plugin version")
            self.server.store.set_state("plugin", {**body, "seen": time.time()})
            return self.reply(200, {"status": "ok"})
        if path == "/updates" and self.command == "GET":
            status = director.read_json(self.server.store.data / "update-status.json", {})
            return self.reply(
                200, status or {"state": "disabled", "message": "Automatic updates have not been set up."}
            )
        if path == "/shutdown" and self.command == "POST":
            self.reply(200, {"status": "stopping"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if path == "/jobs":
            if self.command == "GET":
                return self.reply(200, self.server.store.all())
            return self.reply(201, self.server.store.create(body))
        match = re.fullmatch(r"/jobs/([a-f0-9]{32})(?:/(cancel|resume|publish|report|log))?", path)
        if match:
            job_id, action = match.groups()
            job = self.server.store.get(job_id)
            if self.command == "POST" and action in ("cancel", "resume", "publish"):
                return self.reply(200, self.server.store.action(job_id, action))
            if self.command == "GET":
                if action in ("report", "log"):
                    if action == "report":
                        file = ROOT / "work" / "jobs" / job["slug"] / "report.md"
                    else:
                        files = list((self.server.store.data / "jobs" / job_id).glob("*.log"))
                        file = max(files, key=lambda f: f.stat().st_mtime) if files else None
                    text = "No report/log yet."
                    if file and file.exists():
                        with file.open("rb") as handle:
                            handle.seek(max(0, file.stat().st_size - 64000))
                            text = handle.read().decode("utf-8", errors="replace")
                    return self.reply(200, {"text": text})
                if action is None:
                    return self.reply(200, job)
        return self.reply(404, {"error": "endpoint not found"})

    def handle_request(self):
        try:
            self.dispatch()
        except Maintenance as exc:
            self.reply(503, {"error": str(exc)})
        except KeyError as exc:
            self.reply(404, {"error": str(exc)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.reply(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(f"request failed: {type(exc).__name__}: {exc}", flush=True)
            self.reply(500, {"error": "worker error; see service log"})

    do_GET = handle_request
    do_POST = handle_request


@contextlib.contextmanager
def service_lock(data):
    """An OS-held lease prevents two workers sharing this queue and cast."""
    path = data / "worker.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another Dub Anything worker is already running") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def serve(data=DATA):
    config = settings(data)
    with service_lock(data):
        store = Store(data)
        store.set_state("plugin", None)
        server = Server(config, store)
        store.recover()
        worker = Worker(store)
        worker.thread.start()
        print(
            f"Dub Anything worker: http://{config['host']}:{config['port']} (settings: {data / 'config.json'})",
            flush=True,
        )
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            worker.stop.set()
            worker.thread.join(timeout=30)
            server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "init", "status", "stop"))
    args = parser.parse_args()
    if args.command == "serve":
        serve()
    elif args.command == "init":
        settings()
        print(f"Settings created at {DATA / 'config.json'} (contains the private service token)")
    elif args.command == "stop":
        config = settings()
        request = Request(
            f"http://{config['host']}:{config['port']}/shutdown",
            data=b"{}",
            headers={"Authorization": "Bearer " + config["token"], "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            print(response.read().decode())
    else:
        print(json.dumps(capabilities(), indent=2))


if __name__ == "__main__":
    main()
