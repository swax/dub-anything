"""Coordinated Windows plugin/worker updates. Never replace a running dub.

The small watch process launches the current release's coordinator each minute.
Only public release metadata is fetched; no media or job data leaves the machine.
"""

import argparse
import base64
import contextlib
import json
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import release_info
from build_release import RUNTIME_DIRS, RUNTIME_FILES, digest, write_json
from dub_service import ACTIVE, Store, service_lock

RUNNABLE = (*ACTIVE, "queued", "publish_queued")
MAX_PACKAGE = 64 * 1024 * 1024


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", value):
        raise ValueError("Expected a four-part release version")
    return tuple(map(int, value.split(".")))


def trusted_url(url, feed=False):
    parsed = urlsplit(url)
    allowed = (
        parsed.hostname == "raw.githubusercontent.com"
        and parsed.path == f"/{release_info.REPOSITORY}/main/releases/updates.json"
        if feed
        else parsed.hostname == "github.com"
        and parsed.path.startswith(f"/{release_info.REPOSITORY}/releases/download/jellyfin-v")
    )
    if (
        parsed.scheme != "https"
        or parsed.port not in (None, 443)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not allowed
    ):
        raise ValueError("Update URL must belong to the configured public release repository")
    return url


def fetch(url, limit, feed=False):
    trusted_url(url, feed)
    with urlopen(Request(url, headers={"User-Agent": "Dub-Anything-Updater/1"}), timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Release download exceeds its size limit")
    return data


def select_release(feed, installed, server):
    if feed.get("schema") != 1 or not isinstance(feed.get("releases"), list):
        raise ValueError("Unsupported release feed format")
    candidates = []
    for entry in feed["releases"]:
        current = version(entry["version"])
        if current <= version(installed):
            continue
        if not version(entry["jellyfin_min"]) <= version(server) < version(entry["jellyfin_max"]):
            continue
        if (
            entry["worker_protocol"] != release_info.PROTOCOL
            or entry["updater_protocol"] != release_info.UPDATER_PROTOCOL
        ):
            continue
        trusted_url(entry["url"])
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Missing release SHA-256")
        candidates.append(entry)
    return max(candidates, key=lambda item: version(item["version"]), default=None)


def unpack(package, destination, entry):
    if digest(package) != entry["sha256"]:
        raise ValueError("Release checksum mismatch")
    with zipfile.ZipFile(package) as archive:
        seen, total = set(), 0
        for item in archive.infolist():
            path = pathlib.PurePosixPath(item.filename)
            bad_part = any(
                part.rstrip(" .") != part
                or part.split(".")[0].upper()
                in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
                for part in path.parts
            )
            if (
                item.orig_filename != item.filename
                or "\\" in item.filename
                or path.is_absolute()
                or ".." in path.parts
                or ":" in item.filename
                or not path.parts
                or path.as_posix() != item.filename
                or bad_part
            ):
                raise ValueError("Unsafe release archive path")
            if item.filename.casefold() in seen or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Duplicate or symbolic release entry")
            seen.add(item.filename.casefold())
            total += item.file_size
            if total > MAX_PACKAGE or item.file_size > 16 * 1024 * 1024:
                raise ValueError("Expanded release exceeds its size limit")
            if item.is_dir():
                raise ValueError("Release archives must contain regular files only")
            allowed = item.filename in (
                "release.json",
                "plugin/meta.json",
                "plugin/Jellyfin.Plugin.DubAnything.dll",
            ) or (
                path.parts[0] == "runtime"
                and len(path.parts) > 1
                and (path.parts[1] in RUNTIME_DIRS or "/".join(path.parts[1:]) in RUNTIME_FILES)
            )
            if not allowed:
                raise ValueError("Unexpected release archive entry")
        if not {"release.json", "plugin/meta.json", "plugin/jellyfin.plugin.dubanything.dll"} <= seen:
            raise ValueError("Incomplete plugin release")
        identity = json.loads(archive.read("release.json"))
        for key in ("version", "worker_protocol", "updater_protocol", "jellyfin_min", "jellyfin_max"):
            if identity[key] != entry[key]:
                raise ValueError("Plugin and worker release identities differ")
        expected = {"runtime/" + name for name in identity["files"]}
        actual = {item.filename for item in archive.infolist() if item.filename.startswith("runtime/")}
        if (
            expected != actual
            or not {"Directory.Build.props", "scripts/dub_service.py", "scripts/dub_updates.py"}
            <= identity["files"].keys()
        ):
            raise ValueError("Incomplete worker release")
        meta = json.loads(archive.read("plugin/meta.json"))
        if (
            meta["guid"] != release_info.PLUGIN_ID
            or meta["version"] != entry["version"]
            or meta.get("autoUpdate") is not False
        ):
            raise ValueError("Unsafe or mismatched Jellyfin plugin metadata")
        # All names are checked before writing anything, including on Windows.
        archive.extractall(destination)
    for name, checksum in identity["files"].items():
        if digest(destination / "runtime" / name) != checksum:
            raise ValueError("Worker file checksum mismatch")
    if release_info.metadata(destination / "runtime")["version"] != entry["version"]:
        raise ValueError("Worker assembly version differs from the feed")


def runtime_changes(runtime):
    baseline = runtime.parent / "release.json"
    if not baseline.is_file():
        return []  # Initial development checkout; automatic updates never write into it.
    expected = json.loads(baseline.read_text(encoding="utf-8"))["files"]
    changed = [
        name
        for name, checksum in expected.items()
        if not (runtime / name).is_file() or digest(runtime / name) != checksum
    ]
    for directory in RUNTIME_DIRS:
        for path in (runtime / directory).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md") and "__pycache__" not in path.parts:
                name = path.relative_to(runtime).as_posix()
                if name not in expected:
                    changed.append(name)
    return changed


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def powershell(code):
    encoded = base64.b64encode(("$ErrorActionPreference = 'Stop'\n" + code).encode("utf-16-le")).decode()
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        timeout=90,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        raise RuntimeError("Windows updater action failed: " + result.stderr[-1500:])
    return result.stdout


class Windows:
    def __init__(self, config, data):
        self.config, self.data = dict(config), data
        for key in ("jellyfin_exe", "jellyfin_data"):
            self.config[key] = str(pathlib.Path(self.config[key]).resolve())
        self.worker = json.loads((data / "config.json").read_text(encoding="utf-8"))

    def request(self, path, body=None):
        url = f"http://{self.worker['host']}:{self.worker['port']}/{path}"
        headers = {"Authorization": "Bearer " + self.worker["token"], "Content-Type": "application/json"}
        with urlopen(
            Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers), timeout=5
        ) as response:
            return json.load(response)

    def healthy(self):
        try:
            return self.request("health")
        except (OSError, ValueError):
            return None

    def server_version(self):
        with urlopen(self.config["jellyfin_url"].rstrip("/") + "/System/Info/Public", timeout=5) as response:
            document = json.load(response)
        value = document.get("Version", document.get("version"))
        if not isinstance(value, str):
            raise ValueError("Jellyfin has not reported its server version yet")
        return value + ".0" if len(value.split(".")) == 3 else value

    def stop_worker(self):
        try:
            # Health reads SQLite. During the exclusive admission lease it may
            # block, while shutdown deliberately needs no database access.
            self.request("shutdown", {})
        except HTTPError:
            raise
        except OSError:
            pass
        deadline = time.monotonic() + 65
        while time.monotonic() < deadline:
            try:
                with service_lock(self.data):
                    return
            except RuntimeError:
                time.sleep(0.25)
        raise RuntimeError("Worker did not stop gracefully; no forced termination was attempted")

    def start_worker(self, runtime):
        if self.healthy():
            raise RuntimeError("A worker is already listening; refusing a second worker")
        launcher = self.data / "start-managed-worker.ps1"
        launcher.write_text(
            "$ErrorActionPreference = 'Stop'\n"
            f"$env:VOICESTUDIO_DIR = {ps_quote(self.config['voicestudio'])}\n"
            f"Set-Location -LiteralPath {ps_quote(runtime)}\n"
            f"& {ps_quote(self.config['python'])} -u scripts/dub_service.py serve *>> {ps_quote(self.data / 'service.log')}\n",
            encoding="utf-8",
        )
        command = subprocess.list2cmdline(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)]
        )
        powershell(
            "$taskStartup = New-CimInstance Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}\n"
            f"$taskResult = Invoke-CimMethod Win32_Process Create -Arguments @{{CommandLine={ps_quote(command)};ProcessStartupInformation=$taskStartup}}\n"
            "if ($taskResult.ReturnValue -ne 0) { throw 'Worker launch failed' }"
        )

    def link_data(self, runtime):
        source = pathlib.Path(self.config["source_root"]).resolve()
        for name in ("work", "videos", "vendor"):
            target, link = source / name, runtime / name
            if target.is_dir() and not link.exists():
                powershell(f"New-Item -ItemType Junction -Path {ps_quote(link)} -Target {ps_quote(target)} | Out-Null")

    def stop_jellyfin(self):
        # Restrict process management to the configured Jellyfin installation.
        if self.config.get("jellyfin_service"):
            powershell(f"Stop-Service -Name {ps_quote(self.config['jellyfin_service'])} -ErrorAction Stop")
            return
        exe = ps_quote(self.config["jellyfin_exe"])
        powershell(
            f"$taskDataPattern = '(?i)' + [regex]::Escape({ps_quote(self.config['jellyfin_data'])}) + '(\"|\\s|$)'\n"
            f"Get-CimInstance Win32_Process -Filter \"Name='jellyfin.exe'\" | Where-Object {{ $_.ExecutablePath -eq {exe} -and $_.CommandLine.Replace('/', '\\') -match $taskDataPattern }} | ForEach-Object {{ $taskServerId = $_.ProcessId; Stop-Process -Id $taskServerId -ErrorAction Stop; Wait-Process -Id $taskServerId -Timeout 30 -ErrorAction SilentlyContinue }}"
        )

    def start_jellyfin(self):
        if self.config.get("jellyfin_service"):
            powershell(f"Start-Service -Name {ps_quote(self.config['jellyfin_service'])} -ErrorAction Stop")
            return
        # Detached process service avoids inheriting the updater's captured
        # stdout pipe or job lifetime. ShowWindow=0 keeps the server hidden.
        command = subprocess.list2cmdline([self.config["jellyfin_exe"], "--datadir", self.config["jellyfin_data"]])
        powershell(
            "$taskStartup = New-CimInstance Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}\n"
            f"$taskResult = Invoke-CimMethod Win32_Process Create -Arguments @{{CommandLine={ps_quote(command)};CurrentDirectory={ps_quote(pathlib.Path(self.config['jellyfin_exe']).parent)};ProcessStartupInformation=$taskStartup}}\n"
            "if ($taskResult.ReturnValue -ne 0) { throw 'Jellyfin launch failed' }"
        )

    def wait_ready(self, expected, legacy=False):
        deadline = time.monotonic() + 100
        while time.monotonic() < deadline:
            health = self.healthy()
            if health and health.get("service") == "Dub Anything":
                plugin = health.get("plugin") or {}
                if legacy or (
                    health.get("version") == expected
                    and plugin.get("version") == expected
                    and time.time() - plugin.get("seen", 0) < 90
                ):
                    try:
                        self.server_version()
                        return
                    except (OSError, ValueError):
                        pass
            time.sleep(1)
        raise RuntimeError("Matched plugin/worker startup verification timed out")


def checked_move(source, destination, source_root, destination_root):
    # Never recursively move a computed Windows path without containment checks.
    if not source.resolve().is_relative_to(source_root.resolve()) or not destination.resolve().is_relative_to(
        destination_root.resolve()
    ):
        raise ValueError("Move escaped the configured plugin or backup directory")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def add_catalog(jellyfin_data, backup):
    path = jellyfin_data / "config/system.xml"
    if not path.is_file():
        return
    tree = ET.parse(path)
    repositories = tree.find("PluginRepositories")
    if repositories is None:
        repositories = ET.SubElement(tree.getroot(), "PluginRepositories")
    if any(node.findtext("Url") == release_info.CATALOG for node in repositories):
        return
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup / "system.xml")
    repository = ET.SubElement(repositories, "RepositoryInfo")
    ET.SubElement(repository, "Name").text = "Dub Anything"
    ET.SubElement(repository, "Url").text = release_info.CATALOG
    ET.SubElement(repository, "Enabled").text = "true"
    temporary = path.with_suffix(".xml.dub-update")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(path)


class Coordinator:
    def __init__(self, install, platform=None):
        self.install = pathlib.Path(install).resolve()
        self.data = self.install.parent
        self.config = json.loads(self.install.read_text(encoding="utf-8-sig"))
        self.platform = platform or Windows(self.config, self.data)
        self.store = Store(self.data)
        self.journal = self.data / "update-journal.json"

    def status(self, state, message, **extra):
        write_json(
            self.data / "update-status.json",
            {
                "state": state,
                "message": message,
                "installed_version": self.config["version"],
                "checked": time.time(),
                "feed": release_info.FEED,
                "catalog": release_info.CATALOG,
                **extra,
            },
        )

    def idle(self, db):
        return not any(json.loads(row[0])["state"] in RUNNABLE for row in db.execute("SELECT document FROM jobs"))

    def stage(self, entry):
        releases = self.data / "releases"
        releases.mkdir(exist_ok=True)
        destination = releases / entry["version"]
        if not destination.exists():
            temporary = releases / (entry["version"] + ".staging-" + uuid.uuid4().hex)
            temporary.mkdir()
            package = temporary / "download.zip"
            package.write_bytes(fetch(entry["url"], MAX_PACKAGE))
            unpack(package, temporary, entry)
            temporary.rename(destination)
        else:
            package = destination / "download.zip"
            if not package.is_file() or digest(package) != entry["sha256"] or runtime_changes(destination / "runtime"):
                raise ValueError("Staged release differs from its verified download")
        self.platform.link_data(destination / "runtime")
        return destination

    def restore(self, transaction):
        old = transaction["previous"]
        with self.store.connect() as db:
            db.execute("BEGIN EXCLUSIVE")
            if not self.idle(db):
                self.status("waiting", "Interrupted update recovery is waiting for the running/queued jobs to finish.")
                return False
            db.execute(
                "INSERT OR REPLACE INTO service_state VALUES ('maintenance', ?)",
                (json.dumps({"version": transaction["version"]}),),
            )
            self.platform.stop_worker()
        self.platform.stop_jellyfin()
        plugins = pathlib.Path(old["jellyfin_data"]) / "plugins"
        new = pathlib.Path(transaction["plugin"])
        backup = pathlib.Path(transaction["backup"])
        if new.exists():
            checked_move(new, backup / ("failed-plugin-" + uuid.uuid4().hex), plugins, self.data / "backups")
        saved = backup / "plugin"
        if saved.exists():
            checked_move(saved, pathlib.Path(old["plugin_path"]), self.data / "backups", plugins)
        if (backup / "system.xml").is_file():
            shutil.copy2(backup / "system.xml", pathlib.Path(old["jellyfin_data"]) / "config/system.xml")
        old["active_runtime"] = old.pop("rollback_runtime", old["active_runtime"])
        old["failed_version"] = transaction["version"]
        write_json(self.install, old)
        self.config = old
        self.platform.start_worker(pathlib.Path(old["active_runtime"]))
        self.platform.start_jellyfin()
        self.platform.wait_ready(old["version"], legacy=old.get("legacy", False))
        write_json(self.journal, {"phase": "idle", "rolled_back": transaction["version"]})
        self.store.set_state("maintenance", None)
        return True

    def apply(self, entry, staged):
        old = dict(self.config)
        plugins = pathlib.Path(old["jellyfin_data"]) / "plugins"
        target = plugins / ("Dub Anything_" + entry["version"])
        if target.exists():
            raise FileExistsError("Target plugin folder already exists; refusing to replace it")
        backup = self.data / "backups" / uuid.uuid4().hex
        transaction = {
            "phase": "applying",
            "previous": old,
            "version": entry["version"],
            "plugin": str(target),
            "backup": str(backup),
        }
        # BEGIN EXCLUSIVE also gates the initial pre-updater worker. Retain the
        # SQLite lease until it exits, so even its old admission code cannot race.
        with self.store.connect() as db:
            db.execute("BEGIN EXCLUSIVE")
            if not self.idle(db):
                self.status(
                    "waiting",
                    "Update available; waiting for all running and queued jobs to finish.",
                    available_version=entry["version"],
                )
                return False
            if db.execute("SELECT 1 FROM service_state WHERE key = 'maintenance'").fetchone():
                raise RuntimeError("Another maintenance operation owns job admission")
            write_json(self.journal, transaction)
            db.execute(
                "INSERT INTO service_state VALUES ('maintenance', ?)", (json.dumps({"version": entry["version"]}),)
            )
            self.status(
                "installing",
                "Idle queue reserved. Updating plugin and worker together.",
                available_version=entry["version"],
            )
            try:
                self.platform.stop_worker()
            finally:
                db.commit()  # Keep admission paused even when graceful shutdown fails.
        try:
            self.platform.stop_jellyfin()
            original = pathlib.Path(old["plugin_path"])
            if original.exists():
                checked_move(original, backup / "plugin", plugins, self.data / "backups")
            if target.exists():
                raise FileExistsError("Target plugin folder already exists; refusing to replace it")
            shutil.copytree(staged / "plugin", target)
            add_catalog(pathlib.Path(old["jellyfin_data"]), backup)
            self.config = {
                **old,
                "active_runtime": str(staged / "runtime"),
                "plugin_path": str(target),
                "version": entry["version"],
                "legacy": False,
            }
            self.config.pop("rollback_runtime", None)
            self.config.pop("failed_version", None)
            write_json(self.install, self.config)
            self.platform.start_worker(staged / "runtime")
            self.platform.start_jellyfin()
            self.platform.wait_ready(entry["version"])
            write_json(self.journal, {"phase": "idle", "installed": entry["version"], "backup": str(backup)})
            self.store.set_state("maintenance", None)
            self.status("up_to_date", "Automatic updates enabled. Plugin and worker are on " + entry["version"] + ".")
            return True
        except Exception:
            self.restore(transaction)
            raise

    def once(self):
        journal = json.loads(self.journal.read_text(encoding="utf-8")) if self.journal.exists() else {}
        if journal.get("phase") == "applying":
            if self.restore(journal):
                self.status("rolled_back", "Recovered the previous installation after an interrupted update.")
            return
        if journal.get("phase") == "idle" and self.store.state("maintenance"):
            self.platform.wait_ready(self.config["version"], legacy=self.config.get("legacy", False))
            self.store.set_state("maintenance", None)
        if not self.config.get("enabled", True):
            self.status("disabled", "Automatic updates are disabled.")
            return
        if not self.platform.healthy():
            if any(job["state"] in ACTIVE for job in self.store.all()):
                self.status(
                    "waiting",
                    "Worker unavailable with an unfinished job recorded. Restart it manually before updating.",
                )
                return
            self.platform.start_worker(pathlib.Path(self.config["active_runtime"]))
            # Allow a worker to start without waiting for a plugin heartbeat.
            for _ in range(20):
                if self.platform.healthy():
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("Worker could not be started")
        cache = self.data / "release-feed.json"
        if not cache.exists() or time.time() - cache.stat().st_mtime > self.config.get("check_seconds", 21600):
            feed = json.loads(fetch(release_info.FEED, 1024 * 1024, feed=True))
            write_json(cache, feed)
        else:
            feed = json.loads(cache.read_text(encoding="utf-8"))
        entry = select_release(feed, self.config["version"], self.platform.server_version())
        if not entry:
            self.status("up_to_date", "Automatic updates enabled. No newer compatible release is available.")
            return
        if self.config.get("failed_version") == entry["version"]:
            self.status(
                "rolled_back",
                "Previous update failed and was rolled back. This version will not be retried automatically.",
                available_version=entry["version"],
            )
            return
        if not self.idle_probe():
            self.status(
                "waiting",
                "Update available; waiting for all running and queued jobs to finish.",
                available_version=entry["version"],
            )
            return
        if runtime_changes(pathlib.Path(self.config["active_runtime"])):
            self.status(
                "local_changes",
                "Update deferred: the worker has local code or language changes. Review them before updating.",
                available_version=entry["version"],
            )
            return
        staged = self.stage(entry)
        self.apply(entry, staged)

    def idle_probe(self):
        with self.store.connect() as db:
            return self.idle(db)


@contextlib.contextmanager
def update_lock(data, name="update.lock"):
    # Reuse the worker's tested OS lease with a separate lock-file directory.
    folder = data / name
    folder.mkdir(exist_ok=True)
    with service_lock(folder):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("once", "watch"))
    parser.add_argument("--install", type=pathlib.Path, required=True)
    args = parser.parse_args()
    data = args.install.resolve().parent
    if os.name != "nt":
        raise SystemExit("The automatic installation host currently supports Windows only")
    if args.command == "watch":
        with update_lock(data, "watch.lock"):
            while True:
                config = json.loads(args.install.read_text(encoding="utf-8-sig"))
                runtime = pathlib.Path(config["active_runtime"])
                if not (runtime / "scripts/dub_updates.py").is_file():
                    runtime = pathlib.Path(config["source_root"])
                subprocess.run(
                    [config["python"], str(runtime / "scripts/dub_updates.py"), "once", "--install", str(args.install)],
                    check=False,
                )
                time.sleep(60)
    else:
        with update_lock(data):
            coordinator = Coordinator(args.install)
            try:
                coordinator.once()
            except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error, URLError) as exc:
                coordinator.status("error", str(exc)[:1200])
                print(f"Update deferred/failed: {exc}", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
