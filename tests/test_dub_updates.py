"""Release validation, atomic idle admission and recoverable paired installs."""

import hashlib
import contextlib
import json
import pathlib
import sys
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import dub_service as service
import dub_updates as updates
import release_info
from build_release import runtime_files, write_json


def entry(version="0.2.0.0"):
    return {
        "version": version,
        "jellyfin_min": "12.2.0.0",
        "jellyfin_max": "13.0.0.0",
        "worker_protocol": 2,
        "updater_protocol": 1,
        "sha256": "a" * 64,
        "url": f"https://github.com/swax/dub-anything/releases/download/jellyfin-v{version}/dub-anything-{version}.zip",
    }


class ReleaseTests(unittest.TestCase):
    def test_packager_excludes_private_hidden_untracked_and_media_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            subprocess.run(["git", "init", "-b", "main", str(root)], check=True, capture_output=True)
            names = [
                "scripts/tool.py",
                "scripts/.env",
                "scripts/token",
                "work/config.json",
                "videos/clip.mkv",
                "docs/setup.md",
            ]
            for name in names + ["scripts/untracked.py"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("generated fixture")
            subprocess.run(["git", "add", *names], cwd=root, check=True, capture_output=True)
            self.assertEqual({name for name, _ in runtime_files(root)}, {"scripts/tool.py", "docs/setup.md"})

    def test_catalog_registration_preserves_existing_settings_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "config").mkdir()
            source = "<ServerConfiguration><ServerName>Sample server</ServerName><PluginRepositories><RepositoryInfo><Name>Existing</Name><Url>https://example.invalid/manifest.json</Url><Enabled>true</Enabled></RepositoryInfo></PluginRepositories></ServerConfiguration>"
            (root / "config/system.xml").write_text(source)
            updates.add_catalog(root, root / "backup")
            updates.add_catalog(root, root / "backup")
            tree = updates.ET.parse(root / "config/system.xml")
            self.assertEqual(tree.findtext("ServerName"), "Sample server")
            self.assertEqual(len(tree.findall("PluginRepositories/RepositoryInfo")), 2)
            self.assertEqual((root / "backup/system.xml").read_text(), source)

    def test_only_newer_compatible_matched_releases_are_selected(self):
        newer = entry("0.3.0.0")
        feed = {"schema": 1, "releases": [entry(), newer]}
        self.assertEqual(updates.select_release(feed, "0.1.0.0", "12.2.0.0"), newer)
        self.assertIsNone(updates.select_release(feed, "0.3.0.0", "12.2.0.0"))
        self.assertIsNone(updates.select_release(feed, "0.1.0.0", "13.0.0.0"))
        newer["worker_protocol"] = 999
        self.assertEqual(updates.select_release(feed, "0.1.0.0", "12.2.0.0"), entry())

    def test_untrusted_hosts_and_malformed_versions_are_rejected(self):
        for url in (
            "http://github.com/swax/dub-anything/releases/download/jellyfin-v1/a.zip",
            "https://github.com/other/repository/releases/download/jellyfin-v1/a.zip",
            "https://github.com.evil.test/swax/dub-anything/releases/download/jellyfin-v1/a.zip",
            "https://github.com/swax/dub-anything/releases/download/jellyfin-v1/a.zip?redirect=1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                updates.trusted_url(url)
        for value in ("../../1", "1.0", "0.2.0.0-preview", 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                updates.version(value)

    def test_checksum_path_traversal_and_windows_aliases_are_rejected_before_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            package = root / "release.zip"
            for name in (
                "../escape.py",
                "runtime/scripts/CON.py",
                "runtime/scripts/a.py.",
                "runtime\\scripts\\a.py",
                "runtime/scripts/./a.py",
                "/escape.py",
            ):
                with zipfile.ZipFile(package, "w") as archive:
                    archive.writestr(name, "unsafe")
                candidate = {**entry(), "sha256": hashlib.sha256(package.read_bytes()).hexdigest()}
                with self.subTest(name=name), self.assertRaises(ValueError):
                    updates.unpack(package, root / "out", candidate)
                self.assertFalse((root / "out").exists())
            with self.assertRaisesRegex(ValueError, "checksum"):
                updates.unpack(package, root / "out", entry())

    def test_extraction_requires_a_complete_consistent_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            files = {
                "Directory.Build.props": "<Project><PropertyGroup><Version>0.2.0.0</Version><JellyfinVersion>12.2.0</JellyfinVersion></PropertyGroup></Project>",
                "scripts/dub_service.py": "# worker",
                "scripts/dub_updates.py": "# updater",
            }
            identity = {
                **entry(),
                "files": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()},
            }
            package = root / "release.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("release.json", json.dumps(identity))
                archive.writestr(
                    "plugin/meta.json",
                    json.dumps({"guid": release_info.PLUGIN_ID, "version": "0.2.0.0", "autoUpdate": False}),
                )
                archive.writestr("plugin/Jellyfin.Plugin.DubAnything.dll", "sample")
                for name, text in files.items():
                    archive.writestr("runtime/" + name, text)
            candidate = {**entry(), "sha256": hashlib.sha256(package.read_bytes()).hexdigest()}
            updates.unpack(package, root / "out", candidate)
            self.assertFalse(updates.runtime_changes(root / "out/runtime"))
            (root / "out/runtime/scripts/dub_service.py").write_text("# local change")
            self.assertIn("scripts/dub_service.py", updates.runtime_changes(root / "out/runtime"))
            (root / "out/runtime/scripts/language.py").write_text("# new language support")
            self.assertIn("scripts/language.py", updates.runtime_changes(root / "out/runtime"))


class FakePlatform:
    def __init__(self, store, fail_ready=False):
        self.store, self.fail_ready, self.calls = store, fail_ready, []

    def stop_worker(self):
        self.calls.append("stop_worker")

    def stop_jellyfin(self):
        self.calls.append("stop_jellyfin")

    def start_worker(self, runtime):
        self.calls.append("start_worker")
        self.maintenance_during_start = bool(self.store.state("maintenance"))
        self.assert_no_admission = self.store.claim() is None

    def start_jellyfin(self):
        self.calls.append("start_jellyfin")

    def wait_ready(self, expected, legacy=False):
        self.calls.append("ready_" + expected)
        if self.fail_ready and expected == "0.2.0.0":
            raise RuntimeError("simulated startup failure")

    def healthy(self):
        return {"service": "Dub Anything", "version": "0.1.0.0"}

    def server_version(self):
        return "12.2.0.0"


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.data = self.root / "work/service"
        self.store = service.Store(self.data)
        self.plugin = self.root / "jellyfin/plugins/Dub Anything_0.1.0.0"
        self.plugin.mkdir(parents=True)
        (self.plugin / "original.dll").write_bytes(b"original plugin")
        self.staged = self.root / "staged"
        (self.staged / "runtime").mkdir(parents=True)
        (self.staged / "plugin").mkdir()
        (self.staged / "plugin/new.dll").write_bytes(b"new plugin")
        self.install = self.data / "install.json"
        write_json(
            self.install,
            {
                "version": "0.1.0.0",
                "active_runtime": str(self.root),
                "source_root": str(self.root),
                "plugin_path": str(self.plugin),
                "jellyfin_data": str(self.root / "jellyfin"),
                "legacy": False,
            },
        )
        self.platform = FakePlatform(self.store)
        self.coordinator = updates.Coordinator(self.install, self.platform)

    def add_job(self, state):
        job = {"id": "a" * 32, "state": state, "cancel_requested": False, "title": "Sample video"}
        with self.store.connect() as db:
            db.execute("INSERT INTO jobs VALUES (?, ?)", (job["id"], json.dumps(job)))
        return job

    def test_every_running_or_queued_state_defers_without_stopping_anything(self):
        job = self.add_job("queued")
        for state in updates.RUNNABLE:
            self.store.update(job["id"], state=state)
            with self.subTest(state=state):
                self.assertFalse(self.coordinator.apply(entry(), self.staged))
                self.assertEqual(self.platform.calls, [])
                self.assertTrue((self.plugin / "original.dll").exists())
                self.assertIsNone(self.store.state("maintenance"))

    def test_success_updates_both_components_and_retains_previous_plugin(self):
        job = self.add_job("awaiting_review")
        self.assertTrue(self.coordinator.apply(entry(), self.staged))
        installed = json.loads(self.install.read_text())
        self.assertEqual(installed["version"], "0.2.0.0")
        self.assertEqual(pathlib.Path(installed["active_runtime"]), self.staged / "runtime")
        self.assertTrue((pathlib.Path(installed["plugin_path"]) / "new.dll").exists())
        self.assertEqual(len(list((self.data / "backups").glob("*/plugin/original.dll"))), 1)
        self.assertTrue(self.platform.maintenance_during_start)
        self.assertTrue(self.platform.assert_no_admission)
        self.assertIsNone(self.store.state("maintenance"))
        self.assertEqual(self.store.get(job["id"])["state"], "awaiting_review")

    def test_failed_startup_restores_both_components_and_quarantines_failed_plugin(self):
        self.platform.fail_ready = True
        with self.assertRaisesRegex(RuntimeError, "startup failure"):
            self.coordinator.apply(entry(), self.staged)
        installed = json.loads(self.install.read_text())
        self.assertEqual(installed["version"], "0.1.0.0")
        self.assertEqual(installed["failed_version"], "0.2.0.0")
        self.assertTrue((self.plugin / "original.dll").exists())
        self.assertFalse((self.plugin.parent / "Dub Anything_0.2.0.0").exists())
        self.assertEqual(len(list((self.data / "backups").glob("*/failed-plugin-*/new.dll"))), 1)
        self.assertIsNone(self.store.state("maintenance"))

    def test_crash_recovery_also_waits_for_active_jobs(self):
        self.add_job("rendering")
        transaction = {"previous": self.coordinator.config, "version": "0.2.0.0"}
        self.assertFalse(self.coordinator.restore(transaction))
        self.assertEqual(self.platform.calls, [])

    def test_a_job_claimed_during_download_is_checked_again_before_installing(self):
        job = self.add_job("queued")
        claimed = self.store.claim()
        self.assertEqual(claimed["state"], "staging")
        self.assertFalse(self.coordinator.apply(entry(), self.staged))
        self.assertEqual(self.store.get(job["id"])["state"], "staging")
        self.assertEqual(self.platform.calls, [])

    def test_maintenance_blocks_new_jobs_resume_and_publish_but_allows_cancel(self):
        job = self.add_job("awaiting_review")
        self.store.set_state("maintenance", {"version": "0.2.0.0"})
        with self.assertRaises(service.Maintenance):
            self.store.action(job["id"], "publish")
        self.store.action(job["id"], "cancel")
        with self.assertRaises(service.Maintenance):
            self.store.action(job["id"], "resume")
        with self.store.connect() as db, self.assertRaises(service.Maintenance):
            self.store.check_admission(db)
        self.assertIsNone(self.store.claim())

    def test_failed_version_is_not_automatically_retried(self):
        self.coordinator.config["failed_version"] = "0.2.0.0"
        write_json(self.data / "release-feed.json", {"schema": 1, "releases": [entry()]})
        with patch.object(self.coordinator, "stage") as stage:
            self.coordinator.once()
            stage.assert_not_called()
        self.assertEqual(json.loads((self.data / "update-status.json").read_text())["state"], "rolled_back")

    def test_shutdown_does_not_probe_health_while_the_database_is_exclusively_locked(self):
        write_json(self.data / "config.json", {"host": "127.0.0.1", "port": 8769, "token": "test-only"})
        platform = updates.Windows(
            {"jellyfin_exe": "server.exe", "jellyfin_data": str(self.root / "jellyfin")}, self.data
        )
        with (
            patch.object(platform, "request") as request,
            patch.object(platform, "healthy", side_effect=AssertionError),
            patch.object(updates, "service_lock", return_value=contextlib.nullcontext()),
        ):
            platform.stop_worker()
        request.assert_called_once_with("shutdown", {})

    def test_moves_outside_configured_paths_are_refused(self):
        with self.assertRaises(ValueError):
            updates.checked_move(self.plugin, self.root / "elsewhere", self.plugin.parent, self.data / "backups")


if __name__ == "__main__":
    unittest.main()
