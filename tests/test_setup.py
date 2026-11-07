"""Pinned backend discovery and first-run setup without model downloads."""

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import backend
import setup as bootstrap


class BackendDiscoveryTests(unittest.TestCase):
    def test_bundled_fork_precedes_legacy_installations_on_every_platform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "dub-anything"
            bundled = root / "vendor" / "VoiceStudio"
            sibling = root.parent / "VoiceStudio"
            desktop = root / "home" / "Library" / "Application Support" / "VoiceStudio" / "runtime" / "project"
            for checkout in (bundled, sibling, desktop):
                (checkout / "backend").mkdir(parents=True)
            with (
                patch.object(backend, "ROOT", root),
                patch.object(pathlib.Path, "home", return_value=root / "home"),
                patch.dict(os.environ, {}, clear=True),
            ):
                for platform in ("win32", "linux", "darwin"):
                    with self.subTest(platform=platform), patch("sys.platform", platform):
                        self.assertEqual(backend.find_vs_dir(), bundled)
                with patch.dict(os.environ, {"VOICESTUDIO_DIR": str(sibling)}):
                    self.assertEqual(backend.find_vs_dir(), sibling)

    def test_uninitialized_submodule_keeps_the_sibling_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "dub-anything"
            (root / "vendor" / "VoiceStudio").mkdir(parents=True)
            sibling = root.parent / "VoiceStudio"
            (sibling / "backend").mkdir(parents=True)
            with (
                patch.object(backend, "ROOT", root),
                patch.dict(os.environ, {}, clear=True),
                patch("sys.platform", "linux"),
            ):
                self.assertEqual(backend.find_vs_dir(), sibling)

    def test_macos_keeps_the_desktop_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "dub-anything"
            desktop = root / "home" / "Library" / "Application Support" / "VoiceStudio" / "runtime" / "project"
            (desktop / "backend").mkdir(parents=True)
            with (
                patch.object(backend, "ROOT", root),
                patch.object(pathlib.Path, "home", return_value=root / "home"),
                patch.dict(os.environ, {}, clear=True),
                patch("sys.platform", "darwin"),
            ):
                self.assertEqual(backend.find_vs_dir(), desktop)

    def test_fresh_clone_points_at_the_bundled_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            with (
                patch.object(backend, "ROOT", root),
                patch.dict(os.environ, {}, clear=True),
                patch("sys.platform", "linux"),
            ):
                self.assertEqual(backend.find_vs_dir(), root / "vendor" / "VoiceStudio")


class SetupTests(unittest.TestCase):
    def test_init_only_checks_out_the_pin_and_never_installs_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bundled = root / bootstrap.SUBMODULE
            (bundled / "backend").mkdir(parents=True)
            with (
                patch.object(bootstrap, "ROOT", root),
                patch("shutil.which", side_effect=lambda name: "/tools/git" if name == "git" else None),
                patch("subprocess.run") as run,
            ):
                bootstrap.setup(init_only=True)
            self.assertEqual(
                [c.args[0] for c in run.call_args_list],
                [
                    ["git", "submodule", "sync", "--recursive", "--", bootstrap.SUBMODULE],
                    ["git", "submodule", "update", "--init", "--recursive", "--", bootstrap.SUBMODULE],
                ],
            )
            self.assertTrue(all(c.kwargs["check"] for c in run.call_args_list))
            self.assertTrue(all(c.kwargs["cwd"] == root for c in run.call_args_list))

    def test_failed_submodule_checkout_stops_before_dependency_installation(self):
        with (
            patch("shutil.which", return_value="/tools/git"),
            patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")) as run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                bootstrap.setup()
        self.assertEqual(run.call_count, 1)

    def test_dependency_setup_uses_its_own_venv_and_respects_an_explicit_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bundled, external = root / "bundled", root / "external"
            for checkout in (bundled, external):
                (checkout / "backend").mkdir(parents=True)
            for override in (None, external):
                env = {"VIRTUAL_ENV": "/another/venv", "UV_PROJECT_ENVIRONMENT": "/another/project"}
                if override:
                    env["VOICESTUDIO_DIR"] = str(override)
                with (
                    self.subTest(override=override),
                    patch.dict(os.environ, env, clear=True),
                    patch.object(bootstrap, "initialize_voicestudio", return_value=bundled),
                    patch("shutil.which", return_value="/tools/uv"),
                    patch("subprocess.run") as run,
                ):
                    bootstrap.setup()
                self.assertEqual(
                    [c.args[0] for c in run.call_args_list],
                    [
                        ["/tools/uv", "sync", "--locked"],
                        ["/tools/uv", "run", "--no-sync", "python", "scripts/setup.py"],
                    ],
                )
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs["cwd"], override or bundled)
                    self.assertNotIn("VIRTUAL_ENV", call.kwargs["env"])
                    self.assertNotIn("UV_PROJECT_ENVIRONMENT", call.kwargs["env"])
                    self.assertTrue(call.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
