"""Integration launchers must work after moving and through an older updater."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from integrations import director, dub_service, language_setup, release_info
from integrations.build_release import compatibility_launchers
from integrations.dub_updates import component_command

ROOT = pathlib.Path(__file__).resolve().parents[1]


class IntegrationLayoutTests(unittest.TestCase):
    def test_moved_modules_resolve_the_original_repo_and_data_paths(self):
        for module in (director, dub_service, language_setup, release_info):
            self.assertEqual(module.ROOT, ROOT)
        self.assertEqual(dub_service.DATA, ROOT / "work/service")
        self.assertRegex(release_info.metadata()["version"], r"^\d+\.\d+\.\d+\.\d+$")

    def test_each_component_launches_from_outside_the_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            for component in ("service", "updates", "build-release"):
                with self.subTest(component=component):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "scripts/integrations"), component, "--help"],
                        cwd=tmp,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_old_updater_paths_delegate_to_the_integration_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            scripts = pathlib.Path(tmp) / "scripts"
            package = scripts / "integrations"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            for name, data in compatibility_launchers():
                path = pathlib.Path(tmp) / name
                path.write_bytes(data)
                module = path.stem
                (package / f"{module}.py").write_text('import sys; print("delegated", sys.argv[1])')
                result = subprocess.run(
                    [sys.executable, str(path), "probe"],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "delegated probe")

    def test_coordinator_accepts_both_layouts_for_upgrade_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = pathlib.Path(tmp)
            legacy = runtime / "scripts/dub_service.py"
            self.assertEqual(component_command(runtime, "service"), [str(legacy)])
            package = runtime / "scripts/integrations"
            package.mkdir(parents=True)
            (package / "__main__.py").write_text("# package entry point")
            self.assertEqual(component_command(runtime, "service"), [str(package), "service"])
            self.assertEqual(component_command(runtime, "updates"), [str(package), "updates"])


if __name__ == "__main__":
    unittest.main()
