"""Unit tests for macOS cross-platform support and fallbacks."""

import os
import pathlib
import runpy
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import backend
import core
import dub_anything
import frames


class FixedHomeTests(unittest.TestCase):
    def setUp(self):
        self.home = pathlib.Path("/mock/home")
        self.enterContext(patch.object(pathlib.Path, "home", return_value=self.home))


class PlatformPathTests(FixedHomeTests):
    def test_default_data_dir_resolution(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("sys.platform", "darwin"):
                p = core.default_data_dir()
                self.assertEqual(p, pathlib.Path.home() / "Library" / "Application Support" / "OmniVoice")

            with patch("sys.platform", "win32"):
                with patch.dict(os.environ, {"APPDATA": "/mock/AppData"}):
                    p = core.default_data_dir()
                    self.assertEqual(p, pathlib.Path("/mock/AppData/OmniVoice"))

            with patch("sys.platform", "linux"):
                p = core.default_data_dir()
                self.assertEqual(p, pathlib.Path.home() / ".local" / "share" / "OmniVoice")

    def test_default_cache_dir_resolution(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("sys.platform", "darwin"):
                p = core.default_cache_dir()
                self.assertEqual(p, str(pathlib.Path.home() / "Library" / "Caches" / "OmniVoice" / "hf_cache"))

            with patch("sys.platform", "win32"):
                with patch.dict(os.environ, {"LOCALAPPDATA": "/mock/LocalAppData"}):
                    p = core.default_cache_dir()
                    self.assertEqual(p, str(pathlib.Path("/mock/LocalAppData/OmniVoice/hf_cache")))

    def test_explicit_env_overrides_paths(self):
        with patch.dict(
            os.environ,
            {"OMNIVOICE_DATA_DIR": "/custom/data", "OMNIVOICE_CACHE_DIR": "/custom/cache"},
            clear=True,
        ):
            self.assertEqual(core.default_data_dir(), pathlib.Path("/custom/data"))
            self.assertEqual(core.default_cache_dir(), "/custom/cache")
            self.assertEqual(core.cache_root(), pathlib.Path("/custom/cache"))

    def test_hf_hub_cache_does_not_move_the_model_folder(self):
        # separate.py keeps audio-separator/ where it was; only the Hugging Face cache follows HF_HUB_CACHE.
        with (
            patch.dict(os.environ, {"HF_HUB_CACHE": "/hf", "LOCALAPPDATA": "/mock/LocalAppData"}, clear=True),
            patch("sys.platform", "win32"),
        ):
            self.assertEqual(core.cache_root(), pathlib.Path("/mock/LocalAppData/OmniVoice"))
            self.assertEqual(core.default_cache_dir(), "/hf")


class BackendPlatformTests(FixedHomeTests):
    def launch(self, platform, venv, uv="/tools/uv"):
        with (
            patch("sys.platform", platform),
            patch.object(backend, "venv_python", return_value=venv),
            patch("shutil.which", return_value=uv),
            patch.object(pathlib.Path, "is_file", return_value=False),
        ):
            return backend.launch_cmd({"PATH": ""})

    def test_windows_starts_through_uv_beside_a_venv(self):
        cmd = self.launch("win32", "C:/VoiceStudio/.venv/Scripts/python.exe")
        self.assertEqual(cmd[:4], ["/tools/uv", "run", "--no-sync", "uvicorn"])

    def test_windows_still_needs_uv(self):
        with self.assertRaises(SystemExit):
            self.launch("win32", "C:/VoiceStudio/.venv/Scripts/python.exe", uv=None)

    def test_macos_starts_with_the_venv_whatever_python_runs_this(self):
        for exe in ("/usr/bin/python3", "/vs/.venv/bin/python"):
            with self.subTest(exe=exe), patch("sys.executable", exe):
                self.assertEqual(
                    self.launch("darwin", "/vs/.venv/bin/python")[:3], ["/vs/.venv/bin/python", "-m", "uvicorn"]
                )

    def test_macos_without_a_venv_uses_uv(self):
        self.assertEqual(self.launch("darwin", None)[:3], ["/tools/uv", "run", "--no-sync"])

    def test_backend_health_with_requests_asks_once(self):
        requests = types.ModuleType("requests")
        requests.RequestException = type("RequestException", (OSError,), {})
        requests.get = MagicMock(side_effect=requests.RequestException("timed out"))
        with patch.dict(sys.modules, {"requests": requests}), patch("urllib.request.urlopen") as urlopen:
            self.assertIsNone(backend.health(timeout=0.1))
        requests.get.assert_called_once()
        urlopen.assert_not_called()

    def test_backend_log_path_darwin_default(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("sys.platform", "darwin"):
                log = backend.log_path({})
                self.assertEqual(
                    log,
                    pathlib.Path.home() / "Library" / "Application Support" / "OmniVoice" / "backend.log",
                )

    def test_backend_health_fallback_urllib(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"status":"ok","device":"mps"}'
        with patch.dict(sys.modules, {"requests": None}), patch("urllib.request.urlopen", return_value=response):
            self.assertEqual(backend.health(timeout=0.1), {"status": "ok", "device": "mps"})

    def test_backend_health_down(self):
        with (
            patch.dict(sys.modules, {"requests": None}),
            patch("urllib.request.urlopen", side_effect=URLError("offline")),
        ):
            h = backend.health(timeout=0.1)
            self.assertIsNone(h)

    def test_full_env_includes_homebrew_on_darwin(self):
        with (
            patch("sys.platform", "darwin"),
            patch.dict(os.environ, {"PATH": "/bin"}, clear=True),
            patch("os.path.isdir", side_effect=lambda p: p.startswith("/opt/homebrew/")),
        ):
            env = backend.full_env()
            self.assertIn("/opt/homebrew/bin", env["PATH"].split(":"))
            self.assertIn("/opt/homebrew/sbin", env["PATH"].split(":"))

    def test_stop_posix_only_signals_the_listener(self):
        with (
            patch("sys.platform", "darwin"),
            patch.object(backend, "listening_pid", side_effect=[4242, None, None]),
            patch("os.kill") as kill,
            patch("os.killpg", create=True) as killpg,
        ):
            backend.stop()
        kill.assert_called_once_with(4242, 15)
        killpg.assert_not_called()

    def test_stop_posix_escalates_only_the_same_listener(self):
        with (
            patch("sys.platform", "darwin"),
            patch.object(backend, "listening_pid", return_value=4242),
            patch("os.kill") as kill,
            patch("os.killpg", create=True) as killpg,
            patch("time.sleep"),
        ):
            backend.stop()
        self.assertEqual([c.args for c in kill.call_args_list], [(4242, 15), (4242, 9)])
        killpg.assert_not_called()

    def test_stop_posix_does_not_escalate_a_replacement_listener(self):
        with (
            patch("sys.platform", "darwin"),
            patch.object(backend, "listening_pid", side_effect=[4242, 5678, 5678]),
            patch("os.kill") as kill,
        ):
            backend.stop()
        kill.assert_called_once_with(4242, 15)


class DubAnythingDependencyTests(unittest.TestCase):
    def commands(self, platform):
        with patch("sys.platform", platform), patch.object(backend, "full_env", return_value={}):
            return runpy.run_path(str(pathlib.Path(dub_anything.__file__)))

    def test_darwin_separate_dependencies_do_not_contain_cu128(self):
        cmd_str = " ".join(self.commands("darwin")["SEPARATE"])
        self.assertNotIn("cu128", cmd_str)
        self.assertNotIn("torch==2.8.0", cmd_str)
        self.assertIn("audio-separator", cmd_str)

    def test_darwin_subs_dependencies_do_not_contain_onnxruntime_gpu(self):
        cmd_str = " ".join(self.commands("darwin")["SUBS"])
        self.assertNotIn("onnxruntime-gpu", cmd_str)
        self.assertIn("onnxruntime", cmd_str)
        self.assertIn("rapidocr", cmd_str)

    def test_windows_keeps_cuda_dependencies(self):
        commands = self.commands("win32")
        self.assertIn("https://download.pytorch.org/whl/cu128", commands["SEPARATE"])
        self.assertIn("torch==2.8.0", commands["SEPARATE"])
        self.assertIn("onnxruntime-gpu[cuda,cudnn]", commands["SUBS"])


class FramesFallbackTests(unittest.TestCase):
    def test_has_drawtext_boolean(self):
        with patch.object(frames.subprocess, "run", return_value=types.SimpleNamespace(stdout="T.C drawtext V->V")):
            self.assertTrue(frames.has_drawtext())
        with patch.object(frames.subprocess, "run", return_value=types.SimpleNamespace(stdout="scale V->V")):
            self.assertFalse(frames.has_drawtext())
        with patch.object(frames.subprocess, "run", side_effect=FileNotFoundError):
            self.assertFalse(frames.has_drawtext())


class AudioTrackLanguageTests(unittest.TestCase):
    def test_original_languages_are_preserved_by_default(self):
        for lang in ("eng", "fra", "jpn"):
            self.assertEqual(core.original_audio_language(lang, "en"), lang)

    def test_quicktime_option_only_changes_english_source_tracks(self):
        self.assertEqual(core.original_audio_language("eng", "en", True), "und")
        self.assertEqual(core.original_audio_language("fra", "en", True), "fra")
        self.assertEqual(core.original_audio_language("jpn", "ja", True), "jpn")
        self.assertEqual(core.original_audio_language("eng", "ja", True), "eng")

    def test_runner_forwards_quicktime_option_to_assembly_and_delivery(self):
        import deliver

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                job = {
                    "slug": "review",
                    "source": "source.mp4",
                    "voices": ["actor"],
                    "default": "actor",
                    "langs": ["en", "es"],
                    "quicktime_audio": enabled,
                }
                with (
                    patch.object(dub_anything, "tool", return_value=(True, "-> assembled.mp4")) as tool,
                    patch.object(deliver, "deliver", return_value=pathlib.Path("delivered.mp4")) as delivery,
                    patch.object(dub_anything, "save_job"),
                ):
                    self.assertTrue(dub_anything.deliver_job(job, []))
                self.assertEqual("--quicktime-audio" in tool.call_args.args[2], enabled)
                self.assertEqual(delivery.call_args.kwargs["quicktime_audio"], enabled)

    def test_changed_playback_option_requires_redelivery(self):
        self.assertNotEqual(dub_anything.delivery_sha({}, []), dub_anything.delivery_sha({"quicktime_audio": True}, []))


if __name__ == "__main__":
    unittest.main()
