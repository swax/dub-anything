"""VoiceStudio's PyTorch build, read from its venv, and what a ROCm build changes."""

import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import gpu


class VoiceStudioTorchTests(unittest.TestCase):
    def site(self, *records):
        """A site-packages folder holding only these package records."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        site = pathlib.Path(tmp.name)
        for name in records:
            (site / name).mkdir()
        self.enterContext(patch.object(gpu, "_site_packages", return_value=site))

    def test_rocm_build_and_its_device(self):
        self.site(
            "torch-2.9.1+rocm7.13.0.dist-info",
            "torchvision-0.24.0+rocm7.13.0.dist-info",
            "amd_torch_device_gfx11-2.9.1+rocm7.13.0.dist-info",
            "amd_torch_device_gfx1150-2.9.1+rocm7.13.0.dist-info",
            "torch_audiomentations-0.12.0.dist-info",
        )
        self.assertEqual(gpu.vs_torch(), ("2.9.1", "rocm7.13.0", "gfx1150"))
        args = gpu.uv_torch_args()
        self.assertIn(gpu.ROCM_INDEX, args)
        self.assertIn("torch[device-gfx1150]==2.9.1+rocm7.13.0", args)
        self.assertIn("torchvision[device-gfx1150]==0.24.0+rocm7.13.0", args)
        self.assertFalse(any(a.startswith("torchaudio") for a in args))
        self.assertIn("torchaudio==2.9.0+rocm7.13.0", gpu.uv_torch_args(audio=True))

    def test_cuda_and_cpu_builds_change_nothing(self):
        for record in ("torch-2.8.0+cu128.dist-info", "torch-2.8.0.dist-info"):
            with self.subTest(record=record):
                self.site(record)
                self.assertIsNone(gpu.rocm_torch())
                self.assertIsNone(gpu.uv_torch_args())
                env = {}
                gpu.rocm_env(env)
                self.assertEqual(env, {})

    def test_no_venv(self):
        self.site()
        self.assertIsNone(gpu.vs_torch())
        self.assertIsNone(gpu.uv_torch_args())

    def test_rocm_env_keeps_what_the_user_set(self):
        self.site("torch-2.9.1+rocm7.13.0.dist-info", "amd_torch_device_gfx1100-2.9.1+rocm7.13.0.dist-info")
        env = gpu.rocm_env({"KMP_DUPLICATE_LIB_OK": "FALSE"})
        self.assertEqual(env["KMP_DUPLICATE_LIB_OK"], "FALSE")
        self.assertEqual(env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"], "1")

    def test_rocm_settings_turn_miopen_off_on_rocm_only(self):
        for hip, expected in (("7.13", False), (None, True)):
            with self.subTest(hip=hip):
                torch = types.SimpleNamespace(
                    version=types.SimpleNamespace(hip=hip),
                    backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(enabled=True)),
                )
                with patch.dict("os.environ", {}, clear=True):
                    gpu.rocm_settings(torch)
                self.assertIs(torch.backends.cudnn.enabled, expected)


if __name__ == "__main__":
    unittest.main()
