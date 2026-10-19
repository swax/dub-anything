"""Voice-converter initialization failures must preserve CUDA recovery, and only an
AMD GPU runs the flow decoder in bf16."""

import contextlib
import io
import pathlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import vc


class ChatterboxInitializationTests(unittest.TestCase):
    def load(self, device, error, hip=None, env=None):
        torch = types.ModuleType("torch")
        torch.version = types.SimpleNamespace(hip=hip)
        torch.cuda = types.SimpleNamespace(is_available=lambda: device == "cuda")
        torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: device == "mps"), cudnn=types.SimpleNamespace(enabled=True)
        )
        torch.bfloat16 = "bf16"
        torch.autocast = lambda device_type, dtype: lambda f: ("autocast", device_type, dtype, f)
        tokenizer = types.ModuleType("chatterbox.models.s3tokenizer")
        tokenizer.S3_SR = 16000
        converter = types.ModuleType("chatterbox.vc")
        self.forward = object()
        estimator = types.SimpleNamespace(forward=self.forward)
        s3gen = types.SimpleNamespace(flow=types.SimpleNamespace(decoder=types.SimpleNamespace(estimator=estimator)))
        loaded = types.SimpleNamespace(sr=24000, device="cpu" if error else device, s3gen=s3gen)
        self.loader = MagicMock(side_effect=[error, loaded] if error else [loaded])
        converter.ChatterboxVC = types.SimpleNamespace(from_pretrained=self.loader)
        self.output = io.StringIO()
        with (
            patch.dict(
                sys.modules, {"torch": torch, "chatterbox.models.s3tokenizer": tokenizer, "chatterbox.vc": converter}
            ),
            patch.dict("os.environ", env or {}, clear=True),
            patch.object(pathlib.Path, "exists", return_value=True),
            contextlib.redirect_stdout(self.output),
        ):
            return vc.Chatterbox()

    def test_rocm_runs_the_flow_estimator_in_bf16(self):
        model = self.load("cuda", None, hip="7.13")
        self.assertEqual(model.vc.s3gen.flow.decoder.estimator.forward, ("autocast", "cuda", "bf16", self.forward))

    def test_rocm_keeps_fp32_on_request(self):
        model = self.load("cuda", None, hip="7.13", env={"DUB_ANYTHING_VC_FP32": "1"})
        self.assertIs(model.vc.s3gen.flow.decoder.estimator.forward, self.forward)

    def test_nvidia_and_mac_stay_fp32(self):
        for device in ("cuda", "mps"):
            with self.subTest(device=device):
                model = self.load(device, None)
                self.assertIs(model.vc.s3gen.flow.decoder.estimator.forward, self.forward)

    def test_rocm_build_on_the_cpu_stays_fp32(self):
        model = self.load("cpu", None, hip="7.13")
        self.assertIs(model.vc.s3gen.flow.decoder.estimator.forward, self.forward)

    def test_cuda_oom_reaches_runner(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            self.load("cuda", RuntimeError("CUDA out of memory"))
        self.loader.assert_called_once_with("cuda")

    def test_successful_cuda_load_stays_on_cuda(self):
        model = self.load("cuda", None)
        self.assertEqual(model.vc.device, "cuda")
        self.loader.assert_called_once_with("cuda")

    def test_cpu_failure_is_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, "invalid checkpoint"):
            self.load("cpu", RuntimeError("invalid checkpoint"))
        self.loader.assert_called_once_with("cpu")

    def test_mps_unsupported_operation_falls_back_with_a_message(self):
        model = self.load("mps", NotImplementedError("operation not supported on MPS"))
        self.assertEqual(model.vc.device, "cpu")
        self.assertEqual([c.args[0] for c in self.loader.call_args_list], ["mps", "cpu"])
        self.assertIn("retrying on CPU", self.output.getvalue())

    def test_mps_oom_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.load("mps", RuntimeError("MPS backend out of memory"))
        self.loader.assert_called_once_with("mps")

    def test_mps_runtime_unsupported_operation_can_fall_back(self):
        model = self.load("mps", RuntimeError("operator not currently implemented for the MPS device"))
        self.assertEqual(model.vc.device, "cpu")

    def test_unrelated_mps_failure_is_not_hidden(self):
        with self.assertRaisesRegex(RuntimeError, "invalid checkpoint"):
            self.load("mps", RuntimeError("invalid checkpoint"))
        self.loader.assert_called_once_with("mps")

    def test_missing_checkpoint_is_not_hidden(self):
        with self.assertRaises(FileNotFoundError):
            self.load("mps", FileNotFoundError("checkpoint"))
        self.loader.assert_called_once_with("mps")
