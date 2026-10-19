"""Voice conversion: give finished takes the actor's timbre, keeping their words,
accent, timing and level.

Usage: scripts/vc.py --pairs pairs.json   (dub.py generate runs it for parts rendered with
                                            "voice": "american"; own environment, see UV_RUN)
  pairs.json: [{"in": "engine take.wav", "target": "actor reference.wav", "out": "converted.wav"}, ...]

The model is Chatterbox's voice converter (Resemble AI, MIT licence): the take is
tokenized into speech units and re-synthesized with the target's voice from its
first 10 s. Delivery comes through mostly intact but not exactly (the pitch
contour is re-generated, typically a little lower), and the accent stays the
take's. The library's inaudible Perth watermark is applied.

Each output keeps the input's sample rate, channel count and exact length (the
model runs up to about 1% long; the tail is trimmed) and is level-matched to the
input (RMS over its louder frames), with a peak guard at -0.1 dBFS. It is silent
outside the input's own speech (as VoiceStudio trims takes): the model now and
then turns the silence around a take into sound, a stray "and" before the line,
which made the take longer than its window.

On an AMD GPU (a ROCm build of PyTorch) the flow-matching decoder's estimator,
ten steps that take nearly all of a take's time, runs under bf16 autocast:
fp32 matrix math is several times slower there. The vocoder and its pitch
predictor stay fp32. DUB_ANYTHING_VC_FP32=1 keeps fp32 throughout.

Setup: a clone of github.com/resemble-ai/chatterbox in vendor/chatterbox (or set
CHATTERBOX_SRC). Only its voice converter is imported: the package's own
__init__ loads the TTS models too and wants pip metadata. Weights
(ResembleAI/chatterbox, ~1 GB) download to the Hugging Face cache on first use.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import soundfile as sf

from gpu import rocm_settings, uv_torch_args
from gpulock import hold_gpu

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The environment it runs in (dub.py prepends this to "python vc.py ...").
DEPS = [
    "--with",
    "numpy<2",
    "--with",
    "scipy",
    "--with",
    "librosa==0.11.0",
    "--with",
    "soundfile",
    "--with",
    "huggingface_hub==0.25.2",
    "--with",
    "s3tokenizer",
    "--with",
    "diffusers==0.29.0",
    "--with",
    "conformer==0.3.2",
    "--with",
    "safetensors==0.5.3",
    "--with",
    "omegaconf",
    "--with",
    "einops",
    "--with",
    "resemble-perth",
    "--with",
    "setuptools<81",
    "--with",
    "tqdm",
]
ROCM_TORCH = uv_torch_args(audio=True)  # AMD: VoiceStudio's ROCm build of torch, and its torchaudio
if sys.platform == "darwin":
    UV_RUN = ["uv", "run", "--no-project", "--python", "3.11", "--with", "torch", "--with", "torchaudio"] + DEPS
elif ROCM_TORCH:
    UV_RUN = ["uv", "run", "--no-project", "--python", "3.11", *ROCM_TORCH] + DEPS
else:
    UV_RUN = [
        "uv",
        "run",
        "--no-project",
        "--python",
        "3.11",
        "--index",
        "https://download.pytorch.org/whl/cu128",
        "--index-strategy",
        "unsafe-best-match",
        "--with",
        "torch==2.8.0",
        "--with",
        "torchaudio==2.8.0",
    ] + DEPS


def resample(x, sr, to):
    if sr == to:
        return x.astype("float32")
    from math import gcd
    import scipy.signal as ss

    g = gcd(int(sr), int(to))
    return ss.resample_poly(x, int(to) // g, int(sr) // g).astype("float32")


def gate(y, x, sr, floor_db=-50.0, margin=0.05, fade=0.01):
    """Silence `y` outside `x`'s speech: its first to last sample above -50 dBFS
    plus 50 ms (VoiceStudio's trim_speech_padding), with 10 ms fades, so a
    converted take is never longer than the take it was made from."""
    voiced = np.nonzero(np.abs(x) > 10 ** (floor_db / 20))[0]
    if not len(voiced):
        return y
    a, b = max(0, voiced[0] - int(margin * sr)), min(len(y), voiced[-1] + 1 + int(margin * sr))
    w = np.zeros(len(y), "float32")
    w[a:b] = 1.0
    n = min(int(fade * sr), (b - a) // 2)
    if n:
        w[a : a + n] = np.linspace(0, 1, n)
        w[b - n : b] = np.linspace(1, 0, n)
    return y * w


def active_rms(x, sr):
    """RMS over the frames within 20 dB of the loudest, so pauses don't count."""
    hop = max(1, int(0.02 * sr))
    n = len(x) // hop
    if n == 0:
        return float(np.sqrt(np.mean(x**2) + 1e-12))
    e = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    loud = e[e > e.max() * 0.1]
    return float(np.sqrt(np.mean(loud**2)))


def flow_dtype(torch, device):
    """What the flow-matching decoder's estimator runs in under autocast: bf16 on an
    AMD GPU (above), None (fp32) on any other device or with DUB_ANYTHING_VC_FP32=1."""
    if device != "cuda" or getattr(torch.version, "hip", None) is None:
        return None
    return None if os.environ.get("DUB_ANYTHING_VC_FP32") == "1" else torch.bfloat16


class Chatterbox:
    def __init__(self):
        import types
        import torch

        src = pathlib.Path(os.environ.get("CHATTERBOX_SRC") or ROOT / "vendor" / "chatterbox")
        if not (src / "src" / "chatterbox" / "vc.py").exists():
            raise SystemExit(
                f"no Chatterbox source at {src}: git clone --depth 1 "
                "https://github.com/resemble-ai/chatterbox vendor/chatterbox (or set CHATTERBOX_SRC)"
            )
        pkg = types.ModuleType("chatterbox")
        pkg.__path__ = [str(src / "src" / "chatterbox")]
        sys.modules["chatterbox"] = pkg
        from chatterbox.models.s3tokenizer import S3_SR
        from chatterbox.vc import ChatterboxVC

        self.torch, self.S3_SR = torch, S3_SR
        rocm_settings(torch)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        try:
            self.vc = ChatterboxVC.from_pretrained(device)
        except (RuntimeError, NotImplementedError) as exc:
            # CUDA failures must reach the runner's OOM/restart handling. Only
            # unsupported MPS operations get a CPU retry; memory errors still fail.
            message = str(exc).lower()
            unsupported = isinstance(exc, NotImplementedError) or (
                "mps" in message
                and any(
                    s in message
                    for s in ("not implemented", "not currently implemented", "not supported", "unsupported")
                )
            )
            if device != "mps" or not unsupported or "out of memory" in message:
                raise
            print(f"  Chatterbox: MPS initialization failed ({exc}); retrying on CPU", flush=True)
            self.vc = ChatterboxVC.from_pretrained("cpu")
        if dtype := flow_dtype(torch, self.vc.device):
            estimator = self.vc.s3gen.flow.decoder.estimator
            estimator.forward = torch.autocast("cuda", dtype=dtype)(estimator.forward)
        self.sr = self.vc.sr
        self.target = None

    def convert(self, x, sr, target):
        if self.target != target:
            self.vc.set_target_voice(target)
            self.target = target
        with self.torch.inference_mode():
            audio = self.torch.from_numpy(resample(x, sr, self.S3_SR)).float().to(self.vc.device)[None]
            tokens, _ = self.vc.s3gen.tokenizer(audio)
            wav, _ = self.vc.s3gen.inference(speech_tokens=tokens, ref_dict=self.vc.ref_dict)
            wav = wav.squeeze(0).detach().cpu().numpy()
            wav = self.vc.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return np.asarray(wav, dtype="float32"), self.sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pairs = json.loads(pathlib.Path(args.pairs).read_text(encoding="utf-8"))
    t0 = time.time()
    hold_gpu()  # shared with dub.py when it starts this
    model = Chatterbox()
    print(f"  voice conversion: Chatterbox loaded in {time.time() - t0:.0f} s, {len(pairs)} takes", flush=True)
    for p in pairs:
        x, sr = sf.read(p["in"], dtype="float32", always_2d=True)
        ch, subtype, x = x.shape[1], sf.info(p["in"]).subtype, x.mean(1)
        y, ysr = model.convert(x, sr, p["target"])
        y = resample(y, ysr, sr)
        n = len(x)
        y = y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), "float32")])
        y = gate(y * (active_rms(x, sr) / max(active_rms(y, sr), 1e-6)), x, sr)
        peak = float(np.abs(y).max())
        if peak > 0.989:
            y = y * (0.989 / peak)
        sf.write(
            p["out"],
            np.repeat(y[:, None], ch, 1) if ch > 1 else y,
            sr,
            subtype=subtype if subtype.startswith("PCM") else "PCM_16",
        )
    print(f"  voice conversion: {len(pairs)} takes in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
