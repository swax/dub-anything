"""Which PyTorch build runs the models, and what an AMD GPU needs from it.

VoiceStudio's virtualenv decides: a CUDA build (NVIDIA), a ROCm build (AMD; on
Windows AMD's own wheels, docs/setup.md) or a CPU/MPS one. Tools in their own
uv environments (separate.py) install the same ROCm build, read from that venv
without importing it.

With a ROCm build, every process gets ROCM_ENV (rocm_env: backend.full_env()
for the tools it starts, core.py for the scripts themselves):
- AOTriton attention (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1). PyTorch
  calls it experimental on consumer Radeons and otherwise takes the math path,
  about 10x slower.
- KMP_DUPLICATE_LIB_OK=TRUE. ROCm's torch brings LLVM's OpenMP runtime and
  CTranslate2 (faster-whisper) Intel's, and Intel's aborts the process when it
  starts after the other ("OMP: Error #15"). CUDA builds share Intel's.
And a process about to run a model takes rocm_settings(torch): PyTorch's own
convolutions instead of MIOpen's. MIOpen searches and compiles kernels for
every new input shape, which can take minutes, and each clip has a new length.
DUB_ANYTHING_MIOPEN=1 keeps it. VoiceStudio's backend does the same
(patches/voicestudio-local-fixes.patch). Standard library only, so other
environments can import it.
"""

import os
import re
import sys

ROCM_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"
ROCM_ENV = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"}


def _site_packages():
    from backend import VS_DIR

    venv = VS_DIR / ".venv"
    if sys.platform == "win32":
        return venv / "Lib" / "site-packages"
    return next(iter(sorted(venv.glob("lib/python3*/site-packages"))), venv / "lib" / "site-packages")


def _installed(site, name):
    """(version, build) of a package in a site-packages folder, from its record
    ("2.9.1", "rocm7.13.0"); None when it isn't there."""
    for d in site.glob(f"{name}-*.dist-info"):
        if m := re.fullmatch(rf"{name}-([^+]+)\+?(.*)\.dist-info", d.name):
            return m.groups()
    return None


def vs_torch():
    """VoiceStudio's torch as (version, build, gfx), e.g. ("2.9.1", "rocm7.13.0",
    "gfx1150") or ("2.8.0", "cu128", None), from its venv's package records;
    None without one. gfx is the GPU the ROCm build's device kernels are for (the
    most specific of AMD's amd-torch-device-* packages)."""
    site = _site_packages()
    torch = _installed(site, "torch")
    if not torch:
        return None
    gfx = [
        m.group(1)
        for d in site.glob("amd_torch_device_gfx*")
        if (m := re.match(r"amd_torch_device_(gfx\w+?)-", d.name))
    ]
    return *torch, max(gfx, key=len) if gfx else None


def rocm_torch():
    """VoiceStudio's ROCm build as (version, build, gfx), else None (NVIDIA, Apple or CPU)."""
    t = vs_torch()
    return t if t and t[1].startswith("rocm") and t[2] else None


def rocm_env(env=os.environ):
    """Add ROCM_ENV to `env` (this process's own by default) with a ROCm build;
    nothing changes with any other."""
    if rocm_torch():
        for k, v in ROCM_ENV.items():
            env.setdefault(k, v)
    return env


def uv_torch_args(audio=False):
    """uv run arguments that install VoiceStudio's ROCm torch (and its torchvision,
    which would otherwise resolve to one built for a newer torch) in a tool's own
    environment, or None when VoiceStudio's torch isn't a ROCm build. With `audio`,
    AMD's torchaudio for that torch too (X.Y.0 for torch X.Y.*, as AMD's index has
    it up to 2.11; VoiceStudio itself keeps torchaudio 2.8, docs/setup.md)."""
    t = rocm_torch()
    if not t:
        return None
    version, build, gfx = t
    args = ["--index", ROCM_INDEX, "--index-strategy", "unsafe-best-match"]
    args += ["--with", f"torch[device-{gfx}]=={version}+{build}"]
    vision = _installed(_site_packages(), "torchvision")
    if vision and vision[1] == build:
        args += ["--with", f"torchvision[device-{gfx}]=={vision[0]}+{build}"]
    if audio:
        args += ["--with", f"torchaudio=={'.'.join(version.split('.')[:2])}.0+{build}"]
    return args


def rocm_settings(torch):
    """What a process about to run a model needs from a ROCm build (above);
    nothing on other builds."""
    if getattr(torch.version, "hip", None) is None:
        return
    for k, v in ROCM_ENV.items():
        os.environ.setdefault(k, v)
    if os.environ.get("DUB_ANYTHING_MIOPEN") != "1":
        torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    print(vs_torch(), uv_torch_args())
