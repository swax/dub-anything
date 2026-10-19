# Setup for agents

**If you're a human, start with the [README](../README.md#getting-started).**
Ask your desktop AI coding app to dub your movie or show; it handles setup for you.

Agents: follow this guide to install and verify any missing prerequisites before
starting the dub. Reuse working installations and handle routine setup issues
yourself. Ask the user when a step needs their sign-in, license acceptance or
permissions you don't have.

Dub Anything needs a running [VoiceStudio](https://github.com/debpalash/VoiceStudio)
backend, its models, and ffmpeg.

## 1. Tools

### macOS (Apple Silicon or Intel)
- macOS 13.3+ (Ventura or newer). On Apple Silicon (M1/M2/M3/M4), Metal (MPS) acceleration is used automatically.
- Install tools via Homebrew:
  ```bash
  brew install git uv ffmpeg
  ```
  *(Note: Standard Homebrew `ffmpeg` is sufficient. Dub Anything includes automatic PIL fallbacks for casting previews when FFmpeg is built without `drawtext`.)*
- In the clone, turn on pre-commit hooks and git blame filters:
  ```bash
  git config core.hooksPath .githooks
  git config blame.ignoreRevsFile .git-blame-ignore-revs
  ```

### Windows (NVIDIA)
- An NVIDIA GPU with ≥ 6 GB of VRAM. `nvidia-smi` should work.
- NVIDIA driver ≥ 610 if you want `scripts/split.py --encoder h264_nvenc` (current
  FFmpeg builds need NVENC API 13.1). Otherwise use the default x264.
- `winget install Git.Git astral-sh.uv Gyan.FFmpeg`, then open a new terminal.
- In the clone, `git config core.hooksPath .githooks` turns on the pre-commit
  check (ruff format and lint) and `git config blame.ignoreRevsFile
  .git-blame-ignore-revs` keeps formatting commits out of `git blame`.

### Windows (AMD Radeon)
- A Radeon that AMD's PyTorch wheels for Windows support (RDNA 3, 3.5 and 4,
  the Ryzen AI 300 and Max iGPUs included) and a current Adrenalin driver. Run
  as for NVIDIA, with PyTorch swapped for AMD's ROCm build after `uv sync`
  (section 3). Integrated GPUs share system memory: 32 GB of RAM or more.
- Tools as for NVIDIA (`winget install Git.Git astral-sh.uv Gyan.FFmpeg`).
  `split.py` encodes with x264, and Whisper runs on the CPU: CTranslate2
  (faster-whisper) has no AMD backend.

## 2. Models and Data Directories

### macOS
By default, Dub Anything and VoiceStudio place data in:
- Data / dub jobs: `~/Library/Application Support/OmniVoice/dub_jobs`
- Model caches: `~/Library/Caches/OmniVoice/hf_cache`

To customize these or keep them on a specific drive, add to your `~/.zshrc`:
```bash
export OMNIVOICE_DATA_DIR="$HOME/VoiceStudioData/data"
export OMNIVOICE_CACHE_DIR="$HOME/VoiceStudioData/models"
```

### Windows
Put VoiceStudio's data, its model cache and uv's Python installs somewhere
neutral. The VoiceStudio GUI can share them, and they survive sandboxed shells.
Some sandboxed agent apps (the Claude desktop app, for one) redirect AppData
writes, which breaks `uv sync` ("Missing expected target directory for Python
minor version link").
In PowerShell, then open a new terminal:

```powershell
$d = "C:\VoiceStudioData"
[Environment]::SetEnvironmentVariable("OMNIVOICE_DATA_DIR", "$d\data", "User")
[Environment]::SetEnvironmentVariable("OMNIVOICE_CACHE_DIR", "$d\models", "User")
[Environment]::SetEnvironmentVariable("UV_PYTHON_INSTALL_DIR", "$d\uv\python", "User")
[Environment]::SetEnvironmentVariable("UV_CACHE_DIR", "$d\uv\cache", "User")
[Environment]::SetEnvironmentVariable("HF_TOKEN_PATH", "$env:USERPROFILE\.cache\huggingface\token", "User")
```

Dub Anything's scripts find VoiceStudio's jobs through `OMNIVOICE_DATA_DIR` (or system defaults), so set it in any shell that runs them if you use a custom path.

## 3. VoiceStudio, pinned, with local fixes

### macOS
VoiceStudio can be installed either via the official desktop application or by checking out the source:

#### Option A: Official VoiceStudio Desktop App (Recommended)
1. Download and install `VoiceStudio-Electron-*-mac-arm64.dmg` from the VoiceStudio releases.
2. Launch VoiceStudio once so it initializes its runtime at `~/Library/Application Support/VoiceStudio/runtime/project`.
Dub Anything automatically detects this installation, its runtime backend, and its bundled virtualenv.

#### Option B: From Source
```bash
git clone https://github.com/debpalash/VoiceStudio.git ~/python/VoiceStudio
git -C ~/python/VoiceStudio checkout -b local-fixes c52618af
git -C ~/python/VoiceStudio apply <path-to-dub-anything>/patches/voicestudio-local-fixes.patch
cd ~/python/VoiceStudio && uv sync
uv run python backend/main.py --diagnose
```
The scripts find a source checkout only beside this repo (or the desktop app's
runtime). For one elsewhere, like this one, add to your `~/.zshrc`:
```bash
export VOICESTUDIO_DIR="$HOME/python/VoiceStudio"
```
*(On macOS, PyTorch on Apple Silicon uses Metal / MPS acceleration; `scripts/setup.py` cuDNN 8 installation is Windows-only).*

### Windows
```powershell
git clone https://github.com/debpalash/VoiceStudio.git C:\git\VoiceStudio
git -C C:\git\VoiceStudio checkout -b local-fixes c52618af
git -C C:\git\VoiceStudio apply <path-to-dub-anything>\patches\voicestudio-local-fixes.patch
cd C:\git\VoiceStudio; uv sync; uv run --no-sync python scripts/setup.py
uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"   # True
```

`uv sync` installs the CUDA 12.8 PyTorch build on Windows. `scripts/setup.py`
adds the cuDNN 8 libraries that faster-whisper needs next to PyTorch's cuDNN 9.
The patch fixes three bugs. The dubbing transcriber ignored the source-language
hint. OmniVoice ran in float16 on CPU (about 7× slower). And a render that
stopped at a timing overflow threw its takes away: the next attempt
re-synthesized every line instead of only the changed ones. It also adds two
general-purpose voice presets, `echo` for flashbacks and `pa` for
announcements over a public-address system, and lets reverb ring out instead of
stopping dead where the dry take ends. And it lets each dub line carry its own
voice reference (`scripts/dub.py generate --line-ref`), and its own emotion
reference for engines that take one (IndexTTS 2,
`--engine indextts2 --emo-alpha`). For IndexTTS 2.5 it passes a line's speed on as the
engine's duration factor (it was dropped before), and it gives each line a seed,
so the `"american"` voice's fill re-times the same take. With ROCm's PyTorch
on Windows (an AMD GPU, below) it turns MIOpen off and AOTriton attention on,
in the backend and in IndexTTS's sidecar, and fits IndexTTS to that torch. See
[directing.md](directing.md#gotchas).

#### AMD Radeon: ROCm PyTorch

`uv sync` installs the CUDA build, which runs on the CPU here. Swap in AMD's
ROCm build for your GPU's architecture (`gfx1150` for a Radeon 890M or 880M,
`gfx1151` for Ryzen AI Max, `gfx1100`/`gfx1101`/`gfx1102` for an RX 7900/7800/7600,
`gfx1200`/`gfx1201` for an RX 9060/9070):

```powershell
$py = "C:\git\VoiceStudio\.venv\Scripts\python.exe"; $gfx = "gfx1150"
uv pip install --python $py --index-url https://repo.amd.com/rocm/whl-multi-arch/ --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch[device-$gfx]==2.9.1+rocm7.13.0" "torchvision[device-$gfx]==0.24.0+rocm7.13.0"
uv pip install --python $py --no-deps --reinstall-package torchaudio "torchaudio==2.8.0" --index-url https://pypi.org/simple
$lib = "C:\git\VoiceStudio\.venv\Lib\site-packages\torchaudio\lib"
foreach ($f in "_torchaudio.pyd", "libtorchaudio.pyd") { Rename-Item "$lib\$f" "$f.disabled-for-rocm" }
& $py -c "import torch, torchaudio, pyannote.audio; print(torch.version.hip, torch.cuda.get_device_name(0))"
```

Torch 2.9.1 is the oldest ROCm build for Windows. Its torchaudio (2.9) no
longer has `torchaudio.info` and `list_audio_backends`, which pyannote 3.x
needs, so torchaudio stays at 2.8. Its compiled extension doesn't load against
torch 2.9, and without it torchaudio is plain Python (soundfile I/O,
resampling, filterbanks), all VoiceStudio and the scripts use. `uv sync` puts
the CUDA build back; the backend runs with `uv run --no-sync`.

`scripts/gpu.py` reads the build from the venv: `separate.py`'s environment
installs the same one, and every process gets three settings ROCm on Windows
needs (the patch makes the first two in the backend). AOTriton attention
(`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`): PyTorch calls it experimental
on consumer Radeons and otherwise uses its math attention, which made
OmniVoice twice as slow on a Radeon 890M. PyTorch's own convolutions instead
of MIOpen's: MIOpen searches and compiles kernels for each new input shape, so
the first OmniVoice take sat over 5 minutes in one convolution, and every take
has a new length. And `KMP_DUPLICATE_LIB_OK=TRUE`: ROCm's torch brings LLVM's
OpenMP runtime, CTranslate2 brings Intel's, and Intel's aborts a script that
uses both ("OMP: Error #15"); CUDA's torch uses Intel's too. A test process that
ran new kernels for the first time sometimes stayed busy at exit for over an
hour; if a step's log is complete but the step doesn't end, end that process.

IndexTTS 2.5 is optional (about 12 GB). Install it through the running backend:
`POST http://127.0.0.1:3900/engines/sidecar/indextts2/install`, then poll
`.../install/status`.

With an AMD GPU, IndexTTS's own venv gets the CUDA build too. Swap it the same
way (it has no torchvision), then `scripts/backend.py restart`:

```powershell
$py = "C:\VoiceStudioData\data\engines\indextts2\index-tts-2.5\.venv\Scripts\python.exe"   # under OMNIVOICE_DATA_DIR
uv pip install --python $py --index-url https://repo.amd.com/rocm/whl-multi-arch/ --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch[device-$gfx]==2.9.1+rocm7.13.0"
uv pip install --python $py --no-deps --reinstall-package torchaudio "torchaudio==2.8.0" --index-url https://pypi.org/simple
$lib = "$(Split-Path (Split-Path $py))\Lib\site-packages\torchaudio\lib"
foreach ($f in "_torchaudio.pyd", "libtorchaudio.pyd") { Rename-Item "$lib\$f" "$f.disabled-for-rocm" }
```

The backend log should then say `IndexTTS venv resolved from
OMNIVOICE_INDEXTTS_DIR`. When its import probe of that venv fails, the backend
quietly builds a CUDA venv of its own in `backend\engines\indextts\.venv`,
which runs on the CPU (IndexTTS logs "CPU mode", about a minute a line). The
patch makes the probe and the sidecar fit this torch, which has no distributed
backend: transformers 4.52 imports `torch.distributed.tensor` unguarded, and
audiotools names `dist.ReduceOp` as a default argument. It also turns
TorchScript's fuser off, whose bf16 kernels corrupt the heap (the process ends
with 0xC0000374), and runs IndexTTS's s2mel diffusion in bf16 instead of
upstream's fp32. On a Radeon 890M that is about 10 s a line against 44 s, with
an output 0.3% different; `OMNIVOICE_INDEXTTS_FP16=0` keeps fp32 throughout.
`vc.py` (the `"american"` voice) installs AMD's torchaudio 2.9 for its torch.
Chatterbox only resamples and takes filterbanks with it.

## 4. Start the backend

### macOS
From this repo:
```bash
python3 scripts/backend.py start
```
Or directly from VoiceStudio:
```bash
cd ~/python/VoiceStudio && uv run --no-sync uvicorn main:app --app-dir backend --host 127.0.0.1 --port 3900
```

### Windows
From this repo, with VoiceStudio's Python:

```powershell
& C:\git\VoiceStudio\.venv\Scripts\python.exe scripts\backend.py start
```

It starts the backend detached (it keeps running after the terminal or the
agent's session closes), logs to `backend.log` beside the data folder, and
waits until `http://127.0.0.1:3900/health` returns
`{"status":"ok","device":"cuda (...)"}` (or `"device":"mps"` on Apple Silicon). `scripts/backend.py status`, `stop` and
`restart` do what they say; `scripts/dub_anything.py run` starts it when it's down.
Or run it yourself in a terminal you leave open:

```powershell
cd C:\git\VoiceStudio; uv run --no-sync uvicorn main:app --app-dir backend --host 127.0.0.1 --port 3900
```

Restart it after an NVIDIA driver update. A backend that was running across the
update reports "CUDA out of memory" on an empty GPU and falls back to the CPU,
while `/health` still says `cuda`.

## 5. Models (~15 GB)

### macOS
```bash
api="http://127.0.0.1:3900"
for r in "k2-fsa/OmniVoice" "Systran/faster-whisper-large-v3" "audio-cpp/audio.cpp-gguf"; do
  curl -s -X POST "$api/models/install" -H "Content-Type: application/json" -d "{\"repo_id\":\"$r\"}"
done
curl -s -X POST "$api/engines/audiocpp/runtime/install"
curl -s "$api/models/install/status"
```

### Windows
```powershell
$api = "http://127.0.0.1:3900"
foreach ($r in "k2-fsa/OmniVoice", "Systran/faster-whisper-large-v3", "audio-cpp/audio.cpp-gguf") {
  Invoke-RestMethod -Method Post "$api/models/install" -ContentType application/json -Body (@{repo_id=$r} | ConvertTo-Json) }
Invoke-RestMethod -Method Post "$api/engines/audiocpp/runtime/install"      # Sortformer runtime (optional)
Invoke-RestMethod "$api/models/install/status"                               # progress
```

**pyannote** (speaker diarisation, needed for clips longer than 2 minutes). Accept
the licenses of `pyannote/speaker-diarization-3.1` and
`pyannote/segmentation-3.0` on Hugging Face. Then log in with a read token and
install it:

#### macOS
```bash
hf auth login   # or hf login
curl -s -X POST "http://127.0.0.1:3900/models/install" -H "Content-Type: application/json" -d '{"repo_id":"pyannote/speaker-diarization-3.1"}'
curl -s -X POST "http://127.0.0.1:3900/engines/diarisation/select" -H "Content-Type: application/json" -d '{"engine_id":"pyannote"}'
```

#### Windows
```powershell
C:\git\VoiceStudio\.venv\Scripts\hf.exe auth login
Invoke-RestMethod -Method Post "$api/models/install" -ContentType application/json -Body '{"repo_id":"pyannote/speaker-diarization-3.1"}'
Invoke-RestMethod -Method Post "$api/engines/diarisation/select" -ContentType application/json -Body '{"engine_id":"pyannote"}'
```

Installing the audio.cpp bundle can switch the active diarisation engine to
Sortformer, so select pyannote again afterwards. Sortformer is limited to 120 s
clips and 4 speakers.

If the audio.cpp runtime install fails with `PermissionError ... Access is denied`
on the final rename, Defender was scanning the new DLLs. Post the install again.

**The `"american"` voice** (optional) needs IndexTTS 2.5 (above) and Chatterbox's
voice converter, which `vc.py` runs in its own uv environment. Clone its source
into the repo's `vendor` folder (git-ignored):

```powershell
git clone --depth 1 https://github.com/resemble-ai/chatterbox vendor\chatterbox
```

The weights (`ResembleAI/chatterbox`, about 1 GB) download into the Hugging Face
cache (`HF_HUB_CACHE`, else `OMNIVOICE_CACHE_DIR`) on the first conversion.

## Playback in QuickTime

Original audio keeps its language tags by default. If QuickTime selects an
English original instead of the default dub, add `--quicktime-audio` when
creating the job with `scripts/dub_anything.py new`. This optional workaround
tags retained English original audio as `und` while preserving other languages.
For manual assembly and delivery, pass `--quicktime-audio` to both
`scripts/assemble.py` and `scripts/deliver.py`.

## 6. Check

### macOS
```bash
python3 scripts/dub.py prep <some short clip>.mp4      # prints a Japanese transcript
```

### Windows
```powershell
cd <path-to-dub-anything>
$py = "C:\git\VoiceStudio\.venv\Scripts\python.exe"
& $py scripts\dub.py prep <some short clip>.mp4      # prints a Japanese transcript
```

Five tools run in their own throwaway uv environments, so they can't disturb
VoiceStudio's:
- `separate.py`: audio-separator (on macOS with PyPI torch; on Windows with CUDA 12.8 PyTorch). Its Roformer model
  (about 900 MB) downloads on first use into the model cache (`audio-separator`).
- `second.py`: sherpa-onnx (its models, about 850 MB for Japanese speech,
  1 GB for English, 900 MB for Spanish and 320 MB for Korean, download into the
  model cache on first use).
- `subs.py`: RapidOCR (its Korean recogniser downloads on first use, for a
  Korean source; uses standard onnxruntime on macOS, onnxruntime-gpu on Windows).
- `vc.py`: Chatterbox, which `scripts/dub.py generate` starts itself for the
  `"american"` voice.
- `languages/kana.py`: pykakasi, which `lineqa.py` starts itself to read takes in
  Japanese as kana (dubs into Japanese).

The commands are in [AGENTS.md](../AGENTS.md).

Voice profiles are rebuilt automatically from `work/voices.json` on first use.
Profile ids are machine-specific and cached in `work/profiles.json`.
