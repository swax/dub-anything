# Setup for agents

**If you're a human, start with the [README](../README.md#getting-started).**
Ask your desktop AI coding app to dub your movie or show; it handles setup for you.

Agents: follow this guide to install and verify any missing prerequisites before
starting the dub. Reuse working installations and handle routine setup issues
yourself. Ask the user when a step needs their sign-in, license acceptance or
permissions you don't have.

Dub Anything needs a running [VoiceStudio fork](https://github.com/swax/VoiceStudio)
backend, its models, and ffmpeg. The fork is pinned as the `vendor/VoiceStudio`
Git submodule; setup initializes it automatically.

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

### Linux (NVIDIA), and Windows through WSL2
- An NVIDIA GPU with ≥ 6 GB of VRAM and its driver. `nvidia-smi` should work.
  Checked on Ubuntu 22.04 under WSL2, where the Windows NVIDIA driver provides
  CUDA and NVENC inside Linux: don't install a Linux NVIDIA driver in WSL.
- `sudo apt install git ffmpeg psmisc build-essential` and uv:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`. psmisc has `fuser`,
  which `scripts/backend.py` uses when `lsof` is missing. The compiler builds
  audio-separator's `diffq` for `separate.py` (section 6). Ubuntu's ffmpeg has
  `h264_nvenc`, which `split.py` picks when the driver takes it.
- For dubs into Japanese, Korean or Chinese, `sudo apt install fonts-noto-cjk`:
  subtitle styles name Windows fonts, and fontconfig substitutes one that has
  the characters (in signs previews, and in players on this machine).
- In WSL, keep the clones, `videos/` and the data and model folders in the
  Linux filesystem (`~/`), not under `/mnt/c`: reading the Windows drive from
  WSL is many times slower (about 25 MB/s copying models), and a dub reads and
  writes gigabytes of audio.
- In WSL a missing command fails with "Permission denied" rather than "not
  found": WSL appends the Windows PATH, and some of its folders can't be
  searched from Linux. "Permission denied: 'cc'" means there is no compiler.
  The backend log's "torch.compile disabled ... Permission denied: 'nvcc'"
  is harmless: without the CUDA toolkit OmniVoice runs uncompiled, as on Windows.
- In the clone, `git config core.hooksPath .githooks` and
  `git config blame.ignoreRevsFile .git-blame-ignore-revs`, as on the others.

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

### Linux
By default VoiceStudio keeps its data in `~/.omnivoice` (dub jobs in
`~/.omnivoice/dub_jobs`) and its models in Hugging Face's own cache
(`~/.cache/huggingface/hub`); the scripts' own models, like `separate.py`'s,
go in `~/.cache/OmniVoice`. To put them elsewhere, add to your `~/.bashrc`:
```bash
export OMNIVOICE_DATA_DIR="$HOME/VoiceStudioData/data"
export OMNIVOICE_CACHE_DIR="$HOME/VoiceStudioData/models"
```
In WSL, models already downloaded on the Windows side can be copied over
instead of downloaded again: the `models--*` folders of the Windows cache
(`C:\VoiceStudioData\models` above, `/mnt/c/VoiceStudioData/models` in WSL) go
into the Hugging Face cache as they are, and `audio-separator/` into the
scripts' folder (with `OMNIVOICE_CACHE_DIR` set, both are that folder).

Dub Anything's scripts find VoiceStudio's jobs through `OMNIVOICE_DATA_DIR` (or system defaults), so set it in any shell that runs them if you use a custom path.

These locations have different retention rules: data contains job media and
voice profiles; models contains reusable downloads; uv's cache contains
regenerable packages and tool environments, while its Python directory contains
the interpreters those tools need. Follow
[storage cleanup](directing/package.md#7-storage-cleanup) after each accepted job. Retire
the job media, including timestamped backend exports, before considering model
removal. On Windows, account for hard links rather than adding folder sizes.

## 3. VoiceStudio, pinned fork

From the Dub Anything repository root, with Git, uv and a standard Python 3.11+
installed, run:

### Windows
```powershell
python scripts/setup.py
& .\vendor\VoiceStudio\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"   # True on NVIDIA
```

### macOS and Linux
```bash
python3 scripts/setup.py
vendor/VoiceStudio/.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # True on NVIDIA
```

Setup runs `git submodule update --init --recursive` for `vendor/VoiceStudio`,
then `uv sync --locked` and VoiceStudio's own prerequisite setup in that checkout.
It checks out the exact commit recorded by Dub Anything, including nested
submodules. A normal Git clone is enough; `git clone --recurse-submodules` also
works. Use a Git clone rather than GitHub's source ZIP, which omits submodules
and Git metadata. Setup installs Python dependencies, not model weights.

For source initialization alone, run `python scripts/setup.py --init-only`
(`python3` on macOS/Linux). This also updates an existing submodule to the pin
after pulling Dub Anything. It does not use `--remote` or force local changes.
Use this option when keeping a customized GPU environment: a full setup's
`uv sync` restores the default torch packages.

The backend checks `VOICESTUDIO_DIR` first, then the bundled fork, then a macOS
desktop runtime, then a sibling `VoiceStudio` checkout. Existing installations
still work as fallbacks. To deliberately keep using one after initializing the
submodule, set `VOICESTUDIO_DIR` to its path; full setup also installs dependencies
there. The bundled fork is the default and includes the changes this pipeline
needs; an unmodified desktop release may lack them. Data and model directories
stay in the locations from section 2, so they can be shared with an existing
installation.

On macOS, PyTorch on Apple Silicon uses Metal / MPS acceleration. On Windows and
Linux, `uv sync` installs CUDA 12.8 PyTorch. VoiceStudio's `scripts/setup.py` adds
the cuDNN 8 libraries that faster-whisper needs next to PyTorch's cuDNN 9. The
scripts load them before Whisper runs (`core.whisper_setup`); on Linux without
them, Whisper on the GPU fails with "Could not load library libcudnn_ops_infer.so.8".

The fork forwards the source-language hint during dubbing transcription and
adds per-line voice references (`scripts/dub.py generate --line-ref`), emotion
references for engines that support them, and IndexTTS seeds. These controls
also participate in take fingerprints. Its `echo` and `pa` effects preserve
their reverb tails. CPU float32 selection and IndexTTS duration/speed control
are now supplied by upstream.

The integration retains upstream's atomic take publication. A successful
render publishes reusable takes; failed or cancelled attempts cannot replace
them. The old patch's reuse of newly made takes after a failed first render
was retired, so resolve predicted overflows before rendering. Windows AMD
adjustments apply only to the verified legacy `2.9.1+rocm7.13.x` Torch build;
other builds retain upstream's dispatch and precision policy. See
[the machine gotchas](directing/machine.md).

#### AMD Radeon: ROCm PyTorch

`uv sync` installs the CUDA build, which runs on the CPU here. Swap in AMD's
ROCm build for your GPU's architecture (`gfx1150` for a Radeon 890M or 880M,
`gfx1151` for Ryzen AI Max, `gfx1100`/`gfx1101`/`gfx1102` for an RX 7900/7800/7600,
`gfx1200`/`gfx1201` for an RX 9060/9070):

```powershell
$py = ".\vendor\VoiceStudio\.venv\Scripts\python.exe"; $gfx = "gfx1150"
uv pip install --python $py --index-url https://repo.amd.com/rocm/whl-multi-arch/ --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch[device-$gfx]==2.9.1+rocm7.13.0" "torchvision[device-$gfx]==0.24.0+rocm7.13.0"
uv pip install --python $py --no-deps --reinstall-package torchaudio "torchaudio==2.8.0" --index-url https://pypi.org/simple
$lib = ".\vendor\VoiceStudio\.venv\Lib\site-packages\torchaudio\lib"
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
needs (the fork sets the first two in the backend). AOTriton attention
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
fork makes the probe and the sidecar fit this torch, which has no distributed
backend: transformers 4.52 imports `torch.distributed.tensor` unguarded, and
audiotools names `dist.ReduceOp` as a default argument. It also turns
TorchScript's fuser off, whose bf16 kernels corrupt the heap (the process ends
with 0xC0000374), and runs IndexTTS's s2mel diffusion in bf16 instead of
upstream's fp32. On a Radeon 890M that is about 10 s a line against 44 s, with
an output 0.3% different; `OMNIVOICE_INDEXTTS_FP16=0` keeps fp32 throughout.
`vc.py` (the `"american"` voice) installs AMD's torchaudio 2.9 for its torch.
Chatterbox only resamples and takes filterbanks with it. `vc.py` runs
Chatterbox's flow-matching decoder (its ten steps take nearly all of a take's
time) under bf16 autocast, and the vocoder in fp32. On a Radeon 890M that is
2.7 s a take against 7.2 s, and the converted take moves about 6% as far from
the fp32 one as a fresh seed does (log-mel distance), with the same speaker
similarity, pitch and transcriptions. `DUB_ANYTHING_VC_FP32=1` keeps fp32.

## 4. Start the backend

### macOS
From this repo:
```bash
python3 scripts/backend.py start
```
Or directly from VoiceStudio:
```bash
cd vendor/VoiceStudio && uv run --no-sync uvicorn main:app --app-dir backend --host 127.0.0.1 --port 3900
```

### Windows
From this repo, with VoiceStudio's Python:

```powershell
& .\vendor\VoiceStudio\.venv\Scripts\python.exe scripts\backend.py start
```

It starts the backend detached (it keeps running after the terminal or the
agent's session closes), logs to `backend.log` beside the data folder, and
waits until `http://127.0.0.1:3900/health` returns
`{"status":"ok","device":"cuda (...)"}` (or `"device":"mps"` on Apple Silicon). `scripts/backend.py status`, `stop` and
`restart` do what they say; `scripts/dub_anything.py run` starts it when it's down.
Or run it yourself in a terminal you leave open:

```powershell
cd .\vendor\VoiceStudio; uv run --no-sync uvicorn main:app --app-dir backend --host 127.0.0.1 --port 3900
```

Restart it after an NVIDIA driver update. A backend that was running across the
update reports "CUDA out of memory" on an empty GPU and falls back to the CPU,
while `/health` still says `cuda`.

### Linux
```bash
vendor/VoiceStudio/.venv/bin/python scripts/backend.py start
```
As on Windows it starts detached, in its own session with no terminal input,
and waits for `/health`. With `OMNIVOICE_DATA_DIR` set the log is `backend.log`
beside the data folder, else in the VoiceStudio checkout.

In WSL, run one backend on port 3900 at a time: WSL forwards its ports to
Windows, so a WSL backend answers the Windows scripts too, with Linux paths.
Stop the Windows one before starting it in WSL (and the other way round), or
give one of them another port through `VOICESTUDIO_API`. They share the GPU,
and the GPU lock (`work/gpu.lock`) is per clone, so don't dub on both sides
at once either. A WSL backend outlives the shell that started it, but not the
distribution: WSL may stop that once no WSL terminal or command is left
running. `scripts/dub_anything.py run` keeps one running while it works.

## 5. Models (~15 GB)

### macOS and Linux
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

#### macOS and Linux
```bash
hf auth login   # or hf login; on Linux vendor/VoiceStudio/.venv/bin/hf
curl -s -X POST "http://127.0.0.1:3900/models/install" -H "Content-Type: application/json" -d '{"repo_id":"pyannote/speaker-diarization-3.1"}'
curl -s -X POST "http://127.0.0.1:3900/engines/diarisation/select" -H "Content-Type: application/json" -d '{"engine_id":"pyannote"}'
```

#### Windows
```powershell
.\vendor\VoiceStudio\.venv\Scripts\hf.exe auth login
Invoke-RestMethod -Method Post "$api/models/install" -ContentType application/json -Body '{"repo_id":"pyannote/speaker-diarization-3.1"}'
Invoke-RestMethod -Method Post "$api/engines/diarisation/select" -ContentType application/json -Body '{"engine_id":"pyannote"}'
```

Installing the audio.cpp bundle can switch the active diarisation engine to
Sortformer, so select pyannote again afterwards. Sortformer is limited to 120 s
clips and 4 speakers.

The backend also reapplies its saved performance profile at startup. Set the
ASR and diarisation families to Quality when you need large-v3 and pyannote
to remain selected across restarts. Check the resulting selections before prep.

Some installed models require a local acknowledgment of their model terms
before first use. Present the backend's disclosure and linked terms to the user,
then record acceptance only after their explicit answer. Existing Hugging Face
access and downloaded weights do not replace that acknowledgment.

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
vendor/VoiceStudio/.venv/bin/python scripts/dub.py prep <some short clip>.mp4      # prints a Japanese transcript
```

### Windows
```powershell
cd <path-to-dub-anything>
$py = ".\vendor\VoiceStudio\.venv\Scripts\python.exe"
& $py scripts\dub.py prep <some short clip>.mp4      # prints a Japanese transcript
```

### Linux
```bash
cd <path-to-dub-anything>
vendor/VoiceStudio/.venv/bin/python scripts/dub.py prep <some short clip>.mp4   # prints a Japanese transcript
```

Five tools run in their own throwaway uv environments, so they can't disturb
VoiceStudio's:
- `separate.py`: audio-separator (on macOS with PyPI torch; on Windows and Linux with CUDA 12.8 PyTorch). Its Roformer model
  (about 900 MB) downloads on first use into the model cache (`audio-separator`).
  On Linux and macOS its `diffq` dependency compiles from source (it has no
  wheel for Python 3.11 there): without a C compiler the environment fails to
  build (on Ubuntu with "error: [Errno 13] Permission denied: 'cc'").
- `second.py`: sherpa-onnx (its models, about 850 MB for Japanese speech,
  1 GB for English, 900 MB for Spanish and 320 MB for Korean, download into the
  model cache on first use).
- `subs.py`: RapidOCR (its Korean recogniser downloads on first use, for a
  Korean source; uses standard onnxruntime on macOS, onnxruntime-gpu on Windows and Linux).
- `vc.py`: Chatterbox, which `scripts/dub.py generate` starts itself for the
  `"american"` voice.
- `languages/kana.py`: pykakasi, which `lineqa.py` starts itself to read takes in
  Japanese as kana (dubs into Japanese).

The commands are in [AGENTS.md](../AGENTS.md).

Voice profiles are rebuilt automatically from `work/voices.json` on first use.
Profile ids are machine-specific and cached in `work/profiles.json`.
