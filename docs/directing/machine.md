# Machine gotchas

Part of [directing a dub](../directing.md): read its file formats and workflow overview first.

The backend's and the hardware's traps: stems, GPU memory, drivers, OCR pools, AMD and
Apple GPUs. Directing's own are in [gotchas.md](gotchas.md).

- **VoiceStudio reuses separated stems by video content.** `prep` copies the
  stems of the newest job made from the same video, including stems that
  `separate.py` swapped in. So after re-separating one job, a new job for the
  same video (an A/B copy, say) starts with Roformer stems, not htdemucs, even
  though VoiceStudio only runs htdemucs. Check `separator.json` in the new job
  folder, and use `scripts/separate.py --restore` or copy the stems you want.
- **A long-running backend can fill the GPU.** After many renders (other engines
  included), PyTorch's cache can hold almost all the VRAM while only OmniVoice
  is loaded (`GET /model/loaded` shows the free memory). Renders then slow to
  tens of seconds a line, and long per-line references fail with "Generation
  failed". `POST /system/flush-memory` on the backend releases it without
  unloading anything; `scripts/dub.py generate` calls it before and after each run.
  Prepping a whole video fills it too: transcription, diarisation and
  separation stay cached, and the first render then fails with "out of memory",
  even in the flush itself. Only a restart (`scripts/backend.py restart`) frees
  that; `dub_anything.py` restarts the backend before the first render after a prep, and
  when a render runs out of memory. The prep-to-render restart flag is local to
  the runner process: restarting the runner after prep loses it. In that case,
  finish current GPU work, check the backend through this machine's approved
  local access, and restart it before the first render; prepared parts and
  existing takes are reused. With both voices, the backend keeps the
  American voice's engine loaded after its render, and the next part's actor
  render ran out of memory on a long per-line reference about one part in six;
  the restart cleared it each time. A program outside the dub using the GPU
  (another project's job) makes the retry fail too: two failures block the
  part, and the next `scripts/dub_anything.py run` retries it.
  Long references can also fill VRAM within a single native synthesis batch.
  If those batches crawl, try `OMNIVOICE_DUB_BATCH_WIDTH=2` in the backend's
  startup environment to generate fewer lines together. Wait for your current
  tools to finish before restarting the backend, and carry the setting in the
  job runner's environment so automatic restarts keep it.
  The GPU lock schedules active tools, but idle backend models still occupy
  VRAM. After prep, if standalone transcription or window checks slow down,
  finish backend tasks and unload its idle models or restart it before those
  checks. Run broad test suites with GPU access disabled while a dub is active.
  In PowerShell, use `$env:CUDA_VISIBLE_DEVICES='-1'`; assigning an empty string
  can remove the variable and leave the GPU available.
- **After an NVIDIA driver update, restart the backend.** A backend that was
  running during the update keeps "CUDA out of memory" errors on an empty GPU
  and silently runs Whisper on the CPU.
- **CPU OCR needs bounded inference pools.** The CPU OCR environment uses
  `onnxruntime` instead of `onnxruntime-gpu`. It can scan on-screen text beside
  a render on a discrete GPU; test a short range first, since CPU scans can take
  much longer. Keep at most two scans running together. The OCR
  engine caps each inference pool at four CPU threads. Limiting OpenMP alone
  does not limit ONNX Runtime's own pools. On an integrated GPU, keep these
  memory-intensive tasks sequential as described below.
- **With an AMD GPU (or on a Mac), Whisper runs on the CPU.** CTranslate2 has
  no backend for either, so the tools that only listen (`asr_words.py`,
  `voices.py check`, `lineqa.py`, `qa.py`, the window check in `fit.py` and
  `generate`, `draft.py` with its speaker embeddings) don't wait for the GPU
  lock and run beside a render. They take the CPU instead: on a 12-core laptop
  a draft took about 10 minutes a 3-minute part, and two of them at once each
  ran at half speed. The OmniVoice renders and the separation stay on the GPU.
  Running `lineqa.py` beside a render doesn't pay on an integrated GPU, which
  shares the system's memory with the CPU. On a Radeon 890M, renders beside a
  check ran 1.9-2.4x slower and the check 1.7-2x slower, though the CPU was
  only half busy and the check ran below normal priority. A two-part test dub
  in both voices took 2 h 27 min that way against 2 h 9 min with each check
  in turn. An actor line also ran past VoiceStudio's 300 s batch limit
  ("Generation failed"), and IndexTTS then started in CPU mode until
  `scripts/backend.py restart`. So `dub_anything.py` runs each check in turn
  there too. `lineqa.py` hears in int8 there instead: on the CPU CTranslate2
  turns large-v3's float16 into float32, and int8 heard a part's 48 takes in
  280-290 s instead of 450 s (the American voice's in 315 s instead of 435 s),
  flagging the same lines in all 96; ten transcripts were a word or a comma
  apart. Takes heard in float32 before are heard again once.
- **Speaker detection.** Automatic diarisation can merge similar voices or
  split one speaker. Confirm casting with multiple signals. Sortformer can't
  handle clips over 120 s or more than 4 speakers, ignores
  count hints, and has CC-BY-NC weights.
- **VoiceStudio integration (`vendor/VoiceStudio`).** The fork forwards the
  source-language hint, supports per-line voice/emotion/seed conditioning with
  matching cache fingerprints, and preserves echo/PA reverb tails. Its Windows
  AMD compatibility shims cover the verified legacy `2.9.1+rocm7.13.x` build
  only. Current upstream supplies CPU float32 and IndexTTS speed control, and
  its atomic publication rules keep failed or cancelled renders from replacing
  published takes. See the fork's maintenance notes before changing that policy.
- **On an AMD GPU, check that IndexTTS is on it.** "CPU mode" in `backend.log`
  after the sidecar spawns means it is not (setup.md); a line then takes about
  a minute.
