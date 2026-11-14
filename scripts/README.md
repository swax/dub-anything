# Script entry points

For a dub, start with [AGENTS.md](../AGENTS.md) and
[docs/directing.md](../docs/directing.md). Run scripts from the repository root
with VoiceStudio's Python. The agent directs the job and handles failures as
the tools produce evidence and render each part.

| Task | Entry point |
|---|---|
| Initialize the pinned VoiceStudio fork and its Python environment | `scripts/setup.py` (`--init-only` for sources alone) |
| Create, run, inspect, and finish a dub (or subtitles: `--voices none`) | `scripts/dub_anything.py` |
| Start or inspect the voice backend | `scripts/backend.py` |
| Cast voices and inspect speakers | `voices.py`, `cast.py`, `frames.py` |
| Read speech and on-screen text | `asr_words.py`, `subs.py`, `signs.py`, `second.py` |
| Check a directed part before it renders, in one pass | `check.py` (`edges.py`, `tails.py` and `fit.py`, with only the flags printed; `--ready` marks it ready) |
| Fit dialogue windows and check takes | `edges.py`, `tails.py`, `fit.py`, `qa.py`, `lineqa.py` |
| Time subtitles for reading and check their speed | `subtitles.py` (shared), `fit.py` |

`dub_anything.py new ... --auto-text` reads both caption styles and looks for
signs. For a supervised pause and resume, see the directing guide's
`DUB_PAUSE_FILE` instructions. Shared modules support the tools: `core.py`
(where things live, and the JSON files the tools share), `voicecast.py` (the
cast, voice fingerprints and line ids), `references.py` (per-line references
and the window check), `lengths.py` (take lengths, their prediction and fit),
`gpulock.py`, `gpu.py` (the PyTorch build and what an AMD GPU needs),
`whisper_safe.py`, and `languages/`. They call one another through the module
(`core.load_json`, `voicecast.voice_sig`), so a test patches a name once, in
the module it lives in.

A tool that `dub_anything.py run` acts on reports what it decided with
`core.report_result(...)` instead of leaving the runner to read its messages:
change a message freely, but keep the reported fields (`no_speech`, `overflow`,
`parts`, `output`). Write shared JSON with `core.save_json`, which replaces the
file in one step.

The tests in `tests/` need no GPU, backend or models. Run them from the
repository root with VoiceStudio's Python, `python -m unittest discover -s tests`,
or without that environment as [tests/requirements.txt](../tests/requirements.txt)
shows. Four of them mux real files and skip without ffmpeg on PATH. CI
(`.github/workflows/ci.yml`) runs ruff and the tests on every push and pull
request.
