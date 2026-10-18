# Script entry points

For a dub, start with [AGENTS.md](../AGENTS.md) and
[docs/directing.md](../docs/directing.md). Run scripts from the repository root
with VoiceStudio's Python. The agent directs the job and handles failures as
the tools produce evidence and render each part.

| Task | Entry point |
|---|---|
| Create, run, inspect, and finish a dub (or subtitles: `--voices none`) | `scripts/dub_anything.py` |
| Start or inspect the voice backend | `scripts/backend.py` |
| Cast voices and inspect speakers | `voices.py`, `cast.py`, `frames.py` |
| Read speech and on-screen text | `asr_words.py`, `subs.py`, `signs.py`, `second.py` |
| Fit dialogue windows and check takes | `edges.py`, `tails.py`, `fit.py`, `qa.py`, `lineqa.py` |
| Time subtitles for reading and check their speed | `subtitles.py` (shared), `fit.py` |

`dub_anything.py new ... --auto-text` reads both caption styles and looks for
signs. For a supervised pause and resume, see the directing guide's
`DUB_PAUSE_FILE` instructions. Shared modules such as `core.py`, `gpulock.py`,
`whisper_safe.py`, and `languages/` support the tools.
