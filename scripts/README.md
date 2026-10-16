# Script entry points

For the usual request to dub a video, start with [AGENTS.md](../AGENTS.md) and
[docs/directing.md](../docs/directing.md). Run scripts from the repository root
with VoiceStudio's Python. Jellyfin is optional.

| Task | Entry point |
|---|---|
| Create, run, inspect, and finish a direct dub | `scripts/dub_anything.py` |
| Start or inspect the voice backend | `scripts/backend.py` |
| Direct a part: casting, evidence, timing, and QA | Tools named in `docs/directing.md`, including `voices.py`, `frames.py`, `asr_words.py`, `edges.py`, `tails.py`, `fit.py`, and `lineqa.py` |
| Install and manage the Jellyfin integration | [docs/jellyfin.md](../docs/jellyfin.md) and `integrations/jellyfin/` |

## Integration internals

These modules support the optional Jellyfin integration. A direct dubbing
request does not need them started or configured.

| Module | Purpose |
|---|---|
| `dub_service.py` | Persistent library-job queue and local HTTP worker |
| `director.py` | Launch Codex or Claude Code for a queued job's directing assignment |
| `language_setup.py` | Test missing language support for a queued job in a scratch workspace |
| `dub_updates.py` | Coordinate idle plugin and worker updates |
| `build_release.py`, `release_info.py` | Package releases and identify matching versions |

Both entry points use the same dubbing engine. Shared helpers such as `core.py`,
`gpulock.py`, `whisper_safe.py`, and `languages/` support that engine; they are
imported by the tools rather than launched as extra services. A Jellyfin worker
already running a job owns its pipeline, so do not launch a second copy for it.
