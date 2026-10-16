# Jellyfin integration

## Why this work is on a branch

As of 2026-10-15, this integration is deferred on `codex/jellyfin`. A dub can
take hours and still needs an agent to investigate errors, handle exceptions,
and make directing decisions throughout the run. Moving the job into Jellyfin
added service, queue, plugin, and updater maintenance before the dubbing workflow
was ready to run unattended. An AI harness is the better home for supervising
the work for now.

The branch preserves every commit from the initial Jellyfin addition onward.
`main` resumes from the preceding pull-request merge, with the independent core
dubbing improvements extracted into a separate commit. Revisit the integration
when the pipeline can reliably handle unattended jobs and their recovery.

The setup and release information below records the experiment. Release feeds
and automatic updates are no longer maintained; use `main` for direct dubbing.

The local Jellyfin plugin queues a dub and shows its progress. A separate Python
worker runs the existing pipeline and launches either Codex or Claude Code to
direct each part. The worker and Jellyfin currently run on the same computer.
The plugin targets Jellyfin 12.2 and .NET 10, and builds against the installed
server assemblies rather than assuming a compatible plugin ABI.

## Install on Windows

Set up VoiceStudio and the dubbing tools as described in [setup.md](setup.md).
Install the .NET 10 SDK and the desired director CLI. Sign into that CLI as the
Windows user who will run the worker (`codex login` or `claude auth login`).
The worker reuses the CLI's authentication; it does not store model API keys.

From the repo root:

```powershell
# Build and stage the DLL and private configuration under ignored work/service/.
powershell -NoProfile -File scripts/integrations/setup-jellyfin.ps1

# Install the staged plugin/configuration and start the background worker.
powershell -NoProfile -File scripts/integrations/setup-jellyfin.ps1 -Install -StartWorker

# Enable the matched plugin/worker updater and start it now.
# Existing jobs keep running; installation waits until the queue is idle.
powershell -NoProfile -File scripts/integrations/setup-jellyfin.ps1 -EnableAutoUpdates
```

The installer accepts `-JellyfinInstallDir`, `-JellyfinDataDir`, and `-Python`
for installations outside the defaults. Preserve the same Windows account for
the server-side worker so its CLI logins, models, and environment remain available.
Restart Jellyfin once after installing or updating the plugin. The installer
preserves existing model choices, default director, and review settings.

In Jellyfin's dashboard, open **Plugins → Dub Anything**, or its **Dub Anything**
sidebar entry. Administrator access is required. The installer fills in the
loopback worker address and private worker token. Those can also be changed
under **Worker settings**. Leave model fields empty to use the corresponding
CLI's configured model. `DUB_CODEX_BIN` and `DUB_CLAUDE_BIN` can override CLI
discovery for unusual installations.

## Updates and releases

The dashboard shows the loaded plugin version, repository link, and update
status under **About and updates**. One version in `Directory.Build.props`
identifies both components. The public feeds are:

- [Coordinated worker/plugin releases](https://raw.githubusercontent.com/swax/dub-anything/main/releases/updates.json)
- [Jellyfin plugin repository](https://raw.githubusercontent.com/swax/dub-anything/main/releases/manifest.json)
- [Release notes and downloads](https://github.com/swax/dub-anything/releases)

The Windows coordinator checks the release feed every six hours and checks for
an idle queue every minute. Running **and queued** jobs defer installation.
Awaiting-review previews, completed jobs, and failed/cancelled jobs are retained.
An exclusive SQLite transaction reserves the idle queue, pauses new job/resume/
publish admission, and stops the worker gracefully before changing any code.
Jellyfin then restarts with the matched plugin and worker. New admission resumes
only after both report the expected version. Other Jellyfin playback may be
interrupted by this restart; the idle gate concerns dubbing jobs.

Downloads are restricted to this repository's HTTPS release URLs and checked
against SHA-256. The archive is checked for traversal, Windows path aliases,
symlinks, duplicate paths, size limits, and matching component identities. Worker
releases run from `work/service/releases/<version>/runtime/`; the development
checkout is preserved. Junctions share the existing `work/`, `videos/`, and
`vendor/` folders. VoiceStudio, models, CLI logins and Python remain in their
existing environments. They are not upgraded by this feed.

The prior plugin moves into `work/service/backups/`, and the prior worker release
remains available. A failed or interrupted update restores the previous pair.
A failed version is not automatically retried. The transaction journal, status,
and updater log live in `work/service/update-journal.json`, `update-status.json`,
and `updates.log`. If rollback itself cannot start the previous pair, admission
stays paused and the status reports the error rather than starting jobs with
incompatible components. An unavailable worker with an active job recorded also
defers updates until the worker is restarted manually.

Local changes in a managed worker, including agent-added language support, defer
updates for review; the updater does not overwrite them. Review and contribute
those changes, then reconcile the installed release before updating. Keep all
working files, configuration, the queue database, and backups.

The coordinator adds the Jellyfin repository at the first idle update. This
plugin's packaged `meta.json` sets native Jellyfin `autoUpdate` to false, so
Jellyfin's independent scheduled task cannot update its DLL ahead of the worker
or bypass the idle gate. Automatic updates for this plugin are managed by the
coordinator. Other plugins keep their own settings.

`-EnableAutoUpdates` registers **Dub Anything Updates 8767** in Windows Task
Scheduler for the current user's logon, starts it immediately, and uses that
same user's CLI authentication. Keep that user account and the source checkout.
For a Windows-service Jellyfin installation, set `jellyfin_service` to its exact
service name in `work/service/install.json`. Custom ports/paths can be set there
too. Set `enabled` to false to pause updates; the separate worker can still be
started with `setup-jellyfin.ps1 -StartWorker`. Manual `-Install` is refused once
a coordinated installation is configured.

To publish a release, update `Directory.Build.props` and `releases/notes.md`,
commit to `main`, and push. The **Publish Jellyfin and worker release** workflow
runs formatting, lint, tests, and a build against the published Jellyfin API.
It publishes a `jellyfin-v<version>` GitHub release with both archives and feeds,
then commits the feeds back to `main`. Increment the version for every release;
published version contents are immutable. The build tool packages only source
files from explicit code/documentation directories, never `work/`, media,
credentials, model caches, or local configuration.

## Use

1. Search for a movie or episode in the library and select the video.
2. Enter source and target language names or codes. Suggestions are convenient
   examples, not a restriction. Choose Codex or Claude Code and the voice.
3. Queue the dub. The browser can be closed while the worker runs.
4. When it is ready for review, open its report and play the preview file listed
   in the job. Check the flagged timestamps for acting, voice likeness, and mix.
5. Publish when satisfied. Jellyfin scans the library automatically.

Captions, outlined subtitles, signs, and title cards are detected automatically.
The director compares the text with audio and frames, since text that translates
foreign dialogue must not be mistaken for a transcript of spoken words. OCR
appearance labels are evidence, not a guarantee. Only useful text becomes subtitles.

The original library file is preserved. The worker makes a separate staged copy
before the pipeline runs its normal delivery, then publishes another video with
` - Dub Anything <language> <voice> <job>` in its filename. The result includes
the original soundtrack, dub audio, and subtitles. Jellyfin's grouping of that
version depends on the library's naming/layout; it may appear as another item.
Staging, rendering, and publication require additional disk space.

**Review before publishing** is on by default. Turning it off in Worker settings
publishes automatically after the pipeline finishes. Neither mode certifies the
acting or removes the need for listening.

## Language support

The form accepts any language; the engine still needs verified support for the
requested pair. Existing pairs run immediately. For a missing pair, the chosen
director gets a separate scratch copy of the scripts, docs, and tests. It follows
[the language workflow](languages/README.md#adding-a-language): language modules,
QA rules, notes, and a short actual test dub with and without per-line references.

The supervisor independently runs the tests, per-line transcription QA, timing,
and reference-mode scoring. It imports only allowed language files and new
language tests, preserves existing modules/pair settings, and refuses to overwrite
concurrent edits. Source-specific test media and measurements stay in ignored
`work/service/jobs/<id>/language-workspace/`. New support remains uncommitted;
review it and consider contributing it upstream after listening to the test.

If a language needs changes to a generic recognizer, voice engine, or another
shared tool, the job stops with a concrete blocker for normal implementation and
review. Entering a language does not guarantee that the installed models support
it. New pairs take extra time and GPU work on their first run.

## Worker lifecycle and troubleshooting

```powershell
$py = 'C:\git\VoiceStudio\.venv\Scripts\python.exe'
& $py scripts/integrations service serve    # foreground, useful for development
& $py scripts/integrations service status   # installed directors, language pairs, ffmpeg
& $py scripts/integrations service stop     # stop gracefully; keep unfinished work
```

Use `setup-jellyfin.ps1 -StartWorker` to start it detached again. With automatic
updates enabled, the logon task starts the worker when its queue is idle. An
unfinished active job recorded after a crash needs a manual worker restart and
explicit resume. One worker owns the persistent SQLite queue, and it runs
one library job at a time. The pipeline's GPU lock still coordinates other tools.

Python integration modules and the setup script live in `scripts/integrations/`.
Run them through `python scripts/integrations service ...`, `updates ...`, or
`build-release ...`; see [the integration guide](../scripts/integrations/README.md).
Direct dubbing continues to use `scripts/dub_anything.py` without those services.
Release bundles include two small compatibility launchers at the old worker and
updater paths so already-installed updaters can upgrade and roll back safely.

- **Cancel** stops the active process tree and keeps working files.
- Each card shows the current tool and part number (including batches), the
  current stage's completed/remaining counts, and parts still unfinished overall.
  Expand **Progress by stage** for the preparation, direction, and rendering
  breakdown. A part finishes only
  when every requested voice has current render/QA results. Parts kept entirely
  original count as finished. Counts come from saved files and survive resume.
- **Active runtime** counts processing time across resumes, excluding queue waits,
  pauses, review, and worker downtime. Older jobs show `~` because their earlier
  runtime is reconstructed from logs. New active intervals are tracked directly.
- **Estimated remaining** learns separate per-part timings for each stage and
  voice, including render/QA rounds. A stage estimate appears first; the full
  estimate waits for samples from all remaining stages. It is approximate and
  excludes final assembly and publishing, which are shown separately. Paused
  estimates describe the work left once resumed. Retries and content complexity
  can change estimates; the display does not promise a completion deadline.
- **Pause** stops at the next completed tool step. A running director session and
  background checks finish first, so pausing can take time. The card says
  "Pausing" until the worker has stopped. Queued jobs pause immediately.
- **Resume** also restarts paused jobs from saved progress. An interrupted step
  may repeat. Paused jobs stay paused across restarts and allow idle updates.
- **Resume** queues a failed, interrupted, or cancelled job from its saved work.
- A worker restart marks in-progress jobs interrupted; they need an explicit resume.
- CLI login expiry, usage limits, denied tool permissions, render failures, or
  unresolved timing are reported as failures with logs rather than silently approved.
- Errors show a short summary on the job card; expand **Error details** for the
  original diagnostic. **Refresh log** below an open log fetches its latest text.
- The worker uses Jellyfin's bundled ffmpeg/ffprobe when system PATH lacks them.
- The local worker's log is `work/service/service.log`; per-job CLI and pipeline
  logs are under `work/service/jobs/<id>/`. The normal pipeline report is under
  `work/jobs/<slug>/report.md`.

Codex uses `exec --sandbox workspace-write --json` with model-cache/job directories
added explicitly. Claude Code uses print mode and `dontAsk`, with file access and
the known directing commands allowed. Neither adapter enables a global permission
bypass. The director is told to preserve uncertain lines with listening notes.

The HTTP worker binds to loopback and requires its bearer token for job data and
mutations. The health endpoint exposes only service availability. Jellyfin proxies
requests through administrator-authorized plugin endpoints, so no Jellyfin API key
is needed. Keep `work/service/config.json` and the installed plugin XML private.
Remote workers, NAS path mapping, playback-time dubbing, and automatic cleanup are
not implemented in this version.

## Development checks

```powershell
uvx ruff format
uvx ruff check
& $py -m unittest discover -s tests -v
dotnet build integrations/jellyfin/Jellyfin.Plugin.DubAnything.csproj --configuration Release
```

The automated tests use generated temporary fixtures and avoid GPU/model calls.
Live headless connection checks and the signed-in Jellyfin dashboard should also
be verified when changing the adapters. A complete new media dub still needs a
short listening test before claiming its quality is verified.
