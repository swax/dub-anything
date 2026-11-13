# Package and clean up

Part of [directing a dub](../directing.md): read its file formats and workflow overview first.

## 6. Package

```powershell
& $py scripts\assemble.py episode --parts 1-4      # or: scripts\assemble.py <clip> for an unsplit clip
```

This produces one MP4. The video is stream-copied from the source, so part cuts
leave no seams. Audio 1 is the dub (default) and audio 2 is the original
soundtrack. The subtitles are English (the dub's text) and Japanese (the
corrected original), both including `captions`. Any text subtitle tracks already
in the source are kept, and `--sub file.srt:lang[:title]` adds more (for example
official subtitles). It also writes sidecar `.en.srt` / `.ja.srt` files.
`--voice american` assembles the American renders instead.

```powershell
& $py scripts\deliver.py <source video> actor=<actor assembled.mp4> american=<american assembled.mp4>
```

`deliver.py` puts the dubs into the source file itself: each voice's English
track first (the default voice's plays; with both, the actor's), then the
source's own audio and subtitles untouched, with English subtitles per voice and
the Japanese transcript. Every copied stream is checked against its input before
the new file replaces the source, renamed `<name> (dubbed)`. Delivering again
replaces the dub tracks a "(dubbed)" file already has. `dub_anything.py` does both
steps when every part is done.

When any part has `signs` or `subtitle_layout`, `assemble.py` also writes the
English track styled (`.en.ass`: the dialogue, with each sign placed by its
text, `ass.py`) and the signs alone (`.signs.ass`, only when signs exist).
`deliver.py` delivers Matroska,
`<name> (dubbed).mkv`, in place of the source: MP4's subtitles can't be placed.
The English tracks are then the styled ones, and the signs-only track follows
them, marked forced, for watching the dub with subtitles off. A player has to
render ASS subtitles to place them (mpv and VLC do, with libass).

Subtitle presence and a correct extracted ASS file do not prove playback works.
Long overlapping signs can hold back later packets during packaging; a player
then receives dialogue after its display time, although offline extraction and
preview rendering still recover it. Delivery disables early interleave flushing
(`-max_interleave_delta 0`) and checks the new tracks' packet positions against
the video before replacement. Check the packaged track, including dialogue
under a persistent label, rather than relying only on per-part previews.

For an independent package check, compare the extracted original-language
transcript's text and timing with the assembled sidecar. Compare styled ASS
dialogue together with every used style and layout setting. At audio joins,
check aligned level and residual error as well as correlation: a uniformly
quieter copy can still correlate perfectly with its reference.
For exact checks of retained audio, decode both full tracks to a common sample
rate and slice by sample index. Separately seeking to each crop can shift its
sample alignment and give a false mismatch.
Even after full-track decoding, resampled audio can need sub-sample alignment,
especially for bright sounds. Preserve the coarse result and refine alignment
with band-limited interpolation before judging a mismatch; keep the level,
residual and correlation limits unchanged.

Every subtitle track, a dub's included, is timed for reading as it's assembled
(`subtitles.py`, in the [tools](../directing.md#tools)), and a line longer than about 42 letters is broken in
two near its middle.

## 7. Storage cleanup

Every delivered dub or subtitle job needs a recorded storage outcome. While
the listener is reviewing it, keep the takes and evidence, record cleanup as
pending in the job's `NOTES.md`, and report the retained size. After acceptance or
an explicit cleanup request, finish the authorized cleanup in the same task.
The delivery flag alone does not establish listener acceptance. An idle
machine does not establish that an unfinished job has been abandoned.

1. **Check what must survive.** Locate the actual delivered video, including
   any move into the user's library, and verify it against its delivery proof
   when available. Keep an original source or a verified delivery containing
   the copied original video and audio. Preserve scripts, transcripts, states,
   reports, notes, the job's `voices/*.wav` (later jobs may borrow them), VoiceStudio voice profiles and
   references, and requested review pages, clips and experiments. Check that
   shared voice references have been materialized before retiring their source
   jobs. Check retained jobs' persisted paths and voice profiles too: cached
   stems or clone references can point into another job's directory. Preserve
   that dependency or migrate and verify it before removing the older job.
   Keep installed engines, Python interpreters and downloaded production
   models. Stop or wait for job watchers, backend tasks and cache users before
   touching their files.
2. **Stage the job's disposable bulk.** Run
   `scripts/dub_anything.py cleanup <slug>` after checking its inputs. It moves
   the job's VoiceStudio folders (every `dub_jobs/<slug>-...`, with their
   `exports/` history), `media/split/` (split parts and renders), per-line
   references and kept takes, the `media/` input video and job logs into
   `videos/_to-delete/<slug>/`. The job's folder keeps its scripts, cast, notes,
   report and `scratch/`; review `scratch/` for experiments worth keeping before
   deleting it. Keep a manifest outside that folder. If
   the workspace input is the only retained source, restore it from the holding
   folder before deletion. Preserve requested experiments and provenance logs
   separately. Do not use `--force` on an unfinished job without the user's
   explicit instruction to abandon it. Inventory jobs left over from earlier
   cleanup or experiments separately; a folder's age is not proof it is unused.
3. **Actually reclaim space when authorized.** A move on the same volume does
   not free space. Check every resolved absolute target is inside the intended
   holding folder; reject reparse points and unexpected files. Verify staged
   paths, sizes and file identities or hashes against the manifest, and check
   retained deliveries and references again. With the user's cleanup
   authorization, remove only those staged disposable files. Otherwise report
   their path and size for the user to delete. Record what was removed, what
   remains resumable, and the before/after free space. Keep partial cleanup
   distinct from the runner's full `cleaned` state.
4. **Review dependency caches separately.** Once all users of the cache are
   idle, `uv cache prune` removes dangling entries and cached environments.
   A broader, authorized `uv cache clean` clears regenerable package and
   environment caches; later tools may need downloads and environment rebuilds.
   Keep uv's Python installation directory, existing project/engine virtual
   environments, the model cache and voice data. Follow the same authorization
   and inventory discipline for retired experimental model downloads.

Windows hard links can make folder totals greatly exceed the unique stored
data. Removing one cached link does not reclaim a file still linked into an
installed environment. Count file identities and link counts when estimating
reclaimable bytes, and report the measured free-space increase after cleanup.
Keep source-specific inventories and measurements in ignored `work/` files.
