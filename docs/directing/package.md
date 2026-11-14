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
& $py scripts\deliver.py <job-local source video> actor=<actor assembled.mp4> american=<american assembled.mp4>
```

`deliver.py` puts the dubs into its input file, so give it a **job-local working
source copy**, preserving the original library file and its name
(`dub_anything.py new` makes one in `bulk/media/source/`). It puts each
voice's English track first (the default voice's plays; with both, the actor's),
then the source's own audio and subtitles untouched, with English subtitles per voice and
the Japanese transcript. Every copied stream is checked against its input before
the new file replaces that working copy, renamed `<name> (dubbed)`. Delivering again
replaces the dub tracks a "(dubbed)" file already has. `dub_anything.py` does both
steps when every part is done.

The final handover belongs **beside the original library video**, using its
original basename plus ` (dubbed)` and the delivered container extension.
`dub_anything.py` publishes it there itself after delivery (`publish`): each
file (with a subtitles-only job, the `.srt` sidecar too) is copied under a
hidden `.partial` name in the original's folder, compared with the verified
delivery by SHA-256, and only then renamed. It replaces only its own earlier
publication (the same path, with the hash it recorded) and stops with a problem
rather than overwrite any other file. The published path and hash are in
`job.json` (`delivered.published`) and the report; note them in the job's
`NOTES.md` and link the file in the handover. A delivery that exists only in
`bulk/` is not a finished handover.

When any part has `signs` or `subtitle_layout`, `assemble.py` also writes the
English track styled (`.en.ass`: the dialogue, with each sign placed by its
text, `ass.py`) and the signs alone (`.signs.ass`, only when signs exist).
`deliver.py` delivers Matroska,
`<name> (dubbed).mkv`, in place of the working source copy: MP4's subtitles can't be placed.
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

1. **Check what must survive.** Everything at the top of the job's folder
   stays: scripts, transcripts, states, reports, notes, the cast with its
   `voices/*.wav` (later jobs may borrow them) and `scratch/`. So do the
   published dub and the original beside it, VoiceStudio voice profiles,
   installed engines, Python interpreters and downloaded models. Check that
   shared voice references have been materialized before retiring their source
   jobs, and that no retained job's paths point into this one's bulk. Stop or
   wait for job watchers and backend tasks before deleting their files.
2. **List, then delete.** `scripts/dub_anything.py cleanup <slug>` prints what
   the job keeps and what it can delete, with sizes (a hard-linked file counted
   once): the job's `bulk/` (the source copy, split parts and renders, per-line
   references, kept takes, sign sheets, logs) and VoiceStudio's folders for its
   parts (every `dub_jobs/<slug>-...`, with their `exports/` history). It checks
   the published dub against the hashes recorded when it was published. With
   the user's acceptance or cleanup request, `cleanup <slug> --delete` deletes
   exactly that list, only the job's own folders and never through a link, and
   records `cleaned` (when, bytes freed) in `job.json`. It refuses while the dub
   isn't verified beside the original, since `bulk/` then holds the only copy;
   `--force` deletes anyway, for a job the user has explicitly abandoned.
   Deletion is permanent: rendering the job again needs a fresh prep. Record
   the result and the measured free-space change in the job's `NOTES.md`.
3. **Review leftovers separately.** Inventory jobs left from earlier cleanup or
   experiments on their own; a folder's age is not proof it is unused. Keep
   requested review clips and experiments out of `bulk/` (in `scratch/`).
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
