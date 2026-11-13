# Subtitles only

Part of [directing a dub](../directing.md): read its file formats and workflow overview first.

`scripts/dub_anything.py new <video> --voices none [--source .. --target ..]
[--signs ...]` makes a job that subtitles the video instead of dubbing it. Any
two languages with a module will do: `PAIRS` lists the dubs checked end to end,
and a subtitle needs only the source heard and the target written. Nothing is
cloned or rendered, so the VoiceStudio backend isn't needed, and `run` doesn't
start it:

- **Prep** (`dub.py prep --subtitles-only`) only takes each part's audio out
  into a job folder of VoiceStudio's shape (`<part>-local` beside its own jobs).
  `separate.py` then separates it (Roformer's are its only stems), and
  `draft.py` transcribes it as for a dub, without suggesting speakers.
  `asr_words.py`, `second.py`, `subs.py`, `signs.py` and `frames.py` work as
  they do for a dub. A part prepped this way can't be dubbed: make a dub job
  for that.
- **The script** is the same `script.json` without `voice`: `start`/`end`,
  the original under the source language's code, `text` for the subtitle,
  `listen` for a guess (the report lists them under "Check against the
  original"), and `signs`. `captions` still work, but there is nothing to keep
  original: give everything heard a line, the walk-ons, announcements and names
  a dub would leave in the original audio included.
- **Write for reading, not for the lips.** A cue has at most two lines of about
  42 letters, and a viewer reads about 17 letters a second in comfort (20 is
  fast; `READING_CPS` in the target's module). Condense where the speech is
  fast: drop the fillers, hedges and restarts a dub keeps to fill the actor's
  time. Split at the speaker's pauses and sentence ends, so each cue is a whole
  thought, and keep cues under 7 s. A script written for a dub of the same
  video is a start for the meaning, not for the cues: it was fitted to the
  actor's time.
- **Windows** can be looser than a dub's: no clean-window rule, no `fit.py`
  length prediction, and `edges.py` is optional. Start each at the speech (the
  draft's windows are snapped already). `assemble.py` then times every cue for
  reading (`subtitles.timed`): at least 0.83 s, long enough to read at the
  comfortable rate (up to 7 s), half a second past the speech where there's
  room, never into the next cue, and a gap under 0.6 s closes, so the text
  changes instead of blinking off and on.
- **Check** with `scripts/fit.py <part>`: for a part prepped this way it gives
  each line's reading speed once timed, instead of the dub's length (`--read`
  does the same for any part). Fix every `FAST` (fewer words, or merge with a
  neighbour so the cue has the time), `LONG` (more than two lines: split it at
  a pause) and `SPLIT?` (spoken for longer than a cue should stay).
- **Signs** as for a dub (`--signs`). Text that repeats what is said (a variety
  show's captions of the host's own words) needs no sign: the subtitle says it.
  A travel or food show is full of captions that don't: the dish, its price
  and maker, a station's name, a map's labels, a note on local history. Those
  are worth translating. A narration card on a plain white screen needs
  `boxed`: white letters with a dark outline don't read on white. On a 30-min
  episode (nine 3-min parts) local prep took 1-2 s a part, separation and
  drafting a few minutes per four parts, and `signs.py` 1.5-4.5 min a part,
  the slowest step: place the first parts' signs while it reads the rest.
- Then `scripts/dub_anything.py ready <slug> <NN>`. A ready part is done; when
  every part is, `run` delivers. `assemble.py --subtitles` writes the subtitle
  files alone, and `deliver.py --subtitles` puts them into the source:
  `<name> (subtitled).mkv`, its video and audio copied and verified, the
  translation on by default (styled with the signs placed, when there are any)
  and the transcript, with the translation as an SRT beside it
  (`<name> (subtitled).en.srt`) for players that read subtitles from a file.
  The SRT carries the signs too, at the top of the picture (`{\an8}`, which most
  players honour). Delivering again replaces the subtitle tracks it added.
