# Directing a dub

The machines separate, transcribe and synthesize. Directing is everything
between the transcript and the render: what was really said, who said it, what
it means in English, and exactly when it is spoken. This guide is the workflow
that produced good results, plus the traps found along the way.

This page holds what every director needs: the files, the workflow's outline
and the tools. Each phase has its own page in [directing/](directing/); read
the ones for the work in hand (see "Workflow" below for which).

Run everything from the repo folder with VoiceStudio's Python
(`$py = ".\vendor\VoiceStudio\.venv\Scripts\python.exe"` on Windows, or
`vendor/VoiceStudio/.venv/bin/python` on macOS/Linux),
with the backend up (see [setup.md](setup.md)).

Whisper tools import `WhisperModel` from `scripts/whisper_safe.py`. Keep that
wrapper when adding a transcription tool: a trailing window with fewer than two
mel frames can crash older CTranslate2 builds during word alignment. The wrapper
omits timings for that unusable window and leaves ordinary word timings alone,
without changing the machine's CUDA/cuDNN dependencies.

## Files

Everything a job makes is in one folder, `work/jobs/<slug>/` (`<slug>` is the
name `dub_anything.py new` gave it):

| Path in the job's folder | What |
|---|---|
| `job.json`, `report.md` | What the job is, where it stands; the report of what to listen for |
| `NOTES.md` | The show's notes: story, characters, running jokes, word choices, decisions pending |
| `voices.json`, `voices/<key>.wav`, `profiles.json` | The cast: per character, clean clips of the actor plus their exact transcript; the joined references; their VoiceStudio profiles |
| `parts/NN/script.json` | Part `<slug>-NN`'s directed dub, the real work product |
| `parts/NN/segments.json`, `ja.srt` | Raw machine transcript, the VoiceStudio job id and the language pair (`prep` overwrites them; the `.srt` is named for the source language) |
| `parts/NN/` | The rest of the part's work: `draft.json`, `state.json`, `refs/`, `takes/`, `signs/`... `parts/NN.us/` is its American voice |
| `media/` | A link to the source video; `media/split/` its parts and `parts.json`; `media/split/dubbed/` the renders |
| `logs/` | The runner's logs, one per step |
| `scratch/` | Your own scripts, probes, frames and listening experiments: never anywhere else in the repo |

A part is still named `<slug>-NN` on every command line; the scripts find its
folder (`core.part_dir`). The docs write a part's folder as `<part>/`.
VoiceStudio keeps each part's stems and takes in its own data folder,
`dub_jobs/<slug>-NN` (`<slug>-NN-us` for the American voice). `work/NOTES.md`
holds only what is true of this machine (paths, hardware, installs). A clip
prepped by hand, outside a job, keeps the older layout: `work/<name>/` and the
shared `work/voices.json`.

`work/` and `videos/` are git-ignored: they hold the source, transcripts of it
and references to real voices.

### voices.json

```json
{
  "speaker_1": {
    "from": "episode-01",
    "clips": [[10.25, 13.95], [46.75, 48.65], [167.3, 174.2]],
    "ref_text": "<exactly what is said in those clips, in order>"
  }
}
```

`from` names a prepped clip. Its separated `vocals.wav` supplies the audio.
A clip can also come from any other prepped part, written `["episode-03", 81.2,
84.9]`, so a bit part can pool clean lines from every part it appears in.
`dub.py` joins the clips (with 0.3 s gaps) into the job's `voices/<key>.wav` and
creates a VoiceStudio profile on first use. Changing a voice's entry re-renders
every line that voice speaks, in every part, final ones included: `run` takes
them up again, and `status` shows them as `a voice changed`. A voice other
lines only fall back to counts the same way. To give a character a better
voice without re-rendering parts already approved, add it under a new key
(`speaker_1_v2`) and use that from then on. A remark is the exception: a
`note` in an entry (who the character is, why these clips) is for the record
and changes no take.

Each job has its own cast. A later episode can borrow an earlier one's:
`dub_anything.py new <video> --voices-from <earlier slug>` lets its scripts use
the earlier job's voices as they were built there, read-only. A new voice goes
in the new job's own `voices.json`, under a key the borrowed cast doesn't use.

A character can have a second reference in another register, built by
`scripts/register.py build` (see "The reference sets the register" in [gotchas](directing/gotchas.md)). Its entry
names the character: `"register_of": "speaker_1", "register": "calm"`. Casting,
pace estimates and QA treat it as the same character; only the delivery
differs.

### script.json

```json
{"lines": [
  {"start": 4.30, "end": 5.90, "voice": "speaker_1", "ja": "こんにちは", "text": "Hello."}
]}
```

- `start`/`end`: the window the English line may occupy.
- `voice`: a key in `voices.json`.
- `ja`: the corrected original. The field is named for the source language:
  `en` when dubbing from English, `es` from Spanish, `ko` from Korean, `zh` from Chinese (see
  "Other languages").
- `text`: the English that is spoken (the dub's language: Japanese or Spanish
  when dubbing into it).
- `instruct` (optional): OmniVoice tags, comma-separated, from a fixed
  vocabulary only. The useful one is `whisper`; the others cover gender, age,
  pitch and accent. `dub.py` rejects anything else, because free text aborts the
  whole render inside VoiceStudio. There is no emotion control.
- `effect` (optional): VoiceStudio's voice processing for this line, overriding
  `--effect`. The options are `broadcast` (default), `raw`, `cinematic`, `warm`,
  `bright`, `podcast`, and from the pinned fork `echo` for flashbacks and `pa`
  for public-address announcements; see [gotchas](directing/gotchas.md).
- `take` (optional): a number that gives a line repeated word for word its own
  take. VoiceStudio keys takes by the line's voice and text, so two identical
  lines in one part would collide. Repeated English replies with different
  original references need distinct take values; generation rejects a shared
  ID when it would make one line borrow the other's original audio.
  Check these identities again after shortening dialogue: two previously
  different sentences can become identical while retaining the same `take`
  value. Give the later occurrence an unused value before rendering.
- `keep_voice` (optional): `true` stops `scripts/register.py route` from moving
  the line to another of the character's references, e.g. when one of them can't
  say a word in it.
- `line_ref` (optional): `false` keeps the character's reference for this line
  instead of its own original (see [Render](directing/render.md)).
- `fallback_voice` (optional): a separate reference key in the assigned
  speaker's `register_of` family, used only when the line has no usable own
  reference. Overrides `render.fallback_voices`; it does not change casting.
- `ref_prefix` (optional): `false` uses only this line's original audio and
  transcript when its reference passes the source checks. It overrides the
  render block's choice; `true` adds the character's clips before the original.
  Rejected references still fall back to the character's clean reference.
- `actor_end` (optional): the checked time of the source actor's last sound,
  in seconds within the part, inside this line's window. Use only after inspecting
  the source when speech detection misses a shout or filtered announcement.
  It overrides the detected endpoint for `fit.py`, fill and timing reports;
  it does not move the window or prove that a reference is clean.
- `speed` (optional): a directed pace from 0.8 to 1.25, overriding automatic
  fill for this line. Use the recorded pace of a published take after checking
  its actual fit and handoffs when further automatic slowing overshoots. `1.0`
  keeps natural pace. Changed text or references still need fresh take QA;
  with both voices, check the override in each voice.
- `text_us` (optional): the English for the American voice when it needs to
  differ, usually fuller text for a line `scripts/fit.py --voice american` calls
  `SHORT?`. `dub_anything.py` renders that voice from a copy of the script in
  `<part>.us`.
- `listen` (optional): why the user should check this line (a casting guess, a
  misheard word); `dub_anything.py` puts it in the report with the film time.
- `note` (optional): a remark for the record (why a window moved, what was
  tried).

`listen`, `note` and `notes` are remarks wherever they appear (a line, a
caption, a sign, the script, a voice): nothing renders from them. Adding or
changing one on a final part updates the report and leaves the part final and
the delivered file as it is. `text_us` renders the American voice only. Any
other change to a line renders its part again in each voice, re-using the takes
of the lines whose voice and words stayed.

A top-level `"ready": true` tells `dub_anything.py` the part is directed
(`scripts/dub_anything.py ready <slug> <NN>`); it renders nothing before that.

A top-level `"render": {"line_ref": true, "fill": true, "dips": true, "cast_lines": true}`
records how the part is rendered (see [Render](directing/render.md)). `scripts/dub.py generate`
writes it on the first run: all on for a part never rendered, off for a part
rendered before they became the default, so its approved takes don't change.
For a new part with tight windows, also set `"predict_speed": true` so its first
pass uses the estimated pace; failed first passes cannot supply takes to fill.

With `concise` timing (the default), each line starts at the window start. With
`fill` on, its pace is then adjusted so it ends where the actor stops (see
[Render](directing/render.md)); otherwise it's spoken at its natural pace. It may run up to 0.3 s past
the end (`--overflow`). That allowance can reach a neighboring reply, so check
actual generated handoffs against the source. Outside all windows the original
soundtrack plays.

A part with no lines at all (a commercial break recorded with the show, the
closing credits) gets a `script.json` with an empty `"lines"` list, and
`assemble.py` uses its original audio. Such a part needs no `prep`; a part
without any speech (a silent channel card) gets one from `dub_anything.py` itself
when VoiceStudio finds nothing to transcribe. A scene
entirely in a third language is the same, with its `captions` (below) still in
the subtitles.
Keeping original audio is independent of translating visual text. With
`--signs` or `--auto-text`, every part still gets a text scan, including empty-line
parts, and delivery waits for those scans. Review film titles, credits,
introductory cards and other meaningful text even over music or silence.
Exclude unrelated commercial clutter only after reviewing the picture.
An unprepped part can be scanned with
`scripts/signs.py <part> --video <part.mp4> --source ja --target en` in the OCR
environment; it needs no voice job or audio preparation. The scan saves its
video and language metadata for later previews and assembly. Put translations
in that part's `signs` while leaving `lines` empty to preserve all its audio.
A trailer for the next film or episode, recorded with the show, is different: its
dialogue is the same cast, so dub it like the show with the same voices. Its title
cards and on-screen text need their own review too.

A top-level `"captions": [{start, end, ja, text}]` list (`en` for `ja` dubbing
from English, `es` from Spanish, `ko` from Korean, `zh` from Chinese) holds speech that stays in the original audio but still belongs in
the subtitles. Examples: dialogue already in
English, names, whispers, and bit parts too short to clone. `assemble.py` adds the
list to both subtitle tracks.

A top-level `"signs": [{start, end, text, box}]` list translates text in the
picture: a title card, a diagram's labels, a page shown to be read. `box` is
where the original is, `[x0, y0, x1, y1]` as fractions of the picture (from
`signs.py`). The English goes over it, or `place` puts it `above`, `below`,
`left` or `right` of the box, or `page`: from the box's top left corner,
left-aligned and wrapped in the box. Optional: `box_end` (the box at the end,
when the camera pans: the English moves with it, during `seen`, `[t0, t1]`, if
given; the candidate's `seen` holds the times of its two boxes, but a pan is
often shorter, so find it in the frames), `size` (relative font size),
`width` (the width it wraps in, as a fraction of the picture), `boxed` (a dark
box behind it, over busy writing), and `ja` and `note` for the record. Signs
render nothing and don't make a part render again; they change the delivery
(see [Package](directing/package.md)).

When subtitles would cover original graphics that stay in the picture, use
`"subtitle_layout": [{"start": 10, "end": 18, "top": 0.19}]` at the top level.
Times are within the part; `top` is a fraction of the picture's height, where
the top-centre of the subtitle sits. Preview the actual frames to find clear
space. Include the final cue's reading linger in the interval. An intersecting
cue uses this placement for its whole display; the last matching entry wins,
overriding automatic sign avoidance. This affects the styled translation track,
leaves plain SRT files plain, and does not render the voices again.

## Workflow

Use `scripts/dub_anything.py new ... --auto-text` when the picture's text styles
are uncertain or mixed. It reads both caption appearances and outlined bottom
subtitles with one OCR inference per changed frame, and looks for signs. The
appearance labels are not proof that text matches the spoken language: compare
the words with audio and frames before using them to correct a transcript.
Foreign-language dialogue translated in the picture may need to stay original
with translated captions. Auto detection provides evidence; the director still
decides which text the viewer needs.

To pause a long run at a completed tool step, set `DUB_PAUSE_FILE` to an absolute
marker-file path before starting `run --watch`, then create that file when a
pause is needed. The current tool and background second-opinion checks finish
before the runner exits with code 75. Remove the marker and run the same command
again to resume from saved progress. Without that variable, runs behave as usual.

For a new show, use the actor's voice with their native accent unless the user
requests American English or both (see "Voice" under [Render](directing/render.md)). The
language pair is Japanese → English unless the job says otherwise (`--source
en --target ja`, `--source en --target es`, `--source es`, `--source ko` or `--source zh`; see "Other
languages").
`scripts/dub_anything.py` runs the machine steps below for a whole video, around
your directing (AGENTS.md, "A dub, start to finish"); the pages below are what
each step does and how to direct it. The scripts are in `scripts/` and run from
the repo root.

| Phase | Page | Read it when |
|---|---|---|
| 1–2 | [Prep, draft and cast](directing/prep-and-cast.md) | casting a show's voices, or a part brings a new character |
| 3 | [Translate and time](directing/translate.md) | writing any part's `script.json`: every director, every part |
| 4 | [Render](directing/render.md) | choosing render settings and references, or a render goes wrong |
| 5 | [QA](directing/qa.md) | triaging what `lineqa.py` and the report flag |
| 6–7 | [Package and clean up](directing/package.md) | assembling, delivering and reclaiming the job's storage |
| | [Subtitles only](directing/subtitles.md) | a `--voices none` job, instead of 3–5 |
| | [Gotchas](directing/gotchas.md) | before directing a show: the traps of references, windows and takes |
| | [Machine gotchas](directing/machine.md) | the backend, GPU memory, drivers, OCR, AMD and Apple GPUs misbehave |

A director writing parts of a show already cast (a subagent on a batch of a
long film) needs this page, [Translate and time](directing/translate.md),
[Gotchas](directing/gotchas.md) and the two languages' notes.

**Checking a part before it renders:** `scripts/check.py <part>` runs the
structural check, `edges.py --apply`, `tails.py` and `fit.py` in each of the
job's voices in one pass, and prints only what needs a look: `!` on what must
be fixed (`OVER?`, `risk` on a window under 1 s, a broken script, `FAST`,
`LONG` and `SPLIT?` in subtitles), the rest to review (`SHORT?`, edges left alone, wet
lines, the lines falling back to the character's reference and why). It
rewrites the windows edges moves, so re-read those lines before editing them.
Fix, run it again, and `scripts/check.py <part> --ready` marks the part ready
once nothing must be fixed. The tools still run on their own, with every line
(`--all` prints every line here too).

## Other languages

This guide was written from Japanese → English dubs. For any other pair, read
the notes on both its languages in [languages/](languages/README.md) as well:
what differs with each as the source and as the target
([ja.md](languages/ja.md), [en.md](languages/en.md),
[es.md](languages/es.md), [ko.md](languages/ko.md), [zh.md](languages/zh.md)). Write what you learn about a
language there.

## Tools

The loop, in the order it runs:

| Script | Does |
|---|---|
| `scripts/dub_anything.py new\|run\|status\|ready\|report\|cleanup` | A whole dub (or with `--voices none`, subtitles): runs every machine step below in order, renders each part when its script is ready, QA rounds, delivery, the report |
| `scripts/split.py <video>` | Cut a long video into ~N-minute parts at pauses in speech (frame-accurate) |
| `scripts/dub.py prep <video> [--source en\|es\|ko\|zh --target ja\|en\|es] [--subtitles-only]` | Upload (VoiceStudio doesn't separate it), separate the voices with `separate.py`, attach the stems to the job, transcribe, diarize → `<part>/segments.json` (with the language pair), source-language `.srt`; `--subtitles-only`: the audio alone, no backend |
| `scripts/dub.py fork <part>.us` | Another voice of a prepped part: a VoiceStudio job over the same hard-linked media and stems, nothing uploaded or separated again (`dub_anything.py` does it for the American voice) |
| `scripts/separate.py <clip> ... [--restore]` | Separate prepped clips with Mel-Band Roformer, which leaves far less music in the vocals than htdemucs. `dub.py prep` runs it; run it by hand for subtitles-only parts, parts prepped before that, or another `--model` (own environment; see AGENTS.md) |
| `scripts/subs.py <clip> [--open]` | Read burned-in closed captions (OCR) → `subs.json`: exact words, speaker names, colours, off-screen marks; song lyrics become bare ♪ markers (own environment). `--open`: a film's own subtitles for foreign-language speech |
| `scripts/signs.py <clip> [--preview]` | Text in the picture to translate (signs, diagrams, title cards, pages): candidates with times and boxes → `signs.json`, a sheet per candidate and shot thumbnails (subs.py's environment); `--preview` draws the script's signs into their frames |
| `scripts/draft.py <name> ...` | Careful transcript snapped to real speech, missed lines recovered, speaker suggested per line, English speech found, long hallucinations re-transcribed, captions shown under each phrase → `draft.json` |
| `scripts/cast.py <name> ...` | Group lines across parts by voice, so each character gets one voice |
| `scripts/voices.py spans\|add\|check\|sim\|score` | Build the cast: speech spans around candidate clips, add a voice (locked, for parallel directors), transcribe a reference clip by clip, compare references, score any stretch against the cast |
| `scripts/frames.py <out.jpg> <video> <t>[:label] ...` | Labelled frame grid: who is on screen at each line |
| `scripts/asr_words.py <clip> <t0> <t1>` | Careful re-listen of one stretch of a clip's vocals, with word timings and speech spans (also takes a wav; `--lang en` for English speech) |
| `scripts/second.py <clip> ... [--script]` | Two more recognisers over each line (ReazonSpeech and SenseVoice for Japanese, Parakeet and SenseVoice for English, Parakeet v3 and Canary for Spanish, the Korean Zipformer and SenseVoice for Korean, Zipformer CTC and SenseVoice for Chinese): flags lines where both disagree with Whisper (own environment) |
| `scripts/edges.py <name> [--apply]` | Find windows that cut the actor mid-sound and move the clear ones; move touching windows onto the dip between the voices |
| `scripts/tails.py <name>` | How long each line's original voice rings on: flashback echo, PA announcements, big rooms |
| `scripts/check.py <name> ... [--ready] [--all] [--dry-run]` | The pre-render checks in one pass (structure, `edges.py --apply`, `tails.py`, `fit.py` in each of the job's voices), only what needs a look printed; `--ready` marks a part with nothing to fix ready |
| `scripts/fit.py <name> [--voice american] [--read]` | Each line's length against its window and the actor's speech, predicted before rendering with OmniVoice's own length rule (or IndexTTS's, for the `"american"` voice): `OVER?`, `risk`, `SHORT?`. A part to subtitle (or `--read`): each cue's reading speed, `FAST`, `LONG`, `SPLIT?` |
| `scripts/dub.py generate <name> --match-level` | Render `script.json` in the chosen voice (per-line references and fill by default, or `"american"`), levelled against the original actors → MP4 + SRT |
| `scripts/lineqa.py <name> ... [--reroll]` | Transcribe each rendered line alone (Japanese compared by its reading in kana, Spanish without accents); flag leaked, dropped or garbled words and flipped meanings; queue fresh takes |
| `scripts/shorttakes.py <name> --lines 1,3 [--candidates 3]` | Opt-in actor/OmniVoice candidates on an unready, fully rendered part; retain audio/settings/evidence, measure actual timing and transcription, install only with explicit `--apply` |
| `scripts/assemble.py <stem> [--voice american] [--parts 1-4] [--subtitles]` | Join parts: dub + original audio tracks, soft subtitles in both languages, sidecar SRTs (and styled ASS with the signs); a part without lines keeps its original audio. `--subtitles`: the subtitle files alone |
| `scripts/deliver.py <source> actor=<mp4> [american=<mp4>] [--source en --target ja]` | Put the dubs and subtitles into the source file, verified, renamed `<name> (dubbed)` (Matroska when there are signs); `--subtitles <stem>.subtitled.mkv`: a subtitle job's tracks, `<name> (subtitled).mkv` and an SRT beside it |

Used now and then:

| Script | Does |
|---|---|
| `scripts/vc.py --pairs <file>` | Convert takes to the actor's timbre (Chatterbox); `scripts/dub.py generate` runs it itself for the `"american"` voice |
| `scripts/register.py report\|build\|route\|check\|drop` | Optional calm or raised reference for a character's lines that can't use their own original |
| `scripts/qa.py <name>` | Transcribe the finished dub back and line it up with the script |
| `onsets.py`, `energy.py`, `pitch.py`, `contour.py` | Onset, loudness and pitch checks (the last two need librosa) |
| `eval/timing.py`, `eval/score.py` | Compare renders of the same part (A/B variants): timing against the actor; delivery, likeness, quality and accent. See [research.md](research.md) |
| `scripts/backend.py status\|start\|stop\|restart` | Keep the VoiceStudio backend running, detached from the agent's shell |
| `core.py`, `gpulock.py`, `gpu.py`, `ass.py`, `subtitles.py`, `languages/` | Not tools: paths, voices, line ids, takes and length prediction shared by the scripts; the lock that keeps GPU jobs from overlapping; which PyTorch build VoiceStudio has and what a ROCm one needs; the styled subtitles that place the signs; subtitle timing for reading and the reading-speed check; what the scripts know of each language (one module per language: Whisper's hallucinations, the second opinions, how QA compares a take, subtitles and track names; `kana.py` reads Japanese as kana, with pykakasi in its own environment) and the pairs they know |
