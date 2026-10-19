# Directing a dub

The machines separate, transcribe and synthesize. Directing is everything
between the transcript and the render: what was really said, who said it, what
it means in English, and exactly when it is spoken. This guide is the workflow
that produced good results, plus the traps found along the way.

Run everything from the repo folder with VoiceStudio's Python
(`$py = "C:\git\VoiceStudio\.venv\Scripts\python.exe"` on Windows, or
`~/Library/Application Support/VoiceStudio/runtime/project/.venv/bin/python` on macOS),
with the backend up (see [setup.md](setup.md)).

Whisper tools import `WhisperModel` from `scripts/whisper_safe.py`. Keep that
wrapper when adding a transcription tool: a trailing window with fewer than two
mel frames can crash older CTranslate2 builds during word alignment. The wrapper
omits timings for that unusable window and leaves ordinary word timings alone,
without changing the machine's CUDA/cuDNN dependencies.

## Files

| Path | What |
|---|---|
| `videos/` | Source media (git-ignored). Split parts go in `videos/<stem>-parts/` and dubs in `.../dubbed/` |
| `work/voices.json` | The cast: per character, clean clips of the actor plus their exact transcript |
| `work/<clip>/script.json` | The directed dub, the real work product |
| `work/<clip>/segments.json`, `ja.srt` | Raw machine transcript, the VoiceStudio job id and the language pair (`prep` overwrites them; the `.srt` is named for the source language) |
| `work/NOTES.md` | Per-show notes: story, characters, running jokes, word choices |

`work/` is git-ignored: it holds transcripts of the source and references to
real voices.

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
`dub.py` joins the clips (with 0.3 s gaps) into `work/voices/<key>.wav` and
creates a VoiceStudio profile on first use. Changing a voice's entry re-renders
every line that voice speaks, in every part. To give a character a better
voice without re-rendering parts already approved, add it under a new key
(`speaker_1_v2`) and use that from then on.

A character can have a second reference in another register, built by
`scripts/register.py build` (see "The reference sets the register" below). Its entry
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
  `bright`, `podcast`, and from the local patch `echo` for flashbacks and `pa`
  for public-address announcements; see the gotchas below.
- `take` (optional): a number that gives a line repeated word for word its own
  take. VoiceStudio keys takes by the line's voice and text, so two identical
  lines in one part would collide.
- `keep_voice` (optional): `true` stops `scripts/register.py route` from moving
  the line to another of the character's references, e.g. when one of them can't
  say a word in it.
- `line_ref` (optional): `false` keeps the character's reference for this line
  instead of its own original (see "Render").
- `text_us` (optional): the English for the American voice when it needs to
  differ, usually fuller text for a line `scripts/fit.py --voice american` calls
  `SHORT?`. `dub_anything.py` renders that voice from a copy of the script in
  `<part>.us`.
- `listen` (optional): why the user should check this line (a casting guess, a
  misheard word); `dub_anything.py` puts it in the report with the film time.

A top-level `"ready": true` tells `dub_anything.py` the part is directed
(`scripts/dub_anything.py ready <slug> <NN>`); it renders nothing before that.

A top-level `"render": {"line_ref": true, "fill": true, "dips": true, "cast_lines": true}`
records how the part is rendered (see "Render"). `scripts/dub.py generate`
writes it on the first run: all on for a part never rendered, off for a part
rendered before they became the default, so its approved takes don't change.

With `concise` timing (the default), each line starts at the window start. With
`fill` on, its pace is then adjusted so it ends where the actor stops (see
"Render"); otherwise it's spoken at its natural pace. It may run up to 0.3 s past
the end (`--overflow`), but not into the next window. Outside all windows the
original soundtrack plays.

A part with no lines at all (a commercial break recorded with the show, the
closing credits) gets a `script.json` with an empty `"lines"` list, and
`assemble.py` uses its original audio. Such a part needs no `prep`; a part
without any speech (a silent channel card) gets one from `dub_anything.py` itself
when VoiceStudio finds nothing to transcribe. A scene
entirely in a third language is the same, with its `captions` (below) still in
the subtitles.
A trailer for the next film or episode, recorded with the show, is different: its
dialogue is the same cast, so dub it like the show with the same voices. Its title
cards and on-screen text stay as they are.

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
(see "Package").

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
requests American English or both (see "Voice" under Render). The
language pair is Japanese → English unless the job says otherwise (`--source
en --target ja`, `--source en --target es`, `--source es`, `--source ko` or `--source zh`; see "Other
languages").
`scripts/dub_anything.py` runs the machine steps below for a whole video, around
your directing (AGENTS.md, "A dub, start to finish"); this section is what each
step does and how to direct it. The scripts are in `scripts/` and run from the
repo root.

### 1. Prep and draft

```powershell
& $py scripts\split.py videos\episode.mkv          # only for long videos: ~3-minute parts
& $py scripts\dub.py prep videos\episode-parts\episode-01.mp4
& $py scripts\draft.py episode-01                  # after the cast exists: --cast key,key,...
```

`scripts/split.py --whole A-B` cuts exactly at A and B and keeps that stretch as one
part. Use it for a block that stays in the original (a run of songs, a
commercial break) and at a splice between two recordings, such as a live
stream that went off air and came back with another programme. Otherwise the
~3-minute cuts can land inside the block or straddle the splice.

`split.py` prints the pause each cut found. `pause 0.00s` means it found none:
music under the speech can hide every gap from its full-mix VAD, and the cut
may then fall inside a phrase. Transcribe a few seconds around such a cut
(`asr_words.py` on an extract of the video). If it splits a phrase, re-split
with `--whole` bounds placed in a real pause. A long pause can be false too:
shouts over music hide from the same VAD, and in one anime two cuts placed in
the middle of 4–6 s "pauses" each landed on a line's first syllable.

`prep` separates the voices and prints Whisper's rough transcript. It also
switches VoiceStudio to pyannote when it's installed.

For cleaner references, try `scripts/separate.py episode-01` (its own environment;
see AGENTS.md). Residual music in the vocals can affect clones, transcripts,
register measurements and level matching. Compare the vocals and background
on your material before choosing a separator. Run separation before drafting
and building voices. `--restore` restores htdemucs stems; changing stems under
approved takes also changes their background and level measurements.

`draft.py` produces the first draft:

- a careful Whisper pass over the separated vocals;
- every phrase snapped to where Silero VAD hears speech;
- focused re-transcription of speech the full pass skipped;
- known Whisper hallucinations flagged ("ご視聴ありがとうございました" on music);
- a suggested speaker for each phrase, with score and runner-up;
- speech already in English, found by Whisper's language detection and
  transcribed in English. Forced to Japanese, Whisper silently *translates*
  English speech, so without this a scene in English reads like Japanese
  dialogue. Speech in a third language (Finnish in a film set in Finland) is
  found the same way and transcribed in that language, when Whisper is also
  surer of it there than in Japanese;
- long hallucinations re-transcribed. A stock phrase ("ご視聴ありがとうございました")
  stretched over many seconds can hide real speech. Each speech stretch is
  re-transcribed in Japanese and English; review the result in context.

Short or weak matches get a `?`, and any speech left uncovered is listed.

Whisper still mishears. Re-listen to stretches that look garbled with
`scripts/asr_words.py episode-01 <t0> <t1>`. It transcribes that window of the
clip's separated vocals with word times and prints where the VAD hears speech, and a
focused pass often gets what the full pass missed. Correct remaining errors
using surrounding dialogue, on-screen text and consistent terminology.
Keep source excerpts in `work/NOTES.md`.

`scripts/draft.py --model` and `scripts/asr_words.py --model` accept
faster-whisper models. Compare recognisers on your own corrected samples;
confidence alone does not prove that a transcription is right.

`scripts/second.py <clip>` compares ReazonSpeech and SenseVoice with the draft and
flags lines where both disagree with Whisper (for English speech, Parakeet and
SenseVoice; for Spanish, Parakeet v3 and Canary; for Korean, k2's Korean Zipformer
and SenseVoice; for Chinese, Zipformer CTC and SenseVoice). Re-listen with `asr_words.py`
and decide from context. It needs its own environment (see AGENTS.md).

Some recordings have the broadcaster's closed captions burned into the
picture. They are the best transcript there is: the exact words, the names
spelled right, a bracketed name when the speaker isn't obvious, "≪" for a
voice off screen, and often a colour of their own for each main character.
`scripts/subs.py <clip>` reads them (OCR over sampled frames) into
`work/<clip>/subs.json`. Run it before `draft.py`, which then prints the
captions on screen under each phrase and lists captions no phrase covers.
Caption times follow the display, not the speech, so windows still come from
the VAD. The OCR misreads a character now and then (a kana for its katakana
twin, a rare kanji), so weigh it against Whisper and the context. Song lyrics
become bare `♪` markers: songs stay in the original audio. A caption counts
when its text is light on a dark box, or sits on a broadcaster's
semi-transparent box, found by its edges: the picture is about half as bright
inside it. Before the second test, a box over a bright scene was dropped, and in
one anime most lines had a caption in the picture and none in `subs.json`. A
line without `cc:` still proves little: look at the frames (wide enough to read
the colour).

A film set abroad subtitles its foreign-language speech in the picture
itself: white text with a dark outline at the bottom, no box.
`scripts/subs.py <clip> --open` reads those instead (only the bottom band, 2
frames a second), and `scripts/dub_anything.py new --subtitles` runs it on every
part. The subtitles are the film's own translation, so use them for the
captions' `ja`; for the English, read Whisper's transcript of the speech as
well, which keeps puns and names the Japanese subtitle drops. A short line Whisper doesn't tag (under 1 s, or
detected without confidence) can still be in that language: the subtitle on
screen gives it away. Reading them costs about as much GPU time as preparing the
part, more than drafting it, and `--subtitles` reads every part. For a film
with only a few scenes in another language, leave `--subtitles` off and run
`scripts/subs.py --open` by hand on the parts whose draft tags that language.

### 2. Cast

Automatic diarisation is only a hint. It merges similar voices and splits
others. Decide each line's speaker from:

1. **Speaker embeddings.** They're the most reliable signal. `draft.py` scores each
   phrase against the cast using WeSpeaker, which ships with pyannote: the
   voices built from the show's own parts, or every voice while the show has
   none yet (`--cast` picks them by hand). Each
   character is represented by its voice clips plus up to 30 lines already cast
   to it in the show's other scripts, so casting gets surer with every directed
   part. A clear winner usually scores 0.5–0.9 and beats the runner-up by 0.2 or
   more. Lines under ~0.6 s are unreliable. `scripts/voices.py score <clip> a-b ...`
   scores any stretch the same way: half of a line that two people may share,
   or a window you are about to draw. For a new show or a batch of new
   characters, run `scripts/cast.py part-05 part-06 ...` first: it clusters
   every line across parts by voice, so each cluster is one person. The same
   person in a different register (a quiet monologue, a campy aside) can form a
   separate cluster, so check the nearest-cast hint. Name the clusters once,
   build their voices, then run `scripts/draft.py --recast` on those parts: it
   keeps their transcripts and only suggests the speakers again.
2. **Video frames** at the line's time: who is in close-up with their mouth open.
   On-screen names and role captions can help identify a speaker. Burned-in
   closed captions (`subs.py`) often give each main character a colour of
   their own and put a name in brackets when the speaker isn't obvious;
   `draft.py` prints them under each phrase.
   ```powershell
   & $py frames.py grid.jpg videos\episode-parts\episode-02.mp4 84:announce 121.5:interjection 147.5
   ```
3. **Pitch** (`pitch.py`, `contour.py`). Rough guide: male 80–180 Hz, female
   200–450 Hz. Actors go outside their range when playful or shouting, so a
   theatrical male lead can hit 400 Hz. Never cast on pitch alone.
4. The dialogue itself: who would say it, and who answers.

For a new character, build a voice from 5–15 s of **clean, single-speaker**
speech (see the reference-leakage gotcha below). 3–4 s works for a bit part.
Pick only clips bounded by real pauses. `scripts/voices.py spans <clip> a-b ...`
shows the VAD's speech spans around each candidate (CPU, instant): if one runs
across the boundary into another speaker's line, skip that clip.
`scripts/voices.py add <key> <clip> '<clips>' '<ref_text>'` writes the entry and
builds the joined reference, and `scripts/voices.py check <key>` transcribes it
clip by clip. For anyone with less than 3 s of clean speech (walk-ons, a single
whisper), or for a line that's a name, keep the original audio and put the
line in `captions`. Voices over a
phone are often too quiet and filtered for the VAD to hear: keep them original
and caption them too.

`voices.json` holds every show's cast, and `draft.py` only scores a show's lines
against the voices built from its own parts. A key another show already uses
(a common surname, `wife`, `boy`) is taken: `scripts/voices.py add` refuses it,
so pick another. When directors working in parallel each build a voice for what
may be the same character, `scripts/voices.py sim <key> <key>` compares the
references: the same person scores well above unrelated voices (0.5
against 0.1-0.3 in one film). Keep one, so the other doesn't pull in other
people's lines.

When one performer voices every character (a storyteller acting out a
dialogue, a narrator doing the voices), give them one voice and cast every
line to it. Per-line references carry the change of voice from character to
character, since each take follows the performer's own delivery of that
line. Separate voices for the characters would only confuse the "voice is
nearer another character" check. Build the reference from their plainest
register; it is only used for lines that can't have their own reference. Such performers often switch
characters without a pause, so many windows won't end in one: merge an
exchange into one line where the speech runs straight on. Expect Whisper's
check on each line's reference to reject more lines than usual in fast
dialect speech. Those lines fall back to the performer's reference, which
still sounds like them. Audience laughter separates mostly into the
background stem, between lines, so windows kept tight to the speech leave it
in the mix.

Pick reference lines at the speaker's usual pace, too. Every line that falls
back to the reference takes its pace, and the clips are also the first part of
each per-line reference. A presenter's read-out intro or an excited burst can
be a third faster than how that person talks, and every take on it then ends
early (in one interview, two in three lines came out short until the clips
were swapped). `fit.py` warns when a reference is more than 15% faster or
slower than the speaker's lines in the show's scripts. Swap clips before the
first render; afterwards a changed voice re-renders its lines. A swap can
overshoot: one reference 19% fast, rebuilt from formal lines only (a polite
self-introduction), came out 21% slow; one brisk clip and one formal clip
landed near the speaker's median. The warning measures each line by its
window, pauses included, so a deadpan speaker whose windows hold long pauses
and drawn-out words reads as slow: in one anime such a character stayed 42%
"fast" after his clips were swapped for slower ones, while `fit.py` flagged
only three of his lines. Go by the per-line `SHORT?` flags there, not the
warning alone.

Check the speaker of every reference clip with a frame, even when the draft
and the casting look settled. In a two-person scene the draft gave both
speakers one suggestion, and a reference built from it held two lines of the
other girl; the per-line references would have carried her voice into every
fallback line of the first.

Pick reference lines in the character's usual register. Each line's own
original normally sets its delivery (see "Render"). The character's reference
holds the voice steady, and it alone sets the delivery of lines that can't use
their own original (see "The reference sets the register").

Keep a main character's reference near 8–10 s, not the full 15. A per-line
reference is the clips, a 0.3 s pause and the line's window, and must fit
OmniVoice's 20 s: a 15-s reference (16.7 s joined) left room only for windows
under 3 s, and `fit.py` reported the rest as "over 20 s with the voice's clips".

Build references at normal speaking level, too. A murmured reference, about
14 dB below the show's other voices, scored nearer than their own voices on
most short lines in its scenes, other characters' included, and the window
check sent them all to their fallback. Its own speaker's lines scored low
against it. Rebuilt from lines at his usual level, it pulled nothing.

### 3. Translate and time

- **Translate for meaning, in the character's voice**, not word for word. Keep
  running jokes and motifs consistent across a show, and record the choices in
  `work/NOTES.md`.
- **Windows must cover all of the original speech.** Anything outside a window
  plays in the original language. Leave pure hums, laughs and eating noises
  (うーん, んー) uncovered: they're language-neutral and the actor's own.
  Sung lines stay original too, captioned if the words matter: an announcement
  over a PA can turn into a sung jingle halfway through, and a clone would
  speak it. Slow lines whose pitch holds steady notes give it away.
- **Snap windows to real speech, not Whisper's timestamps.** Whisper's word times
  run about 0.4 s early. Find true onsets with Silero VAD on `vocals.wav`
  (`faster_whisper.vad.get_speech_timestamps`, `min_silence_duration_ms≈120`) or
  with `contour.py`. `onsets.py` gives a quicker energy-based check, but it hits
  its ±0.6 s search limit on noise. The VAD misses shouted and drawn-out lines
  (an excited "Found one!", a sing-song "so yummy"), even when they're the
  loudest thing in the vocals: `draft.py` then drops the line or its end, and `fit.py` shows no actor
  end for it. The loudness of `vocals.wav` in 50 ms steps shows where such a
  line really starts and ends. Some voices it misses altogether: in one anime,
  two soft, high female voices showed no speech spans at all through ordinary
  lines held at the level of everyone else's speech. Draft windows for such a
  character are Whisper's (early), `voices.py spans` reports their clips as
  clean however they are bounded, and their reference clips have to be cut by
  loudness too. Dialogue heard from a distance, 15-20 dB below the scene's
  other voices, is better kept original with captions.
- **Split long lines at the actor's pauses** so the English lands where the lips
  move. Write each piece so it stands on its own. A take whose text starts
  mid-sentence ("goes well with the rice") can come out with a subject added
  ("It goes well...").
- **End each window in a pause, with nobody else inside it.** A line whose
  window is clean gets the actor's own delivery (see "Render"). One where the
  next speaker starts before the window ends, or the actor is cut mid-sound,
  falls back to the character's reference. In overlapping dialogue, trimming a
  window's end back to the actor's last pause is often enough. Where two lines
  run straight into each other, the voice usually still dips for a moment
  between them; `scripts/edges.py --apply` moves the shared boundary onto that dip.
- **Size the text:** each clone follows its actor's pace, from about 8 to 16
  letters per second. Short lines cost more than their length suggests: even
  "Really?" takes 0.8–1.2 s. `scripts/fit.py <clip>` computes the length
  OmniVoice will give every line not rendered yet, with OmniVoice's own rule and
  the reference the render will use: typically within 0.2 s, nine lines in ten
  within 0.4 s, about as close as two takes of one line are to each other. It
  needs no earlier renders, so it works on a new show's first part. It first
  lists the lines that will fall back to the character's reference (snap those
  windows if you can), then flags `OVER?` and `risk` against the room each line
  has, and `SHORT?` for English that ends well before the actor stops at natural
  pace. Run it before the first render and fix the flags. For `OVER?`: widen short
  windows into following silence, merge them with a neighbour, or cut words.
  Treat `risk` on a window under about 1 s as `OVER?`: short English is
  stretched (see below), so even one word takes 0.9–1.3 s, more than fill's
  fastest speed can recover. The estimate is least reliable for such one-word
  lines. It runs short on two- to four-word lines in windows up to about
  1.3 s too: in one fast-talking comedy, eight parts blocked on lines fit.py had
  passed ("Oh. She hung up.", "What is it, <name>?"), whose takes came out
  30–60% longer than estimated. Give such lines a third of slack, or keep a
  one-word reply original with a caption.
  For `SHORT?`: write fuller English, closer to everything the actor says.
  Don't count on fill to slow a short line down (see "Render"). In the
  `"american"` voice, `fit.py` estimates IndexTTS's length instead, from the
  English syllables and how long the actor speaks in the window (IndexTTS
  follows the pace of its emotion reference). There fill does slow a take, so
  `SHORT?` means the English ends early even at fill's slowest speed. Where
  `fit.py` finds no actor end (`-`: a shout the VAD missed), its American
  estimate can run short: three shouted lines of one anime passed it and came
  out 15-25% over their room, blocking the part. Give such lines a fifth of
  slack, or shorter `text_us`; a wail ("Waaah!") is drawn out. Unscripted
  talk (interviews, chat, a host's patter) is full of hedges, restarts and
  restatements. English written as tight as its meaning comes out a third
  short there, so write the first draft closer to all that was said, hedges
  included, rather than fixing it after `fit.py`.
- **Why the pace varies.** OmniVoice fixes a take's length before generating it:
  the reference's duration, times the English text's weight, divided by the
  weight of the reference's transcript. A kanji weighs 3, kana 2.2, a digit
  3.5, a Latin letter 1 and a space 0.2 (`omnivoice/utils/duration.py`).
  Anything under 2 s is stretched towards 2 s, which is why short lines run
  long. With per-line references the reference includes the line's own
  original, so a line the actor says quickly, dense in kanji, comes out fast in
  English, and one said slowly comes out slow. The same English can take very
  different times on two lines of one character; `fit.py` accounts for this.
- **Keep what's already in the target language.** If characters speak English in
  the original, leave it and caption it. `draft.py` tags such lines
  `[English]`; `scripts/asr_words.py <clip> t0 t1 --lang en` re-listens to them.
  If the video has burned-in Japanese subtitles there, use them for the
  captions' `ja`.
- **Keep a third language too.** Speech in neither Japanese nor English (the
  locals in a film set abroad) stays in the original audio with English
  captions, as an English-language film would subtitle it. `draft.py` tags it
  with its language (`[Finnish]`); `scripts/asr_words.py ... --lang fi` re-listens.
  A Japanese character speaking that language keeps their own voice there too.
- **Voice-over can be freer.** A narrator isn't on screen, so merge clipped
  narration fragments into full sentences. Short sentence fragments can be
  harder to synthesize clearly.
- **Translate the text in the picture the viewer needs** (`"signs"`, with
  `scripts/dub_anything.py new --signs` or by hand): title cards and disclaimers, the
  labels of a diagram the narration explains, a letter or a screen a character
  reads, a page put up to be read with the pause button. Leave the clutter: shop
  signs in passing, posters, number plates, the channel's logo.
  `scripts/signs.py <part>` (subs.py's environment, about 40 s of GPU per
  minute) lists candidates in `signs.json` with times and boxes, draws a sheet
  for each (`work/<part>/signs/NN.jpg`) and a thumbnail of every shot
  (`shots-N.jpg`). Read the sheets, not the OCR: it misreads stylised writing
  and misses vertical brush lettering, handwriting and arched labels altogether,
  so scan the shot thumbnails for text it missed and take the box from the
  frame. A label that repeats a word of the narration is still a label (only
  longer text matching the dialogue is dropped as a caption). A sign shows from
  cut to cut unless it appears mid-shot, but the cut detector misses a hard cut
  between two similar pictures (two pages on white), so check the times of
  anything that changes in place. Diagrams come in stages: a label can appear
  seconds into its shot. `scripts/signs.py <part> --preview [N ...]` draws the
  part's signs and dialogue into their frames as delivered: check that each sits
  by its text, reads against the picture and covers nothing that matters. A line
  of dialogue that would cover a sign (its English or its original) moves to the
  top, if the top is clear; where both are taken, move the sign's English
  instead (`place`). Lay short labels over their original without a box (the
  user's choice; a player can't switch the box on or off); keep `boxed` for
  sentences over busy writing. A page of text shown for a few seconds can't
  be read live in either language: translate it in full anyway, for the pause
  button. Leave what the original itself cuts off at the edge of the frame.
  A handwritten letter in close-up fills most of the frame: put its English in
  the clear area beside it (`place: "page"` with a `box` drawn over that area
  and `width` to fit), in short lines, rather than over the writing.
  `dub_anything.py` runs `signs.py` only when no part is waiting to render, so on
  a long video with parts marked ready, delivery waits for it after the last
  render. Run `signs.py` on the parts yourself while the renders go (it takes
  turns with them on the GPU) and place the signs meanwhile.

### 4. Render

```powershell
& $py scripts\dub.py generate episode-01 --match-level
```

If a line overflows, VoiceStudio stops at the first one. `dub.py` then prints
every line that doesn't fit, using VoiceStudio's exact rule: the whole clip
against the window, plus up to 0.25 s of the following gap, plus the 0.3 s
overflow budget. Fix them all in one pass. `fit.py` prints the same table
anytime. `generate` also warns about lines ending in "…", which tend to swallow
their last word.

Only new, changed or re-rolled lines are synthesized, so fixing an overflow
costs just the lines you touched, and timing or level changes are a re-mix of
about 5 s. (Unpatched VoiceStudio ignores partial re-renders until a job's
first render succeeds, so every failed first attempt re-rolls every take;
`patches/voicestudio-local-fixes.patch` fixes that.)

`--match-level` sets each line's gain from the original actor's level. The
clip-wide offset is matched fully and each line's deviation by half, so
whispers stay quieter than shouts. Without it, dialogue comes out about 10 dB
louder than the film mix.

`--steps` defaults to 32, OmniVoice's quality preset; VoiceStudio's own default
is 16. `--effect` picks the voice processing. The default is `broadcast`;
compare alternatives on your source. `dub.py` records what each line was made
with (`work/<clip>/rendered.json`), so changing steps, effect or instruct
re-synthesizes exactly the affected lines.

By default (the script's `render` block) each take follows the actor's own
delivery of its line and ends where the actor stops:

- **Per-line references** (`line_ref`). Each line's reference is the voice's
  clips from `voices.json`, a pause, then the actor's original of this very
  line, with its `ja` after the voice's transcript. OmniVoice copies its
  reference's delivery and accent, so the take follows the actor on that line
  and keeps their accent, while the clips hold the voice steady. This replaces
  sorting lines into calm and raised: the delivery comes from the line itself.
  Dubbing into Japanese it's off by default (see [languages/ja.md](languages/ja.md)).
  A line keeps the character's reference when it has `"line_ref": false`, a
  room effect (`echo`, `pa`; the original's room would be cloned), a window too
  long to fit OmniVoice's 20 s limit alongside the clips, or a window that isn't
  clean. A window isn't clean if speech runs on past its end: the reference would
  end mid-sound, and OmniVoice finishes that sound at the start of the take.
  With `dips` (the `render` block's third switch) that doesn't count where the
  voice dips 20 dB below the line's level within 0.15 s of the end, too briefly
  for the 50 ms pauses speech detection needs: a turn change in fast dialogue,
  or the actor's own tail. The line's original is then cut at the dip. In a
  quick family argument this gave 5 of 13 run-on lines their own reference;
  the rest overlapped for real. Nor is a window clean if Whisper hears
  something other than the line's `ja` (crosstalk), or if the voice is nearer
  another character. With `cast_lines` (the `render` block's fourth switch) that
  last test needs two votes: the line must be nearer another voice's reference
  *and* nearer another character's cast centroid (its clips plus up to 30 lines
  cast to it in the show's other parts, as `draft.py` scores them). A short
  reference can miss most of a character's range: in one anime whose young voices
  all scored about 0.5 against each other, the references alone sent 45% of the
  lines to their fallback, a whole confession included ("nearer" a girl who had
  one line in the scene); with both votes, 20%. The comparison reads katakana as hiragana, but a word
  Whisper writes in kana where `ja` has kanji still counts against it: write
  `ja` the way it is said when a short line falls back only for that. `generate` prints each line that
  falls back and why. A line whose window overlapped the next speaker came out
  as gibberish on every take until this check excluded it, so don't force
  one. Snap its window instead (see "Translate and time"). A take's reference
  depends on its window, so moving a window re-synthesizes that line.
- **Fill** (`fill`). After the natural-pace render, `generate` finds each actor's
  last sound in the window (Silero VAD on the vocals). It then re-renders every
  take ending more than 0.25 s away from it, at the OmniVoice speed that lands it
  there, within 0.8–1.25. OmniVoice sets the take's length from the speed;
  nothing is time-stretched. Two passes converge, and a line that overflows at
  its natural pace gets sped up instead of stopping the render. Fill only
  aligns where a line ends, not pauses inside it. Lines stuck at 0.8 have
  English much shorter than the actor's speech (`fit.py` flags them `SHORT`):
  give them fuller text, and listen to slowed lines for drawl. A take asked
  for at 0.8 often isn't slower at all: OmniVoice says the words at its own
  pace and leaves the extra time silent, and the silence is trimmed. Only
  fuller text fills those lines. Fill also absorbs most natural-pace
  overflows, which otherwise stop the render and each need a reword or a wider
  window. A take that still overflows at 1.25 gets a fresh take on the next
  pass, since takes of one line differ by a few tenths of a second. If it
  overflows after the last pass, `generate` stops and lists it. In the
  `"american"` voice a slowed take really is slower (see "Voice"), so there
  fill can make up for English somewhat short of the actor.
  `"predict_speed": true` in the `render` block (or `--predict-speed`) renders
  each new line at the speed `fit.py`'s prediction asks for instead of at
  natural pace first. That halves the re-renders but gave the same timing
  (see [research.md](research.md)), so it is off by default.

`--no-line-ref` and `--no-fill` (or `--line-ref`, `--fill`, `--predict-speed`)
override the script for one run, e.g. for an A/B job. Both change takes, so try them on a copy of an
approved part rather than on the part itself.

**Voice** (the `render` block's `"voice"`, or `--voice`) is how the English
sounds. Use the actor's native accent by default; use American English or both
when the user requests it.

- `"actor"` (the default): OmniVoice clones each actor, so the English carries
  the actor's accent. Timing is tightest and the sound cleanest, and per-line
  references give each take the actor's delivery.
- `"american"`: IndexTTS 2.5 speaks native American English, following each
  line's original as an emotion reference (below), and `vc.py` then converts
  every take to the actor's timbre with Chatterbox's voice converter.
  `generate` does both. Fill judges timing on IndexTTS's own takes (kept as
  `src_en_<id>.wav` in the job); once the timing has settled, the new takes
  are converted and the part re-mixed. Fill is exact here: speed reaches
  IndexTTS as its duration factor, which stretches a take without changing
  its pitch, and each take keeps its seed, so a re-timed take is the same
  performance and one pass lands it on the actor (with the local patch). It
  is slower (about 5 s a line, plus conversion), and conversion re-generates
  some of the intonation, a little lower in pitch. Per-line references don't
  apply.

When requested, `scripts/dub_anything.py new --voices both` renders
every part in each voice and delivers both tracks, the actor's playing by
default. In trials, `"american"` came out closer to the actors than the clones on an
animated series. On a live-action drama it stayed less like the actors than the
clones, and conversion cost some of the native accent. Record the choice in
`work/NOTES.md`; the job and each part's `render` block keep it, so later parts
follow it and approved parts don't change. Two things that look like options
aren't. OmniVoice's `american accent` instruct tag doesn't change the accent
while the reference carries the actor's, which per-line references always do.
IndexTTS without the conversion speaks American English but doesn't sound like
the actors.

`--engine indextts2 --emo-alpha 0.9` alone renders with IndexTTS 2.5 instead of
OmniVoice, without the conversion. Each line gets the actor's own line from the separated
vocals as an emotion reference, and the voice still comes from `voices.json`.
The engine is VoiceStudio's global setting: `dub.py` switches it for the render
and switches back afterwards. It needs the local patch, which adds the
per-line reference, and it's slow, about 8 s a line against OmniVoice's 1 s.
The reference carries the actor's pacing as well as their delivery: without
it, IndexTTS runs well past the actor and overflows many windows. On a scene
with a wide emotional range, its takes followed the actors' delivery more
closely than any OmniVoice approach. On a quiet scene there was little to
follow and no gain. The costs are likeness, which is lower than OmniVoice's,
and the accent. OmniVoice keeps the actor's accent from the reference, while
IndexTTS speaks its own native English. Whisper's language confidence on the
takes measures this cheaply: clones of non-native actors score well below
native speech. A listener may find the actor's own accent more authentic, so
render it as a separate job and compare by ear.

How these were chosen, and what else was tried, is in
[research.md](research.md).

### 5. QA

- `scripts/qa.py episode-01` transcribes the finished MP4 in English and lines
  it up with the script. Word timestamps drift by a line or so in quiet
  passages, so a word appearing in the neighbouring window is usually fine.
  Whisper also invents words on background noise; check `energy.py` before
  worrying.
- **Check each rendered line on its own too:** `scripts/lineqa.py episode-01`. The
  full-mix QA hid these problems:
  - junk at the start of lines (reference leakage, below)
  - dropped words in fast lists
  - meaning flips like "can" heard as "can't" (any change in negation is flagged)
    or "seventy" as "forty" (so is any number heard as another)
  - words a voice can't say
  - lines that came out silent

  Numbers, units, contractions and "Dr." are normalized, so formatting doesn't
  count as a failure. Triage each flag. A random glitch (a dropped word, a
  mangled first word) usually clears with a fresh take:
  `scripts/lineqa.py episode-01 --reroll` queues the flagged lines, and the next
  `generate` re-rolls them.
  A word the voice can't say needs rewording. Every take is a new roll of the
  dice, so re-check after each render. A fresh take can also come out worse
  than the one it replaced: `--reroll` keeps each replaced take
  (`work/<clip>/takes/`), and the next check puts the best one back when the
  new take transcribes worse. The next `generate` then only re-mixes. What
  Whisper heard in each take is kept by the take's hash
  (`work/<clip>/heard.json`), so the check after a round listens only to the
  re-rolled takes, and one after a re-mix loads no model at all: on a film's
  first parts a 34-line check after a round heard one take, and one after a
  re-mix took 5 s instead of about 50-200 s per check before. A line that
  fails on its fresh take too is *settled*: it gets no more rolls (later
  rounds almost never fixed one, [research.md](research.md), finding 22) and
  goes to the report for rewording or the listener. In the
  `"american"` voice most flags come from the conversion rather than from
  IndexTTS, so when IndexTTS's own take of a flagged line says it right,
  `--reroll` queues that take to be converted again instead (conversion is
  random too), keeping its performance and timing; after two failed
  conversions it gets a fresh take. Judge each flag against the intended
  text and the audio. Names and specialist vocabulary can confuse QA even when the take is
  correct; flag uncertain cases for the listener. On a feature film, the first
  round cleared the real glitches; later rounds mostly re-rolled names and
  dropped fillers ("Hmm", "huh") that ended up accepted anyway. So reword a line
  that fails twice rather than waiting on another roll: a one-word line
  misheard on every take ("Fine." heard as "Find.") usually clears with one
  more word of context ("That's fine."). Leave names to the listener. Reword
  at once a line QA hears as profanity or a slur, even if the take is probably
  right: a listener can hear it the same way. A
  hesitation ("Um...", "Well...") or a one-word answer that fails twice goes
  back to the original audio, with a caption if it says anything. A take that
  starts with the line's own Japanese (QA's "leading extra" in Japanese, the
  same on every take) is the per-line reference leaking: give the line
  `"line_ref": false`. When the American voice's conversion keeps garbling a
  line that IndexTTS said right, other words in `text_us` usually clear it.
- Listen. Machines can't judge acting.

### 6. Package

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

When any part has `signs`, `assemble.py` also writes the English track styled
(`.en.ass`: the dialogue, with each sign placed by its text, `ass.py`) and the
signs alone (`.signs.ass`), and `deliver.py` delivers Matroska,
`<name> (dubbed).mkv`, in place of the source: MP4's subtitles can't be placed.
The English tracks are then the styled ones, and the signs-only track follows
them, marked forced, for watching the dub with subtitles off. A player has to
render ASS subtitles to place them (mpv and VLC do, with libass).

Every subtitle track, a dub's included, is timed for reading as it's assembled
(`subtitles.py`, below), and a line longer than about 42 letters is broken in
two near its middle.

## Subtitles only

`scripts/dub_anything.py new <video> --voices none [--source .. --target ..]
[--signs ...]` makes a job that subtitles the video instead of dubbing it. Any
two languages with a module will do: `PAIRS` lists the dubs checked end to end,
and a subtitle needs only the source heard and the target written. Nothing is
cloned or rendered, so the VoiceStudio backend isn't needed, and `run` doesn't
start it:

- **Prep** (`dub.py prep --subtitles-only`) only takes each part's audio out
  into a job folder of VoiceStudio's shape (`local_<id>` beside its own jobs).
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

## Other languages

This guide was written from Japanese → English dubs. For any other pair, read
the notes on both its languages in [languages/](languages/README.md) as well:
what differs with each as the source and as the target
([ja.md](languages/ja.md), [en.md](languages/en.md),
[es.md](languages/es.md), [ko.md](languages/ko.md), [zh.md](languages/zh.md)). Write what you learn about a
language there.

## Gotchas

- **Takes can only be replaced in a job that has rendered.** On a job's first
  render VoiceStudio synthesizes every line, whatever files are already in the
  job folder; after that it re-mixes the takes it finds. So post-processing a
  take (`vc.py`) happens after a successful render, and a take made elsewhere
  can't be dropped into a fresh job.

- **Voice-reference leakage (the big one).** If a clip in `voices.json` cuts a
  word off, or includes someone else's voice, OmniVoice "finishes" it at the
  start of *every* line in that voice. Incomplete references can leak extra
  syllables into each take. Clips taken
  straight from Whisper timestamps are cut off because those run early. Snap
  clips to real speech boundaries, keep neighbours out, then check the joined
  reference with `scripts/asr_words.py work\voices\<key>.wav`. Every phrase must be
  complete, and `ref_text` must match what is actually said.
- **Keep screams out of voice clips.** Speech detection doesn't count a scream
  or a wail as speech, so a clip can look like a short word with silence around
  it while it holds seconds of screaming. That blurs the voice's speaker
  embedding: other characters' lines, even of the other sex, then score nearer
  it, and the per-line reference check sends them to their fallback. Build
  references from spoken lines, even for a character who mostly shouts.
- **A click just before the English starts = the actor chopped mid-sound.** At
  each window start VoiceStudio crossfades from the original mix, voice
  included, to the voice-free bed in only 10 ms. Speech detection misses soft
  attacks (a breath, a consonant) by 20–100 ms. If the actor is already
  sounding when the window opens, that sliver of Japanese plays and is then
  cut. It's rare enough to sound random ("sometimes the start is glitchy").
  `scripts/edges.py <clip> --apply` finds these and starts the window 20 ms
  before the true attack. It only moves clear cases: a voice within 16 dB of the
  line at the cut that began 20–130 ms before the window. Shorter fragments weren't
  heard, and longer ones are usually someone else's sound, except where the
  sound rises out of silence: the separated vocals hold zeros (or nearly)
  between phrases, so a sound that begins there and runs unbroken into the
  window is the line's own start, and `edges.py` moves the window up to 400 ms
  then. In one film the drafts started 150–400 ms late on many lines this way,
  the actor at full level at the cut. It's timing only, so
  the re-render is a re-mix and approved takes stay. `draft.py` applies the same
  rule to new drafts. Windows you set by hand bypass it, so run `edges.py`
  after writing a script. A `skip` with a long shift (150–300 ms) and a voice
  near the line's level at the cut is often the same speaker starting early.
  Confirm it with `asr_words.py` (Whisper's start plus ~0.35 s) and move it by
  hand. Leave it when the sound could be someone else, such as a sigh louder
  than the line itself. The fix is best checked by ear on the flagged lines.
  Window ends are the mirror image: a voice still sounding when the window
  closes plays a clipped tail of the original after the English. `edges.py`
  moves the end past short tails (same rule, so up to 400 ms into silence) and notes the rest for manual review.
  Where two windows touch, it moves their shared boundary onto the dip between
  the voices when there is one (see `dips` under "Render"). In a noisy room (a
  bar, a crowd) the separated vocals keep some background chatter: long `skip`s
  with a voice at or above the line's level there are the room, not the actor.
- **Match reverberation to the source.** `scripts/tails.py <clip>` compares the
  original voice tail with the rendered take. Music or a following sound can
  look like reverb, so review the scene before choosing an effect. `echo`
  provides heavy reverb with a 1 s tail; `pa` provides public-address tone
  and reverb with a 0.5 s tail. These are starting presets: compare them on
  your material. Extend the speech window over the original tail without
  covering the next speaker. Effects lengthen the take, which `fit.py` does
  not predict; leave room and check the rendered duration.
- **Timing strategy.** Use `concise`, dub.py's default: natural pace, placed at the
  window start, and it fails loudly on overflow. `strict_slot` (the VoiceStudio UI
  default) stretches every line to fill its window, which gives slowed or rushed
  speech.
- **Words a cloned voice cannot say.** A fresh take can fix a random
  glitch; persistent pronunciation problems need a clearer phrase, an
  expanded abbreviation, or a different reference. Keep production-specific
  examples in `work/NOTES.md`. A trailing ellipsis can swallow the last word.

- **Repeated phrases collapse.** Give each repetition its own line at
  the speaker's pauses, with `"take": n` so each gets a distinct take.
- **Short replies need care.** Allow room after short lines where possible.
  If a very short acknowledgement has no room and repeated takes fail,
  retain the original audio and add a caption.
- **No acting direction with OmniVoice.** VoiceStudio's `direction` field and
  free-text `instruct` values (such as "excited") crash the dub. Its only style
  tag is `whisper`. Delivery comes from the reference, which is why each line's
  reference ends with the actor's own original of it (per-line references).
- **The reference sets the register.** A clone copies delivery as well as
  voice identity. With per-line references this matters only for lines that
  fall back to the character's reference. When many of a character's lines
  fall back and they sound wrong, e.g. a character whose reference is projected
  but whose fallback lines are quiet asides, give them a reference in their
  register with `register.py`:
  1. `report` measures pitch, vocal effort and loudness relative to each
     speaker's own lines and flags mismatches.
  2. `build <voice> calm|raised` builds another reference from clean lines
     with matching voice and transcript. It requires at least 5 s of speech
     and a register difference of at least 0.8.
  3. `route <clip> --apply` selects a reference only when it is nearer by
     at least 0.5. Only moved lines need new takes.
  4. Render and QA. `fit.py` learns each reference's pace. Use
     `"keep_voice": true` for lines that should keep a particular reference.
  5. `check` verifies that the takes deliver the intended register;
     `drop <voice>` retires a reference that fails this check.

  Loudness in the mix is not the same as vocal effort. Review both the
  source and the takes; these tools do not provide emotion control.
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
  when a render runs out of memory. With both voices, the backend keeps the
  American voice's engine loaded after its render, and the next part's actor
  render ran out of memory on a long per-line reference about one part in six;
  the restart cleared it each time. A program outside the dub using the GPU
  (another project's job) makes the retry fail too: two failures block the
  part, and the next `scripts/dub_anything.py run` retries it.
- **After an NVIDIA driver update, restart the backend.** A backend that was
  running during the update keeps "CUDA out of memory" errors on an empty GPU
  and silently runs Whisper on the CPU.
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
  there too.
- **Speaker detection.** Automatic diarisation can merge similar voices or
  split one speaker. Confirm casting with multiple signals. Sortformer can't
  handle clips over 120 s or more than 4 speakers, ignores
  count hints, and has CC-BY-NC weights.
- **VoiceStudio bugs (fixed by `patches/voicestudio-local-fixes.patch`).** Dubbing
  transcription ignored the source-language hint, so short Japanese clips were
  heard as English (`backend/api/routers/dub_core.py`). OmniVoice loaded in
  float16 on CPU, where it is about 7× slower (`backend/services/model_manager.py`).
  A render that stopped at a timing overflow didn't record its takes as
  reusable, so the next one re-synthesized everything
  (`backend/api/routers/dub_generate.py`). Reverb presets stopped dead where
  the dry take ended, cutting their own tail (`backend/services/audio_dsp.py`,
  which also gains the `echo` preset). With ROCm's PyTorch on Windows, MIOpen
  tuned every new convolution shape for minutes and attention took PyTorch's
  slow math path (`backend/services/model_manager.py`; see
  [setup.md](setup.md#amd-radeon-rocm-pytorch)). IndexTTS's venv probe
  rejected a ROCm venv and the backend fell back to a CUDA venv on the CPU,
  and s2mel ran in fp32, 4-5x slower than bf16 on a Radeon
  (`backend/engines/indextts/`). All are worth reporting upstream.
- **On an AMD GPU, check that IndexTTS is on it.** "CPU mode" in `backend.log`
  after the sidecar spawns means it is not (setup.md); a line then takes about
  a minute.

## Tools

The loop, in the order it runs:

| Script | Does |
|---|---|
| `scripts/dub_anything.py new\|run\|status\|ready\|report\|cleanup` | A whole dub (or with `--voices none`, subtitles): runs every machine step below in order, renders each part when its script is ready, QA rounds, delivery, the report |
| `scripts/split.py <video>` | Cut a long video into ~N-minute parts at pauses in speech (frame-accurate) |
| `scripts/dub.py prep <video> [--source en\|es\|ko\|zh --target ja\|en\|es] [--subtitles-only]` | Upload, separate, transcribe, diarize → `work/<name>/segments.json` (with the language pair), source-language `.srt`; `--subtitles-only`: the audio alone, no backend |
| `scripts/separate.py <clip> ... [--restore]` | Re-separate prepped clips with Mel-Band Roformer: far less music left in the vocals, so cleaner clones (own environment; see AGENTS.md) |
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
| `scripts/fit.py <name> [--voice american] [--read]` | Each line's length against its window and the actor's speech, predicted before rendering with OmniVoice's own length rule (or IndexTTS's, for the `"american"` voice): `OVER?`, `risk`, `SHORT?`. A part to subtitle (or `--read`): each cue's reading speed, `FAST`, `LONG`, `SPLIT?` |
| `scripts/dub.py generate <name> --match-level` | Render `script.json` in the chosen voice (per-line references and fill by default, or `"american"`), levelled against the original actors → MP4 + SRT |
| `scripts/lineqa.py <name> ... [--reroll]` | Transcribe each rendered line alone (Japanese compared by its reading in kana, Spanish without accents); flag leaked, dropped or garbled words and flipped meanings; queue fresh takes |
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
