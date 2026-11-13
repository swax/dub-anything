# Prep, draft and cast

Part of [directing a dub](../directing.md): read its file formats and workflow overview first.

## 1. Prep and draft

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
The whole-block choice controls splitting, not whether its picture needs
translation. Record the audio and on-screen-text decisions separately at intake.

`split.py` prints the pause each cut found. `pause 0.00s` means it found none:
music under the speech can hide every gap from its full-mix VAD, and the cut
may then fall inside a phrase. Transcribe a few seconds around such a cut
(`asr_words.py` on an extract of the video). If it splits a phrase, re-split
with `--whole` bounds placed in a real pause. A long pause can be false too:
shouts over music hide from the same VAD, and in one anime two cuts placed in
the middle of 4–6 s "pauses" each landed on a line's first syllable.

Review both sides of cuts beside an original-audio block, too, especially the
last spoken sentence before credits. A closing shot or music cue can begin
before the actor finishes a clause. Check the source words and image rather
than assuming the whole-block boundary is a speech boundary. Repair a bad cut
before generating dialogue, preserve the earlier preparation, and update the
manifest and both neighboring scripts together. If a voice reference came
from an affected part, preserve its exact materialized audio and provenance;
rebuilding from newly separated stems can change the voice despite identical
clip times.

`prep` uploads the part without letting VoiceStudio separate it, separates the
voices with `separate.py` (Mel-Band Roformer), attaches those stems to the job,
and only then has VoiceStudio transcribe and diarize them. It prints Whisper's
rough transcript and switches VoiceStudio to pyannote when it's installed.

In a sandbox, FFmpeg may be unable to write to Python's temporary directory even
when Python can create it. If frame extraction reports a permission error, set
`TEMP` and `TMP` to an existing, writable folder inside `work/` before running
the tools. This keeps temporary frames in the workspace; the extraction error
is different from a timestamp past the video's end.

Residual music in the vocals can affect clones, transcripts, register
measurements and level matching. To compare another separator model, run
`scripts/separate.py episode-01 --model <model>` (its own environment; see
AGENTS.md) before drafting and building voices. Parts prepped before prep
separated them may still have VoiceStudio's htdemucs stems; `--restore` puts
those back. Changing stems under approved takes also changes their background
and level measurements. An American-voice copy shares its part's stems by hard
link: `separate.py` replaces a stem file, never rewrites it, so re-separating one
leaves the other as it was.

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
Keep source excerpts in the job's `NOTES.md`.

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
`<part>/subs.json`. Run it before `draft.py`, which then prints the
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

## 2. Cast

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
  In a workspace with many other shows, `scripts/cast.py ... --cast key,key,...`
  limits the nearest-cast comparisons to the selected voices, saving the cost of
  embedding unrelated references. Every supplied part's lines still take part
  in the clustering.
2. **Video frames** at the line's time: who is in close-up with their mouth open.
   On-screen names and role captions can help identify a speaker. Burned-in
   closed captions (`subs.py`) often give each main character a colour of
   their own and put a name in brackets when the speaker isn't obvious;
   `draft.py` prints them under each phrase.
   An open mouth in a wide group shot can be a reaction or laughter while
   someone else speaks off-screen. Compare several frames and complete clauses
   against checked references. A voice key named for clothing in an earlier
   scene does not establish who wears that item in a later scene.
   ```powershell
   & $py frames.py grid.jpg videos\episode-parts\episode-02.mp4 84:announce 121.5:interjection 147.5
   ```
3. **Pitch** (`pitch.py`, `contour.py`). Rough guide: male 80–180 Hz, female
   200–450 Hz. Actors go outside their range when playful or shouting, so a
   theatrical male lead can hit 400 Hz. Never cast on pitch alone.
4. The dialogue itself: who would say it, and who answers.

Credits identify performers, but do not map every brief appearance to a role.
A credited character seen later does not prove that a similar-looking speaker
in an earlier scene is the same person. Build a reference from the uncertain
speaker's own clean speech and keep a separate key until frames and voice
comparisons support joining it to an existing character.

For a new character, build an identity sample from 5–15 s of **clean,
single-speaker** speech (see the reference-leakage gotcha in [gotchas](gotchas.md)). 3–4 s works
for a bit part. In a Japanese → English actor dub, give recurring characters a
separate complete three-to-five-second conversational fallback (see [Render](render.md)).
The longer identity sample can help casting without being prepended to every
line or setting the delivery of unsafe lines.
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
original normally sets its delivery (see [Render](render.md)). The identity reference
supports casting; the selected fallback supplies the voice and delivery when
a line can't use its own original (see "The reference sets the register" in [gotchas](gotchas.md)).

With a shared prefix enabled, keep its reference near 8–10 s, not the full 15.
The combined reference is the clips, a 0.3 s pause and the line's window, and must fit
OmniVoice's 20 s: a 15-s reference (16.7 s joined) left room only for windows
under 3 s, and `fit.py` reported the rest as "over 20 s with the voice's clips".

Build references at normal speaking level, too. A murmured reference, about
14 dB below the show's other voices, scored nearer than their own voices on
most short lines in its scenes, other characters' included, and the window
check sent them all to their fallback. Its own speaker's lines scored low
against it. Rebuilt from lines at his usual level, it pulled nothing.
