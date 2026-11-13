# Render

Part of [directing a dub](../directing.md): read its file formats and workflow overview first.

## 4. Render

```powershell
& $py scripts\dub.py generate episode-01 --match-level
```

For repeated take or timing repairs that do not yet need a new video, add
`--no-export`. This still generates takes and the backend mix. Run once without
that flag before reviewing the finished part, assembling or delivering: the
previous MP4 and subtitles remain stale until a fresh export. Each export
creates another timestamped video under the VoiceStudio job's `exports/`, even
when the downloaded part MP4 is overwritten. Include those backend copies in
the job's eventual storage cleanup.

If a line overflows, VoiceStudio stops at the first one. `dub.py` then prints
every line that doesn't fit, using VoiceStudio's exact rule: the whole clip
against the window, plus up to 0.25 s of the following gap, plus the 0.3 s
overflow budget. Fix them all in one pass. `fit.py` prints the same table
anytime. `generate` also warns about lines ending in "…", which tend to swallow
their last word.

The following gap is measured against the next dubbed line. When retained
original speech starts sooner (including a `captions` entry), check the actual
take's end against that start; the allowed overflow does not prove that the
reply stays clear. These lengths include trimmed edge silence. For a small
nominal crossing, inspect the generated speech end and the next original
voice's actual onset before rewriting; put any remaining uncertain handoff
on the listen list.

Account for the backend's leading-silence trim when placing raw take timestamps.
For retained speech, compare the source and actual export at the onset itself.
A correlation check that excludes a generous cloned-audio mask cannot establish
that the excluded beginning survived. Conversely, quiet padding can cross a
nominal boundary while the actual reply remains intact; check the waveform
before requesting another take.

Check the next generated English voice's onset too. A take can fit the overflow
budget yet collide with the next English line. Compare both speech boundaries
with the original exchange: shorten the outgoing line if the dub turns a
source pause into overlapping speech. A source interruption can be kept as a
directing choice.

After a successful render, unchanged published takes are reused; changing timing
or level alone is a quick re-mix. New takes from a failed or cancelled render
are not published. The integrated fork keeps upstream's atomic publication
safeguard, so the former patch's reuse of a failed first render's new takes is
retired. Resolve predicted overflows before starting the first render.

A clone can take longer than a very brief source reaction even for two words.
Group adjacent fragments by the same speaker when they form one turn and no
other voice is inside the combined window. This gives the performance room
without widening it across somebody else's reply.

`--match-level` sets each line's gain from the original actor's level. The
clip-wide offset is matched fully and each line's deviation by half, so
whispers stay quieter than shouts. Without it, dialogue comes out about 10 dB
louder than the film mix.

`--steps` defaults to 32, OmniVoice's quality preset; VoiceStudio's own default
is 16. `--effect` picks the voice processing. The default is `broadcast`;
compare alternatives on your source. `dub.py` records what each line was made
with (`<part>/rendered.json`), so changing steps, effect or instruct
re-synthesizes exactly the affected lines.

By default (the script's `render` block) each take follows the actor's own
delivery of its line and ends where the actor stops:

- **Per-line references** (`line_ref`). Each clean line can use its actor's
  original audio and matching source transcript as the reference. OmniVoice
  copies the reference's delivery and accent. For new Japanese → English actor dubs, set
  `"render": {"line_ref": true, "ref_prefix": false, ...}` and use the current
  line alone when clean, with a conversational fallback for unsafe lines.
  Existing scripts keep the shared prefix by default: the voice's clips from
  `voices.json`, a pause, then the current line, with their transcripts joined.
  Those clips are intended to stabilize identity, but the engine receives one
  combined reference without separate roles. The prefix can compete with the
  current line's performance. The original trials established the benefit of
  adding per-line speech, not of retaining the prefix versus using the line
  alone (see [research finding 6](../research.md#findings)). A line's own
  `"ref_prefix": true` can opt back into that combined reference. Generation and
  timing prediction use the same choice, and source-only cache identities
  include the original audio and transcript. This changes conditioning, so
  compare on an isolated job before changing an approved dub.
  Dubbing into Japanese it's off by default (see [languages/ja.md](../languages/ja.md)).
  A line keeps the character's reference when it has `"line_ref": false`, a
  room effect (`echo`, `pa`; the original's room would be cloned), a window too
  long to fit OmniVoice's 20 s limit with any enabled prefix, or a window that isn't
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
  one. Snap its window instead (see [Translate and time](translate.md)). A take's reference
  depends on its window, so moving a window re-synthesizes that line.
- **Fill** (`fill`). After a successful baseline render, `generate` finds each actor's
  last sound in the window (Silero VAD on the vocals). It then re-renders every
  take ending more than 0.25 s away from it, at the OmniVoice speed that lands it
  there, within 0.8–1.25. OmniVoice sets the take's length from the speed;
  nothing is time-stretched. With published takes to measure, two passes usually
  converge. Fill only aligns where a line ends, not pauses inside it. For a
  meaningful internal beat, direct separate complete phrases at the source's
  phrase starts, keeping their own clean source references where enabled.
  Check every generated phrase and its following onset: segmentation alone
  does not make a take fit. Preserve complete words and nonverbal responses;
  do not cut a word or add meaning to manufacture a pause. The engine's prompt
  and output silence processing can shorten gaps, so supplying the original
  recording does not guarantee the same pauses in English.

  A `SHORT?` prediction is evidence to review, not proof of undertranslation.
  Very quiet references are a particular hazard: OmniVoice raises their level
  before silence removal, while the empirical duration predictor omits that
  gain. The two preparations can retain quite different amounts of speech.
  Check source words, pauses and a measured take before expanding a complete
  translation or replacing a clean reference. Applying the gain to every
  prediction has not established a general improvement; keep actual fit and
  protected onsets as the delivery gates.

  A matching controlled audition can supply an actual duration before the
  whole part renders. Use that evidence when the predictor is clearly wrong,
  keeping the engine, wording, reference and generation settings in view.
  Shorten an overlong translation faithfully or choose a justified pace; leave
  the earlier trial and its failed fit unchanged. Another engine's duration
  does not establish how this engine will fit.

  For lines stuck at 0.8, check whether the English omits meaning or needs a
  more natural full expression before writing fuller text; do not pad a
  complete translation to satisfy `SHORT`. Listen to slowed lines for drawl. A take asked
  for at 0.8 often isn't slower at all: OmniVoice says the words at its own
  pace and leaves the extra time silent, and the silence is trimmed. A complete
  translation may legitimately leave a source pause; expand it only when the
  meaning or natural phrasing calls for more. With a published baseline, fill also absorbs most natural-pace
  overflows, which otherwise stop the render and each need a reword or a wider
  window. A take that still overflows at 1.25 gets a fresh take on the next
  pass, since takes of one line differ by a few tenths of a second. If it
  overflows after the last pass, `generate` stops and lists it. In the
  integrated backend, an unsuccessful pass leaves the last published takes
  intact. `fit.py` can therefore show those older takes fitting even when the
  rejected pass overflowed. For an automatic OmniVoice pace adjustment, the
  client can reuse the published pace when the reference and other settings
  still match and the take fits the current room. It resubmits the mix for
  backend validation. Explicit paces and forced regenerations are never
  replaced with an older cached success. If recovery is unavailable, use the
  reported line number to revise its text or checked window. After reviewing a
  fitting natural take, `"speed": 1.0` can keep its pace fixed for later passes.
  In the `"american"` voice a slowed take really is slower (see "Voice"), so there
  fill can make up for English somewhat short of the actor.
  `"predict_speed": true` in the `render` block (or `--predict-speed`) renders
  each new line at the speed `fit.py`'s prediction asks for instead of at
  natural pace first. A prediction can speed up a new take, but slowing waits
  until a published take supplies its measured length. Use it for a new part when its natural-rate takes may
  overflow: atomic publication leaves no new takes for fill to measure after
  a failed first pass. With no published baseline, fill stops at that failure
  instead of repeating the same whole-part synthesis without a measurement.
  The earlier comparison halved re-renders with similar
  timing (see [research.md](../research.md)); it remains an explicit render option.

`--no-line-ref` and `--no-fill` (or `--line-ref`, `--fill`, `--predict-speed`)
override the script for one run, e.g. for an A/B job. Both change takes, so try them on a copy of an
approved part rather than on the part itself.

For reference comparisons, keep the English, seed, speed, engine and effects
fixed within each pair. Preserve the exact reference audio and transcript,
raw outputs and generation settings. Use level-matched playback copies and
ask the listener to judge delivery and voice likeness separately. Keep these
trials distinct from the preview's fitted takes; different timing or wording
would confound a reference-only comparison.

Record a listener's declared overall preference as one vote per pair, with
delivery and likeness unspecified. Selecting the same take in two controls
does not establish two independent judgments when the listener used both for
overall sound. Preserve the original feedback and annotate its interpretation.

Equal active-speech RMS can still leave perceptual loudness differences. Check
loudness and peaks separately when a listener reports jarring variants, and
keep their report of harsh or projected delivery in the evidence. Preserve the
reviewed audio; any revised playback copies need a separate version so saved
votes remain tied to what the listener heard.

Check a very short original reference with an actual take before choosing its
fallback from duration alone: verify the words, complete source boundaries and
generated handoffs. When the original contains overlapping speakers, use a
checked clean reference from the same speaker's nearby conversation, or retain
the original with a caption. Keep those exceptions explicit so an automatic
fallback cannot silently replace the source-only policy on substantial lines.

Review fallback lines together by reference as well as in their surrounding
dialogue. A clean reference still needs to suit the scene's delivery. Several
jarring takes sharing one fallback are a reason to audition that reference's
register before changing the engine or every line's settings. A source rejection
for speaker mismatch also calls for an attribution check; substituting the
assigned character's reference does not resolve that casting uncertainty.

For fallback duration, OmniVoice recommends starting with
[3–10 seconds](https://github.com/k2-fsa/OmniVoice#voice-cloning) and warns that
longer audio can reduce cloning quality. A duration comparison should use
shorter and longer cuts from the same clean conversational passage; changing
the passage and its length together cannot isolate the effect of duration.

For a conversational fallback, a complete utterance around four seconds is a
useful starting candidate. Preserve natural word boundaries; three to five
seconds is preferable to cutting a word to meet an exact duration. Assess
neutrality relative to that speaker's ordinary conversation and recording
conditions. Screen pitch, pitch variation, level, speaking rate and spectral
balance separately for extremes; quiet whispers and unnaturally flat speech
also need review. The existing register measurements are heuristic screening
features, not a validated neutrality classifier. Audition shortlisted references
through the same small set of translated lines in context, then keep the
approved fallback consistent through the scene. Clean per-line Japanese
references still supply their own original performance.

Treat casting and fallback selection as separate decisions. A clean sample that
establishes the speaker's identity still needs a generated audition before it
becomes a recurring fallback. For an unreviewed recurring voice, try a small set
of target-language lines: a statement, a question and a short reply in context.
Check complete words, articulation, perceived pace, pitch and transitions to the
neighboring dialogue. Keep the ordinary conversational sample as the baseline;
choose a different register when the scene supports it. Reserve uncertain
choices for a small post-dub listening page rather than blocking the whole dub.

When comparing candidates, hold the requested text, seed, configured speed,
steps and effects fixed, and match playback loudness. The same speed setting
can still produce different perceived pace and clarity with another reference.
Record an overall listener preference as an overall preference; it does not
separately establish emotion transfer or voice likeness. Keep qualified or tied
preferences qualified, and verify wider changes on more than one line or seed.
Label each trial's reference choice and the delivered line's actual reference:
a preferred fallback audition does not evaluate an own-reference production take.

For narration, choose a complete clean sample in the narrator's register. A
separate three-to-five-second narration fallback can serve a short rejected
line while ordinary dialogue keeps its conversational fallback.

`scripts/neutral_refs.py speaker_1 episode-01 episode-02 --out work/neutral.json`
ranks complete, source-checked three-to-five-second phrases near four seconds.
It needs at least eight clean phrases for a speaker-relative baseline. Pitch,
pitch range, spectral balance, level, level range and a rough source-text rate
contribute separate distances; large outliers receive an extra penalty. An
optional `--benchmark work/jobs/<slug>/voices/approved.wav` compares candidates with a
listener-approved reference instead of treating the statistical center as the
desired performance. The report is a shortlist, not an emotion verdict.

After reviewing the source and complete phrase boundaries, add
`--pick episode-02:18.20:22.30 --key speaker_1_neutral4` to build a separate
reference. The tool verifies its transcript and speaker similarity before
adding the key. In the script, use
`"render": {"line_ref": true, "ref_prefix": false, "fallback_voices":
{"speaker_1": "speaker_1_neutral4"}}` alongside the other render settings.
Clean lines retain their own source audio and cache identity. Fallback takes
include the selected reference's fingerprint, so changing that reference
regenerates those takes and recomputes their pace. `fit.py --predict` likewise
estimates from the new reference instead of measuring an incompatible old take.
This fallback routing applies to the actor voice; the American voice keeps its
existing IndexTTS reference path.

Check pace independently of the acoustic ranking. A dense transcript in a short
reference can make the engine assign too little time to faithful translated
speech. Recover a clean own window first; for an unavoidable fallback, compare
another complete conversational phrase near the scene's usual pace. Keep its
reference consistent through the scene. Do not invent dialogue or repeated
words to fill a timing estimate. Inspect whether a long detected source span
includes laughter or another speaker, then check the actual take and both
handoffs before changing its text or pace.

Transcript agreement alone does not establish a useful acting reference. A
complete sentence followed by laughter can pass both the transcript and speaker
checks, yet condition a markedly different voice. Keep laughter outside the
reference where a clean, complete speech boundary permits it; otherwise compare
the same speaker's conversational fallback in context. Preserve deliberate
dramatic delivery rather than treating every expressive line as contamination.

Borrowed gaps and overflow budgets can carry an English take into an original
laugh even after the reference window is corrected. Check the actual take's end
and its audible tail against the original sound being protected.

New source checks save recognized word margins in `refs/source-review.json`.
`fit.py` and generation print a reference-review prompt when at least 0.8 seconds
and a quarter of the reference follow its last recognized word, with at least
0.35 seconds of audible energy after a 0.25-second alignment allowance. Quiet
window padding does not prompt a review. This adds no transcription pass and
does not trim, reject or reroute the reference. The
untranscribed tail may contain laughter, residual noise, crosstalk or speech the
recognizer missed; inspect it before deciding. Cached checks made before this
report existed have no timing evidence and produce no such prompt.

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
  performance and one pass lands it on the actor (with the pinned fork). It
  is slower (about 5 s a line, plus conversion), and conversion re-generates
  some of the intonation, a little lower in pitch. Per-line references don't
  apply.

When requested, `scripts/dub_anything.py new --voices both` renders
every part in each voice and delivers both tracks, the actor's playing by
default. In trials, `"american"` came out closer to the actors than the clones on an
animated series. On a live-action drama it stayed less like the actors than the
clones, and conversion cost some of the native accent. Record the choice in
the job's `NOTES.md`; the job and each part's `render` block keep it, so later parts
follow it and approved parts don't change. Two things that look like options
aren't. OmniVoice's `american accent` instruct tag doesn't change the accent
while the reference carries the actor's, which per-line references always do.
IndexTTS without the conversion speaks American English but doesn't sound like
the actors.

`--engine indextts2 --emo-alpha 0.9` alone renders with IndexTTS 2.5 instead of
OmniVoice, without the conversion. Each line gets the actor's own line from the separated
vocals as an emotion reference, and the voice still comes from `voices.json`.
The engine is VoiceStudio's global setting: `dub.py` switches it for the render
and switches back afterwards. It needs the pinned fork, which adds the
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
[research.md](../research.md).

### Optional short-line candidates

`shorttakes.py` is an explicit actor/OmniVoice experiment; the normal runner
does not invoke it. Choose zero-based line indices before looking at generated
results. Eligibility is at most 1.5 seconds from the first to last detected
source speech inside the directed window, not the window or WAV duration.
The part must stay unready and have a complete normal concise render from the
current client. Keep its wording, voices, reference bytes and render rules fixed.

For a controlled comparison, archive the first render before normal QA repairs:

```text
python scripts/shorttakes.py example-01 --lines 1,3 --archive-baseline
```

Then complete normal isolated QA, its conditional fresh take and best-prior-take
restoration. Capture that control with `--capture-control`; `--control-cost`
accepts a JSON manifest of the generation and QA cost receipts. Restore the
common first render with `--restore-baseline` before measuring the trial:

```text
python scripts/shorttakes.py example-01 --lines 1,3 --candidates 3
```

Three means the archived first take plus two extra seeded takes. Each extra uses
a private clone profile with identical reference bytes/text and a persisted seed.
Fixed seeds disable native multi-row batching. Generation, profiles, settings,
hash-bound hearings and selection evidence stay under the part's `shorttakes/`;
matching resumes reuse them. A changed source, wording, reference or policy
invalidates that comparison. Repeated identical lines need distinct `take` fields.
An initial render that cannot assemble counts as a producer failure. Preserve
its takes, logs and costs; declare a repaired comparison context that records
reused takes before continuing.

Selected-only generation also saves a durable before-state and command/task
outcome receipt. A lost POST response or event stream can leave the backend
writing after the client exits. An unknown outcome stops the trial and blocks
generation, capture, restore and apply for that part. Keep the active receipt
and before-state; explicit reconciliation needs terminal backend evidence and
verified audio/settings restoration. Process exit, elapsed time and successful
metrics alone do not establish that the writer has stopped.

The selector rejects transcription/meaning failures, missing speech, clipping,
underfill and audio that reaches the next protected English or retained-original
turn. Speech end determines endpoint quality; a reverb tail cannot fill missing
words. Audible tails still count against the protected boundary. Validity comes
first, then endpoint error; timing alone must improve by at least 100 ms to
replace a valid baseline. Ties retain the baseline, then the lower ordinal.
Correct transcription cannot establish acting or voice likeness.

Blind preference can disagree with both transcription scores and endpoint
rankings. Keep an acceptable baseline when the only proposed gain is a closer
endpoint; review the performance before applying that replacement. A preferred
take can still have an unresolved word error. Record preference and correctness
separately, and use extra candidates to address supported defects.

Measurement restores the baseline. `--apply` explicitly installs the measured
choice and matching metadata; `--apply-control` restores the archived normal
repair result. Exit 2 reports unresolved lines, with evidence retained. Rephrase
or correct a checked window rather than generating candidates indefinitely.
After installation, check actual fit, current-take transcription, handoffs and
the finished mix as usual. If a later fill or repair changes a take, its archived
selection is no longer proof of the current audio.

For experiment costs, `dub.py generate` and `lineqa.py` accept `--metrics PATH`.
Use distinct files per attempt, including failures. Report common first-render
costs, control repairs, trial extras, QA, remixes and restoration separately;
GPU reservation, backend-task and model timings overlap and must not be added
as active GPU compute. Ordinary behavior remains the default pending evidence.
Both commands also accept `--only` for zero-based indices. Generation requires
every current line ID to have a complete natural-rate baseline; after changing
wording, use ordinary incremental generation to create its new ID. Selected-only
QA leaves the other lines' repair counters alone.
