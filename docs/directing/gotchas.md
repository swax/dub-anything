# Gotchas

Part of [directing a dub](../directing.md): read its file formats and workflow overview first.

The traps of directing: references, windows, takes and delivery. The backend's and the
hardware's are in [machine.md](machine.md).

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
  complete, and `ref_text` must match what is actually said. Per-line references
  must contain the rebuilt voice too: `core.line_refs` fingerprints its audio
  in the cached filename, so an old joined WAV cannot survive a voice correction
  under the new reference transcript.
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
  the voices when there is one (see `dips` under [Render](render.md)). Caption windows
  protect speech kept original: automatic moves stop before crossing them,
  including captions with empty English text. Review existing overlaps yourself;
  a genuine interruption may need both voices, while an overlong window can
  cut off a short original reply. In a noisy room (a
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
- **Compare retained audio from matching samples.** Decode both streams from
  the same starting point and crop the PCM afterward, or align them before
  comparing. Independent fast seeks can shift the starting samples or resampling
  phase, lowering correlation even when the retained audio is intact.
  For near-silent windows, compare uncentered peaks and RMS over the same central
  interval, excluding any lag-search padding. Check both stereo channels before
  calling a mono window near-silent: phase cancellation can hide stereo energy.
  Report verified near-silent intervals separately from measured correlation and
  lag; neither measurement is meaningful without a signal.
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

  Check voice continuity across adjacent takes as well as each take's words.
  One correctly cast speaker can sound like different people when successive
  sentence references pull the clone into different deliveries. On a copy,
  compare an unchanged clean character reference across the passage with the
  sentence references, holding the seed and other controls fixed. Speaker
  embeddings can support the comparison; the listener decides whether the
  voice stays consistent. Recheck timing and handoffs before adopting a trial.

  Loudness in the mix is not the same as vocal effort. Review both the
  source and the takes; these tools do not provide emotion control.
