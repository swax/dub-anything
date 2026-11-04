# Roadmap

## Where it stands

Dub Anything dubs Japanese, Spanish and Korean video into English end to end, and
English video into Japanese or Spanish (the actor's voice only; see each
language's notes in [languages/](languages/README.md)):

- **Transcript and cast.** Parts are separated with Mel-Band Roformer and
  transcribed by Whisper, with two other recognisers as a second opinion and
  burned-in captions read where a recording has them. Lines are cast by
  speaker embeddings across the show.
- **Voice and timing.** The actor's own voice: OmniVoice clones each actor,
  prompted with their delivery of each line (per-line references). Or American:
  IndexTTS converted to the actor's timbre. Either way, each take is re-timed
  to end where the actor stops (fill), and `fit.py` predicts each voice's
  lengths before the first render.
- **Checks and packaging.** Every take is transcribed back and cached. A flagged
  line gets one fresh take, then stays with the director for rewording or
  review. The dubs go into the source file itself, renamed "(dubbed)": one
  dub track per voice, the original
  audio, and subtitles in both languages.
- **Running it.** `dub_anything.py` runs a whole dub around the agent's directing:
  the machine steps in order, each part rendered as soon as its script is
  ready, one GPU job at a time, delivery and a report of what to listen for.
  The agent directs; on a long video, helper agents can direct batches of
  parts against the show's notes.

How each piece was chosen, and the measurements behind it, are in
[research.md](research.md). The tools are listed in
[directing.md](directing.md#tools).

## Next, by payoff for what the listener hears

1. **Steadier short lines.** Lines under about 1.5 s cause most overflows, fill
   re-renders and garbled takes; one-word lines and stutters are the worst,
   and conversion makes them worse still. A re-roll no longer loses a better
   take (`lineqa.py --reroll` keeps each one and puts the best back).
   An optional candidate workflow is now implemented in `shorttakes.py`: keep
   the current natural baseline, generate fixed seeded alternatives, and compare
   their actual safe speech timing and transcription without changing the
   wording, voice, references or render rules. It archives audio, settings and
   attempt costs, and resumes completed attempts without regenerating them.
   See [directing.md](directing.md#optional-short-line-candidates) for its archive, measure and apply
   steps.

   A prospective trial found an occasional automatic validity gain and closer
   endpoints, alongside lines that failed every candidate. Extra candidates
   required substantially more generation work than the control's conditional
   fresh takes. Routine renders keep the existing behavior. Next: review the
   private blind comparisons before deciding where the extra cost is worthwhile;
   listening preference, acting and likeness remain unmeasured. Detailed results
   stay in ignored `work/` files (research.md, finding 23).
2. **More lines on their own reference, and pauses inside lines.** In
   overlapping dialogue many lines fall back to the character's reference.
   Cutting a reference at the dip between two voices (`dips`) and snapping
   touching windows to it recovered about a third of the run-on lines in fast
   argument, though in a blind listen the recovered lines sounded about the
   same as on the character's reference (research.md, finding 14). So press on
   only where fallbacks are frequent and heard. That could be reduced further by:
   - a transcript per voice clip, for long lines;
   - a kana-aware window check (katakana already reads as hiragana; kanji
     against kana still fails a short line, and needs a reading dictionary).

   Fill aligns only where a line ends, so an actor who pauses mid-sentence
   still gets continuous English.
3. **Captions, second pass.** A Japanese-trained OCR model, which would misread
   fewer kana and kanji. Turn caption colours and bracketed names into speaker
   suggestions in `draft.py`.
4. **Keep the words and the intonation through conversion.** The American
   voice's converter garbles about one line in twelve that IndexTTS said
   right; converting again fixes about half, and the rest need fresh takes
   (research.md, finding 15). It also re-generates part of the pitch contour.
   A converter that kept both while still gaining likeness would keep
   IndexTTS's acting; Seed-VC's pitch-preserving model kept the contour but
   gained no likeness.
5. **By ear, as listeners flag them:**
   - line edges: fragments `edges.py` skips, and VoiceStudio's 10 ms
     crossfade;
   - room acoustics per scene, starting from `tails.py` and the `echo` and
     `pa` presets.

## Later

- **Lip sync** (LatentSync / MuseTalk); feasible on a decent NVIDIA GPU.
- **Other language pairs and hardware.** What a pair beyond Japanese ↔ English,
  Spanish ↔ English and Korean → English, or other hardware, needs is listed in
  the README, "Other languages and hardware".
- **A native Japanese or Spanish voice** for dubs into those languages, the
  counterpart of the American voice (research.md, open questions).
- **Upstream to VoiceStudio:**
  - the fixes in `patches/voicestudio-local-fixes.patch`, the `echo` and `pa`
    presets, and per-segment voice and emotion references in dubs;
  - a Roformer separator as an option next to htdemucs;
  - a fix for free-text `direction`/`instruct` aborting OmniVoice dubs.
