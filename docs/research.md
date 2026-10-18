# Research notes: making the clone act like the actor

What was measured and tried to get from "the right voice saying the right words"
to "the actor's performance, in English", and why the pipeline now works as it
does. The trials used the Japanese→English dramas and travel shows this toolkit
was built on. Their content and exact numbers stay in the local `work/NOTES.md`.

## How the variants were compared

Each idea became a **variant**: the same part rendered with the same script
(identical English), in a separate VoiceStudio job, so approved takes never
moved. Every variant got the same treatment: overflowing lines re-rolled, then
a small overflow allowance if needed, then `lineqa.py` re-rolls. Variants were
rendered twice where it mattered, because two renders of the same settings
differ. A listener then compared the parts by ear, and their verdict decided.

A change that touches only some lines is easier to judge line by line. A new
VoiceStudio job synthesizes every line on its first render, even with the
takes copied in, so the variant's other lines differ too. Splice instead:
give the variant the part's stems, render it, then cut each changed line's
window (plus the 0.3 s a take may run on) from the variant into the delivered
mix. Inside a window only that line's take plays over the shared bed, so
everything else in the clip stays as delivered. Outside the windows, two such
mixes matched to within codec noise. Present each pair blind, with the
original to compare against.

The instruments, all run against the actor's own original line
(`eval/score.py` for delivery, likeness, quality and accent; `eval/timing.py`
for timing):

| What | How |
|---|---|
| Delivery | Per line: pitch (semitones), vocal effort (spectral tilt, 1–4 kHz against 0.1–1 kHz) and loudness, combined relative to that actor's own lines (`register.py`'s measure). Rank correlation between each take and its original, per character. |
| Likeness | WeSpeaker similarity of each take to the actor's own lines, and whether the nearest actor is the right one. |
| Timing | The actor's last sound in each window (Silero VAD on the separated vocals) against where the take ends. "Dead air": the actor still talking after the English stopped. "Overrun": the reverse. |
| Quality | torchaudio's SQUIM (reference-free PESQ and SI-SDR estimates), and the level of the quietest frames inside the speech (noise between words). |
| Accent | Whisper's confidence that a take is English (native speech scores near 1; a clone carrying the actor's accent scores well below), plus an English-accent classifier. |
| Intelligibility | `lineqa.py`. |

Pick test material with range. The first trial scene was quiet, so every
variant followed the actor about equally and the test showed little. A scene
whose delivery swings from murmurs to shouts separated the variants clearly.
`register.py`'s arousal measure finds such scenes: take the spread of the
originals per part.

## Findings

1. **Likeness was not the gap.** Clones were identified as their own actor
   about as reliably as the actor's original lines were.
2. **Delivery was.** With one reference per character, the takes' intensity
   barely followed the originals. OmniVoice copies the delivery of its
   reference, so every line got the reference's delivery. Calm/raised
   references (`register.py`) helped the characters who had enough clean speech
   for them, and only as far as two buckets can.
3. **Timing was the other gap.** Concise English at the clone's natural pace
   ended well before the actor stopped on a large share of lines, and no check
   flagged it: `fit.py` only warned about lines running long.
4. **Cleaner separation cleans the clones.** References cut from Roformer
   vocals gave takes with less noise between words and better estimated
   quality than htdemucs references, with no change in delivery.
5. **IndexTTS 2.5 with an emotion reference acts best, but loses the actor.**
   With each line's original as its emotion reference, it followed the actors'
   delivery most closely of everything tried, on a scene with range; on a quiet
   scene it had nothing to add. The reference also carries the actor's pacing:
   without it the takes ran long and overflowed. But it speaks its own native
   English instead of the actor's accent, its likeness is lower, and it is
   several times slower. The listener found the actor's own accent more
   authentic, so it stays an option (`--engine indextts2 --emo-alpha`), not the
   default.
6. **A per-line reference does most of that within OmniVoice.** Each line's
   reference became the voice's clips followed by the actor's original of that
   line. The takes followed the actor's delivery clearly better than with the
   character's usual reference, most for characters the calm/raised references
   hadn't helped, while keeping accent and likeness. The risk is leakage: a
   window with another speaker in it, or cut mid-sound, produced gibberish on
   every take. A check on each window (speech running past its end, Whisper
   hearing something else, the voice nearer another character) now excludes
   those. In overlapping dialogue it excludes many lines, which then keep the
   character's reference.
7. **Filling the actor's time is what the listener heard most.** Re-rendering
   takes at 0.8–1.25× so each ends where the actor stops cut the dead air
   substantially, and cut the overruns too. The listener preferred the
   variant with per-line references and fill over every other. They did not
   prefer per-line references without fill over the plain dub. The rhythm of
   the exchange — lines taking the time the actor takes, turns landing where
   they did in the original — mattered at least as much as the acting. Most
   re-timed lines were slowed, and many hit the 0.8× floor: the English was
   written much shorter than the original speech.
8. **Fill carries the timing, per-line references carry the delivery.** On a
   new show, one part with range was rendered three ways from the same script:
   plain, fill only, and fill with per-line references. Fill alone halved the
   distance between the take's end and the actor's, as much as the full
   default did. But its delivery followed the originals no better than the
   plain dub: the rank correlation stayed near zero. Only per-line references
   made it clearly positive. Likeness, estimated quality and accent were about
   the same in all three. So the preference in finding 7 isn't timing alone, and
   the listener confirmed it by ear: of the three, the one with per-line
   references and fill sounded best. The plain variant also needed the old
   chores: five lines overflowed at natural
   pace and had to be reworded or widened, and re-rolls changed short takes'
   lengths enough to overflow again. Fill absorbed all of that.
9. **OmniVoice decides a take's length by rule.** Its length is the
   reference's duration scaled by the weight of the English against the weight
   of the reference's transcript, with a stretch for anything under 2 s (see
   [directing.md](directing.md#3-translate-and-time)). With per-line
   references each line's own original enters the rule, so pace varies from
   line to line. `fit.py` used to estimate lengths from each voice's average
   rate on lines already rendered. That needs renders first, and it can't
   follow per-line pace: on a new show's first parts, nine in ten of its
   estimates were within 0.6 s, and they were biased long. Now it applies the
   rule itself, to the reference each line will get: over about 1,000 takes,
   typically within 0.2 s, nine in ten within 0.4 s, with no bias and no
   renders needed. That is about as close as two takes of one line are.
   Takes speak only about half of the stretch given to short lines. A take
   asked for at another speed changes length by about speed^0.7, not in
   proportion: slowed, the model often speaks at its own pace and leaves
   silence that is trimmed. On the lines where slowing mattered most, it added
   almost nothing. So `fit.py` flags `SHORT?` against the natural pace and
   gives fill no credit. Before, it caught three of the lines that ended at the
   0.8× floor on a new show's first render; now it catches thirteen of
   seventeen.
10. **Windows written to end in pauses keep most lines on their own
    reference.** When every window was ended in a pause while writing the
    script, rather than snapped afterwards, about nine lines in ten passed the
    window check on a dialogue-heavy show, against about seven in ten before.
11. **Choosing each line's speed from the prediction saves renders, not
    timing.** On a third show, three parts of an ensemble scene (about 110
    lines) were rendered twice from the same script, each in its own job. One
    used the current fill: natural pace first, then re-timing. The other
    rendered each new line at the speed the length prediction asked for, and
    corrected afterwards by speed^0.7. The prediction put about two lines in
    three within fill's tolerance on the first pass, where natural pace managed
    about one in three. Fill passes re-rendered about half as many lines, and
    the whole render made about 15% fewer takes. Everything measured on the
    result was the same:
    the distance between each take's end and the actor's, lines ending late,
    `lineqa.py` flags, delivery, likeness and estimated quality. Wall time saved
    little: the first pass dominates, and every run has fixed costs. It also
    had a cost. Short takes vary so much from roll to roll that the speed^0.7
    correction over-steered them. One short line swung from well under the
    actor's time at 1.25× to far over it at 0.8×, and three times as many
    lines ended at the 0.8× floor. So fill still
    starts at natural pace; the prediction is a switch (`predict_speed`). Both
    variants also stopped on lines still overflowing at 1.25×, sometimes by a
    hundredth of a second, because fill never re-rendered a take already at
    its fastest speed. Now such a take gets a fresh roll.

12. **Burned-in captions are the best transcript, and a casting signal.** A
    fourth show, an animated series, was recorded with the broadcaster's closed
    captions burned into the picture. Read by OCR (`subs.py`), they were on
    screen for nine dubbed lines in ten. Whisper's draft was more than a quarter
    wrong on about one line in eight: names swapped for other names, a key word
    heard as a similar-sounding one, slang it didn't know, and three stock
    "thank you for watching" phrases over silence. The captions had most of
    those right. They also carried the cast: the lead's lines were in a colour
    of their own on most of them, bit parts too short for speaker embeddings
    had their name in brackets, and an off-screen or inner voice was marked.
    Where the embeddings' first choice and the final cast disagreed (under one
    line in ten), the captions usually decided. In one gag, the colour showed
    that a line the context gave to the lead was the other character's. The OCR
    misreads a character now and then, so captions and Whisper are read side by
    side, and window timing still comes from the VAD.

13. **American English in the actor's voice takes conversion, not a tag.** On
    two shows, an animated series and a live-action drama, one scene was
    rendered five ways from the same script:
    * the clones (the actor's accent);
    * the clones with OmniVoice's `american accent` tag;
    * IndexTTS, speaking native English with each line's original as its
      emotion reference;
    * IndexTTS followed by one of two voice converters, Seed-VC and
      Chatterbox, which change only the timbre, towards the actor.

    The tag changed nothing measurable: the reference carries the actor's
    accent, and with per-line references it always does. IndexTTS came out
    native (Whisper was about 0.97 sure it was English, against 0.5–0.6 for the
    clones) and followed the delivery at least as well as the clones, but it
    was the least like the actors. Conversion won likeness back while keeping
    most of the accent. On the animated series it went past the clones; on the
    drama it recovered about half the gap. Its costs:
    * Part of the intonation is re-generated, a little lower in pitch.
    * IndexTTS's takes end short of the actor more often, so timing is looser
      than with the clones.

    Chatterbox (MIT) kept the sound quality closer to the clones'. Seed-VC
    (GPL) kept slightly more likeness, but it made shouted lines hoarse, and its
    pitch-preserving model gained no likeness at all. A longer, pooled target
    of the actor's speech didn't help either converter. OpenVoice made voices
    less like the actors. Some characters converted better than others, so the
    genre result is a tendency, not a rule. The listener kept the clones for
    the drama and chose conversion (Chatterbox) for the animation. It is now
    the `"american"` voice.

14. **Most fallbacks in fast dialogue are a pause too short to see.** A line
    keeps its own reference only when its window ends in a pause. On one
    show's cache of window checks, 50 of 58 fallbacks were speech running on
    past the window's end, and only 8 were Whisper hearing something else
    (reading both sides as kana would have saved 4 of those). In a quick
    family argument, 13 of 44 windows ran on. At 6 of them the voice still
    dipped 20–32 dB below the line's level within 0.15 s of the end, too
    briefly for speech detection's 50 ms pauses; at the rest it dipped only
    5–18 dB, a real overlap. Cutting each reference at such a dip (`dips`)
    and moving touching windows onto it gave 5 of the 13 their own reference.
    On the earlier show's run-on ends the dip was always deep (37–70 dB): the
    window had cut the actor's own tail, and the pause followed it.
    Over a whole feature film, `dips` gave 61 of about 1,190 lines their own
    reference (5%). In a blind A/B of all 61 against the takes the old rule
    makes (the character's reference), the listener heard them as close, with
    a preference that went either way line by line. The rule stays: it costs
    nothing and keeps those lines on the actor's delivery, but on its own it
    isn't an audible gain.

15. **The American voice's loose timing was a dropped setting.** VoiceStudio
    passed fill's speed to OmniVoice but not to IndexTTS, so a take fill asked
    to slow down came back at natural pace: on the animated series, takes
    asked for at 0.8 spoke no more slowly per character than takes at 1.0,
    and only 55% of lines ended within 0.25 s of the actor. IndexTTS 2.5 has
    its own control, a duration factor that sets the length of the take's
    acoustic stage, so a take stretches exactly while its words and pitch stay
    put. Speed now goes there, and each take carries a seed, so re-timing a
    line keeps its performance instead of drawing a new one. On a feature
    film rendered both ways from the same script (about 1,200 lines), one
    fill pass then put 92% of the American takes within 0.25 s of where the
    actor stops, against 80% for the clones (median distance 0.09 against
    0.13 s), with half the overrun.
    IndexTTS has no length rule to read, but its takes follow the pace of
    the emotion reference: a take's length is about 0.33 s, plus 0.10 s per
    English syllable, plus 0.44 s per second the actor speaks in the window.
    Fitted on about 780 takes from two shows, that is typically within
    0.2 s and nine in ten within 0.5 s, and a fit on either show predicts the
    other about as well; the English alone predicts it worse (0.26 s, 0.68
    s). With fill able to slow a take, `fit.py` flagged only 3 lines of the
    film as too short for the American voice.
    Most of the American voice's QA flags come from the conversion, not from
    IndexTTS: on the film, 133 of 1,240 lines were flagged on the first
    pass, and IndexTTS's own take said three in four of them correctly.
    Converting the same take again fixed about half of those within three
    tries (conversion samples noise, but the speech units it starts from are
    fixed, so some garbles repeat); fresh takes and the odd accepted name
    left 1% flagged after four rounds. The converter also now and then turns
    the silence around a take into sound, a stray "and" before the line,
    which once made a take longer than its window, so converted takes are now
    silenced outside the original take's speech.
16. **Into Japanese, the per-line reference carries the English accent.** A
    live-action sketch was dubbed from English into Japanese (32 lines, three
    clones) twice from the same script, with per-line references and
    without. Whisper's confidence that a take is Japanese held at 0.99 for the
    narrator either way, but the most expressive and most strongly accented
    actor's takes fell from a median 0.96 without her own English original in
    the reference to 0.53 with it, half of them under 0.5. Without per-line
    references the takes followed the actors' arousal more closely, not less
    (rank correlation 0.84 against 0.58), matched their voices as well
    (WeSpeaker 0.53 against 0.52) and ended as near the actor (median 0.06
    against 0.05 s). The first QA check flagged none of them against three
    with per-line references. Dubbing into Japanese now leaves per-line
    references off (`PAIRS` in `scripts/languages/`). One sketch is little to go on:
    it awaits a listener, and a calmer show could come out the other way.
    OmniVoice's length rule held for Japanese takes as it did for English:
    with each take's reference and speed, a median error of +0.02 s, 84%
    within 0.2 s and 97% within 0.4 s.
17. **From Spanish, the per-line reference helps, as from Japanese.** The
    first ten minutes of a telenovela (three parts, 116 lines, seven clones,
    fast dialogue) were rendered twice from the same scripts, with per-line
    references and without. With them the takes followed the actors' arousal
    and effort more closely (rank correlation 0.46 and 0.44 against 0.36 and
    0.26), matched their voices a little better (WeSpeaker 0.54 against
    0.52) and sounded a little cleaner (SQUIM PESQ 2.20 against 2.15, SDR
    11.9 against 11.1 dB). Whisper's confidence that the takes were English
    hardly moved (0.78 against 0.80): the Spanish original didn't carry a
    heavier accent into the English, as the English one did into Japanese
    (finding 16). Both ended as near the actor (0.11 s on average), and
    without per-line references two lines overflowed their windows at fill's
    fastest speed. So Spanish → English keeps them on. OmniVoice's length
    rule held here too: with each take's reference and speed, a median error
    of +0.02 s, 77% within 0.2 s and 91% within 0.4 s.
18. **Into Spanish, the per-line reference helps too.** A live-action sketch
    was dubbed from English into Spanish (31 lines, four clones) twice from the
    same script, with per-line references and without. Unlike into Japanese
    (finding 16), the English original in the reference cost the Spanish
    nothing: Whisper's confidence that the takes were Spanish was 0.60 with
    it and 0.61 without, low either way as the clones keep their actors'
    English accent. With it the takes followed the actors' pitch far more
    closely (rank correlation 0.40 against -0.05; arousal and effort were
    level, near zero and 0.2 on 24 scored lines), matched their voices as
    well (WeSpeaker 0.58 against 0.57), sounded cleaner (SQUIM PESQ 2.46
    against 2.28, SDR 14.4 against 12.2 dB) and passed the first QA check
    without a flag, against two (one line heard half in English). Without it
    the takes ended a little nearer the actor (median 0.10 against 0.14 s)
    and fewer needed fill's fastest speed (5 against 8). So English →
    Spanish keeps them on. OmniVoice's length rule held, better with them:
    with each take's reference and speed, a median error of +0.06 s, 71%
    within 0.2 s and all within 0.4 s (without them +0.14 s, 45% and 84%).
    One sketch is little to go on; it awaits a listener.
19. **From Korean, the per-line reference helps too.** The first 12 minutes
    of a live-action drama (four parts, 107 lines, five clones: narration, a
    formal hearing, quick banter, a meeting) were rendered twice from the same scripts, with per-line references and
    without. With them the takes followed the actors' arousal, pitch and
    effort far more closely (rank correlation 0.40, 0.47 and 0.43 against
    0.01, -0.01 and 0.20), a wider gap than from Spanish (finding 17), most
    for one character (arousal 0.80 against 0.17). They matched
    the actors' voices a little better (WeSpeaker 0.54 against 0.52), and
    Whisper's confidence that the takes were English didn't move (0.58
    against 0.59): the Korean original carried no heavier accent into the
    English. Estimated quality (SQUIM PESQ 2.73 against 2.77, SDR 17.0 against
    17.2 dB) and timing (mean distance from the actor's end 0.13 against
    0.12 s; eight takes more than 0.25 s off against ten) were level, and 103
    of the 107 lines kept their own reference. The first QA check
    flagged 6 lines against 4, setting aside four flags that came from
    problems fixed before the second variant rendered. So Korean → English
    keeps them on. OmniVoice's length rule held: with each take's reference
    and speed, a median error of +0.00 s, 70% within 0.2 s and 93% within
    0.4 s (without them -0.01 s, 75% and 94%). It awaits a listener.

20. **Chinese (Mandarin) → English: keep per-line references.** A short sample
    exercised preparation, Chinese transcription and independent recognition,
    casting, English cloning, fitting, per-line QA, assembly and verified
    delivery. The same script rendered without per-line references also passed
    English QA, but keeping them improved measured voice likeness and timing
    without weakening recognition of the English. OmniVoice's length estimates
    held for the pair. The sample had too little speech per character for the
    acting correlations used in the earlier comparisons, so it establishes a
    working pipeline and a starting default, not an acting verdict. The actor
    voice was exercised; other Chinese varieties and the American voice await
    tests. Source-specific measurements and listening notes stay in ignored
    `work/` files. It awaits a listener.

21. **English → Cantonese: references help delivery, but speech needs review.**
    A short Windows sample completed the actor-voice pipeline in both reference
    modes, including independent recognition, separation, signs, fitting,
    rerolls, assembly and verified delivery. Per-line references followed the
    actors' pitch and intensity more closely, with slightly better measured
    likeness and similar timing. Without them, reference-free audio-quality
    estimates were better. Both variants still produced mispronounced words
    and English-like phonetics, especially on names and short stylized lines;
    an independent recognizer confirmed problems that spelling normalization
    could not explain. Whisper generally detected Chinese rather than
    Cantonese in both, so its Cantonese language-ID probability alone does not
    establish fluency. Keep per-line references as the starting default and
    compare by ear. Length estimates were useful, but rerolls still required
    a shorter reply and a wider closing pause. Source-specific measurements
    and listening notes stay in ignored `work/` files. It awaits a listener.

22. **QA rounds cost more than they fix: hear each take once, and stop after
    one fresh take.** On a 90-minute film (28 parts, one voice) the three QA
    rounds after each part's render took 139 checks averaging 52 s (about two
    hours) and 92 rounds of fresh takes; most parts used all three rounds, and
    the lines still flagged at the end were names and near-misses. Two changes,
    tried on the next film of the same length:
    - `lineqa.py` keeps what Whisper heard in each take, keyed by the take's
      hash, and loads Whisper only when a take is new. A check after a round
      heard 1-5 takes instead of the part's 11-34, and one after a re-mix
      loaded nothing (5 s). The film's 56 checks took 32 s each on average (30
      minutes in all, against two hours); hearing itself was 136 s for 311
      takes, and the rest was waiting for the GPU behind directors' tools and
      loading the model, so what remains to win is in the queue, not in
      Whisper.
    - A flagged line gets one fresh take; if that fails too, it is *settled*:
      listed for the report and the director, with no further rolls
      (`SETTLE_TAKES` in `lineqa.py`). On the first parts rendered with three
      rounds, the first fresh take fixed 5 of the 7 lines it re-rolled, and
      rounds 2-3 fixed none of the other 2: one, a common noun, came out garbled
      differently on every take, a word the voice can't say, which a stricter
      "heard the same way twice" rule would have kept re-rolling. Rewording such
      a line fixes it; another roll doesn't. With the rule on, the film took 22
      rounds (18 first rounds; the 4 later ones before or at the switch)
      against the earlier film's 92, and 13 lines settled; the director
      reworded the garbled ones after delivery (each re-render about 20 s, QA
      about 10 s, once the GPU was free) and left names and fillers ("Aw"
      heard as "Oh") to the listener.
    Two kinds of flag were the check's own: Whisper writes a letter grade
    "B-minus" as "B-" and "eighth" as "8th" (both now normalised,
    [languages/en.md](languages/en.md)). A filler "nah" heard as "now", on the
    other hand, is a real flip worth rewording.

## What the pipeline does now

`dub.py generate` renders new parts with per-line references, fill and `dips` by
default, recorded in each script's `render` block (see
[directing.md](directing.md#4-render)). `fit.py` predicts each line's length with
OmniVoice's own rule before the first render. It flags `SHORT?` lines, whose
English is too short for the actor's speech at natural pace, so the director
writes fuller text before rendering. `register.py`'s calm/raised
references are now a fallback for characters whose lines can't use their own
original. A show can instead take the `"american"` voice: IndexTTS with the
emotion reference and fill, then `vc.py` converts each settled take to the
actor's timbre. Fill re-times its seeded takes exactly through IndexTTS's
duration factor, `fit.py` predicts their length from the English and the
actor's speech, and `lineqa.py` converts a take again when only the
conversion garbled it. A part dubbed from English into Japanese renders
without per-line references (finding 16), and `lineqa.py` compares its takes by
their reading in kana. One dubbed from Spanish into English keeps them
(finding 17), and so does one dubbed from English into Spanish (finding 18),
whose takes `lineqa.py` compares word by word without accents, and one dubbed
from Korean into English (finding 19) and Chinese into English (finding 20).
English into Cantonese keeps them too (finding 21), with pronunciation review
still needed in either mode. `lineqa.py` hears each take once and gives a
flagged line one fresh take before settling it for the director (finding 22).

## Open questions

- **A native Japanese or Spanish voice.** Dubbing into Japanese or Spanish,
  every clone keeps its actor's English accent; the counterpart of the
  `"american"` voice would be an engine that speaks the language natively
  (CosyVoice 3 is in VoiceStudio's catalogue, not installed here; IndexTTS
  speaks only English and Chinese) converted to the actor's timbre.

- **Steadier short takes.** Most fill passes, overflows and re-rolls come
  from lines under about 1.5 s, whose takes vary most from roll to roll (see
  finding 11). Rendering several takes of a short line and keeping the one
  that fits might settle them in one pass.
- **Pauses inside a line.** Fill aligns where a line ends. An actor who pauses
  mid-sentence still gets continuous English.
- **Coverage of per-line references.** Snapping windows to pauses helps. Long
  lines hit OmniVoice's 20 s reference limit next to the voice's clips; using
  fewer clips needs a transcript per clip. The window check compares
  transcripts as raw text, so kanji against kana counts as a difference, and it
  only rejects gross mismatches; a kana-aware comparison could be stricter.
- **A Japanese recogniser for captions.** `subs.py` reads captions with
  PP-OCRv5, a Chinese model that also reads kana. It confuses a kana with its
  katakana twin and garbles the odd rare kanji. A recogniser trained on
  Japanese (manga-ocr, say) run on the caption boxes might read them cleanly.
- **Conversion that keeps the intonation.** Chatterbox and Seed-VC re-generate
  part of the pitch contour; Seed-VC's pitch-preserving model kept it but won
  no likeness. A converter that does both would give the `"american"` voice
  the delivery IndexTTS had before conversion.
