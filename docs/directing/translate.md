# Translate and time

Part of [directing a dub](../directing.md): read its file formats and workflow overview first.

## 3. Translate and time

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
- **Compare changes to delivery by ear.** For adjacent clauses from one speaker,
  a grouped take is worth comparing with separate takes when no other voice
  intrudes. Preserve deliberate calls and long pauses as distinct beats unless
  listening supports joining them. Grouping changes reference use and fill speed
  as well as phrasing; record those settings with the comparison. A brief source
  clip passing its reference check does not establish that it sounds better than
  the character's reference. One fresh take per condition establishes a preference
  for those takes, without isolating which setting caused it.
- **End each window in a pause, with nobody else inside it.** A line whose
  window is clean gets the actor's own delivery (see [Render](render.md)). One where the
  next speaker starts before the window ends, or the actor is cut mid-sound,
  falls back to the character's reference. In overlapping dialogue, trimming a
  window's end back to the actor's last pause is often enough. Where two lines
  run straight into each other, the voice usually still dips for a moment
  between them; `scripts/edges.py --apply` moves the shared boundary onto that dip.
  A quiet dip after someone else's short reply does not make the preceding
  reference single-speaker. Word alignment can also attach a brief response to
  a later sentence across a long quiet gap. When one word appears to span that
  gap, inspect source energy and focused hearings on both sides before placing
  the boundary or assigning the response.
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
  For `SHORT?`: first check the separated vocals and source speech spans. The
  actor's first-to-last speech span can include a long internal silence. A VAD
  gap can also hide murmured words: confirm the source before splitting it. If it
  separates utterances, give them separate windows and preserve the pause;
  keep a brief original call with a caption when a clone cannot fit it. Re-run
  `fit.py` on those windows before adding words. For continuous speech, write
  fuller English, closer to everything the actor says.
  Don't count on fill to slow a short line down (see [Render](render.md)). In the
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
  anything that changes in place. When text carries across a part boundary, put
  its translation in both parts with the same placement, and check the frames
  on each side of the splice. Assembly must offset a moving sign's `seen` times
  with its start and end; otherwise a correct part preview can become a label
  that jumps to its final position in the full video. For camera reversals, use adjacent sign entries
  with measured start/end boxes; one move across the whole shot cannot follow
  them. Check the first frame after a sign's shot too: ASS rounds to
  centiseconds, so an end rounded past the cut can leave the label on the new
  shot. End it just before the cut when needed. Diagrams come in stages: a label can appear
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
