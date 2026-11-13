# Spanish

What differs when Spanish is the source or the target. Everything in
[directing.md](../directing.md) applies; read this beside it, and the other
language's notes ([en.md](en.md)). The code is `scripts/languages/es.py`.

## As the source: Spanish → English

- **The job.** `scripts/dub_anything.py new <video> --source es` (the target is
  English; by hand, `scripts/dub.py prep <video> --source es`). Each part's
  `segments.json` records the pair: Whisper listens in Spanish, a line keeps
  its original in `es` (captions too), and the delivered file carries the
  Spanish transcript (`.es.srt`). Both voices apply: the actor's voice speaks
  English with the actor's Spanish accent, as it keeps a Japanese actor's
  accent from Japanese. The original audio is tagged Spanish in the
  delivered file whatever the source's tag says: web downloads often tag
  every track English.
- **English that isn't.** Whisper's language detection calls short Spanish
  phrases English, above all over music, and forced into English it
  translates them: a one-word greeting came out "Thank you.", a two-word
  question another English question. In the first ten minutes of a
  telenovela it tagged nine phrases English and one Korean, and none was
  English speech. So `draft.py` keeps a
  target-language tag only when Whisper also transcribes the phrase more
  surely in English than in Spanish. The rest keep their Spanish and are
  tagged `[English?]` with the English reading: six of the nine. Check what
  stays `[English]` as well (`asr_words.py ... --lang es` and `--lang en`, and
  a frame): of the other three, two were crowd noise and one was a Spanish
  exclamation, written as an unrelated English sentence.
- **"¡Gracias!" from nowhere.** On a laugh, crowd noise or a song, Whisper
  writes "¡Gracias!" (and "¡Suscríbete!") where it hears no words, as it
  writes "ご視聴ありがとうございました" in Japanese. Real thanks are everywhere in
  dialogue, so the draft only marks a recovered phrase that says nothing
  else `[likely Whisper hallucination]`: three of the four it marked in a
  telenovela's party scenes were noise.
- **Songs.** Without burned-in captions there are no ♪ markers, and the draft
  transcribes sung lyrics as dialogue: a telenovela's theme song over its
  opening titles, a pop song at a party. Keep them original. `cast.py` puts a
  singer's lines in a cluster of their own, which finds the theme song at
  once.
- **Second opinions.** `second.py` runs NVIDIA's Parakeet TDT 0.6B v3 and
  Canary 180M Flash in Spanish (about 900 MB) and compares them with the
  draft word by word, accents aside. On two parts of fast dialogue it flagged
  6 of 77 and 6 of 100 lines, mostly words a draft window cut off or took from
  the next speaker, and a few mishearings (a possessive both models heard
  as a demonstrative). Parakeet returns nothing on some short stretches, and
  Canary can loop on crowd noise ("sí" eighty times); no line is flagged
  where either fails. Word-level comparison also flags a word the draft
  writes as one and the models as two (a long adjective).
- **Fast dialogue.** Telenovela dialogue runs from one speaker to the next
  with no pause the VAD can find, so many windows share a boundary and many
  reference clips can't be bounded by pauses. Cut references at a dip of
  30 dB or more between the voices, and set a shared boundary by transcribing
  each side of it: on one line `edges.py` moved a boundary onto a dip inside
  the next speaker's first words, and the window check then heard only the
  rest of her line. Look at the loudness where each reference clip ends, too:
  one ended with the first syllable of the speaker's next phrase, which
  Whisper left out of its transcript, so `voices.py check` passed it. At a
  party the crowd stays in the separated vocals and keeps the VAD on past
  the actors: 6 of 44 lines in one party scene fell back to the character's
  reference for speech running on past their windows, and every "voice
  sounds on" note from `edges.py` and "WET" line from `tails.py` there was
  music or crowd at -25 to -30 dB, not the actors.
- **Translate concisely.** OmniVoice's length rule counts Latin letters in
  both languages, so a take of English lasts about as long as the actor took
  for that many letters of Spanish. Spanish needs more letters than English
  for the same meaning, so natural English fits, and padded English doesn't:
  in one telenovela scene 7 of 15 first drafts, written out in full, came out
  `OVER?` in `fit.py` (up to 1.6 s on a 3-s window), and all fit once
  trimmed to natural, tight English. Write it the way an English-language
  show would say it, and keep the forms of address the characters use
  ("Doña" or "Don" before a name). Spanish names stay as they are; the
  clones say them as the actors do.
- **Per-line references stay on.** Into Japanese they carried the English
  accent into the takes (research.md, finding 16). From Spanish they didn't:
  over 116 lines of a telenovela, Whisper's confidence that the takes were
  English was 0.78 with them and 0.80 without, and with them the takes
  followed the actors' delivery more closely (arousal rank correlation 0.46
  against 0.36), matched their voices a little better and fit their windows
  more easily (research.md, finding 17). `fit.py`'s estimates held as for
  Japanese → English: median error +0.02 s, 91% within 0.4 s.
- **QA** is the same as for Japanese → English. Read the per-line
  transcripts (`lineqa.py <part> --all`) for a meaning QA lets pass: a
  compliment opening "My, don't you…" passed while Whisper heard "Why don't
  you…?", a jab a listener may hear too; opening it with "Wow," instead
  cleared it.
- **Captions and signs.** `subs.py` and `signs.py` read with a Chinese and
  English model: Latin letters come out right, accents not always (a title
  card's "Adaptación" read as "Adaptacion"). Correct them in `es`.

## As the target: English → Spanish

Much of what dubbing into Japanese needs applies here too ([ja.md](ja.md), "As
the target").

- **The job.** `scripts/dub_anything.py new <video> --source en --target es` (by
  hand, `scripts/dub.py prep <video> --source en --target es`). A line keeps
  its original in `en`, takes are made and heard in Spanish
  (`seg_es_<id>.wav`), and the delivered file carries the Spanish dub and
  subtitles and the English transcript. There is one voice, the actor's,
  speaking Spanish with the actor's English accent; the American voice is
  English only.
- **Which Spanish.** Write neutral Latin American Spanish, as US studios dub
  for Latin America, unless the user asks for Spain's: `ustedes`, never
  `vosotros`. Use the established Spanish names of characters and of the
  films and franchises a script mentions (their Latin American titles), and
  record them in the job's `NOTES.md`. People's names stay as they are, but a
  nickname with a known Spanish form takes it. Latin American Spanish says
  `z` and `c` like `s`, so Whisper writing "casa" for "caza" is not a fault.
- **Per-line references stay on.** Into Japanese the actor's English original
  carried the accent into the takes (research.md, finding 16). Into Spanish it
  didn't: on a sketch rendered both ways (31 lines, four clones), Whisper's
  confidence that the takes were Spanish was 0.60 with them and 0.61 without,
  and with them the takes followed the actors' pitch far more closely, sounded
  cleaner and passed the first QA check without a flag, against two
  (finding 18). That confidence is low for every take, as for any clone that
  keeps an English accent, so read the per-line transcripts
  (`lineqa.py <part> --all`) rather than trusting the pass count.
- **Translate concisely.** OmniVoice's length rule counts Latin letters in both
  languages, and Spanish needs more of them than English for the same meaning,
  so a faithful first draft runs long, as into Japanese. Cut what the picture
  makes clear and prefer the short word, but write fuller where the actor
  pauses mid-line: a narrator's or a hero's slow, dramatic lines came out
  `SHORT?` in tight Spanish, and a line with two long pauses needed a third
  more words. A slow-talking clone is the tightest: one character who spoke in
  a low growl needed a four-word line cut to three to fit a 2-s window.
  `fit.py`'s estimates held: with each take's reference and speed, a median
  error of +0.06 s, 71% within 0.2 s and every take within 0.4 s. Even so,
  eight of 31 takes ran at fill's fastest speed.
- **Short lines.** One- and two-word replies stretch towards 2 s, and `fit.py`
  runs short on them here too: a two-word question estimated at 1.07 s came out
  at 1.46 s and blocked the first render, since VoiceStudio stops at an
  overflow before fill can re-time anything. Use the shortest natural form (a
  colloquial "¿Que qué?" for "You what?") or keep the reply original with a
  caption.
- **Wordplay.** Rebuild a pun from Spanish words rather than translating its
  words: a mishearing joke between two English near-homophones works as two
  Spanish ones, and a tagline punning on a swear word as a mild Spanish curse
  with the same image. Note the choice in the line's `listen` so the user can
  judge it.
- **Numbers.** VoiceStudio spells digits out in Spanish, so a year can stay in
  digits in the script (and the subtitles), unlike in Japanese. It leaves a
  number with a thousands point ("150.000") as digits, so `dub.py` drops the
  point (`speakable` in `es.py`).
- **QA** compares Spanish word by word, as English: accents and `¿¡` aside
  (Whisper writes "esta" or "está"), numbers spelled out in Spanish
  (num2words), `Sr.`, `Sra.`, `Dr.` and `Ud.` expanded, and "negation differs"
  counting no, nunca, nada, nadie, ni, ningún, jamás and tampoco. A take that
  runs words together can still pass on letters: a cast-card line sped up to
  fit came back as one invented word where "como" met a short name. Reword
  such a line (shorter, or another verb) rather than re-rolling it.
- **Subtitles and signs** stay in Arial, which has the accents, `ñ` and `¿¡`.
  A cast card the narration reads out ("<actor> as <character>") needs no
  sign: the viewer hears it in Spanish.
