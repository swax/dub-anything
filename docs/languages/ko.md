# Korean

What differs when Korean is the source. Everything in
[directing.md](../directing.md) applies; read this beside it, and the other
language's notes ([en.md](en.md)). The code is `scripts/languages/ko.py`.

## As the source: Korean → English

- **The job.** `scripts/dub_anything.py new <video> --source ko` (the target is
  English; by hand, `scripts/dub.py prep <video> --source ko`). Each part's
  `segments.json` records the pair: Whisper listens in Korean, a line keeps its
  original in `ko` (captions too), and the delivered file carries the Korean
  transcript (`.ko.srt`). Both voices apply (only the actor's has been tried
  from Korean): the actor's voice speaks English with the actor's Korean
  accent. The original audio is tagged Korean in the
  delivered file whatever the source's own tag says (a web download tagged it
  English).
- **Whisper's stock phrases.** Over music and silence Whisper writes
  "감사합니다" ("thank you"), "다음 영상에서 만나요" ("see you in the next
  video") and subtitle credits ("자막 제공…", "한글자막 by …"), as it writes
  "ご視聴ありがとうございました" in Japanese. In one drama's opening, title
  cards and a wordless scene with music came back as six such lines. They also turn up over real speech: Whisper heard the first 30 s of a
  meeting as subtitle credits, and all ten of the draft's lines there came
  from its focused passes (`[recovered]`). Each uncovered stretch of the draft
  keeps only its own words, so one stock phrase can come back as several
  pieces ("자막", "영상에서", "만나요"); `draft.py` flags a piece of a stock
  phrase as well. A lone "감사합니다" from the full pass isn't flagged, as
  real thanks are common: check it against a frame. Where Whisper
  hallucinates, both second opinions write a syllable or nothing, a sign worth
  reading even where `second.py` flags nothing.
- **Spacing.** Korean is written with spaces, but recognisers place them
  differently ("할 수 있어", "할수있어"), and the Korean Zipformer writes none
  at all. So `second.py` compares transcripts by syllable, and the window check
  (`core.bare`) drops spaces as it does for every language.
- **Second opinions.** `second.py` runs k2's Korean Zipformer (trained on
  spontaneous Korean speech) and SenseVoice (about 320 MB). SenseVoice is the
  stronger on drama dialogue; the Zipformer misreads more but independently.
  Over 165 draft lines of a drama they flagged 2 (one a "감사합니다" over
  noise). Their real value was in the stretches already in doubt: the
  Zipformer caught an English loanword Whisper had turned into another word,
  and SenseVoice a common idiom Whisper had heard as its near-homophone.
  Neither settles names: they split on surnames a vowel apart (주 and 조, 최
  and 채).
- **Names.** Whisper spells a name differently on each hearing (one lead's
  name came out three ways in a single scene), so correct it from context and
  the picture. Characters call each other by name and title constantly
  ("<name> 팀장."); keep those calls in the actor's own audio with a caption,
  and leave a name the recognisers can't agree on out of the English rather
  than guess it ("a presidential candidate"). Titles translate: 본부장
  "Director", 팀장 "Team Lead"; record the choices in the job's `NOTES.md`.
- **Speech levels carry meaning.** A switch between polite speech (존댓말) and
  casual speech (반말) is a plot point as often as a habit: when a junior asks
  a senior to drop the formalities (말씀 편하게 하세요), the senior answers in
  casual speech from then on. English has no grammar for it, so say it in
  words ("Please, no formalities"), and let the register show in word choice.
  A joke on a word's connotation needs an English word with the same one: an
  old-fashioned word for a partner against a young person's slang wants an
  English pair with the same contrast, and the same gender (or lack of it)
  if the punchline turns on it.
- **Translate concisely, and mind numbers.** OmniVoice's length rule weighs a
  Hangul syllable 2.5 and a Latin letter 1, so English follows how dense the
  actor's Korean is. Over four parts of a drama, `fit.py` flagged 22 of 108
  first drafts `OVER?` or `risk`, 9 of 44 in one rapid-fire meeting, and
  numbers most of all: Korean writes them compactly ("15만", "0.5%") and
  English spells them out ("a hundred and fifty thousand"). Cut what the
  picture shows, say "half a percent" rather than "zero point five percent",
  and say a number once where the Korean repeats it. Slow narration and a
  hearing's measured questions ran the other way (10 `SHORT?`): write fuller
  lines there.
- **Numbers with a thousands comma.** VoiceStudio spells English numbers out,
  but not one written with a thousands comma ("150,000"): those reach the
  clone as digits, and the clone read them in its reference's language:
  "150,000" came back as Korean-sounding syllables ("Baek-woong-op-shi-cheon")
  on two lines. `dub.py` now sends English text without thousands commas
  (`speakable` in `en.py`), so write numbers as you like.
- **Short replies and fragments.** 네, 예, 그래? and the like stay in the
  original with a caption. A window that holds only the end of a sentence,
  as the Korean word order leaves it (the verb alone, after a long dramatic
  pause), garbled on all four takes, as a two-word English predicate with no
  subject; given its own subject it passed at once. Write each piece so it
  stands on its own.
- **The picture often settles what the audio can't.** A brand name that
  looked like Whisper mishearing a real brand was, on the product in the
  shot, a fictional one spelled just as Whisper heard it. Korean dramas open
  with a product-placement notice and a disclaimer card, and show phone
  screens, search results and headlines: run with `--signs`.
- **Captions and signs.** `subs.py` and `signs.py` read Hangul with
  PP-OCRv5's Korean recogniser (`OCR` in `ko.py`; the Chinese one has none),
  which reads Latin letters and digits too. On a drama it read the disclaimer,
  the product-placement notice, the title card, a time card ("six months
  earlier"), an on-screen definition, a phone chat and a news headline
  exactly; read the sheets anyway. Burned-in Korean captions haven't been
  tried: `subs.py` finds songs by the Chinese recogniser reading ♪ as 月, which
  the Korean one may read otherwise.
- **Per-line references stay on.** Over 107 lines of a drama rendered both
  ways, the takes followed the actors' delivery far more closely with them
  (arousal rank correlation 0.40 against 0.01, pitch 0.47 against -0.01),
  and Whisper's confidence that they were English didn't move (0.58 against
  0.59); likeness, quality and timing were level (research.md, finding 19).
  `fit.py`'s estimates held as for Japanese → English: median error +0.00 s,
  70% within 0.2 s, 93% within 0.4 s.
- **QA** is the same as for Japanese → English. The accent shows on numbers
  ("thirty" heard as "thirteen") and on made-up names close to English words
  (heard as the word); both cleared with a fresh take.

## As the target

Not built: nothing dubs into Korean yet (`PAIRS`). `ko.py` has what a dub into
it would need to start from, untested: QA by syllable with spacing aside
(`compare`, with 않, 못, 없, 아니 and 말 as negations and numbers in digits
checked), Malgun Gothic for styled subtitles, and track names in Korean.
VoiceStudio leaves digits as digits in Korean, and Korean reads a number two
ways (Sino-Korean 삼, native 셋), so a script would spell numbers in Hangul as
`ja.md` says for kana.
