# Languages

The directing notes for each language, on what differs with it as the source
and as the target, are beside this file: [ja.md](ja.md), [en.md](en.md),
[es.md](es.md), [ko.md](ko.md), [zh.md](zh.md), [yue.md](yue.md). Read the job's two beside
[directing.md](../directing.md).

The scripts dub the pairs listed in `PAIRS` (`scripts/languages/__init__.py`):
Japanese → English, Spanish → English, Korean → English, Chinese (Mandarin) →
English, English → Japanese, English → Spanish, and English → Cantonese.
What they know about a language is in one module, `scripts/languages/<code>.py`,
named for its ISO 639-1 code, or its ISO 639-3 code when it has no two-letter
one (`yue`). A file with a longer name there is a helper, not a language. What
was measured for a pair is its entry in
`PAIRS`. The scripts look a language up by its code (`LANGUAGES["es"]`) and
never branch on it, so a new language is a new module and a new pair is a new
entry, with no change anywhere else unless a tool itself can't handle the
language (see "Adding a language").

## A language's module

Each module defines the names below (`REQUIRED`; the package won't load a
module that lacks one). A module imports only the standard library at its top,
because `second.py` and `signs.py` load it in their own environments. Anything
more is imported inside the function that needs it (num2words in `es.py`,
pykakasi through `kana.py`).

| Name | What it is | Used by |
|---|---|---|
| `NAME`, `ISO3`, `NATIVE` | The language's English name (in messages), its ISO 639-2 code (track tags) and its name in itself (subtitle track titles) | everything, `assemble.py`, `deliver.py` |
| `HALLUCINATIONS` | Whisper's stock phrases on hums, music and silence when it's told to hear this language, in lowercase | `draft.py` |
| `THANKS` | Whole phrases Whisper writes on a breath, a laugh or crowd noise ("thanks"). Real thanks are common, so only a recovered phrase that says nothing else is suspect. May be empty | `draft.py` |
| `SECOND_OPINIONS` | Two sherpa-onnx recognisers, `{family: Hugging Face repo}`, the family one that `second.py` builds (`parakeet`, `canary`, `sensevoice`, `reazon`, `zipformer`, `zipformer_ctc`). `{}` for none | `second.py` |
| `units(text)` | What two transcripts are compared by: a list of words, or a string of characters (the edit distance runs over its items) | `second.py` |
| `OCR` | The recogniser that reads captions and signs in the language: RapidOCR's PP-OCRv5 `(LangRec, ModelType)`, `("CH", "SERVER")` for Chinese, Japanese, English and Spanish, `("KOREAN", "MOBILE")` for Korean | `subs.py`, `signs.py` |
| `VOICES` | The render voices a dub into the language can have: `("actor",)`, plus `"american"` for English | `dub_anything.py new`, `dub.py` |
| `JOIN` | What joins Whisper's segments of a take: `" "`, or `""` for a language written without spaces | `lineqa.py` |
| `compare(want, got)` | QA's check of a take's transcript against its line: `(notes, similarity)`, one note per problem (`match 0.71`, `negation differs`, `number differs: ...`), none when it passes | `lineqa.py` |
| `TITLES` | Track names in the language: `dub` and `subs` per voice, `signs`, `subtitles` (a subtitle job's translation), `transcript` and `original` | `assemble.py`, `deliver.py` |
| `FONT` | A subtitle font that has the language's letters | `ass.py` |
| `subtitle(text)` | A subtitle cue's text, broken in two when it runs past a line: at the space nearest the middle, or by hand where there are no spaces | `assemble.py`, `signs.py`, `subtitles.py` |
| `READING_CPS` | Characters a second a subtitle in the language can be read at in comfort (17 for English, 7 for Japanese). Optional (17); `subtitles.py` times cues and flags fast ones by it | `subtitles.py` |

Beside them, `latin.py` holds what the languages written in Latin letters with
spaces share: `second.py`'s word units, subtitles broken in two at a space, and
`compare_words`, QA's word-by-word check. A module gives `compare_words` its own
normalizer (numbers spelled out in its words in each reading that matters,
abbreviations, contractions), its negations and its number words. `kana.py`
reads Japanese as hiragana for `ja.py`. Korean has spaces, but recognisers place
them differently, so `ko.py` compares by syllables, as `ja.py` does by kana.
Chinese's `zh.py` compares characters with punctuation, spacing and width aside;
it preserves simplified/traditional variants and homophones for review.

A language that is only a source still defines every name: `ko.py`'s `compare`
has never checked a rendered take, since nothing dubs into Korean yet.

A language a clip only speaks in passing (kept in the original and captioned)
needs no module: its English name and ISO 639-2 code go in `OTHERS`.

## A pair's entry

`PAIRS[(source, target)]` records what was measured for the pair
([research.md](../research.md)):

- `line_ref`: whether the pair's lines render with per-line references (the
  voice's clips, then the actor's own original of the line) unless the script
  says otherwise. They help where the original doesn't carry its accent into
  the dub, so they're on into English from Japanese, Spanish, Korean and Chinese and
  from English into Spanish, and off from English into Japanese (findings 16
  to 20).

A pair goes into `PAIRS` once a video has been dubbed in it end to end and
checked. `dub.py prep`, `dub_anything.py new` and `deliver.py` refuse any pair that
isn't there.

## Adding a language

1. **The module.** Copy the closest one (`es.py` for a language written in Latin
   letters, `ja.py` for one without spaces, `ko.py` for one whose spacing
   recognisers don't agree on) and fill it in:
   - `HALLUCINATIONS`: run `draft.py` over stretches of music and of silence and
     note what Whisper writes there.
   - `SECOND_OPINIONS`: two sherpa-onnx recognisers that know the language. A
     family `second.py` doesn't build yet needs its own branch in
     `recogniser()`. With `{}`, `second.py` skips the language. Run both over a
     drafted part and read what they write: Korean's Zipformer writes no spaces
     at all, which `units` has to allow for.
   - `OCR`: the RapidOCR recogniser that reads the language's script
     (`rapidocr.LangRec`; PP-OCRv5 has Chinese, which reads kana and Latin
     letters too, Korean, Latin, Cyrillic, Greek, Thai and others).
   - `compare`: normalize both sides so spelling doesn't count. Write numbers
     the way Whisper may hear them, in every reading that matters (English
     reads 1979 in three ways). List the negations and number words, so a
     flipped meaning is still caught in a close match. Then run `lineqa.py` on a
     few rendered takes and read every flag.
   - `TITLES` in the language itself, and a `FONT` that has its letters.
     `deliver.py` recognises its own tracks in a delivered file by these names,
     so if a name ever changes, add the old one to `LEGACY` there.
2. **The pair.** Dub a short video end to end, twice from the same script: once
   with per-line references, once without. Measure both as findings 16 to 19
   did (`eval/score.py`, `eval/timing.py`): Whisper's confidence that the takes are in the target language, how
   closely they follow the actors' delivery, voice likeness, and timing.
   Record the result as a finding and add the pair to `PAIRS`. Check that
   OmniVoice's length rule holds for the pair too (`fit.py` against the rendered
   takes).
3. **What the tools can't do yet.** The OCR for burned-in captions and signs
   (`subs.py`, `signs.py`) reads the scripts PP-OCRv5 has a recogniser for,
   Latin letters not always with their accents. The American voice is English
   only. Note anything else that falls short.
4. **The directing notes.** Start `<code>.md` here, with what differs when
   the language is the source and when it's the target, as the notes for the
   languages so far do, and link it from the list at the top.
