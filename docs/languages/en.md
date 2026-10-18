# English

What differs when English is the source or the target. Everything in
[directing.md](../directing.md) applies; read this beside it, and the other
language's notes ([ja.md](ja.md), [es.md](es.md), [yue.md](yue.md)). The code is
`scripts/languages/en.py`.

## As the source: English → Japanese, English → Spanish, English → Cantonese

- **The script.** A line keeps its original in `en`:
  `{"start": 4.3, "end": 5.9, "voice": "speaker_1", "en": "Hello.", "text": "こんにちは。"}`,
  and so do captions. The draft tags speech already in the target language
  (`[Japanese]`, `[Spanish]`, `[Cantonese]`): keep it and caption it. `second.py` runs
  Parakeet TDT 0.6B v2 and SenseVoice in English and compares them with the
  draft word by word.
- **Casting** is as in directing.md. English drafts come out with clean
  punctuation and phrase-length lines, but Whisper's word times run early here
  too, and the VAD merges quick exchanges into one span: snap turn changes by
  loudness.
- **One voice.** The dub has the actor's voice only, speaking the target
  language with the actor's English accent. The American voice is English
  only. Whether per-line references carry that accent into the takes depends
  on the target: off into Japanese, on into Spanish and Cantonese (see the
  target's notes for measured benefits and remaining speech errors).

## As the target

[directing.md](../directing.md) was written for dubs into English, from
Japanese: the actor's voice and the American one, per-line references,
`"text_us"` for a line the American voice needs worded differently. QA
compares takes word by word: numbers spelled out in any reading ("1979" as
"nineteen seventy nine"), contractions expanded, "Dr." as "doctor", letter
grades and ordinals as said (Whisper writes "a B-minus" as "a B-", which once
flagged a quarter of a scene's lines, all of them right, and "eighth" as
"8th"), a take
passing at 85% of words, and "negation differs" and "number differs" catching
a flipped meaning in a close match. Numbers can stay in digits: VoiceStudio
spells them out, and `dub.py` drops a thousands comma ("150,000"), which would
have kept them digits for the clone to read in its reference's language
(`speakable` in `en.py`; [ko.md](ko.md)). From Spanish, see [es.md](es.md), "As
the source"; from Korean, [ko.md](ko.md); from Chinese, [zh.md](zh.md).
