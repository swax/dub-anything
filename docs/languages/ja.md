# Japanese

What differs when Japanese is the source or the target. Everything in
[directing.md](../directing.md) applies; read this beside it, and the other
language's notes ([en.md](en.md)). The code is `scripts/languages/ja.py`.

## As the source

[directing.md](../directing.md) was written from Japanese → English dubs, so a
Japanese source needs nothing beyond it: a line keeps its original in `ja`,
`second.py` runs ReazonSpeech k2 v2 and SenseVoice and compares them with the
draft as kana, and `subs.py` reads burned-in Japanese captions. Into English
there are both voices (the actor's and the American), and per-line references
are on (research.md, findings 6 and 8).

Reazon can return only the last phrase of a long focused range while SenseVoice
recovers the whole stretch. For disputed words, split the range at pauses and
check one short phrase at a time. Confirm the earlier phrases before relying
on that output.

## As the target: English → Japanese

- **The job.** `scripts/dub_anything.py new <video> --source en --target ja` (by
  hand, `scripts/dub.py prep <video> --source en --target ja`). Each part's
  `segments.json` records the pair and every script reads it from there:
  Whisper listens in English (the draft, `asr_words.py`, `voices.py check`, the
  window check), takes are made and heard in Japanese (`seg_ja_<id>.wav`), and
  the subtitles, track names and styled signs follow. There is one voice, the
  actor's: each actor speaks Japanese with their own accent, as the Japanese
  actors speak English with theirs. The American voice is English only, and
  no engine for native Japanese is wired in.
- **No per-line references by default.** The actor's English original in a
  line's reference carried their accent into the Japanese: on one sketch, a
  strongly accented actor's takes went from 0.96 to 0.53 on Whisper's
  confidence that they were Japanese, and followed her delivery less closely,
  not more (research.md, finding 16). So a part dubbed into Japanese renders
  each take from the character's clips alone (`"render": {"line_ref": false}`,
  written on its first render) and the window check doesn't run. That makes the
  character's reference the register of every line: build it from lines in
  their usual register, and use `register.py` if one sounds wrong. A part can
  still opt in (`"line_ref": true`, for a calmer cast whose accent is light);
  compare by ear.
- **Translate concisely.** Japanese needs more morae than English needs
  syllables for the same meaning, so a faithful first draft runs long. In one
  sketch 14 of 32 first drafts came out 10–35% over the actor's time in
  `fit.py`, and only a two-word reply short.
  Cut, don't pad: drop the subjects and pronouns the picture makes clear, and
  prefer the short word (声 over メッセージ). `scripts/fit.py` estimates
  Japanese takes as well as English ones: on that sketch's 32 lines, a
  median error of +0.02 s, 84% within 0.2 s and 97% within 0.4 s.
- **Give each character a Japanese voice.** Japanese marks who is talking
  through first-person pronouns (俺, 僕, 私, あたし), sentence endings (だ, です,
  じゃん, よ, ね) and politeness, which English mostly leaves to the delivery.
  Pick them per character once and keep them in `work/NOTES.md`. Narration
  reads like a Japanese documentary's: plain form, questions in 〜なのか.
  Names go in katakana, spelled as on screen when a caption shows them.
- **Spell numbers in kana** (はちじゅう年, さんじゅう年代). The clones misread
  numbers written in kanji: in one sketch 八十年 was heard as 「1年」 and
  七十年代 as 「40年代」, and both came out right once written in kana. Kana
  also gives a number its real length: a kanji numeral weighs 3 in OmniVoice's
  length rule though it is said in two or three morae. Digits are worse still:
  VoiceStudio leaves them as digits in Japanese. QA checks a number written in
  digits or kanji against the one it hears; one in kana, by its reading only.
- **Mind the accent.** The clone speaks Japanese with the actor's English
  accent, and it shows where Japanese leans on length: a small っ, a long vowel.
  一家 (ikka) was heard as イカ ("squid") on two lines, and 家族 says the same
  without the っ. Others were near-misses a listener may or may not catch
  (暖炉 heard as 難路, 邸 as ディ). QA's reading match lets one mora pass, so
  read the per-line transcripts (`scripts/lineqa.py <part> --all`) for lines
  whose sense hangs on one word, and reword or list them for the user.
- **Short lines.** One-word replies stretch towards 2 s as in English. A
  0.8-s window held 「できない。」 (four morae) at natural pace; anything longer
  needs room or stays original with a caption. Short lines are also where a
  clone garbles most: two of one character's short lines failed QA twice
  (幽霊 heard as ゆで, 真っ暗 as マアン) and passed reworded with plainer words
  (おばけ, 暗い).
- **QA** hears takes in Japanese and compares readings in hiragana
  (`languages/kana.py`, pykakasi in its own uv environment), so kanji or kana,
  katakana or hiragana and digits or kanji numerals don't count as
  differences. A take passes at 80% kana-level agreement (English: 85% of
  words). "negation differs" counts ない, なかっ and ません. "English in the
  take" means Whisper heard Latin letters in a take whose text has none:
  usually the per-line reference leaking the actor's English, so try
  `"line_ref": false` on a line that does it on every take.
- **Subtitles.** Players wrap subtitles only at spaces, which Japanese doesn't
  have, so a cue over 22 characters is broken in two at the punctuation
  nearest its middle. Styled subtitles and signs are set in Yu Gothic (Arial
  has no Japanese).
- **Signs.** Japanese laid over a large English name or title can't be read:
  put it `above` or `below` the original at `size` 1.3–1.5. Lower thirds
  (a name and a role) translate as two lines, the name in katakana.
