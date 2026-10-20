# Chinese

What differs when Chinese is the source. Read beside
[directing.md](../directing.md) and [en.md](en.md). The code is
`scripts/languages/zh.py`.

## As the source: Chinese → English

- **The job.** `scripts/dub_anything.py new <video> --source zh --voices actor`
  (English is the default target), or `scripts/dub.py prep <video> --source zh`
  by hand. Each part records `zh` → `en`; original dialogue and captions use
  the `zh` field. Delivery includes a Chinese transcript (`.zh.srt`) and tags
  the original audio `zho`, even when a download incorrectly labels it English.
  The actor voice has been tested on Mandarin. The English target also permits
  the American voice; it has not been exercised for this pair.
- **Scope.** `zh` follows Whisper's Chinese language code. The checked sample
  speaks Mandarin; Cantonese and other Chinese varieties have not been tested.
  Keep their language and pronunciation in mind when picking recognisers.
- **Second opinions.** `second.py` runs a Chinese Zipformer CTC and SenseVoice
  on the CPU. The CTC model has one `model.int8.onnx`, unlike the Korean
  Zipformer's encoder/decoder/joiner. Compare characters, ignoring punctuation,
  spaces, letter case and full-width variants. Simplified and traditional
  spellings, homophones and numbers written in digits versus characters still
  differ: review those flags against captions and context.
  Model details: [sherpa-onnx's Chinese CTC model](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html)
  and [SenseVoice](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17).
- **Captions.** PP-OCRv5's Chinese server recogniser reads Chinese captions
  and signs. For white outlined subtitles without a box use `--subtitles`
  (`subs.py --open`); `--captions` expects boxed closed captions. Caption
  display times are not speech windows. Check frames for proper names and
  captions the OCR missed; the subtitle may simplify what the actor says.
- **Translate and fit.** Spoken Mandarin packs meaning into a short window.
  Translate naturally, then use `fit.py` before rendering. OmniVoice weighs a
  Han character 3 units versus 1 for a Latin letter; the reference's pace and
  the spoken English numbers determine the result. Keep short acknowledgements,
  names and nonverbal sounds original when there is too little room or clean
  speech to clone. Make each phrase stand on its own in English.
- **Names and titles.** Characters may share pronunciations or be written
  differently by recognisers. Use the picture and story to settle spelling;
  keep unresolved names original with translated captions and a `listen` note.
  Translate titles and forms of address for their relationship in the scene,
  and keep those choices in `work/NOTES.md`.
- **QA.** Rendered English takes use the same word, number and negation checks
  as the other pairs into English. Review each line independently and hand
  uncertain pronunciation, voice likeness and acting to the listener.
- **Per-line references stay on.** In the checked Mandarin sample, both
  versions passed per-line English QA. Keeping the actor's own line in the
  reference improved measured voice likeness and timing without weakening
  the English (research.md, finding 20). The length estimates held as well.
  There was too little speech per character to measure acting correlations;
  the listener still judges delivery and likeness. Detailed sample results
  stay in the ignored test report and work files.
- **Subtitles.** Long Chinese transcript cues are broken near their middle,
  preferably at punctuation, because players cannot rely on spaces to wrap
  them. Styled Chinese subtitles use Microsoft YaHei on Windows.

## As the target

Not enabled: English → Chinese is not in `PAIRS`. The module's Chinese
synthesis helpers satisfy the language interface but have not been checked
on rendered Chinese takes. `compare` reads numbers written in digits or
characters alike (`hanzi.py`, shared with Cantonese); simplified/traditional
normalization needs further work before adding that direction.
