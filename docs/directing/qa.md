# QA

Part of [directing a dub](../directing.md): read its file formats and workflow overview first.

## 5. QA

For the optional observation-only triage workflow, see the
[QA pilot](../plans/qa-triage-pilot.md). Keep raw hearings, machine flags and
director decisions separate. A second recognizer agreeing with the first is
evidence, not proof; disputed numbers, negation, missing words and turn boundaries
still need review. A review decision belongs to the current audio, wording,
voice and occurrence window, and must be revisited when those inputs change.

Set `DUB_QA_OBSERVATIONS` to an ignored study directory before the normal runner
starts, or give `lineqa.py --observations DIR`. This archives the first checked
audio and contextual mix before ordinary repair or restoration, along with
settings, raw readings, comparisons and control actions. It leaves the normal
repair policy in charge. After primary QA finishes, freeze the declared sample
with `scripts/qa_triage.py enroll DIR --freeze`; use `--allow-incomplete` only
at the final freeze when a stratum is genuinely sparse. Then run
`scripts/qa_triage.py hear DIR --reader sensevoice` in `second.py`'s environment
and `scripts/qa_triage.py report DIR` with VoiceStudio's Python. Secondary
hearings are unprompted, bounded and cached. The report is a review queue: it
cannot queue synthesis, restore takes, mark readiness or approve delivery.

The normal runner records exact current-take hashes after QA. A re-mix can
regenerate audio while fitting, so it is final only when those checked hashes
still match; changed takes return to ordinary QA. After two such rechecks,
another changed re-mix blocks for a timing/reference review instead of cycling.
Missing takes also block. New final receipts are checked again before delivery;
older final jobs without these receipts stay unchanged until reopened or revised.

- `scripts/qa.py episode-01` transcribes the finished MP4 in English and lines
  it up with the script. Word timestamps drift by a line or so in quiet
  passages, so a word appearing in the neighbouring window is usually fine.
  Whisper also invents words on background noise; check `energy.py` before
  worrying.
  A mix retaining source-language speech can make full-clip ASR omit whole
  stretches that pass the per-line checks. Extract a short stretch of the
  assembled MP4 and check it with `asr_words.py <wav> <offset> --lang en`
  (or the target language), without VAD. Narrowing the context helps distinguish
  a transcription miss from masked speech; check this before raising the gain
  or replacing a good take.
- **Check each rendered line on its own too:** `scripts/lineqa.py episode-01`. The
  full-mix QA hid these problems:
  - junk at the start of lines (reference leakage, in [gotchas](gotchas.md))
  - dropped words in fast lists
  - meaning flips like "can" heard as "can't" (any change in negation is flagged)
    or "seventy" as "forty" (so is any number heard as another)
  - words a voice can't say
  - lines that came out silent

  Numbers, units, contractions and "Dr." are normalized, so formatting doesn't
  count as a failure. Skim the cached per-line hearings even when no lines are
  flagged: a high overall match can hide an omitted short name or clause, or
  a pronoun substitution that changes whom the actor means.
  Check disputed words in the actual mixed movie too: an isolated take can
  transcribe correctly while music or effects reduce its clarity. Compare
  narrow and contextual crops without prompting the expected words, and use a
  second recognizer. Context can recover words that isolated transcriptions
  missed; record both readings before judging a substitution. When these checks
  recover the intended words, record any remaining clarity question for the
  listener. Repeated unwanted profanity readings need rewording.
  If repeated takes lose a short setup clause, check it with a second recognizer.
  When both omit it, try folding its meaning into the main sentence instead of
  another separate lead, then check the whole line and its actual timing again.
  Triage each flag. A random glitch (a dropped word, a
  mangled first word) usually clears with a fresh take:
  `scripts/lineqa.py episode-01 --reroll` queues the flagged lines, and the next
  `generate` re-rolls them.
  A word the voice can't say needs rewording. Every take is a new roll of the
  dice, so re-check after each render. A fresh take can also come out worse
  than the one it replaced: `--reroll` keeps each replaced take
  (`<part>/takes/`), and the next check puts the best one back when the
  new take transcribes worse. The next `generate` then only re-mixes. What
  Whisper heard in each take is kept by the take's hash
  (`<part>/heard.json`), so the check after a round listens only to the
  re-rolled takes, and one after a re-mix loads no model at all: on a film's
  first parts a 34-line check after a round heard one take, and one after a
  re-mix took 5 s instead of about 50-200 s per check before. A line that
  fails on its fresh take too is *settled*: it gets no more rolls (later
  rounds almost never fixed one, [research.md](../research.md), finding 22) and
  goes to the report for rewording or the listener. In the
  `"american"` voice most flags come from the conversion rather than from
  IndexTTS, so when IndexTTS's own take of a flagged line says it right,
  `--reroll` queues that take to be converted again instead (conversion is
  random too), keeping its performance and timing; after two failed
  conversions it gets a fresh take. Judge each flag against the intended
  text and the audio. Names and specialist vocabulary can confuse QA even when the take is
  correct; flag uncertain cases for the listener. On a feature film, the first
  round cleared the real glitches; later rounds mostly re-rolled names and
  dropped fillers ("Hmm", "huh") that ended up accepted anyway. So reword a line
  that fails twice rather than waiting on another roll: a one-word line
  misheard on every take ("Fine." heard as "Find.") usually clears with one
  more word of context ("That's fine."). Leave names to the listener. Reword
  at once a line QA hears as profanity or a slur, even if the take is probably
  right: a listener can hear it the same way. A
  hesitation ("Um...", "Well...") or a one-word answer that fails twice goes
  back to the original audio, with a caption if it says anything. A take that
  starts with the line's own Japanese (QA's "leading extra" in Japanese, the
  same on every take) is the per-line reference leaking: give the line
  `"line_ref": false`. A leaked ending can also look like an English syllable
  in the recogniser's output. Compare repeated extras with the reference's
  last words, then verify the new take's prefix with the checked character
  reference. When the American voice's conversion keeps garbling a
  line that IndexTTS said right, other words in `text_us` usually clear it.
- Listen. Machines can't judge acting.

When a cached take appears in several windows, review each occurrence's timing
and handoffs separately. Persist approvals by occurrence as well as take hash so
one occurrence cannot overwrite another.
