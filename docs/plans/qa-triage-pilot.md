# Next dub: QA triage pilot

Status: observation mode implemented in `scripts/qa_triage.py`, with optional
capture from `scripts/lineqa.py`. A first prospective frozen sample has received
bounded secondary checks and director reviews, including passing controls.
A cached resume reused every hearing without loading a model or changing
production files. Unresolved cases stay unresolved; private listener outcomes
and wider policy evaluation remain pending. This document does not start or
schedule a movie or change the default repair policy.

## Question and scope

Can contextual evidence distinguish real speech defects from recognizer
disagreements, while preserving acceptable takes and catching errors that the
primary check misses? Fewer flags and closer endpoints alone do not establish
better speech or listener preference.

Use one Japanese-to-English movie with actor voices and native accents. Follow
the normal intake, parallel directing and verified-delivery workflow. Keep the
existing runner behavior as control and default. Select the source at execution
after checking existing jobs and deliveries; keep its identity and all dialogue,
cast, measurements and audio in ignored `work/` files.

## Implement a bounded observation mode

1. Add optional structured evidence to `lineqa.py`: current take hash, expected
   text, raw hearing, model/decoding identity, normalized comparison, individual
   flag reasons, settings, and the existing repair decision. Preserve the normal
   output, cache and repair policy when the option is absent.
2. Add an optional audit-only triage command that reads pinned observations and
   audio. It may write its own evidence and hearing cache. It must not queue TTS,
   restore takes, change render metadata, mark a part ready or approve delivery.
   Avoid calling a repair routine merely to collect evidence.
3. For sampled flags, compare an isolated reading with a short contextual crop
   of the current mix and an existing second recognizer where needed. Use no
   expected-text prompt. Record VAD/crop settings and preserve conflicting raw
   readings. Run a bounded second check; rephrase persistent genuine failures
   rather than cycling through recognizers or seeds until one passes.
4. Separate physical timing defects, suspected word/meaning defects, probable
   recognition disagreements and unresolved cases. Physical silence, clipping,
   missing speech and protected-onset violations remain failures. Suspected
   number, unit, negation, relationship or omitted-clause errors require director
   review even when another recognizer passes. Model agreement alone cannot
   clear them.
5. Bind hearings to audio/model/decoding hashes. Bind review decisions to the
   wording, voice, source evidence and occurrence window too. A reused waveform
   in two windows shares a hearing, not a timing or meaning approval. Reuse
   completed checks on resume; invalidate affected decisions after a change.

The first production run observes the normal repair process without changing
its decisions. Compare recommendations with what actually happened. Any claim
about a repair the new checks would have avoided is counterfactual until that
policy is deliberately tested; it is not measured saved GPU time.

## Freeze the comparison before secondary checks

During the first normal QA pass, enroll the first dozen flagged occurrences in
source recording order (part, then line index), independent of parallel workers'
completion order. Preserve actual capture times as a separate timeline. Also
enroll a dozen passing occurrences: half short and half
longer, spread across speakers where available. Declare those duration and
speaker-sampling rules before enrollment, then freeze identities before the new
triage readings or repairs are observed. If a category is sparse, report its
actual size instead of filling it with favorable cases.

Use the existing short-line definition: at most 1.5 seconds between the first and
last detected source speech inside the checked window. Container or window
duration alone does not define that category.

Archive the common baseline audio and matching settings before repair. Keep the
control's ordinary allowed fresh take, best-prior restoration, attempts and
costs. Attach later normal repairs to the same enrolled occurrence. A wording,
voice/reference or window change creates a new context and is reported
separately; it must not improve the frozen comparison retroactively.

Existing reviewed clips may help develop the checks, but their disclosed votes
are calibration evidence. New-movie cases provide the fresh evaluation. Include
passing lines because a flag-only sample cannot reveal false negatives.

For each case, the director records supported defect, probable recognition issue,
no defect found in the bounded review, or unresolved, with evidence. The negative
review category is useful for passing controls; it does not certify acting,
likeness or all possible speech errors. Record listener preference separately. An overall
preference vote cannot certify the exact words or diagnose its own cause.

## Verification before production use

- Add meaningful regression cases for silence, dropped words, leaked prefixes,
  changed numbers/negation, names confused with ordinary words, and an error
  hidden by a high overall match. Include known defective controls that the new
  checks must still flag. Do not weaken global thresholds to fit the pilot.
- Verify invalidation after audio, text, model/settings or window changes;
  separate approvals for repeated takes; cached resumes; and no mutation of
  takes, repair queues, render metadata or readiness in observation mode.
- Format and lint changed Python, run the relevant regressions, and verify real
  rendered evidence plus a cached resume. Reuse installed models and the shared
  GPU lock; record queue/model-load costs separately.
- Review every proposed cleared flag and every sampled passing line. Keep
  unresolved semantic risks visible. A sampled missed defect blocks wider
  adoption until its cause is repaired and checked; a clean sample is limited
  evidence, not proof that all passing lines are safe.

## Measure and hand over

Report supported defects, probable recognition disagreements, unresolved cases
and sampled missed defects. Include director review burden, actual fresh-take
requests, changed/restored audio, extra repair rounds and secondary-check cache
hits. Separate observed generation/QA/remix time, queue wait, model loading,
restoration work and any unmeasured active GPU compute. Retain all failures.

Prepare six to eight private blind comparisons from cases identified before
triage outcomes. Freeze the first available flagged cases and a smaller passing
control subset in the enrollment manifest. Compare the normal repair result with
the audio the new checks recommend retaining or repairing; use actual available
audio and include unchanged pairs. Include Japanese references and sufficient
context, match gain and timing, and keep the arm key outside served assets. Ask
separately about word clarity and preferred delivery, with optional likeness
comments. Reveal assignments only after saved feedback; subsequent ratings of
those assignments are informed feedback rather than new blind evidence.

Keep the review local or offline. Preserve existing review pages and all prior
artifacts. Any public sharing needs specific later authorization for the clips
and destination. Human feedback can remain pending while the movie completes.

Finish the full movie through ordinary current-take, actual-fit, handoff, signs,
mixed/final-audio, copied-stream and subtitle gates. Preserve originals, working
takes, voice references and comparisons. Report retained storage and cleanup
status under the repository's storage policy.

Adopt a new repair policy only after its checked defect coverage, listener
outcomes and measured work justify it. Otherwise deliver with the established
workflow and retain the observation mode for further study. Record general
findings in the roadmap/research notes and source-specific evidence in `work/`.
