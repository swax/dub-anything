# Roadmap

Remaining work, in priority order. Current capabilities are in the
[README](../README.md); findings and completed experiments are in
[research.md](research.md).

## Next dub: improve QA

- [ ] Distinguish speech defects, recognition disagreements and timing problems
  before requesting another take. Add structured evidence and contextual second
  checks in observation mode, keeping the normal repair policy as control.
- [ ] Review flagged and passing lines to measure missed defects as well as
  unnecessary alarms. Preserve checks for meaning changes and speaker boundaries.
- [ ] Measure repair work and private listening outcomes before changing the
  default. Preserve acceptable performances rather than chasing endpoint scores.

Implementation and evaluation: [QA pilot plan](plans/qa-triage-pilot.md).

## After that

- **More usable per-line references:** transcripts for long voice clips and
  kana-aware window checks, where fallback references cause audible problems.
- **Pauses inside lines:** preserve internal pauses during fill.
- **Caption recognition:** better Japanese OCR; use caption colours and name
  brackets as speaker suggestions.
- **American-voice conversion:** preserve words and intonation while improving
  likeness.
- **Listener-reported mix issues:** fix line edges and scene acoustics as heard.

## Parked

Short-line candidates are implemented and remain optional. The listening trial
did not establish a benefit worth broader use. Revisit only for specific failures
that normal QA and rewording cannot resolve; see [research finding 23](research.md).

## Later

- Lip sync.
- Additional language pairs and hardware; native Japanese/Spanish target voices.
- Upstream VoiceStudio patches, per-segment references and effects, optional
  Roformer separation, and the free-text instruction fix.
