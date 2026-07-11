# Completeness-only Runtime Profile A/B

This frozen zero-Pro experiment tests whether one short, confirmed
`completeness_checks` instruction reduces structural omissions without causing
new lexical drift.

## Isolation

- `CURRENT_RUNTIME` uses the current production-style Flash request.
- `COMPLETENESS_ONLY_RUNTIME` uses the identical frozen request and adds only
  `RuntimeProfile.render()` to the system message.
- The Profile has exactly one completeness check and no semantic or style
  directives, terminology, format instructions, examples, or preferred
  translations.
- Each of 16 cases freezes Reference, Feedback Memory, technical terms, and the
  OCR user message once before the 64 A/B jobs start.
- Four POC-only Reference entries and three Feedback Memory rules live in a
  temporary isolated directory. They do not change settings or real user data.
- Formal design: 0 Pro, 64 Flash, 0 judge, 0 retry.
- Dry-run uses local fixtures and never accesses the network. It validates the
  harness and token estimate, not translation quality.

## Decision gates

A formal Flash-only result passes only when all of these hold:

- major structural-completeness errors fall by at least 30%;
- at least two different cases improve consistently in both repetitions;
- zero new major lexical errors (one is an immediate failure);
- control-group semantics do not decline;
- local validation does not decline;
- input-token increase is at most 5%;
- repetition stability declines in at most one case;
- p95 latency increases by at most 15%.

If the dry-run token estimate exceeds 5%, the summary explicitly reports
`token gate likely failed before formal run`; the gate is never relaxed.

Run the offline harness from the repository root:

```powershell
.\.venv\Scripts\python.exe -m poc.poc_runtime_profile_completeness --dry-run
```

Do not omit `--dry-run` unless a separate formal Flash-only run is explicitly
authorized.
