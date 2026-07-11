# Pro Semantic Agent Profile A/B POC

## Question

Does replacing the current format-only Pro digest with a structured semantic
Agent Profile improve Flash semantic fidelity and natural Japanese terminology
without regressing hard-format compliance or latency?

## Groups

- `FORMAT_PROFILE`: Pro receives the current production digest instruction,
  which asks primarily for character, terminology-format, spacing, and
  punctuation rules.
- `SEMANTIC_PROFILE`: Pro receives a strict JSON contract covering semantic
  fidelity, natural target-language expression, terminology strategy,
  ambiguity handling, completeness, and the same format checklist.

Both profiles are rendered through the same single `<AGENT_PROFILE>` system
block. Both Flash groups share the exact production prompt, policy, reference
package, request wrapper, model, temperature, OCR source, normalizer, and
validator. The only variable is Profile content.

## Isolation and trust boundary

- The 12 evaluation sources never enter either Pro payload or Profile.
- Eight cases are exact failures/quality problems observed in the 2026-07-10
  runtime log; four are non-software controls.
- Pro may produce soft semantic guidance but cannot create glossary tables,
  source-to-target mappings, fixed readings, or local hard rules.
- User Knowledge References remain the only authoritative term mappings.
- There is no Pro judge, semantic auto-score, Thinking retry, or automatic API
  retry. Blind human review determines semantic quality.

## Fixed call count

- 2 Pro calls: one Profile per group, `thinking=enabled`.
- 48 Flash calls: 12 cases × 2 groups × 2 repetitions,
  `thinking=disabled`, `temperature=0.0`.
- 0 Pro judge calls and 0 retries.

## Offline validation

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_poc_agent_semantic_profile
.\.venv\Scripts\python.exe -m poc.poc_agent_semantic_profile --dry-run
```

`--dry-run` must create 48 production-shaped results without making network
requests. Dry fixtures are only plumbing checks and are not semantic evidence.

## Formal run

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_semantic_profile
```

Run it exactly once. Return the complete terminal output and the three printed
paths: `Results`, `Profile state`, and `Blind review`. Do not modify files,
score translations, analyze results, retry failed calls, or perform Git
operations during execution.

## Blind review fields

For every blind row, fill:

- `semantic_fidelity_0_to_5`
- `naturalness_0_to_5`
- `terminology_0_to_5`
- `complete_sentence`
- `major_error`
- `review_notes`

Group and repetition are intentionally hidden from the blind file.

## Predeclared decision gate

Integrate the semantic Profile only if all conditions hold:

- blind semantic-fidelity mean improves by at least 0.4;
- blind naturalness mean improves by at least 0.4;
- major errors on the eight runtime regressions fall by at least 30%;
- incomplete sentences do not increase;
- hard-format pass loses at most one result;
- Flash P95 latency increases by no more than 30%.

Otherwise, do not replace the production Profile. A successful dry run or
machine format summary cannot establish semantic improvement.

