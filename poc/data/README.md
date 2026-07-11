# Reference injection POC data

`reference_poc_dataset.json` was reviewed and approved on 2026-06-29.  It now
contains the 20-entry small glossary baseline, a 200-entry large glossary, and
24 test cases across six categories.  The approval followed joint checks for:

- 24-30 cases spanning `semantic`, `terminology`, `format`, `style`,
  `forbidden`, and `ocr_noise`;
- a 200-entry glossary that contains the original 20 entries and adds real
  related, ambiguous, aliased, and cross-domain terms rather than filler;
- an `expected_entry_ids` list for every case where a fixed mapping is
  expected;
- optional generic machine checks (`allowed_pattern`,
  `no_edge_whitespace`, `required_patterns`, `forbidden_patterns`);
- a final `status` of `approved` after project-owner review.

Validate without using the API:

```powershell
.\.venv\Scripts\python.exe -m poc.poc_reference_injection --validate-only
```

Inspect exact request payloads without spending API quota:

```powershell
.\.venv\Scripts\python.exe -m poc.poc_reference_injection --allow-draft --dry-run
```

The dry run writes exact request payloads to `poc/results/*.jsonl`.  It never
writes the API key.  A formal run compares A, B2, and B3 at both 20- and
200-entry scales, randomizes variant order per case, and records full request
messages, response, provider token usage (when returned), estimated tokens,
latency, placeholder integrity, terminology coverage, and generic machine
checks.

Semantic fidelity and naturalness are deliberately not auto-scored by this
runner.  Those fields require the agreed Pro blind review and human review;
adding a self-grading translation model here would contaminate the first-round
comparison.
