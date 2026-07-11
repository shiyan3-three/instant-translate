# Agent Bootstrap POC

This POC tests the project's main hypothesis rather than another checker chain:
a thinking model understands the complete prompt/reference package once,
persists that understanding locally, and a Flash model performs all routine
translations with thinking disabled.

## Formal run

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_bootstrap
```

The run performs one Pro bootstrap call followed by 72 Flash translation calls
(24 cases × 3 groups). It does not run an LLM checker or a voting system.

Groups:

- `CURRENT_FULL`: full 200-entry reference in the system prompt.
- `RETRIEVAL_RAW`: relevant-reference injection without Pro startup state.
- `AGENT_BOOTSTRAP`: relevant-reference injection plus persisted Pro state.

Version 2 uses a single-owner term protocol: exact matches become opaque
placeholders and their targets are restored locally, so Flash never receives
both a placeholder and its target mapping. OCR-uncertain cases may receive at
most two edit-distance candidates as non-binding hints. Pro state contains only
intent/style/OCR/risk understanding deltas; confirmed hard rules remain in the
deterministic prompt layer and are not repeated by Pro.

The script writes three files under `poc/results/`:

- complete request/response JSONL;
- reusable bootstrap state JSON;
- randomized blind-review JSONL with group names removed.

The summary calls automatic results `machine_constraint_pass`. It intentionally
does not calculate a translation-quality pass rate. Fill in semantic fidelity
and naturalness in the blind-review file before comparing group identities.

After every blind row has a persisted 0-5 score, reveal and aggregate groups:

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_bootstrap_review `
  --results poc\results\agent-bootstrap-YYYYMMDD-HHMMSS.jsonl `
  --review poc\results\agent-bootstrap-blind-YYYYMMDD-HHMMSS.jsonl
```

The review command refuses missing, non-numeric, out-of-range, duplicate, or
unmatched scores. A narrative report without this auditable summary is not a
completed blind review.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_bootstrap --dry-run
```

Dry-run uses deterministic fixture state, sends no API requests, and verifies
that all 72 Flash payloads and blind-review paths can be generated.

## Reuse the local session

To prove that startup understanding survives a later process, reuse the saved
state without calling Pro:

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_bootstrap --state poc\results\agent-bootstrap-state-YYYYMMDD-HHMMSS.json
```

The persisted state contains explicit conclusions, not hidden reasoning. API
models do not share activations; local persistence works by reconstructing the
stable system context from this state on every Flash request.

States produced by version 1 are intentionally rejected. Run Pro bootstrap
again after a prompt package or state-schema change.
