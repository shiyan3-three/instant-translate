# Runtime Profile 0 Pro Flash A/B POC

This frozen experiment isolates one runtime change: injecting a locally
confirmed `RuntimeProfile.render()` block into the Flash system message.

## Groups

- `CURRENT_RUNTIME`: the current production-style Flash bootstrap and OCR
  wrapper, including the format checklist.
- `RUNTIME_PROFILE_RUNTIME`: the identical frozen request context, with the
  confirmed semantic Runtime Profile appended to the system message.

The Profile contains semantic interpretation, natural style, and completeness
checks only. Format policy, term wrapping, glossary mappings, punctuation, and
spacing remain owned by the existing Policy and Reference layers.

## Frozen design

- 12 new cases: 8 runtime regressions and 4 ordinary-life controls.
- 2 repetitions and 2 groups: 48 Flash requests in a formally authorized run.
- 0 Pro, 0 judge, and 0 retry.
- Reference matches, feedback memory, technical terms, and OCR user content are
  read once per case and frozen before any A/B jobs execute.
- The dry run uses local fixture translations and performs no network calls. It
  validates artifact shape and request isolation, not translation quality.

Run the offline check from the repository root:

```powershell
.\.venv\Scripts\python.exe -m poc.poc_runtime_profile_ab --dry-run
```

Do not run the command without `--dry-run` unless a separate formal Flash-only
experiment has been explicitly authorized.
