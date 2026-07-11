# Visible Session Replay POC

This is the core Agent hypothesis test. It preserves the visible assistant
`content` produced by one Pro startup turn and replays it to Flash with its
original role. It never stores or sends `reasoning_content`.

The formal run makes one Pro call and 144 Flash calls:

```powershell
.\.venv\Scripts\python.exe -m poc.poc_session_replay
```

The three groups are `DIRECT`, `NEUTRAL_SESSION`, and `PRO_SESSION`. Every group
receives the same complete 200-entry rule/reference package. There is no
retrieval, placeholder, fuzzy matching, checker, or repair pass. Two repetitions
measure output stability.

Run mechanics can be delegated to a low-cost model. It should only execute the
command and return the three generated paths. It must not modify files, score
translations, or write conclusions.

## Blind review

The blind file contains all outputs with group and repetition hidden. Reviewers
must judge naturalness under the confirmed constraints and must not reward
kanji/katakana or normal spacing when the user explicitly required otherwise.

After every row is scored, validate and aggregate:

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_bootstrap_review `
  --groups DIRECT,NEUTRAL_SESSION,PRO_SESSION `
  --results poc\results\session-replay-YYYYMMDD-HHMMSS.jsonl `
  --review poc\results\session-replay-blind-YYYYMMDD-HHMMSS.jsonl
```

The summary reports overall blind scores, hard-constraint pass rate, and the
combined score only among hard-compliant outputs. Hard-rule failures therefore
cannot win merely by looking like unrestricted Japanese.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m poc.poc_session_replay --dry-run
```

Dry-run generates all 144 request payloads with deterministic visible Pro
fixture content and sends no API requests.

## Reusing the visible session

```powershell
.\.venv\Scripts\python.exe -m poc.poc_session_replay `
  --transcript poc\results\session-replay-transcript-YYYYMMDD-HHMMSS.json
```

The transcript is rejected if its prompt-package hash differs or if it does not
explicitly declare that no `reasoning_content` is stored.
