# POC workspace

All translation experiments now live under one directory:

- `poc_*.py`: runnable experiment and review modules;
- `data/`: reviewed datasets and experiment-specific instructions;
- `results/`: generated requests, responses, state, blind reviews, and summaries;
- `legacy/scripts/`: early standalone scripts migrated from `C:\tmp`;
- `results/legacy/`: early generated JSONL files migrated from `C:\tmp`.

Run current experiments from the repository root as modules, for example:

```powershell
.\.venv\Scripts\python.exe -m poc.poc_agent_retrieved_policy --dry-run
.\.venv\Scripts\python.exe -m poc.poc_agent_retrieved_policy_judge
.\.venv\Scripts\python.exe -m poc.poc_semantic_guard --dry-run
.\.venv\Scripts\python.exe -m poc.poc_agent_semantic_profile --dry-run
.\.venv\Scripts\python.exe -m poc.poc_agent_runtime_projection --dry-run --reuse-current-profile-state poc\results\runtime-projection-state-20260710-190955.json
.\.venv\Scripts\python.exe -m poc.poc_runtime_profile_ab --dry-run
.\.venv\Scripts\python.exe -m poc.poc_runtime_profile_completeness --dry-run
```

Generated results are intentionally excluded from Git. Reviewed datasets and
scripts remain ordinary project files.
