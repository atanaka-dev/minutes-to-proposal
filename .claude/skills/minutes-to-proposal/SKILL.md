---
name: minutes-to-proposal
description: Run this repository's presales agent flow without Streamlit and generate proposal artifacts from meeting notes or RFP files. Use when asked to execute the same core pipeline as the GUI (Ask/Assume extraction, knowledge lookup, proposal build, quality critique, demo-app generation, trace persistence) from Claude Code or compatible agents.
---

# Minutes to Proposal

## Goal

Execute the same `AgentLoop` pipeline used by `app/main.py`, but from CLI so GUI is not required.

## Run Workflow

1. Assume current directory is repository root (`pyproject.toml` exists).
2. Verify dependencies are installed once with `poetry install` when environment is not ready.
3. Run the pipeline:

```bash
poetry run python scripts/run.py --input <path-to-note-or-rfp>
```

4. Read JSON output and report at minimum:
- `success`
- `run_id`
- `run_dir`
- `artifacts`
- `ask_blocker_count`
- `confirmation_count`
- `demo_app_type`

5. If user asks for model changes, pass overrides:

```bash
poetry run python scripts/run_presales_agent.py \
  --input <path> \
  --extract-model gpt-5-nano \
  --generate-model gpt-5-mini \
  --planner-model gpt-5-mini
```

6. If user requests a machine-readable artifact summary, add `--output-json <path>`.

## Expected Artifacts

Inspect `artifacts/<client_project>/<run_id>/` and confirm:

- `proposal.html`
- `demo_app/app.py`
- `trace.jsonl`
- `metadata.json`
- `input_snapshot.md`
- `agent_snapshot.json`

## Input Rules

- Prefer UTF-8 `.md` or `.txt`.
- Reject empty input files.
- If user does not provide input, suggest examples under `demo_inputs/`.

## Error Handling

- Exit code `0`: run succeeded.
- Exit code `1`: agent completed with failure state.
- Exit code `2`: invalid input file (missing, non-UTF-8, or empty).
