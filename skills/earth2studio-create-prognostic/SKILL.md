---
name: earth2studio-create-prognostic
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags:
    - earth2studio
    - earth2
    - python
    - prognostic
    - model-wrapper
    - pytorch
description: >
  Create a new Earth2Studio prognostic model (px) wrapper end-to-end.
  Do NOT use for diagnostic models, data sources, or non-inference tasks.
argument-hint: URL or local path to reference inference script/repo (optional)
---

# Create and Validate Prognostic Model Wrapper

## Purpose

Create a new Earth2Studio prognostic model (px) wrapper end-to-end — from
analyzing a reference inference script through implementation, testing,
reference comparison, and PR submission.

## Prerequisites

- Earth2Studio dev environment (`uv sync --all-extras` or `.venv` active)
- `uv run python` for all Python commands (never system Python)
- Git repository with write access to a fork of `NVIDIA/earth2studio`
- Reference inference script or model repository available

## Reference Files

Load these on demand during the relevant steps:

| File | Content | Load at |
|---|---|---|
| `references/implementation-guide.md` | Skeleton, templates, coordinates, loading | Steps 3–8 |
| `references/testing-guide.md` | Smoke tests, pytest patterns, coverage | Steps 9–10 |
| `references/validation-guide.md` | Reference comparison, plots, PR, code review | Steps 11–13 |

---

## Instructions

Follow every step in order. Each `[CONFIRM]` gate requires explicit user
approval before proceeding.

### Workflow Overview

```text
Step 0: Obtain reference → Step 1: Analyze & propose deps
→ Step 2: Add deps → Step 3: Create skeleton → Step 4: Coordinates
→ Step 5: Forward pass → Step 6: Model loading → Step 7: Register
→ Step 8: Format/lint/docs → Step 9: Smoke tests → Step 10: Pytest
→ Step 11: Reference comparison → Step 12: PR → Step 13: Code review
```

---

## Step 0 — Obtain Reference Script / Repository

If `$ARGUMENTS` is provided, use it (URL → WebFetch; file path → Read).

If not provided, ask:

> Please provide a reference inference script or repository URL/path that
> demonstrates how this model runs inference.

---

## Step 1 — Analyze Reference & Propose Dependencies

### 1a. Analyze the reference code

Identify: Python packages, model architecture (PyTorch/ONNX/JAX), input/output
shapes, time step, spatial resolution, checkpoint format.

### 1b. Propose pyproject.toml dependency group

Follow the group format in `references/implementation-guide.md § Dependencies`.
Group name: lowercase-hyphenated. Also add to `all` aggregate (px models line).

### [CONFIRM — Dependencies]

Present: 1) Group name, 2) Package list with licenses, 3) Ask if correct.

---

## Step 2 — Add Dependencies to pyproject.toml

Edit `pyproject.toml`: add group alphabetically, add to `all` aggregate.

Use the optional dependency imports pattern (try/except with
`OptionalDependencyFailure`).

---

## Step 3 — Create Skeleton Class File

> **Load `references/implementation-guide.md` from here through Step 8.**

### 3a. Determine naming

- **Class**: PascalCase (e.g., `Pangu24`, `Aurora`)
- **File**: `earth2studio/models/px/<lowercase>.py`

### 3b. Write skeleton with pseudocode

Use the skeleton template from the implementation guide. Must use triple
inheritance: `torch.nn.Module + AutoModelMixin + PrognosticMixin`.

### [CONFIRM — Skeleton]

Present: 1) Class name, 2) File path, 3) Ask if acceptable.

---

## Step 4 — Implement Coordinate System

### 4a. Map variables to E2STUDIO_VOCAB

Read `earth2studio/lexicon/base.py`. Verify all model variables exist (282 entries).
Flag missing variables for discussion.

### 4b. Implement input_coords

Rules: `batch` first with `np.empty(0)`, `time` is `np.empty(0)`, `lead_time`
starts at `np.timedelta64(0, "h")`, `lat`: 90→-90, `lon`: 0→360.

### 4c. Implement output_coords

Include `handshake_dim`/`handshake_coords` validation. Increment `lead_time`.

### [CONFIRM — Coordinates]

Present: 1) Variable list + mapping issues, 2) Spatial dimensions, 3) History
needs, 4) Time step.

---

## Step 5 — Implement Forward Pass

### 5a. Implement `__call__`

Key notes:

- `@batch_func()` handles batch dimension
- Input shape: `(batch, time, lead_time, variable, lat, lon)`
- Reshape to model format → call model → reshape back

### 5b. Implement create_iterator

- MUST yield initial condition first (step 0)
- Use `front_hook`/`rear_hook` for perturbation injection

### [CONFIRM — Forward Pass]

Ask: 1) Is reshape logic correct? 2) Special considerations?

---

## Step 6 — Implement Model Loading

### 6a. Implement load_default_package

Lock HuggingFace URLs to commit: `hf://org/repo@commit`.

### 6b. Implement load_model

Use `package.resolve()`, `map_location="cpu"`, `eval()` mode,
`@check_optional_dependencies()`.

### 6c. Implement .to() (optional)

Only override if non-PyTorch state exists (ONNX, JAX). Otherwise omit.

### [CONFIRM — Model Loading]

Present: 1) Checkpoint URL, 2) File names/loading logic, 3) `.to()` needed?

---

## Step 7 — Register the Model

See `references/implementation-guide.md § Registration & Documentation` for details.

- Add import alphabetically to `earth2studio/models/px/__init__.py`
- Verify `pyproject.toml` dependency group

---

## Step 8 — Format, Lint, Documentation & CHANGELOG

### 8a. Run checks

```bash
make format && make lint && make license
```

### 8b. Documentation

- Add to `docs/modules/models_px.rst` (alphabetically)
- Add to `docs/userguide/about/install.md` Prognostics section
- NumPy-style docstrings on all public methods
- Reference URL in class `Note` section

### 8c. CHANGELOG

Update `CHANGELOG.md` (see `references/implementation-guide.md § Update CHANGELOG.md`
for templates).

---

## Step 9 — Smoke Tests

> **Load `references/testing-guide.md` for Steps 9–10.**

Run quick smoke test and data fetch test scripts to verify basic forward pass
and data pipeline compatibility. Report results.

---

## Step 10 — Write & Run Pytest Unit Tests

### 10a. Write test file

Create `test/models/px/test_<filename>.py` with:

- `PhooModelName` dummy model adapted to actual model interface
- `TestModelNameMock` class: `test_model_call`, `test_model_iter`, `test_model_exceptions`
- `test_model_package` integration test (`@pytest.mark.package`)

### 10b. Run mock tests

```bash
uv run python -m pytest test/models/px/test_<filename>.py -m "not package" -v
```

### 10c. Run package test

### [CONFIRM — Package Test]

Warn user: downloads checkpoint (potentially GB), requires GPU, takes time.
Only run after confirmation.

### 10d. Coverage

```bash
uv run python -m pytest test/models/px/test_<filename>.py -v \
    --slow --timeout=300 \
    --cov=earth2studio/models/px/<filename> \
    --cov-report=term-missing \
    --cov-fail-under=90
```

Target: >= 90% line coverage.

---

## Step 11 — Reference Comparison & Sanity-Check

> **Load `references/validation-guide.md` for Steps 11–13.**

### 11a. Create reference scripts (do NOT commit)

1. Vanilla reference script (third-party only)
2. Earth2Studio reference script
3. Comparison script

Run all three and report metrics.

### 11b. Create sanity-check plot script

Side-by-side visualization comparing reference vs E2S output.

### [CONFIRM — Reference Comparison]

Present comparison metrics and scripts. User must confirm results look correct.

---

## Step 12 — Branch, Commit & Open PR

### [CONFIRM — Ready to Open PR]

Verify all complete: implementation, triple inheritance, coordinates,
forward pass, iterator, model loading, registration, docs, CHANGELOG,
format/lint, tests passing >= 90% coverage.

### 12a. Branch and commit

Do NOT add reference scripts, comparison scripts, or images.

### 12b. Push to fork and open PR

See `references/validation-guide.md § Open PR` for the full `gh pr create`
command and PR body template.

### 12c. Post sanity-check validation as PR comment

---

## Step 13 — Automated Code Review

1. Poll for Greptile review (5 min timeout)
2. Categorize feedback (bug/style/perf/docs/suggestion/false-positive)
3. Present triage table to user

### [CONFIRM — Review Triage]

User approves which comments to address.

1. Implement accepted fixes
2. Respond to PR comments
3. Push

---

## Examples

Typical invocation:

> "Wrap the Pangu-Weather 24h model as an Earth2Studio prognostic. Here's
> the reference script: <https://github.com/198808xc/Pangu-Weather/blob/main/inference.py>"

---

## Limitations

- One model per invocation — do not batch multiple models
- Requires network access for checkpoint download and HuggingFace resolution
- GPU required for package tests and reference comparison
- Cannot wrap non-autoregressive models (those are diagnostic, not prognostic)

---

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `OptionalDependencyFailure` | Missing model extras | `uv sync --extra <model-name>` |
| `handshake_dim` assertion | Mismatched dim index | Check dimension order in `output_coords` |
| Coverage < 90% | Untested branches | Add parametrized edge cases |
| `load_model` hangs | Large checkpoint, slow network | Increase timeout, verify URL |
| `create_iterator` wrong shape | Missing batch dim handling | Verify `@batch_func()` decorator |

---

## Reminders

- **DO** use `uv run python` for all Python commands
- **DO** use `loguru.logger`, never `print()`, inside `earth2studio/`
- **DO** maintain alphabetical order in `__init__.py`, RST, CHANGELOG
- **DO** follow canonical method ordering (see implementation guide)
- **DO** use triple inheritance: `torch.nn.Module + AutoModelMixin + PrognosticMixin`
- **DO** yield initial condition first in `create_iterator` (step 0)
- **DO** use `front_hook()`/`rear_hook()` in `create_iterator`
- **DO** include `lead_time` starting at `np.timedelta64(0, "h")`
- **DO** use `handshake_dim` indices matching dimension position
- **DO** include reference URLs in class docstrings
- **DO** update CHANGELOG.md under current unreleased version
- **DO** run `make format && make lint` before finalizing
- **DO NOT** make a general base class for reuse across models
- **DO NOT** over-populate `load_model()` API
- **NEVER** commit reference scripts, comparison scripts, or images
- **NEVER** commit API keys, secrets, tokens, or credentials
