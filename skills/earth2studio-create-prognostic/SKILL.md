---
name: earth2studio-create-prognostic
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team <agent-skills@nvidia.com>
  tags:
    - earth2studio
    - earth2
    - python
    - prognostic-model
    - integration
description: >
  Create Earth2Studio prognostic (time-stepping forecast) model wrappers from
  reference scripts. Covers implementation, testing, validation, and PR.
  Do NOT use for diagnostic models, data sources, or installation.
argument-hint: URL or local path to reference inference script/repo (optional)
---

# Create and Validate Prognostic Model Wrapper

## Quick Start Checklist

For agents and developers — follow this order:

1. **Read this SKILL.md first** (you're doing it now)
2. **Load `references/skeleton-template.py`** — copy and adapt
3. **Write `earth2studio/models/px/<model>.py`** with triple inheritance
4. **Write `test/models/px/test_<model>.py`** with Phoo mock
5. **Run `uv run pytest test/models/px/test_<model>.py -v`**
6. **Run `make format && make lint`**

Skip Steps 8-12 (registration, docs, PR) unless explicitly requested.

> **Minimize exploration.** The templates in `references/` contain the key
> patterns. Read `skeleton-template.py` and `testing-guide.py` first. If you
> need clarification on imports or coordinate systems, briefly check
> `earth2studio/models/px/__init__.py` or one existing model — but do not
> launch subagents or read multiple model files.

---

## Purpose

End-to-end workflow for implementing a new Earth2Studio prognostic model wrapper
that connects a third-party ML weather model to Earth2Studio's inference
infrastructure — from analysis through implementation, testing, validation, and
PR submission.

> **What is a prognostic model?** Prognostic models are forecast models that
> time-integrate forward — given an initial state, they predict future states
> by stepping through time (e.g., 6-hour increments). Examples include Pangu,
> GraphCast, FourCastNet, and Aurora. In contrast, diagnostic models compute
> derived quantities from a single time step without time integration.

## Prerequisites

- Earth2Studio dev environment with `uv` (`uv run python` must work)
- Git configured with fork (`origin`) and upstream (`upstream`) remotes
- Access to the model checkpoint (HuggingFace, NGC, S3, or local)
- Python 3.10+

## Workspace

| Context | Where to work |
|---|---|
| Harbor eval (default) | Skill at `/workspace/skills/`; write deliverables under `/workspace/output/` preserving paths like `earth2studio/models/px/...` and `test/models/px/...` |
| Harbor eval with `--copy-repo` | Full checkout at `/workspace/repo` (`EARTH2STUDIO_ROOT`) |
| Local clone | Directory containing `pyproject.toml` |

**Never read `evals/targets/`** — grader references only. Use
`references/skeleton-template.py`, `references/method-templates.py`, and
`references/testing-guide.py`.

## Instructions

> **Python Environment:** Always use `uv run python` or the local `.venv`.
> Never use the system Python directly.

Follow every step in order.

> **[CONFIRM] gates:** Only Step 1 (Dependencies) and Step 10 (Sanity-Check
> Plots) require explicit user approval. All other steps summarize decisions
> inline and proceed without blocking.
>
> **Deliverables first:** Write the model file and test file (Steps 3–7)
> before extended exploration, documentation, registration, CHANGELOG, or PR
> work. Skip Steps 8–12 when the user asks for implementation only.
>
> **Before you finish:** Run verification commands in the repo root so results
> appear in the session log:
>
> ```bash
> uv run pytest test/models/px/test_<model>.py -x
> make format && make lint
> ```
>
> **Be concise:** Avoid long architecture reports; summarize decisions in a
> few sentences and move on to file writes.

### Reference Files

Load these on demand during the relevant steps:

| File | Content | Load at |
|---|---|---|
| `references/skeleton-template.py` | Skeleton model class with canonical method ordering | Steps 3–5 |
| `references/method-templates.py` | Method implementation templates with docstrings | Steps 3–5 |
| `references/testing-guide.py` | Test skeleton with mock and package tests | Step 7 |
| `references/validation-guide.md` | Reference comparison, plots, PR, code review | Steps 10–12 |

---

### Workflow Overview

```text
Step 0: Obtain reference → Step 1: Dependencies → Step 2: Add deps
→ Step 3: Create skeleton → Step 4: Coordinates → Step 5: Forward pass
→ Step 6: Model loading → Step 7: Tests → Step 8: Register
→ Step 9: Documentation → Step 10: Validate & plots
→ Step 11: PR → Step 12: Code review
```

---

### Step 0 — Obtain Reference Script / Repository

If `$ARGUMENTS` is provided, use it (URL → WebFetch; file path → read).

If empty, ask:

> Please provide a reference inference script or repository URL/path that
> demonstrates how this model runs inference.

---

### Step 1 — Examine Reference & Propose Dependencies

**Analyze:** Python packages, model architecture (PyTorch/ONNX/JAX), input/output
shapes, time step, spatial resolution, checkpoint format.

**Propose pyproject.toml group:**

```toml
model-name = ["package1>=version", "package2"]
```

Group name: lowercase-hyphenated. Also add to `all` aggregate (px models line).

#### [CONFIRM — Dependencies]

Present: 1) Group name, 2) Package list with licenses, 3) Ask if correct.

---

### Step 2 — Add Dependencies to pyproject.toml

Edit `pyproject.toml`: add group alphabetically, add to `all` aggregate.

---

### Step 3 — Create Skeleton Class File

> **Load `references/skeleton-template.py` and `references/method-templates.py`
> from here through Step 6.**

**Naming:**
- Class: PascalCase (e.g., `Pangu24`, `Aurora`)
- File: `earth2studio/models/px/<lowercase>.py`

**Canonical method ordering:**
1. `__init__` 2. `input_coords` 3. `output_coords` (@batch_coords)
4. `load_default_package` 5. `load_model` 6. `to` (optional)
7. Private methods 8. `__call__` (@batch_func) 9. `_default_generator`
10. `create_iterator`

Every `.py` file MUST start with the SPDX Apache-2.0 license header.

Summarize: class name, file path — then proceed.

---

### Step 4 — Implement Coordinate System

**Map variables to E2STUDIO_VOCAB** (282 entries in `earth2studio/lexicon/base.py`).
Pressure levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000.
Flag missing variables for discussion.

**input_coords rules:**
- `batch` first with `np.empty(0)`
- `time` is `np.empty(0)` (dynamic)
- `lead_time` starts at `np.timedelta64(0, "h")`
- `lat`: 90 to -90 (north to south)
- `lon`: 0 to 360

**output_coords:** Validate with `handshake_dim`/`handshake_coords`, increment
`lead_time` by time step.

Summarize: variable list, grid size, time step — then proceed.

---

### Step 5 — Implement Forward Pass

**`__call__`:** @batch_func decorated, input shape (batch, time, lead_time,
variable, lat, lon). Reshape to model format → call model → reshape back.

**`create_iterator`:** MUST yield initial condition first (step 0). Use
`front_hook`/`rear_hook` for perturbation injection.

Summarize: reshape logic, special considerations — then proceed.

---

### Step 6 — Implement Model Loading

**`load_default_package`:** Lock HuggingFace URLs to commit: `hf://org/repo@commit`.

**`load_model`:** Use `package.resolve()`, load with `map_location="cpu"`,
set `eval()` mode, use `@check_optional_dependencies()`.

**`.to()`:** Only override for non-PyTorch state (ONNX, JAX).

Summarize: checkpoint URL, file names — then proceed.

---

### Step 7 — Write Tests

> **Load `references/testing-guide.py` for this step.**

Create `test/models/px/test_<filename>.py` with:

| Function | Marks | Purpose |
|---|---|---|
| `test_<model>_call` | parametrized device/time | Single forward pass |
| `test_<model>_iter` | parametrized device/ensemble | Iterator produces sequence |
| `test_<model>_exceptions` | none | Invalid coords raise errors |
| `test_<model>_package` | `@pytest.mark.package` | Real weights integration |

Create `PhooModelName` dummy that matches model interface for mock tests.

**Run tests:**

```bash
uv run pytest test/models/px/test_<filename>.py -m "not package" -v
```

**Note:** Package test downloads checkpoint and needs GPU — skip if not requested.

---

### Step 8 — Register the Model

- `earth2studio/models/px/__init__.py` — alphabetical import
- Verify pyproject.toml deps exist

---

### Step 9 — Update Documentation & CHANGELOG

- Add to `docs/modules/models_px.rst` (alphabetical)
- Add to `docs/userguide/about/install.md` Prognostics section
- Update CHANGELOG.md under `### Added`:
  ```markdown
  - Added <ModelName> prognostic model (`<ClassName>`) with <brief description>
  ```

**Format and lint:**

```bash
make format && make lint && make license
```

---

### Step 10 — Reference Comparison & Validation

> **Load `references/validation-guide.md` for Steps 10–12.**

Create three scripts (do NOT commit):
1. `reference_<model>_vanilla.py` — third-party only inference
2. `reference_<model>_e2s.py` — E2S wrapper inference
3. `reference_<model>_compare.py` — numerical comparison

Run and verify single-step and multi-step (8 steps) match within tolerance.

#### [CONFIRM — Sanity-Check Plots]

User MUST visually inspect plots. Do not proceed without confirmation.

---

### Step 11 — Branch, Commit & Open PR

1. Create branch `feat/prognostic-model-<name>`
2. Commit (do NOT add comparison scripts/images)
3. Push to fork
4. Create PR with template from `references/validation-guide.md` — include:
   - Model details (architecture, time step, resolution)
   - **License information** (model weights, code, training data)
   - Dependencies added with licenses
   - Reference comparison results
5. `gh pr create --repo NVIDIA/earth2studio --body-file pr-body.md`
6. Post reference comparison as PR comment

---

### Step 12 — Automated Code Review

1. Poll for Greptile review (5 min timeout)
2. Categorize feedback (bug/style/perf/docs/suggestion/false-positive)
3. Present triage table — implement obvious fixes, ask only for ambiguous items
4. Respond to PR comments
5. Push

---

## Examples

### Example 1: Pangu-Weather

```text
User: Add a prognostic model wrapper for Pangu-Weather
      - Paper: https://arxiv.org/abs/2211.02556
      - GitHub: https://github.com/198808xc/Pangu-Weather
      - Inference script: https://github.com/198808xc/Pangu-Weather/blob/main/inference.py
      - Call it Pangu24 (24-hour model)

Agent: [loads skill, reads inference.py, creates earth2studio/models/px/pangu.py
        with class Pangu24, writes tests, runs pytest and lint]
```

### Example 2: GraphCast

```text
User: Wrap GraphCast for Earth2Studio
      - Paper: https://arxiv.org/abs/2212.12794
      - GitHub: https://github.com/google-deepmind/graphcast
      - Checkpoint: HuggingFace deepmind/graphcast
      - Name: GraphCast

Agent: [loads skill, analyzes graphcast repo, creates earth2studio/models/px/graphcast.py,
        handles JAX->PyTorch conversion if needed, writes tests]
```

### Example 3: Simple test model

```text
User: Create an IdentityModel that returns input unchanged, 6h step, 181x360 grid,
      variables t2m, u10m, v10m, msl

Agent: [loads skill, skips Step 0-1 (no external deps), writes identity.py and
        test_identity.py, runs pytest and lint]
```

---

## Limitations

- One model per invocation
- Cannot create diagnostic models (use different skill)
- Requires checkpoint access for validation (Step 10)
- SPDX license headers are required in all generated files

## Troubleshooting

| Error | Cause | Solution |
|---|---|---|
| `OptionalDependencyFailure` | Missing optional pkg | `uv add --optional <group> <pkg>` |
| Coordinate handshake fails | Wrong dim order/values | Check `handshake_dim` indices |
| Iterator yields wrong shapes | Reshape logic error | Debug with random input |
| Package test fails | Checkpoint issue | Verify URL and format |

---

## Reminders

- **DO** use `uv run python` for all Python commands
- **DO** use `loguru.logger`, never `print()`
- **DO** maintain alphabetical order in `__init__.py`, RST, CHANGELOG
- **DO** follow canonical method ordering
- **DO** yield initial condition first in `create_iterator`
- **DO** use `front_hook()`/`rear_hook()` in `_default_generator`
- **DO** include reference URLs in class docstrings
- **DO** inherit `torch.nn.Module + AutoModelMixin + PrognosticMixin`
- **DO** use `handshake_dim` indices matching dim position in CoordSystem
- **DO NOT** make a general base class for reuse across models
- **DO NOT** over-populate `load_model()` API
- **NEVER** commit API keys, secrets, or credentials
- **NEVER** commit comparison scripts or images
