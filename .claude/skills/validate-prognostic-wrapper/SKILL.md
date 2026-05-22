---
name: validate-prognostic-wrapper
description: Validate a newly created Earth2Studio prognostic model wrapper by running tests with coverage, performing reference comparison (single-step and multi-step), generating sanity-check plots, and opening a PR with automated code review. Use this skill after completing the create-prognostic-wrapper and create-prognostic-tests skills.
argument-hint: Name of the prognostic model class and test file (optional — will be inferred from recent changes if not provided)
---

# Validate Prognostic Model Wrapper

Validate a newly created Earth2Studio prognostic model (px) wrapper by
running tests with coverage, performing reference comparison (single-step
and multi-step), generating sanity-check plots, and opening a PR with
automated code review.

**Prerequisites:** This skill assumes:

1. The prognostic wrapper has been implemented (`create-prognostic-wrapper` skill)
2. Unit tests have been written (`create-prognostic-tests` skill)

> **Python Environment:** This project uses **uv** for dependency
> management. Always use the local `.venv` virtual environment
> (`source .venv/bin/activate` or prefix with `uv run python`) for all
> Python commands — installing packages, running tests, executing
> scripts, etc. Use `uv add` / `uv pip install` / `uv lock` instead of
> `pip install`.

Each confirmation gate marked by:

```markdown
### **[CONFIRM — <Title>]**
```

requires **explicit user approval** before proceeding.

---

## Step 1 — Run Tests

### 1a. Run the new test file

```bash
uv run python -m pytest test/models/px/test_<filename>.py -v --timeout=60
```

All tests must pass. Fix failures and re-run until green.

### 1b. Run coverage report with `--slow` tests

Run the new test file **with coverage** and the `--slow` flag to
include integration tests. The new prognostic model file must achieve
**at least 90% line coverage**:

```bash
uv run python -m pytest test/models/px/test_<filename>.py -v \
    --slow --timeout=300 \
    --cov=earth2studio/models/px/<filename> \
    --cov-report=term-missing \
    --cov-fail-under=90
```

- `--slow` enables integration tests (marked `@pytest.mark.slow`)
- `--cov=earth2studio/models/px/<filename>` scopes coverage to the
  new model module only
- `--cov-report=term-missing` shows which lines are not covered
- `--cov-fail-under=90` fails the run if coverage is below 90%

If coverage is below 90%, add additional tests or mock tests to
cover the missing lines. Common coverage gaps for px models:

- Error handling in `output_coords` (wrong variable names, wrong dims)
- Device management paths (CPU vs CUDA)
- `create_iterator` edge cases (initial condition yield, hook calls)
- `load_model` and `load_default_package` (needs mock or package test)
- ONNX / non-PyTorch backend `.to()` logic

Re-run until coverage is at or above 90%.

### 1c. Run the full model test suite (optional but recommended)

```bash
make pytest TOX_ENV=test-models
```

Confirm no regressions in existing model tests.

---

## Step 2 — Branch, Commit & Open Initial PR

### **[CONFIRM — Ready to Open Initial PR]**

Before proceeding, confirm with the user:

> All implementation and basic validation steps are complete:
>
> - Prognostic model class implemented with correct method ordering
> - Triple inheritance: `torch.nn.Module + AutoModelMixin + PrognosticMixin`
> - Coordinate system with proper `handshake_dim` indices
> - `__call__` (single step) and `create_iterator` (multi-step) implemented
> - `create_iterator` yields initial condition first, uses front/rear hooks
> - Model loading implemented (`load_default_package`, `load_model`)
> - Registered in `earth2studio/models/px/__init__.py`
> - Documentation added to `docs/modules/models_px.rst`
> - Reference URLs included in class docstrings
> - CHANGELOG.md updated
> - Format, lint, and license checks pass
> - Unit tests written and passing with >= 90% coverage
> - Dependencies in pyproject.toml confirmed
>
> Would you like to open an initial PR now? This allows reviewers to
> see the implementation while you continue with reference comparison
> validation.

If the user declines, skip to Step 3 and return to Step 2 after
Step 3 is complete.

### 2a. Create branch and commit

```bash
git checkout -b feat/prognostic-model-<name>
git add earth2studio/models/px/<filename>.py \
        earth2studio/models/px/__init__.py \
        test/models/px/test_<filename>.py \
        pyproject.toml \
        CHANGELOG.md \
        docs/modules/models_px.rst
git commit -m "feat: add <ClassName> prognostic model

Add <ClassName> prognostic model for <brief description>.
Includes unit tests and documentation."
```

Do **NOT** add the reference scripts, comparison scripts, or
their output images.

### 2b. Identify the fork remote and push branch

The working repository is typically a **fork** of
`NVIDIA/earth2studio`. Before pushing, confirm which git remote
points to the user's fork:

```bash
git remote -v
```

Ask the user:

> Which git remote is your fork of `NVIDIA/earth2studio`?
> (Usually `origin` — e.g., `git@github.com:<user>/earth2studio.git`)

Then push the feature branch to the **fork** remote:

```bash
git push -u <fork-remote> feat/prognostic-model-<name>
```

### 2c. Open Pull Request (fork -> NVIDIA/earth2studio)

> **Important:** PRs must be opened **from the fork** to the
> **upstream source repository** `NVIDIA/earth2studio`. The branch
> lives on the fork; the PR targets `main` on the upstream repo.

Use `gh pr create` with explicit `--repo` and `--head` flags:

```bash
gh pr create \
  --repo NVIDIA/earth2studio \
  --base main \
  --head <fork-owner>:feat/prognostic-model-<name> \
  --title "feat: add <ClassName> prognostic model" \
  --body "..."
```

Where `<fork-owner>` is the GitHub username that owns the fork.

Use the PR body template from this skill's directory:

```text
.claude/skills/validate-prognostic-wrapper/pr-body-template.md
```

Read the template, fill in the placeholders with actual values, and use
as the `--body` argument. Key fields to populate:

- Model details table (architecture, time step, variables, resolution, etc.)
- **Checkpoint license** — if unknown, mark as "Unknown" with a note
- Dependencies table with license info (or "No new dependencies needed")
- Note that reference comparison is pending (will be added as PR comment)

---

## Step 3 — Reference Comparison & Validation

This step validates the prognostic model wrapper produces correct
output by comparing against the original reference implementation.
Create three scripts in the repo root (do NOT commit them):

1. **Vanilla reference script** — runs inference using only third-party
   packages (no Earth2Studio)
2. **Earth2Studio reference script** — runs the same inference using
   the E2S wrapper
3. **Comparison script** — loads outputs from both and compares

### 3a. Create vanilla reference inference script

Create `reference_<model_name>_vanilla.py` in the repo root:

```python
"""Vanilla reference inference for <ModelName> (no Earth2Studio).

Runs inference using only third-party packages to establish
ground truth outputs for comparison.

This script is for validation only — do NOT commit to the repo.
"""

import numpy as np
import torch

# ============================================================
# PART 1: Load the reference model (adapt from original repo)
# ============================================================
# TODO: Fill in the model loading code from the reference repo
# Example:
# from <package> import <Model>
# model = <Model>.from_pretrained("<checkpoint>")
# model = model.to("cuda").eval()

raise NotImplementedError(
    "Fill in the reference model loading code above, then remove this line."
)

# ============================================================
# PART 2: Prepare input data
# ============================================================
# Use a fixed random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# TODO: Define input shape matching reference model expectations
# input_shape = (batch, time, vars, lat, lon)  # or whatever the model expects
# x = torch.randn(input_shape, device="cuda", dtype=torch.float32)

# ============================================================
# PART 3: Run inference
# ============================================================
N_STEPS = 8  # Adjust based on model time step

with torch.no_grad():
    # Single-step output
    # y_single = model(x)

    # Multi-step outputs
    outputs = []
    current = x
    for step in range(N_STEPS):
        current = model(current)
        outputs.append(current.cpu())

# ============================================================
# PART 4: Save outputs
# ============================================================
torch.save({
    "input": x.cpu(),
    "single_step": y_single.cpu(),
    "multi_step": outputs,
    "n_steps": N_STEPS,
}, "ref_<model_name>_vanilla_outputs.pt")

print(f"Saved vanilla reference outputs ({N_STEPS} steps)")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {y_single.shape}")
```

### 3b. Create Earth2Studio reference inference script

Create `reference_<model_name>_e2s.py` in the repo root:

```python
"""Earth2Studio reference inference for <ModelName>.

Runs inference using the E2S wrapper for comparison against
vanilla reference.

This script is for validation only — do NOT commit to the repo.
"""

import numpy as np
import torch

from earth2studio.models.px import <ModelName>

# ============================================================
# PART 1: Load the E2S model
# ============================================================
model = <ModelName>.load_model(<ModelName>.load_default_package())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ============================================================
# PART 2: Prepare input data (same seed as vanilla)
# ============================================================
torch.manual_seed(42)
np.random.seed(42)

input_coords = model.input_coords()
# Build input tensor matching E2S format: (batch, time, lead_time, variable, lat, lon)
shape = tuple(max(len(v), 1) for v in input_coords.values())
x = torch.randn(shape, device=device, dtype=torch.float32)

# ============================================================
# PART 3: Run inference
# ============================================================
N_STEPS = 8  # Must match vanilla script

with torch.no_grad():
    # Single-step output
    y_single, out_coords = model(x, input_coords)

    # Multi-step outputs via create_iterator
    iterator = model.create_iterator(x, input_coords)

    # Skip initial condition (step 0)
    step0_x, step0_coords = next(iterator)

    outputs = []
    coords_list = []
    for step in range(N_STEPS):
        y_step, c_step = next(iterator)
        outputs.append(y_step.cpu())
        coords_list.append({k: np.array(v) for k, v in c_step.items()})

# ============================================================
# PART 4: Save outputs
# ============================================================
torch.save({
    "input": x.cpu(),
    "input_coords": {k: np.array(v) for k, v in input_coords.items()},
    "single_step": y_single.cpu(),
    "multi_step": outputs,
    "multi_step_coords": coords_list,
    "n_steps": N_STEPS,
}, "ref_<model_name>_e2s_outputs.pt")

print(f"Saved E2S reference outputs ({N_STEPS} steps)")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {y_single.shape}")
```

### 3c. Create comparison script

Create `reference_<model_name>_compare.py` in the repo root:

```python
"""Compare vanilla vs E2S outputs for <ModelName>.

Loads outputs from both reference scripts and compares them
numerically.

This script is for validation only — do NOT commit to the repo.
"""

import torch
import numpy as np

# ============================================================
# Load outputs
# ============================================================
vanilla = torch.load("ref_<model_name>_vanilla_outputs.pt", weights_only=False)
e2s = torch.load("ref_<model_name>_e2s_outputs.pt", weights_only=False)

assert vanilla["n_steps"] == e2s["n_steps"], "Step count mismatch!"
N_STEPS = vanilla["n_steps"]

# ============================================================
# Helper functions
# ============================================================
def compare_tensors(ref: torch.Tensor, test: torch.Tensor, name: str) -> dict:
    """Compare two tensors and return metrics."""
    # Ensure same shape (may need reshaping if formats differ)
    ref = ref.float()
    test = test.float()

    diff = (ref - test).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    # Relative error (avoid division by zero)
    rel_diff = diff / (ref.abs() + 1e-8)
    max_rel = rel_diff.max().item()

    # Correlation
    corr = torch.corrcoef(torch.stack([ref.flatten(), test.flatten()]))[0, 1].item()

    return {
        "name": name,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "correlation": corr,
    }

def print_metrics(m: dict):
    """Pretty-print comparison metrics."""
    print(f"  {m['name']}:")
    print(f"    Max absolute diff: {m['max_abs']:.2e}")
    print(f"    Mean absolute diff: {m['mean_abs']:.2e}")
    print(f"    Max relative diff: {m['max_rel']:.2e}")
    print(f"    Correlation: {m['correlation']:.8f}")

# ============================================================
# Compare single-step
# ============================================================
print("=" * 60)
print("SINGLE-STEP COMPARISON")
print("=" * 60)

# NOTE: You may need to reshape/reorder tensors if vanilla and E2S
# use different layouts. Adapt the indexing below as needed.
single_metrics = compare_tensors(
    vanilla["single_step"],
    e2s["single_step"],
    "Single step"
)
print_metrics(single_metrics)

# ============================================================
# Compare multi-step
# ============================================================
print("\n" + "=" * 60)
print(f"MULTI-STEP COMPARISON ({N_STEPS} steps)")
print("=" * 60)

multi_metrics = []
for i in range(N_STEPS):
    m = compare_tensors(
        vanilla["multi_step"][i],
        e2s["multi_step"][i],
        f"Step {i + 1}"
    )
    multi_metrics.append(m)
    print_metrics(m)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

all_pass = True
TOLERANCE_ABS = 1e-4  # Adjust based on model precision
TOLERANCE_CORR = 0.9999

if single_metrics["max_abs"] > TOLERANCE_ABS:
    print(f"WARNING: Single-step max_abs {single_metrics['max_abs']:.2e} > {TOLERANCE_ABS}")
    all_pass = False

for m in multi_metrics:
    if m["correlation"] < TOLERANCE_CORR:
        print(f"WARNING: {m['name']} correlation {m['correlation']:.6f} < {TOLERANCE_CORR}")
        all_pass = False

if all_pass:
    print("PASS: All comparisons within tolerance")
else:
    print("FAIL: Some comparisons outside tolerance — review above warnings")
```

### 3d. Run scripts and verify

Execute the scripts in order:

```bash
uv run python reference_<model_name>_vanilla.py
uv run python reference_<model_name>_e2s.py
uv run python reference_<model_name>_compare.py
```

### **[CONFIRM — Reference Comparison]**

Present the comparison results to the user:

> The reference comparison scripts have been run. Results:
>
> **Single-step comparison:**
>
> - Max absolute difference: `<value>`
> - Correlation: `<value>`
>
> **Multi-step comparison:**
>
> - Step 1: max_abs=`<value>`, corr=`<value>`
> - Step N: max_abs=`<value>`, corr=`<value>`
>
> **Scripts created (in repo root, NOT committed):**
>
> - `reference_<model_name>_vanilla.py`
> - `reference_<model_name>_e2s.py`
> - `reference_<model_name>_compare.py`
>
> Please review the metrics and scripts. Do the results look correct?
> If yes, I'll post these as a PR comment with placeholders for you
> to upload reference plots.

If the user reports problems, debug and fix the scripts, re-run,
and ask again.

### 3e. Post reference comparison comment on PR

After user confirms the comparison looks good, post a comment on the
PR with the reference scripts and placeholders for plots.

Create the comment body:

<!-- markdownlint-disable MD033 -->

```markdown
## Reference Comparison Validation

**Model:** `<ClassName>`
**Comparison date:** YYYY-MM-DD
**Environment:** GPU_MODEL / CUDA version

### Results Summary

| Metric | Single-step | Step 4 | Step 8 |
|--------|-------------|--------|--------|
| Max abs diff | X.XXe-XX | X.XXe-XX | X.XXe-XX |
| Correlation | 0.XXXXXX | 0.XXXXXX | 0.XXXXXX |

### Reference Plots

> **TODO:** Upload reference comparison plots by editing this comment.
>
> Suggested plots:
> 1. Side-by-side spatial maps (vanilla vs E2S) at step 1 and step 8
> 2. Difference maps showing spatial error pattern
> 3. Time series of error growth (max_abs vs step)

<!-- Drag and drop images here -->

### Vanilla Reference Script

<details>
<summary>Click to expand vanilla script</summary>

```python
PASTE_VANILLA_SCRIPT_HERE
```

</details>

### Earth2Studio Reference Script

<details>
<summary>Click to expand E2S script</summary>

```python
PASTE_E2S_SCRIPT_HERE
```

</details>

### Comparison Script

<details>
<summary>Click to expand comparison script</summary>

```python
PASTE_COMPARISON_SCRIPT_HERE
```

</details>
```

<!-- markdownlint-enable MD033 -->

Post the comment:

```bash
gh api -X POST repos/NVIDIA/earth2studio/issues/<PR_NUMBER>/comments \
  -F "body=@/tmp/pr_reference_comment.md" \
  --jq '.html_url'
```

Inform the user:

> Reference comparison comment posted to PR.
>
> **Next steps for you:**
>
> 1. Edit the PR comment in the browser
> 2. Upload/drag reference plots into the placeholder section
> 3. Verify the scripts are complete and runnable
>
> Comment URL: `<URL>`

---

## Step 4 — Automated Code Review (Greptile)

After the PR is created and pushed, an automated code review from
**greptile-apps** (Greptile) will be posted as PR review comments.
Wait for this review, then process the feedback.

### 4a. Wait for Greptile review

Poll for review comments from `greptile-apps[bot]` every 30 seconds
for up to **5 minutes**. Time out gracefully if no review arrives:

```bash
# Poll loop — check every 30s, timeout after 5 minutes (10 attempts)
for i in $(seq 1 10); do
  REVIEW_ID=$(gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
    --jq '.[] | select(.user.login == "greptile-apps[bot]") | .id' 2>/dev/null)
  if [ -n "$REVIEW_ID" ]; then
    echo "Greptile review found: $REVIEW_ID"
    break
  fi
  echo "Attempt $i/10 — no review yet, waiting 30s..."
  sleep 30
done
```

If no review after 5 minutes, inform the user:

> Greptile hasn't posted a review after 5 minutes. This can happen if
> the review bot is busy or the PR hasn't triggered it. You can:
>
> 1. Ask me to check again later
> 2. Skip this step and proceed without automated review
> 3. Manually request a review from Greptile on the PR page

### 4b. Pull and parse review comments

Once the review is posted, fetch all comments:

```bash
# Get all review comments on the PR
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") |
    {path: .path, line: .diff_hunk, body: .body}'
```

Also fetch the top-level review body:

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") | .body'
```

### 4c. Categorize and present to user

Parse each comment and categorize it:

| Category | Description | Default action |
| --- | --- | --- |
| **Bug / correctness** | Logic errors, wrong behavior | Fix |
| **Style / convention** | Naming, formatting, patterns | Fix if valid |
| **Performance** | Inefficiency, resource waste | Evaluate |
| **Documentation** | Missing/wrong docs, docstrings | Fix |
| **Suggestion** | Alternative approach, nice-to-have | User decides |
| **False positive** | Incorrect or irrelevant feedback | Dismiss |

### **[CONFIRM — Review Triage]**

Present each comment to the user in a summary table:

```markdown
| # | File | Line | Category | Summary | Proposed Action |
|---|------|------|----------|---------|-----------------|
| 1 | <model>.py | 142 | Bug | Missing null check | Fix: add guard |
| 2 | <model>.py | 305 | Style | Use f-string | Fix: convert |
| 3 | <model>.py | 45 | Suggestion | Add type alias | Skip: not needed |
| ... | ... | ... | ... | ... | ... |
```

For each comment, briefly explain:

- What Greptile flagged
- Whether you agree or disagree (with reasoning)
- Your proposed fix (or why to skip)

Ask the user to confirm which comments to address. The user may:

- Accept all proposed fixes
- Select specific fixes
- Override your recommendation on any comment
- Add their own fixes

### 4d. Implement fixes

For each accepted fix:

1. Make the code change
2. Run `make format && make lint` after all fixes
3. Run the relevant tests:

   ```bash
   uv run python -m pytest test/models/px/test_<filename>.py -v --timeout=60
   ```

4. Commit with a message like:

   ```bash
   git commit -m "fix: address code review feedback (Greptile)"
   ```

### 4e. Respond to review comments

For each Greptile comment, post a reply on the PR:

**For fixed comments:**

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Fixed in <commit_sha>. <brief description of fix>"
```

**For dismissed comments (false positives / won't fix):**

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Won't fix — <brief justification>"
```

### 4f. Push and resolve

```bash
git push <fork-remote> feat/prognostic-model-<name>
```

After pushing, resolve all addressed review threads if possible.

Inform the user of the final state:

- How many comments were fixed
- How many were dismissed (with reasons)
- Any remaining open threads

---

## Reminders

- **DO** use the repo's local `uv` `.venv` to run Python with
  `uv run python`
- **DO NOT** commit reference scripts, comparison scripts, or images to
  the repo
- **DO** use `loguru.logger` for logging, never `print()`, inside
  `earth2studio/`
- **DO** ensure all public functions have full type hints (mypy-clean)
- **DO** maintain alphabetical order in `__init__.py` exports,
  RST file entries, and CHANGELOG entries
- **DO** follow the canonical PX method ordering:
  `__init__`, `input_coords`, `output_coords`,
  `load_default_package`, `load_model`, `to`, private methods,
  `__call__`, `_default_generator`, `create_iterator`
- **DO** use `handshake_dim` indices matching each dimension's
  position in the `CoordSystem` OrderedDict — check existing px
  models for the predominant convention
- **DO** include reference URLs in class docstrings
- **DO** always update CHANGELOG.md under the current unreleased
  version
- **DO** add the model to `docs/modules/models_px.rst` in the
  `earth2studio.models.px` section (alphabetical order)
- **DO** inherit from `torch.nn.Module + AutoModelMixin + PrognosticMixin`
  (triple inheritance)
- **DO** yield initial condition first in `create_iterator`
  (step 0 = unchanged input)
- **DO** use `self.front_hook()` and `self.rear_hook()` in
  `create_iterator` for perturbation injection / post-processing
- **DO** include `lead_time` in `input_coords` starting at
  `np.timedelta64(0, "h")` and increment it in `output_coords`
- **DO NOT** make a general base class with intent to reuse the
  wrapper across models
- **DO NOT** over-populate the `load_model()` API — only expose
  essential parameters
- **NEVER** commit, hardcode, or include API keys, secrets, tokens,
  or credentials in source code, sample scripts, commit messages,
  PR descriptions, or any file tracked by git
