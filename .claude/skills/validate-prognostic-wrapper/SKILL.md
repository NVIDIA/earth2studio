---
name: validate-prognostic-wrapper
description: Validate a newly created Earth2Studio prognostic model wrapper by running tests, performing reference comparison (single-step and multi-step), generating sanity-check plots, and opening a PR with automated code review. Use this skill after completing prognostic model implementation (create-prognostic-wrapper skill Steps 0-12).
argument-hint: Name of the prognostic model class and test file (optional — will be inferred from recent changes if not provided)
---

# Validate Prognostic Model Wrapper

Validate a newly created Earth2Studio prognostic model (px) wrapper by
running tests, performing reference comparison (single-step and
multi-step), generating sanity-check plots, and opening a PR with
automated code review. This skill picks up after the
`create-prognostic-wrapper` skill completes implementation
(Steps 0-12).

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

## Step 2 — Reference Comparison & Sanity-Check

This step validates the prognostic model wrapper produces correct
output by comparing against the original reference implementation
for both a single time step and a multi-step forecast, then
generating visual sanity-check plots.

### 2a. Create reference comparison script

Create a **standalone Python script** in the repo root. This is for
validation only and should **NOT** be committed to the repo.

The script loads the reference model and the E2S wrapper side by side,
runs both on identical input (same random seed or real data), and
compares outputs with tolerance. For prognostic models, test **both**
single-step (`__call__`) and multi-step (`create_iterator`):

```python
"""Reference comparison for <ModelName> prognostic model.

Compares the Earth2Studio wrapper output against the original reference
implementation to verify numerical agreement for both single-step and
multi-step forecasts.

This script is for validation only — do NOT commit to the repo.
"""
import torch
import numpy as np

# --- Reference model ---
# TODO: Load original model per reference repo instructions
# Uncomment and adapt the following lines:
# ref_model = ...
# ref_input = ...
# ref_output_single = ref_model(ref_input)  # single step
# ref_outputs_multi = [ref_output_single]
# current = ref_output_single
# for step in range(N_STEPS):
#     current = ref_model(current)
#     ref_outputs_multi.append(current)
raise NotImplementedError(
    "Fill in the reference model code above, then remove this line."
)

# --- Earth2Studio wrapper ---
from earth2studio.models.px import ModelName

model = ModelName(...)  # or ModelName.load_model(package)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

input_coords = model.input_coords()
# Construct input tensor matching the reference input
# Use the same random seed or identical real data for both
shape = tuple(max(len(v), 1) for v in input_coords.values())
torch.manual_seed(42)
x = torch.randn(shape, device=device)

# --- Single-step comparison ---
e2s_output_single, out_coords = model(x, input_coords)

ref_output_single = ref_output_single.to(e2s_output_single.device)
max_abs_single = (ref_output_single - e2s_output_single).abs().max().item()
max_rel_single = (
    (ref_output_single - e2s_output_single).abs()
    / (ref_output_single.abs() + 1e-8)
).max().item()
corr_single = torch.corrcoef(
    torch.stack([
        ref_output_single.flatten(),
        e2s_output_single.flatten(),
    ])
)[0, 1].item()

print("=== Single-step comparison ===")
print(f"Max absolute difference: {max_abs_single:.2e}")
print(f"Max relative difference: {max_rel_single:.2e}")
print(f"Correlation: {corr_single:.8f}")

assert torch.allclose(
    ref_output_single, e2s_output_single, rtol=1e-4, atol=1e-5
), f"Single-step mismatch! Max abs diff: {max_abs_single:.2e}"

# --- Multi-step comparison ---
N_STEPS = 5  # Adapt to model time step (e.g., 5 steps of 6h = 30h)
iterator = model.create_iterator(x, input_coords)

# Skip initial condition (step 0)
step0_x, step0_coords = next(iterator)

print(f"\n=== Multi-step comparison ({N_STEPS} steps) ===")
for step_i in range(N_STEPS):
    e2s_step, e2s_coords = next(iterator)
    ref_step = ref_outputs_multi[step_i + 1].to(e2s_step.device)

    max_abs = (ref_step - e2s_step).abs().max().item()
    corr = torch.corrcoef(
        torch.stack([ref_step.flatten(), e2s_step.flatten()])
    )[0, 1].item()
    lead = e2s_coords["lead_time"]

    print(f"Step {step_i + 1} (lead_time={lead}): "
          f"max_abs={max_abs:.2e}, corr={corr:.8f}")

    assert torch.allclose(ref_step, e2s_step, rtol=1e-3, atol=1e-4), \
        f"Multi-step mismatch at step {step_i + 1}! Max abs: {max_abs:.2e}"

print("\nPASS: Reference comparison successful (single + multi-step).")
```

### 2b. Summarize model capabilities to user

Before generating sanity-check plots, **present a summary table** to
the user covering the model's capabilities:

> **Model Summary for `<ClassName>`:**
>
> | Property | Value |
> |---|---|
> | **Input variables** | `var1`, `var2`, ... (N total) |
> | **Output variables** | `out1`, `out2`, ... (M total) |
> | **Time step** | Xh (e.g., 6h, 24h) |
> | **Spatial resolution** | X.XX deg x Y.YY deg (NxM) |
> | **History required** | None / Xh (e.g., -6h, 0h) |
> | **Checkpoint size** | XX MB |
> | **Checkpoint source** | NGC / HuggingFace / S3 |
> | **Inference time** | ~XX ms per step (on GPU/CPU) |

This summary helps the user verify the wrapper matches their
expectations for the model.

### 2c. Generate sanity-check plot script

Create a **standalone Python script** in the repo root. This is for
PR reviewer reference only and should **NOT** be committed to the
repo.

Choose the appropriate template based on the model's output type:

#### Spatial forecast evolution (most common — gridded weather models)

```python
"""Sanity-check plot for <ModelName> prognostic model.

This script is for PR review only — do NOT commit to the repo.
Runs a multi-step forecast and visualizes the evolution of key
variables over lead time.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import ModelName

# Load model
model = ModelName.from_pretrained()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Prepare input
time = np.array([np.datetime64("2024-01-01T00:00")])
input_coords = model.input_coords()
input_coords["time"] = time
ds = Random(input_coords)
x, coords = fetch_data(ds, time, input_coords["variable"], device=device)

# Run multi-step forecast
N_STEPS = 5
iterator = model.create_iterator(x, coords)

# Collect outputs
steps = []
for i, (step_x, step_coords) in enumerate(iterator):
    steps.append((step_x.cpu().numpy(), dict(step_coords)))
    if i >= N_STEPS:
        break

# Pick 2-3 representative variables
var_list = list(steps[0][1]["variable"])
plot_vars = var_list[:3]  # First 3 variables, or pick specific ones
n_vars = len(plot_vars)

# Plot: rows = variables, columns = time steps (0, mid, final)
step_indices = [0, N_STEPS // 2, N_STEPS]
n_cols = len(step_indices)
fig, axes = plt.subplots(n_vars, n_cols, figsize=(5 * n_cols, 4 * n_vars))
if n_vars == 1:
    axes = axes[np.newaxis, :]
if n_cols == 1:
    axes = axes[:, np.newaxis]

for row, var in enumerate(plot_vars):
    var_idx = var_list.index(var)
    for col, si in enumerate(step_indices):
        data, sc = steps[si]
        # Shape: (batch, time, lead_time, variable, lat, lon)
        data_2d = data[0, 0, 0, var_idx, :, :]
        lead = sc["lead_time"]
        im = axes[row, col].contourf(data_2d, cmap="turbo", levels=20)
        axes[row, col].set_title(f"{var} | lead={lead}")
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)

plt.suptitle(f"<ModelName> forecast evolution", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_<model_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<model_name>.png")
```

#### Time-series at selected grid points

For models where spatial patterns are less meaningful (e.g., random
input), or to supplement the spatial plot, show how variables evolve
over lead time at specific grid points:

```python
"""Sanity-check time series for <ModelName> prognostic model.

Shows how selected variables evolve over forecast lead time at
specific grid points.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import ModelName

# Load model and run forecast
model = ModelName.from_pretrained()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

time = np.array([np.datetime64("2024-01-01T00:00")])
input_coords = model.input_coords()
input_coords["time"] = time
ds = Random(input_coords)
x, coords = fetch_data(ds, time, input_coords["variable"], device=device)

N_STEPS = 10
iterator = model.create_iterator(x, coords)

# Collect time series at a central grid point
lat_idx = len(input_coords["lat"]) // 2
lon_idx = len(input_coords["lon"]) // 2
var_list = list(input_coords["variable"])
plot_vars = var_list[:4]  # Pick representative variables

lead_times = []
values = {v: [] for v in plot_vars}

for i, (step_x, step_coords) in enumerate(iterator):
    lead_times.append(step_coords["lead_time"][0])
    for var in plot_vars:
        var_idx = var_list.index(var)
        val = step_x[0, 0, 0, var_idx, lat_idx, lon_idx].cpu().item()
        values[var].append(val)
    if i >= N_STEPS:
        break

# Convert lead times to hours for plotting
lead_hours = [lt / np.timedelta64(1, "h") for lt in lead_times]

fig, axes = plt.subplots(len(plot_vars), 1,
                         figsize=(10, 3 * len(plot_vars)),
                         sharex=True)
if len(plot_vars) == 1:
    axes = [axes]

for ax, var in zip(axes, plot_vars):
    ax.plot(lead_hours, values[var], marker="o", markersize=3)
    ax.set_ylabel(var)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{var} at lat_idx={lat_idx}, lon_idx={lon_idx}")

axes[-1].set_xlabel("Lead time (hours)")
plt.suptitle(f"<ModelName> — time series at grid center", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_<model_name>_timeseries.png",
            dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<model_name>_timeseries.png")
```

#### Iterator behavior validation

For verifying `create_iterator` mechanics (initial condition yield,
lead_time progression, hook application):

```python
"""Iterator validation for <ModelName> prognostic model.

Verifies create_iterator yields correct initial condition, increments
lead_time correctly, and that front/rear hooks are called.
"""
import numpy as np
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import ModelName

model = ModelName.from_pretrained()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

time = np.array([np.datetime64("2024-01-01T00:00")])
input_coords = model.input_coords()
input_coords["time"] = time
ds = Random(input_coords)
x, coords = fetch_data(ds, time, input_coords["variable"], device=device)

N_STEPS = 5
iterator = model.create_iterator(x, coords)

print("=== Iterator validation ===")
for i, (step_x, step_coords) in enumerate(iterator):
    lead = step_coords["lead_time"]
    print(f"Step {i}: shape={step_x.shape}, lead_time={lead}, "
          f"device={step_x.device}")

    if i == 0:
        # Step 0 must be the initial condition (unchanged input)
        assert torch.allclose(step_x, x), \
            "Step 0 should yield the initial condition unchanged"
        assert np.array_equal(lead, coords["lead_time"]), \
            "Step 0 lead_time should match input lead_time"
        print("  -> Initial condition verified")

    if i >= N_STEPS:
        break

# Verify lead_time progression
expected_step = input_coords["lead_time"][0]
print(f"\nExpected time step: {expected_step}")
print(f"Final lead_time after {N_STEPS} steps: {step_coords['lead_time']}")
print("PASS: Iterator validation successful.")
```

### 2d. Run comparison and sanity-check scripts

Execute all scripts:

```bash
uv run python reference_comparison_<model_name>.py
uv run python sanity_check_<model_name>.py
```

Verify that:

- The reference comparison passes for both single-step and multi-step
- The sanity-check script runs without errors
- Output PNGs are generated
- `create_iterator` yields correct initial condition at step 0
- `lead_time` increments correctly at each step

### 2e. **[CONFIRM — Sanity-Check & Comparison]**

**You MUST ask the user to visually inspect the generated plot(s)
before proceeding.** Do not skip this step even if the scripts ran
without errors — a successful run does not guarantee the plots are
correct (e.g., empty axes, wrong colorbar range, garbled data).

Tell the user the absolute path to the generated image file(s) and
the reference comparison metrics, then ask them to inspect:

> The reference comparison and sanity-check scripts ran successfully.
>
> **Single-step reference comparison:**
>
> - Max absolute difference: `<value>`
> - Max relative difference: `<value>`
> - Correlation: `<value>`
>
> **Multi-step reference comparison (N steps):**
>
> - Step 1: max_abs=`<value>`, corr=`<value>`
> - Step N: max_abs=`<value>`, corr=`<value>`
> - (error may grow with lead time — this is expected)
>
> **Sanity-check plot saved to:**
> `/absolute/path/to/sanity_check_<model_name>.png`
>
> **Please open this image and confirm it looks correct.** Check:
>
> 1. Data is visible on the axes at all time steps (not blank/empty)
> 2. Values are in physically reasonable ranges
> 3. Spatial patterns evolve smoothly over lead time
> 4. No sudden jumps, NaN explosions, or frozen fields
> 5. Lead time labels increment correctly
>
> Does the plot look correct and do the reference comparison metrics
> look acceptable?

**Do not proceed to Step 3 until the user explicitly confirms.** If
the user reports problems, debug and fix the issue, re-run the
scripts, and ask the user to inspect again.

---

## Step 3 — Branch, Commit & Open PR

### **[CONFIRM — Ready to Submit]**

Before proceeding, confirm with the user:

> All implementation and validation steps are complete:
>
> - Prognostic model class implemented with correct method ordering
> - Triple inheritance: `torch.nn.Module + AutoModelMixin + PrognosticMixin`
> - Coordinate system with proper `handshake_dim` indices
> - `__call__` (single step) and `create_iterator` (multi-step) implemented
> - `create_iterator` yields initial condition first, uses front/rear hooks
> - Model loading implemented (`load_default_package`, `load_model`)
> - Registered in `earth2studio/models/px/__init__.py`
> - Documentation added to `docs/modules/models.rst`
> - Reference URLs included in class docstrings
> - CHANGELOG.md updated
> - Format, lint, and license checks pass
> - Unit tests written and passing with >= 90% coverage
> - Dependencies in pyproject.toml confirmed
> - Reference comparison passes (single-step and multi-step)
> - Sanity-check plots generated and confirmed by user
>
> Ready to create a branch, commit, and prepare a PR?

### 3a. Create branch and commit

```bash
git checkout -b feat/prognostic-model-<name>
git add earth2studio/models/px/<filename>.py \
        earth2studio/models/px/__init__.py \
        test/models/px/test_<filename>.py \
        pyproject.toml \
        CHANGELOG.md \
        docs/modules/models.rst
git commit -m "feat: add <ClassName> prognostic model

Add <ClassName> prognostic model for <brief description>.
Includes unit tests and documentation."
```

Do **NOT** add the sanity-check scripts, comparison scripts, or
their output images.

### 3b. Identify the fork remote and push branch

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

### 3c. Open Pull Request (fork -> NVIDIA/earth2studio)

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

The PR body should follow this prognostic-model-specific template:

````markdown
## Description

Add `<ClassName>` prognostic model for <brief description>.

Closes #<issue_number> (if applicable)

### Model details

| Property | Value |
|---|---|
| **Architecture** | PyTorch / ONNX / JAX |
| **Time step** | Xh (e.g., 6h, 24h) |
| **Input variables** | N variables (list or link) |
| **Output variables** | M variables (list or link) |
| **Spatial resolution** | X° x Y° (NxM grid) |
| **History required** | None / Xh (e.g., [-6h, 0h]) |
| **Checkpoint source** | NGC / HuggingFace / S3 |
| **Checkpoint size** | XX MB |
| **Reference** | <paper/repo URL> |

### Dependencies added

| Package | Version | License | License URL | Reason |
|---|---|---|---|---|
| `<package>` | `>=X.Y` | <License> | [link](<URL>) | <reason> |

*(or "No new dependencies needed")*

When filling this table, look up each new dependency's license:

1. Check the package's PyPI page or repository for the license type
2. Link directly to the license file
3. Flag any **non-permissive licenses** (GPL, AGPL, SSPL) — these
   may be incompatible with the project's Apache-2.0 license

### Reference comparison

**Single step:**

- Max absolute difference: <value>
- Max relative difference: <value>
- Correlation: <value>

**Multi-step (N steps):**

- Step 1: max_abs=<value>, corr=<value>
- Step N: max_abs=<value>, corr=<value>

### Validation

See sanity-check plots in PR comments below.

## Checklist

- [x] I am familiar with the [Contributing Guidelines][contrib].
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [x] The [CHANGELOG.md][changelog] is up to date with these changes.
- [ ] An [issue][issues] is linked to this pull request.
- [ ] Assess and address Greptile feedback (AI code review bot).

[contrib]: https://github.com/NVIDIA/earth2studio/blob/main/CONTRIBUTING.md
[changelog]: https://github.com/NVIDIA/earth2studio/blob/main/CHANGELOG.md
[issues]: https://github.com/NVIDIA/earth2studio/issues
````

### 3d. Post sanity-check as PR comment

After the PR is created, post the sanity-check visualization as a
separate **PR comment** so it is immediately visible to reviewers.

#### Image upload limitation

**GitHub has no CLI or REST API for uploading images to PR comments.**
The only way to embed an image is via the browser's drag-and-drop
editor or by referencing an already-hosted URL.

**Practical workflow:**

1. Write the comment body to a temp file (avoids shell quoting issues
   with heredocs containing backticks and markdown).
2. Post the comment **without** the image — include the validation
   table, reference comparison metrics, the full sanity-check script,
   and a placeholder line.
3. Tell the user to drag the image into the browser editor.

```bash
# 1. Write body to a temp file (use your editor tool, not heredoc)

# 2. Post the comment
gh api -X POST repos/NVIDIA/earth2studio/issues/<PR_NUMBER>/comments \
  -F "body=@/tmp/pr_comment_body.md" \
  --jq '.html_url'
```

Do **not** waste time trying `curl` uploads, GraphQL file mutations,
or the `uploads.github.com` asset endpoint — they do not work for
issue/PR comment images.

#### Comment content template

```markdown
## Sanity-Check Validation

**Model:** `<ClassName>` — <brief description>
**Architecture:** PyTorch / ONNX / JAX
**Time step:** Xh
**Test environment:** <GPU model or CPU>

### Reference Comparison

**Single step:**

| Metric | Value |
|--------|-------|
| Max absolute difference | <value> |
| Max relative difference | <value> |
| Correlation | <value> |

**Multi-step (N steps):**

| Step | Lead time | Max abs diff | Correlation |
|------|-----------|-------------|-------------|
| 1 | Xh | <value> | <value> |
| ... | ... | ... | ... |
| N | NXh | <value> | <value> |

### Model Summary

| Property | Value |
|----------|-------|
| Input variables | <list or count> |
| Output variables | <list or count> |
| Spatial resolution | X° x Y° (NxM) |
| Time step | Xh |
| History required | None / [-Xh, 0h] |
| Inference time | ~XX ms per step |

**Key findings:**

- <bullet summarizing single-step numerical agreement>
- <bullet on multi-step error growth behavior>
- <bullet on spatial pattern quality over forecast horizon>

> **TODO:** Attach sanity-check image by editing this comment in
> the browser.

<details>
<summary>Sanity-check script (click to expand)</summary>

```python
PASTE THE FULL WORKING SCRIPT HERE — not a truncated excerpt.
The script must be copy-pasteable and produce the plot end-to-end.
```

</details>
```

**Important:** Always paste the **complete, runnable** script — not
a shortened version. Reviewers should be able to reproduce the plot
by copying the script directly.

#### Finalize

After posting, inform the user of:

1. The comment URL
2. The local path to the image file for manual attachment
3. Instructions: *"Edit the comment in your browser and drag the
   image file into the editor to embed it."*

> **Note:** The sanity-check image and script are for PR review
> purposes only — they must NOT be committed to the repository.

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
|---|---|---|
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
- **DO NOT** commit sanity-check/comparison scripts or images to
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
- **DO** add the model to `docs/modules/models.rst` in the
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
