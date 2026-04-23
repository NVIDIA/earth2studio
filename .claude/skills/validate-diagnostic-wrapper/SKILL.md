---
name: validate-diagnostic-wrapper
description: Validate a newly created Earth2Studio diagnostic model wrapper by running tests, performing reference comparison, generating sanity-check outputs, and opening a PR with automated code review. Use this skill after completing diagnostic model implementation (create-diagnostic-wrapper skill Steps 0-12).
argument-hint: Name of the diagnostic model class and test file (optional — will be inferred from recent changes if not provided)
---

# Validate Diagnostic Model Wrapper

Validate a newly created Earth2Studio diagnostic model (dx) wrapper by
running tests, performing reference comparison, generating sanity-check
outputs, and opening a PR with automated code review. This skill picks
up after the `create-diagnostic-wrapper` skill completes implementation
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
uv run python -m pytest test/models/dx/test_<filename>.py -v --timeout=60
```

All tests must pass. Fix failures and re-run until green.

### 1b. Run coverage report with `--slow` tests

Run the new test file **with coverage** and the `--slow` flag to
include integration tests. The new diagnostic model file must achieve
**at least 90% line coverage**:

```bash
uv run python -m pytest test/models/dx/test_<filename>.py -v \
    --slow --timeout=300 \
    --cov=earth2studio/models/dx/<filename> \
    --cov-report=term-missing \
    --cov-fail-under=90
```

- `--slow` enables integration tests (marked `@pytest.mark.slow`)
- `--cov=earth2studio/models/dx/<filename>` scopes coverage to the
  new model module only
- `--cov-report=term-missing` shows which lines are not covered
- `--cov-fail-under=90` fails the run if coverage is below 90%

If coverage is below 90%, add additional tests or mock tests to
cover the missing lines. Common coverage gaps for dx models:

- Error handling in `output_coords` (wrong variable names, wrong dims)
- Device management paths (CPU vs CUDA)
- Edge cases in physics calculations (zero inputs, extreme values)
- `load_model` and `load_default_package` (NN-based models, needs mock)

Re-run until coverage is at or above 90%.

### 1c. Run the full model test suite (optional but recommended)

```bash
make pytest TOX_ENV=test-models
```

Confirm no regressions in existing model tests.

---

## Step 2 — Reference Comparison & Sanity-Check

This step validates the diagnostic model wrapper produces correct
output by comparing against the original reference implementation and
generating visual sanity-check plots.

### 2a. Create reference comparison script

Create a **standalone Python script** in the repo root. This is for
validation only and should **NOT** be committed to the repo.

The script loads the reference model and the E2S wrapper side by side,
runs both on identical input (same random seed or real data), and
compares outputs with tolerance:

```python
"""Reference comparison for <ModelName> diagnostic model.

Compares the Earth2Studio wrapper output against the original reference
implementation to verify numerical agreement.

This script is for validation only — do NOT commit to the repo.
"""
import torch
import numpy as np

# --- Reference model ---
# TODO: Load original model per reference repo instructions
# Uncomment and adapt the following lines:
# ref_model = ...
# ref_input = ...
# ref_output = ref_model(ref_input)
raise NotImplementedError(
    "Fill in the reference model code above, then remove this line."
)

# --- Earth2Studio wrapper ---
from earth2studio.models.dx import ModelName

model = ModelName(...)  # or ModelName.load_model(package)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

input_coords = model.input_coords()
# Construct input tensor matching the reference input
# Use the same random seed or identical real data for both
shape = tuple(max(len(v), 1) for v in input_coords.values())
torch.manual_seed(42)
x = torch.randn(shape, device=device)

e2s_output, out_coords = model(x, input_coords)

# --- Compare outputs ---
# Ensure both tensors are on the same device and dtype
ref_output = ref_output.to(e2s_output.device)

max_abs_diff = (ref_output - e2s_output).abs().max().item()
max_rel_diff = (
    (ref_output - e2s_output).abs() / (ref_output.abs() + 1e-8)
).max().item()
correlation = torch.corrcoef(
    torch.stack([ref_output.flatten(), e2s_output.flatten()])
)[0, 1].item()

print(f"Max absolute difference: {max_abs_diff:.2e}")
print(f"Max relative difference: {max_rel_diff:.2e}")
print(f"Correlation: {correlation:.8f}")

assert torch.allclose(ref_output, e2s_output, rtol=1e-4, atol=1e-5), \
    f"Output mismatch! Max abs diff: {max_abs_diff:.2e}"

print("PASS: Reference comparison successful.")
```

**For physics-based models**, compare against hand-calculated expected
values instead of a reference model:

```python
# Known test case: e.g., u=3, v=4 -> wind_speed=5
u_input = torch.tensor([3.0])
v_input = torch.tensor([4.0])
expected_ws = torch.tensor([5.0])

# Run through E2S wrapper
# ...
assert torch.allclose(e2s_output, expected_ws, atol=1e-6), \
    f"Physics check failed: expected {expected_ws}, got {e2s_output}"
```

### 2b. Summarize model capabilities to user

Before generating sanity-check plots, **present a summary table** to
the user covering the model's capabilities:

> **Model Summary for `<ClassName>`:**
>
> | Property | Value |
> |---|---|
> | **Input variables** | `var1`, `var2`, ... |
> | **Output variables** | `out1`, `out2`, ... |
> | **Spatial resolution** | X.XX deg x Y.YY deg (NxM) / Flexible |
> | **Checkpoint size** | XX MB / N/A (physics-based) |
> | **Checkpoint source** | NGC / HuggingFace / N/A |
> | **Inference time** | ~XX ms per forward pass (on GPU/CPU) |

This summary helps the user verify the wrapper matches their
expectations for the model.

### 2c. Generate sanity-check plot script

Create a **standalone Python script** in the repo root. This is for
PR reviewer reference only and should **NOT** be committed to the
repo.

Choose the appropriate template based on the model's output type:

#### Spatial gridded outputs (e.g., precipitation, solar radiation)

```python
"""Sanity-check plot for <ModelName> diagnostic model.

This script is for PR review only — do NOT commit to the repo.
Run it to produce a quick visualization confirming the model works.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.models.dx import ModelName

# Load model
model = ModelName(...)  # or ModelName.load_model(package)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create or fetch input data
input_coords = model.input_coords()
shape = tuple(max(len(v), 1) for v in input_coords.values())
x = torch.randn(shape, device=device)

# Run forward pass
output, out_coords = model(x, input_coords)
output_np = output.cpu().numpy()

# Plot contourf for each output variable
variables = list(out_coords["variable"])
n_vars = len(variables)
fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))
if n_vars == 1:
    axes = [axes]

for ax, i_var in zip(axes, range(n_vars)):
    var = variables[i_var]
    data_2d = output_np[0, 0, i_var, :, :]  # batch=0, time=0
    im = ax.contourf(data_2d, cmap="turbo", levels=20)
    ax.set_title(f"{var}")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle(f"<ModelName> diagnostic output", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_<model_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<model_name>.png")
```

#### Physics-based outputs (e.g., derived quantities like wind speed)

```python
"""Sanity-check for <ModelName> physics-based diagnostic.

This script is for PR review only — do NOT commit to the repo.
Validates exact physics results against known test cases.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.models.dx import ModelName

model = ModelName(...)

# Known test cases
# e.g., u=3, v=4 -> wind_speed=5
test_inputs = {
    "u": [0.0, 3.0, -5.0, 10.0],
    "v": [0.0, 4.0, 12.0, 0.0],
}
expected_outputs = {
    "ws": [0.0, 5.0, 13.0, 10.0],
}

# Construct input tensor from test cases and run model
input_coords = model.input_coords()
n_cases = len(test_inputs["u"])
n_vars = len(input_coords["variable"])
# Build tensor: shape (1, n_vars, n_cases, 1) — batch=1, spatial=n_cases x 1
x = torch.zeros(1, n_vars, n_cases, 1)
# Map test inputs to the correct variable channels
# Adapt these indices to match model.input_coords()["variable"]
var_list = list(input_coords["variable"])
x[:, var_list.index("u1000"), :, 0] = torch.tensor(test_inputs["u"])
x[:, var_list.index("v1000"), :, 0] = torch.tensor(test_inputs["v"])

# Update coords to match tensor shape
test_coords = input_coords.copy()
test_coords["lat"] = np.arange(n_cases, dtype=np.float64)
test_coords["lon"] = np.array([0.0])

output, out_coords = model(x, test_coords)

# Validate physics
for key, expected in expected_outputs.items():
    actual = output[..., :, 0, 0].cpu().numpy().flatten()
    expected_arr = np.array(expected)
    print(f"{key}: expected={expected_arr}, actual={actual}")
    assert np.allclose(actual, expected_arr, atol=1e-6), \
        f"Physics validation failed for {key}"

print("PASS: All physics test cases validated.")

# Side-by-side plot: input vs output
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(range(len(test_inputs["u"])), test_inputs["u"], label="u")
axes[0].bar(range(len(test_inputs["v"])), test_inputs["v"],
            alpha=0.7, label="v")
axes[0].set_title("Input components")
axes[0].legend()

axes[1].bar(range(len(expected_outputs["ws"])), expected_outputs["ws"],
            label="Expected", alpha=0.7)
axes[1].bar(range(len(expected_outputs["ws"])),
            output[..., :, 0, 0].cpu().numpy().flatten(),
            label="E2S output", alpha=0.5)
axes[1].set_title("Output: expected vs actual")
axes[1].legend()

plt.suptitle("<ModelName> — physics validation")
plt.tight_layout()
plt.savefig("sanity_check_<model_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<model_name>.png")
```

#### Scalar/classification outputs (e.g., TC tracking, severity index)

```python
"""Sanity-check for <ModelName> scalar/classification diagnostic.

This script is for PR review only — do NOT commit to the repo.
Generates histogram and summary statistics of output values.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.models.dx import ModelName

# Load model and run inference
model = ModelName(...)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

input_coords = model.input_coords()
shape = tuple(max(len(v), 1) for v in input_coords.values())
x = torch.randn(shape, device=device)

output, out_coords = model(x, input_coords)
output_np = output.cpu().numpy().flatten()

# Summary statistics
print(f"Output shape: {output.shape}")
print(f"Min: {output_np.min():.4f}")
print(f"Max: {output_np.max():.4f}")
print(f"Mean: {output_np.mean():.4f}")
print(f"Std: {output_np.std():.4f}")

# Histogram of output values
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(output_np, bins=50, edgecolor="black", alpha=0.7)
ax.set_xlabel("Output value")
ax.set_ylabel("Count")
ax.set_title(f"<ModelName> output distribution (n={len(output_np)})")
ax.axvline(output_np.mean(), color="red", linestyle="--", label=f"Mean: {output_np.mean():.3f}")
ax.legend()

plt.tight_layout()
plt.savefig("sanity_check_<model_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<model_name>.png")
```

### 2d. Run comparison and sanity-check scripts

Execute both scripts:

```bash
uv run python reference_comparison_<model_name>.py
uv run python sanity_check_<model_name>.py
```

Verify that:

- The reference comparison passes (all assertions hold)
- The sanity-check script runs without errors
- Output PNGs are generated
- Metrics are printed (max abs diff, max rel diff, correlation)

### 2e. **[CONFIRM — Sanity-Check & Comparison]**

**You MUST ask the user to visually inspect the generated plot(s)
before proceeding.** Do not skip this step even if the scripts ran
without errors — a successful run does not guarantee the plots are
correct (e.g., empty axes, wrong colorbar range, garbled data).

Tell the user the absolute path to the generated image file(s) and
the reference comparison metrics, then ask them to inspect:

> The reference comparison and sanity-check scripts ran successfully.
>
> **Reference comparison metrics:**
>
> - Max absolute difference: `<value>`
> - Max relative difference: `<value>`
> - Correlation: `<value>`
>
> **Sanity-check plot saved to:**
> `/absolute/path/to/sanity_check_<model_name>.png`
>
> **Please open this image and confirm it looks correct.** Check:
>
> 1. Data is visible on the axes (not blank/empty)
> 2. Values are in physically reasonable ranges
> 3. No obvious artifacts (all-NaN regions, garbled values)
> 4. For spatial outputs: geographic patterns look plausible
> 5. For physics outputs: results match expected analytical values
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
> - Diagnostic model class implemented with correct method ordering
> - Coordinate system with proper `handshake_dim` indices
> - Forward pass implemented (NN-based or physics-based)
> - Model loading implemented (NN-based) or skipped (physics-based)
> - Registered in `earth2studio/models/dx/__init__.py`
> - Documentation added to `docs/modules/models.rst`
> - Reference URLs included in class docstrings
> - CHANGELOG.md updated
> - Format, lint, and license checks pass
> - Unit tests written and passing with >= 90% coverage
> - Dependencies in pyproject.toml confirmed (NN-based)
> - Reference comparison passes with acceptable tolerance
> - Sanity-check plots generated and confirmed by user
>
> Ready to create a branch, commit, and prepare a PR?

### 3a. Create branch and commit

```bash
git checkout -b feat/diagnostic-model-<name>
git add earth2studio/models/dx/<filename>.py \
        earth2studio/models/dx/__init__.py \
        test/models/dx/test_<filename>.py \
        pyproject.toml \
        CHANGELOG.md \
        docs/modules/models.rst
git commit -m "feat: add <ClassName> diagnostic model

Add <ClassName> diagnostic model for <brief description>.
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
git push -u <fork-remote> feat/diagnostic-model-<name>
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
  --head <fork-owner>:feat/diagnostic-model-<name> \
  --title "feat: add <ClassName> diagnostic model" \
  --body "..."
```

Where `<fork-owner>` is the GitHub username that owns the fork.

The PR body should follow this diagnostic-model-specific template:

````markdown
## Description

Add `<ClassName>` diagnostic model for <brief description>.

Closes #<issue_number> (if applicable)

### Model details

| Property | Value |
|---|---|
| **Model type** | NN-based / Physics-based |
| **Architecture** | PyTorch / ONNX / Analytical |
| **Input variables** | <list> |
| **Output variables** | <list> |
| **Spatial resolution** | X° x Y° (NxM) / Flexible |
| **Checkpoint source** | NGC / HuggingFace / N/A |
| **Reference** | <paper/repo URL> |

### Dependencies added

| Package | Version | License | License URL | Reason |
|---|---|---|---|---|
| `<package>` | `>=X.Y` | <License> | [link](<URL>) | <reason> |

*(or "No new dependencies — physics-based model")*

### Reference comparison

- Max absolute difference: <value>
- Max relative difference: <value>
- Correlation: <value>

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
**Type:** NN-based / Physics-based
**Test environment:** <GPU model or CPU>

### Reference Comparison

| Metric | Value |
|--------|-------|
| Max absolute difference | <value> |
| Max relative difference | <value> |
| Correlation | <value> |

### Model Summary

| Property | Value |
|----------|-------|
| Input variables | <list or count> |
| Output variables | <list or count> |
| Output shape | <shape> |
| Spatial resolution | X° x Y° / Flexible |
| Inference time | ~XX ms |

**Key findings:**
- <bullet summarizing numerical agreement with reference>
- <bullet on output quality / physical reasonableness>
- <bullet on performance or notable behavior>

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
   uv run python -m pytest test/models/dx/test_<filename>.py -v --timeout=60
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
git push <fork-remote> feat/diagnostic-model-<name>
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
- **DO** follow the canonical DX method ordering:
  `__init__`, `input_coords`, `output_coords`, `__call__`,
  `_forward`, `to`, `load_default_package`, `load_model`
- **DO** use `handshake_dim` indices matching each dimension's position in the
  `CoordSystem` OrderedDict — check existing dx models for the predominant convention
- **DO** include reference URLs in class docstrings
- **DO** always update CHANGELOG.md under the current unreleased
  version
- **DO** add the model to `docs/modules/models.rst` in the
  `earth2studio.models.dx` section (alphabetical order)
- **DO NOT** inherit from `PrognosticMixin` or add `create_iterator`
  — diagnostic models are single-pass, not time-stepping
- **DO NOT** add `lead_time` dimension unless the model genuinely needs temporal
  context (e.g., solar radiation, wind gust models that depend on forecast lead time)
- **DO NOT** make a general base class with intent to reuse the
  wrapper across models
- **DO NOT** over-populate the `load_model()` API — only expose
  essential parameters
- **NEVER** commit, hardcode, or include API keys, secrets, tokens,
  or credentials in source code, sample scripts, commit messages,
  PR descriptions, or any file tracked by git
