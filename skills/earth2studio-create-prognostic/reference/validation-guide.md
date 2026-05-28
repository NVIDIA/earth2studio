# Validation Guide — Prognostic Model Wrapper

> **Table of Contents**
>
> 1. [Reference Comparison](#reference-comparison)
> 2. [Sanity-Check Plots](#sanity-check-plots)
> 3. [Branch, Commit & Open PR](#branch-commit--open-pr)
> 4. [Automated Code Review](#automated-code-review)

---

## Reference Comparison

Validate the wrapper produces correct output by comparing against the original
reference implementation. Create three scripts in the repo root (do NOT commit).

### Vanilla reference script

Create `reference_<model_name>_vanilla.py`:

```python
"""Vanilla reference inference for <ModelName> (no Earth2Studio).

Runs inference using only third-party packages to establish
ground truth outputs for comparison.

This script is for validation only — do NOT commit to the repo.
"""
import numpy as np
import torch

# Load the reference model (adapt from original repo)
# ref_model = ...

raise NotImplementedError(
    "Fill in the reference model loading code above, then remove this line."
)

# Prepare input data
torch.manual_seed(42)
np.random.seed(42)

# Run inference
N_STEPS = 8

with torch.no_grad():
    # Single-step output
    # y_single = model(x)

    # Multi-step outputs
    outputs = []
    current = x
    for step in range(N_STEPS):
        current = model(current)
        outputs.append(current.cpu())

# Save outputs
torch.save({
    "input": x.cpu(),
    "single_step": y_single.cpu(),
    "multi_step": outputs,
    "n_steps": N_STEPS,
}, "ref_<model_name>_vanilla_outputs.pt")
```

### Earth2Studio reference script

Create `reference_<model_name>_e2s.py`:

```python
"""Earth2Studio reference inference for <ModelName>.

This script is for validation only — do NOT commit to the repo.
"""
import numpy as np
import torch
from earth2studio.models.px import ModelName

# Load E2S model
model = ModelName.load_model(ModelName.load_default_package())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Prepare input (same seed as vanilla)
torch.manual_seed(42)
np.random.seed(42)

input_coords = model.input_coords()
shape = tuple(max(len(v), 1) for v in input_coords.values())
x = torch.randn(shape, device=device, dtype=torch.float32)

# Run inference
N_STEPS = 8

with torch.no_grad():
    y_single, out_coords = model(x, input_coords)

    iterator = model.create_iterator(x, input_coords)
    step0_x, step0_coords = next(iterator)  # Skip initial condition

    outputs = []
    coords_list = []
    for step in range(N_STEPS):
        y_step, c_step = next(iterator)
        outputs.append(y_step.cpu())
        coords_list.append({k: np.array(v) for k, v in c_step.items()})

# Save outputs
torch.save({
    "input": x.cpu(),
    "input_coords": {k: np.array(v) for k, v in input_coords.items()},
    "single_step": y_single.cpu(),
    "multi_step": outputs,
    "multi_step_coords": coords_list,
    "n_steps": N_STEPS,
}, "ref_<model_name>_e2s_outputs.pt")
```

### Comparison script

Create `reference_<model_name>_compare.py`:

```python
"""Compare vanilla vs E2S outputs for <ModelName>.

This script is for validation only — do NOT commit to the repo.
"""
import torch
import numpy as np

vanilla = torch.load("ref_<model_name>_vanilla_outputs.pt", weights_only=False)
e2s = torch.load("ref_<model_name>_e2s_outputs.pt", weights_only=False)

assert vanilla["n_steps"] == e2s["n_steps"]
N_STEPS = vanilla["n_steps"]


def compare_tensors(ref: torch.Tensor, test: torch.Tensor, name: str) -> dict:
    ref, test = ref.float(), test.float()
    diff = (ref - test).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_diff = diff / (ref.abs() + 1e-8)
    max_rel = rel_diff.max().item()
    corr = torch.corrcoef(torch.stack([ref.flatten(), test.flatten()]))[0, 1].item()
    return {"name": name, "max_abs": max_abs, "mean_abs": mean_abs, "max_rel": max_rel, "correlation": corr}


def print_metrics(m: dict):
    print(f"  {m['name']}:")
    print(f"    Max absolute diff: {m['max_abs']:.2e}")
    print(f"    Mean absolute diff: {m['mean_abs']:.2e}")
    print(f"    Max relative diff: {m['max_rel']:.2e}")
    print(f"    Correlation: {m['correlation']:.8f}")


# Single-step
print("=" * 60)
print("SINGLE-STEP COMPARISON")
print("=" * 60)
single_metrics = compare_tensors(vanilla["single_step"], e2s["single_step"], "Single step")
print_metrics(single_metrics)

# Multi-step
print(f"\n{'=' * 60}")
print(f"MULTI-STEP COMPARISON ({N_STEPS} steps)")
print("=" * 60)
multi_metrics = []
for i in range(N_STEPS):
    m = compare_tensors(vanilla["multi_step"][i], e2s["multi_step"][i], f"Step {i + 1}")
    multi_metrics.append(m)
    print_metrics(m)

# Summary
print(f"\n{'=' * 60}")
print("SUMMARY")
print("=" * 60)

TOLERANCE_ABS = 1e-4
TOLERANCE_CORR = 0.9999
all_pass = True

if single_metrics["max_abs"] > TOLERANCE_ABS:
    print(f"WARNING: Single-step max_abs {single_metrics['max_abs']:.2e} > {TOLERANCE_ABS}")
    all_pass = False
for m in multi_metrics:
    if m["correlation"] < TOLERANCE_CORR:
        print(f"WARNING: {m['name']} correlation {m['correlation']:.6f} < {TOLERANCE_CORR}")
        all_pass = False

print("PASS: All comparisons within tolerance" if all_pass else "FAIL: Review warnings above")
```

### Run and verify

```bash
uv run python reference_<model_name>_vanilla.py
uv run python reference_<model_name>_e2s.py
uv run python reference_<model_name>_compare.py
```

---

## Sanity-Check Plots

Create `sanity_check_<model_name>.py` (do NOT commit):

```python
"""Sanity-check plot for <ModelName> prognostic model.

This script is for PR review only — do NOT commit to the repo.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

# --- PART 1: Reference model inference (third-party only) ---
# TODO: Load and run original model
raise NotImplementedError("Fill in reference model code, then remove this line.")

# --- PART 2: Earth2Studio wrapper inference ---
from earth2studio.models.px import ModelName

model = ModelName.load_model(ModelName.load_default_package())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

time = np.array([np.datetime64("2024-01-01T00:00")])
input_coords = model.input_coords()
input_coords["time"] = time

# Create input with same seed
torch.manual_seed(42)
shape = tuple(max(len(v), 1) for v in input_coords.values())
x = torch.randn(shape, device=device)

# Run multi-step forecast
N_STEPS = 5
iterator = model.create_iterator(x, input_coords)
e2s_steps = []
for i, (step_x, step_coords) in enumerate(iterator):
    e2s_steps.append((step_x.cpu().numpy(), dict(step_coords)))
    if i >= N_STEPS:
        break

# --- PART 3: Side-by-side visualization ---
var_list = list(e2s_steps[0][1]["variable"])
plot_vars = var_list[:3]
n_vars = len(plot_vars)

step_indices = [0, N_STEPS]
n_cols = len(step_indices) * 2
fig, axes = plt.subplots(n_vars, n_cols, figsize=(4 * n_cols, 4 * n_vars))
if n_vars == 1:
    axes = axes[np.newaxis, :]

for row, var in enumerate(plot_vars):
    var_idx = var_list.index(var)
    for col_idx, si in enumerate(step_indices):
        # Reference output (TODO: adapt indexing)
        ref_data_2d = np.zeros((100, 100))  # Replace with actual ref

        # E2S output
        e2s_data, sc = e2s_steps[si]
        e2s_data_2d = e2s_data[0, 0, 0, var_idx, :, :]
        lead = sc["lead_time"]

        ax_ref = axes[row, col_idx * 2]
        im_ref = ax_ref.contourf(ref_data_2d, cmap="turbo", levels=20)
        ax_ref.set_title(f"REF: {var} | step={si}")
        plt.colorbar(im_ref, ax=ax_ref, shrink=0.8)

        ax_e2s = axes[row, col_idx * 2 + 1]
        im_e2s = ax_e2s.contourf(e2s_data_2d, cmap="turbo", levels=20)
        ax_e2s.set_title(f"E2S: {var} | lead={lead}")
        plt.colorbar(im_e2s, ax=ax_e2s, shrink=0.8)

plt.suptitle("<ModelName> — Reference vs Earth2Studio comparison", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_<model_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<model_name>.png")
```

---

## Branch, Commit & Open PR

### Create branch and commit

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

Do NOT add reference scripts, comparison scripts, or images.

### Identify fork remote

```bash
git remote -v
```

Ask user which remote is their fork. Push to fork:

```bash
git push -u <fork-remote> feat/prognostic-model-<name>
```

### Open PR (fork -> NVIDIA/earth2studio)

```bash
gh pr create \
  --repo NVIDIA/earth2studio \
  --base main \
  --head <fork-owner>:feat/prognostic-model-<name> \
  --title "feat: add <ClassName> prognostic model" \
  --body "..."
```

### PR body template

```markdown
## Description

Add `ClassName` prognostic model for BRIEF_DESCRIPTION.

Closes #ISSUE_NUMBER (if applicable)

### Model details

| Property | Value |
|---|---|
| **Architecture** | PyTorch / ONNX / JAX |
| **Time step** | Xh |
| **Input variables** | N variables |
| **Output variables** | M variables |
| **Spatial resolution** | X° x Y° (NxM grid) |
| **History required** | None / Xh |
| **Checkpoint source** | NGC / HuggingFace / S3 |
| **Checkpoint size** | XX MB |
| **Checkpoint license** | LICENSE_NAME / Unknown |
| **Reference** | PAPER_OR_REPO_URL |

### Dependencies added

| Package | Version | License | License URL | Reason |
|---|---|---|---|---|
| `PACKAGE` | `>=X.Y` | LICENSE | [link](URL) | REASON |

(or "No new dependencies needed")

### Reference comparison

**Single step:** max_abs=VALUE, corr=VALUE
**Multi-step (N steps):** Step 1: max_abs=VALUE, corr=VALUE

### Validation

See sanity-check plots in PR comments below.

## Checklist

- [x] I am familiar with the Contributing Guidelines.
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [x] The CHANGELOG.md is up to date with these changes.
- [ ] An issue is linked to this pull request.
- [ ] Assess and address Greptile feedback.
```

### Post sanity-check as PR comment

```markdown
## Sanity-Check Validation

**Model:** `ClassName` — BRIEF_DESCRIPTION
**Architecture:** PyTorch / ONNX / JAX
**Time step:** Xh
**Test environment:** GPU_MODEL

### Reference Comparison

- SINGLE_STEP_AGREEMENT
- MULTISTEP_ERROR_GROWTH
- SPATIAL_PATTERN_QUALITY

> **TODO:** Attach sanity-check image by editing this comment.

<details>
<summary>Sanity-check script (click to expand)</summary>

```python
PASTE FULL WORKING SCRIPT HERE
```

</details>
```

Post comment:
```bash
gh api -X POST repos/NVIDIA/earth2studio/issues/<PR_NUMBER>/comments \
  -F "body=@/tmp/pr_reference_comment.md" \
  --jq '.html_url'
```

---

## Automated Code Review

### Poll for Greptile review

Wait up to 5 minutes (poll every 30s):

```bash
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

### Fetch review comments

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") |
    {path: .path, line: .diff_hunk, body: .body}'

gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") | .body'
```

### Categorize feedback

| Category | Description | Default action |
|---|---|---|
| **Bug / correctness** | Logic errors, wrong behavior | Fix |
| **Style / convention** | Naming, formatting, patterns | Fix if valid |
| **Performance** | Inefficiency, resource waste | Evaluate |
| **Documentation** | Missing/wrong docs, docstrings | Fix |
| **Suggestion** | Alternative approach, nice-to-have | User decides |
| **False positive** | Incorrect or irrelevant feedback | Dismiss |

Present triage table to user:

```markdown
| # | File | Line | Category | Summary | Proposed Action |
|---|------|------|----------|---------|-----------------|
| 1 | <model>.py | 142 | Bug | Missing null check | Fix: add guard |
| 2 | <model>.py | 305 | Style | Use f-string | Fix: convert |
```

### Implement accepted fixes

1. Make code changes
2. Run `make format && make lint`
3. Run tests: `uv run python -m pytest test/models/px/test_<filename>.py -v --timeout=60`
4. Commit: `git commit -m "fix: address code review feedback (Greptile)"`

### Respond to comments

Fixed:
```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Fixed in <commit_sha>. <brief description>"
```

Dismissed:
```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Won't fix — <brief justification>"
```

### Push and report

```bash
git push <fork-remote> feat/prognostic-model-<name>
```

Report: comments fixed, dismissed (with reasons), remaining open threads.
