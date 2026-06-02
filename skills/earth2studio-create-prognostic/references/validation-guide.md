# Validation Guide for Prognostic Model Wrappers

This guide covers Steps 10-12: reference comparison, PR submission, and code review.

## Step 10 — Reference Comparison & Validation

Create three scripts in the repo root (do NOT commit):

### 10a. Vanilla Reference Script

Create `reference_<model>_vanilla.py`:

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

# ============================================================
# PART 2: Prepare input data
# ============================================================
torch.manual_seed(42)
np.random.seed(42)

# TODO: Define input shape matching reference model expectations
# input_shape = (batch, channel, lat, lon)
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
}, "ref_<model>_vanilla_outputs.pt")

print(f"Saved vanilla reference outputs ({N_STEPS} steps)")
```

### 10b. Earth2Studio Reference Script

Create `reference_<model>_e2s.py`:

```python
"""Earth2Studio reference inference for <ModelName>.

Runs inference using the E2S wrapper for comparison against
vanilla reference.

This script is for validation only — do NOT commit to the repo.
"""

import numpy as np
import torch

from earth2studio.models.px import ModelName

# ============================================================
# PART 1: Load the E2S model
# ============================================================
model = ModelName.load_model(ModelName.load_default_package())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ============================================================
# PART 2: Prepare input data (same seed as vanilla)
# ============================================================
torch.manual_seed(42)
np.random.seed(42)

input_coords = model.input_coords()
shape = tuple(max(len(v), 1) for v in input_coords.values())
x = torch.randn(shape, device=device, dtype=torch.float32)

# ============================================================
# PART 3: Run inference
# ============================================================
N_STEPS = 8

with torch.no_grad():
    # Single-step output
    y_single, out_coords = model(x, input_coords)

    # Multi-step via create_iterator
    iterator = model.create_iterator(x, input_coords)
    step0_x, step0_coords = next(iterator)  # Initial condition

    outputs = []
    for step in range(N_STEPS):
        y_step, c_step = next(iterator)
        outputs.append(y_step.cpu())

# ============================================================
# PART 4: Save outputs
# ============================================================
torch.save({
    "input": x.cpu(),
    "single_step": y_single.cpu(),
    "multi_step": outputs,
    "n_steps": N_STEPS,
}, "ref_<model>_e2s_outputs.pt")

print(f"Saved E2S reference outputs ({N_STEPS} steps)")
```

### 10c. Comparison Script

Create `reference_<model>_compare.py`:

```python
"""Compare vanilla vs E2S outputs for <ModelName>."""

import torch

vanilla = torch.load("ref_<model>_vanilla_outputs.pt", weights_only=False)
e2s = torch.load("ref_<model>_e2s_outputs.pt", weights_only=False)

def compare(ref, test, name):
    diff = (ref - test).abs()
    print(f"{name}:")
    print(f"  Max abs diff: {diff.max().item():.2e}")
    print(f"  Mean abs diff: {diff.mean().item():.2e}")
    corr = torch.corrcoef(torch.stack([ref.flatten(), test.flatten()]))[0, 1]
    print(f"  Correlation: {corr.item():.8f}")
    return diff.max().item(), corr.item()

# Single step
print("=" * 60)
print("SINGLE-STEP")
print("=" * 60)
compare(vanilla["single_step"], e2s["single_step"], "Single step")

# Multi-step
print("\n" + "=" * 60)
print(f"MULTI-STEP ({vanilla['n_steps']} steps)")
print("=" * 60)
for i in range(vanilla["n_steps"]):
    compare(vanilla["multi_step"][i], e2s["multi_step"][i], f"Step {i+1}")
```

### 10d. Run Scripts

```bash
uv run python reference_<model>_vanilla.py
uv run python reference_<model>_e2s.py
uv run python reference_<model>_compare.py
```

### 10e. Acceptance Criteria

- Single-step max absolute difference < 1e-4
- Multi-step correlation > 0.9999
- No NaN/Inf values

---

## Step 11 — Branch, Commit & Open PR

### 11a. Create Branch

```bash
git checkout -b feat/prognostic-model-<name>
```

### 11b. Stage Files

```bash
git add earth2studio/models/px/<filename>.py \
        earth2studio/models/px/__init__.py \
        test/models/px/test_<filename>.py \
        pyproject.toml \
        CHANGELOG.md \
        docs/modules/models_px.rst \
        docs/userguide/about/install.md
```

Do NOT add: reference scripts, comparison outputs, images.

### 11c. Commit

```bash
git commit -m "feat: add <ClassName> prognostic model

Add <ClassName> prognostic model for <brief description>.
Includes unit tests and documentation."
```

### 11d. Push to Fork

```bash
git push -u origin feat/prognostic-model-<name>
```

### 11e. Create PR

```bash
gh pr create \
  --repo NVIDIA/earth2studio \
  --base main \
  --head <your-username>:feat/prognostic-model-<name> \
  --title "feat: add <ClassName> prognostic model" \
  --body-file pr-body.md
```

### PR Body Template

```markdown
## Description

Add `ClassName` prognostic model for BRIEF_DESCRIPTION.

### Model details

| Property | Value |
|---|---|
| **Architecture** | PyTorch / ONNX / JAX |
| **Time step** | Xh |
| **Input variables** | N variables |
| **Spatial resolution** | X° x Y° |
| **Checkpoint source** | NGC / HuggingFace / S3 |
| **Checkpoint license** | LICENSE_NAME |
| **Reference** | PAPER_URL |

### Dependencies added

| Package | Version | License |
|---|---|---|
| `package` | `>=X.Y` | MIT |

### Reference comparison

**Single step:** max_abs=X.XXe-XX, corr=0.XXXXXX
**Multi-step (8 steps):** Step 8 corr=0.XXXXXX

## Checklist

- [x] Tests cover these changes
- [x] Documentation updated
- [x] CHANGELOG.md updated
```

### 11f. Post Reference Comparison Comment

Post comparison results and script snippets as a PR comment.

---

## Step 12 — Automated Code Review (Greptile)

### 12a. Wait for Review

Poll for greptile-apps[bot] review (5 min timeout):

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR>/reviews \
  --jq '.[] | select(.user.login == "greptile-apps[bot]")'
```

### 12b. Categorize Feedback

| Category | Action |
|---|---|
| Bug/correctness | Fix |
| Style/convention | Fix if valid |
| Performance | Evaluate |
| Documentation | Fix |
| Suggestion | User decides |
| False positive | Dismiss |

### 12c. Implement Fixes

```bash
# Make fixes
make format && make lint
uv run pytest test/models/px/test_<filename>.py -v
git commit -m "fix: address code review feedback"
git push
```

### 12d. Respond to Comments

For fixed:
```
Fixed in <commit_sha>. <description>
```

For dismissed:
```
Won't fix — <justification>
```

---

## Sanity-Check Plot Template

```python
"""Generate sanity-check plots for <ModelName>."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.models.px import ModelName
from earth2studio.data import GFS

# Load model
model = ModelName.load_model(ModelName.load_default_package())
model = model.to("cuda")

# Fetch real data
time = np.array([np.datetime64("2024-01-15T00:00")])
gfs = GFS()
x, coords = gfs(time, model.input_coords()["variable"])
x = x.to("cuda")

# Run 5-day forecast
iterator = model.create_iterator(x, coords)
outputs = []
for i, (y, c) in enumerate(iterator):
    outputs.append((y.cpu(), c.copy()))
    if i >= 20:  # 5 days at 6h step
        break

# Plot t2m at day 0, 2, 5
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, day in zip(axes, [0, 8, 20]):
    y, c = outputs[day]
    t2m_idx = list(c["variable"]).index("t2m")
    im = ax.imshow(y[0, 0, 0, t2m_idx].numpy(), cmap="RdBu_r")
    ax.set_title(f"T2M at {c['lead_time'][0]}")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("sanity_check_<model>.png", dpi=150)
print("Saved sanity_check_<model>.png")
```
