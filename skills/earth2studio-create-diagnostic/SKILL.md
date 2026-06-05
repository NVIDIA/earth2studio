---
name: earth2studio-create-diagnostic
version: 0.17.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team <agent-skills@nvidia.com>
  tags: [earth2studio, diagnostic-model, python]
description: >
  Create Earth2Studio diagnostic (single-step transformation) model wrappers.
  Use when wrapping models that transform data at a single time point without
  time integration. Covers deterministic, automodel-based, and generative
  (diffusion) diagnostics like CorrDiff. Do NOT use for prognostic models,
  data sources, or installation tasks.
argument-hint: URL or local path to reference inference script (optional)
---

## Quick Start Checklist

**Do these steps IN ORDER. Do not skip any step.**

- [ ] Read this SKILL.md completely first
- [ ] Get reference script (Step 0)
- [ ] Create `earth2studio/models/dx/<name>.py` with appropriate inheritance
- [ ] Create `test/models/dx/test_<name>.py` with mock tests
- [ ] Run: `uv run pytest test/models/dx/test_<name>.py -v`
- [ ] Run: `make format && make lint`

> **CRITICAL:** Always use `uv run` for Python commands:
> - `uv run pytest ...` / `uv run python ...`
> - `pytest ...` / `python ...` (missing dependencies)
>
> **Stuck or wrong output:** Do not keep retrying the same fix. Follow
> [Self-Improvement](#self-improvement) to patch this skill before continuing.

## Purpose

Implement a diagnostic model wrapper connecting third-party ML models to
Earth2Studio. Diagnostic models transform physical data at a single time point
(no time integration)—they take input fields and produce output fields without
stepping through time.

**Key difference from prognostic models:**
- NO `create_iterator` method (diagnostics don't time-integrate)
- NO `PrognosticMixin` (not needed without iterator hooks)
- NO `lead_time` coordinate (single point-in-time transformation)

## Diagnostic Model Types

| Type | Description | Inheritance | Example |
|------|-------------|-------------|---------|
| **Simple** | Basic transformation, no AutoModel | `torch.nn.Module` only | `Identity` |
| **AutoModel** | Loads from package (NGC/HuggingFace) | `torch.nn.Module + AutoModelMixin` | `PrecipitationAFNO` |
| **Generative** | Produces samples (diffusion, VAE) | `torch.nn.Module + AutoModelMixin` | `CorrDiff` |

## Workspace

| Context | Location |
|---------|----------|
| Harbor eval | Write to `/workspace/output/earth2studio/models/dx/...` |
| Harbor + `--copy-repo` | Full checkout at `/workspace/repo` |
| Local clone | Directory with `pyproject.toml` |

**Never read `evals/targets/`** — grader references only.

### Reference Files

Load on demand during the matching step:

| File | Content | Load at |
|------|---------|---------|
| `references/skeleton-template.py` | Full model skeleton with FILL comments | Steps 3–5 |
| `references/testing-guide.py` | Test skeleton and mock patterns | Step 6 |

---

## Workflow Steps

### Step 0 — Get Reference Script

If `$ARGUMENTS` provided, use it. Otherwise ask:
> Please provide a reference inference script URL/path.

### Step 1 — Analyze & Classify Model Type

Analyze the reference script to determine:
- **Model type**: Simple, AutoModel, or Generative
- **Input/output variables**: What fields are consumed and produced
- **Grid resolution**: Lat/lon dimensions
- **Dependencies**: Required packages (e.g., `physicsnemo` for CorrDiff)
- **Normalization**: Center/scale parameters

**Classification criteria:**

| If the model... | Then use... |
|-----------------|-------------|
| Has no checkpoints, simple transform | Simple (no AutoModelMixin) |
| Loads weights from package | AutoModel |
| Produces multiple samples (diffusion, VAE) | Generative |

### Step 2 — Propose Dependencies (if needed)

For AutoModel and Generative types, propose `pyproject.toml` group:
```toml
model-name = ["package1>=version", "package2"]
```

**[CONFIRM]** Present dependencies and ask user to approve.

### Step 3 — Create Model File

**File:** `earth2studio/models/dx/<lowercase>.py`

**Required inheritance by type:**

```python
# Simple (no checkpoint loading)
class ModelName(torch.nn.Module):

# AutoModel (loads from package)
class ModelName(torch.nn.Module, AutoModelMixin):

# Generative (diffusion models with samples)
class ModelName(torch.nn.Module, AutoModelMixin):
```

**Note:** Diagnostic models do NOT inherit from `PrognosticMixin`.

**Required imports:**
```python
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.type import CoordSystem
```

**SPDX header (required at top of every .py file):**
```python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
```

**Canonical method order:**
1. `__init__` — constructor
2. `input_coords` — input coordinate system
3. `output_coords` — output coordinate system (@batch_coords)
4. `load_default_package` — classmethod returning default Package (AutoModel only)
5. `load_model` — classmethod loading model from package (AutoModel only)
6. `to` — device management (optional)
7. Private methods (e.g., `_normalize`, `_forward`)
8. `__call__` — single-step forward (@batch_func)

### Step 4 — Implement Coordinates

**input_coords rules:**
- `batch`: `np.empty(0)` — MUST be first
- `variable`: Input variable names (map to `E2STUDIO_VOCAB`)
- `lat`: 90 to -90 (north to south) for global grids
- `lon`: 0 to 360 (typically)
- NO `lead_time` dimension (unlike prognostic models)

**output_coords:** Use `handshake_dim`/`handshake_coords`, update `variable`.

```python
def input_coords(self) -> CoordSystem:
    return OrderedDict({
        "batch": np.empty(0),
        "variable": np.array(INPUT_VARIABLES),
        "lat": np.linspace(90, -90, 720, endpoint=False),
        "lon": np.linspace(0, 360, 1440, endpoint=False),
    })

@batch_coords()
def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
    target = self.input_coords()
    handshake_dim(input_coords, "variable", 1)
    handshake_dim(input_coords, "lat", 2)
    handshake_dim(input_coords, "lon", 3)
    handshake_coords(input_coords, target, "variable")
    handshake_coords(input_coords, target, "lat")
    handshake_coords(input_coords, target, "lon")

    output_coords = input_coords.copy()
    output_coords["variable"] = np.array(OUTPUT_VARIABLES)
    return output_coords
```

**For Generative models with samples:**
```python
@batch_coords()
def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
    # ... handshake validation ...
    output_coords = OrderedDict({
        "batch": input_coords["batch"],
        "sample": np.arange(self.number_of_samples),  # NEW: sample dimension
        "variable": np.array(OUTPUT_VARIABLES),
        "lat": self.lat_output,
        "lon": self.lon_output,
    })
    return output_coords
```

### Step 5 — Implement Forward Pass

**`__call__`:** @batch_func decorated, single transformation.

```python
@torch.inference_mode()
@batch_func()
def __call__(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    """Forward pass of diagnostic model."""
    output_coords = self.output_coords(coords)

    # Move to device
    device = next(self.parameters()).device
    x = x.to(device)

    # Normalize → forward → denormalize
    x = self._normalize(x)
    out = self.core_model(x)
    out = self._denormalize(out)

    return out, output_coords
```

**For Generative models (CorrDiff pattern):**
```python
@batch_func()
def __call__(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    output_coords = self.output_coords(coords)

    # Generate samples
    out = torch.zeros(
        [len(v) for v in output_coords.values()],
        device=x.device,
        dtype=torch.float32,
    )

    for i in range(out.shape[0]):  # batch
        out[i] = self._forward(x[i])  # produces [samples, var, lat, lon]

    return out, output_coords
```

### Step 6 — Implement Model Loading (AutoModel only)

**`load_default_package`:** Lock HuggingFace/NGC URLs.

```python
@classmethod
def load_default_package(cls) -> Package:
    return Package(
        "ngc://models/org/model@version",
        cache_options={
            "cache_storage": Package.default_cache("model_name"),
            "same_names": True,
        },
    )
```

**`load_model`:** Use `package.resolve()`, `map_location="cpu"`, `eval()` mode.

```python
@classmethod
@check_optional_dependencies()
def load_model(cls, package: Package) -> DiagnosticModel:
    checkpoint_path = package.resolve("model.pt")
    core_model = torch.load(checkpoint_path, map_location="cpu")
    core_model.eval()
    core_model.requires_grad_(False)

    # Load normalization
    center = torch.from_numpy(np.load(package.resolve("center.npy")))
    scale = torch.from_numpy(np.load(package.resolve("scale.npy")))

    return cls(core_model, center=center, scale=scale)
```

### Step 7 — Write Tests

**File:** `test/models/dx/test_<name>.py`

**Required tests:**
| Function | Purpose |
|----------|---------|
| `test_<model>_call` | Forward pass with mock model |
| `test_<model>_exceptions` | Invalid coords raise errors |
| `test_<model>_package` | Real weights (`@pytest.mark.package`) |

**For Generative models, add:**
| Function | Purpose |
|----------|---------|
| `test_<model>_samples` | Correct number of samples produced |
| `test_<model>_deterministic_seed` | Reproducibility with seed |

Create `Phoo<ModelName>` dummy matching interface for mock tests.

**Run tests:**
```bash
uv run pytest test/models/dx/test_<name>.py -m "not package" -v
```

### Step 8 — Register Model (if requested)

- Add to `earth2studio/models/dx/__init__.py` (alphabetical)
- Verify deps in pyproject.toml

### Step 9 — Documentation

- Add to `docs/modules/models_dx.rst` (alphabetical)
- Add to `docs/userguide/about/install.md` if optional deps
- Update `CHANGELOG.md` under `### Added`

**Format and lint:**
```bash
make format && make lint && make license
```

### Step 10 — Validation (if requested)

Create comparison scripts (do NOT commit):
1. `reference_<model>_vanilla.py` — third-party inference
2. `reference_<model>_e2s.py` — E2S wrapper inference
3. `reference_<model>_compare.py` — numerical comparison

**[CONFIRM]** User must visually inspect plots before proceeding.

### Step 11 — PR (if requested)

1. Branch: `feat/diagnostic-model-<name>`
2. Commit (exclude comparison scripts)
3. `gh pr create --repo NVIDIA/earth2studio`

---

## Examples

### Simple Diagnostic (no checkpoints)
```text
User: Create a diagnostic that computes wind speed from u10m and v10m

Agent: [reads SKILL.md, creates windspeed.py inheriting torch.nn.Module only,
        creates test_windspeed.py, runs pytest]
```

### AutoModel Diagnostic
```text
User: Add PrecipNet wrapper from this reference script
      URL: https://example.com/precipnet.py

Agent: [reads SKILL.md, analyzes script, creates precipnet.py with AutoModelMixin,
        creates test_precipnet.py, runs pytest]
```

### Generative Diagnostic (CorrDiff-like)
```text
User: Wrap this diffusion-based super-resolution model
      Reference: https://example.com/sr_diffusion/inference.py

Agent: [reads SKILL.md, classifies as Generative, creates sr_diffusion.py
        with sample dimension in output_coords, creates test_sr_diffusion.py]
```

---

## Key Patterns

### Deterministic Diagnostic Template
```python
class MyDiagnostic(torch.nn.Module, AutoModelMixin):
    def __init__(self, core_model, center, scale):
        super().__init__()
        self.core_model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def input_coords(self) -> CoordSystem:
        return OrderedDict({
            "batch": np.empty(0),
            "variable": np.array(INPUT_VARS),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        })

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        target = self.input_coords()
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(OUTPUT_VARS)
        return output_coords

    @torch.inference_mode()
    @batch_func()
    def __call__(self, x, coords):
        output_coords = self.output_coords(coords)
        x = (x - self.center) / self.scale
        out = self.core_model(x)
        return out, output_coords
```

### Generative Diagnostic Template (CorrDiff pattern)
```python
class MyGenerative(torch.nn.Module, AutoModelMixin):
    def __init__(self, ..., number_of_samples=1, seed=None):
        super().__init__()
        self.number_of_samples = number_of_samples
        self.seed = seed
        # ... register models and buffers

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        # ... validation ...
        return OrderedDict({
            "batch": input_coords["batch"],
            "sample": np.arange(self.number_of_samples),
            "variable": np.array(OUTPUT_VARS),
            "lat": self.lat_output,
            "lon": self.lon_output,
        })

    @batch_func()
    def __call__(self, x, coords):
        output_coords = self.output_coords(coords)
        out = torch.zeros([len(v) for v in output_coords.values()], ...)

        for batch_idx in range(out.shape[0]):
            out[batch_idx] = self._forward(x[batch_idx])

        return out, output_coords

    def _forward(self, x):
        # Generate number_of_samples using diffusion/VAE
        samples = torch.concat([self._generate_sample(x, i)
                                for i in range(self.number_of_samples)], dim=0)
        return samples
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `OptionalDependencyFailure` | `uv add --optional <group> <pkg>` |
| Coordinate handshake fails | Check `handshake_dim` indices match dim position |
| Wrong output shape | Verify output_coords matches returned tensor shape |
| `ModuleNotFoundError: pytest` | Use `uv run pytest` not `pytest` |

---

## Reminders

**DO:**
- Use `uv run python` for ALL Python commands
- Use `loguru.logger`, never `print()` inside `earth2studio/`
- Use `@batch_func()` on `__call__`
- Use `@batch_coords()` on `output_coords`
- `batch` must be first dimension with `np.empty(0)`
- Validate coordinates with `handshake_coords()` and `handshake_dim()`
- For Generative: add `sample` dimension in output_coords
- Include SPDX header in every .py file

**DON'T:**
- Inherit from `PrognosticMixin` (not for diagnostics)
- Include `lead_time` coordinate (not for diagnostics)
- Create `create_iterator` method (not for diagnostics)
- Create general base classes for reuse
- Commit API keys or comparison scripts
- Read from `evals/targets/`

---

## Self-Improvement

If this skill produces incorrect outputs, update it before continuing:

1. Identify the issue in the generated code
2. Edit `SKILL.md` to fix the guidance
3. Commit: `git add SKILL.md && git commit -m "fix: <description>"`
4. Continue with the corrected workflow
