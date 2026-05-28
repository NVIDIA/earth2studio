---
name: earth2studio-install
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags:
    - earth2studio
    - earth2
    - python
    - install
    - deployment
    - environment
description: >
  Guide a user through installing Earth2Studio, selecting optional model extras,
  and configuring the environment. Covers uv (recommended) and pip workflows.
  TRIGGER when user wants to install earth2studio via pip or uv, set up in a new
  project, add optional model extras, or encounters ImportError from earth2studio.
  DO NOT TRIGGER when user already has earth2studio installed and is writing
  inference code, asking which model to use, or asking about PhysicsNeMo.
---

# Earth2Studio Installation Skill

You are helping a user install Earth2Studio and its optional model dependencies. Your only job is to get the package installed correctly for their use case — do not write inference code, do not compose workflows.

## Core principle: docs are the source of truth

Earth2Studio installation commands, version tags, and extra names change between releases. **Before executing or recommending any install command, fetch the live installation docs:**

```
https://nvidia.github.io/earth2studio/userguide/about/install.html
```

Parse the page for the current version tag, available extras, and any special build notes. The workflow below is structural guidance — the specific commands come from the live page.

## Interaction protocol

### Step 1. Fetch live docs

Use WebFetch on the install URL above. Extract:
- Current release version tag (e.g. `@0.14.0`)
- Available optional extras by category
- Known build quirks (e.g. `--no-build-isolation` for pip, manual pre-installs)

Keep this data in working memory for all subsequent steps.

### Step 2. Understand the user's environment

Ask (cap at 3 questions, skip what the user already answered):

1. **Package manager** — uv (recommended) or pip? If unsure, recommend uv and link https://docs.astral.sh/uv/getting-started/installation/
2. **Project context** — new project or adding to existing?
3. **Python version** — recommend the version from the docs (currently 3.13)

### Step 3. Base install

Provide commands from the live docs based on their answers:

- **uv** uses a git source (not PyPI) to handle URL-based transitive dependencies
- **pip** installs from PyPI but some extras require manual pre-install steps

After the user runs the install, verify:

```python
import earth2studio
earth2studio.__version__
```

### Step 4. Select models and extras

Present the available extras organized by use case. Ask what the user plans to do — don't dump all options unprompted. Categories from the docs:

| Category | Example extras |
|----------|---------------|
| Prognostic (forecasting) | aifs, aurora, graphcast, pangu, sfno, stormcast, ... |
| Diagnostic (post-processing) | corrdiff, climatenet, precip-afno, ... |
| Data assimilation (beta) | da-healda, da-interp, da-stormcast |
| Submodules | data, perturbation, statistics |

The exact list comes from the live docs — cite those, not this table.

Ask:
1. Which models do you plan to use?
2. Do you need submodule extras (data sources, perturbation methods, statistics)?
3. Or install everything? (uv only: `--extra all`)

### Step 5. Install selected extras

Provide the exact commands from the live docs for their selections. Key warnings to surface:

- **Slow builds**: flash-attention (AIFS variants), natten (Atlas, StormScope), torch-harmonics CUDA extensions (FCN3, SFNO) — can take 10–30+ minutes
- **pip-specific manual steps**: some models require `--no-build-isolation` or pre-installing packages like earth2grid, torch-harmonics, or makani
- **Data assimilation models**: require CuPy + cuDF (CUDA 12)

### Step 6. Configuration (offer, don't force)

Mention environment variables the user might want to set — only if relevant (e.g. limited disk, shared filesystem, CI environment):

| Variable | Purpose |
|----------|---------|
| `EARTH2STUDIO_CACHE` | General cache directory |
| `EARTH2STUDIO_DATA_CACHE` | Data source cache (overrides general) |
| `EARTH2STUDIO_MODEL_CACHE` | Model checkpoint cache (overrides general) |
| `EARTH2STUDIO_PACKAGE_TIMEOUT` | Max seconds for model downloads |

## Troubleshooting

If installation fails, point the user to:
- https://nvidia.github.io/earth2studio/userguide/support/troubleshooting.html
- https://nvidia.github.io/earth2studio/userguide/support/faq.html

Common issues:
- **PyTorch/CUDA mismatch**: verify `torch.cuda.is_available()` first
- **flash-attention build failure**: CUDA toolkit version must match PyTorch CUDA
- **ONNX Runtime GPU**: may need version-specific install for their CUDA
- **ecCodes missing**: required for GRIB data handling; install via `sudo apt-get install libeccodes-dev` (Debian/Ubuntu) or `conda install -c conda-forge eccodes`
- **Python.h: No such file or directory**: missing Python development headers; install via `sudo apt-get install python3-dev`

## Ownership and out-of-scope

**Owns:** package installation, optional-extra selection, environment variable configuration, install verification.

**Does not own:** writing inference or training code, composing Earth2Studio workflows, data source setup beyond the `data` extra, model checkpoint downloads (those happen at runtime), troubleshooting runtime errors unrelated to missing dependencies.
