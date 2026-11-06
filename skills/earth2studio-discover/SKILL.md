---
name: earth2studio-discover
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags:
    - earth2studio
    - earth2
    - python
    - discovery
    - models
    - data-sources
description: >
  Find Earth2Studio models, data sources, and examples for a weather/climate use
  case. Do NOT use for writing inference code, downloading data, or installation.
---

# Earth2Studio Discoverability Skill

## Purpose

Help users find the right Earth2Studio models, data sources, and examples for a
weather/climate task: compare models by class, region, and GPU memory, find data
sources that cover the variables a model needs, and locate a relevant example.

## Core principle: discover from the repo and live docs, do not memorize

Models, data sources, and examples are added every release. Any list written
into this skill will go stale. Read the sources below, filter, and cite them.

| Need | Where to look (repo path first, live page as a fallback) |
|------|------------------------------------------------|
| Prognostic, diagnostic, and data assimilation models, with their badges (class, region, GPU memory, year, provider) | `docs/modules/models_px.md`, `models_dx.md`, `models_da.md` |
| Data sources (analysis, forecast, dataframe) | `docs/modules/datasources_analysis.md`, `datasources_forecast.md`, `datasources_dataframe.md` |
| What a model needs (variables, grid, lead times) | the model class in `earth2studio/models/{px,dx,da}/` (`input_coords`, `output_coords`) and its docstring |
| What a data source provides | the class in `earth2studio/data/` and its lexicon in `earth2studio/lexicon/` |
| Variable naming conventions | `docs/userguide/advanced/lexicon.md` and `E2STUDIO_VOCAB` in `earth2studio/lexicon/base.py` |
| Examples, by workflow type | `examples/` (numbered category folders, each with a `README.rst`) and the gallery page |
| Concepts (what px / dx / da / data / io mean) | `docs/userguide/components/` |

Live pages (note the `/main/` segment; URLs without it return 404):

```text
https://nvidia.github.io/earth2studio/main/modules/models_px/
https://nvidia.github.io/earth2studio/main/modules/models_dx/
https://nvidia.github.io/earth2studio/main/modules/models_da/
https://nvidia.github.io/earth2studio/main/modules/datasources_analysis/
https://nvidia.github.io/earth2studio/main/modules/datasources_forecast/
https://nvidia.github.io/earth2studio/main/modules/datasources_dataframe/
https://nvidia.github.io/earth2studio/main/examples/
```

If working in a local clone, read the files directly. Otherwise fetch from
`https://github.com/NVIDIA/earth2studio` or the live pages. Fetch only what the
question needs, not everything.

## Instructions

1. **Understand the task.** Ask at most 3 questions, skipping what is known:
   task type (forecast, nowcast, downscaling, assimilation, ensemble), region,
   time horizon, variables, GPU memory, deterministic vs. ensemble.
2. **Find candidate models** in the model pages. Filter with the badges shown
   there. Chains are common (a prognostic model feeding a diagnostic one), so
   check both kinds when relevant.
3. **Find data sources** that cover the user's region, variables, and need
   (historical, operational, or observations).
4. **Check compatibility from code, not by name.** Read the model's
   `input_coords` for its variable list and check the data source's lexicon
   (`earth2studio/lexicon/`) covers every one. The model docs and API pages
   are a fallback when only the docs are available, but they are less precise,
   so say so.
5. **Pick 1-3 relevant examples** from `examples/` and say what each shows.
6. **Reply concisely** with the use case, 2-4 recommended models (with the
   tradeoff between them), compatible data sources, examples, and next steps.
   Always include links to the pages or file paths you used. Send installation
   questions to `earth2studio-install`.

## Limitations

- Only as current as the branch or docs you read; unreleased models are not
  discoverable.
- Badges can lag for newly added models. Fall back to the model's class
  docstring.
- Compatibility checks need source access to be exact.

## Ownership and out-of-scope

**Owns:** component discovery, model and data-source compatibility checks,
hardware-fit assessment, example recommendations.

**Does not own:** installation (use `earth2studio-install`), writing inference
code, training, custom model development, runtime debugging, PhysicsNeMo model
discovery.
