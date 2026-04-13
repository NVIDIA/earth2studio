# Assimilation Model Skills Design

## Goal

Create two Claude skills for Earth2Studio data assimilation (DA) models:

1. `create-assimilation-wrapper` — guide creation of a new DA model wrapper
2. `validate-assimilation-wrapper` — guide validation, PR, and code review

These follow the established create+validate pattern from the diagnostic (dx) and
prognostic (px) skill pairs, but with **fundamental I/O differences** that make DA
models distinct.

## Critical Differences from px/dx Models

DA models are **not** tensor-in / tensor-out like px and dx models. The entire I/O
contract is different:

| Aspect | px / dx | da |
| ------------------- | ------------------------------ | ---------------------------------------- |
| **Primary input** | `Tensor` + `CoordSystem` | `pd.DataFrame` or `xr.DataArray` |
| **Primary output** | `Tensor` + `CoordSystem` | `xr.DataArray` or `pd.DataFrame` |
| **Coord schemas** | `CoordSystem` only | `FrameSchema` + `CoordSystem` |
| **Coord returns** | Single dict | **Tuple** (even for single) |
| **Batch handling** | `@batch_func` / `@batch_coords` | **Neither** — N/A for DataFrame I/O |
| **PrognosticMixin** | Yes (px) / No (dx) | **No** |
| **Time integration** | `create_iterator` yields | `create_generator` with **send** |
| **Generator prime** | Yields initial condition | Yields `None` / state, `.send()` |
| **Generator cleanup** | N/A | Must handle `GeneratorExit` |
| **Init data** | No | Optional via `init_coords()` |
| **Time metadata** | Coordinate dimension | `obs.attrs["request_time"]` |
| **GPU data** | `Tensor` on device | **cupy** arrays, **cudf** DFs |
| **Input validation** | `handshake_dim`/`handshake_coords` | `validate_observation_fields()` |
| **Time filtering** | N/A | `filter_time_range()` |
| **Tensor conversion** | Already tensors | `dfseries_to_torch()` |
| **Device tracking** | `device_buffer` / `parameters()` | `device_buffer` + `@property` |

These differences pervade every step of the skill — skeleton, coordinates, forward
pass, generator, testing, and validation.

## Approach

Single template for the general **stateful** case, with clear notes on simplifications
for **stateless** models (those returning `None` from `init_coords()`). No NN/physics
branching — DA models are all NN-backed or algorithmic with external packages.

Reference input: user provides reference script/repo URL or local path (same as px
skill).

---

## Skill 1: `create-assimilation-wrapper`

### Frontmatter (create)

```yaml
---
name: create-assimilation-wrapper
description: >-
  Create a new Earth2Studio data assimilation model (da) wrapper from a
  reference inference script or repository. DA models ingest sparse
  observations (DataFrames) and/or gridded state arrays (DataArrays) and
  produce analysis output — they do NOT use @batch_func, @batch_coords,
  or PrognosticMixin.
argument-hint: URL or local path to reference inference script/repo
---
```

### Steps 0-8

The create skill handles **implementation only** — skeleton, coordinates, forward
pass, model loading, registration, and code quality. All testing, validation,
comparison scripts, and PR submission live in the validate skill.

#### Step 0 — Obtain Reference Script

- Accept `$ARGUMENTS` or ask user for reference
- Analyze: detect input types (DataFrame? DataArray?), output types, whether
  stateful (needs init data) or stateless, dependency packages, model architecture
- Determine if `@torch.inference_mode()` is safe (no gradient flow needed)
- Present summary to user

No NN/physics branching — all DA models get the same flow.

**[CONFIRM — Model Analysis]**

#### Step 1 — Examine Reference and Propose Dependencies

- Identify Python packages (physicsnemo, scipy, healpy, cudf, cupy, etc.)
- Propose pyproject.toml dependency group named `da-<model-name>`
- Highlight cudf/cupy as optional GPU acceleration packages

**[CONFIRM — Dependencies]**

#### Step 2 — Add Dependencies to pyproject.toml

#### Step 3 — Create Skeleton Class File

Dual inheritance: `torch.nn.Module + AutoModelMixin` (NO PrognosticMixin).
Class-level `@check_optional_dependencies()` decorator.

**Canonical DA method ordering:**

1. `__init__` — register `device_buffer`, store model params, normalize tolerance
2. `device` property — `return self.device_buffer.device`
3. `init_coords` — `None` for stateless, tuple for stateful
4. `input_coords` — tuple of `FrameSchema` (DF) or `CoordSystem` (DA)
5. `output_coords` — accept `input_coords` tuple + `request_time`, return tuple
6. `load_default_package` — classmethod
7. `load_model` — classmethod with `@check_optional_dependencies()`
8. `to` — device management, return `AssimilationModel`
9. Private/support methods (e.g., `_interpolate`, `_forward`, spatial lookups)
10. `__call__` — stateless forward, accept `*args: pd.DataFrame | xr.DataArray | None`
11. `create_generator` — bidirectional generator with send protocol

**DA-specific skeleton elements:**

- `FrameSchema` for observation inputs (time, lat, lon, observation, variable)
- `CoordSystem` for gridded outputs
- `request_time` from `obs.attrs`, NOT a coordinate dimension
- `validate_observation_fields()` call in `__call__`
- `filter_time_range()` for time-window filtering
- `dfseries_to_torch()` for zero-copy DataFrame→tensor
- cupy/cudf support: `cp.asarray()` for GPU output, `.cpu().numpy()` for CPU
- Generator: `yield None` to prime, `observations = yield result`, handle `GeneratorExit`
- `@torch.inference_mode()` unless gradient flow is required (document reason if omitted)
- No `@batch_func()`, no `@batch_coords()`, no `PrognosticMixin`

**[CONFIRM — Skeleton]**

#### Step 4 — Implement Coordinate System

Key differences from px/dx:

- `init_coords()` returns `None` for stateless models or tuple for stateful
- `input_coords()` returns **tuple** of `FrameSchema` — one per `__call__` arg
- `FrameSchema` keys are DF column names (time, lat, lon, observation, variable)
- `output_coords()` accepts tuple + `request_time`, returns tuple of `CoordSystem`
- Use `handshake_dim`/`handshake_coords`/`handshake_size` only for `CoordSystem`
- Use `validate_observation_fields()` for `FrameSchema` inputs

**[CONFIRM — Coordinates]**

#### Step 5 — Implement Forward Pass

Two methods:

- `__call__`: Extract `request_time` from `obs.attrs`, validate with
  `validate_observation_fields()`, filter with `filter_time_range()`, convert with
  `dfseries_to_torch()`, run model, build `xr.DataArray` output with cupy/numpy
- `create_generator`: Prime with `yield None` (stateless) or `yield initial_state`
  (stateful), loop with `observations = yield result`, handle `GeneratorExit`

**[CONFIRM — Forward Pass]**

**Step 6 — Implement Model Loading**
`load_default_package`, `load_model` with `@check_optional_dependencies()`

**[CONFIRM — Model Loading]**

**Step 7 — Register in `__init__.py`**
Add to `earth2studio/models/da/__init__.py`

**Step 8 — Verify Style/Format/Lint**
`make format`, `make lint`, `make license`

The create skill ends here. All testing, validation, comparison, and PR work is
handled by the `validate-assimilation-wrapper` skill.

**Reminders section** — DA-specific DO/DON'T:

- DO return tuples from `input_coords` and `output_coords`
- DO use `FrameSchema` for tabular inputs, `CoordSystem` for gridded outputs
- DO validate `request_time` from `obs.attrs`
- DO use `validate_observation_fields()`, `filter_time_range()`, `dfseries_to_torch()`
- DO prime generator with `yield None` and handle `GeneratorExit`
- DO return cupy arrays on GPU, numpy on CPU
- DO register `device_buffer` and expose `device` property
- DO NOT use `@batch_func()` or `@batch_coords()`
- DO NOT use `PrognosticMixin`
- DO NOT use `create_iterator` — DA uses `create_generator` with send protocol
- DO NOT assume tensor inputs — inputs are DataFrames/DataArrays
- DO NOT forget cudf/cupy optional import pattern

---

## Skill 2: `validate-assimilation-wrapper`

### Frontmatter (validate)

```yaml
---
name: validate-assimilation-wrapper
description: >-
  Validate a newly created Earth2Studio data assimilation model wrapper by
  writing unit tests (90% coverage required), performing reference comparison
  with DataFrame/DataArray outputs, generating sanity-check plots, and
  opening a PR with automated code review. Use after completing
  create-assimilation-wrapper Steps 0-8.
argument-hint: Name of the DA model class and test file
---
```

### Steps 1-6

#### Step 1 — Write Pytest Unit Tests

DA-specific test patterns:

- PhooModelName dummy that returns known output shapes
- **Parametrize over pandas AND cudf** (skip cudf if unavailable)
- **Parametrize over CPU AND GPU** devices
- Test `__call__` with DataFrame input, verify `xr.DataArray` output
- Test generator protocol: prime → send → close
- Test `init_coords` returns correct type (None or tuple)
- Test time tolerance filtering
- Test empty DataFrame handling
- Test invalid `obs.attrs` (missing `request_time`)
- Test `validate_observation_fields` raises on bad columns
- `@pytest.mark.package` integration test

**[CONFIRM — Package Test]** (for `@pytest.mark.package` integration test only)

#### Step 2 — Run Tests and Achieve 90% Coverage

- Run test file with `-v --timeout=60`, all must pass
- Coverage with `--slow`, `--cov`, **`--cov-fail-under=90`** — the new DA model
  file must achieve **at least 90% line coverage**
- DA-specific coverage gaps to watch: generator cleanup path (`GeneratorExit`),
  cudf code paths, empty DataFrame handling, time tolerance edge cases,
  `obs.attrs` validation branches, cupy vs numpy output paths
- If coverage is below 90%, add tests to cover missing lines and re-run

#### Step 3 — Reference Comparison and Sanity-Check

3a. Reference comparison — compare `__call__` output AND multi-step generator
output against reference implementation. For DataArray output: compute tolerance
metrics (max abs diff, correlation, allclose). For DataFrame output: compare row
counts, value ranges, spatial coverage.

3b. Model summary table — input schema, output grid, variables, stateful/stateless,
observation types, cudf support.

3c. Three sanity-check plot templates:

1. **Spatial assimilated output** — contourf of gridded DataArray output (like dx
   spatial plot)
2. **Observation overlay** — scatter of input DataFrame observations overlaid on
   assimilated grid output (unique to DA — shows sparse→dense mapping)
3. **Generator sequence** — multi-step assimilation evolution over time (for
   stateful models) or repeated independent calls (for stateless)

3d. Side-by-side comparison scripts — reference inference vs Earth2Studio equivalent

3e-3f. Run scripts, user confirms plots.

**[CONFIRM — Sanity-Check & Comparison]**

#### Step 4 — Branch, Commit and Open PR

DA-specific PR template fields:

- Model type: Stateless / Stateful
- Input format: DataFrame / DataArray / Mixed
- Output format: DataArray / DataFrame
- Observation schema: columns and types
- Grid specification: lat-lon / HRRR / HealPix / etc.
- Time tolerance: default value
- cudf/cupy support: Yes / No
- Reference comparison metrics table

**[CONFIRM — Ready to Submit]**

**Step 5 — Automated Code Review (Greptile)**
Same polling/triage/fix pattern as dx/px.

**[CONFIRM — Review Triage]**

**Reminders** — same DA-specific rules as create skill.

---

## Reference Files

| File | Purpose |
| --------------------------------------------------- | ------------------------------------------ |
| `.cursor/rules/e2s-013-assimilation-models.mdc` | Authoritative DA rules (577 lines) |
| `earth2studio/models/da/base.py` | AssimilationModel Protocol definition |
| `earth2studio/models/da/interp.py` | Simple stateless DA example (593 lines) |
| `earth2studio/models/da/sda_stormcast.py` | Complex stateful DA example (919 lines) |
| `earth2studio/models/da/utils.py` | DA utilities (validate, filter, convert) |
| `test/models/da/test_da_interp.py` | DA test patterns (359 lines) |
| `test/models/da/test_da_healda.py` | Complex DA test patterns with mocks |

## CONFIRM Gates Summary

### Create skill (6 gates)

1. `[CONFIRM — Model Analysis]`
2. `[CONFIRM — Dependencies]`
3. `[CONFIRM — Skeleton]`
4. `[CONFIRM — Coordinates]`
5. `[CONFIRM — Forward Pass]`
6. `[CONFIRM — Model Loading]`

### Validate skill (4 gates)

1. `[CONFIRM — Package Test]`
2. `[CONFIRM — Sanity-Check & Comparison]`
3. `[CONFIRM — Ready to Submit]`
4. `[CONFIRM — Review Triage]`
