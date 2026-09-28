# Configurable Array Backends Implementation Plan

> Execute inline using the executing-plans skill. Design:
> `dev/scratch/2026-09-28-array-backends-design.md`.

**Goal:** Preserve graphs through existing conversion calls with configurable
xarray payload backends, warning on gradient-dropping exports.

**Architecture:** Import-time environment default plus ContextVar scope and
per-call policy; private Torch duck array; shared conversion/batching helpers.
**Stack:** Python, NumPy, CuPy, Torch, xarray, pytest.

## Completed work

- [x] Establish `.venv` with uv; baseline conversion tests: 3 passed, including CUDA.
- [x] Implement backend validation, precedence, nested scopes and isolation.
  Test environment defaults in-process with monkeypatch.
- [x] Preserve non-leaf identity/history and tracking semantics; centralize
  warning-and-detach exports, explicit devices and metadata-preserving conversion.
- [x] Implement bounded Torch array dispatch, differentiable xarray operations
  and explicit rejection of unsupported operations/implicit NumPy conversion.
- [x] Extend shared batching with Torch views/copies; verify an unchanged
  two-component conversion chain with zero, one and multiple leading dimensions.
- [x] Verify CPU/CUDA transfers, NumPy/CuPy exports and fresh-leaf reimports.
- [x] Self-review: fix empty-axis NaN sums, integer reduction dtype handling and
  scalar-left comparisons. Regression cases failed before fixes.
- [x] Consolidate dispatch argument validation and parameterized operation tests;
  share value/gradient assertions. Retain positional-option rejection coverage.
- [x] Shorten the design and replace stale plan status with verified results.

Production file: `earth2studio/utils/cupy.py`. Tests: `test/utils/test_cupy.py`.
Publication targets upstream with base `ngeneva/projected-grid-followup` (#1181),
title `[E2S 1.0] Add gradient-preserving Torch xarray backend` and label `1.0.0`.
No model edits. Known compatibility gap:
`handshake_device` rejects Torch payloads; see the design's supported limits.

## Verification commands

Run from the worktree using its virtualenv:

```bash
.venv/bin/python -m pytest test/utils/test_cupy.py test/utils/test_coords.py \
  test/models/test_batch_xarray.py test/models/test_batch.py test/grids -q --tb=short
.venv/bin/python -m black --check --target-version py312 earth2studio/utils/cupy.py test/utils/test_cupy.py
.venv/bin/python -m ruff check earth2studio/utils/cupy.py test/utils/test_cupy.py
.venv/bin/python -m mypy --follow-imports=silent earth2studio/utils/cupy.py
git diff --check
```

Review results: **170 passed**, including CUDA; **96% statement coverage** of
`cupy.py` (339/354 statements). Full-repository `make format` and `make lint`
passed using `UV_NO_SYNC=1` with the provisioned virtualenv; focused Black,
Ruff, mypy and interrogate checks also passed.
Existing pytest configuration, NumPy timedelta and Torch indexing warnings
remain. Full repository/model test suites and older dependency versions were not run.
