---
name: earth2studio-create-assimilation
description: >
  Create, migrate, or review Earth2Studio data assimilation model wrappers and
  their unit tests, including independent analyses and explicit-state rollouts
  with DataFrame observations. Use for DA test alignment; not for fetching data
  or pure prognostic/diagnostic wrappers.
---

# Create an Assimilation Model

Read the [DA contract](../../dev/spec/DA_MODEL_CONTRACT_SPEC.md),
[protocols](../../earth2studio/models/da/base.py), and
[implementation rules](../../.agents/rules/e2s-013-assimilation-models.mdc).

## Workflow

1. Classify execution: independent analyses use `AssimilationModel`; evolving
   trajectories use `StatefulAssimilationModel`. A background input or caller's
   loop over times does not imply statefulness.
2. Record grids, variables, observation columns/windows, optional slots, sources,
   normalization, sampling and checkpoints. Configure the domain before planning.
3. Declare fixed named inputs and matching `input_coords`/`output_coords` slots.
   Use DataArray/CoordinateSystem for fields and DataFrame/FrameSchema for tables.
   Preserve optional `None` slots. Only stateful models declare `forcing_coords`.
   Callers fetch data; `default_sources` follows input-then-forcing order.
4. Implement `__call__`, device movement, `stochastic` and `set_rng(seed, reset=True)`.
   Deterministic setters are no-ops; seeded sampling advances an isolated stream.
   For stateful models add `initialize`, `step`, and `create_iterator`: first output
   is computed, state is serializable/replayable, and hooks are iterator-only.
5. Reuse existing numerical cores and package loaders where appropriate. Guard
   optional dependencies; propose dependency changes before editing extras.
6. Follow the [test structure](references/testing-guide.md). Extend native model
   tests, consolidate overlapping cases, and retain numerical observation checks.
   Do not create a separate base-protocol test file or a bespoke conformance framework.
7. Export the wrapper and update API/install documentation. Verify with `uv run`
   or the project virtualenv; report dependency skips and untested real weights.

```bash
uv run pytest test/models/da/test_da_<model>.py -m "not package" -v
make format && make lint && make license
```

Use SPDX headers, typed public methods, NumPy-style docstrings, and `loguru` in
library code. Follow the repository testing rules; keep this skill and its test
guide aligned when the protocol changes.
