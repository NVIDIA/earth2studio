---
name: earth2studio-create-assimilation
description: >
  Create, migrate, or review Earth2Studio data assimilation model wrappers and
  their unit tests, including independent analyses and explicit-state rollouts
  with DataFrame observations. Use for DA test alignment; not for fetching data
  or pure prognostic/diagnostic wrappers.
---

# Create an Assimilation Model

## Scope

Default to a protocol-compatible wrapper in the user's separate Python project.
Unless intent is already explicit, ask:
**"Are you planning to contribute this to NVIDIA Earth2Studio?"**
Only an affirmative answer or explicit contribution intent enables contributor steps.
For local wrappers, use the project's layout, dependencies, style and test conventions;
no Earth2Studio repository changes or full repository test suite are required.
Use `earth2studio-install` for dependency setup, including NATTEN and FlashAttention.
Repository paths below are source references or contributor destinations, not local output paths.
If unavailable locally, consult matching-version sources on GitHub or the documentation site.
Preserve existing licenses in copied code, including reference skeleton SPDX headers.
These samples do not impose NVIDIA copyright on newly authored local code.

Read the [local DA contract](references/da-contract.md) before coding; it holds
the slot, cadence, state, iterator, RNG and ownership requirements. Then read the
[protocols](../../earth2studio/models/da/base.py) (their docstrings are
normative), a
current wrapper of the same kind in `earth2studio/models/da/` (independent:
`interp.py`, `healda.py`, `sda_corrdiff_cosmo_era5.py`; stateful: `sda_stormcast.py`,
`sda_stormcastconus.py`) and, when migrating a 0.x wrapper, the data assimilation
section of [MIGRATION.md](../../MIGRATION.md). Prognostic-style state semantics
are in the [prognostic contract](../earth2studio-create-prognostic/references/model-contract.md).

## Workflow

1. Classify execution: independent analyses use `AssimilationModel`; evolving
   trajectories use `StatefulAssimilationModel`. A background input or a caller's
   loop over times does not imply statefulness, and independent models get no
   artificial iterator. Implement in the user's project; only contributions use
   `earth2studio/models/da/<name>.py` as the destination.
2. Record grids, variables, observation columns and time windows, which slots are
   optional, sources, normalization, sampling and checkpoints. Configure the
   domain and variable set on the instance before querying signatures.
3. Declare fixed named inputs and matching `input_coords`/`output_coords` slots.
   Use DataArray/CoordinateSystem for fields and DataFrame/FrameSchema for tables
   (empty column array = unconstrained, non-empty = allowed values). Preserve
   optional `None` slots. Only stateful models declare `forcing_coords`.
   Callers fetch data; `default_sources` follows input-then-forcing order and
   models never fetch. Document each observation window and any required
   DataFrame `attrs`; drivers request DataFrame forcing at the analysis valid
   time and pass `None` when no rows return.
   Stateful gridded outputs carry `lead_time`; output offsets are added to the
   final input lead time. The step interval is the output's last lead minus the
   later of the first input's last lead and zero, so a background ending before
   zero (brought up to the analysis time by `initialize`) does not lengthen it
   and publishes no initial condition. Several output leads per step are
   declared together and returned together.
4. Implement `__call__`, device movement, `stochastic` and `set_rng(seed, reset=True)`.
   Deterministic setters are no-ops; stochastic models keep a local generator that
   each call or new trajectory advances, `reset=False` creates it only if absent,
   and global RNG state is left unchanged.
   For stateful models add `initialize`, `step`, and `create_iterator`.
   `initialize` returns `(first_analysis, state)` after one core computation and
   keeps static forcing in the state. `step(*outputs, *new_forcing, state)` takes
   the previous outputs, new dynamic forcing only and `state` as a
   positional-or-keyword parameter after the arrays. `(output, state)` is the whole
   continuation: serializable (RNG as counter or snapshot), never mutated,
   replayable bit for bit including stochastic draws. `__call__` equals
   `initialize(...)[0]`. The iterator yields computed analyses only (no initial
   state or priming `None`); `send` supplies the next forcing in declared order,
   required conditioning is never reused (reject `send(None)`), observations
   may be `None`. Hooks, if offered, are iterator-only.
5. Reuse existing numerical cores and package loaders where appropriate. Guard
   optional dependencies and use `earth2studio-install` for setup. Follow the
   local project's packaging; only contributors propose repository extras changes.
6. For local wrappers, run a small numerical analysis and relevant slot, ownership,
   invalid-input and RNG checks; stateful wrappers also check step/iterator agreement
   and replay. Use the project's test layout. There is no DA conformance checker.
   For contributions, follow the full [test structure](references/testing-guide.md).
   Extend native model tests, consolidate overlapping cases, and retain numerical observation checks.
   Do not create a separate base-protocol test file or a bespoke conformance framework.
7. Contributor only: export the wrapper in `earth2studio/models/da/__init__.py`,
   add it to `docs/modules/models_da.md` alphabetically and update `CHANGELOG.md`.
   For optional extras, update `docs/userguide/about/install_options.yml`. Verify with
   `uv run` or the project virtualenv; report dependency skips and untested real weights.

Contributor verification commands (add the new component's test suite and run
relevant regression tests as well):

```bash
uv run pytest test/models/da/test_da_<model>.py -m "not package" -v
make format && make lint && make license
```

## Contributor Checklist

Only for confirmed contributions, follow [CONTRIBUTING.md](../../CONTRIBUTING.md)
and the [developer guide](../../docs/userguide/developer/overview.md):

- Register approved extras, exports, API/catalog entries and changelog updates.
- Apply repository style and SPDX requirements; preserve copied licenses.
- Add and run component tests and relevant regressions, plus `make format`, `make lint`,
  `make license`; report unavailable dependencies and skipped coverage.
- Follow contribution/PR guidance only when requested; commit, push and open a PR
  only when explicitly requested.

## Using the wrapper elsewhere

- Independent models are called directly in a caller-owned loop. They are not
  `CoupledRunner` components or `PrognosticRunner` models.
- A `StatefulAssimilationModel` can be a `CoupledRunner` component (see the
  `earth2studio-couple-models` skill) when its first input is gridded with lead
  times, its outputs are gridded with a `variable` dimension and share lead
  times, and its observation slots accept `None`. The coupler calls
  `initialize`/`step`, not the iterator.

For contributions, use SPDX headers in newly authored repository code, typed public
methods, NumPy-style docstrings, and `loguru` in library code. Follow the repository
testing conventions; keep this skill and its contract and test guides aligned with
`base.py` when the protocol changes.
