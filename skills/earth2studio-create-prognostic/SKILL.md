---
name: earth2studio-create-prognostic
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team <agent-skills@nvidia.com>
  tags: [earth2studio, prognostic-model, python]
description: >
  Use when creating or migrating Earth2Studio prognostic time-stepping model
  wrappers, including explicit-state, forced, coupled, stochastic and multi-output
  forecasts. Not for diagnostic models, data sources, or installation.
argument-hint: URL or local path to reference inference script (optional)
---

# Create a Prognostic Model

Implement the [PrognosticModel protocol](../../earth2studio/models/px/base.py).
Read [the local prognostic contract](references/model-contract.md) before coding;
it specifies slots, state, forcing, forecasts-only iteration, hooks and RNG.
Follow every rule in its [P1–P24 checklist](references/model-contract.md#rules-and-verification).
Use [PrognosticMixin](../../earth2studio/models/px/utils.py) optionally: its public
execution methods are stubs; explicit wrapper methods can delegate to its private
helpers.

## Workflow

1. **Understand the reference.** Use the supplied script/repository/paper; ask if
   absent. Record inputs, history windows, output chunks, grids, statistics,
   forcing, statics, latent state, randomness, core shapes and checkpoints.
   Configure domains and variables before querying signatures.
2. **Propose dependencies.** Every packaged prognostic gets a named optional
   extra, even if empty. Confirm dependencies before editing `pyproject.toml`;
   add approved extras alphabetically and include them in `all`.
3. **Implement** `earth2studio/models/px/<name>.py` using the
   [single-frame skeleton](references/skeleton-template.py) and
   [history methods](references/method-templates.py). `torch.nn.Module` supplies
   device movement; `AutoModelMixin` is for packaged weights. Protocol compliance,
   not triple inheritance, is required.
4. **Declare slots and sources.** Implement `input_coords`, `output_coords`,
   `forcing_coords` and `default_sources`. The mixin supplies the latter two as
   `None`. Planning is allocation-free and validates finite relative history.
   Pass fields, then forcing, as individual positional DataArrays.
5. **Implement explicit execution methods.** Declare fixed named parameters for
   `__call__`, `initialize`, `step` and `create_iterator`; `state` follows the
   arrays as a positional-or-keyword parameter. `initialize` computes the first
   forecast. `step` consumes previous outputs plus new forcing and serializable
   state, returning the next `(y, state)` without modifying its inputs. Use
   `_default_call` and `_default_create_iterator` for thin wrapper delegates.
   Each iterator yield is a complete forecast, including all core-produced leads.
6. **Handle weights and RNG.** Resolve immutable `Package` assets, load on CPU,
   call `eval()`, register buffers and guard optional backend dependencies.
   Override `to` for non-Torch state. Stochastic models implement `set_rng` and
   retain per-rollout RNG position in explicit state for replay.
7. **Test** in `test/models/px/test_<name>.py` using
   [testing patterns](references/testing-guide.py). Cover numerical values,
   initialization/step/iterator equivalence, replay, hooks, ownership, metadata,
   invalid coordinates and forcing, slots and seeding. The current conformance
    checker still expects legacy iteration; include the sample
    `test_model_conformance` using `check_prognostic_contract(model, rollout=False)`
    and retain the direct iterator, replay and signature tests for rules it does
    not yet cover. Assert the exact expected skips. Mock tests require no downloads;
   real-weight tests use `@pytest.mark.package`.
8. **Integrate public models.** Add alphabetical exports in
   `earth2studio/models/px/__init__.py`, API entries in
   `docs/modules/models_px.md`, and `CHANGELOG.md`. For extras, update
   `docs/userguide/about/install_options.yml` with install notes and `api_refs`.
9. **Verify** using the commands below. Report exact failures and skipped
   coverage; do not claim unrun checks passed.

```bash
uv run pytest test/models/px/test_<name>.py -m "not package" -v
# When real-weight validation is requested and dependencies are available:
uv run pytest test/models/px/test_<name>.py -m package --package -v
make format && make lint && make license
```

Use `uv run` or the project virtualenv for Python. Include SPDX headers, typed
public methods and NumPy-style docstrings; use `loguru.logger` in library code.

## Reference Map

| Reference | Read when |
| --- | --- |
| [Model contract](references/model-contract.md) | Always, before implementation |
| [Skeleton](references/skeleton-template.py) | Starting a single-frame wrapper |
| [Method examples](references/method-templates.py) | Retaining history in explicit state |
| [Forcing and slots](references/model-contract.md#forcing-and-slots) | Forced or coupled execution |
| [Testing guide](references/testing-guide.py) | Building mock/contract tests |
| [Validation guide](references/validation-guide.md) | Comparing with upstream inference |
| [PR body](references/pr-body-template.md), [validation comment](references/pr-comment-template.md) | A PR is requested |

## Common Mistakes

- Copying `*x`/`*y` from the protocol into wrappers: declare fixed named slots.
- Yielding the initial condition: first yield is the first forecast; drivers use
  `initial_condition(x)` separately when publishing the starting fields.
- Keeping history/latents in a generator frame: save everything needed alongside
  `y` in serializable state so a checkpoint can resume with `step`.
- Applying hooks inside `__call__`, `initialize` or `step`: hooks are iterator-only.
- Copying hook inputs: both hooks receive `y` directly and their returned values
  feed recurrence; in-place front hooks can edit a previously yielded array.
- Fetching conditioning internally: declare forcing and require caller-supplied
  data. Static forcing is supplied only at initialization.

For local work, use the checkout containing `pyproject.toml`. Harbor output goes
under `/workspace/output/earth2studio/models/px/`; a copied repo is at
`/workspace/repo`. Never read `evals/targets/` (grader-only material). Keep
validation scripts, checkpoints, images and credentials out of commits. If a
reference produces an incorrect wrapper, fix that guidance and verify it before
continuing.
