---
name: earth2studio-create-diagnostic
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team <agent-skills@nvidia.com>
  tags: [earth2studio, diagnostic-model, python]
description: >
  Use when creating or migrating Earth2Studio diagnostic model wrappers,
  including derived quantities, packaged models, generative diagnostics, and
  multi-input or multi-output transformations. Not for prognostic forecasts,
  data sources, or installation.
argument-hint: URL or local path to reference inference script (optional)
---

# Create a Diagnostic Model

Implement the [DiagnosticModel protocol](../../earth2studio/models/dx/base.py).
Read [the local diagnostic contract](references/model-contract.md) before coding;
it contains the coordinate, slot, ownership, source and RNG requirements.
Follow every rule in its [D1–D11 checklist](references/model-contract.md#rules-and-verification).
Diagnostics transform fields without a rollout, forcing API or iterator hooks.

## Workflow

1. **Understand the reference.** Use the supplied script, repository or paper;
   ask for one if absent. Record variables, grids, units/statistics, input/output
   slots, core shapes, checkpoints and sampling behavior. Configure domains and
   variables before querying signatures.
2. **Choose packaging.** Simple derived diagnostics usually need no dependency
   extra. Packaged/generative models use `AutoModelMixin` and a named optional
   extra, even if empty. Propose dependencies before editing `pyproject.toml`;
   add approved extras alphabetically and include them in `all`.
3. **Implement** `earth2studio/models/dx/<name>.py` using the
   [runnable skeleton](references/skeleton-template.py) and, as needed,
   [method examples](references/method-templates.py). Declare fixed named inputs
   in `__call__`, one DataArray per slot. Implement `input_coords`,
   `output_coords`, `default_sources` and `to` (normally inherited from
   `torch.nn.Module`), plus a boolean `stochastic`. Protocol inheritance is optional.
4. **Handle the core boundary.** Plan and validate coordinates without allocating
   fields. Use `.e2s.to_torch()`/`from_torch()` for Torch conversion. Pack leading
   axes only where needed; clone borrowed storage before in-place kernels.
   Declare sample axes and changed grids for generative outputs. Use
   `set_rng(seed, reset=True)` for stochastic models, with isolated RNG draws.
5. **Load weights when applicable.** Resolve immutable `Package` assets, load on
   CPU, call `eval()`, register buffers, and guard actual optional dependencies.
   Override `to` for non-Torch state. Follow
   [packaging guidance](references/model-contract.md#packaging).
6. **Test** in `test/models/dx/test_<name>.py` using
   [testing patterns](references/testing-guide.py). Cover numerical results,
   coordinate errors, leading axes, input ownership, fixed signatures, source
   slots and conformance. Include the sample `test_model_conformance` using
   `check_diagnostic_contract(model)` for both single- and multi-slot models;
   assert the exact expected skips. Retain the sample slot test for per-output
   numerical assertions. Add seeded sampling cases when relevant.
   Mock tests need no downloads; real-weight tests use `@pytest.mark.package`.
7. **Integrate public models.** Add alphabetical exports in
   `earth2studio/models/dx/__init__.py` and API entries in
   `docs/modules/models_dx.md`; update `CHANGELOG.md`. For extras, update
   `docs/userguide/about/install_options.yml` including install notes and `api_refs`.
8. **Verify** with the commands below. Report exact failures or skipped coverage;
   do not claim unrun checks passed.

```bash
uv run pytest test/models/dx/test_<name>.py -m "not package" -v
# When real-weight validation is requested and dependencies are available:
uv run pytest test/models/dx/test_<name>.py -m package --package -v
make format && make lint && make license
```

Use `uv run` or the project virtualenv for Python. Include SPDX headers, typed
public methods and NumPy-style docstrings; use `loguru.logger` in library code.

## Reference Map

| Reference | Read when |
| --- | --- |
| [Model contract](references/model-contract.md) | Always, before implementation |
| [Skeleton](references/skeleton-template.py) | Starting a single-input wrapper |
| [Method examples](references/method-templates.py) | Packaging or multiple slots |
| [Testing guide](references/testing-guide.py) | Building mock/contract tests |
| [Validation guide](references/validation-guide.md) | Comparing with upstream inference |
| [PR body](references/pr-body-template.md), [validation comment](references/pr-comment-template.md) | A PR is requested |

## Common Mistakes

- Copying the protocol's `*x` into a wrapper: use fixed named parameters instead.
- Passing `(a, b)` to execution: use `model(a, b)`; coordinate planning still
  accepts `model.output_coords((a, b))`.
- Omitting `default_sources()`: return `None` when there is no recommendation.
- Treating a zero-length spatial axis as a wildcard: configure a concrete grid.
- Forbidding temporal axes: diagnostics may consume or preserve time/lead time
  when needed; this does not make them prognostic models.
- Seeding constructors or resetting on every call: expose `set_rng` and let the
  stream advance between calls.

For local work, use the checkout containing `pyproject.toml`. Harbor output goes
under `/workspace/output/earth2studio/models/dx/`; a copied repo is at
`/workspace/repo`. Never read `evals/targets/` (grader-only material). Keep
validation scripts, checkpoints, images and credentials out of commits. If a
reference here produces an incorrect wrapper, fix that guidance and verify the
correction before continuing.
