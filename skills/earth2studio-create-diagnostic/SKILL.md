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

Implement the [DiagnosticModel protocol](../../earth2studio/models/dx/base.py).
Read [E2S-06: Model Contract][model-spec] before coding, including its diagnostic
rules and conformance sections. The selected version's spec is authoritative;
use its requirements directly rather than maintaining a separate skill contract.
Diagnostics transform fields without a rollout, forcing API or iterator hooks.

## Workflow

1. **Understand the reference.** Use the supplied script, repository, paper or
   specification; ask for missing details only when needed. Record variables, grids,
   units/statistics, input/output slots, core shapes, checkpoints and sampling behavior. Configure domains and
   variables before querying signatures.
   Unless already specified, ask: “Do you want checkpoint/restart integration? I recommend no for the
   initial implementation because it adds storage, restore and testing complexity.”
   Add it only if requested; consult the selected version's package and execution APIs.
2. **Choose packaging.** Use the local project's dependency conventions and
   `earth2studio-install` for setup. `AutoModelMixin` supports packaged weights.
   For contributions only, packaged/generative models get a named optional extra,
   even if empty; simple derived diagnostics usually need none. Confirm changes
   to the repository's `pyproject.toml`; add extras alphabetically and include them in `all`.
3. **Implement** in the user's project, or for contributions in
   `earth2studio/models/dx/<name>.py`, using the
   [runnable skeleton](references/skeleton-template.py) and, as needed,
   [method examples](references/method-templates.py). Follow the spec's
   [diagnostic interface][diagnostic-spec] for methods and slots.
4. **Handle the core boundary** using the spec's coordinate, execution-boundary,
   ownership and stochasticity requirements. Consult [domain selection][domain-spec]
   for models that support sub-domains.
5. **Load weights when applicable.** Resolve immutable `Package` assets, load on
   CPU, call `eval()`, register buffers, and guard actual optional dependencies.
   Override `to` for non-Torch state. Use the packaging examples in the templates.
6. **Validate locally** with a small numerical smoke case and
   `check_diagnostic_contract(model)` using mock weights; check relevant sampling
   behavior and report skipped checks. Follow the project's test layout.
   **For contributions**, use `test/models/dx/test_<name>.py` and the full
   [testing patterns](references/testing-guide.py). Cover numerical results,
   coordinate errors, leading axes, input ownership, fixed signatures, source
   slots and conformance. Include the sample `test_model_conformance` using
   `check_diagnostic_contract(model)` for both single- and multi-slot models;
   assert the exact expected skips. Add per-output numerical assertions and
   seeded sampling cases when relevant.
   Mock tests need no downloads; real-weight tests use `@pytest.mark.package`.
7. **Contributor only: integrate public models.** Add alphabetical exports in
   `earth2studio/models/dx/__init__.py` and API entries in
   `docs/modules/models_dx.md`; update `CHANGELOG.md`. For extras, update
   `docs/userguide/about/install_options.yml` including install notes and `api_refs`.
8. **Verify** local wrappers with the project's relevant smoke/contract checks.
   For contributions, run the new component's test suite, relevant regression tests
   and the repository checks below. Report exact failures or skipped coverage;
   do not claim unrun checks passed.

```bash
uv run pytest test/models/dx/test_<name>.py -m "not package" -v
# When real-weight validation is requested and dependencies are available:
uv run pytest test/models/dx/test_<name>.py -m package --package -v
make format && make lint && make license
```

Use `uv run` or the project virtualenv for Python. For contributions, include
SPDX headers in newly authored repository code, typed public methods and
NumPy-style docstrings; use `loguru.logger` in library code.
Use the complete docstrings in the templates. Describe the actual variables,
grids, sample axes and output slots rather than a generic "forward pass."

## Contributor Checklist

Only for confirmed contributions, follow [CONTRIBUTING.md](../../CONTRIBUTING.md)
and the [developer guide](../../docs/userguide/developer/overview.md):

- Register approved extras, exports, API/catalog entries and changelog updates.
- Apply repository style and SPDX requirements; preserve copied licenses.
- Add and run component tests and relevant regressions, plus `make format`, `make lint`,
  `make license`; report unavailable dependencies and skipped coverage.
- Use the PR references when submission is requested. Commit, push and open a PR
  only when explicitly requested.

## Reference Map

| Reference | Read when |
| --- | --- |
| [Model contract][model-spec] | Always, before implementation |
| [Skeleton](references/skeleton-template.py) | Starting a single-input wrapper |
| [Method examples](references/method-templates.py) | Packaging or multiple slots |
| [Regional views][domain-spec] | The model supports sub-domains |
| [Testing guide](references/testing-guide.py) | Building mock/contract tests |
| [Validation guide](references/validation-guide.md) | Comparing with upstream inference |
| [PR body](references/pr-body-template.md), [validation comment](references/pr-comment-template.md) | A PR is requested |

## Common Mistakes

- Copying the protocol's `*x` into a wrapper: use fixed named parameters instead.
- Passing `(a, b)` as one argument: use `model(a, b)` and
  `model.output_coords(a, b)` for separate input slots.
- Omitting `default_sources()`: return `None` when there is no recommendation.
- Treating a zero-length spatial axis as a wildcard: configure a concrete grid.
- Forbidding temporal axes: diagnostics may consume or preserve time/lead time
  when needed; this does not make them prognostic models.
- Seeding constructors or resetting on every call: expose `set_rng` and let the
  stream advance between calls.

For local work, use the user's project root. For contributor Harbor tasks only,
output goes under `/workspace/output/earth2studio/models/dx/`; a copied repo is at
`/workspace/repo`. Never read `evals/targets/` (grader-only material). Keep
validation scripts, checkpoints, images and credentials out of contribution commits.
If a template or implementation conflicts with the selected version's spec, report
the discrepancy; do not redefine the contract in this skill.

[model-spec]: ../../docs/specs/E2S-06_MODEL_CONTRACT_SPEC.md
[diagnostic-spec]: ../../docs/specs/E2S-06_MODEL_CONTRACT_SPEC.md#diagnostic-models
[domain-spec]: ../../docs/specs/E2S-06_MODEL_CONTRACT_SPEC.md#domain-selection
