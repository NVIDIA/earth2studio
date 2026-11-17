---
name: earth2studio-create-prognostic
version: 1.0.0
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

Implement the [PrognosticModel protocol](../../earth2studio/models/px/base.py).
Read [E2S-06: Model Contract][model-spec] before coding, including its prognostic
rules and conformance sections. The selected version's spec is authoritative;
use its requirements directly rather than maintaining a separate skill contract.
Use [PrognosticMixin](../../earth2studio/models/px/utils.py) optionally: it supplies
no public execution methods; explicit wrapper methods can delegate to its private
helpers.

## Workflow

1. **Understand the reference.** Use the supplied script, repository, paper or
   specification; ask for missing details only when needed. Record inputs, history
   windows, output chunks, grids, statistics, forcing, statics, latent state,
   randomness, core shapes and checkpoints.
   Configure domains and variables before querying signatures.
   Unless already specified, ask: “Do you want checkpoint/restart integration? I recommend no for the
   initial implementation because it adds storage, restore and testing complexity.”
   Add it only if requested; see [explicit state][state-spec].
   Explicit `initialize`/`step` state remains required either way.
2. **Propose dependencies.** Use the local project's dependency conventions and
   `earth2studio-install` for setup. For contributions only, every packaged
   prognostic gets a named optional extra, even if empty. Confirm changes to the
   repository's `pyproject.toml`; add extras alphabetically and include them in `all`.
3. **Implement** in the user's project, or for contributions in
   `earth2studio/models/px/<name>.py`, using the
   [single-frame skeleton](references/skeleton-template.py) and
   [history methods](references/method-templates.py). `torch.nn.Module` supplies
   device movement; `AutoModelMixin` is for packaged weights. Protocol compliance,
   not triple inheritance, is required.
4. **Declare slots and sources** following the spec's [model interface][interface-spec]
   and coordinate requirements. Use the templates as implementation examples.
5. **Implement explicit execution methods** following the spec's
   [state and continuation requirements][state-spec], iteration and hooks sections.
   Check the selected version's `PrognosticMixin` before using its private helpers.
6. **Handle weights and RNG.** Resolve immutable `Package` assets, load on CPU,
   call `eval()`, register buffers and guard optional backend dependencies.
   Override `to` for non-Torch state. Follow the spec's stochasticity requirements
   and [domain selection][domain-spec] when applicable.
7. **Validate locally** with a small numerical smoke case and
   `check_prognostic_contract(model)` using mock weights; check replay and relevant
   forcing/RNG behavior and report skipped checks. Follow the project's test layout.
   **For contributions**, use `test/models/px/test_<name>.py` and the full
   [testing patterns](references/testing-guide.py). Cover numerical values,
   initialization/step/iterator equivalence, replay, hooks, ownership, metadata,
   invalid coordinates and forcing, slots and seeding. Include the sample
   `test_model_conformance` using `check_prognostic_contract(model)` for the new
   forecasts-only, explicit-state contract. Retain direct numerical tests and
   assert the exact expected skips. Mock tests require no downloads;
   real-weight tests use `@pytest.mark.package`.
8. **Contributor only: integrate public models.** Add alphabetical exports in
   `earth2studio/models/px/__init__.py`, API entries in
   `docs/modules/models_px.md`, and `CHANGELOG.md`. For extras, update
   `docs/userguide/about/install_options.yml` with install notes and `api_refs`.
9. **Verify** local wrappers with the project's relevant smoke/contract checks.
   For contributions, run the new component's test suite, relevant regression tests
   and the repository checks below. Report exact failures and skipped
   coverage; do not claim unrun checks passed.

```bash
uv run pytest test/models/px/test_<name>.py -m "not package" -v
# When real-weight validation is requested and dependencies are available:
uv run pytest test/models/px/test_<name>.py -m package --package -v
make format && make lint && make license
```

Use `uv run` or the project virtualenv for Python. For contributions, include
SPDX headers in newly authored repository code, typed public methods and
NumPy-style docstrings; use `loguru.logger` in library code.
Use the complete docstrings in the templates. Adapt descriptions to the model's
state, forcing windows and forecast chunk; avoid stale initial-condition-first
or single-frame wording from older wrappers.

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
| [Skeleton](references/skeleton-template.py) | Starting a single-frame wrapper |
| [Method examples](references/method-templates.py) | Retaining history in explicit state |
| [Forcing and slots][interface-spec] | Forced or coupled execution |
| [Regional views][domain-spec] | The model supports sub-domains |
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

For local work, use the user's project root. For contributor Harbor tasks only,
output goes under `/workspace/output/earth2studio/models/px/`; a copied repo is at
`/workspace/repo`. Never read `evals/targets/` (grader-only material). Keep
validation scripts, checkpoints, images and credentials out of contribution commits.
If a template or implementation conflicts with the selected version's spec, report
the discrepancy; do not redefine the contract in this skill.

[model-spec]: ../../docs/specs/E2S-06_MODEL_CONTRACT_SPEC.md
[interface-spec]: ../../docs/specs/E2S-06_MODEL_CONTRACT_SPEC.md#model-interface
[state-spec]: ../../docs/specs/E2S-06_MODEL_CONTRACT_SPEC.md#explicit-state
[domain-spec]: ../../docs/specs/E2S-06_MODEL_CONTRACT_SPEC.md#domain-selection
