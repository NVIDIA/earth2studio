# DA test structure

Use the same native-test organization as the DX/PX
[diagnostic](../../earth2studio-create-diagnostic/references/testing-guide.py) and
[prognostic](../../earth2studio-create-prognostic/references/testing-guide.py) guides.
Adapt existing fixtures, rather than copying entire suites or creating a test framework.

| Case | Required behavior |
| --- | --- |
| `test_<model>_coords` | Input schemas, output geometry/time, source order; forcing only for stateful models. Fold into an existing init case when sufficient. |
| `test_<model>_call` | Numerical analysis, metadata/device, input ownership; optional/empty observations and independent batches where supported. |
| `test_<model>_iter` | Stateful only: call/initialize/first-yield equality, two steps with fresh forcing, replay after reseeding, unchanged inputs/state. |
| `test_<model>_exceptions` | Representative invalid coordinates/columns; missing required inputs. Stateful cases include invalid sends/state. |
| `test_<model>_deterministic_seed` | Stochastic only: same seed reproduces, different seeds differ, stream advances, `reset=False` preserves it, global RNG unchanged. |
| `test_<model>_package` | Marked `package`; real weights/downloads only when enabled. Omit for un-packaged models. |

- These are behavior groups, not a requirement to add six new functions. Extend or
  rename existing cases. Keep one device matrix for each numerical execution path.
- Put observation-window, guidance and normalization assertions in call/iterator
  cases where possible. Separate regressions only when they exercise distinct logic
  (e.g. duplicate-observation reduction, halo cropping, sampler failure isolation).
- Preserve numerical expectations; shape-only tests cannot prove observations matter.
  Stochastic mocks must exercise the actual wrapper RNG path.
- Keep coordinate/protocol assertions with model tests. There is no DA conformance
  checker; do not use the DX/PX checkers or add classification-only suites.
- Reuse builders/fixtures for expensive setup. Inline single-use helpers; avoid
  generic dispatchers, signature-only tests and exhaustive parameter products.
- Prefer `test/models/da/test_da_<model>.py`. A thin wrapper may share the existing
  core's test module when this avoids duplicating substantial mock scaffolding.
- Do not erase domain-specific numerical regressions merely to hit a test-count
  target. Consolidate repeated setup and overlapping assertions instead.
