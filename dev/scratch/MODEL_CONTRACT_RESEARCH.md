# Model Contract Research Notes

**Status:** Non-normative background for the model contract work
**Last updated:** September 2026

## Purpose

This note records how other projects specify and enforce a model interface, and what that
survey implies for Earth2Studio. The normative contract remains in
`dev/spec/MODEL_CONTRACT_SPEC.md`, and the checker that enforces it lives in
`earth2studio/models/conformance.py`. It was written in response to review feedback asking
how this system compares and contrasts with existing solutions.

Two questions motivated the search. Is "prose specification plus an executable conformance
checker" an established pattern? It is, in several mature forms, and they disagree in
instructive ways about how to record known failures. Is it established for machine-learning
weather models specifically? It is not: the packages closest to Earth2Studio in structure
have no behavioral enforcement at all.

For orientation, the current implementation is a rule table with stable identifiers
(`P1`-`P16`, `D1`-`D10`), a checker that probes a live model instance and reports every
violation at once, and a completeness gate that introspects the public namespaces so a new
model must be either covered by a conformance test or listed in an exemption dictionary
with a written reason.

## Package Findings

### scikit-learn

[The estimator checks](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html)
are the closest analogue in shape. Checks are module-level functions named `check_*` whose
function name serves as the stable identifier, discovered through a generator cascade
(`_yield_all_checks`) that selects check families by estimator type. `check_estimator`
runs them against a live instance and synthesizes its own data, the same relationship
Earth2Studio has between a model and a probe tensor built from `input_coords()`.

Three mechanisms are worth naming. Capability declaration happens through
`__sklearn_tags__()`, returning a `Tags` dataclass that gates which checks are generated:
`non_deterministic` suppresses the invariance checks exactly as a `stochastic` declaration
would. Known failures live in `PER_ESTIMATOR_XFAIL_CHECKS`, a `dict[type, dict[str, str]]`
mapping a class to individual check names and prose reasons, so an estimator can be
non-conformant on one check while still regression-tested on the rest. Completeness is
structural rather than curated: `test_common.py` parametrizes over `all_estimators()`,
which walks the package and collects public `BaseEstimator` subclasses.

Results are structured. Each check yields a record carrying the check name, status in
`passed`, `failed`, `skipped`, or `xfail`, and the expected-failure reason, with `on_fail`
and `on_skip` selecting whether to raise, warn, or collect.

**Delta.** Earth2Studio's identifiers are better keys than long function names, and its
completeness gate covers a case sklearn does not, namely forcing an explicit decision when
a new model appears. Earth2Studio is behind on granularity: exemptions are per model rather
than per rule, so an exempt model is checked for nothing at all.

### Gymnasium

[`check_env`](https://gymnasium.farama.org/api/utils/) validates a live environment against
the `reset`/`step` protocol. Two of its checks are near-duplicates of Earth2Studio rules
that were derived independently: `check_returned_data_not_reused` asserts that successive
returns do not share objects, and `check_reset_seed_determinism` resets with a repeated
seed and compares both the observations and the generator bit state.

Two design choices stand out. Severity is split structurally rather than by configuration:
hard protocol violations raise, while style and plausibility issues emit warnings, and
reproducibility specifically is graded, with approximate inequality raising but exact
inequality only warning. Second, the same check bodies run in two modes. `PassiveEnvChecker`
is a wrapper that runs the checks lazily on the first `reset`, `step`, and `render` call and
then passes through, and it is applied by default inside `gymnasium.make`, so ordinary user
runs perform conformance checking at negligible steady-state cost.

The weaknesses are equally instructive. There are no stable check identifiers, `check_env`
fails on the first assertion so later rules go unevaluated, and the project records its own
accepted deviations as exact-match allowlists of warning strings, which silently stop
matching when a message is reworded.

**Delta.** The passive mode is the strongest idea here and has no Earth2Studio equivalent.
The warning-string allowlist is the anti-pattern that stable rule identifiers exist to
avoid, and is worth citing as justification for them.

### Python Array API test suite

[`array-api-tests`](https://github.com/data-apis/array-api-tests) is the closest match in
mechanism. It is a standalone, portable suite run against any implementation by pointing
`ARRAY_API_TESTS_MODULE` at a namespace, with inputs generated from declared dtypes and
shapes rather than fixed fixtures. Implementations negotiate which revision they are graded
against through `ARRAY_API_TESTS_VERSION`, defaulting to the library's own declared version.

Its treatment of known failures is the single most adoptable idea found. Failures are
recorded in `xfails.txt`, a plain list of test identifiers with comments permitted, and
crucially those tests still run: an implementation that quietly fixes a violation reports
`XPASS`, which forces the waiver list to be cleaned up. Skips are a separate file for tests
that must not run at all. The waiver file lives in the implementer's repository, not the
suite's, and consumers pin a specific suite commit.

The standard itself has no numbered requirement identifiers; tests map to spec clauses only
by function name, and the documentation concedes that some clauses are untestable.

**Delta.** Earth2Studio has better traceability and worse bookkeeping. An expectation file
whose entries expire loudly is strictly stronger than a static exemption dictionary, which
cannot distinguish a live violation from a stale note.

### ONNX

ONNX separates two concerns Earth2Studio currently fuses.
[`onnx.checker.check_model`](https://onnx.ai/onnx/api/checker.html) validates that a model
*artifact* is well formed, while the
[backend test suite](https://github.com/onnx/onnx/blob/main/docs/OnnxBackendTest.md) is what
*implementations* run to prove they behave correctly. Earth2Studio's checker is doing the
second job.

Versioning is the identifier system: operators are identified by domain, type, and
`since_version`, and a model declares what it requires through `opset_import`, so tightening
a definition is a version bump rather than a silent break. The backend runner supports
regex-based `include`, `exclude`, and `xfail` selection, and defines a dedicated
`BackendIsNotSupposedToImplementIt` exception so "legitimately out of scope" is a distinct
signal from "failed". The test cases double as the code samples embedded in the operator
documentation, so one artifact is specification, example, and test.

**Delta.** Contract versioning and a first-class not-applicable signal are both missing from
Earth2Studio, which currently collapses out-of-scope, unevaluable, and checker-error into a
single "skip" string.

### Container and Kubernetes conformance

These represent the mature end of the pattern.
[The OCI distribution specification](https://github.com/opencontainers/distribution-spec/blob/main/spec.md)
adopts RFC 2119 keywords explicitly, assigns stable identifiers (`end-1` through `end-14`)
to normative endpoints, and groups them into capability categories where a registry claiming
a category must implement all of it. Its conformance runner reports a five-valued status
that distinguishes a disabled category from a skip, a failure, and an error in the test
engine itself.

[Kubernetes](https://github.com/kubernetes/community/blob/main/contributors/devel/sig-architecture/conformance-tests.md)
inverts authorship: each conformance test carries a mandatory comment block whose
`Description` states the required behavior in RFC 2119 language, and the canonical clause
list is generated from those comments. Promotion into the suite is gated on the feature
being generally available and stable, and
[certification](https://github.com/cncf/k8s-conformance/blob/master/instructions.md)
permits no skipped tests at all, so there is no waiver mechanism to rot.

**Delta.** Two practices transfer. Grouping rules into declared profiles with all-or-nothing
semantics is a cleaner model than per-rule exemption, and generating one of the specification
and the checker from the other prevents the drift that a hand-maintained pair guarantees.

### Advisory contracts

Not every project enforces. [MLflow model signatures](https://mlflow.org/docs/latest/ml/model/signatures/)
validate inputs at serving time, applying safe casts and rejecting unsafe ones, but extra
fields are ignored and unknown parameters only warn. Its one borrowable idea is the stored
input example, whose round trip is validated automatically; for models where synthesizing a
plausible probe is ambiguous, a shipped example is more honest than a generated one.

[The Open Inference Protocol](https://github.com/kserve/open-inference-protocol) is the
cautionary case. It is widely adopted, uses no RFC 2119 keywords, marks optionality
informally, lets servers advertise arbitrary undocumented extensions at runtime, and has no
conformance suite whatsoever. A well-adopted prose specification with capability
advertisement and no executable check is what Earth2Studio would become if the checker stays
optional rather than gating.

### Typeclass laws

The functional-programming tradition has been shipping behavioral contracts as reusable test
suites for years. [`cats-laws`](https://typelevel.org/cats/typeclasses/lawtesting.html)
splits the problem in two: a laws layer expressing each property as a pure symbolic equality
with no test framework in it, and a `discipline` layer that packages those into a named,
composable `RuleSet` with inheritance, so a stronger contract automatically includes the
rules of the weaker one it extends. Haskell's
[`quickcheck-classes`](https://hackage.haskell.org/package/quickcheck-classes) is the same
idea with law bundles as values.

The load-bearing detail is who supplies the data. The library ships the laws; the implementer
supplies the generator and equality for their own type, and invokes the bundle in their own
test file with a single call.

**Delta.** Earth2Studio fuses the law predicates and the runner into one module, and the
checker supplies the probe. Separating them would let the same rules be driven by a fixed
probe, by generated inputs, or by a fixture the wrapper author provides, and would let rules
be composed per model category instead of enumerated in a flat list.

### Property-based testing

[Hypothesis](https://hypothesis.readthedocs.io/en/latest/stateful.html) supplies the missing
generation machinery. `hypothesis.extra.numpy.arrays` takes an `elements` strategy, which is
the hook for physically plausible values, and `RuleBasedStateMachine` with `@rule` and
`@invariant` models exactly the shape of a rollout: step, reseed, and reset as rules, with
no-aliasing and coordinate consistency as invariants checked after every one. For continuous
integration, `derandomize` plus a bounded `max_examples` and a persisted example database
give variety without flakiness, which is how `array-api-tests` runs.

Separately, the mechanics of the properties Earth2Studio checks deserve attention. Comparing
by value is the wrong primitive twice over. Aliasing is answerable exactly and in constant
time by comparing `untyped_storage().data_ptr()` across yields, rather than by observing that
a tensor's contents changed, which depends on whether the overwriting values happened to
differ. In-place mutation is detectable through the autograd version counter `_version`,
which is bumped by any in-place operation regardless of `requires_grad` and is shared by
views and detached aliases, rather than by cloning and comparing. And `torch.allclose` and
`torch.testing.assert_close` both default to `equal_nan=False`, so a model that legitimately
produces `NaN` fails a determinism check that compares it against itself.

**Delta.** Both live defects in the current checker are instances of this. The `P16` aliasing
rule compares values and is therefore order-dependent, which is what made the DLESyM
conformance test pass alone and fail inside the full suite. The `D9` failure recorded for
`DerivedRH` is not a model defect at all: the probe feeds standard-normal values into a
formula expecting Kelvin, overflowing to `inf` and then `NaN`, and the default `equal_nan`
does the rest.

### Weather and climate packages

The domain has almost nothing comparable, which is the most consequential finding here.

[ECMWF's `ai-models`](https://github.com/ecmwf-lab/ai-models) is the closest structural
sibling, hosting many third-party models behind one plugin interface. Its `Model` base class
is a plain class rather than an abstract one, `run()` is not declared on it at all so a
plugin missing it fails at runtime, and models declare their data requirements as class
attributes that are never cross-checked against what they consume. There is no validation of
plugins, and the test suites in the framework and its model plugins are empty stubs.

[`anemoi-inference`](https://github.com/ecmwf/anemoi-inference/blob/main/src/anemoi/inference/runner.py)
is the only project found with comparable runtime checking, but it validates payloads and
provenance rather than the model object: `_check_state` validates each state dictionary,
`check_data` reports a per-variable pass and fail table rather than stopping at the first
problem, and `anemoi-inference validate` diffs the runtime environment against the one
recorded in the checkpoint, with `--on-difference` and `--exempt-packages` controlling
severity and waivers.

[PhysicsNeMo](https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/core/module.py) has
the closest thing to a capability declaration in `ModelMetaData`, which advertises support
for AMP, CUDA graphs, ONNX export, and similar, and it ships shared validators under
`test/common` that model authors call by hand. Application is voluntary, and its entry-point
test parametrizes over a hardcoded list of names, which is precisely the failure mode the
introspecting completeness gate was written to avoid.

The remaining packages constrain artifacts rather than models. WeatherBench2 documents
forecast dataset conventions and coerces names rather than validating them, with checks
scattered through the evaluation pipeline instead of exposed as an API. `earthkit-data`
defines a genuine abstract base class, but for data objects. Single-model packages validate
aggressively at construction: [Aurora](https://github.com/microsoft/aurora/blob/main/aurora/batch.py)
raises on out-of-range or non-monotonic coordinates in its `Metadata` dataclass, and
[NeuralGCM](https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/legacy/api.py) checks
variables and coordinates on entry.

**Delta.** Two details are worth carrying back. Both Aurora's `rollout` and anemoi's
forecast generator omit the initial condition, and NeuralGCM exposes the same question as a
configuration flag, `start_with_input`. That "does the rollout include the initial state"
is answered three different ways by three packages is good evidence that `P7` is pinning down
a real and recurring ambiguity rather than codifying an arbitrary preference.

## Implications for Earth2Studio

### What the survey confirms

The rules that looked most speculative are the ones with the strongest independent support.
Aliasing of successive returns, seeded reproducibility, and refusal to mutate caller data are
all checked by both sklearn and Gymnasium, arrived at separately. Global RNG isolation has no
analogue anywhere in the survey and appears to be genuinely novel.

Two Earth2Studio properties lead the field. Stable rule identifiers that appear in the
failure message are rarer than expected: only the OCI specification and, indirectly,
Kubernetes have an equivalent, and sklearn, Gymnasium, ONNX, and the Array API suite all
identify checks by function or test name. The namespace-introspecting completeness gate has
no equivalent at all in any surveyed project; nothing else prevents a newly added model from
shipping with no conformance coverage and no recorded reason.

### Expected failures rather than a static exemption list

The highest-value change. Today an entry in `_PROGNOSTIC_EXEMPT` removes a model from
checking entirely and can never tell us the waiver has gone stale. Both sklearn and the Array
API suite solve this the same way, and their solution composes with the rule identifiers
already in place:

```python
EXPECTED_FAILURES: dict[str, dict[str, str]] = {
    "DLESyM": {
        "P7": "0th yield carries the full input lead_time window",
        "P13": "two rollouts from one input disagree",
    },
}
```

A model listed this way is still checked on every other rule, and a rule that starts passing
is reported rather than ignored, which forces the list to be maintained. The current
per-model granularity means an exempt model is regression-tested on nothing.

### Capability declaration and status granularity

`stochastic` is already a capability flag in the sklearn `Tags` sense. Generalizing it would
convert today's inferred skips into declared ones, so that an unexpected skip becomes a
failure rather than a shrug. Separately, the checker currently returns a single list of skip
strings that conflates three different situations that OCI's runner keeps apart: a rule the
model never claimed, a rule claimed but not evaluable in this environment, and a rule the
checker itself failed to run. The third is the dangerous one, because a broken check is
currently indistinguishable from an inapplicable one.

### Keeping the specification and the checker aligned

`iter_contract_rules()` already exposes a machine-readable identifier and summary for every
rule, which is most of the machinery needed. A test asserting that the identifiers in
`MODEL_CONTRACT_SPEC.md` are exactly those the checker emits would close the drift risk
cheaply. Adopting RFC 2119 keywords in the rule table would also let severity be derived
rather than assumed, so a `SHOULD` rule warns where a `MUST` rule fails, which is the split
Gymnasium makes structurally and Earth2Studio currently cannot express.

### Reaching models the project did not write

`earth2studio.models.conformance` is not re-exported from `earth2studio.models` and is not
referenced in the documentation, so a third-party model author cannot currently discover it.
Two precedents apply. The Array API suite is portable and vendor-run, with the waiver file
owned by the implementer. Gymnasium's passive checker runs by default on ordinary user code,
which catches violations in models that never see the test suite. Either would extend the
contract past the wrappers in this repository, which is the only way it reaches models added
by users.

## Current Risks and Follow-up

- `P16` detects aliasing by value comparison, which is order-dependent; storage pointer
  identity answers the same question exactly. The DLESyM test helpers were seeded to stop
  the symptom, but the rule itself remains value-based
- `P13` and `D9` compare with `torch.allclose`, whose `equal_nan` default is `False`; a model
  that legitimately returns `NaN` cannot pass, as `DerivedRH` currently demonstrates
- The probe is `torch.randn`, which is physically implausible for every model in the tree;
  a variable-to-range table, or a fixture supplied by the wrapper, would prevent the overflow
  that produced the `DerivedRH` finding, and probe finiteness deserves to be its own rule
- Exemptions are per model rather than per rule, so an exempt model is checked on nothing and
  a fixed violation is never noticed
- Skips conflate "not claimed", "not evaluable here", and "the checker broke"
- There is no contract version, so tightening a rule silently invalidates every wrapper
- The checker is not part of the public API and has no documentation page
- Conformance tests run only as part of the general suite; there is no dedicated gate, and
  the completeness registry is the only thing forcing the question for a new model
- Whether to move from one fixed probe to generated inputs is unresolved; the bounded
  approach used by `array-api-tests` is the model to copy if we do

## References

- [scikit-learn check_estimator](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html)
- [scikit-learn developing estimators](https://scikit-learn.org/stable/developers/develop.html)
- [Gymnasium utils](https://gymnasium.farama.org/api/utils/)
- [Gymnasium PassiveEnvChecker](https://gymnasium.farama.org/api/wrappers/misc_wrappers/)
- [array-api-tests](https://github.com/data-apis/array-api-tests)
- [Array API test suite verification](https://data-apis.org/array-api/latest/verification_test_suite.html)
- [ONNX checker](https://onnx.ai/onnx/api/checker.html)
- [ONNX backend test](https://github.com/onnx/onnx/blob/main/docs/OnnxBackendTest.md)
- [ONNX versioning](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
- [OCI distribution specification](https://github.com/opencontainers/distribution-spec/blob/main/spec.md)
- [Kubernetes conformance tests](https://github.com/kubernetes/community/blob/main/contributors/devel/sig-architecture/conformance-tests.md)
- [CNCF certification instructions](https://github.com/cncf/k8s-conformance/blob/master/instructions.md)
- [MLflow model signatures](https://mlflow.org/docs/latest/ml/model/signatures/)
- [Open Inference Protocol](https://github.com/kserve/open-inference-protocol)
- [Cats law testing](https://typelevel.org/cats/typeclasses/lawtesting.html)
- [quickcheck-classes](https://hackage.haskell.org/package/quickcheck-classes)
- [Hypothesis stateful testing](https://hypothesis.readthedocs.io/en/latest/stateful.html)
- [torch.testing](https://docs.pytorch.org/docs/stable/testing.html)
- [ai-models](https://github.com/ecmwf-lab/ai-models)
- [anemoi-inference runner](https://github.com/ecmwf/anemoi-inference/blob/main/src/anemoi/inference/runner.py)
- [PhysicsNeMo Module](https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/core/module.py)
- [WeatherBench2 schema](https://github.com/google-research/weatherbench2/blob/main/weatherbench2/schema.py)
- [Aurora batch](https://github.com/microsoft/aurora/blob/main/aurora/batch.py)
- [NeuralGCM API](https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/legacy/api.py)
