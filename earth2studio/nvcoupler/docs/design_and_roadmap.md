# Design and roadmap

Why nvcoupler is shaped the way it is: the NUOPC inheritance, the decisions
that diverge from it, how the implementation was verified, and what is
honestly not done yet. For what each abstraction *means*, see
[concepts](concepts.md); for how to use them, the [user guide](user_guide.md)
and [API reference](api_reference.md).

## Why NUOPC concepts for ML inference

AI Earth-system models are coupled today in three ad-hoc ways (see prior art
below), each of which welds the coupling decision to code that should not own
it. The Earth-system-modeling community solved this problem once already:
NUOPC/ESMF's component/connector/mediator/driver decomposition is thirty years
of institutional knowledge about which seams matter. nvcoupler ports the
concepts, not the code.

**What transfers directly:**

- Components with advertise/realize/initialize/run/finalize phases, so a
  misconfigured system fails at initialize with a named, actionable error
  rather than mid-rollout.
- Connectors as the sole path between components, matching by a field
  dictionary's standard names rather than model vocabularies.
- Mediators for cadence-bridging reductions.
- A driver executing an ordered run sequence on a shared clock, each
  component gated by its own declared timestep (slot alignment) — which is
  exactly what a 6 h atmosphere and a 48 h ocean need to coexist.
- The "data component" move (prescribed forcing is just another component) and
  the "split a monolithic executable into gridded components" move
  ([`dlesym_split.py`](../dlesym_split.py) is the latter, applied to DLESyM).

**What was deliberately dropped for v1:**

- **Concurrency.** NUOPC runs components on disjoint PE layouts; nvcoupler
  executes slot actions strictly sequentially in one process. ML inference
  steps are GPU-bound and fast; the ordering-as-semantics model (below) also
  *requires* deterministic sequencing. Multi-GPU concurrency is roadmap, not
  regret.
- **Conservative regridding and flux exchange.** ESMF couplers conserve energy
  and mass across grids because the physics demands it. ML emulators are not
  conservation-constrained, exchange states rather than fluxes, and v1 ships
  bilinear interpolation only (plus a custom-`regridder=` escape hatch).
- **Fortran-era config machinery** in favor of a small Python API, a
  NUOPC-flavored [runSeq DSL, and YAML round-tripping](dsl_and_yaml_reference.md).

**The fundamental caveat:** a physical model tolerates any dynamically
consistent forcing; an ML model only tolerates forcing that looks like its
training data. You can re-plumb *how* DLESyM's halves exchange SST, but you
cannot make its atmosphere accept hourly SST, a new variable, or an
out-of-distribution ocean and expect skill. nvcoupler makes coupling
structure explicit and swappable; it cannot make models coupleable in ways
they were not trained for. (Coupled fine-tuning, below, is the remedy the
architecture is built to enable.)

## Key decisions

### The adapter owns the model call

Alternatives considered: (a) require every model to implement a common
coupled-model interface — rejected, because the point is to couple *existing*
models unmodified; (b) have components translate imports into each model's
call shape — rejected, because it multiplies component subclasses by call
shapes. Instead a small `ImportAdapter` protocol owns the invocation, and the
three built-ins are transcriptions of the three call shapes observed in the
wild: state-channel overwrite (prescribed forcing), StormScope's
`call_with_conditioning` kwarg, and the DLESyM/PhysicsNeMo extra coupling
tensor. A new call shape is a ~20-line adapter, not a framework change.
Corollary decision: stacking adapters demand an explicit `field_order=` for
multiple imports, because alphabetical stacking would feed channel-permuted
inputs that run fine and predict garbage.

### Physical-units exchange

Alternative: exchange in each model's normalized space, avoiding a
denormalize/renormalize round trip per step. Rejected because normalization
statistics are private per model — normalized exchange couples every
component to every other's training pipeline and makes a `DataComponent`
(observations, in physical units) a special case. The round trip through the
same per-variable constants is exact up to float rounding (asserted for the
DLESyM split in `test/nvcoupler/test_dlesym_split.py`). Units are checked at
match time, not converted — a wrong-units pairing should be a loud
configuration error, not a silent multiply (see
[errors and troubleshooting](errors_and_troubleshooting.md)).

### Slot ordering as coupling semantics

Alternative: a `mode="lagged"` flag on connectors. Rejected in favor of the
NUOPC convention that a runSeq *is* the coupling semantics: a connect before
the source's run delivers the source's previous state (lagged), after it
the fresh state (sequential). Derived sequences make the canonical lagged
shape the default without giving up the mechanism — `derive_sequence`'s
`lagged=` parameter only chooses where each connect is *placed*, never adds
a second mechanism. One mechanism, zero redundant configuration to
disagree with itself, and a coupling-order experiment is a one-line DSL edit
([example 02](../../../examples/09_nvcoupler/02_lagged_vs_sequential.py)).
`describe()` derives and displays the mode per connect so the ordering is
never implicit knowledge.

### Dictionary CellMethods over string parsing

Alternative: infer "48 h mean of z1000" by parsing the suffix of
`geopotential_at_1000hpa_48h_mean`. Rejected — name-grammar coupling is how
lexicons rot. A derived field is a first-class `FieldEntry` carrying
`CellMethod(base, method, window)`, which is what lets windowed
`Connector`s, `AccumulationMediator`, and `couple()`'s windowed-connector
synthesis operate on data, not regexes
([concepts](concepts.md#cellmethod-derived-fields-as-first-class-dictionary-entries)).

### Pure-torch exchange for training readiness

Every stage of the exchange path — regrid gathers, mask-fill gathers,
log-pressure vertical interpolation, mediator running reductions, functional
import injection (clone + `index_copy_`) — is differentiable torch. That is a
tax during inference (numpy would sometimes be simpler) paid for one payoff:
`driver.rollout(n_steps)` keeps the autograd graph across the whole coupled
system, while `run()`/`steps()` execute under `torch.inference_mode()` and
record/IO paths detach so collection never pins graphs.

```python
import torch
import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

gain = torch.tensor(1.0, requires_grad=True)   # a shared "parameter"
driver = nvc.couple(
    fake_atmos(gain), fake_ocean(gain), start="2024-01-01", stop="2024-01-05"
)
driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
with torch.enable_grad():
    states = driver.rollout(16)                # 16 x 6 h, graph intact
loss = states["atmos"]["geopotential_at_1000hpa"].data.mean()
loss.backward()
gain.grad                                      # non-zero: crossed the exchange
```

What coupled fine-tuning enables: jointly training coupled emulators so each
learns to tolerate the other's imperfect output — the standard remedy for
coupled drift, previously unavailable because the exchange lived in numpy
datapipes. Optimizer loops, truncated BPTT, and per-component device placement
stay out of scope for v1 ([example 05](../../../examples/09_nvcoupler/05_coupled_finetuning.py)
shows a complete training step).

### Framework coupling, not datapipe or model-internal coupling

The three prior-art patterns this package factors out:

- **Datapipe-level** — PhysicsNeMo's `ConstantCoupler` /
  `TrailingAverageCoupler` bake exchange policy into data loading; changing
  the coupling means changing the datapipe, and two-way interaction is out of
  reach. nvcoupler keeps both as one-line configurations (`time_policy=
  "constant"`; `Connector(window=, reduce=)` or `TrailingAverageMediator`)
  on a two-way-capable substrate.
- **Model-internal** — `earth2studio/models/px/dlesym.py` hard-codes the
  atmos↔ocean exchange inside `__call__`; swapping the ocean for observations
  means forking the model. `split_dlesym` re-exposes the halves as components.
- **Caller-owned** — StormScope's `call_with_conditioning` leaves coupling
  entirely to user scripts; correct, but unshareable and unvalidated.
  `ConditioningKwargAdapter` gives that pattern the same validation, cadence,
  and regridding machinery as everything else.

## Verification story

The code ships with 176 passing tests (`test/nvcoupler/`; a 177th — the
real-weights gate below — is collected but skipped), built on a deliberate
strategy:

- **Hand-computed toys.** The `testing.py` components are linear maps with
  spatially constant ICs, so every intermediate value of a 96 h coupled run is
  computable on paper; `test_driver.py::test_cadence_and_hand_computed_values`
  asserts the full trajectory, and the lagged/sequential, gradient-flow, and
  reset-reproducibility tests reuse the same closed-form system. Tests are the
  executable specification — mediator dedup, connector history rotation, and
  every error path in [errors_and_troubleshooting.md](errors_and_troubleshooting.md)
  are pinned there.
- **Seam tests** (`test_seams.py`). The historically bug-rich boundary is
  shape conventions between component kinds: a `PrognosticComponent`'s exports
  (model coords carry singleton batch/time/lead_time) feeding a
  `CallableComponent`'s overwrite adapter, stacking with a `DataComponent`'s
  fields in one import state, and driving a `DiagnosticComponent` end to end.
- **Adversarial reviews.** The design underwent a design review and the
  implementation an execution-verified code review (findings reproduced by
  running code, then fixed and regression-tested). Representative bug classes
  caught this way, each now defended in code and tests: axis renumbering after
  `tensor.select` silently slicing the wrong dimension (see the pointed
  helper comment in `test_dlesym_weights_equivalence.py`); connector history
  collapsing when a slow source's export is re-seen by a fast slot (the
  "rotate only on new valid_time" rule); in-memory records and IO writes
  pinning autograd graphs during `rollout` (detached clones, off the exchange
  path); zarr's default 0.0 fill letting never-written rows masquerade as
  physical values (NaN initialization); and double-counted mediator samples on
  duplicate deliveries.
- **The DLESyM weights-equivalence gate — currently UNRUN.** All
  `dlesym_split` tests run against a mock authored from a *reading* of
  `dlesym.py`, which makes them structurally circular: a misreading
  (normalization order, insolation anchors, window chunking) would pass every
  mock test and fail on real weights.
  `test_dlesym_weights_equivalence.py` is the actual proof — it drives the
  split components through nvcoupler and asserts equality with native
  `DLESyM.__call__` on real checkpoints — but it is gated behind
  `NVCOUPLER_DLESYM_WEIGHTS=1` (needs physicsnemo plus the multi-GB
  `hf://nvidia/dlesym-v1-era5` package) and **has not yet been executed
  anywhere**. Until it passes, treat "nvcoupler can host DLESyM" as
  structurally validated but numerically unverified.

## Honest limitations and v2 roadmap

Current, stated plainly:

- **Real-weights validation is pending** (the gate above). Highest-priority
  item; it either closes the loop or finds the misreading the mocks cannot.
- **Units are checked, not converted.** No pint dependency in v1; a genuine
  mismatch requires a converting Mediator or aligned dictionary entries.
- **HEALPix / curvilinear sources need a user regridder.** The auto path
  handles regular 1D lat/lon only; differing `face` grids raise with an
  `earth2grid` pointer rather than guessing.
- **No checkpoint/restart.** A crashed 3-month coupled run restarts from t0.
  Restart needs serializing component internal state (model windows, mediator
  accumulators, connector history), which the YAML layer deliberately does not
  attempt yet.
- **No coupled ensembles.** Batch dims flow through the exchange, but there is
  no ensemble-aware driver API (perturbations, per-member IO).
- **Single-process, single-device execution.** No concurrent slot execution,
  no per-component GPU placement; components that could run in parallel
  (independent branches of an impact chain) do not.
- **`DiagnosticComponent` is shallow** — single-grid, all-imports-every-step;
  windowed or multi-resolution diagnostics need mediator help.
- **The field dictionary is package-local.** `DEFAULT_DICTIONARY` is a curated
  v1 vocabulary; reconciling it with (or upstreaming it into)
  `earth2studio.lexicon` — whose `normalize_units` it already mirrors — is
  open, and drift between the two is a real risk.
- **Long gradient rollouts hold every step's graph.** `rollout(n)` has no
  gradient checkpointing or truncated-BPTT support; memory grows linearly in
  n, which caps practical fine-tuning horizons.
- **Driver IO** is in-memory xarray plus streaming `IOBackend` writes; the
  `couple()` auto-wiring layer does not yet configure IO.

The v2 order of attack roughly follows that list: run the weights gate, then
checkpoint/restart and gradient checkpointing (they share the
state-serialization work), then ensembles and multi-GPU placement, with unit
conversion and lexicon reconciliation as the dictionary matures.
