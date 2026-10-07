# Unified Pipeline and Coupled Execution Architecture

**Status:** Proposal
**Date:** 22 September 2026
**Target:** Earth2Studio v1.0
**Related proposals:** `alignment.md`, `pipelines.md`, `Impact_modeling.md`
**Reference implementation:** [PR #1114](https://github.com/NVIDIA/earth2studio/pull/1114)

## 1. Scope

This proposal defines how Earth2Studio should align the `Pipeline` execution model
with coupled and impact-model execution. Its central objective is one execution
path that covers both:

- simple workflows containing one prognostic model and optional diagnostics; and
- coupled workflows containing multiple prognostic, diagnostic, impact, and data
  components running at different cadences.

`unification.md` was explicitly excluded from the analysis behind this proposal
and is not a source for any decision recorded here.

This proposal builds on the contracts already being developed in `dev/spec`, in
particular:

- DataArray model inputs and outputs;
- allocation-free coordinate signatures;
- the shared grid protocol and registry; and
- qualified variable labels for temporal statistics.

## 2. Decision

`Pipeline` should be the outer supervisor of a compiled execution graph. It should
not own a model-specific rollout loop and should not be subclassed for each unusual
workflow.

A single-model forecast is a one-component execution graph. A coupled or impact
workflow is a multi-component execution graph. Both compile to the same
`ExecutionPlan` and run through the same per-work-item `RunSession`.

```text
WorkItems
   |
   v
Pipeline
distribution | retries | resume | output ownership
   |
   v
resolve requirements and compile bindings
   |
   v
ExecutionPlan
components | provider bindings | transforms | schedule | output schemas
   |
   v
RunSession
initialize | advance events | emit outputs | snapshot
```

There are two deliberately separate scheduling levels:

1. `Pipeline` schedules independent work, such as initialization times and
   ensemble-member groups, across ranks.
2. `RunSession` schedules component activations and exchanges within one forecast
   or coupled simulation.

These levels are nested rather than competing. The complete coupled system remains
the atomic unit of distributed work initially; component-level distribution can be
added later without changing the public graph or pipeline APIs.

## 3. Goals

1. Use one execution path for single-model and coupled workflows.
2. Keep declarations of model data needs on the model while keeping execution and
   delivery behavior in adapters and the runtime.
3. Resolve the same field requirement from a data source, cache, fallback, or
   component export without changing the consumer.
4. Validate fields, coordinates, grids, units, cadences, temporal policies, and
   cycles before allocating model state or running compute.
5. Preserve the current DataArray and `GridDefinition` direction.
6. Generate predownload requests, output schemas, and plan descriptions from the
   same compiled plan that will execute.
7. Reuse the work-distribution, ensemble, progress, resume, and output-management
   substrate proposed for `Pipeline`.
8. Preserve a path to differentiable coupled exchange without making the v1.0
   inference runtime Torch-only.
9. Minimize bespoke execution subclasses. Model-specific behavior should normally
   be isolated in a small component adapter.

## 4. Non-goals

- A training or optimization framework.
- Concurrent placement of components across devices or process groups in v1.0.
- A general products or application catalog.
- Automatic unit conversion in the initial implementation.
- Removing the legacy tensor-and-coordinate model API as part of this work.
- Hiding scientific time policies behind framework defaults.

## 5. Responsibility boundaries

| Layer | Responsibility |
| --- | --- |
| Model | Declare static field requirements and expose normal coordinate/model contracts |
| Component adapter | Invoke the model; hold state directly if steppable, else own private state; advance one activation |
| Provider resolver | Bind a field requirement to a component export, live source, cache, or fallback |
| Graph compiler | Validate the graph and compile provider bindings, transforms, schedules, and output schemas |
| Run session | Execute one simulation in valid-time order and expose complete snapshot state |
| Pipeline | Distribute work, group ensembles, manage progress/resume, and route output events |
| `couple()` / `Application` | Construct execution graphs as convenience APIs; contain no independent executor |

This boundary implements the rule from `alignment.md`: declaration belongs on the
model, while execution belongs on a neutral wrapper. It also removes the need for
the pipeline and coupling paths to have separate resolution implementations.

## 6. Core abstractions

### 6.1 Field requirement

A field requirement describes what a consumer needs, not where it comes from.

```python
@dataclass(frozen=True)
class FieldRequirement:
    slot: str
    variables: tuple[str, ...]
    signature: CoordinateSystem
    phase: Literal["initialize", "step"]
    schedule: Schedule
    lead_offsets: tuple[np.timedelta64, ...] = ()
    optional: bool = False
    fallback: Fallback | None = None
    freshness: np.timedelta64 | None = None
```

The final names and exact field split remain to be determined, but the declaration
must carry enough static information to support graph validation without an
initialization time:

- input slot;
- canonical or qualified variable labels;
- expected dimensions, grid, and coordinate metadata;
- initialization-time versus step-time delivery;
- cadence or event schedule;
- lead offsets;
- optionality, fallback, and freshness policy; and
- canonical units resolved from the shared lexicon.

Qualified labels such as `tp:sum:24h` are the canonical representation for temporal
statistics. Coupling must not introduce a second `CellMethod` vocabulary for the
same quantity. The shared lexicon is also the authority for canonical units,
including units of qualified temporal quantities. This requires structured unit
metadata rather than parsing units from human-readable vocabulary descriptions.
Dataset-specific lexicons and modifiers convert native values to that canonical
representation. DataArray `units` attributes may mirror the resolved unit for
interoperability and output, but they are descriptive rather than the source of
truth for planning or compatibility checks.

The concrete request list used for fetching or predownload is an expansion of this
static declaration for a particular `WorkItem` and horizon. It is not the static
declaration itself.

### 6.2 Component specification

A `ComponentSpec` is the metadata-only description of an executable participant:

- stable component name;
- input requirements;
- output signatures;
- activation schedule;
- supported execution modes and array backends;
- initialization requirements; and
- checkpoint capability.

For model-backed components, output variables and grids should normally be derived
from `output_coords()`. Input requirements should be derived from the model's field
requirements. A wrapper may override either as an escape hatch for legacy or
external models, but restating declarations should not be the default.

Data sources are providers rather than consumers and therefore declare no input
requirements.

### 6.2.1 Output ports and component granularity

A component may publish more than one named output port, and one activation
may produce a DataArray containing multiple lead times. Every position in a
published DataArray must be semantically valid. Invalid filler values are not
part of the exchange contract.

Variables that differ in grid, cadence, or valid lead-time coordinates belong
on different ports or, when they have independently advancing state, in
different components. This avoids an in-band validity mask that every
connector and output backend would otherwise have to interpret correctly.
The decision rule is: use multiple ports when one atomic state transition
produces several valid products; use multiple components when those products
have independently advancing state, scheduling, or coupling relationships.

DLESyM illustrates the distinction. Its existing fused model returns one
rectangular tensor containing atmospheric and ocean variables, even though
only a subset of ocean lead times is meaningful. That representation remains
acceptable at the fused model's compatibility boundary, but an adapter must
publish only the valid subsets, for example through separate `atmos` and
`ocean` ports with their own coordinate signatures.

The coupled representation should go further: use dedicated atmosphere and
ocean component wrappers, each with its own state and cadence, and connect
them through explicit bindings. DLESyM is a coupled Earth-system model, so
the graph should represent its two independently advancing participants
rather than treating a 96-hour fused call with invalid ocean positions as one
generic component activation. The fused wrapper remains useful for simple
inference and as the numerical reference for the split-component equivalence
gate. The split wrappers must expose the underlying atmosphere and ocean step
boundaries; they must not each invoke the fused wrapper and merely select a
different subset of its output.

Output blocks remain ordinary DataArrays when all of their positions are
valid. They require no special validity-mask abstraction. This rule excludes
structural filler introduced only to force unlike schedules into one rectangle;
it does not prohibit explicit masks or missing values that are meaningful data,
such as land/sea domains or unavailable observations.

### 6.3 Component adapter and session

Execution behavior belongs on a neutral adapter. Two adapter tiers exist,
distinguished by whether the model can externalize its recurrent state. The
steppable protocol is an optional v1.0 capability, but it is required for a
stateful model that participates in step-time coupling or restartable
execution.

**Steppable models** expose state explicitly:

```python
ModelState = Mapping[str, xr.DataArray]

@dataclass(frozen=True)
class StepResult:
    state: ModelState
    output: xr.DataArray
    rng: RNGState

class SteppablePrognosticModel(Protocol):
    def initialize(
        self, x: xr.DataArray, *, rng: RNGState
    ) -> StepResult: ...

    def step(
        self,
        state: ModelState,
        imports: Mapping[str, xr.DataArray] | None = None,
        *,
        rng: RNGState,
    ) -> StepResult: ...

    def state_coords(
        self, input_coords: CoordinateSystem
    ) -> Mapping[str, CoordinateSystem]: ...
```

`state` is a mapping of named, labelled DataArrays, not an opaque handle.
That makes the model-owned portion of checkpoint/restore inspectable, makes
per-member ensemble state an ordinary leading dimension on each state entry
where the state is batchable (§12), and permits replay of one activation from
explicit inputs. `RNGState` is also passed and returned explicitly; it may
represent a local Torch generator state, a functional PRNG key, or a
seed-and-counter scheme. Deterministic models use a null RNG state.

`imports` is a borrowed, read-only mapping keyed by the model's field
requirements. The runtime does not copy DataArrays merely to deliver them,
and the model must not mutate them. Delivery is an explicit call argument,
not a mutable slot the runtime writes before calling.

For steppable models, `step()` is hook-free in the same sense as today's
`__call__()`. Iterator hooks retain their ordering but the front hook evolves
to operate on explicit recurrent state. The rear hook operates on the complete
transition because some existing models feed the hook-modified prediction into
their next recurrent state while others publish a transformed view without
changing the internal state:

```python
StateHook = Callable[[ModelState], ModelState]
TransitionHook = Callable[[StepResult], StepResult]
```

`create_iterator()` becomes a shared derived convenience over
`initialize`/`step`, rather than a second independently implemented loop:

```python
def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
    result = self.initialize(x, rng=self.initial_rng())
    yield result.output
    while True:
        state = self.front_hook(result.state)
        result = self.step(state, rng=result.rng)
        result = self.rear_hook(result)
        yield result.output
```

This is a v1.0 hook-signature change for steppable models. Each steppable wrapper
must explicitly define how its existing DataArray hooks project into and update
`ModelState` and `StepResult`; the runtime must not guess a state key or assume
that output and recurrent state are identical. A narrow helper may be used by a
wrapper when one declared state entry is exactly both the hooked input and the
recurrent output, but that is wrapper-owned adaptation rather than generic
runtime behavior. Non-steppable models retain the current DataArray hook contract
inside their existing `create_iterator()` implementation.

A model whose recurrence lives inside an object the wrapper does not control
cannot honestly implement `state_coords()`. It declares itself non-steppable
and keeps `create_iterator()` as its only execution surface — a declared
limit, reported by the conformance checker, not a silent one.

```python
class ComponentAdapter(Protocol):
    spec: ComponentSpec

    def open(
        self,
        initial: Mapping[str, xr.DataArray],
        *,
        rng: RNGState,
    ) -> ComponentSession: ...


class ComponentSession(Protocol):
    def step(
        self,
        valid_time: np.datetime64,
        inputs: Mapping[str, xr.DataArray],
    ) -> Mapping[str, xr.DataArray]: ...

    checkpoint_capability: CheckpointCapability

    def snapshot(self) -> ComponentSnapshot: ...
    def restore(self, snapshot: ComponentSnapshot) -> None: ...
```

`ComponentSnapshot` contains the model state, RNG state, step index, and any
adapter-owned state needed for deterministic continuation. Labelled
`ModelState` makes the largest part explicit, but does not alone prove that a
session is restartable. Conformance tests must snapshot, advance, restore,
and reproduce the same output.

`CheckpointCapability` distinguishes:

- `STATELESS`: no mutable state needs saving; restore is a no-op;
- `SNAPSHOTTABLE`: `snapshot()` and `restore()` are supported; and
- `UNSUPPORTED`: mutable state exists but cannot be externalized.

This prevents stateless diagnostics and deterministic provider adapters from
being confused with opaque stateful iterators.

The adapter owns the model call shape; the runtime sees declared ports and
emitted DataArrays, not model-specific conditioning kwargs, history buffers,
or normalization details.

Built-in adapters should include:

- `SteppableComponentAdapter` for models implementing
  `SteppablePrognosticModel`, driving `initialize`/`step` directly with full
  checkpoint and per-member state capability;
- `IteratorComponentAdapter` for non-steppable prognostic models, wrapping
  `create_iterator()` as an opaque generator with reduced capability: no
  inter-step import delivery and resume reported as unsupported;
- `CallComponentAdapter` for history-one models;
- `DiagnosticComponentAdapter` for stateless transformations;
- `CallableComponentAdapter` for non-Earth2Studio models and process models;
  and
- specialized adapters for split DLESyM or other genuinely distinct
  invocation shapes.

There is deliberately no hook-injection adapter. Delivering coupled imports
through `front_hook`/`rear_hook` would make the runtime a second, implicit
writer into a seam the model contract reserves for one caller-configured
transformation, silently interleaving coupling delivery with a user-set
perturbation hook. Coupled delivery goes through `step()`'s `imports`
argument; a non-steppable model that needs step-time imports must implement
`SteppablePrognosticModel` to receive them, or stay uncoupled.

The generic prognostic adapter must not call `model.__call__()` repeatedly
and guess how to form the next recurrent input. A multi-history model
without a conforming `SteppablePrognosticModel` implementation runs only
through `IteratorComponentAdapter` and fails planning if the graph requires
capabilities that adapter cannot provide, such as coupled imports or resume.
It may still execute a batched ensemble when its existing iterator supports
the required leading dimensions; opaque state prevents independent member
checkpointing, not ordinary batched inference.

### 6.4 Provider resolution

One resolver interface should serve both ordinary and coupled execution. A resolver
turns a `FieldRequirement` into a `BindingPlan`.

Sources and components remain separate public concepts. A source is externally
queried by time, lead time, and variable and may support asynchronous fetch;
a component is scheduled, stateful execution that publishes ports and may itself
depend on providers. Forcing both into one public protocol would either expose a
large union of irrelevant methods or hide these materially different lifecycles.
Adapters compile both into one internal bound-provider interface used by a
`RunSession`, so consumers and bindings do not branch on provider kind.

Possible providers are:

- another component export;
- a live `DataSource` or `ForecastSource`;
- a prefetched or predownloaded store;
- a static value or configured fallback; or
- absence, when the requirement is optional and its policy permits it.

For example, a StormCast conditioning requirement may resolve to GFS data in an
ordinary forecast and to an upstream forecast component in a coupled run. The
consumer model, component adapter, output path, and pipeline loop remain unchanged.

Resolution must be complete before execution. Every required input must have one
unambiguous provider or produce a teach-the-fix error.

### 6.5 Binding and transform plan

A binding connects one provider port to one consumer port. Its identity must include
the ports and field selection, not only the source and destination component names:

```text
(source component, source port, destination component, destination port, fields)
```

This permits multiple routes with different aggregation, regridding, or timing
policies between the same component pair.

A binding may compile the following operations:

- variable selection or renaming;
- temporal aggregation;
- cadence bridging and freshness checks;
- spatial regridding or reprojection;
- masking and fill policy;
- vertical interpolation;
- point or geometry sampling; and
- user-supplied or learned transforms.

Every value-touching transform must have two phases:

1. resolve a metadata-only plan; and
2. apply that plan through an array-backend implementation.

Regridding consumes shared `GridDefinition` objects and uses their fingerprints for
validation and weight caching. Regridders expose their weights or equivalent plan,
not only an opaque apply method.

### 6.6 Execution graph and plan

An `ExecutionGraph` is a user-constructible collection of component specifications
and unresolved bindings. An `ExecutionPlan` is the validated, executable result of
combining that graph with:

- a run request and horizon;
- concrete provider registrations;
- execution device and backend capabilities;
- output subscriptions; and
- checkpoint and ensemble settings.

The plan contains:

- every component activation;
- concrete provider bindings;
- compiled transform plans;
- ordering and data-availability semantics;
- expanded source requests;
- output schemas;
- state and checkpoint capabilities; and
- a stable identity used by the progress store.

`describe()` renders the compiled plan, ensuring the preview and the actual runtime
cannot drift.

### 6.7 Run session

A `RunSession` owns all mutable state for one work item:

- open component sessions;
- connector windows and histories;
- current event or clock position;
- random-number state;
- cached transformation state;
- output cursor; and
- any warm-start or assimilation state.

It emits `OutputEvent` objects rather than writing through driver-specific IO:

```python
@dataclass(frozen=True)
class OutputEvent:
    component: str
    port: str
    produced_at: np.datetime64
    data: xr.DataArray
```

`produced_at` is the component activation time at which the output became
available to the graph. `data` carries its forecast reference-time,
`lead_time`, and derived `valid_time` coordinates. Keeping production time
separate from forecast valid time prevents a block of future forecasts from
appearing available before the activation that produced it. Every position
in `data` is valid for the named port (§6.2.1).

The shared `OutputManager` owns filtering, output-side transforms, shard ownership,
per-component destinations, flushing, and finalization.

## 7. Data representation

The inference execution path uses `xr.DataArray` at provider, component, connector,
and output boundaries. It must not introduce a second public exchange container made
from a Torch tensor plus an ordered coordinate dictionary.

Important consequences are:

- preserve arbitrary leading dimensions, including ensemble and sample dimensions;
- preserve auxiliary latitude/longitude coordinates on projected and curvilinear
  grids;
- select fields by qualified variable label while retaining variable metadata;
- use existing DataArray-to-Torch conversion only at model boundaries; and
- avoid stripping singleton dimensions merely to create a separate exchange shape.

The initial inference backends are NumPy and CuPy. Differentiable exchange is
deferred, but the graph and transform plans remain backend-neutral. A future Torch
apply implementation or Torch-backed labelled-array boundary can execute the same
plans without changing field declarations or graph semantics.

Conversions that would detach a gradient-requiring tensor must continue to fail
explicitly.

## 8. Scheduling semantics

### 8.1 Explicit scientific timing

Lagged versus current-state delivery is a modeling assumption and must be explicit
on the binding:

```python
Binding(
    source="ocean.sst",
    target="atmos.sst",
    availability="previous",  # alternatively "current"
    time_policy="constant",
)
```

The graph compiler derives an action order consistent with these declarations.
The generated sequence is human-readable and inspectable, but v1.0 does not expose
a raw action-order override as a stable public API. If declared semantics do not
determine a safe order, compilation fails and asks for a declarative timing or
solver policy. A raw developer override may exist while implementing the coupler,
but action position must not become a second source of scientific semantics: an
optimizer, serializer, or topological reorder must not silently change the science.

### 8.2 Event-based execution

The runtime should advance through component activation events rather than stepping
only on every greatest-common-divisor clock tick. Fixed cadence components compile
to regular events; irregular satellite overpasses or observation arrivals compile
to irregular events.

The concrete schedule algebra belongs in the coupler/graph implementation plan,
not in `Pipeline`. It should cover at least fixed cadence with an origin or phase,
explicit or streamed irregular times, and offsets relative to a work item's
forecast reference time. Each schedule must support lazy iteration over a bounded
horizon, metadata-only compatibility checks, a stable fingerprint for plan and
progress identity, and a concise preview. `Pipeline` distributes work carrying the
horizon and persists the compiled identity; it does not interpret component
schedules or expand long runs eagerly.

The plan preview reports:

- each component's activation times;
- the version and valid time of each consumed input;
- hold, interpolation, reduction, and freshness policies;
- the final activation of each component within the requested horizon; and
- uncovered or stale intervals.

Start/stop interval closure and terminal-state expectations must be specified and
validated. A run must not silently report completion at a time for which a
coarse-cadence output was never produced.

Event timing and output coordinates answer different questions. The event
schedule decides when a component activates and therefore when its output
becomes available; each output port's coordinates state the valid times of
the forecasts produced by that activation. Ports contain no invalid filler.

### 8.3 Cycles

Cycles of current-state dependencies are invalid unless the user supplies an
explicit iteration or solver policy. The default error should identify the cycle and
suggest marking a specific binding as previous-state delivery when that is the
intended coupling.

## 9. Pipeline behavior

`Pipeline` becomes a concrete supervisor rather than an ABC whose subclasses own
the inner inference loop.

Conceptually, its per-work-item path is fixed:

```python
def run_item(self, item: WorkItem) -> None:
    plan = self.workflow.compile(item, self.providers)
    session = plan.open(item)
    for event in session.run():
        self.output.write(event)
```

The real implementation may compile once for a homogeneous group of work items and
reuse cached plans. The important constraint is that workflow variation appears in
the graph and adapters, not in a replaced `run_item` loop.

The pipeline owns:

- `WorkItem` identity and rank distribution;
- ensemble-member grouping;
- output-store ownership and collision checks;
- progress and completion records;
- snapshot persistence and restoration;
- retry and failure boundaries;
- output manager lifecycle; and
- optional progress reporting and scoring observers.

Output coordinates are derived from compiled output signatures. Predownload stores
are derived from expanded external-provider requests. Neither requires a
pipeline-specific subclass method.

## 10. Simple workflow mapping

A deterministic forecast compiles to a graph containing:

1. an initialization-data provider;
2. one prognostic component;
3. any external forcing providers;
4. optional diagnostic components connected at the relevant valid time; and
5. output subscriptions.

The user-facing convenience path remains small:

```python
pipeline = Pipeline.from_model(
    model,
    data=source,
    diagnostics=diagnostics,
    output=output,
)
pipeline.run(items)
```

`Pipeline.from_model()` is a graph builder; it does not select a different runtime.
`earth2studio.run.deterministic()` becomes a thin compatibility facade over this
construction.

Perturbations are initialization transforms. Diagnostics are ordinary graph
components with current-state bindings. Pure scoring or writing stages should be
output observers unless they participate in model state or feed later components.

## 11. Coupled workflow mapping

A coupled workflow uses the same graph and pipeline with additional components and
bindings:

```python
graph = couple(
    atmos,
    ocean,
    impact,
    context=[observations],
    bindings=[...],
)

pipeline = Pipeline(
    workflow=graph,
    providers=providers,
    output=output,
    progress=progress,
)
pipeline.run(items)
```

`couple()` returns an unresolved `ExecutionGraph`, not a second public executor.

`Application(...)` similarly translates role-oriented arguments such as
`forcing`, `context`, `transforms`, `model`, `diagnostics`, and `io` into graph
components, bindings, and output subscriptions. It contains no independent driver
or rollout loop.

## 12. Ensembles and distributed work

An initial condition and its coupled component state form one atomic simulation.
The first distributed implementation assigns complete simulations to ranks rather
than splitting component activations across ranks.

Ensemble-member groups are represented by leading DataArray dimensions and remain
present through stateful components. Each per-member `ModelState` entry carries the
same labelled member dimension; genuinely shared immutable entries may omit it.
`ModelState` is not alternatively represented as a sequence of member states.
Allowing both shapes would double the state, checkpoint, transform, and validation
paths and would make a component session's atomic state ambiguous.

The pipeline decides which member IDs a work item carries; the graph runtime treats
those dimensions like any other supported leading dimensions. A component that
cannot batch its recurrent state declares that capability and forces member groups
of size one. Multiple such members are multiple sessions/work items rather than a
sequence hidden inside one `ModelState`. Per-member RNG streams are keyed by stable
member identity so regrouping members or changing world size does not change their
trajectories.

Required validation includes:

- ensemble size one is bit-identical to the unbatched path;
- every component and transform preserves member identity and ordering;
- stochastic models declare whether batched and unbatched member streams are
  reproducible; and
- output ownership is validated before execution.

Per-component device placement, parallel branches, and distributed component
execution are later executor capabilities. They must not require changes to graph
declarations.

## 13. Checkpoint and resume

Output markers alone are insufficient for coupled workflows. A resumable snapshot
must include:

- every component's internal model state and history window;
- connector or mediator accumulation windows and delivery history;
- the event cursor or clock;
- RNG state;
- cached state needed for deterministic continuation; and
- output write position and committed-progress metadata.

Checkpoint capability is validated during planning. If any stateful participant
cannot snapshot and restore, the plan reports that resume is unsupported. It must
not silently advertise output-level resume while restarting component state from
the initial condition.

For a component whose model implements `SteppablePrognosticModel` (§6.3),
the labelled `ModelState` and explicit `RNGState` form the core of its
`ComponentSnapshot`; the session adds its step index and adapter-owned state.
Restartability still requires replay conformance tests. A stateless component
is checkpoint-compatible without serialized state, while an opaque stateful
iterator reports `CheckpointCapability.UNSUPPORTED`. A plan requiring resume
is rejected only for the latter case.

The durable encoding and migration policy can be deferred until the execution
contracts have implementation experience. The initial resume guarantee is
same-version continuation, not restart across arbitrary Earth2Studio releases.
Even the first encoding must record its schema version, Earth2Studio version,
execution-plan fingerprint, and relevant component/model versions and reject an
incompatible snapshot rather than attempting a best-effort restore.

The progress identity should include the workflow/plan fingerprint, initialization
time, ensemble members, scenario parameters, and relevant model/source versions. It
must remain stable across world-size changes.

## 14. Serving integration

The serving `Workflow` abstraction remains responsible for service lifecycle and
request handling. It composes with execution through a thin adapter:

```text
REST Workflow -> PipelineWorkflow adapter -> Pipeline -> ExecutionPlan
```

Serving does not implement a second rollout loop. A local one-item pipeline run and
a distributed evaluation run use the same compiled graph and run-session behavior.

## 15. Use of PR #1114

PR #1114 is a valuable prototype and test source, but it predates the current
DataArray work and should not be merged as a parallel execution subsystem.

### 15.1 Retain

- Component, connector, and driver decomposition.
- Advertise, realize, initialize, step, and finalize lifecycle concepts.
- Fail-before-compute graph validation.
- Derived scheduling and plan inspection.
- Adapter-owned model invocation.
- Connector time, regrid, fill, vertical, and accumulation mechanisms.
- Connector reset and history behavior.
- DLESyM split-component work and its real-weights equivalence gate.
- Hand-computed trajectory, seam, cadence, gradient, and error-path tests.

### 15.2 Refactor

- Replace the public Torch `Field`/`State` exchange representation with DataArrays.
- Replace the package-local field dictionary and `CellMethod` representation with
  the shared lexicon and qualified temporal-statistics vocabulary.
- Derive component imports from model field requirements instead of restating them
  on every wrapper.
- Replace raw coordinate maps and `interp_to` usage with `GridDefinition` and the
  shared regridding interface.
- Replace the runnable public `Driver` with the graph compiler and internal
  per-work-item session runner used by `Pipeline`.
- Replace driver-specific IO and in-memory collection with `OutputEvent` and the
  shared `OutputManager`.
- Replace `(source, destination)` connector identity with port- and field-specific
  binding identity.
- Replace inferred `next_input` behavior and ad hoc conditioning kwargs with
  `SteppablePrognosticModel.step()`'s explicit `imports` argument.
- Represent DLESyM coupling with dedicated atmosphere and ocean components;
  retain the fused wrapper as a simple-inference facade and numerical
  equivalence reference, not as the coupled graph primitive.
- Make lag/current-state delivery explicit on bindings and compile it into the run
  sequence.
- Keep pull-style source injection only as a migration shim for models that still
  own a `DataSource`.

### 15.3 Verification gate

The DLESyM real-weights equivalence test must run successfully after the DataArray
refactor before the coupled implementation is considered validated. Mock-derived
tests are useful structural coverage but cannot prove numerical equivalence to the
existing fused model.

## 16. Implementation sequence

### Phase 0: Freeze the breaking execution contracts

These interfaces change public model and output shapes. Slipping them past
the September freeze costs a second migration pass over every wrapper
already being touched for `dev/spec`.

- `FieldRequirement` and port-specific output declarations (§6.2.1).
- The optional explicit-state capability — `SteppablePrognosticModel`,
  `ModelState`, explicit RNG state, `initialize`/`step`/`state_coords`, and
  state/output hook semantics — and its conformance rules (§6.3). Scope its
  first implementations to models already migrated to the DataArray
  execution API. Non-steppable models keep `create_iterator()` and run at
  `IteratorComponentAdapter`'s reduced capability.
- Component specification shape, since adapter-tier selection (steppable vs.
  iterator-only) depends on it.

Add tests proving that planning does not touch field values or allocate model state.

### Phase 0b: Settle additive interfaces

These extend the surface without breaking a currently conforming model or
component, so they can keep firming up through October without forcing a
second migration.

- Provider resolution and binding plans.
- Event schedules and time-policy semantics.
- Execution-plan identity and inspection.
- Output events.
- `ComponentSnapshot`, restore behavior, and checkpoint capability reporting.

### Phase 1: Provider resolution and planning

- Add static field declarations to the relevant conditioned models.
- Implement `SteppablePrognosticModel` for candidate models whose recurrence
  is already externalized in wrapper-owned tensors. `aifs2ens` is a candidate
  after its separate DataArray migration; its window history, step counter,
  and RNG state must all be included explicitly. Models with genuinely
  opaque external state remain on `IteratorComponentAdapter`.
- Implement provider resolution for live sources, caches, component exports,
  fallbacks, and optional absence.
- Expand static requirements into concrete source requests.
- Generate predownload plans from those requests.
- Compile and validate transformations using the grid and temporal-statistics
  contracts already present in the current branch.

No new `Pipeline` class is required for this phase.

### Phase 2: DataArray graph kernel

- Port the component lifecycle, connector planning, scheduling, validation, and
  description mechanisms from PR #1114.
- Implement them directly against DataArrays and shared grid/statistics metadata.
- Add the standard component adapters.
- Implement a local, single-process `RunSession`.
- Add output events and full session state enumeration.

### Phase 3: Prove the one-component path

- Run one migrated prognostic model through the graph kernel.
- Demonstrate step-for-step equivalence with direct `create_iterator()` execution.
- Add a diagnostic component and external forcing provider.
- Demonstrate that live, prefetched, and predownloaded providers produce the same
  consumer behavior.
- Convert `earth2studio.run.deterministic()` to a thin graph-and-pipeline facade.

### Phase 4: Add the concrete Pipeline

- Move `WorkItem`, work distribution, ensemble grouping, progress storage, and
  `OutputManager` into the shared execution substrate.
- Make the pipeline drive `RunSession` without workflow-specific subclasses.
- Derive output schemas and predownload plans from `ExecutionPlan`.
- Integrate complete snapshot/restore and stable work identity.
- Add a serving adapter over the same pipeline.

### Phase 5: Coupled acceptance gates

- Conditioned forecast whose requirement can switch between a data source and an
  upstream model.
- StormScope-style sequential two-model workflow.
- Multi-rate impact model with optional, freshness-limited observations.
- Dedicated DLESyM atmosphere and ocean components versus fused real-weight
  numerical equivalence.
- Multiple bindings with different policies between the same component pair.
- Irregular observation schedule.
- Ensemble size one equivalence and batched-member identity preservation.
- Coupled checkpoint/restart equivalence.

### Phase 6: Application and recipe migration

- Convert existing eval pipeline subclasses into graph builders and adapters.
- Convert `Application(...)` into a role-oriented graph builder.
- Migrate other `earth2studio.run.*` facades.
- Remove source ownership from models after the compatibility window.
- Remove the pull-style compatibility shim once affected models have migrated.

## 17. Acceptance criteria

The architecture is successful when all of the following are true:

1. A single-model forecast and a multi-component coupled workflow use the same
   `Pipeline`, graph compiler, run-session implementation, and output path.
2. Switching a conditioning field from cached GFS to a live upstream model changes
   only a provider binding.
3. No model's external data requirements are restated independently in pipeline and
   coupling code.
4. No second grid registry, temporal-statistics vocabulary, field dictionary, or
   public labelled-field container exists.
5. Output schemas and predownload requests are generated from the same execution
   plan that runs.
6. Graph errors are reported before model compute or field allocation.
7. Every value-touching connector operation has a tested metadata-only resolver.
8. A model implementing `SteppablePrognosticModel` participates in coupling
   and resume through declared model, RNG, and adapter state; a non-steppable
   multi-history prognostic cannot be driven by a generic repeated-`__call__`
   loop that guesses its recurrent input.
9. Coupled resume restores all component and connector state or is explicitly
   rejected as unsupported.
10. The refactored DLESyM graph passes the real-weight fused-model equivalence gate.
11. Ensemble size one is bit-identical to the unbatched execution path.
12. Existing simple APIs remain thin convenience facades rather than alternate
    execution engines.

## 18. Resolved and deferred questions

1. **Units — resolved.** Canonical per-variable units live in structured shared-
   lexicon metadata, including for qualified temporal quantities. DataArray
   attributes may mirror them but are not authoritative (§6.1).
2. **Provider surface — resolved.** Sources and components remain distinct public
   concepts because their query, scheduling, state, and dependency lifecycles are
   different. Provider adapters converge on one internal bound-provider interface
   after resolution (§6.4).
3. **Schedule representation — delegated with constraints.** The coupler/graph
   implementation plan chooses the concrete schedule types. They must cover regular,
   irregular, and forecast-relative activation lazily and provide stable identity
   and inspection. This affects `Pipeline` only through horizon-bearing work items
   and compiled plan/progress identity; `Pipeline` does not interpret schedules
   (§8.2).
4. **Snapshot encoding — deferred.** The initial contract promises same-version
   continuation with strict compatibility metadata and fail-closed restore. A
   cross-version encoding and migration policy is later hardening work (§13).
5. **Run-sequence override — resolved for v1.0.** The compiled sequence is public
   and inspectable, but raw action-order overrides are not a stable public API.
   Users express timing and iteration semantics declaratively (§8.1).
6. **Ensemble state — resolved.** Per-member mutable state uses a labelled leading
   member dimension. Unbatchable models force size-one sessions rather than changing
   `ModelState` into a sequence representation (§12).
7. **Hook compatibility — resolved.** Steppable wrappers explicitly migrate how
   hooks project into state and transitions. The runtime performs no guessed state-
   key adaptation; wrappers may opt into a narrow helper only when the mapping is
   exact and declared (§6.3).

The explicit-state protocol is likewise settled as an optional v1.0 capability for
DataArray models but mandatory for a stateful model participating in step-time
coupling or restartable execution. Iterator-only models remain valid for ordinary
inference at reduced capability.

The remaining work is implementation-level validation rather than an unresolved
architecture split: define the internal bound-provider method shape, select concrete
schedule classes, and document/test each migrated wrapper's hook projection.
