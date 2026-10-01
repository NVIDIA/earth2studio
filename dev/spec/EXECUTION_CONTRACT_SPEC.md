# Execution Contracts (draft)

Goal: one execution path for single-model and coupled runs, without making the
single-model case pay for coupling's complexity.

The interface between Pipeline work supervision and the execution of one
forward run: how to align things between single-model forecasts and coupled
component graphs.

Pipeline and the graph compiler are not yet implemented. Existing model, source, grid (`GRID_SPEC.md`), and
time-statistics (`TIME_STATISTICS_SPEC.md`) contracts remain authoritative.

## Layers


| Module                        | Holds                                               |
| ----------------------------- | --------------------------------------------------- |
| `earth2studio.models.px.base` | `PrognosticModel`; explicit-state capability        |
| `earth2studio.run.schedules`  | `Schedule` protocol and all concrete schedules      |
| `earth2studio.run.session`    | the shared boundary: plan, session, event, snapshot |
| `earth2studio.run.component`  | component spec, session, adapter, snapshot          |
| `earth2studio.run.single`     | direct one-model loop and output transforms         |
| `earth2studio.run`            | built-in workflows, `WorkItem`; later `Pipeline`    |
| `earth2studio.coupling`       | graph authoring, compilation, and execution         |


**Share model knowledge, not necessarily control flow.** Model requirements,
ports, cadence, and capabilities are derived once by the component adapter.
The single-model executor and graph executor reuse these declarations and
stepping primitives. The direct loop remains independent of graph scheduling;
there is no required migration to a one-node graph compiler.

`Pipeline` depends only on `ExecutionPlan`, never on components or graph topology.
A custom executor needs only the plan/session contract. Graph compilation,
exchange resolution, and connector history stay in `coupling`; `run` does not
import it. A fresh-interpreter test guards this dependency direction.

A one-model graph and a direct rollout should be behaviorally equivalent. That
is an integration check, not an implementation requirement.

The classic single-model rollout stays a four-liner. `Pipeline.from_model`
wraps `SingleModelPlan`; nothing here touches `coupling`:

```python
from earth2studio.data import GFS
from earth2studio.models.px import SFNO
from earth2studio.run import Pipeline

model = SFNO.load_model(SFNO.load_default_package())
pipeline = Pipeline.from_model(model, GFS())
pipeline.run(times=["2024-01-01T00"])
```

A compiled graph is an `ExecutionPlan` like any other; `Pipeline` never
imports `coupling` to accept one. A coupled run is user code composing two
planned public pieces:

```python
from earth2studio.coupling import couple
from earth2studio.run import Pipeline

plan = couple(goes, mrms, bindings=[...]).compile(providers)
Pipeline(plan).run(items)
```



## API tiers

Not all of the complexity needs to be exposed to users. Organize with three tiers:


| Tier      | For                                             |
| --------- | ----------------------------------------------- |
| Public    | users running forecasts                         |
| Extension | advanced users; model wrapper and graph authors |
| Internal  | the package                                     |


**Public:** `run.deterministic`, `run.diagnostic`, `run.ensemble`, `WorkItem`;
later `Pipeline`, `Pipeline.from_model`, and `couple()`. Eventually we may sunset
`run.deterministic` and related functions, or make them thin wrappers around a
`Pipeline`-based approach.

**Extension**, by task:

- **Customize published outputs:** `run.single.OutputTransform` and
`SingleModelPlan`.
- **Write a bespoke loop:** `run.session.LoopPlan`, an ordinary generator plus
output declarations and a caller-owned identity. No component types required.
- **Components:** `run.component` — `ComponentAdapter`, `ComponentSession`,
`ComponentSpec`, `ComponentSnapshot`, `FieldRequirement`, `PrognosticComponent`.
- **Custom execution:** `run.session` — `ExecutionPlan`, `RunSession`,
`OutputEvent`, `RunSnapshot`, `SnapshotCompatibility`, `CheckpointCapability`,
`OutputPort`, `PortRef`, `ResolvedRequest`.
- **Schedules:** `run.schedules` — `Schedule`, `FixedCadence`.
- **Explicit-state models:** `models.px.base` — `SteppablePrognosticModel`,
`StepResult`, `ModelState`, `RNGState`, `StateHook`, `TransitionHook`.
- **Graph wiring:** `earth2studio.coupling` exports — `Binding`,
`TransformSpec`, `ProviderRegistry`.

**Internal:** the concrete `SingleModelSession` loop; `coupling.contracts` —
`ExecutionGraph` (users obtain one from `couple()`), `BoundProvider`,
`ActivationContext`, `GraphSnapshot`,
`ConnectorSnapshot`.

## Responsibility boundary

```text
Pipeline   distribution, member grouping, retry, progress, snapshot storage, IO
  -> ExecutionPlan.open(item, snapshot) -> RunSession.run() -> OutputEvent
Plan producers (any of):
  SingleModelPlan(model, source)      one component, one source
  ExecutionGraph.compile(providers)   many components, bindings
  LoopPlan(run, output_ports=..., identity=...)   a custom generator
  a hand-written ExecutionPlan
```

Pipeline reads plan metadata for routing, sizing, and resume; it never sequences
components or branches on graph topology. The graph never distributes work,
writes output, or tracks progress. Pipeline depends only on the `ExecutionPlan`
protocol, so a compiled graph plan and a hand-written one are interchangeable to
it -- see the dependency rule under Layers.

**Every Pipeline variation point is a constructor parameter or a** `RunSession`**.**
Distribution, ensemble grouping, retry, progress storage, and output handling are
injected strategies; a new execution pattern is a session. Pipeline is never
subclassed itself.

## Plans

A plan is **item-agnostic**: built once, opened per work item, so Pipeline can
compile before distributing. It binds **sources, never data**; the session
fetches field values, including the initial condition. Item-specific
information takes the item: `external_requests(item)` (initial conditions and
forcing, read by predownload without opening a session) and `open(item, snapshot)`.

A plan also reports `identity`, `output_ports`, `describe()`,
`checkpoint_capability` (weakest across what it runs), and
`supports_member_batching` (`False` forces member groups of one). Output schemas
are declared before execution, but do not promise an exact event
count: early stopping is allowed. Declarative plans validate static inputs,
units, grids, cadences, and cycles before compute. Custom loops may resolve
inputs conditionally during execution.

`external_requests(item)` returns a complete tuple of requests, or `None` when
the complete set cannot be known upfront. `()` means no external inputs; it is
not interchangeable with `None`. Complete predownload requires a known set and
must reject `None`; ordinary execution remains available. Do not add a separate
capability flag for information already expressed by this return value.

`LoopPlan(run, output_ports=..., identity=..., requests=...)` supplies the plan
boilerplate for a generator taking a `WorkItem` and yielding `OutputEvent`s.
Requests default to unknown, member batching is disabled, and checkpointing is
unsupported. The caller versions the identity when code, configuration, or
schemas change. A fresh invocation owns its state and releases resources on
completion or generator close. Stateful resumable custom execution implements
the existing `ExecutionPlan` / `RunSession` protocols directly.

## Sessions and events

`RunSession` has three members: `run()` yields `OutputEvent`s in production
order; `snapshot()` captures resume state; `checkpoint_boundary` says whether a
snapshot taken now is restorable.

`OutputEvent(component, port, produced_at, data)`: `produced_at` is graph
availability time; reference, lead, and valid time stay coordinates on `data`.
The buffer must not change while Pipeline or an IO backend owns the event.

## Snapshots (for checkpointing)

`RunSnapshot` holds only what Pipeline reads:

- `compatibility` — schema, Earth2Studio, plan, and component versions, readable
without decoding the payload. Restore rejects any mismatch; the initial promise
is same-version continuation.
- `output_cursor` — after a restore, a session re-emits no event committed at the
cursor and skips none that was not.
- `payload` — opaque to Pipeline. Scientific state lives here
(`GraphSnapshot` for a graph).

The plan owns the codec (`encode_snapshot` / `decode_snapshot`). Pipeline chooses
checkpoint frequency; the session declares where a boundary is legal, and a plan
reporting `UNSUPPORTED` never has one.

## Default single-model execution

`SingleModelPlan(model, source, transforms=())` wraps the model in a
`PrognosticComponent` and takes cadence, ports, requests, and capabilities from
its spec. The direct session steps the component once per scheduled activation,
publishing the initial condition and then each step through the horizon,
matching `run.deterministic`. Its payload is a `ComponentSnapshot`.

Common customization uses `OutputTransform(apply, identity, ports=None)`:

- `apply` maps port-keyed arrays to port-keyed arrays. Transforms compose in
order and return every declared port once per activation.
- `ports`, when provided, maps the input declarations to output declarations.
Omit it for schema-preserving operations such as masking.
- `identity` is caller-owned and includes behavior and configuration. It enters
plan identity so incompatible transform configurations cannot share snapshots.

Transforms are stateless and must not mutate borrowed inputs. Their results are
published, never fed back into recurrent state. Changes to model state belong in
explicit model hooks. Stateful diagnostics, conditional fetching, altered
schedules, and early stopping belong in a custom loop/session or a graph.
The built-in transform path deliberately does not implement another scheduler.

A graph may wrap the same output transform as a diagnostic; users should not
have to rewrite simple transforms into graph vocabulary. The concrete
`SingleModelSession` is not a subclass extension contract.

`PrognosticComponent` resumes by re-seeding the iterator from the last
published step, which is exact only for a single input lead time with identical
input and output variables; other models report `UNSUPPORTED` until phase 2.
RNG state is not captured and member perturbation is not implemented
(`supports_member_batching=False`). Transforms cannot keep hidden state because
that state would not appear in the component snapshot.

`dev/examples/05_single_model_session.py` demonstrates the default path, output
masking, a derived stream, and a bespoke loop with conditional inputs and early
stopping. None requires a session subclass or a graph import.

## Explicit model state

**TBD: This may be handled by a general update to core** `PrognosticModel` **protocol
to support an** `initialize` **+** `step` **API along with support for multiple** `DataArray`
**inputs/outputs/coords.**

`SteppablePrognosticModel` (`initialize`, `step`, `state_coords`) is an optional
model capability for restart and step-time imports. Adapters cannot provide it:
today's recurrent state lives in generator frames (`_default_generator` locals,
DLESyM's `_next_step_inputs` history), which can be neither serialized nor
given imports. For most wrappers the state is the yielded DataArray;
DLESyM's history is why `ModelState` is a mapping.

- `StepResult` carries `state`, port-keyed `outputs`, and `rng` (`None` when
deterministic). Multiple ports are required when outputs differ in grid or
cadence, as DLESyM's atmosphere and ocean do.
- `step` is hook-free and takes borrowed, read-only imports keyed by slot.
Iterators apply a `StateHook` before and a `TransitionHook` after. Each
wrapper must state whether a rear-hook change alters recurrent state; SFNO
and DLESyM differ today.
- Iterator-only models remain supported without restart or imports. The
conformance suite must not require the capability.

See addendum at the end for draft on the above items, though they're likely to be superseded.

## Components

The unit shared by the built-in single-model and graph executors. Custom loops
do not need components. Declarations are metadata-only and constructible without
weights or data.

- `FieldRequirement` — slot, variables, signature, `initialize`/`step` phase,
schedule, lead offsets, optional flag, named fallback, freshness. Derive it
from the model contract where possible rather than restating it per adapter.
- `OutputPort` — one stream of valid outputs. Different grids, cadences, or
lead patterns need separate ports; no structural filler.
- `ComponentSpec` — requirements, ports, activation schedule, checkpoint and
member-batching capability.
- `ComponentAdapter.open(initial)` → `ComponentSession`, which steps one
activation, returns port-keyed outputs, snapshots to a `ComponentSnapshot`, and
`close()`s without losing state. `initial` may be empty when a snapshot is
restored before the first step.
- `PrognosticComponent` adapts any iterator-based prognostic: one
`initial_condition` requirement, one `forecast` port, the model's step as
cadence. Activation `k` publishes step `k`. It takes no step-time inputs.



## Graph wiring

- `Binding` — source port to target slot, selected fields, `current`/`previous`
availability, and fingerprinted transforms that resolve on metadata before
touching values.
- `BoundProvider` (internal) — presents a resolved source or component output
synchronously; `None` means declared optional absence, never an error.
- `GraphSnapshot` (internal) — a graph plan's run payload: per-component
`ComponentSnapshot`s, connector state, and event cursor.

Per-member state carries a labelled leading member dimension. Member IDs and RNG
streams stay stable across regrouping and world-size changes. Variable names use
the shared lexicon; `units` attributes mirror it but are not authoritative.

## Schedules

`Schedule.iter_between(reference_time, start, stop)` yields activations in
`[start, stop)` lazily; `fingerprint()` is stable and feeds plan identity.
`FixedCadence(step, offset)` exists; irregular and forecast-relative schedules
will join it in `run.schedules`.

## Phasing

Each phase leaves a working single-model path; none waits on the next.

1. **Now.** `SingleModelPlan` runs one `PrognosticComponent` with a hand-written
  loop. Shared component primitives live in `run`; graph mechanics stay in
  `coupling`. `LoopPlan` supports bespoke generators without components.
2. **Explicit state.** `PrognosticComponent` uses `SteppablePrognosticModel`
  when a model provides it: exact resume for multi-lead models and step-time
  inputs (forcing). The plan and session interfaces do not change.
3. **Compiler.** Implement graph compilation and execution in `coupling`,
  reusing component declarations and stepping primitives. Preserve the direct
  single-model loop and compare behavior through the common session boundary.
4. **Diagnostics.** Graph diagnostics may reuse output transforms. Add dedicated
  stateful or independently scheduled diagnostics when needed; simple output
  customization does not require a graph.



## Open decisions

1. **Dependent work items** (warm-start cycling) cannot be expressed below
  Pipeline. Support it in v1 as a first-class dependency notion, or declare it
   out of scope? No `recipes/eval/` pipeline needs it today.
2. **Lagged exchange:** `Binding.availability` versus `Impact_modeling.md`'s
  positional run sequence. Pick one before the first coupled implementation.
3. **Async providers:** if live sources need an async session, change provider,
  session, and Pipeline loop together.
4. **RNG:** explicit `RNGState` versus the model contract's `set_rng` seeding.
5. **Checkpoint/commit ordering** and cursor encoding.
6. `ProviderRegistry` registration types; `ComponentSnapshot` field for
  non-array adapter metadata.



## Integration checks

1. A hand-written session runs, snapshots, and resumes without importing
  `coupling` — `test/run/test_session.py`.
2. A synthetic graph: stateless component, one source input, resumable cursor.
3. A DataArray model matches through direct iteration, `SingleModelSession`, and
  Pipeline — session half in `test/run/test_single.py`.
4. A requirement switches between source and component provider by binding only.
5. Split DLESyM atmosphere and ocean reproduce the fused model with real weights.
6. A slower impact component consumes forecast output and irregular,
  freshness-limited observations.
7. A coupled session resumes all state after a world-size change.
8. A `LoopPlan` conditionally fetches inputs and stops early through the same
  supervisor, without component or graph declarations — session half in
  `test/run/test_session.py`. Unknown requests disable complete predownload only.
9. Composed output transforms resume without changing recurrent state —
  `test/run/test_single.py`.

Shared-type changes land in this spec and the package together.


### Addendum: `SteppablePrognosticModel`

Code sketch:
```


# Explicit-state capability, used by restartable and coupled execution.

ModelState: TypeAlias = Mapping[str, xr.DataArray]
"""Named, labelled recurrent model state. Entries may have different grids."""

RNGState: TypeAlias = Mapping[str, xr.DataArray] | None
"""Explicit serializable RNG state; ``None`` denotes a deterministic model."""


@dataclass(frozen=True)
class StepResult:
    """New state, published outputs, and RNG state from one model transition.

    ``outputs`` is keyed by declared output port name. A model that publishes on
    several grids or cadences -- DLESyM's atmosphere and ocean being the case
    that forces this -- returns one entry per port rather than one fused array.
    """

    state: ModelState
    outputs: Mapping[str, xr.DataArray]
    rng: RNGState


StateHook: TypeAlias = Callable[[ModelState], ModelState]
"""Iterator hook applied to recurrent state before a step."""

TransitionHook: TypeAlias = Callable[[StepResult], StepResult]
"""Iterator hook applied to both outputs and recurrent state after a step."""


class SteppablePrognosticModel(Protocol):
    """Optional explicit-state capability for coupling and restartable inference.

    Recurrent state in an iterator-based wrapper lives in a live generator frame,
    where it can be neither serialized nor handed step-time imports. A model that
    needs restart or step-time coupling therefore has to externalize its state;
    an adapter cannot do it on the model's behalf. This protocol is that
    externalization and nothing more.

    It is a capability, not a replacement for :class:`PrognosticModel`. Iterator-only models
    remain fully supported for ordinary inference through an adapter, with
    reduced capability: no step-time imports and no restart.
    """

    def initialize(self, x: xr.DataArray, *, rng: RNGState) -> StepResult:
        """Create labelled recurrent state and the initial published outputs."""
        ...

    def step(
        self,
        state: ModelState,
        imports: Mapping[str, xr.DataArray] | None = None,
        *,
        rng: RNGState,
    ) -> StepResult:
        """Advance once using borrowed, read-only imports keyed by slot.

        Hook-free, like ``__call__``. A steppable iterator applies a
        :data:`StateHook` before the transition and a :data:`TransitionHook`
        after it. Whether a post-transition change also alters recurrent state is
        a per-wrapper decision that must be stated explicitly -- wrappers differ
        on this today, and the runtime cannot infer the projection.
        """
        ...

    def state_coords(
        self, input_coords: CoordinateSystem
    ) -> Mapping[str, CoordinateSystem]:
        """Declare each state entry's coordinate signature without values."""
        ...
```
