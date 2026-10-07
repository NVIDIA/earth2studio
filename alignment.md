# Alignment Across the Three Earth2Studio Design Proposals

**Status:** Draft
**Date:** 2 September 2026
**Covers:** `cupy_design.md`, `pipelines.md`, `Impact_modeling.md`

## 1. Why this document exists

The three proposals read as independent projects. They are not. Each one
separately specifies a grid registry, a temporal aggregation vocabulary, a way for
a model to declare what external data it needs, a regridding interface, and a
revision to the IO backend protocol. Three implementations of those five things
would be the most expensive outcome available to us.

This document records what is shared, resolves the three places where the
proposals actively contradict each other, lists the concrete edits each proposal
needs, and gives a sequencing that lets all three run in parallel.

## 2. The shared foundation

Five items are claimed by more than one proposal. Each must have exactly one
implementation and one owner.

| Shared item | Claimed by | Notes |
| --- | --- | --- |
| Grid registry with coordinate reference systems | All three | The CuPy plan designs it, the execution plan needs it for offline predownload planning, the coupling plan needs it for regridding and reprojection |
| Temporal aggregation vocabulary | CuPy plan, coupling plan | The CuPy plan's `mean:24h` modifier and the coupling plan's cell-method recipes describe the same thing |
| Variable name and unit vocabulary | Coupling plan, existing `earth2studio.lexicon` | The coupling plan already flags this as a seam that must reconcile to one vocabulary |
| Declaration of required external data | Execution plan, coupling plan | The execution plan puts it on the model, the coupling plan puts it on a wrapper |
| Regridding interface | Execution plan, coupling plan, CuPy plan | The CuPy plan reaches it indirectly through interpolation parity |

## 3. Resolved conflicts

### 3.1 Differentiable coupled exchange is deferred, not foreclosed

The coupling plan requires gradients to survive the exchange between components.
The CuPy plan lists gradient-preserving conversion as a non-goal. Deferring the
gradient requirement is acceptable, and adopting the CuPy plan's selection,
aggregation, and grid features does not close the door, provided one rule holds.

**Rule: every accessor operation that touches array values must have a companion
resolver that returns a plan without touching values.**

Selection, temporal aggregation, and regridding are label operations. Turning a
twenty-four hour mean into a start time, an end time, and a reduction dimension is
coordinate arithmetic. Turning a variable selection into gather indices is
coordinate arithmetic. Turning two grid descriptions into interpolation weights is
coordinate arithmetic. None of it touches the payload, so none of it can sever an
autograd graph. Only the apply step does.

The CuPy plan already separates statistic resolution from statistic application.
That separation needs to become a stated architectural rule rather than a
convenient accident. If it holds, adding differentiable exchange later means
writing a Torch apply function per operation against plans that already exist. If
it does not hold, it means redesigning every connector.

Two supporting guardrails:

- Keep the explicit failure on gradient-requiring tensors in the Torch conversion
  helper. Silent detachment yields zero gradients with no error, which is the
  failure mode that would actually cost months.
- Replace the hard NumPy-or-CuPy type check in the batching helper with a dispatch
  point. CuPy has no autograd at all, so a CuPy payload inside a differentiable
  region is a dead end by construction. Dispatch leaves room for a Torch-backed
  labelled array later, which is the only path that gets labels and gradients at
  once.

### 3.2 A model declares what it needs; a wrapper decides how it runs

The execution plan and the coupling plan are not actually competing. They collide
because both bundle declaration with execution.

Only the model knows that it needs conditioning data. If that knowledge lives on a
coupling wrapper, then predownload tooling, an offline run, and a coupled run each
restate it separately and drift apart. **Declaration belongs on the model.**

The coupling plan's three arguments against extending the model contract are all
about execution, not declaration. Push delivery into an import slot, advancing one
timestep, and holding private state genuinely do not fit a tensor-in, tensor-out
signature, and a neutral wrapper is the right answer for those. **Execution belongs
on the wrapper.** A data-source component declares nothing and a partner model gets
an empty default, so the requirement that outside models join without forking
survives intact.

One declaration, two resolvers. The execution pipeline resolves each requirement to
a data source, fetched live or prefetched or read from a predownloaded cache. The
coupling driver resolves the same requirement to an edge from another component and
validates the graph before any compute.

Three extensions are needed for one declaration to serve both:

1. **Separate the static declaration from the expanded request list.** The
   execution plan's current signature takes an initialization time and a step count,
   which only produces concrete requests. Graph validation runs before any time is
   chosen, so field names, grid, cadence, and lead offsets must be readable
   statically.
2. **Carry an aggregation term.** A requirement for daily-summed precipitation is
   simultaneously a data request and the connector the coupling driver synthesizes.
   This is the strongest single argument that the shared vocabulary is one artifact.
3. **Mark requirements optional, with a fallback policy.** Observation nudging runs
   only when an observation is present and recent. The execution plan's version has
   no notion of an optional input.

Naming caution: the current name implies fetching from an archive, but a two-way
coupled import is fed by another model. The declaration should describe the field
requirement, not its source.

### 3.3 Explicit regridding replaces the interpolation argument

Agreed as proposed. The interpolation argument on the data fetch function is
dropped in favour of an explicit regridding interface supporting both host and
device execution.

Two consequences:

- The regridder consumes grid descriptions from the shared registry, not raw
  latitude and longitude arrays. Otherwise grid identity gets rebuilt a fourth time.
- The regridder exposes its weights, not only an apply method. Interpolation
  weights applied as a sparse matrix multiply are differentiable for free in Torch,
  which connects this decision back to section 3.1.

Worth recording for the CuPy plan: device-side regridding is a genuine performance
argument, since a sparse weight application on device beats a host round trip. It
is a stronger justification than the data-movement claim currently in that plan's
motivation section, which does not survive counting the copies on each path.

## 4. Concrete changes to each proposal

### 4.1 CuPy labelled-array plan

**Remove**

- The completion criterion requiring interpolation parity with the legacy path.
  That API is being deleted, so parity with it is wasted effort.
- The grid examples that use the StormCast-specific horizontal dimension names.
  The execution plan normalizes those names like every other projected source.
- The shape-only placeholder array backing the coordinate constructor. Labelled
  arrays coerce their backing data during printing, comparison, alignment, and
  concatenation, so an object that raises on materialization will fail during
  ordinary operations. Declare coordinate contracts with real but empty arrays until
  there is evidence the allocation matters.

**Add**

- A written contract-validation function. The plan names one in pseudocode but
  never specifies it. It is the most important single artifact: ordered dimensions,
  one-dimensional index coordinates, preserved auxiliary coordinates, attributes
  that survive serialization.
- Preservation of two-dimensional latitude and longitude coordinates across the
  legacy conversion boundary. The current conversion keeps only one-dimensional
  dimension coordinates, so any curvilinear-grid data that passes through a legacy
  model loses its geolocation silently. This is more serious than the attribute loss
  the risk section already lists.
- A serializable representation for the batching bookkeeping. It is currently a
  Python dataclass stored in the attributes dictionary, which will fail the first
  time an array reaches a file format or the description helper.
- The plan-versus-apply rule from section 3.1, stated as a design principle.

**Reorder**

- Pull the grid registry and the temporal aggregation vocabulary forward, ahead of
  the model and IO adapter stages. Neither depends on device-backed arrays, both can
  be built against today's coordinate dictionaries, and both are what the other two
  proposals are waiting on.
- Push the model and IO adapter stages back. Nothing outside this proposal is
  blocked on them.

**Reframe**

- The claim that a labelled container simplifies the component protocols does not
  come true within this project. For its whole duration there are two model
  protocols, two IO protocols, and an adapter layer. Simplification arrives only
  after the legacy path is removed, which is explicitly out of scope. Say so.
- Replace the process-wide environment variable as the primary behaviour switch
  with an in-code setting plus a context manager, keeping the environment variable
  as an initializer only. A downstream package cannot otherwise know which mode its
  caller selected.
- Do not begin emitting the future-behaviour warning until the new path is at
  feature parity. Warning users to test a mode that cannot yet do what they need
  produces noise and no signal.

### 4.2 Execution and pipeline plan

**Add**

- The static half of the data-requirement declaration described in section 3.2,
  readable without an initialization time, with an aggregation term and an
  optional-input flag.
- A statement that the grid registry it requires is the shared one from the CuPy
  plan, not a second registry local to this proposal.

**Change**

- The regridding interface takes grid descriptions from the shared registry and
  exposes its weights.
- Note explicitly that the coupling proposal consumes the same data-requirement
  declaration through a different resolver, so the two efforts must review that
  interface together before either implements against it.

**Keep as written**

- The contract specification and conformance test as the first item. It is small,
  non-breaking, and a genuine prerequisite for both other proposals.
- The deferral of the pipeline abstraction to last.

### 4.3 Impact and coupling plan

**Change**

- Accept deferred differentiability for the exchange, conditional on the
  plan-versus-apply rule holding in the shared accessor. Record the condition, since
  it is the thing that keeps the deferral reversible.
- Derive component imports and exports from the model-side data-requirement
  declaration rather than restating them on the wrapper. The wrapper keeps override
  capability as an escape hatch for models that cannot declare.
- Use the shared temporal aggregation vocabulary for cell-method recipes instead of
  a coupling-specific one. The worked example's daily-sum and daily-mean recipes are
  the same objects the data layer resolves.
- Use the shared grid registry for connector regridding and reprojection.

**Keep as written**

- The component, connector, and driver core as the first item, together with the
  equivalence gate against the existing coupled model with real weights. This is the
  longest single piece of work in the whole program and the least dependent on the
  other two proposals. It can start immediately against today's tensor and
  coordinate interfaces.
- The insistence that time policies are documented modelling assumptions rather
  than silent framework defaults.

**Defer**

- Conservative regridding until the shared regridding interface exists, so it is
  implemented once and not twice.

## 5. Phasing

Three months, three teams, one shared foundation. The organizing principle is that
the other two proposals are blocked on *decisions* about the foundation, not on its
*implementation*, so they can code against an agreed interface well before it is
finished.

### September: decide the foundation, start the long pole

| Track | Work |
| --- | --- |
| Foundation | Grid registry with coordinate reference systems. Temporal aggregation vocabulary reconciled with the existing variable lexicon. Written contract-validation function. Data-requirement declaration interface agreed jointly by all three teams. |
| Execution | Iterator and coordinate contract specification plus its conformance test. Source combinators and the regridding interface lifted out of the evaluation recipe into core. |
| Coupling | Component, connector, and driver core, built against today's tensor and coordinate interfaces. Equivalence gate work begins. |
| Labelled arrays | Auxiliary-coordinate preservation across the conversion boundary. Serializable batching bookkeeping. Device placement path. |

**Hard gate, end of September:** the foundation interfaces are frozen. Grid
descriptions, aggregation vocabulary, and the data-requirement declaration stop
changing shape. If this slips, the other two tracks will build their own versions
and the merge cost becomes the dominant risk in the program.

### October: build against frozen interfaces

| Track | Work |
| --- | --- |
| Foundation | Registry populated with the production grids. Aggregation resolution and application, with the plan-versus-apply separation enforced by tests. |
| Execution | Models stop owning data sources and declare their needs instead, with a compatibility shim for one release. Work distribution, progress and resume store, ensemble grouping. |
| Coupling | Equivalence gate passed. Conservative regridding on the shared interface. Coupled ensembles. |
| Labelled arrays | Workflow internals converted to the labelled container with adapters at model and IO boundaries. Native protocols introduced, proven on two or three models including one on a curvilinear grid. |

**Coordination point, mid-October:** the IO backend protocol takes a single
coordinated revision covering labelled writes, lifecycle methods, and per-component
backends. Three separate revisions to the same protocol is the failure mode to
avoid.

### November: converge and document

| Track | Work |
| --- | --- |
| Foundation | Documentation. Migration guidance for the dropped interpolation argument. |
| Execution | Pipeline abstraction over the stabilized substrate. Existing recipe pipelines migrated off configuration blobs. |
| Coupling | First reference application end to end with a real forecast model. |
| Labelled arrays | Data fetch switches to the new default behaviour under the in-code setting. Warning goes live. Documentation. |

## 6. What does not fit by the end of November

Stating this plainly is more useful than a schedule that assumes it away.

- The coupling proposal lists seven strictly ordered items. Realistically the first
  three fit, which is the core, conservative regridding, and coupled ensembles, plus
  a start on the first reference application. Checkpoint and restart, the second
  reference application, and the application-layer hardening do not fit.
- The CuPy plan's default behaviour switch should not happen in November if the
  labelled path has not run in continuous integration for two normal releases. Its
  own text says this, and the November milestone conflicts with it. Treat November
  as switch-ready rather than switched.
- Removing the legacy tensor and coordinate path is outside all three proposals and
  should stay that way.

## 7. How we know it worked

- No team has implemented a second grid registry or a second temporal aggregation
  vocabulary.
- A model declares its external data needs in exactly one place, and both the
  execution pipeline and the coupling driver read that same declaration.
- Every value-touching operation in the shared accessor has a resolver that returns
  a plan, verified by test.
- Curvilinear-grid geolocation survives a round trip through a legacy model.
- The IO backend protocol changed once.
