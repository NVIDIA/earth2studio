# Prognostic Model Contract

Read [PrognosticModel](../../../earth2studio/models/px/base.py) for the public
interface and [PrognosticMixin](../../../earth2studio/models/px/utils.py) for the
optional helpers. This file contains the behavioral contract for new wrappers.
Legacy wrappers and drivers may still follow older iteration semantics.

## Execution and Explicit State

Every wrapper declares fixed named parameters for `__call__`, `initialize`,
`step` and `create_iterator`. Protocol `*x`/`*y` notation expresses differing
fixed arities across models; do not copy it into concrete wrappers. Generic
internal forwarding helpers and caller-side tuple unpacking are allowed.

| Method | Arguments | Result |
| --- | --- | --- |
| `__call__` | Initial fields, then full forcing windows | First forecast |
| `initialize` | Same as `__call__` | First forecast and state |
| `step` | Previous output slots, then newest dynamic forcing, then `state` | Next forecast and state |
| `create_iterator` | Same as `initialize` | Generator of complete forecasts |

Each array slot is a separate positional DataArray. `step` declares `state` as
a positional-or-keyword parameter after the arrays; generic callers use
`state=state`. One output is returned directly; multiple outputs form a tuple.
Return `(y, state)` from initialization and stepping, even when `state is None`.
Do not introduce a required state base class or a `.fields` property.

`initialize` and `step` each perform one core computation. `y` always contains
all declared output variables and lead times, including diagnostics. Keep
everything else needed for continuation in state: older input/forcing history,
caller-supplied statics, latents, noise and RNG position. Weights/configuration
and checkpoint-provided statics stay on the model.

For two-frame input `[-6h, 0h]`, initialization returns `y(+6h)` and retains
`x(0h)`; the next step returns `y(+12h)` and retains the previous `y(+6h)`.
Reconstruct windows by selecting input variables by label and the last needed
lead times, not variable positions. Do not duplicate current `y` in state unless
published fields are clipped/cast/postprocessed and unsuitable for recurrence.

`step` is pure: it changes neither outputs nor state passed to it, and replaying
the same `(y, state)` reproduces the next result. The pair is a complete
checkpoint and supports independent branches. State may contain arrays, tensors,
DataArrays, scalars, dataclasses, tuples, or `None`; save RNG state/counters,
not live generators. A checkpoint resume computes the next forecast, not the
saved one. Test serialization of both fields and state; do not assume an existing
checkpoint utility already supports every payload type.

The mixin provides no-forcing/no-source defaults, `stochastic=False`, hooks and
private helpers. It defines no public execution methods or abstract stubs; do not
delegate execution to `super()`. Explicit wrappers
may delegate `__call__` to `_default_call(x, ...)` and `create_iterator` to
`_default_create_iterator(x, ...)`; implement `initialize` and `step` yourself.
Direct implementations without the mixin must obey the same behavior.

## Optional Checkpoint Integration

Ask whether the user wants integration with Earth2Studio's checkpoint system.
Recommend **no for the initial implementation**: persistence adds payload/schema
design, device restoration and restart testing. Without an opt-in, omit checkpoint
bindings, disk save/restore logic and checkpoint-specific tests. Still implement
the required explicit state and replay semantics above; a resumable `(y, state)`
does not automatically integrate with checkpoint storage.

If requested, consult the [checkpoint guide](../../../docs/userguide/advanced/checkpointing.md)
and [implementation](../../../earth2studio/utils/checkpoint.py):

- `Checkpoint` manages a named run's restart catalog; `NullCheckpoint` is the
  no-op fallback. Components opt in through `bind_checkpoint_state` with a
  component-specific dataclass. Bind inside the active context when construction
  depends on restored state; duplicate dataclass identities in one session collide.
- Agree on supported levels: 0 records workflow progress only, 1 supports restart
  of a workflow item, and 2 supports restart within a rollout. Do not claim a level
  the wrapper cannot restore completely.
- Keep immutable weights out of restart state. Coordinate with the workflow's IO
  so the full forecast fields and continuation state needed for `step` are
  recoverable, even when ordinary outputs save only selected variables.
- Use the existing pickle-free serializer. It supports dataclasses, scalar and
  container values, tensors and non-object NumPy arrays. DataArrays are not a
  supported native payload: explicitly encode and restore their data, coordinates
  and metadata using supported types. Verify the actual payload against the
  serializer rather than assuming protocol state is directly supported.
- Keep `initialize` and `step` independent of ambient checkpoint restoration.
  Restore the saved pair at the orchestration boundary and call `step` for the
  next forecast. Record progress only after successful forecast IO, then flush.
- Test an on-disk restart with a fresh component, matching uninterrupted fields,
  lead times and RNG continuation; also test disabled checkpointing. Use the
  checkpoint API rather than pickle round trips.

## Forcing and Slots

`input_coords()` and `output_coords(...)` return a single allocation-free
signature or a tuple. Coordinate planning receives one separate positional
signature/DataArray per input slot, using fixed named parameters like execution.
`forcing_coords()` returns one signature, a tuple, or `None` for unforced models.

- Inputs are fetched once. Initial arguments concatenate input slots then forcing
  slots, preserving declaration order; this concatenates arguments, not contents.
- Forcing is explicitly declared, even if it shares grid/variables with outputs.
  Supply complete relative forcing windows to initialization.
- A dynamic forcing slot has `lead_time`. Each step needs its final declared
  offset shifted by each lead of the `y` being advanced. Select matching frames
  by label; extra frames are harmless. Retain older forcing history in state.
  For example, a `[0h, 6h]` forcing window needs `y.lead_time + 6h` next.
- Caller-supplied static forcing has no `lead_time`. Store it at initialization;
  omit that slot from every step and generator send thereafter.
- Missing/invalid forcing raises `ValueError`; never fetch, silently reuse stale
  forcing or infer provider names from parameters. Fixed required Python arguments
  naturally raise `TypeError` if omitted; validate supplied payloads, and helper
  slot counts/iterator sends report `ValueError`.
- Split slots when coordinates differ. No two output slots may share identical
  non-variable coordinates. State-variable output slots come first in input-slot
  order; output-only slots follow. Include same-grid/time diagnostics as variables
  in the existing output slot.
- Slot order is append-only API. Signature `.name` is display-only. Automation
  matches providers by variables, grid and valid time, not parameter names.

An example fixed API with two state slots, static terrain and dynamic forcing:

```python
def __call__(self, atmosphere, ocean, terrain, radiation):
    return self._default_call(atmosphere, ocean, terrain, radiation)

def create_iterator(self, atmosphere, ocean, terrain, radiation):
    yield from self._default_create_iterator(atmosphere, ocean, terrain, radiation)

# Implement initialize(self, atmosphere, ocean, terrain, radiation), retaining
# terrain in state, and step(self, atmosphere, ocean, radiation, state).
# Both return ((next_atmosphere, next_ocean), next_state).
```

`default_sources()` is required. Return a raw `DataSource`/`ForecastSource` for
one total slot, a tuple with one source or `None` per input-then-forcing slot, or
top-level `None` for no recommendations. The mixin returns `None`. Drivers use
`earth2studio.models.utils.recommended_sources`, fetch and validate data. A source
on another grid may be composed with a recommended regridder; intrinsic model
transforms stay inside the wrapper. Models never fetch their recommendations.

## Coordinates and Core Boundary

Create allocation-free DataArray signatures with `coord_array()`. Read `.dims`,
`.sizes` and coordinate labels, never field `.values`. Dynamic axes form an
explicit zero-sized leading prefix via `dynamic=`; fixed empty axes are not
wildcards. Configure grid/domain/variables before planning and keep them fixed.
Concrete fields may have arbitrary leading axes or none. Resolve several dynamic
dimensions together or right-to-left.

Use `grid=` for geometry, grid ID, CRS and geographic auxiliaries. Regular public
latitude runs north-to-south; longitude normally spans `[0, 360)`. Flip internally
for differently ordered cores. Use Earth2Studio variable names and qualified
statistic labels such as `tp:sum:6h`.

Initial field history is finite, nonempty, explicitly labelled, 1-D timedelta,
strictly increasing and ends at relative zero. Validate labels before subtracting
the final lead for `handshake_dataarray`. Add output offsets to the final input
lead, not a constant; nonzero starts and shifted rollouts must work. Forcing may
include positive relative offsets when the core requires target-time fields.

`output_coords` accepts declarations or actual fields without reading field data
or modifying input metadata. Use `coord_array_like` for new lead times/variables;
rebuild any dropped dependent auxiliaries. Spatial replacements require a fresh
grid-backed signature. Invalid dimensions, labels or metadata raise `ValueError`.
Use `handshake_time` for time labels and `handshake_nonempty` at execution.

Fields have data on the model device (normally NumPy CPU/CuPy CUDA); document any
movement. Convert via `.e2s.to_torch()` and `from_torch(output, signature)` at the
core boundary. Preserve structural metadata and original leading dimensions;
returned fields omit signature-only attrs. `batch_func` currently handles an
array-to-array numerical method, not arbitrary `(y, state)` results: decorate
the private core helper, not `initialize`/`step`. Multi-slot cores may need
per-slot batching. Do not pack the outer iterator: hooks see original dimensions.

## Iteration, Hooks and Ownership

`create_iterator` returns a `Generator` whose first yield is the initialization
forecast. `nsteps` forecasts take `nsteps` yields. A multi-lead core returns one
whole chunk per step/yield; no partial yields or `front_hook_interval` scheduling.
Drivers publishing the initial condition do so separately using
`earth2studio.models.px.utils.initial_condition(x)`.

After the first `next(iterator)`, send the next step's dynamic forcing as one
DataArray or a tuple in declared order, with static slots omitted. `next(it)`
equals `send(None)`, valid only when no new forcing is needed. Never splat a
single DataArray; normalize it to `(y,)` before generic output unpacking.

Hooks apply only in the iterator, never in `__call__`, `initialize` or `step`:

1. Initialize, apply `rear_hook(y)`, yield its result.
2. Receive/validate forcing, apply `front_hook(y)`, step, apply rear hook, yield.

Each hook accepts/returns the complete output payload (array or tuple), once per
core computation in original leading dimensions. Both returned values feed
recurrence. Hooks receive `y` directly with no defensive copies; an in-place
front hook can edit the previously yielded array. `clear_hooks()` restores
identity. Compose multiple transformations in one callable. Neither hook sees
initial input or private history; edit those explicitly outside iteration.

Models borrow inputs and own outputs. Core execution never modifies caller
inputs, state, coordinates, attrs or encoding, and never overwrites earlier
yields. Clone borrowed tensors before in-place cores. In-place hook edits are
the explicit exception; callers retaining snapshots must copy before advancing.
Downstream IO/unit conversions must not mutate yielded fields because they feed
the next step. Document any latent state that cannot reflect arbitrary hook edits.

## Randomness

Declare a boolean `stochastic`; stochastic models implement
`set_rng(seed: int, reset: bool = True) -> None`. Constructors and loaders do not
take seeds. `reset=True` restarts the initialization stream; `reset=False` only
initializes it when absent. Never-seeded models may use global RNG.

Initialization advances the model's stream for each new rollout. Save that
rollout's seed/key and position in state; steps derive draws from state without
advancing hidden model RNG. Resetting the model mid-rollout must affect only new
initializations, never existing trajectories. Same seeded rollouts reproduce;
different seeds differ.

After seeding, seeding and execution must leave global RNG state unperturbed.
Use a local generator rebuilt from serialized state, a functional key, or a scoped
RNG fork for backends requiring global seeding. Fork only relevant devices and
numerical calls, never across yields. Global RNG forks do not isolate concurrent
threads; run those components serially within a process.

## Rules and Verification

| Rule | Requirement |
| --- | --- |
| P1 | Implements prognostic interface |
| P2 | Allocation-free declarations with explicit dynamic leading axes |
| P3 | Relative input history strictly increases and ends at zero |
| P4 | Planning does not mutate its argument |
| P5 | Invalid coordinates raise `ValueError` |
| P6 | Shifting input leads shifts output leads equally |
| P7 | Iterator yields forecasts only, starting with initialization output |
| P8 | First yield matches planned output coordinates |
| P9 | Every output matches planned coordinates and structural metadata |
| P10 | Both hooks apply only in the iterator |
| P11 | Boolean stochastic declaration |
| P12 | Stochastic models implement `set_rng` |
| P13 | Seeding determines rollouts; different seeds differ |
| P14 | Seeded execution and seeding preserve global RNG state |
| P15 | Execution does not mutate borrowed inputs or coordinates |
| P16 | Advances preserve earlier yields, except explicit in-place hook edits |
| P17 | One positional array per slot, fields then forcing |
| P18 | No duplicate output slots with identical non-variable coordinates |
| P19 | Pure step and exact replay from `(y, state)` |
| P20 | Call, initialization forecast and first hook-free yield agree |
| P21 | Serializable complete continuation `(y, state)` |
| P22 | No internal fetching; invalid/missing forcing rejected |
| P23 | Single/tuple/None default sources in input-then-forcing order |
| P24 | Fixed named execution parameters |

Use `check_prognostic_contract(model)` with mock weights. It probes single/multiple
slots, fixed signatures through decorators, forecasts-only iteration, forcing,
state replay, hooks and ownership. Serialization (`P21`) is skipped and needs
component-specific tests when checkpoint integration is requested. The checker
returns unevaluated rules with reasons; pin those explicitly.
`rollout=False` intentionally disables execution
checks and is not a substitute for full conformance. Keep model-specific tests for
numerical correctness, source ordering and absence of internal fetching, which
generic probes cannot prove. Update drivers that count `nsteps + 1` yields when
integrating migrated models; runtime protocol checks alone cannot catch this.

Tests use mock weights, numerical expectations, invalid/empty inputs, shifted
leads, leading axes, metadata and ownership checks. Stochastic mocks must exercise
real sampling paths. Real-weight tests are marked `package`; dependency skips
are not evidence of conformance.

## Packaging

For packaged weights use `AutoModelMixin`, immutable package revisions,
`Package.default_cache` and `package.resolve`. Load on CPU, set `eval()` and
disable gradients; use `weights_only=False` only for required pickled objects.
Guard optional backends with `OptionalDependencyFailure` and
`check_optional_dependencies` from `earth2studio.utils.imports`. Register Torch
buffers for normalization/device state; override `to` for ONNX/JAX runtime state.
Keep loading and core computation checkpoint-specific.
