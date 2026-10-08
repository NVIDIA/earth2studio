# Diagnostic Model Contract

The public interface is [DiagnosticModel](../../../earth2studio/models/dx/base.py).
This reference supplies the behavioral requirements needed to implement it.
Read the public method docstrings alongside this file; runtime protocol membership
checks member presence, not correct signatures or execution.

## Inputs, Outputs and Sources

- `input_coords()` returns one allocation-free `CoordinateSystem` or a tuple of
  signatures in input-slot order.
- `__call__` takes one positional DataArray per slot. Declare a fixed number of
  named parameters (`x`, or descriptive names for complex models), not `*x`.
  Variadic protocol notation describes different fixed arities across models.
- `output_coords(input_coords)` receives one signature/DataArray or one tuple of
  them, and returns one signature or a tuple of output signatures. Execution
  returns one DataArray directly, or a tuple for multiple output slots.
- Split slots when coordinates differ. Combine variables with identical
  non-variable coordinates into one slot. Slot order is append-only API;
  signature `.name` is display-only. Providers match by variables, grid and time,
  not Python parameter names.
- `default_sources()` is required. Return a raw `DataSource` or `ForecastSource`
  for one slot, a tuple containing one source or `None` per input slot, or
  top-level `None` for no recommendations. Drivers use
  `earth2studio.models.utils.recommended_sources`; models never fetch.
- A recommended source may be composed with its regridder. Provider-independent
  transforms intrinsic to the model remain inside the wrapper.
- `to(device)` returns the model and moves weights and non-Torch runtime state as
  needed. `torch.nn.Module` normally supplies it. Inheritance from the protocol or
  a mixin is not required. Declare `stochastic: bool = False` for deterministic
  diagnostics; do not assume a diagnostic mixin exists.

No `forcing_coords`, `initialize`, `step`, `create_iterator` or iterator hooks
belong to this API. Temporal axes are permitted when the diagnostic uses them.

## Coordinates and Ownership

`CoordinateSystem` is a shape-only xarray DataArray made with `coord_array()`.
Read `.dims`, `.sizes` and coordinate labels through `.coords`; never read field
`.values` on a signature. Planning works on declarations and concrete inputs
without allocating field data or mutating either.

Declare dynamic axes explicitly as a zero-sized leading prefix via `dynamic=`.
Concrete inputs may have arbitrary leading axes or none. All fixed trailing
dimensions, labels, grid/CRS and statistics metadata must match. Empty fixed axes
are not wildcards. Resolve multiple dynamic axes together or right-to-left.
Configure grid/domain and variables before planning, and keep them fixed for a run.

Use `coord_array(grid=...)` for registered or configured geometry, including
projected/curvilinear geographic auxiliaries. Public regular latitude runs
north-to-south; longitude is normally `[0, 360)`. Flip tensors internally if the
backend differs. Use Earth2Studio vocabulary and qualified labels such as
`tp:sum:6h` for temporal statistics.

`handshake_dataarray` validates dimensions, labels and structural metadata;
use `handshake_time` for temporal labels and `handshake_nonempty` at execution
boundaries. Invalid coordinates raise `ValueError`. `coord_array_like` preserves
leading axes and unaffected metadata. Replacing variables or lead time drops
dependent auxiliaries; rebuild them if needed. Spatial changes require a fresh
grid-backed `coord_array`. Generative outputs declare `sample` after the original
leading dimensions.

Fields use array data on the model device (normally NumPy on CPU, CuPy on CUDA).
Document any device movement. Convert at the Torch boundary with
`.e2s.to_torch()` and `from_torch(output, signature)`, which strips signature-only
attributes and preserves field metadata. Use `batch_func` only for a compatible
array-to-array core; multi-slot shapes may need explicit per-slot batching.
Preserve labels, auxiliaries and original leading dimensions.

Inputs are borrowed; outputs belong to the model. Never modify input values,
coordinates, attrs or encoding. Clone borrowed tensors before in-place kernels,
and do not return storage that later calls overwrite.

## Randomness

Stochastic models declare `stochastic = True` and implement
`set_rng(seed: int, reset: bool = True) -> None`. No constructor or loader seed:
load first, then seed. `reset=True` restarts the stream; `reset=False` initializes
only an absent stream. Calls advance it. Never-seeded models may use global RNG.

After seeding, repeatable seeds reproduce outputs, different seeds differ, and
neither seeding nor execution leaves global RNG state perturbed. Use a local
generator/functional key or a scoped RNG fork for backends without generator
injection. Fork only the relevant devices and numerical call. Global RNG forks
isolate sequential execution, not concurrent threads.

## Rules and Verification

| Rule | Requirement |
| --- | --- |
| D1 | Implements the diagnostic interface, including `default_sources` |
| D2 | Allocation-free signatures with explicit dynamic leading dimensions |
| D3 | Coordinate planning does not mutate inputs |
| D4 | Invalid coordinates raise `ValueError` |
| D5 | Outputs match planned coordinates and structural metadata |
| D6 | Execution does not modify input data or coordinates |
| D7 | Boolean stochastic declaration |
| D8 | Stochastic models implement `set_rng` |
| D9 | Same seeds reproduce output; different seeds differ |
| D10 | Seeded execution and seeding preserve global RNG state |
| D11 | Fixed named public `__call__` parameters |

Use `check_diagnostic_contract` with existing mock fixtures for supported checks.
The checker is still single-slot and does not enforce D11; test multi-slot
behavior and `inspect.signature` directly. For single-slot deterministic models,
the expected skip is `D10: model does not declare itself stochastic`, not an empty
list. Pin legitimate skips with reasons, never suppress actual violations.
Verify numerical results, invalid/empty inputs, arbitrary leading axes, metadata,
input nonmutation, source ordering and RNG isolation. Package tests load real
weights only when explicitly enabled; dependency skips are not conformance.

## Packaging

Simple derived diagnostics normally need only `torch.nn.Module`. Packaged and
generative models add `AutoModelMixin`, `load_default_package` and `load_model`.
Use immutable HuggingFace revisions (`hf://org/repo@commit`) or versioned registry
assets, `Package.default_cache` and `package.resolve`. Load on CPU, set `eval()`
and disable gradients. Use `weights_only=False` only for required pickled objects.

Use `OptionalDependencyFailure("model-extra")` and
`check_optional_dependencies` from `earth2studio.utils.imports` for optional
backends. Add an approved named extra (even empty), `all`, install documentation
and API references. Register normalization/device buffers; override `to` only
when state such as ONNX sessions or JAX arrays also needs movement.
