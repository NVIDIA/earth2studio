# Run-sequence DSL and YAML reference

## Run-sequence DSL

Most systems use a sequence derived from components and connectors. Pass DSL
text to `Driver(sequence=...)` when action order is part of the configuration.

```text
@6h
  atmos -> med
  ocean -> atmos
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
```

### Grammar

The format is line-oriented:

- `@<interval>` opens a slot.
- `@` closes the current slot; end of input also closes it.
- `name` is `RunAction(name)`.
- `src -> dst` is `ConnectAction(src, dst)`.
- `name.phase` is `MediateAction(name, phase)`.
- `#` begins a comment; blank lines are ignored.

Intervals are strings accepted by `as_timedelta`, including `6h`, `2D`,
`90m`, and `500ms`, or `np.timedelta64` values in the Python API. Bare
numbers are rejected because their units are ambiguous. Avoid calendar month
and year units for fixed coupling intervals.

### Execution

At each clock time after `start`, a slot executes when elapsed time is a
positive multiple of its interval. Aligned slots execute in document order;
actions inside each slot execute in listed order.

- A run action advances a component and publishes exports at the current time.
- A connect action transfers the source's current exports.
- A mediate action computes the mediator's exports.

A connect before the source's run sees the previous export and is lagged. A
connect after the source's run sees the newly produced export and is
sequential.

Windowed connectors accumulate on every connect action and deliver only at
their window boundaries.

### Validation

`Driver.initialize()` validates:

- slot intervals are positive multiples of clock `dt`;
- each component or mediator runs in a slot equal to its timestep;
- action names and connector endpoints exist;
- every component appears in the sequence;
- connector fields and units match;
- every required import is fed, unless `allow_unfed_imports=True`.

`str(RunSequence)` emits parseable DSL with a final `@`. Parsing that output
reconstructs an equivalent sequence.

## YAML schema

`to_yaml(driver)` serializes configuration. `from_yaml(text_or_path)` returns
an uninitialized Driver.

```yaml
clock:
  start: '2024-01-01T00:00:00'
  stop: '2024-01-05T00:00:00'
  dt: 6h

sequence: |-
  @6h
    ocean -> atmos
    atmos
  @48h
    ocean
  @

components:
  atmos:
    class: my_package.factories.make_atmos
    kwargs:
      timestep: 6h
  ocean:
    class: my_package.factories.make_ocean
    kwargs:
      timestep: 48h

connectors:
  - src: ocean
    dst: atmos
    time_policy: constant
    fill: nearest
```

### Top-level keys

- `clock` (required): `start`, `stop`, and `dt`.
- `sequence` (required): DSL text, or a derived-sequence mapping.
- `components` (required): component name to `class` and `kwargs`.
- `dictionary` (optional): non-default `FieldEntry` definitions.
- `aliases` (optional): alias to standard-name additions.
- `connectors` (optional): connector declarations.

A graph-derived sequence is represented as:

```yaml
sequence:
  derived: true
  text: |-
    @6h
      source -> target
      source
      target
    @
```

`text` is informational. Loading re-derives the sequence from components and
connectors.

### Dictionary entries

```yaml
dictionary:
  - standard_name: air_temperature_2m_24h_max
    canonical_units: K
    description: Daily maximum near-surface temperature
    aliases: []
    cell_method:
      base: air_temperature_2m
      method: max
      window: 24h

aliases:
  t2m_daily_max: air_temperature_2m_24h_max
```

### Connector entries

Supported keys are:

- `src`, `dst`;
- optional `fields`;
- `time_policy`: `constant` or `linear`;
- `fill`: `none`, `zero`, or `nearest`;
- optional `sample`: `nearest` or `bilinear` for a PointGrid destination;
- paired `window` and `reduce` for windowed reduction.

```yaml
connectors:
  - src: atmos
    dst: ocean
    fields: [geopotential_at_1000hpa]
    time_policy: constant
    fill: none
    window: 2D
    reduce: mean
```

Custom regridder callables are not serialized. Configure those connectors in
Python.

### Component reconstruction

Each `class` is a dotted import path to a class or factory called with
`kwargs`. Components wrapping closures or live model objects require a
`yaml_spec`:

```python
component.yaml_spec = {
    "class": "my_package.factories.make_component",
    "kwargs": {"name": "atmos", "timestep": "6h"},
}
```

Accumulation mediators serialize automatically. DataArrays used as initial
conditions, model runtime state, and custom adapter objects are not part of
the YAML document.

Grid definitions are component constructor arguments. Primitive,
factory-understood grid names are suitable for YAML. Rich grid objects such
as a `PointGrid` generally require a factory that reconstructs the object
from YAML-safe keyword arguments.

### Loading and running

```python
driver = nvc.from_yaml("system.yaml")
driver.initialize({
    "atmos": atmos_initial_dataarray,
    "ocean": ocean_initial_dataarray,
})
datasets = driver.run()
```
