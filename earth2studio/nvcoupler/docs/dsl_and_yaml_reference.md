# Run-sequence DSL and YAML reference

This page is the normative reference for the two text formats nvcoupler
understands: the run-sequence DSL (`parse_run_sequence`, `Driver(sequence=...)`)
and the YAML system schema (`to_yaml` / `from_yaml`). For the concepts behind
them see [concepts.md](concepts.md); for task-oriented walkthroughs see
[user_guide.md](user_guide.md). Every snippet below has been executed against
the toy components in `earth2studio.nvcoupler.testing`.

## The run-sequence DSL

Most systems never write a sequence: `Driver(sequence=None)` (the default)
and `couple()` derive the canonical lagged sequence from the coupling graph
via `derive_sequence`. The DSL is the explicit override for hand-tuned
ordering — it mirrors NUOPC's runSeq: slots opened by `@<interval>` headers,
each containing an ordered list of actions.

```
@6h
  atmos -> med          # accumulate atmos exports into the mediator
  ocean -> atmos        # lagged: before atmos runs
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
```

### Grammar

Line-oriented. On every line, everything from the first `#` to the end of the
line is a comment; blank (or comment-only) lines are ignored. The remaining
lines are:

| Line form | Meaning | Regex (after comment strip) |
|---|---|---|
| `@<interval>` | Open a new slot with that interval | `^@(\S+)?$` |
| `@` | Close the current slot (terminator) | same |
| `src -> dst` | `ConnectAction(src, dst)` — transfer matched fields | `^(\w[\w.-]*)\s*->\s*(\w[\w.-]*)$` |
| `name.phase` | `MediateAction(name, phase)` — run a mediator | `^(\w[\w-]*)\.(\w+)$` |
| `name` | `RunAction(name)` — run a component | `^(\w[\w-]*)$` |

Notes on the grammar, all verified against `parse_run_sequence`:

- Actions are matched in the order connect, mediate, run — so a bare name
  containing a dot always parses as a `MediateAction`. Component names must
  therefore not contain dots (allowed characters: word characters and `-`).
- The mediate phase is free-form (`med.compute` by convention; `med.finalize`
  parses equally). The `Driver` ignores the phase at execution time and calls
  the mediator's `run(time)` (which calls `compute`) for every `MediateAction`.
- The trailing bare `@` terminator is what `str(RunSequence)` emits, but it is
  optional on input: end-of-input also closes the last slot. An action line
  appearing after a bare `@` (or before any `@<interval>`) raises
  `SequenceError("... outside any @interval slot")` with the line number.
- An empty sequence (no slots at all) raises `SequenceError("Run sequence is
  empty")`.
- An unparseable action line raises `SequenceError` naming the line and the
  three accepted forms.

### Interval strings (`as_timedelta`)

Slot headers — and every timestep/window/`dt` argument in the package — are
coerced through `earth2studio.nvcoupler.clock.as_timedelta`, which accepts:

- **`np.timedelta64`** values (converted to nanosecond precision).
- **Strings of the form `<int><unit>`**: an integer (optionally negative)
  followed by a numpy timedelta64 unit code, with four convenience spellings
  mapped first: `d → D`, `H → h`, `min → m`, `S → s`. Verified examples:
  `"6h"`, `"12H"`, `"2D"`, `"2d"`, `"1W"`, `"90m"`, `"30min"`, `"45s"`,
  `"500ms"`. Note `m` is minutes; `M` is calendar months, which numpy converts
  using an average-month length — avoid `M`/`Y` for coupling intervals.

Rejected:

- **Bare numbers** — `as_timedelta(6)` raises
  `ValueError: Bare number 6 is ambiguous as a timedelta (hours? steps?) —
  pass a string like '6h' or '2D', or a np.timedelta64`. There is no implicit
  "hours" or "steps" unit anywhere in nvcoupler.
- Strings with no leading digits (`"h6"`) or no unit (`"6"`) raise
  `ValueError: Cannot parse timedelta ...; expected e.g. '6h', '2D'`. Inside
  `parse_run_sequence` this is re-raised as `SequenceError("Line N: ...")`.
- A caveat, honestly: an unknown unit *letter* (e.g. `"6x"`) propagates
  numpy's own `TypeError: Invalid datetime unit "x" in metadata` rather than a
  `SequenceError` — only `ValueError`s are wrapped with the line number.
- Negative intervals parse (`"-6h"`) but fail validation: `is_multiple`
  requires a positive interval, so `validate()` raises `CadenceError`, and
  `Clock` rejects a non-positive `dt` at construction.

### Formal semantics

Let the driver `Clock` run from `start` to `stop` in steps of `dt`. The
execution rule, exactly as implemented by `Driver._execute_time`:

1. **Slot alignment.** At each clock time `t` (the first is `start + dt`;
   actions never execute at `start` itself — the initial conditions seed the
   export states there), a slot with interval `I` is *aligned* iff
   `(t - start) > 0` and `(t - start) % I == 0`. A 6 h slot on a 6 h clock is
   aligned at every step; a 48 h slot at every 8th.
2. **Execution order.** All aligned slots execute at `t` in the order they
   appear in the sequence, and within a slot the actions execute strictly in
   listed order. When a 6 h and a 48 h slot are both aligned (every 48 h), the
   6 h slot's actions run first because it is listed first — the order is
   textual, never cadence-derived.
3. **Action effects.**
   - `RunAction(c)`: calls `c.run(t)` — the component consumes whatever is
     currently in its import `State` and republishes its export `State` with
     `valid_time = t`.
   - `ConnectAction(src, dst)`: the connector copies whatever is *currently*
     in `src.export_state` through its pipeline (time policy → vertical →
     mask fill → regrid) into `dst.import_state`. If `src` has not yet
     produced a matched field, this raises
     `CouplingError("... has not produced <field> yet — check the run
     sequence ordering")`. A *windowed* connector (`window=`/`reduce=`)
     behaves differently: each execute folds the source export into its
     running reduction, and it delivers the derived field only at times
     aligned to its window — mid-window executes leave the destination's
     import untouched.
   - `MediateAction(m, phase)`: calls `m.run(t)`, i.e. the mediator's
     `compute(t)`, which turns its accumulated samples into exported derived
     fields. (Accumulation itself is not an action — it happens as a side
     effect of every connector delivery into the mediator.)

**Lagged vs sequential is purely positional.** A connect transfers the source
state *as of the moment the connect executes*:

- Connect placed **before** the source's run at the same time (or in a slot
  where the source does not run at all this time) delivers the source's
  *previous* export — **lagged** (NUOPC-explicit) coupling. In the example
  above, `ocean -> atmos` before `atmos` delivers the ocean state from its
  last 48 h ring.
- Connect placed **after** the source's run (or mediator's compute) at the
  same time delivers the export *just produced* — **sequential** coupling.
  `med -> ocean` after `med.compute` hands the ocean the freshly reduced
  48 h mean.

Swapping the two flavors is a one-line reorder; see
`examples/09_nvcoupler/02_lagged_vs_sequential.py`
([examples README](../../../examples/09_nvcoupler/README.rst)). `describe()`
labels a connect's mode with exactly this source-relative rule: `sequential`
iff the source ran (or the mediator computed) earlier in the same slot, else
`lagged`.

### Validation

`RunSequence.validate(components, dt)` runs inside `Driver.initialize()`
(after parsing, before anything executes) and performs, in order:

1. **Slot cadence**: every slot interval must be a positive whole multiple of
   the driver `dt`, else `CadenceError` (`@7h` on a 6 h clock fails).
2. Per action:
   - `RunAction`: the component name must be a key of `components` (unknown
     names raise `SequenceError` with a did-you-mean suggestion, e.g.
     `atmoss` → `Did you mean: 'atmos'?`), **and** the component's `timestep`
     must equal the slot interval exactly — a 48 h ocean listed in a `@6h`
     slot raises `CadenceError`.
   - `ConnectAction`: both endpoint names must resolve (same suggestion
     machinery). No cadence constraint — connects may sit in any slot.
   - `MediateAction`: the mediator name must resolve, and its `timestep`
     (its window) must equal the slot interval, exactly like a `RunAction`
     (`med.compute` for a 48 h mediator in a `@6h` slot raises
     `CadenceError`). The phase string is not validated.
3. **Completeness**: every component in `components` must appear in some
   `RunAction` or `MediateAction`; idle components raise
   `SequenceError("Components never run by the sequence: [...] — add a
   RunAction (bare component name) to a slot matching their timestep")`.

`Driver.initialize()` layers further checks on top (connector field matching
and units, unfed-import detection, unconsumed-export warnings) — see
[errors_and_troubleshooting.md](errors_and_troubleshooting.md).

### Round-trip guarantees

`str(RunSequence)` emits valid DSL and `parse_run_sequence(str(seq)) == seq`
(actions are frozen dataclasses, so equality is structural). Formatting rules:

- Intervals that are a whole number of hours print NUOPC-style as hours —
  including multi-day ones: a slot built with `np.timedelta64(2, "D")` prints
  as `@48h`. The parsed interval is identical either way.
- Sub-hourly intervals fall back to `fmt_timedelta`, so `@90m` and `@30m`
  survive round-trips exactly (this was a real regression: an earlier
  formatter truncated `@90m` to `@1h` through YAML round-trips; the fix is
  pinned by `test_str_preserves_subhour_intervals`).
- Output is always two-space-indented actions and a final bare `@`; comments
  are not preserved (they are stripped at parse time).

For hand-written sequences the YAML `sequence` key stores
`str(driver.sequence)` verbatim, so these guarantees are exactly what makes
those YAML round-trips faithful. Derived sequences are stored as
`{derived: true, text: ...}` and re-derived on load instead (see below).

## The YAML schema

`to_yaml(driver)` serializes a `Driver` to a small YAML document;
`from_yaml(text_or_path)` rebuilds an *uninitialized* driver from it (call
`driver.initialize(ics)` afterwards as usual — initial conditions are tensors
and are never serialized).

### Top-level keys

| Key | Required | Type | Meaning |
|---|---|---|---|
| `clock` | yes | mapping `{start, stop, dt}` | ISO-8601 `start`/`stop` strings, `dt` an interval string. Rebuilt as `Clock(start, stop, dt)`. |
| `sequence` | yes | string (literal block) or mapping | Hand-written sequences: the run-sequence DSL, verbatim (`str(driver.sequence)`). Derived sequences (`driver.sequence_derived`): `{derived: true, text: <DSL>}` — the `text` is informational; `from_yaml` re-derives the schedule from components + connectors (deterministic, so round-trips reproduce identical runs). A mapping without `derived: true` raises `CouplingError`. |
| `components` | yes | mapping `name -> {class, kwargs}` | `class` is a dotted import path to a module-level class or factory; it is imported and called as `factory(**kwargs)`. |
| `dictionary` | no | list of entry mappings | Only `FieldEntry` items **absent from or differing from** `DEFAULT_DICTIONARY`. Each has `standard_name`, `canonical_units`, `description`, `aliases` (list), and optional `cell_method: {base, method, window}`. |
| `aliases` | no | mapping `alias -> standard_name` | Alias additions relative to the default dictionary (see below). |
| `connectors` | no | list of `{src, dst, time_policy, fill, fields?, sample?, window?, reduce?}` | Connector settings. `fields` appears only when the connector was built with an explicit list; `time_policy` defaults to `"constant"` and `fill` to `"none"` on load. `sample` (`"nearest"` or `"bilinear"`) appears only when set — a plain string, it round-trips like `time_policy`/`fill`. `window`/`reduce` appear (together) for windowed connectors and rebuild them on load. |

`from_yaml` raises `CouplingError` when the document is not a mapping, when
any of `clock`/`sequence`/`components` is missing, when a component spec lacks
`class`, when the import path cannot be resolved (the error names the module
and attribute), when the factory call itself fails (wrapped with the class
path and kwargs), or when a connector endpoint is not a configured component.
A string argument containing no newline that names an existing file is read as
a file path; anything else is parsed as YAML text.

If a `dictionary` and/or `aliases` section is present, `from_yaml` builds one
extended `FieldDictionary` (a copy of the default plus the entries and
aliases) and passes it as the `dictionary` kwarg to every component factory
that accepts one (inspected via its signature, `**kwargs` counts) and whose
`kwargs` don't already set it.

### Annotated example

This exact document is `to_yaml` output for the toy atmos–ocean–mediator
system, and executing `from_yaml(to_yaml(driver))` reproduces the original
run bit-for-bit (same xarray outputs; the assertion is in the scratch check
for this page and in `test/nvcoupler/test_config.py::test_round_trip_identical_outputs`):

```yaml
clock:
  start: '2024-01-01T00:00:00'   # ISO-8601, second precision
  stop: '2024-01-05T00:00:00'
  dt: 6h                          # interval string (as_timedelta form)
sequence: |-                      # the run-sequence DSL, verbatim
  @6h
    atmos -> med
    ocean -> atmos
    atmos
  @48h
    med.compute
    med -> ocean
    ocean
  @
components:
  atmos:
    class: earth2studio.nvcoupler.testing.fake_atmos   # module-level factory
    kwargs:
      gain: 1.0
      timestep: 6h
  ocean:
    class: earth2studio.nvcoupler.testing.fake_ocean
    kwargs:
      gain: 1.0
      timestep: 48h
  med:                            # AccumulationMediator: auto-serialized
    class: earth2studio.nvcoupler.mediator.TrailingAverageMediator
    kwargs:
      name: med
      fields:
      - geopotential_at_1000hpa_48h_mean
      window: 2D                  # 48 h, printed in fmt_timedelta's day form
```

The Python side, executed to verify:

```python
import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

atmos = fake_atmos(gain=1.0)
atmos.yaml_spec = {
    "class": "earth2studio.nvcoupler.testing.fake_atmos",
    "kwargs": {"gain": 1.0, "timestep": "6h"},
}
ocean = fake_ocean(gain=1.0)
ocean.yaml_spec = {
    "class": "earth2studio.nvcoupler.testing.fake_ocean",
    "kwargs": {"gain": 1.0, "timestep": "48h"},
}
med = nvc.TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"])
driver = nvc.Driver(
    {"atmos": atmos, "ocean": ocean, "med": med},
    "@6h\n  atmos -> med\n  ocean -> atmos\n  atmos\n@48h\n  med.compute\n  med -> ocean\n  ocean\n@",
    nvc.Clock("2024-01-01", "2024-01-05", "6h"),
)
rebuilt = nvc.from_yaml(nvc.to_yaml(driver))   # round-trip
rebuilt.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
datasets = rebuilt.run()                        # identical to driver.run()
```

### Derived sequences and windowed connectors, executed

A driver built declaratively (no `sequence=`) serializes its schedule as the
mapping form, and windowed connectors carry `window`/`reduce`; `from_yaml`
re-derives the sequence and rebuilds the windowed connector, so the
round-trip reproduces the run exactly (pinned in
`test/nvcoupler/test_config.py`):

```python
atmos2, ocean2 = fake_atmos(gain=1.0), fake_ocean(gain=1.0)
atmos2.yaml_spec, ocean2.yaml_spec = atmos.yaml_spec, ocean.yaml_spec
declared = nvc.Driver(
    {"atmos": atmos2, "ocean": ocean2},
    clock=nvc.Clock("2024-01-01", "2024-01-05", "6h"),
    connectors=[
        ("ocean", "atmos"),
        nvc.Connector(atmos2, ocean2, window="48h", reduce="mean"),
    ],
)
text = nvc.to_yaml(declared)
assert "derived: true" in text and "window: 2D" in text and "reduce: mean" in text
rebuilt2 = nvc.from_yaml(text)
assert rebuilt2.sequence_derived
assert str(rebuilt2.sequence) == str(declared.sequence)
```

### Serialization rules

`to_yaml` decides per component, in order:

1. **`yaml_spec` attribute wins.** If the component carries a `yaml_spec`
   attribute, it must be a dict with a `"class"` key (dotted import path) and
   optional `"kwargs"`; anything else raises `CouplingError`. The kwargs are
   sanitized: `np.timedelta64` → interval string, `np.datetime64` → ISO
   string, numpy scalars → Python scalars, arrays/sets/tuples → lists, dicts
   recursively. This is the escape hatch for `CallableComponent`s and anything
   wrapping Python state — the contract is simply *"calling
   `<class>(**kwargs)` rebuilds an equivalent component"*. Note `from_yaml`
   does **not** re-attach `yaml_spec` to rebuilt components; re-tag them if
   you intend to serialize again.
2. **`AccumulationMediator` subclasses auto-serialize** (this includes
   `TrailingAverageMediator`): class path plus
   `{name, fields: <derived names>, window: <interval string>}` are recovered
   from the instance itself — no `yaml_spec` needed.
3. **Everything else raises `CouplingError`**, with the honest explanation:
   the component wraps a closure or model object YAML cannot reconstruct; set
   `yaml_spec` or build the system in Python. Model components referenced by
   load paths (e.g. `{load: 'earth2studio.models.px.Persistence'}`) are
   explicitly out of scope for v1.

Not serialized, honestly stated: connector `regridder=` callables (a rebuilt
connector falls back to the auto lat/lon path — HEALPix/curvilinear systems
need Python construction); `io=` backends; `collect`; and
`allow_unfed_imports`. Custom `Connector` subclasses lose their type: only
`src`/`dst`/`fields`/`time_policy`/`fill`/`sample`/`window`/`reduce`
round-trip. A destination's `points=PointSet(...)` is a *component* kwarg,
not a connector setting — it round-trips the same way `model=` does: only
through that component's own `yaml_spec` (`PointSet` isn't itself YAML-safe
out of the box, being a dataclass of numpy arrays; a component author wanting
one reconstructed needs a `yaml_spec["kwargs"]` shaped for it, same as any
other non-primitive component kwarg).

### The aliases delta mechanism

The `dictionary` section only contains entries that differ from the built-in
default — but an alias added with `FieldDictionary.add_alias("std", "alias")`
*after* registration leaves the `FieldEntry` itself equal to the default and
would be silently lost. The `aliases` key exists to carry exactly that delta:
`to_yaml` walks every component dictionary and emits each
`alias -> standard_name` pair that is neither in the default dictionary nor
explained by the component's own `variable_aliases` kwarg (those are rebuilt
from the component spec). If two components map the same alias to different
standard names, `to_yaml` raises `CouplingError` asking you to make the alias
consistent before serializing. On load, the pairs are re-applied with
`add_alias` to the shared rebuilt dictionary.

### Window and time string forms

All interval-valued fields in the document — `clock.dt`, mediator `window`
kwargs, `cell_method.window` — are emitted by `fmt_timedelta`: whole days as
`"2D"`, else whole hours as `"48h"`, else whole minutes as `"90m"`, else the
raw numpy repr. All are read back by `as_timedelta`, so the day/hour spelling
is cosmetic (`window: 2D` ≡ `window: 48h`; verified equal after parsing).
Clock `start`/`stop` are ISO-8601 strings at second precision, parsed by
`np.datetime64`. The `sequence` block follows the DSL's own round-trip rules
above, including the sub-hourly `@90m` guarantee.

## See also

- [concepts.md](concepts.md) — why sequences and slots look the way they do
- [user_guide.md](user_guide.md) — building systems in Python vs YAML
- [api_reference.md](api_reference.md) — exact signatures for `parse_run_sequence`, `to_yaml`, `from_yaml`
- [errors_and_troubleshooting.md](errors_and_troubleshooting.md) — the full error catalogue
- [design_and_roadmap.md](design_and_roadmap.md) — planned schema extensions (checkpointed models, IO)
