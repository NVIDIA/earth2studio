# Earth2Studio Model Contract

## Goal

Pin the iterator, coordinate, and hook semantics that a prognostic or diagnostic
wrapper must satisfy, so that a shared execution substrate can drive any model
without wrapper-specific knowledge. This contract codifies behavior the correct
wrappers already implement; it does not add capability.

The contract is enforced by `earth2studio.models.conformance`, which reports the rule
identifiers used below. A model that passes is drivable by any conforming caller.

## Coordinate Systems

A coordinate system is an `OrderedDict[str, np.ndarray]`. Order is part of the
contract: the key order is the tensor's dimension order, and callers index
positionally against it.

`input_coords()` is a *declaration*, not a concrete coordinate system. A zero-length
array declares an open dimension whose size the caller chooses; every other entry
declares required coordinate values. The leading dimension is always `batch`.

`output_coords(input_coords)` is a *resolver*. It validates a concrete input
coordinate system and returns the coordinate system the model will produce, without
touching field data. Validation and resolution happen together so that a caller can
plan a rollout — output shapes, lead times, and dimension order — before allocating
anything.

## Rules

### Prognostic

| Rule | Requirement |
| --- | --- |
| `P1` | Structurally satisfies `PrognosticModel` |
| `P2` | `input_coords()` and `output_coords()` return ordered dicts of arrays led by `batch` |
| `P3` | `input_coords()["lead_time"]` is relative, strictly increasing, and ends at zero |
| `P4` | `output_coords()` treats its argument as read-only |
| `P5` | `output_coords()` raises `ValueError` for an invalid coordinate system |
| `P6` | Shifting input `lead_time` by an offset shifts output `lead_time` by the same offset |
| `P7` | `create_iterator()` yields the initial condition as its 0th step |
| `P8` | The 1st yield matches the coordinate system `output_coords()` declared |
| `P9` | Every yielded tensor shape matches its coordinate system |
| `P10` | `create_iterator()` applies both hook chains; `__call__` applies neither |
| `P11` | The model declares a boolean `stochastic` attribute |
| `P12` | A stochastic model implements `set_rng(seed, reset=True)` |
| `P13` | Seeding determines a rollout, and different seeds give different rollouts |
| `P14` | Stepping the model does not modify its input tensor or coordinate system |
| `P15` | A yielded tensor does not change once a later step is produced |

### Diagnostic

| Rule | Requirement |
| --- | --- |
| `D1` | Structurally satisfies `DiagnosticModel` |
| `D2` | Coordinate systems are ordered dicts of arrays led by `batch` |
| `D3` | `output_coords()` treats its argument as read-only |
| `D4` | `output_coords()` raises `ValueError` for an invalid coordinate system |
| `D5` | `__call__` returns the coordinate system `output_coords()` declared |
| `D6` | `__call__` does not modify its input tensor or coordinate system |
| `D7` | The model declares a boolean `stochastic` attribute |
| `D8` | A stochastic model implements `set_rng(seed, reset=True)` |
| `D9` | Seeding determines the output, and different seeds give different output |

## Lead Time

Input `lead_time` is relative to the analysis time and ends at zero, so a model with
two history steps and a six hour step declares `[-6h, 0h]`. The final entry is the
step being advanced from.

Rebasing is the rule that makes a rollout composable: a model resolves output
`lead_time` as its own step offset added to the *final input* `lead_time`, never to a
constant. Consequently, feeding a model's output back to its input advances the
forecast, and a caller can start a model from any lead time — mid-rollout, resumed
from a checkpoint, or driven by an upstream model — without the model knowing.

`P6` is the machine-checkable form: shift every input `lead_time` by a constant and
the output must shift by exactly that constant.

## Iteration

`create_iterator()` yields the initial condition first. That 0th yield carries the
input coordinate system with `lead_time` reduced to its final entry, and the tensor
reduced to match. Forecast steps follow, each advancing `lead_time` by the model's
step.

The 0th yield is the initial condition rather than a forecast, so a caller writing
output for `nsteps` forecast steps draws `nsteps + 1` yields. A model must not
consume its input before the 0th yield, and must not emit a partial step.

## Hooks

`front_hook` and `rear_hook` are the declared mutation points of a step, applied
immediately before and after the model advances. A hook takes and returns
`(tensor, coords)`, so it may rewrite values, coordinates, or both.

**Hooks belong to the iterator.** `create_iterator()` applies both chains on every
forecast step; `__call__` applies neither. This is not a concession to the existing
wrappers, though all thirty of them already behave this way — it is the scope the
feature earns.

`__call__` is a single-step primitive. A caller holding one step already has the
tensor and can transform it directly, so a hook adds nothing there. What a caller
cannot reach is the state fed *back* between steps inside `create_iterator()`:
history buffers, cubed-sphere state, a multi-model cascade. `front_hook` mutates
exactly that state, which is why it cannot be replicated from outside and why the
feature survives the arrival of `Pipeline`. Perturbing a recurrent state between
steps to produce a model-uncertainty ensemble is the load-bearing use case, and
`examples/02_medium_range/02_model_perturbation_hook.py` is it.

`rear_hook` *is* externally replicable — a caller can transform each yield — but it
is implemented on twenty-seven wrappers and costs nothing to keep, so it stays.

Giving two step paths different hook semantics is the ambiguity worth avoiding, so
`P10` checks both directions: the iterator must apply both chains, and `__call__`
must apply none.

Hooks are composable. `add_front_hook()` and `add_rear_hook()` register into a
`HookChain` applied in registration order; `clear_hooks()` restores the pass-through
default. Assigning a bare callable to the slot remains supported and is promoted to
the chain's first element, so registration never silently drops an assigned hook.

```python
@model.add_rear_hook
def clamp_precipitation(x, coords):
    index = np.where(coords["variable"] == "tp")[0]
    x[:, :, index] = x[:, :, index].clamp(min=0)
    return x, coords
```

**Known deviation.** `gencast_mini`, `graphcast_small`, `graphcast_operational`, and
`weathernext2_cyclones_mini` apply `rear_hook` and never `front_hook`, so a front
hook set on them is silently discarded. `P10` detects this; fixing it is four lines
across four wrappers.

## Ownership of Tensors

A model borrows the tensor it is given and owns the tensor it returns.

`__call__` and `create_iterator()` must not write into the caller's input tensor or
coordinate system (`P14`, `D6`). Mutating the input does not save memory: a model
that clones and then mutates holds two copies at peak, exactly as one that builds
its output out-of-place. Mutating the caller's buffer instead of cloning saves that
peak only by destroying data the caller may still need — which forces *every* caller
to clone defensively before calling, turning an optional copy into a mandatory one.

This is already the informal convention — eight wrappers clone their input, two with
the comment `# prevent editing of argument` — and it has already failed once in the
field. `stormcast` wrote forecast results back into the caller's initial condition
while `stormcastconus` cloned, which is
[issue #1133](https://github.com/NVIDIA/earth2studio/issues/1133), fixed per-model in
PR #1134. A per-model fix does not prevent the next occurrence; `P14` does.

`P15` is the related guarantee for the iterator: once a later step is produced, an
earlier yield must not have changed. A model may return a view into its own buffers,
but not a view into a buffer it will overwrite on the next step. Without this, a
caller that holds a yield across steps reads the wrong values, and holding a yield
across steps is exactly what an asynchronous IO backend, a resume buffer, and a
scoring accumulator all do. `AsyncZarrBackend` currently defends itself with an
unconditional copy on the non-blocking write path, commented "prevents race
conditions when the data is mutated in place before the write is completed" — the
guarantee is being purchased per-write at the IO layer because the model layer does
not offer it.

Note that the `batch_func` decorator does *not* protect the input tensor:
`_compress_batch` reshapes with `unsqueeze` and `flatten`, both of which return
views, so a write inside a decorated method reaches the caller. It does shield the
coordinate system, which it rebuilds. `P14`'s tensor half is therefore the load
bearing one.

## Stochasticity

A model declares whether it draws randomness. The requirement is identical for both
protocols — `P11`-`P13` for prognostics, `D7`-`D9` for diagnostics — because the
caller's need is identical: a driver scheduling ensemble members has to know whether
members will differ and how to make them reproducible, and it does not care which
protocol the component implements.

```python
class MyModel(torch.nn.Module, PrognosticMixin):
    stochastic = True

    def set_rng(self, seed: int, reset: bool = True) -> None: ...
```

`stochastic` is a statically readable attribute, defaulting to `False` when absent.
It is a declaration, not a capability: a caller reads it to decide whether ensemble
members will differ at all, and to reject a fifty-member request against a
deterministic model, *before* running anything. `PrognosticMixin` supplies the
`False` default; diagnostics have no shared base class, so for them the default comes
from the attribute being absent. Adding a `DiagnosticMixin` purely to hold one class
attribute is not worth the inheritance change across twenty wrappers, and the
conformance check reads the attribute either way.

`set_rng(seed, reset=True)` is the single seeding entry point, required when
`stochastic` is `True`. The seed is the first positional argument. `reset=False`
leaves an already-initialized generator alone, which is what a caller wants when
reseeding mid-rollout would break a noise trajectory. A model that is never seeded
falls back to the global RNG rather than failing.

The only difference between the protocols is what reproducibility ranges over. `P13`
is a property of a rollout: the same seed reproduces every step. `D9` is a property
of a single call, because a diagnostic has no rollout.

An `int` seed rather than a `torch.Generator` is deliberate: several wrappers
delegate to a core model that accepts only a seed, and every existing implementation
already takes one.

### Seeding is the only entry point

A construction-time `seed` argument must not override a later `set_rng()` call.
`Aurora1p5Ensemble` currently shows why this matters: it stores `seed` on the
instance and `create_iterator()` re-applies `self.set_rng(self.seed)` on every call,
so a caller that does `model.set_rng(42)` and then iterates silently gets
`self.seed` instead. Two mechanisms for one piece of state is the bug; `set_rng` is
the one that survives, because a caller reseeding per ensemble member cannot reach a
constructor argument.

`seed` on `load_model` is not a hypothetical: `corrdiff`, `cbottle_sr`, and
`stormscope_dx_nsrdb` already accept it there and thread it to the constructor. Any
decision to remove constructor seeds has to account for that established diagnostic
convention rather than treating it as a new question — see Open Questions.

`P13` and `D9` are the properties that actually matter for a distributed run: the
same seed reproduces output exactly, so a resumed run matches the run it resumes, and
different seeds produce different output, so ensemble members are not duplicates. A
model that declares `stochastic=False` but produces two different results from one
input fails the same rule.

### Migration

Three wrappers implement `set_rng` today, with two incompatible signatures
(`fcn3` and `dlesym` take `(seed, reset)`; `aurora1p5` takes `(seed)`), and callers
discover it by `hasattr`. Those three now declare `stochastic`; `dlesym` declares it
as a property because it is conditional on `use_cln`.

The remaining stochastic wrappers hold a `seed` attribute and seed the global RNG
instead: prognostic `aifs2ens`, `gencast_mini`, `cbottle_video`,
`weathernext2_cyclones_mini`, `atlas_crps`, and `stormscope`; diagnostic `corrdiff`,
`corrdiff_cosmo_era5`, `cbottle_sr`, and `stormscope_dx_nsrdb`. Each needs a
`stochastic` declaration and a `set_rng` from its owner — they are left undeclared
here rather than guessed at.

The diagnostics share one further pattern worth naming before it is standardized:
`seed=None` means *draw a fresh seed per call*, implemented as
`seed = self.seed if self.seed is not None else np.random.randint(2**32)`. That is a
third state which `set_rng(seed: int)` does not express. Under this spec the
unseeded state is simply "never called `set_rng`", so the `None` sentinel becomes
redundant — but the migration has to preserve today's default of a nondeterministic
run for a caller who seeds nothing.

## Conformance

```python
from earth2studio.models.conformance import check_prognostic_contract

skipped = check_prognostic_contract(model)
```

Every rule is evaluated before the check fails, so one call reports all violations.
The return value lists rules that could not be evaluated and why — a rule that cannot
run is reported rather than passed silently.

`rollout=False` restricts the check to the rules that need no forward pass
(`P1`-`P6`, `P11`, `P12`), for models too expensive to step in continuous
integration. `check_diagnostic_contract(model, forward=False)` is the equivalent,
leaving `D1`-`D4`, `D7`, and `D8`.

The rollout rules build a probe input from `input_coords()`, which requires deriving
a tensor shape from a coordinate system. A one-dimensional coordinate contributes one
dimension of its own length. A multidimensional coordinate spans several dimensions at
once — a curvilinear grid declares `lat` and `lon` as two entries, both shaped
`(ny, nx)`, describing two tensor dimensions between them rather than one each. The
shape is derived by the convention `convert_multidim_to_singledim` already documents:
an `n`-dimensional entry is followed by `n - 1` partners of the same shape, and the
group contributes that shape once.

A coordinate system that does not satisfy that convention implies no shape, so no
probe input can be built and `P7`-`P10` and `P13`-`P15` are reported as skipped
rather than passed.

The probe tensor is pseudo-random rather than zero, so that a model writing into its
input is detectable. Models that reject unphysical input will need a fixture that
supplies a realistic initial condition.

## Open Questions

- `P13` compares rollouts with `torch.allclose`, so a deterministic model running on
  nondeterministic GPU kernels may report as stochastic. The tolerance may need to
  be configurable.
- The forcing and conditioning declaration is deliberately absent. It is shared with
  the coupling and labelled-array proposals and must be agreed across all three
  before it is specified here
- Existing `seed=` constructor arguments currently stay as conveniences and could be
  defined as equivalent to calling `set_rng` at construction. Would it be more clear
  to remove them outright and not permit `seed` constructor args so we really control
  RNG behavior per model from just one single place?
  Reviewers should weigh three facts. First, `seed` is not only a constructor
  argument today: `corrdiff`, `cbottle_sr`, and `stormscope_dx_nsrdb` accept it on
  `load_model` and pass it through, so removal touches the loading API too. Second,
  `Aurora1p5Ensemble` demonstrates the concrete failure of keeping both — its
  `create_iterator` re-applies `self.seed` and silently discards a caller's
  `set_rng`. Third, `seed=None` currently means "fresh seed per call" on the
  diagnostics, so removal needs a replacement for that default, or we need to
  communicate the breaking change.
- Relatedly, should this effort also migrate the existing stochastic wrappers that
don't use `set_rng` but instead hold a `seed` attribute and seed the global RNG?
(`aifs2ens`, `gencast_mini`, `cbottle_video`,
`weathernext2_cyclones_mini`, `atlas_crps`, and `stormscope`; diagnostic `corrdiff`,
`corrdiff_cosmo_era5`, `cbottle_sr`, and `stormscope_dx_nsrdb`)
