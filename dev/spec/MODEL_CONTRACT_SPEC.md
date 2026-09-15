# Earth2Studio Model Contract

## Goal

Pin the iterator, coordinate, and hook semantics that a prognostic or diagnostic
wrapper must satisfy, so that a shared execution substrate can drive any model
without wrapper-specific knowledge. This contract codifies behavior the correct
wrappers already implement; it does not add capability.

The contract is enforced by `earth2studio.models.conformance`, which reports the rule
identifiers used below. A model that passes is drivable by any conforming caller.

`AssimilationModel` is deliberately out of scope, and the omission is a scoping
decision rather than an oversight — see Open Questions for the gaps it has and why
they need their own document.

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
| `P14` | After `set_rng()`, seeding and stepping leave global RNG state unperturbed |
| `P15` | Stepping the model does not modify its input tensor or coordinate system |
| `P16` | A yielded tensor does not change once a later step is produced |

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
| `D10` | After `set_rng()`, seeding and calling leave global RNG state unperturbed |

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
coordinate system (`P15`, `D6`). Mutating the input does not save memory: a model
that clones and then mutates holds two copies at peak, exactly as one that builds
its output out-of-place. Mutating the caller's buffer instead of cloning saves that
peak only by destroying data the caller may still need — which forces *every* caller
to clone defensively before calling, turning an optional copy into a mandatory one.

This is already the informal convention — eight wrappers clone their input, two with
the comment `# prevent editing of argument` — and it has already failed once in the
field. `stormcast` wrote forecast results back into the caller's initial condition
while `stormcastconus` cloned, which is
[issue #1133](https://github.com/NVIDIA/earth2studio/issues/1133), fixed per-model in
PR #1134. A per-model fix does not prevent the next occurrence; `P15` does.

`P16` is the related guarantee for the iterator: once a later step is produced, an
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
coordinate system, which it rebuilds. `P15`'s tensor half is therefore the
important one. Note that `batch_func` will be updated in the xarray/cupy migration.

## Stochasticity

A model declares whether it draws randomness. The requirement is identical for both
protocols — `P11`-`P14` for prognostics, `D7`-`D10` for diagnostics — because the
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

The two arguments serve two different callers, and the distinction is what `reset`
does to an *already-initialized* generator: `reset=True` replaces it, `reset=False`
leaves it drawing from where it left off — so the seed argument is only consulted the
first time a model is seeded, and ignored on every `reset=False` call after that.

An ensemble driver wants replacement: each member is a fresh, independent trajectory,
so it reseeds with `reset=True` (the default) before every rollout.

```python
driver_rng = np.random.default_rng(0)

for member in range(n_members):
    model.set_rng(int(driver_rng.integers(2**32)))  # reset=True: fresh generator
    run(model)
```

A hook installed on the iterator wants the opposite. It runs once per step, *inside*
the rollout the driver already seeded, and its job is to keep drawing from that same
generator across steps — a new draw each step, not a restart to the first draw. If
the hook does not know whether the model has been seeded yet (it may run before or
after the driver, depending on setup order), it seeds defensively:

```python
@model.add_front_hook
def perturb(values, coords):
    # reset=False: a no-op if the driver already seeded the model, so this call
    # cannot clobber the trajectory in progress. Only takes effect as a fallback
    # if perturb runs before the driver has seeded anything.
    model.set_rng(fallback_seed, reset=False)
    noise = torch.randn(values.shape, generator=model.generator)
    return values + noise, coords
```

Had the hook called `set_rng(fallback_seed, reset=True)` instead, every step would
reinitialize the generator to the same state, so `perturb` would draw the *same*
noise at every step — silently collapsing what should be `nsteps` independent
perturbations into one value repeated across the whole forecast.

The only difference between the protocols is what reproducibility ranges over. `P13`
is a property of a rollout: the same seed reproduces every step. `D9` is a property
of a single call, because a diagnostic has no rollout.

An `int` seed rather than a `torch.Generator` is deliberate: several wrappers
delegate to a core model that accepts only a seed, and every existing implementation
already takes one.

### RNG isolation

A seeded model must keep its randomness to itself. `P14` and `D10`: once `set_rng()`
has been called, neither seeding nor stepping may leave the global RNG state
perturbed.

The harm is specific. A wrapper that seeds the global RNG reaches every other
consumer in the process — a second model in a cascade, a perturbation method, a
dataloader — and resets its stream to a fixed point. Two ensemble members that should
differ draw identical noise from an unrelated component. `P13` and `D9` make this
worse rather than catching it: they pass when the model is checked alone, and the
failure only appears once the model is one component of a pipeline.

The rule constrains the effect, not the mechanism. Three implementations satisfy it:

- a local `torch.Generator` seeded in `set_rng` and threaded into every draw;
- a functional PRNG key, as the JAX-backed wrappers already use;
- global seeding confined to a `torch.random.fork_rng()` block, which snapshots the
  RNG state, lets the seeded code run, and restores the state on exit.

The third is what makes the rule satisfiable for a model whose randomness is drawn
inside an external package with no generator injection point — `aifs2ens` calls
`torch.manual_seed(self.seed + step)` because `anemoi` offers nothing else. Forking
keeps that call and removes its blast radius:

```python
def set_rng(self, seed: int, reset: bool = True) -> None:
    if reset or self._seed is None:
        self._seed = seed

# at the step
with torch.random.fork_rng(devices=[x.device] if x.is_cuda else []):
    torch.manual_seed(self._seed + step)
    out = self.core_model(...)
```

Pass `devices` explicitly: the default forks every visible CUDA device and warns. The
residual cost is a state copy per step, negligible against a model forward.

**Known deviation.** Of the three wrappers that implement `set_rng` today, `dlesym`
seeds a local `torch.Generator` and conforms; `fcn3` delegates to its core model, so
conformance depends on what that model does internally; and `aurora1p5` is a bare
`torch.manual_seed(seed)` and fails `P14`. That is the same wrapper whose
constructor seed already conflicts with `set_rng` below, so both of its seeding
defects are fixed by the same rewrite. Tracked as an exemption in
`test/models/test_model_conformance.py` (`Aurora1p5Ensemble`) pending a wrapper fix
in a follow-up PR — this spec change does not alter the wrapper itself.

The rule is scoped to models that implement `set_rng`, and to the state *after* it is
called. An unseeded model drawing from the global generator merely *advances* it,
which is what any program using the default RNG does and is not the harm being
prevented; forbidding it would contradict the global-RNG fallback above. Reseeding is
the harm, because it destroys independence rather than consuming it.

### Seeding is the only entry point

A construction-time `seed` argument must not override a later `set_rng()` call.
`Aurora1p5Ensemble` currently shows why this matters: it stores `seed` on the
instance and `create_iterator()` re-applies `self.set_rng(self.seed)` on every call,
so a caller that does `model.set_rng(42)` and then iterates silently gets
`self.seed` instead. Two mechanisms for one piece of state is the bug; `set_rng` is
the one that survives, because a caller reseeding per ensemble member cannot reach a
constructor argument. Tracked as a known exemption (see above) pending a wrapper fix
in a follow-up PR.

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

The remaining stochastic wrappers each need a `stochastic` declaration and a
`set_rng` from its owner — they are left undeclared here rather than guessed at. They
divide by how their randomness is reached today, which is what determines the work
`P14`/`D10` implies for each:

| Mechanism today | Wrappers | Work |
| --- | --- | --- |
| Functional PRNG key | `gencast_mini`, `weathernext2_cyclones_mini` | declare and wrap |
| Local `torch.Generator` | `corrdiff` | declare and wrap |
| Seed passed to core model | `cbottle_video` | declare and wrap |
| Global `torch.manual_seed` | `aifs2ens`, `cbottle_sr`, `stormscope_dx_nsrdb` | fork the RNG |
| None at all | `atlas_crps`, `stormscope` | add seeding, forked |

"Declare and wrap" means the randomness is already isolated, so only the `stochastic`
declaration and a `set_rng` entry point are missing. "Fork the RNG" means the seeding
call stays as written and moves inside `torch.random.fork_rng()`.

Every wrapper in the tree is therefore reachable, and none needs a change to an
upstream package. The last row is the case that gains most: `atlas_crps` currently
documents "use `torch.manual_seed` for reproducible members", which is to say it
pushes global reseeding onto the caller and offers no per-instance control. A forked
`set_rng` gives it control it does not have today rather than taking any away.

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
(`P1`-`P6`, `P11`, `P12`, and the seeding half of `P14`), for models too expensive to
step in continuous integration. `check_diagnostic_contract(model, forward=False)` is
the equivalent, leaving `D1`-`D4`, `D7`, `D8`, and the seeding half of `D10`.

`P14` and `D10` are evaluated in two halves because only one of them needs a model
step: whether `set_rng()` itself perturbs the global RNG is answerable for free, and
it is the half that catches the common violation of implementing `set_rng` as a bare
`torch.manual_seed`.

The rollout rules build a probe input from `input_coords()`, which requires deriving
a tensor shape from a coordinate system. A one-dimensional coordinate contributes one
dimension of its own length. A multidimensional coordinate spans several dimensions at
once — a curvilinear grid declares `lat` and `lon` as two entries, both shaped
`(ny, nx)`, describing two tensor dimensions between them rather than one each. The
shape is derived by the convention `convert_multidim_to_singledim` already documents:
an `n`-dimensional entry is followed by `n - 1` partners of the same shape, and the
group contributes that shape once.

A coordinate system that does not satisfy that convention implies no shape, so no
probe input can be built and `P7`-`P10` and `P13`-`P16` are reported as skipped
rather than passed.

The probe tensor is pseudo-random rather than zero, so that a model writing into its
input is detectable. Models that reject unphysical input will need a fixture that
supplies a realistic initial condition.

### Enforcement

A model creation skill produces a `test_<model>_conformance` test alongside a model's
other required tests, checking the mock-weight instance the rest of the test file
already builds — no real weights, no network. `test/models/test_conformance.py`
checks the checker itself and `test/models/test_model_conformance.py` is the
completeness gate: every class reachable from `earth2studio.models.px` /
`earth2studio.models.dx` must be listed there as conformant or explicitly exempt with
a reason, discovered by introspecting the namespace rather than trusting a
hand-maintained list, so a new model cannot land without either passing the contract
or documenting why it does not. All three run in CI on every pull request.

Every model that predates this spec is currently listed as exempt pending backfill —
the gate stops new gaps from opening, and existing ones close incrementally as each
model's test file gains a conformance test.

## Open Questions

- `P13` compares rollouts with `torch.allclose`, so a deterministic model running on
  nondeterministic GPU kernels may report as stochastic. The tolerance may need to
  be configurable.
- `stochastic` currently answers two questions at once: *will ensemble members
  differ?* and *can I reproduce them?* `P12` fuses them by requiring `set_rng`
  whenever `stochastic` is `True`. A model whose randomness is drawn by an opaque
  dependency — one that neither accepts a generator nor draws from a forkable global
  RNG — would be stochastic but unseedable, and could not satisfy both. No such model
  is in the tree today: every wrapper is either already isolated or reachable by
  forking, which is why the two are not split here. If one arrives, the split is
  adding a `seedable` declaration and demoting `P13` to a skip, not reopening `P14`.
- `P14` and `D10` are scoped to models that implement `set_rng`. A model declaring
  `stochastic=False` that reseeds the global RNG anyway would go uncaught, but it
  would be pathological, and none exists today. Widening the rule to every model is
  cheap if one shows up.
- `AssimilationModel` needs an equivalent contract, but writing one means *choosing*
  between live divergences rather than codifying settled behavior, which is why it is
  not folded in here. Four gaps, in severity order:
  1. `output_coords()` has no callable contract. The protocol declares
     `(input_coords, *args, **kwargs)`, and the implementations split:
     `InterpEquirectangular` requires `request_time` as a positional with no default,
     `HealDA` takes it as an optional keyword, and `StormCastSDA` and
     `CorrDiffCosmoEra5SDA` accept `input_coords` alone and reject it. No caller can
     resolve output coordinates generically, which forfeits the plan-before-allocate
     property that `P8` and `D5` give the other two protocols.
  2. There is no priming-yield convention. `HealDA`, `InterpEquirectangular`, and
     `CorrDiffCosmoEra5SDA` prime with `yield None`; `StormCastSDA` primes by
     yielding the initial state. `CorrDiffCosmoEra5SDA` documents the split in its
     own docstring — "unlike StormCast, yields no initial state" — so it is a known,
     unresolved divergence. This is the `P7` question for the send-protocol
     generator.
  3. Stochasticity is undeclared. `CorrDiffCosmoEra5SDA` holds `self.seed` and
     derives per-member seeds as `seed + i`, the same pattern the diagnostics use,
     with no `set_rng`. `P11`-`P14` port across unchanged.
  4. Input immutability matters more here, not less. Inputs are `pd.DataFrame` and
     `xr.DataArray`, which are mutated in place far more idiomatically than tensors,
     so the `P15`/`D6` hazard is larger while the defensive-clone convention that
     grew up around the tensor models does not exist.

  A conformance checker is feasible on the same pattern: `FrameSchema` is an
  `OrderedDict[str, np.ndarray]` mapping column names to representative arrays, so a
  probe DataFrame is constructible exactly as `_sample_tensor` builds a probe tensor.
  The generator lifecycle — what `send(None)` means mid-stream, whether `__call__` is
  equivalent to one generator step, who closes the generator — would need deciding
  first.
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
