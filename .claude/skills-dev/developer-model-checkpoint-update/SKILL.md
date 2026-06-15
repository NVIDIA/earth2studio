---
name: developer-model-checkpoint-update
version: 0.16.0
license: Apache-2.0
metadata:
  author: NVIDIA Earth-2 Team
  tags:
    - earth2studio
    - earth2
    - python
    - checkpoint
    - restart
    - models
    - inference
    - testing
description: >
  Add checkpoint restart support to an existing Earth2Studio model or component
  by inspecting its call and iterator semantics, identifying continuation state
  and random state, binding a pickle-free dataclass with bind_checkpoint_state,
  honoring minimal/state/full checkpoint policies, handling checkpoint device
  staging, and adding focused restart unit tests.
---

# Model Checkpoint Update - Add Restart Support

Add checkpoint restart support to an existing Earth2Studio model, perturbation,
or component without expanding its public model API. Use this skill when a
component needs to opt into `earth2studio.utils.checkpoint` restart state.

The goal is a small, explicit implementation: one internal dataclass, one
`bind_checkpoint_state(...)` call during construction, minimal changes to the
existing call/iterator path, and one focused unit test proving restart behavior.

---

## Design Rules

Follow these rules throughout the change:

1. Do not add required user-facing model arguments for checkpointing.
2. Do not add a new model base class, wrapper class, inheritance layer, or global
   registry.
3. Do not use pickle, `torch.save`, or arbitrary object serialization for
   checkpoint state.
4. Keep checkpoint state in a dataclass local to the component module unless
   there is already a better local pattern.
5. Use `bind_checkpoint_state` exactly once per restart state dataclass instance,
   normally in `__init__` after normal model fields are initialized.
6. Treat `minimal`, `state`, and `full` as user intent hints exposed through the
   bound checkpoint proxy.
7. Use `self.<state>.device` or `self.<state>.checkpoint_device` when staging live
   tensors so `Checkpoint(device=...)` controls where restart tensors live.
8. Keep IO backend output separate from model restart state. Do not assume saved
   user-facing output variables are sufficient to rehydrate model internals.
9. Add or update one focused unit test using existing test utility models when
   possible. Avoid broad mocks and avoid large integration fixtures.

---

## Step 1 - Read the Component Before Editing

Inspect the target component and its tests before deciding what to save.

Look for:

- Constructor inputs and persistent attributes.
- `__call__`, `forward`, `_forward`, `create_iterator`, or generator methods.
- Autoregressive state that is advanced between yields.
- Random number generation, including `torch.Generator`, NumPy generators,
  global Torch/NumPy RNG usage, perturbations, dropout-like behavior, or sampling.
- Pre/post hooks that mutate tensors or coordinates.
- Device moves, dtype casts, autocast blocks, and CPU/GPU assumptions.
- Existing test utility models such as Phoo, Random, Persistence, Identity, or
  local dummy modules.

Identify the restart boundary. For prognostic iterators, the saved boundary is
usually the latest completed forecast state. On restore, the iterator should
consume that saved boundary internally and yield the next forecast state.

---

## Step 2 - Choose the State Policy Behavior

Implement behavior for all three checkpoint policies:

- `minimal`: do not store component state. The workflow may still record catalog
  progress and explicit artifacts.
- `state`: store lightweight state needed to resume at workflow or forecast-run
  boundaries, such as RNG state, counters, selected member IDs, or replayable
  metadata. Do not store large continuation tensors here.
- `full`: store everything supported by the component to resume mid-rollout,
  including continuation tensors and coordinate metadata when user-facing IO is
  not restart-complete.

When a model cannot support a policy, make the fallback explicit in code. For
example, `state` may save RNG state while `full` saves RNG plus continuation
state. `minimal` should clear or avoid updating component state.

Do not expose these policies as new model constructor parameters. Read them from
the bound state proxy:

```python
if self.restart.checkpoint_state_policy == "full":
    ...
elif self.restart.checkpoint_state_policy == "state":
    ...
else:
    ...
```

---

## Step 3 - Add a Dataclass State

Create a dataclass containing only pickle-free serializable fields. Good field
examples include:

- `torch.Tensor | None`
- `np.ndarray`
- JSON-like scalars, tuples, lists, and dicts
- `np.datetime64`, `np.timedelta64`, `torch.dtype`, `torch.device`
- coordinate keys and coordinate values needed to reconstruct an `OrderedDict`

Avoid fields that hold live model objects, modules, hooks, data sources, IO
backends, callables, open files, generators, or arbitrary Python objects.

Example:

```python
@dataclass
class MyModelCheckpointState:
    x: torch.Tensor | None = None
    coord_keys: tuple[str, ...] = ()
    coord_values: tuple[np.ndarray, ...] = ()
    rng_state: torch.Tensor | None = None
```

Bind it in the constructor:

```python
self.restart = bind_checkpoint_state(MyModelCheckpointState())
```

Use a component-specific field name such as `restart`, `checkpoint`, or
`_checkpoint_state` that matches local style. Do not name dataclass fields
`device` or `checkpoint_*`; those names are reserved for checkpoint metadata on
the proxy.

---

## Step 4 - Restore State at the Right Boundary

Restore state where the component first has enough runtime context to do so.

For a simple callable component, this may be in `__call__` before generating the
next stochastic value. For a prognostic model, prefer the iterator construction
path so workflow code can still fetch and pass the normal initial condition.

Pattern for an iterator:

```python
restored = False
if self.restart.checkpoint_state_loaded and self.restart.x is not None:
    x = self.restart.x.to(x.device)
    coords = OrderedDict(
        (key, np.asarray(value).copy())
        for key, value in zip(self.restart.coord_keys, self.restart.coord_values)
    )
    restored = True

iterator = super().create_iterator(x, coords)
if restored:
    next(iterator)  # consume the saved boundary internally

for x_out, coords_out in iterator:
    self._save_checkpoint_state(x_out, coords_out)
    yield x_out, coords_out
```

If the component has both `__call__` and `create_iterator`, put shared save logic
in a small private helper so both paths update the dataclass consistently.

If hooks mutate the returned tensor or coordinates, save the post-hook state.
Checkpoint state should match the boundary that future computation will continue
from, not an earlier internal intermediate unless that is intentional and tested.

---

## Step 5 - Handle Random State Explicitly

Prefer a component-owned `torch.Generator` when possible. This avoids saving or
restoring global Torch or NumPy RNG state.

For stochastic components:

1. Create the generator internally from existing seed/input parameters.
2. If checkpoint state is loaded, restore the generator from the saved state.
3. For `state`, save the generator state needed to reproduce the next component
   call or forecast instance.
4. For `full`, save the generator state that correctly continues after the saved
   mid-rollout boundary.

Be precise about pre-state versus post-state:

- Save pre-call state when restart should replay the just-started stochastic
  operation.
- Save post-call state when restart should continue after the completed
  stochastic operation.

Do not gate RNG dataclass updates on `checkpoint_is_flush_due` unless stale state
cannot affect correctness. `flush_interval` should usually control disk writes,
not whether the live dataclass has the latest restart state.

---

## Step 6 - Stage Tensors on the Checkpoint Device

When saving live tensor state, detach and clone before staging it:

```python
self.restart.x = x.detach().clone().to(self.restart.device)
```

Use `self.restart.device` or `self.restart.checkpoint_device`; both are provided
by the bound checkpoint proxy. The default is CPU, but users may set
`Checkpoint(device=torch.device("cuda:0"))` to keep full checkpoint tensors on
the active inference device and reduce device transfers.

When restoring, move staged tensors to the runtime input/model device:

```python
x = self.restart.x.to(x.device)
```

Only store large tensors for `full`. For `minimal` and usually `state`, clear
large fields or leave them unset:

```python
self.restart.x = None
self.restart.coord_keys = ()
self.restart.coord_values = ()
```

---

## Step 7 - Write One Focused Unit Test

Add a restart test close to the component's existing tests. Prefer an existing
small model fixture over a broad mock.

The test should cover:

1. Construct a `Checkpoint(..., state_policy="full")` or `state` as appropriate.
2. Run the component long enough to write a checkpoint row.
3. Re-open or re-select the checkpoint with `checkpoint.select(-1)`.
4. Construct the component inside the selected checkpoint context.
5. Continue the run and assert the restarted result matches the uninterrupted
   reference or expected continuation.
6. Assert the component actually used hydrated state when that is observable.

Example skeleton:

```python
checkpoint = Checkpoint("model", path=tmp_path, mode="append", state_policy="full")

with checkpoint as ckpt:
    model = MyRestartableModel(...)
    x1, coords1 = next(model.create_iterator(x0, coords0))
    ckpt.write(lead_time=coords1["lead_time"][-1])
    ckpt.flush()

with checkpoint.select(-1) as ckpt:
    restarted = MyRestartableModel(...)
    out, coords = next(restarted.create_iterator(x0, coords0))

assert ...
```

For random components, compare the restarted sample sequence against a reference
sequence produced by the same seed. For prognostic models, compare both tensor
values and lead-time progression.

Run the smallest relevant test first, then the local test file if optional
dependencies allow it.

---

## Step 8 - Validate the Change

Run targeted checks before committing:

```bash
uv run ruff check <component-file> <test-file>
uv run pytest <test-file>::<new-test-name> -q
git diff --check
```

If the full test file cannot run because an optional dependency group is missing,
run the targeted new test and clearly report the optional dependency limitation.
Do not skip the focused restart test.

---

## Review Checklist

Before opening the PR, verify:

- Existing model construction still works without a checkpoint.
- Existing public model APIs are unchanged unless the user explicitly requested
  otherwise.
- `bind_checkpoint_state` is called during construction.
- `minimal`, `state`, and `full` behavior is explicit.
- Tensor state is staged with `.detach().clone().to(self.restart.device)` when
  relevant.
- Restore logic moves tensors back to the runtime device.
- Iterator restart consumes the saved boundary internally and yields the next
  forecast state.
- Random state uses a component-owned generator where practical.
- The test proves restart behavior rather than only checking serialization.
