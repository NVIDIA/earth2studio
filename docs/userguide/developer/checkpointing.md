# Checkpointing

Earth2Studio checkpointing is designed for long-running inference jobs that need
to restart without asking every model, perturbation, or custom loop to adopt a
new component API. A `Checkpoint` manages a small set of saved restart points.
Each row is selected by labels such as `time`, `ensemble`, or a user-defined
scenario name, and each row can contain small workflow artifacts plus dataclass
state from components that opt in.

The checkpoint does not store model weights or duplicate data already written by
an IO backend. Forecast fields should continue to be written to the selected IO
backend. The checkpoint stores the position and small state needed to decide
where to restart.

## Basic Use

```python
from earth2studio.run import deterministic
from earth2studio.utils.checkpoint import Checkpoint

checkpoint = Checkpoint("my-forecast", flush_interval=6, state_policy="full")

with checkpoint as ckpt:
    deterministic(
        time=["2024-01-01"],
        nsteps=24,
        prognostic=model,
        data=data,
        io=io,
        checkpoint=ckpt,
    )
```

Use checkpointed workflows inside a checkpoint context so restart-aware models
and perturbations bind to the active session before the run begins. The built-in
deterministic, diagnostic, and ensemble workflows write checkpoint rows after
successful IO writes. `flush_interval` controls durable writes to
disk. The workflow can call `write` on the active checkpoint session every
iteration while the checkpoint decides whether a real disk commit is due.
Ensemble workflow rows are tracked independently by `ensemble_batch` when a
checkpoint is provided.

Use `flush_interval=1` for every write call to commit immediately, or
`flush_interval=None` to keep updates pending until `flush` is called. The
workflow flushes the final pending state before returning. When checkpointing is
omitted, built-in workflows use a no-op checkpoint session with the same
`write` and `flush` methods.

`state_policy` is a user intent hint exposed to bound component state. It does
not force a model to checkpoint a particular payload. Supported values are:

- `minimal`: save only catalog progress and explicit workflow artifacts.
- `state`: save lightweight component restart state such as RNG, generator, or
  counter state needed to resume at workflow or forecast-instance boundaries.
- `full`: save all supported restart state, including heavy tensors needed to
  resume inside a rollout. This is the default.

## Selecting Restart Points

`Checkpoint.select` returns a `CheckpointSession`, which chooses a saved row or a
future label set. The selected labels also become the labels for future writes in
that session.

```python
checkpoint = Checkpoint("my-forecast")

print(checkpoint)

latest = checkpoint.select(-1)
latest_time = checkpoint.select(time=-1)
first_member = checkpoint.select(time="2024-01-01T00:00:00", ensemble=0)
```

The integer positional argument selects catalog rows, with negative indexing
supported. Keyword selections choose the latest matching row. A keyword value of
`-1` means the latest saved value for that label after all other labels are
applied.

## Custom Loops

Custom workflows can call `write` after a safe iteration boundary, usually after
the forecast fields for that iteration have been written to the IO backend.

```python
checkpoint = Checkpoint("ensemble-forecast", mode="append", history_size=8)

with checkpoint.select(time=time, ensemble=member) as ckpt:
    for lead_time in lead_times:
        coords, array = step_model(...)
        io.write(coords, array)

        ckpt.write(
            lead_time=lead_time,
            artifacts={"last_complete_lead_time": lead_time},
        )

    ckpt.flush()
```

`write` accepts explicit `lead_time` and `artifacts` keyword arguments.
`lead_time` records the latest completed forecast position. `artifacts` is for
small user-provided restart metadata. `flush()` with no arguments commits the
latest pending `write`; if there is nothing pending, it is a no-op. Arbitrary
keyword arguments are intentionally not accepted so checkpoint payloads remain
explicit.

`mode="overwrite"` keeps only the latest row for a label set. `mode="append"`
keeps a history, and `history_size` can cap that history.

## Workflow Resume

Built-in workflows always fetch the normal initial condition and feed it to the
prognostic model iterator. When a checkpoint row is selected, the workflow uses
the row's lead time only as the completed workflow position. A checkpoint-aware
model is responsible for using its bound dataclass state inside `create_iterator`
to restore the selected restart point. If state is restored, the iterator should
consume that saved boundary internally and yield the next forecast state. If no
state is restored, the iterator should keep the normal convention of yielding
the initial condition first.

This keeps restart independent from model internals and avoids assuming that
user-facing IO output is restart-complete. Diagnostic workflows still track the
prognostic lead time before diagnostic output is written. Ensemble workflows store
progress per mini-batch using an `ensemble_batch` label, allowing completed
batches to be skipped and partially completed batches to continue from their
latest saved lead time.

For custom loops, print the checkpoint and select the desired row by index, for
example `checkpoint.select(-1)` for the latest row. `Checkpoint` and
`CheckpointSession` both support context-manager use. Built-in workflows accept
either the checkpoint manager or a selected `CheckpointSession`. Passing the
manager while a session is active uses that active session; passing a manager
with no active session chooses the latest matching workflow row, or starts a new
row when no matching checkpoint exists.

## Component State

Models, perturbations, and user-defined components can opt in by binding a
dataclass instance. Existing components do not need to change.

```python
from dataclasses import dataclass

import torch

from earth2studio.utils.checkpoint import bind_checkpoint_state


@dataclass
class NoiseState:
    calls: int = 0
    rng_state: torch.Tensor | None = None


class NoisePerturbation:
    def __init__(self, generator: torch.Generator):
        self.generator = generator
        self.checkpoint = bind_checkpoint_state(NoiseState())

        if self.checkpoint.rng_state is not None:
            self.generator.set_state(self.checkpoint.rng_state)

    def __call__(self, x):
        y = add_noise(x, generator=self.generator)
        self.checkpoint.calls += 1
        if self.checkpoint.checkpoint_enabled:
            self.checkpoint.rng_state = self.generator.get_state()
        return y
```

`bind_checkpoint_state` returns a proxy around the original dataclass. Normal
dataclass fields are accessed directly, while checkpoint metadata is exposed
through `checkpoint_*` properties such as `checkpoint_enabled`,
`checkpoint_state_policy`, `checkpoint_device`, `checkpoint_flush_interval`,
`checkpoint_write_count`, `checkpoint_is_flush_due`, `checkpoint_selected`,
`checkpoint_state_loaded`, and `checkpoint_lead_time`. The shorter `device`
alias is provided for tensor staging, for example `x.to(self.restart.device)`.
Because `device` is reserved checkpoint metadata, component state dataclasses
should use a different field name for their own device values. Live tensor state
is staged on CPU by default; pass `device=` to `Checkpoint` when components
should stage it elsewhere.

When a checkpoint session is active, `bind_checkpoint_state` loads the matching
saved state if one exists and registers the live dataclass instance for future
writes. Model state can be lightweight `state` policy metadata, such as lead
time and RNG state, or heavier `full` policy tensors when the model cannot be
restarted from user-facing IO output alone. Built-in opt-in examples include
`Persistence` and `FCN`/FourCastNet full-state restart support. When no session
is active but a `Checkpoint` has been instantiated, `bind_checkpoint_state`
buffers the live dataclass for that checkpoint. The buffered state is registered
when a session for that checkpoint is entered.


A model can use the checkpoint policy hint without adding another API call:

```python
if self.restart.checkpoint_state_loaded:
    x = self.restart.x.to(x.device)

if self.restart.checkpoint_state_policy == "full":
    self.restart.x = x.detach().clone().to(self.restart.device)
elif self.restart.checkpoint_state_policy == "state":
    self.restart.step = step
    self.restart.rng_state = generator.get_state()
else:
    self.restart.x = None
```

This makes new runs simple while keeping the active checkpoint session explicit:

```python
checkpoint = Checkpoint("my-forecast")

with checkpoint as ckpt:
    model = MyRestartableModel(...)
    deterministic(..., checkpoint=ckpt)
```

For strict restart hydration from an existing row, construct restartable
components inside the selected session:

```python
checkpoint = Checkpoint("my-forecast")

with checkpoint.select(-1) as ckpt:
    model = MyRestartableModel(...)
    perturbation = NoisePerturbation(generator)
    deterministic(..., checkpoint=ckpt)
```

If a component binds state before an existing checkpoint session is entered, the
session will still hydrate that dataclass when it opens, but Earth2Studio emits a
warning because constructor side effects that already used the default state will
not be replayed.

State identity is based on the dataclass type, using its fully qualified module
and class name. Binding the same dataclass type more than once in one checkpoint
session raises an error, because the checkpoint would otherwise not know which
saved payload belongs to which component. Use separate dataclass types for
distinct restartable components.

## Serialization Rules

Checkpoint state is intentionally pickle-free. Supported values include
JSON-like scalars and containers, dataclasses, `datetime`, `date`, `timedelta`,
`numpy.datetime64`, `numpy.timedelta64`, `numpy.dtype`, NumPy scalars,
`torch.device`, `torch.dtype`, `torch.Tensor`, and `numpy.ndarray` with
non-object dtype. Tensors and arrays are stored as separate `.npy` files with
pickle disabled.

Unsupported objects raise `CheckpointSerializationError` during checkpoint
writes. If a dataclass definition changes incompatibly between save and restore,
checkpoint binding raises `CheckpointStateSchemaError` instead of guessing.

## Distributed Runs

Serial runs write directly into the checkpoint directory. Distributed runs write
each process to its own rank directory, such as `rank_000000` or `rank_000001`.
Rank detection first checks PhysicsNeMo's distributed manager when available,
then common distributed environment variables. The checkpoint does not use file
locks across ranks.

## Storage Layout

By default, checkpoints are stored under
`$EARTH2STUDIO_CACHE/checkpoints/<name>` or `~/.cache/earth2studio/checkpoints/<name>`.
Pass `path=` to store them elsewhere.

Each durable write is staged in a temporary directory and then atomically moved
into the commit directory. The catalog file is also written atomically. Incomplete
temporary commits are ignored and cleaned up by later writes.
