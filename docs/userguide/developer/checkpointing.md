# Checkpointing

Earth2Studio checkpointing is designed for long-running inference jobs that need
to restart without asking every model, perturbation, or custom loop to adopt a
new component API. A checkpoint is a small catalog of saved restart points. Each
row is selected by labels such as `time`, `ensemble`, or a user-defined scenario
name, and each row can contain small workflow artifacts plus dataclass state from
components that opt in.

The checkpoint does not store model weights or duplicate data already written by
an IO backend. Forecast fields should continue to be written to the selected IO
backend. The checkpoint stores the position and small state needed to decide
where to restart.

## Basic Use

```python
from earth2studio.run import deterministic
from earth2studio.utils.checkpoint import Checkpoint

checkpoint = Checkpoint("my-forecast", flush_interval=6)

deterministic(
    time=["2024-01-01"],
    nsteps=24,
    prognostic=model,
    data=data,
    io=io,
    checkpoint=checkpoint,
)
```

The built-in deterministic, diagnostic, and ensemble workflows write checkpoint
rows after successful IO writes. `flush_interval` controls durable writes to
disk. The workflow can call `Checkpoint.write` every iteration while the
checkpoint object decides whether a real disk commit is due. Ensemble workflow
rows are tracked independently by `ensemble_batch` when a checkpoint catalog is
provided.

Use `flush_interval=1` for every write call to commit immediately, or
`flush_interval=None` to keep updates pending until `flush` is called. The
workflow flushes the final pending state before returning.

## Selecting Restart Points

`Checkpoint.select` behaves like a small catalog selection. The selected labels
also become the labels for future writes in that context.

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
checkpoint = Checkpoint("ensemble-forecast", mode="append", keep_last=8)

with checkpoint.select(time=time, ensemble=member) as ckpt:
    for lead_time in lead_times:
        coords, array = step_model(...)
        io.write(coords, array)

        ckpt.write(
            coords=coords,
            artifacts={"last_complete_lead_time": lead_time},
        )

    ckpt.flush(coords=coords)
```

`write` accepts explicit `coords` and `artifacts` keyword arguments. `coords` are
used to record the latest workflow position, including `lead_time` when present.
`artifacts` is for small user-provided restart metadata. Arbitrary keyword
arguments are intentionally not accepted so checkpoint payloads remain explicit.

`mode="overwrite"` keeps only the latest row for a label set. `mode="append"`
keeps a history, and `keep_last` can cap that history.

## Workflow Resume

The built-in deterministic workflow can resume from forecast fields already
written to the IO backend. The diagnostic workflow can also resume when the IO
backend contains the prognostic variables needed for the next forecast step.
The ensemble workflow stores progress per mini-batch using an `ensemble_batch`
label, allowing completed batches to be skipped and partially completed batches
to continue from their latest saved lead time.

For custom loops, print the checkpoint catalog and select the desired row by
index, for example `checkpoint.select(-1)` for the latest row.

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
        self.checkpoint.rng_state = self.generator.get_state()
        return y
```

When no checkpoint context is active, `bind_checkpoint_state` simply returns the
dataclass instance unchanged. Inside a selected checkpoint context, it loads the
matching saved state if one exists and registers the live dataclass instance for
future writes.

State identity is based on the dataclass type, using its fully qualified module
and class name. Binding the same dataclass type more than once in one checkpoint
selection raises an error, because the checkpoint would otherwise not know which
saved payload belongs to which component. Use separate dataclass types for
distinct restartable components.

Components that bind state in `__init__` should be constructed inside the
selected checkpoint context when restart hydration is required:

```python
with checkpoint.select(time=time, ensemble=member) as ckpt:
    model = MyRestartableModel(...)
    perturbation = NoisePerturbation(generator)
    run_custom_loop(model, perturbation, ckpt)
```

Components that bind lazily during `__call__` or iterator creation only need that
call to happen inside the context.

## Serialization Rules

Checkpoint state is intentionally pickle-free. Supported values include JSON-like
scalars and containers, dataclasses, `datetime`, `date`, `timedelta`,
`numpy.datetime64`, `numpy.timedelta64`, `numpy.dtype`, NumPy scalars,
`torch.device`, `torch.dtype`, `torch.Tensor`, and `numpy.ndarray` with
non-object dtype. Tensors and arrays are stored as separate `.npy` files with
pickle disabled.

Unsupported objects raise `CheckpointSerializationError` during checkpoint
writes. If a dataclass definition changes incompatibly between save and restore,
checkpoint binding raises `CheckpointStateSchemaError` instead of guessing.

## Distributed Runs

Each process writes to its own rank directory, such as `rank_000000` or
`rank_000001`. Rank detection first checks PhysicsNeMo's distributed manager
when available, then common distributed environment variables. The checkpoint
does not use file locks across ranks.

## Storage Layout

By default, checkpoints are stored under
`$EARTH2STUDIO_CACHE/checkpoints/<name>` or `~/.cache/earth2studio/checkpoints/<name>`.
Pass `path=` to store them elsewhere.

Each durable write is staged in a temporary directory and then atomically moved
into the commit directory. The catalog file is also written atomically. Incomplete
temporary commits are ignored and cleaned up by later writes.
