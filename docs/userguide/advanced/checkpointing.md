# Checkpointing

Earth2Studio checkpoints enable users to restart inference workflows. A
checkpoint is a catalog of restart points for one named run. Each row in that
catalog is a checkpoint session: it records the completed lead time, optional
artifacts, and opt-in component state(s) needed to restart the workflow.

Checkpoint storage is independent of the user-facing IO backend and independent
of any particular model implementation. Forecast fields remain in the IO backend
you choose, model weights are not copied into checkpoints, and the checkpoint
catalog stores only the restart metadata and component state requested by the
configured checkpoint level. Exactly what gets logged is user-configurable
through the checkpoint options and through the checkpoint support implemented
by each component.

```{warning}
Checkpointing support is opt-in for every component. Not all models,
perturbations, or custom components support checkpointed restarts. Always verify
that the model you plan to use has checkpoint support before relying on it
for restartable inference. If checkpointing is missing for a model you need, open
a [feature request](https://github.com/NVIDIA/earth2studio/issues).
```

## Basic Use

```python
from earth2studio.run import deterministic
from earth2studio.utils.checkpoint import Checkpoint

checkpoint = Checkpoint("my-forecast", flush_interval=6, level=2)

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

Use checkpointed workflows inside a checkpoint context. This makes the active
session clear and lets restart-aware models or perturbations bind their state
before the run starts. Built-in workflows automatically interact with the
checkpoint session: they call `write` after successful IO writes and call
`flush` before returning. If no checkpoint is supplied, they use the
`NullCheckpoint` no-op checkpoint.

`Checkpoint` options:

- `flush_interval=1`: commit every workflow write.
- `flush_interval=None`: keep writes pending until `flush()`.
- `mode="overwrite"`: keep only the latest row.
- `mode="append"`: keep a row history; cap it with `history_size`.
- `device=torch.device("cpu")`: device used by components for staged tensor state.

Components can opt into checkpoint state when they need restart information
(RNG state, counters, tensors, etc.). The user chooses the requested level,
and each component decides what it supports:

- `0`: no component logging; workflows still record catalog progress and explicit artifacts.
- `1`: enough component state to restart a workflow item such as an ensemble member.
- `2`: full component state for restarting inside a rollout when the component supports it.

## Selecting Rows

Print a checkpoint to inspect its catalog, then select a row by integer index.
Negative indexing is supported.

```python
checkpoint = Checkpoint("my-forecast")

print(checkpoint)

latest = checkpoint.select(-1)
first = checkpoint.select(0)
```

By default, the context manager `with checkpoint` selects the latest catalog row
when one exists, or opens a new session when the catalog is empty. Use
`with checkpoint.select(-1):` when a specific saved row should be restored.

## Custom Loops

Call `write` after a safe restart boundary, usually after forecast fields have
been written to IO. Call `flush()` to force the latest pending write to disk.

```python
checkpoint = Checkpoint("forecast", mode="append", history_size=8)

with checkpoint as ckpt:
    for lead_time in lead_times:
        x, coords = step_model(...)
        io.write(*split_coords(x, coords))
        ckpt.write(
            lead_time=lead_time,
            artifacts={"last_complete_lead_time": lead_time},
        )

    ckpt.flush()
```

`write` accepts only `lead_time` and `artifacts`. Artifacts are for small explicit
restart metadata; large forecast arrays should remain in the IO backend.

## Workflow Resume

Built-in workflows, like `run.deterministic`, always fetch the normal initial
condition and pass it to the prognostic iterator.
The checkpoint session tells the workflow what has been already completed.
Checkpoint-aware models decide whether to restore their own state and, if restored,
should yield the next forecast state rather than the existing saved boundary.

This decouples user-facing IO from what might be required to resume a run via a
checkpoint: users often save only a subset of generated forecast variables, but
continuing a rollout may require all fields and possibly internal model state.

## Component State

Models, perturbations, and custom components opt in by binding a dataclass.
Existing components do not need to change.

```python
from dataclasses import dataclass

import torch

from earth2studio.utils.checkpoint import bind_checkpoint_state


@dataclass
class NoiseState:
    rng_state: torch.Tensor | None = None


class NoisePerturbation:
    def __init__(self, generator: torch.Generator):
        self.generator = generator
        self.checkpoint = bind_checkpoint_state(NoiseState())

        if self.checkpoint.rng_state is not None:
            self.generator.set_state(self.checkpoint.rng_state)

    def __call__(self, x):
        y = add_noise(x, generator=self.generator)
        if self.checkpoint.checkpoint_level >= 1:
            self.checkpoint.rng_state = self.generator.get_state()
        return y
```

`bind_checkpoint_state` returns a proxy around the dataclass. Normal dataclass
fields are accessed directly. Checkpoint metadata is available through read-only
properties such as `checkpoint_enabled`, `checkpoint_level`,
`checkpoint_state_loaded`, `checkpoint_lead_time`, and `device`. Use `device` for
staging tensor state, for example
`x.detach().clone().to(self.checkpoint.device)`.

Construct restart-aware components inside the checkpoint context when saved
state must be restored during initialization:

```python
checkpoint = Checkpoint("my-forecast")

with checkpoint.select(-1) as ckpt:
    model = MyRestartableModel(...)
    deterministic(..., checkpoint=ckpt)
```

If a component binds before an existing session is active, Earth2Studio restores
it when the session opens and warns because constructor side effects may already
have used default state.

State identity is the dataclass type's fully qualified module and class name.
Binding the same dataclass type twice in one session raises
`CheckpointStateCollision`; use distinct dataclass types for distinct components.

## Serialization

Checkpoint state is pickle-free. Supported values include JSON-like scalars and
containers, dataclasses, `datetime`, `date`, `timedelta`, `numpy.datetime64`,
`numpy.timedelta64`, NumPy scalars and dtypes, `torch.device`, `torch.dtype`,
`torch.Tensor`, and non-object `numpy.ndarray`. Tensors and arrays are stored as
separate `.npy` files with pickle disabled.

Unsupported values raise `CheckpointSerializationError`. Incompatible dataclass
schema changes raise `CheckpointStateSchemaError` during restore.

## Storage

Default location:

```text
$EARTH2STUDIO_CACHE/checkpoints/<name>
```

or, if `EARTH2STUDIO_CACHE` is unset:

```text
~/.cache/earth2studio/checkpoints/<name>
```

Pass `path=` to choose another location. Serial runs write directly into that
folder. Distributed runs write per-rank folders such as `rank_000000`; rank
detection checks PhysicsNeMo's distributed manager first, then common distributed
environment variables.

Checkpoint writes are staged in temporary directories and atomically moved into
place when complete. Catalog JSON writes use the same temporary-file pattern.
Incomplete temporary writes are ignored and cleaned up by later writes.
