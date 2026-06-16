# Checkpointing

Earth2Studio checkpoints are small restart catalogs for inference workflows. A
checkpoint row records progress labels, the latest completed lead time, optional
small artifacts, and dataclass state from components that opt in. Forecast fields
stay in the selected IO backend; model weights are not copied into checkpoints.

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

Use checkpointed workflows inside a checkpoint context. This makes the active
session clear and lets restart-aware models or perturbations bind their state
before the run starts. Built-in workflows call `write` after successful IO writes
and call `flush` before returning. If no checkpoint is supplied, they use the
package no-op checkpoint session.

`Checkpoint` options:

- `flush_interval=1`: commit every workflow write.
- `flush_interval=None`: keep writes pending until `flush()`.
- `mode="overwrite"`: keep the latest row for a label set.
- `mode="append"`: keep a row history; cap it with `history_size`.
- `device=torch.device("cpu")`: device used by components for staged tensor state.

`state_policy` is a hint for opt-in components:

- `minimal`: catalog progress and explicit artifacts only.
- `state`: lightweight restart state such as RNG or counters.
- `full`: all supported restart state, including tensors needed to resume inside a rollout.

## Selecting Rows

Print a checkpoint to inspect its catalog, then select a row by index or labels.
Negative indexing is supported. Label value `-1` means the latest saved value for
that label after applying any other label filters.

```python
checkpoint = Checkpoint("my-forecast")

print(checkpoint)

latest = checkpoint.select(-1)
latest_time = checkpoint.select(time=-1)
member = checkpoint.select(time="2024-01-01T00:00:00", ensemble=0)
```

`with checkpoint` selects the latest catalog row when one exists, or opens a new
session when the catalog is empty. Use `with checkpoint.select(...):` when a
specific row or label set is required.

## Custom Loops

Call `write` after a safe restart boundary, usually after forecast fields have
been written to IO. Call `flush()` to force the latest pending write to disk.

```python
checkpoint = Checkpoint("ensemble-forecast", mode="append", history_size=8)

with checkpoint.select(time=time, ensemble=member) as ckpt:
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

Built-in workflows always fetch the normal initial condition and pass it to the
prognostic iterator. The checkpoint row tells the workflow which writes already
completed. Checkpoint-aware models decide whether to restore their own dataclass
state and, if restored, should yield the next forecast state rather than the
saved boundary.

This avoids assuming that user-facing IO output is restart-complete. It matters
when IO stores only selected variables but a model needs more internal state to
continue a rollout. Ensemble workflows use the `ensemble_batch` label so each
mini-batch can resume independently.

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
        if self.checkpoint.checkpoint_enabled:
            self.checkpoint.rng_state = self.generator.get_state()
        return y
```

`bind_checkpoint_state` returns a proxy around the dataclass. Normal dataclass
fields are accessed directly. Checkpoint metadata is available through read-only
properties such as `checkpoint_enabled`, `checkpoint_state_policy`,
`checkpoint_state_loaded`, `checkpoint_lead_time`, `checkpoint_labels`, and
`device`. Use `device` for staging tensor state, for example
`x.detach().clone().to(self.checkpoint.device)`.

Construct restart-aware components inside the checkpoint context when hydration
must happen during initialization:

```python
checkpoint = Checkpoint("my-forecast")

with checkpoint.select(-1) as ckpt:
    model = MyRestartableModel(...)
    deterministic(..., checkpoint=ckpt)
```

If a component binds before an existing session is active, Earth2Studio hydrates
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
schema changes raise `CheckpointStateSchemaError` during hydration.

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
