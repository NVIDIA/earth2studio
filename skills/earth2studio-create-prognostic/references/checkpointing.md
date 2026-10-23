# Checkpointing Prognostic Models

Use checkpoint support when a prognostic wrapper has mutable state that is
needed to continue a rollout. A model that only applies a stateless forward
map does not need to bind checkpoint state; document that fact in the PR and
test the normal iterator path without checkpoint state.

## Decide What Belongs in State

Keep only restart state in the checkpoint dataclass:

- rollout tensors that are not available from the workflow's next input,
- rolling history or cached coordinates,
- counters, sampler state, or component RNG state needed for the next step.

Do not store model weights, normalization constants, or the complete forecast
history. Weights and static tensors belong to the model package, and forecast
fields belong in the configured IO backend.

## Bind Component State

Define a dataclass for the smallest restartable state and bind it in the model
constructor:

```python
from dataclasses import dataclass

import numpy as np
import torch

from earth2studio.utils.checkpoint import bind_checkpoint_state


@dataclass
class _ModelCheckpointState:
    x: torch.Tensor | None = None
    coord_keys: tuple[str, ...] = ()
    coord_values: tuple[np.ndarray, ...] = ()


class ModelName(...):
    def __init__(self, ...) -> None:
        super().__init__()
        self.checkpoint = bind_checkpoint_state(_ModelCheckpointState())
```

Construct the model inside the active `Checkpoint` or
`checkpoint.select(...)` context when an existing state must be restored during
initialization. The bound proxy exposes `checkpoint_level`,
`checkpoint_state_loaded`, `checkpoint_enabled`, and `device`.

## Rollout Lifecycle

1. At iterator setup, restore saved state only when
   `checkpoint_level == 2` and `checkpoint_state_loaded` is true.
2. On a fresh run, yield the supplied initial condition at lead time zero.
3. On a resumed run, use the saved boundary and yield the next forecast state;
   do not emit the already committed state twice.
4. After a successful forward step and rear hook, save the state needed by the
   next step. Clone and detach tensors, and stage them on
   `self.checkpoint.device`.
5. Let the workflow call `ckpt.write(...)` after the corresponding IO write;
   component state becomes durable when the checkpoint is flushed.

Checkpoint levels are user-configured: level 0 records workflow progress only,
level 1 supports component state needed to restart workflow items, and level 2
supports state needed to resume inside a rollout. A model must not assume that
its component state is available at lower levels.

## Round-Trip Test

Add a focused test with a temporary checkpoint directory. Run one or more
steps under `Checkpoint(..., level=2)`, call `ckpt.write` at the saved lead
time, then construct a new model inside `with checkpoint.select(-1):`. Verify
that the restored model emits the next lead time and the same state as an
uninterrupted rollout. Also cover the no-checkpoint path and, when applicable,
level 0 or level 1 behavior.

Use `test/models/px/test_fcn.py` and
`test/models/px/test_persistence.py` as reference implementations.
