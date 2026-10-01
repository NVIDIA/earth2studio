# Checkpointing Diagnostic Models

Most deterministic diagnostic models are stateless: they transform one input
state into one output state and do not need checkpoint support. Do not bind a
checkpoint dataclass merely because a model loads weights or normalization
buffers. Record that the model is stateless and test its ordinary forward path.

## When State Is Required

Bind state only when a diagnostic has mutable information that changes the next
call, such as:

- a random-number-generator state or sampling counter,
- a stateful sampler, diffusion schedule, or cache,
- mutable auxiliary data required to reproduce the next generated sample.

There is no diagnostic iterator. A stateful diagnostic restores state during
construction and updates it after a successful call. Model weights, static
normalization tensors, and generated output history do not belong in the
checkpoint.

## Bind Component State

Use a small dataclass and bind it in `__init__`:

```python
from dataclasses import dataclass

import torch

from earth2studio.utils.checkpoint import bind_checkpoint_state


@dataclass
class _DiagnosticCheckpointState:
    rng_state: torch.Tensor | None = None
    sample_count: int = 0


class StatefulDiagnostic(...):
    def __init__(self, ...) -> None:
        super().__init__()
        self.checkpoint = bind_checkpoint_state(_DiagnosticCheckpointState())
        if self.checkpoint.rng_state is not None:
            self.generator.set_state(self.checkpoint.rng_state)
```

Construct the diagnostic inside the active `Checkpoint` or
`checkpoint.select(...)` context so saved state is restored before the first
sample. After sampling, update only the state needed for the next call. Save
RNG tensors with `detach().clone()` on `self.checkpoint.device` when required.
Use a dedicated `torch.Generator` where possible instead of changing global
RNG state.

## Round-Trip Test

For a stateful diagnostic, use a temporary `Checkpoint(..., level=1)` or the
level required by the component. Call the diagnostic, record a checkpoint
boundary with `ckpt.write(...)`, then construct a new instance inside
`with checkpoint.select(-1):`. Verify that the resumed call matches the next
call of an uninterrupted instance and that counters or RNG state advance once.

For a stateless diagnostic, do not add a synthetic checkpoint field just to
make this test pass. A normal forward test is sufficient; package tests should
continue to verify loading and inference separately.
