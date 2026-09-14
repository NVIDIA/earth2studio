# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
Earth2Studio Model Contract
===========================

Resolve coordinates, iterate a rollout, compose hooks, and check conformance.

A conforming model can be driven by any caller without wrapper-specific knowledge.
This tutorial walks the contract on a model that needs no weights, then checks it.
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
from collections import OrderedDict
from itertools import islice

import numpy as np
import torch

from earth2studio.models.conformance import (
    ContractException,
    check_prognostic_contract,
)
from earth2studio.models.px import Persistence
from earth2studio.utils.type import CoordSystem

domain = OrderedDict(
    {
        "lat": np.linspace(90, -90, 8, endpoint=False),
        "lon": np.linspace(0, 360, 16, endpoint=False),
    }
)
model = Persistence("t2m", domain, history=2)

# %%
# Read the Declaration
# --------------------
# ``input_coords()`` declares what a model accepts. Order is part of the contract,
# and a zero-length array marks an open dimension whose size the caller chooses.

# %%
declared = model.input_coords()
print(list(declared))
print(declared["batch"].size, declared["lead_time"])

# %%
# Lead time is relative and ends at zero, so two history steps six hours apart
# declare ``[-6h, 0h]``. The final entry is the step being advanced from.

# %%
# Resolve Output Coordinates
# --------------------------
# ``output_coords()`` validates a concrete coordinate system and returns what the
# model will produce, without touching field data. A caller can size a rollout
# before allocating anything.

# %%
coords = model.input_coords()
coords["batch"] = np.arange(1)
print(model.output_coords(coords)["lead_time"])

# %%
# Rebasing is what makes a rollout composable: the model adds its own step to the
# *final input* lead time, never to a constant. Shift the input and the output
# shifts with it, so a model can start from any point in a forecast.

# %%
rebased = coords.copy()
rebased["lead_time"] = coords["lead_time"] + np.timedelta64(24, "h")
print(model.output_coords(rebased)["lead_time"])

# %%
# Invalid coordinates are rejected rather than silently coerced.

# %%
misordered = OrderedDict(reversed(list(coords.items())))
try:
    model.output_coords(misordered)
except (ValueError, KeyError) as error:
    print(type(error).__name__)

# %%
# Iterate a Rollout
# -----------------
# ``create_iterator()`` yields the initial condition first, then forecast steps. A
# caller writing ``nsteps`` forecast steps therefore draws ``nsteps + 1`` yields.

# %%
x = torch.randn(1, 2, 1, 8, 16, generator=torch.Generator().manual_seed(0))
for step, (values, step_coords) in enumerate(model.create_iterator(x, coords.copy())):
    print(step, step_coords["lead_time"], tuple(values.shape))
    if step == 2:
        break

# %%
# The 0th yield carries the input coordinates with lead time reduced to its final
# entry — it is the initial condition, not a forecast.

# %%
# Compose Hooks
# -------------
# Hooks are the declared mutation points of a step, applied immediately before and
# after the model advances. They compose into a chain applied in registration order.
#
# Hooks belong to the iterator. ``__call__`` is a single-step primitive whose caller
# already holds the tensor and can transform it directly; what no caller can reach is
# the state fed back *between* steps, which is exactly what ``front_hook`` mutates.

# %%
# .. literalinclude:: ../../earth2studio/models/px/utils.py
#    :language: python
#    :start-after: # sphinx - hook chain start
#    :end-before: # sphinx - hook chain end

# %%
applied: list[np.ndarray] = []


@model.add_front_hook
def record(
    values: torch.Tensor, hook_coords: CoordSystem
) -> tuple[torch.Tensor, CoordSystem]:
    """Record the lead time each step is advanced from."""
    applied.append(hook_coords["lead_time"].copy())
    return values, hook_coords


@model.add_rear_hook
def offset(
    values: torch.Tensor, hook_coords: CoordSystem
) -> tuple[torch.Tensor, CoordSystem]:
    """Shift every predicted value, standing in for a bias correction."""
    return values + 1, hook_coords


iterator = model.create_iterator(x, coords.copy())
next(iterator)
step_values, _ = next(iterator)
print(len(applied), float(step_values.max()))

model.clear_hooks()

# %%
# Borrow the Input, Own the Output
# --------------------------------
# A model must not write into the tensor it is handed. The caller's initial
# condition survives the call, so it can be reused, retried, or written out.

# %%
before = x.clone()
model(x, coords.copy())
print(torch.equal(before, x))

# %%
# Yields must stay independent too: once a later step is produced, an earlier yield
# must not have changed. Anything that holds a yield across steps — an asynchronous
# IO write, a resume buffer, a scoring accumulator — depends on this.

# %%
held, snapshots = [], []
for values, _ in islice(model.create_iterator(x, coords.copy()), 3):
    held.append(values)
    snapshots.append(values.clone())

# Snapshot during iteration, not after: aliased yields have already converged on the
# final step's values by the time the rollout finishes.
print([torch.equal(a, b) for a, b in zip(held, snapshots)])

# %%
# Declare Stochasticity
# ---------------------
# A model declares whether it draws randomness, so a caller can decide whether
# ensemble members will differ at all before running anything. The declaration is a
# plain attribute that defaults to ``False``.

# %%
print(model.stochastic)


# %%
# A stochastic model implements ``set_rng(seed, reset=True)`` as its single seeding
# entry point. Reseeding with the same seed must reproduce a rollout exactly, which
# is what lets a resumed distributed run match the run it resumes.


# %%
class NoisyPersistence(Persistence):
    """Persistence with additive noise, standing in for a stochastic model."""

    stochastic = True

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the generator used to draw per-step noise."""
        if reset or getattr(self, "generator", None) is None:
            self.generator = torch.Generator().manual_seed(seed)

    def _forward(
        self, values: torch.Tensor, step_coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        out, out_coords = super()._forward(values, step_coords)
        # An unseeded stochastic model falls back to the global RNG rather than failing
        generator = getattr(self, "generator", None)
        return out + torch.randn(out.shape, generator=generator), out_coords


noisy = NoisyPersistence("t2m", domain, history=2)


def rollout(seed: int) -> float:
    """Run one forecast step from a given seed and return its first value."""
    noisy.set_rng(seed)
    iterator = noisy.create_iterator(x, coords.copy())
    next(iterator)
    return float(next(iterator)[0].flatten()[0])


print(rollout(0), rollout(0), rollout(1))

# %%
# Check Conformance
# -----------------
# One call evaluates every rule and reports all violations, so a wrapper author sees
# the full picture rather than the first failure. The return value lists rules that
# could not be evaluated and why.

# %%
print(check_prognostic_contract(model, rollout=False))
print(check_prognostic_contract(model))
print(check_prognostic_contract(noisy))

# %%
# A violation names the rule and what to do about it. Here a model declares itself
# stochastic but never reaches its randomness from ``set_rng``, so every ensemble
# member would be a duplicate.


# %%
class UnseededPersistence(NoisyPersistence):
    """A stochastic model whose set_rng misses a source of randomness."""

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Accept a seed and ignore it."""
        self.generator = torch.Generator().manual_seed(0)


try:
    check_prognostic_contract(UnseededPersistence("t2m", domain, history=2))
except ContractException as error:
    print(error)

# %%
# Keep Randomness Local
# ---------------------
# A seeded model must not leave the global RNG state perturbed. Seeding it reaches
# every other consumer in the process — a second model in a cascade, a perturbation
# method, a dataloader — and resets its stream. The failure hides during a
# single-model check and only appears once the model is one part of a pipeline, so
# the contract checks for it directly.


# %%
class GlobalSeedPersistence(NoisyPersistence):
    """A stochastic model that seeds the global generator."""

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the global RNG, reaching every other consumer in the process."""
        torch.manual_seed(seed)


try:
    check_prognostic_contract(GlobalSeedPersistence("t2m", domain, history=2))
except ContractException as error:
    print(error)

# %%
# The rule constrains the effect rather than the mechanism, which matters for a model
# whose randomness is drawn inside an external package that exposes no generator.
# Seeding globally is still allowed as long as it is confined to a
# ``torch.random.fork_rng()`` block, which restores the state on exit.


# %%
class ForkedSeedPersistence(NoisyPersistence):
    """A model that seeds globally, but only inside a fork."""

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Record the seed; the draw itself is seeded inside a fork."""
        if reset or getattr(self, "_seed", None) is None:
            self._seed = seed

    def _forward(
        self, values: torch.Tensor, step_coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        out, out_coords = Persistence._forward(self, values, step_coords)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._seed)
            noise = torch.randn(out.shape)
        return out + noise, out_coords


print(check_prognostic_contract(ForkedSeedPersistence("t2m", domain, history=2)))
