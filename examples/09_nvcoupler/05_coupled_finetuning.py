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
Gradients Across the Exchange: Coupled Fine-Tuning
==================================================

Backpropagating a coupled-rollout loss into both components.

Separately-trained emulators drift when coupled: each was trained against
truth forcing, not against the other model's imperfect output. The remedy is
coupled fine-tuning — optimizing both models jointly on coupled rollouts.
nvcoupler's exchange path (regrid gathers, mediator reductions, functional
import injection) is autograd-clean end to end, and ``driver.rollout()``
keeps the graph, so a loss on one component's final state reaches the
parameters of every component upstream through the exchanges.

This example fine-tunes the two scalar "physics" parameters of the toy
system so the 96 h coupled forecast hits a target. The point is not the toy
optimization — it is that gradients cross the coupler.

In this example you will learn:

- How rollout() differs from run()/steps() (graph kept vs inference mode)
- How to verify gradients reach parameters through the exchange
- The shape of a minimal coupled training step
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Trainable Components
# --------------------
# Each toy takes a gain parameter inside its step function. gain_ocean can
# influence the atmosphere ONLY through the exchange chain:
# ocean sst -> connector regrid -> import injection -> atmos step.

import torch

import earth2studio.nvcoupler as nvc
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

SEQUENCE = """
@6h
  atmos -> med
  ocean -> atmos
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
"""

gain_atmos = torch.tensor(1.0, requires_grad=True)
gain_ocean = torch.tensor(1.0, requires_grad=True)


def coupled_forecast() -> torch.Tensor:
    """One 96 h coupled rollout; returns the final mean z1000 (graph kept)."""
    driver = nvc.Driver(
        {
            "atmos": fake_atmos(gain=gain_atmos),
            "ocean": fake_ocean(gain=gain_ocean),
            "med": nvc.TrailingAverageMediator(
                "med", ["geopotential_at_1000hpa_48h_mean"]
            ),
        },
        sequence=SEQUENCE,
        clock=nvc.Clock("2024-01-01", "2024-01-05", "6h"),
        collect=False,
    )
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    states = driver.rollout(16)  # full 96h, autograd graph retained
    return states["atmos"]["geopotential_at_1000hpa"].data.mean()


# %%
# Gradients Cross the Coupler
# ---------------------------
# With gains at 1.0 the forecast lands at 19.2336 (see example 01). A loss on
# the atmosphere's final state produces nonzero gradients for BOTH gains —
# the ocean's only route to the loss is through two connectors and the
# mediator.

with torch.enable_grad():
    z96 = coupled_forecast()
    z96.backward()

print(f"z1000(96h) = {z96.item():.4f}")
print(f"d(loss)/d(gain_atmos) = {gain_atmos.grad.item():.4f}")
print(f"d(loss)/d(gain_ocean) = {gain_ocean.grad.item():.6f}")
if gain_atmos.grad == 0 or gain_ocean.grad == 0:
    raise ValueError("a gradient failed to cross the exchange path")
print("gradients reached both components through the exchange ✓\n")

# %%
# A Minimal Coupled Training Loop
# -------------------------------
# Fine-tune both gains so the coupled forecast hits z1000(96h) = 25. Each
# iteration rebuilds the system from the same initial conditions (a fresh
# clock) and takes one optimizer step — the skeleton of coupled fine-tuning.

TARGET = 25.0
optimizer = torch.optim.Adam([gain_atmos, gain_ocean], lr=0.1)

for it in range(60):
    optimizer.zero_grad()
    with torch.enable_grad():
        z96 = coupled_forecast()
        loss = (z96 - TARGET) ** 2
        loss.backward()
    optimizer.step()
    if it % 10 == 0 or it == 59:
        print(
            f"iter {it:2d}: z96 = {z96.item():7.4f}  loss = {loss.item():9.5f}  "
            f"gains = ({gain_atmos.item():.4f}, {gain_ocean.item():.4f})"
        )

if abs(z96.item() - TARGET) >= 0.5:
    raise ValueError("coupled fine-tuning failed to reach the target forecast")
print(f"\ncoupled system fine-tuned to the target ({z96.item():.3f} ≈ {TARGET}) ✓")
