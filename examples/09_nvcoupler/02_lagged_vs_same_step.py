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
Lagged versus Same-Step Coupling
================================

Coupling order follows from what the destination declares.

NUOPC distinguishes *explicit* (lagged) coupling, where a model sees its
partner's previous state, from *sequential* coupling, where it sees the state
just produced. In Earth2Studio a model declares the lead time of the forcing
frame each step needs, so the coupler reads the mode off the declaration and
orders the run sequence to satisfy it.

In this example you will learn:

- how a forcing lead of ``0h`` versus ``+6h`` changes the exchange
- how :func:`~earth2studio.nvcoupler.derive_sequence` orders same-cadence models
- what happens when an explicit run sequence violates a dependency
"""

# /// script
# dependencies = [
#   "earth2studio[utils] @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
import numpy as np

from earth2studio.nvcoupler import (
    Connector,
    Coupler,
    CouplingError,
    RunSequence,
    Slot,
)
from earth2studio.nvcoupler.testing import ToyPrognostic, toy_grid, toy_source
from earth2studio.run import WorkItem

GRID = toy_grid(32, 64)
ITEM = WorkItem(np.datetime64("2026-01-01T00"), np.timedelta64(18, "h"))


def forced_by(forcing_lead: str) -> ToyPrognostic:
    """A 6-hourly model whose state ``b`` adds its partner's ``a`` each step."""
    return ToyPrognostic(
        ["b"],
        forcing=["a"],
        forcing_lead=forcing_lead,
        advance=lambda x, f: x + f.isel(lead_time=-1, drop=True).values,
    )


def run(destination: ToyPrognostic) -> Coupler:
    """Couple ``a`` (which counts 0, 1, 2, ...) into ``destination``."""
    coupler = Coupler(
        {"b": destination, "a": ToyPrognostic(["a"])},
        connectors=[Connector("a", "b")],
        sources={"a": toy_source(GRID, 0.0), "b": toy_source(GRID, 0.0)},
    )
    print(coupler.sequence)
    for step in coupler.run_item(ITEM):
        if "b" in step:
            lead = int(
                step["b"]["lead_time"].values[0].astype("timedelta64[h]").astype(int)
            )
            print(f"  b at {lead:2d}h = {float(step['b'].values.ravel()[0]):.0f}")
    return coupler


# %%
# Lagged: forcing at lead 0h
# --------------------------
# ``b`` declares the frame of ``a`` valid at its own input time. That frame was
# produced by the previous step (or is the initial condition), so no ordering
# constraint exists and models keep declaration order: ``b`` before ``a``.

# %%
lagged = run(forced_by("0h"))
print(lagged.models["b"].calls)

# %%
# Same step: forcing at lead +6h
# ------------------------------
# ``b`` declares the frame of ``a`` valid at the *end* of its step. Only ``a``'s
# current step produces it, so the coupler moves ``a`` ahead of ``b``.

# %%
same_step = run(forced_by("6h"))
print(same_step.models["b"].calls)

# %%
# The lagged run adds 0, 1, 2 and the same-step run adds 1, 2, 3. Both are
# correct couplings; which one a model wants is part of its contract.

# %%
# Overriding the sequence
# -----------------------
# A :class:`~earth2studio.nvcoupler.RunSequence` can be passed explicitly, but it
# must still satisfy every same-step dependency the declarations imply.

# %%
try:
    Coupler(
        {"b": forced_by("6h"), "a": ToyPrognostic(["a"])},
        connectors=[Connector("a", "b")],
        sources={"a": toy_source(GRID, 0.0), "b": toy_source(GRID, 0.0)},
        sequence=RunSequence((Slot("6h", ("b", "a")),)),
    )
except CouplingError as error:
    print(error)
