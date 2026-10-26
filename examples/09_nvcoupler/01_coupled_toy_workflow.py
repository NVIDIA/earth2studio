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
Coupled Atmosphere and Ocean
============================

Couple two prognostic models with different cadences and grids.

A 6-hourly atmosphere on a 32 x 64 grid takes sea-surface temperature from a
48-hourly ocean on a 16 x 32 grid; the ocean takes the 48 h mean of the
atmosphere's ``z1000``. Both are ordinary prognostic models: the coupler reads
what each one needs from its ``forcing_coords()`` and what it provides from its
``output_coords()``, and nothing is coupler-specific about them.

In this example you will learn:

- declaring exchanges with :class:`~earth2studio.nvcoupler.Connector`
- how the coupler derives cadences, regridders and the run sequence
- windowed exchange through qualified labels such as ``z1000:mean:48h``
- stepping a :class:`~earth2studio.nvcoupler.Coupler` and reading its streams
"""

# /// script
# dependencies = [
#   "earth2studio[utils] @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Toy models
# ----------
# ``earth2studio.nvcoupler.testing`` ships two toy prognostic models with
# hand-computable arithmetic in place of networks. The atmosphere declares a
# forcing slot ``sst`` at lead ``0h``: each step from ``T`` to ``T + 6h`` uses the
# SST valid at ``T``. The ocean declares ``z1000:mean:48h`` at lead ``+48h``: its
# step from ``T`` to ``T + 48h`` uses the atmosphere's mean over ``[T, T + 48h)``.
#
# .. math::
#
#     z_{1000} \leftarrow z_{1000} + 1 + 0.1\,\mathrm{sst}, \qquad
#     \mathrm{sst} \leftarrow \mathrm{sst} + 0.01\,\overline{z_{1000}}^{48h}

# %%
import numpy as np

from earth2studio.nvcoupler import Connector, Coupler
from earth2studio.nvcoupler.testing import (
    ATMOS_SHAPE,
    OCEAN_SHAPE,
    ToyAtmosphere,
    ToyOcean,
    toy_grid,
    toy_source,
)
from earth2studio.run import WorkItem

atmos, ocean = ToyAtmosphere(), ToyOcean()
print(
    atmos.forcing_coords()["variable"].values,
    atmos.forcing_coords()["lead_time"].values,
)
print(
    ocean.forcing_coords()["variable"].values,
    ocean.forcing_coords()["lead_time"].values,
)

# %%
# Declare the coupled system
# --------------------------
# A :class:`~earth2studio.nvcoupler.Connector` is a frozen declaration: which
# destination labels a source serves. With no ``variables`` it serves every
# destination import the source exports; ``z1000:mean:48h`` is served by reducing
# the atmosphere's ``z1000`` frames (see ``dev/spec/TIME_STATISTICS_SPEC.md``).
#
# Whatever connectors do not serve comes from each model's bound data source, as
# in :class:`~earth2studio.run.PrognosticRunner`: here only the initial conditions.

# %%
coupler = Coupler(
    {"atmos": atmos, "ocean": ocean},
    connectors=[Connector("ocean", "atmos"), Connector("atmos", "ocean")],
    sources={
        "atmos": (toy_source(toy_grid(*ATMOS_SHAPE), 0.0), None),
        "ocean": (toy_source(toy_grid(*OCEAN_SHAPE), 2.0), None),
    },
)
print(coupler.describe())

# %%
# The plan shows what the coupler derived: a 6 h clock (the GCD of the cadences),
# bilinear regridders between the two grids, and a run sequence in which the
# atmosphere advances before the ocean because the ocean needs the atmosphere's
# frames from the *same* 48 h step, while the atmosphere's SST is lagged.

# %%
# Run and read the streams
# ------------------------
# The coupler is a :class:`~earth2studio.run.Runner`. Each step of
# :meth:`~earth2studio.nvcoupler.Coupler.run_item` yields the outputs of the
# models that advanced, keyed by model name, with ``time`` and ``lead_time``
# dimensions ready for IO. The first step publishes the initial conditions.

# %%
item = WorkItem(np.datetime64("2026-01-01T00"), np.timedelta64(96, "h"))
steps = list(coupler.run_item(item))
print(len(steps), "steps;", "first step holds", sorted(steps[0]))


def corner(step: dict, stream: str) -> float:
    """One grid point of a published field."""
    return float(step[stream].isel(lat=1, lon=1).values.ravel()[0])


print("z1000 at 6h:", corner(steps[1], "atmos"))  # 0 + 1 + 0.1 * 2
print("sst at 48h:", corner(steps[8], "ocean"))  # 2 + 0.01 * mean(0, 1.2, ..., 8.4)
print("z1000 at 54h:", corner(steps[9], "atmos"))  # 9.6 + 1 + 0.1 * 2.042

# %%
# The ocean appears only at 48 h and 96 h; the atmosphere every 6 h. Its SST from
# 48 h on is ``2.042``, and the atmosphere's increment grows accordingly.

# %%
np.testing.assert_allclose(corner(steps[8], "ocean"), 2.042)
np.testing.assert_allclose(corner(steps[9], "atmos"), 10.8042)
print([sorted(step) for step in steps[7:10]])
