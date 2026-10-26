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

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import NearestRegridder
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

T0 = np.datetime64("2026-01-01T00")
ATMOS_GRID = toy_grid(*ATMOS_SHAPE)
OCEAN_GRID = toy_grid(*OCEAN_SHAPE)


def _masked_sst() -> tuple[xr.DataArray, np.ndarray]:
    """Ocean initial condition: NaN land over the northern half, 2 K elsewhere."""
    ocean = np.ones(OCEAN_SHAPE, dtype=bool)
    ocean[: OCEAN_SHAPE[0] // 2, :] = False
    values = np.where(ocean, 2.0, np.nan).astype(np.float32)
    ic = xr.DataArray(
        values[None, None],
        dims=("lead_time", "variable", "lat", "lon"),
        coords={
            "lead_time": np.array([0], dtype="timedelta64[h]"),
            "variable": ["sst"],
            **OCEAN_GRID.coords(only_index=True),
        },
    )
    return ic, ocean


def test_land_masked_sst_is_filled_from_the_nearest_ocean_point():
    ic, ocean = _masked_sst()
    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        [
            Connector(
                "ocean",
                "atmos",
                regridder=NearestRegridder(OCEAN_GRID, ATMOS_GRID, source_mask=ocean),
            ),
            Connector("atmos", "ocean"),
        ],
        {"atmos": toy_source(ATMOS_GRID, 0.0), "ocean": toy_source(OCEAN_GRID, 2.0)},
    )
    steps = list(
        coupler.run_item(WorkItem(T0, np.timedelta64(6, "h")), ics={"ocean": ic})
    )
    z1000 = steps[1]["atmos"]
    # Every atmosphere point, land included, received a finite SST of 2 K.
    assert not np.isnan(z1000.values).any()
    np.testing.assert_allclose(z1000.values, 1.2)


def test_without_the_mask_land_nans_reach_the_atmosphere():
    ic, _ = _masked_sst()
    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        [Connector("ocean", "atmos"), Connector("atmos", "ocean")],
        {"atmos": toy_source(ATMOS_GRID, 0.0), "ocean": toy_source(OCEAN_GRID, 2.0)},
    )
    steps = list(
        coupler.run_item(WorkItem(T0, np.timedelta64(6, "h")), ics={"ocean": ic})
    )
    assert np.isnan(steps[1]["atmos"].values).any()


def test_mask_must_match_the_source_grid():
    with pytest.raises(ValueError, match="does not match the source grid"):
        NearestRegridder(OCEAN_GRID, ATMOS_GRID, source_mask=np.ones(ATMOS_SHAPE, bool))
