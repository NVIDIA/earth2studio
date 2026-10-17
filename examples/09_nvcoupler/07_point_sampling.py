# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# %%
"""Sample a gridded DataArray onto a canonical PointGrid."""

import numpy as np
import xarray as xr

import earth2studio.nvcoupler as nvc
from earth2studio.grids import PointGrid
from earth2studio.nvcoupler.testing import grid_coords

NLAT, NLON = 32, 64
grid = grid_coords(NLAT, NLON)
lat, lon = grid["lat"], grid["lon"]

atmos = nvc.CallableComponent(
    "atmos",
    lambda array: array,
    "6h",
    exports=["air_temperature_2m"],
)
temperature = lat[:, None] + 0.1 * lon[None, :]
atmos_ic = xr.DataArray(
    temperature[None, ...],
    dims=("variable", "lat", "lon"),
    coords={"variable": ["air_temperature_2m"], **grid},
)

stations = PointGrid(
    latitude=np.array([lat[4], lat[10], (lat[6] + lat[7]) / 2]),
    longitude=np.array([lon[3], lon[20], (lon[12] + lon[13]) / 2]),
    x=np.array(["boulder", "denver", "midpoint"]),
)
site = nvc.CallableComponent(
    "stations",
    lambda array: array,
    "6h",
    imports=["air_temperature_2m"],
    grid=stations,
)

clock = nvc.Clock("2024-01-01", "2024-01-02", "6h")
atmos.realize(clock)
site.realize(clock)
atmos.initialize(atmos_ic)

nvc.Connector(atmos, site, sample="nearest").execute(clock.start)
nearest = site.import_state["air_temperature_2m"].array.copy()
nvc.Connector(atmos, site, sample="bilinear").execute(clock.start)
bilinear = site.import_state["air_temperature_2m"].array.copy()

expected_exact = lat[[4, 10]] + 0.1 * lon[[3, 20]]
expected_midpoint = (lat[6] + lat[7]) / 2 + 0.1 * (lon[12] + lon[13]) / 2
if not np.allclose(nearest[:2], expected_exact):
    raise ValueError("nearest sampling did not match grid points")
if not np.allclose(bilinear, [*expected_exact, expected_midpoint]):
    raise ValueError("bilinear sampling did not reproduce the linear field")
print(xr.Dataset({"nearest": nearest, "bilinear": bilinear}))
