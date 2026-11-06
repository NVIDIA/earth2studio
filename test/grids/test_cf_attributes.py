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

from earth2studio.grids import (
    CurvilinearGrid,
    HEALPixGrid,
    PointGrid,
    ProjectedGrid,
    resolve_grid,
)
from earth2studio.utils.coords import coord_array


@pytest.mark.parametrize("name", ["latlon-1deg", "latlon-1.5deg", "gaussian-f90"])
def test_geographic_cf_attributes(name):
    grid = resolve_grid(name)
    array = coord_array(grid.dims, grid=name)
    assert set(array.coords) == {"lat", "lon"}
    assert array.attrs["grid_mapping_name"] == "latitude_longitude"
    assert "grid_mapping" not in array.attrs
    for dim, standard_name, units, axis in (
        ("lat", "latitude", "degrees_north", "Y"),
        ("lon", "longitude", "degrees_east", "X"),
    ):
        assert array[dim].attrs["standard_name"] == standard_name
        assert array[dim].attrs["units"] == units
        assert array[dim].attrs["axis"] == axis
        assert array[dim].attrs == grid.coords(only_index=True)[dim].attrs


def test_projected_cf_attributes_with_renamed_explicit_coordinates():
    parent = resolve_grid("hrrr")
    grid = ProjectedGrid(parent.y[:2], parent.x[:3], parent.crs)
    explicit_y = xr.Variable("hrrr_y", grid.y, attrs={"source": "caller"})
    array = coord_array(
        ("hrrr_y", "hrrr_x"),
        coords={"hrrr_y": explicit_y},
        grid=grid,
        grid_dims={"y": "hrrr_y", "x": "hrrr_x"},
    )
    assert set(array.coords) == {"hrrr_y", "hrrr_x", "lat", "lon"}
    assert array.attrs["grid_mapping_name"] == "lambert_conformal_conic"
    assert array.attrs["standard_parallel"] == (38.5, 38.5)
    assert array.attrs["earth2studio_crs"] == grid.crs.to_string()
    assert "grid_mapping" not in array.attrs
    for dim, axis in (("hrrr_y", "Y"), ("hrrr_x", "X")):
        assert (
            array[dim].attrs["standard_name"] == f"projection_{axis.lower()}_coordinate"
        )
        assert array[dim].attrs["axis"] == axis
        assert array[dim].attrs["units"] == "metre"
    assert array.hrrr_y.attrs["source"] == "caller"
    assert explicit_y.attrs == {"source": "caller"}
    for dim in ("lat", "lon"):
        assert "axis" not in array[dim].attrs
    indexes = grid.coords(only_index=True)
    assert set(indexes) == {"y", "x"}
    assert indexes["x"].attrs == array.hrrr_x.attrs
    np.testing.assert_array_equal(array.hrrr_y, grid.y)


@pytest.mark.parametrize(
    "grid",
    [
        HEALPixGrid(1),
        HEALPixGrid(1, ordering="xy", layout="face"),
        CurvilinearGrid(np.zeros((2, 3)), np.ones((2, 3))),
        PointGrid(np.zeros(3), np.ones(3)),
    ],
)
def test_auxiliary_geographic_cf_attributes(grid):
    index_coords = grid.coords(only_index=True)
    assert set(index_coords) == set(grid.dims)
    coordinates = grid.coords()
    assert set(coordinates) == {*grid.dims, "lat", "lon"}
    for dim, standard_name, units in (
        ("lat", "latitude", "degrees_north"),
        ("lon", "longitude", "degrees_east"),
    ):
        assert coordinates[dim].attrs["standard_name"] == standard_name
        assert coordinates[dim].attrs["units"] == units
        assert "axis" not in coordinates[dim].attrs
    if grid.topology == "healpix":
        signature = coord_array(grid.dims, grid=grid)
        assert set(signature.coords) == set(grid.dims)
        assert "grid_mapping_name" not in signature.attrs
