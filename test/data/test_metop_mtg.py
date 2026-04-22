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

import pathlib
import shutil
from datetime import datetime
from unittest.mock import patch

import netCDF4
import numpy as np
import pytest

from earth2studio.data import MetOpMTG


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time,variable",
    [
        (datetime(2025, 3, 1, 12, 0), "fci09"),
        (
            [datetime(2025, 3, 1, 12, 0), datetime(2025, 3, 1, 12, 10)],
            ["fci09", "fci12"],
        ),
    ],
)
def test_metop_mtg_fetch(time, variable):
    ds = MetOpMTG(resolution="2km", cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(time, datetime):
        time = [time]
    if isinstance(variable, str):
        variable = [variable]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 5568
    assert shape[3] == 5568
    assert not np.all(np.isnan(data.values))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("cache", [True, False])
def test_metop_mtg_cache(cache):
    ds = MetOpMTG(resolution="2km", cache=cache)
    time = datetime(2025, 3, 1, 12, 0)
    variable = "fci09"
    data = ds(time, variable)
    assert data.shape[0] == 1

    assert pathlib.Path(ds.cache).is_dir() == cache

    # Reload from cache
    data2 = ds(time, variable)
    assert data2.shape[0] == 1

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_metop_mtg_bbox_fetch():
    bbox = (35.0, -10.0, 60.0, 30.0)  # Europe (lat_min, lon_min, lat_max, lon_max)
    ds = MetOpMTG(resolution="2km", lat_lon_bbox=bbox, cache=False)
    time = datetime(2025, 3, 1, 12, 0)
    variable = "fci09"
    data = ds(time, variable)
    shape = data.shape
    assert shape[0] == 1
    assert shape[1] == 1
    # Cropped output must be smaller than full disk
    assert shape[2] < 5568
    assert shape[3] < 5568
    assert not np.all(np.isnan(data.values))


def _build_body_nc(path: pathlib.Path, channel: str, rows: int, cols: int) -> None:
    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    grp = ds.createGroup("data").createGroup(channel).createGroup("measured")
    grp.createDimension("y", rows)
    grp.createDimension("x", cols)
    var = grp.createVariable("effective_radiance", "f4", ("y", "x"))
    var[:] = np.random.default_rng(42).random((rows, cols), dtype=np.float32)
    ds.close()


def test_metop_mtg_call_mock(tmp_path):
    rows, cols = 32, 32
    product_dir = tmp_path / "mtg_mock"
    product_dir.mkdir()
    _build_body_nc(product_dir / "BODY_001.nc", "ir_87", rows, cols)

    lat = np.linspace(-10, 10, rows * cols).reshape(rows, cols).astype(np.float32)
    lon = np.linspace(-10, 10, rows * cols).reshape(rows, cols).astype(np.float32)

    small_grid_params = {
        "2km": (81918.0, 2784.5, 81918.0, 2784.5, (rows, cols)),
        "1km": (163836.0, 5569.0, 163836.0, 5569.0, (rows * 2, cols * 2)),
    }

    with (
        patch.object(MetOpMTG, "_fetch_product", return_value=str(product_dir)),
        patch.object(MetOpMTG, "_ensure_grid", return_value=(lat, lon)),
        patch("earth2studio.data.metop_mtg._GRID_PARAMS", small_grid_params),
    ):
        ds = MetOpMTG(resolution="2km", cache=False, verbose=False)
        data = ds(datetime(2025, 3, 1, 12, 0), "fci09")

        assert data.shape == (1, 1, rows, cols)
        assert not np.all(np.isnan(data.values))
        assert "_lat" in data.coords
        assert "_lon" in data.coords


@pytest.mark.timeout(15)
def test_metop_mtg_grid():
    lat, lon = MetOpMTG.grid(resolution="2km")
    assert lat.shape == (5568, 5568)
    assert lon.shape == (5568, 5568)
    assert not np.all(np.isnan(lat))
    assert not np.all(np.isnan(lon))


@pytest.mark.timeout(60)
def test_metop_mtg_grid_1km():
    lat, lon = MetOpMTG.grid(resolution="1km")
    assert lat.shape == (11136, 11136)
    assert lon.shape == (11136, 11136)


@pytest.mark.timeout(15)
def test_metop_mtg_available():
    assert MetOpMTG.available(datetime(2024, 1, 16, 12, 0))
    assert MetOpMTG.available(datetime(2025, 3, 1, 0, 0))
    assert not MetOpMTG.available(datetime(2023, 12, 31, 0, 0))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
def test_metop_mtg_mixed_resolution_error():
    ds = MetOpMTG(resolution="2km")
    with pytest.raises(ValueError, match="resolution"):
        ds(
            datetime(2025, 1, 1, 0, 0),
            ["fci01", "fci09"],
        )
