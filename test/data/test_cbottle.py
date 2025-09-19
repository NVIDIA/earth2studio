# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import datetime
import importlib

import numpy as np
import pytest
import torch
import xarray as xr

try:
    import cbottle
    import earth2grid
    from cbottle.datasets import base
    from cbottle.inference import MixtureOfExpertsDenoiser
except ImportError:
    pytest.skip("cbottle dependencies not installed", allow_module_level=True)

from earth2studio.data import CBottle3D


@pytest.fixture(scope="class")
def mock_core_model() -> torch.nn.Module:
    # Real model checkpoint has
    # {"model_channels": 192, "label_dim": 1024, "out_channels": 45, "condition_channels": 1}
    model_config = cbottle.config.models.ModelConfigV1()
    model_config.model_channels = 4
    model_config.label_dim = 1024
    model_config.out_channels = 45
    model_config.condition_channels = 1
    model_config.level = 2
    model1 = cbottle.models.get_model(model_config)
    return MixtureOfExpertsDenoiser(
        [model1],
        (),
        batch_info=base.BatchInfo(CBottle3D.VARIABLES),
    )


@pytest.fixture(scope="class")
def mock_sst_ds() -> torch.nn.Module:
    times = [np.datetime64("1870-01-16T12:00:00"), np.datetime64("2022-12-16T12:00:00")]
    lats = np.arange(-89.5, 90, 1.0)
    lons = np.arange(0.5, 360, 1.0)
    data = np.full((2, 180, 360), -1.8)  # In Celcius
    return xr.Dataset(
        data_vars=dict(
            tosbcs=xr.DataArray(
                data=data,
                dims=["time", "lat", "lon"],
                coords={"time": times, "lat": lats, "lon": lons},
            )
        )
    )


class TestCBottleMock:
    @pytest.mark.parametrize(
        "time",
        [
            datetime.datetime(year=1959, month=1, day=31),
            [
                datetime.datetime(year=1971, month=6, day=1, hour=6),
                datetime.datetime(year=2021, month=11, day=23, hour=12),
            ],
            np.array([np.datetime64("1993-04-05T00:00")]),
        ],
    )
    @pytest.mark.parametrize("variable", ["tcwv", ["u500", "u200"]])
    def test_cbottle_fetch(self, time, variable, mock_core_model, mock_sst_ds):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ds = CBottle3D(mock_core_model, mock_sst_ds).to(device)
        ds.sampler_steps = 4  # Speed up sampler
        data = ds(time, variable)
        shape = data.shape

        if isinstance(variable, str):
            variable = [variable]

        if isinstance(time, datetime.datetime):
            time = [time]

        assert shape[0] == len(time)
        assert shape[1] == len(variable)
        assert shape[2] == 721
        assert shape[3] == 1440
        assert np.array_equal(data.coords["variable"].values, np.array(variable))
        assert not np.isnan(data.values).any()

    @pytest.mark.parametrize(
        "time",
        [
            [
                datetime.datetime(year=1971, month=6, day=1, hour=6),
                datetime.datetime(year=2021, month=11, day=23, hour=12),
                datetime.datetime(year=1971, month=6, day=1, hour=6),
                datetime.datetime(year=2021, month=11, day=23, hour=12),
            ],
        ],
    )
    @pytest.mark.parametrize("variable", [["sst", "sic", "t700"]])
    @pytest.mark.parametrize("batch_size", [4, 2, 1])
    def test_cbottle_batches(
        self, time, variable, batch_size, mock_core_model, mock_sst_ds
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ds = CBottle3D(mock_core_model, mock_sst_ds, batch_size=batch_size).to(device)
        ds.sampler_steps = 4  # Speed up sampler
        data = ds(time, variable)
        shape = data.shape

        if isinstance(variable, str):
            variable = [variable]

        if isinstance(time, datetime.datetime):
            time = [time]

        assert shape[0] == len(time)
        assert shape[1] == len(variable)
        assert shape[2] == 721
        assert shape[3] == 1440
        assert np.array_equal(data.coords["variable"].values, np.array(variable))
        assert not np.isnan(data.values).any()

    @pytest.mark.parametrize(
        "time",
        [
            [
                datetime.datetime(year=1959, month=1, day=31),
                datetime.datetime(year=1971, month=6, day=1, hour=6),
            ]
        ],
    )
    @pytest.mark.parametrize("variable", [["v10m"], ["rlut", "tpf"]])
    def test_cbottle_hpx(self, time, variable, mock_core_model, mock_sst_ds):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ds = CBottle3D(mock_core_model, mock_sst_ds, lat_lon=False, seed=1).to(device)
        ds.sampler_steps = 4  # Speed up sampler
        data_hpx = ds(time, variable)
        shape = data_hpx.shape

        assert shape[0] == len(time)
        assert shape[1] == len(variable)

        ds = CBottle3D(mock_core_model, mock_sst_ds, lat_lon=True, seed=1).to(device)
        ds.sampler_steps = 4  # Speed up sampler
        data_latlon = ds(time, variable)
        shape = data_latlon.shape

        assert shape[0] == len(time)
        assert shape[1] == len(variable)
        assert shape[2] == 721
        assert shape[3] == 1440

        # Manually regrid the hpx
        nlat, nlon = 721, 1440
        latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
            nlat, nlon, includes_south_pole=True
        )
        src_grid = earth2grid.healpix.Grid(
            level=6, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        regridder = earth2grid.get_regridder(src_grid, latlon_grid).to(device)
        field_regridded = regridder(
            torch.tensor(data_hpx.values, device=device)
        ).squeeze(2)

        # Check both grids from the same seed are the same
        assert np.allclose(data_latlon.values, field_regridded.cpu().numpy())


@pytest.mark.slow
@pytest.mark.ci_cache
@pytest.mark.timeout(60)
@pytest.mark.parametrize("time", [datetime.datetime(year=2000, month=12, day=31)])
@pytest.mark.parametrize("variable", [["sic", "u10m", "t2m"]])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_cbottle_package(time, variable, device, model_cache_context):
    # Test the cached model package
    with model_cache_context():
        package = CBottle3D.load_default_package()
        ds = CBottle3D.load_model(package).to(device)

    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert not np.isnan(data.values).any()
    # Check physical ranges
    assert (data.sel(variable="sic").values >= -0.1).all() and (
        data.sel(variable="sic").values <= 1.1
    ).all()
    assert (data.sel(variable="u10m").values >= -40).all() and (
        data.sel(variable="u10m").values <= 40
    ).all()
    assert (data.sel(variable="t2m").values >= 184).all() and (
        data.sel(variable="t2m").values <= 330
    ).all()


@pytest.mark.slow
@pytest.mark.ci_cache
@pytest.mark.timeout(60)
@pytest.mark.parametrize("time", [datetime.datetime(year=2000, month=12, day=31)])
@pytest.mark.parametrize("variable", [["sic", "u10m", "t2m"]])
@pytest.mark.skipif(
    importlib.util.find_spec("apex.contrib.group_norm") is not None,
    reason="Test requires apex.contrib.group_norm to not be installed",
)
def test_cbottle_package_cpu(time, variable, model_cache_context):
    # Test the cached model package
    with model_cache_context():
        package = CBottle3D.load_default_package()
        ds = CBottle3D.load_model(package)
        ds.sampler_steps = 2  # Speed up sampler for testing

    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
