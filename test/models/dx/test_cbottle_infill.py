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

from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
import xarray as xr

try:
    import cbottle
except ImportError:
    pytest.skip("cbottle dependencies not installed", allow_module_level=True)

from earth2studio.models.dx import CBottleInfill
from earth2studio.utils import handshake_dim


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
    return cbottle.models.get_model(model_config)


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
        "input_variables",
        [
            np.array(["u10m", "v10m"]),
            np.array(["t2m", "z1000", "sic"]),
        ],
    )
    @pytest.mark.parametrize(
        "time,lead_time",
        [
            (
                np.array(
                    [datetime(2020, 1, 1, 6, 2, 3), datetime(1990, 5, 6, 7, 8, 9)]
                ),
                np.array([timedelta(0)]),
            ),
            (
                np.array([datetime(2006, 12, 13, 12, 36)]),
                np.array([timedelta(hours=6), timedelta(hours=12, minutes=5)]),
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_infill(
        self, input_variables, time, lead_time, device, mock_core_model, mock_sst_ds
    ):
        dx = CBottleInfill(mock_core_model, mock_sst_ds, input_variables).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        x = torch.randn(
            time.shape[0], lead_time.shape[0], input_variables.shape[0], 721, 1440
        ).to(device)
        coords = OrderedDict(
            {
                "time": time,
                "lead_time": lead_time,
                "variable": input_variables,
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        out, out_coords = dx(x, coords)

        assert out.shape == torch.Size(
            [
                time.shape[0],
                lead_time.shape[0],
                out_coords["variable"].shape[0],
                721,
                1440,
            ]
        )
        assert np.all(out_coords["variable"] == dx.output_coords(coords)["variable"])
        assert np.all(out_coords["time"] == time)
        assert np.all(out_coords["lead_time"] == lead_time)
        handshake_dim(out_coords, "lon", 4)
        handshake_dim(out_coords, "lat", 3)
        handshake_dim(out_coords, "variable", 2)
        handshake_dim(out_coords, "lead_time", 1)
        handshake_dim(out_coords, "time", 0)
        assert not torch.isnan(out).any()
        # Assert the provided fields the same (fairly close, theres still interpolation)
        torch.allclose(out[:, :, dx.input_variable_idx], x, rtol=0.05)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_infill_exceptions(self, device, mock_core_model, mock_sst_ds):

        dx = CBottleInfill(mock_core_model, mock_sst_ds, ["t2m"]).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        x = torch.randn(1).to(device)
        wrong_coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "time": dx.input_coords()["time"],
                "lead_time": dx.input_coords()["lead_time"],
                "wrong": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )

        with pytest.raises((KeyError, ValueError)):
            dx(x, wrong_coords)

        wrong_coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "time": dx.input_coords()["time"],
                "variable": dx.input_coords()["variable"],
                "lon": dx.input_coords()["lon"],
                "lat": dx.input_coords()["lat"],
            }
        )

        with pytest.raises(ValueError):
            dx(x, wrong_coords)

        wrong_coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "time": dx.input_coords()["time"],
                "lead_time": dx.input_coords()["lead_time"],
                "variable": dx.input_coords()["variable"],
                "lat": np.linspace(-90, 90, 720),
                "lon": dx.input_coords()["lon"],
            }
        )
        with pytest.raises(ValueError):
            dx(x, wrong_coords)

    @pytest.mark.parametrize(
        "input_variables",
        [
            np.array(["u10m"]),
            np.array(["sst", "z1000", "sic"]),
            np.array(["v1000", "sst"]),
        ],
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_cbottle_infill_sst(
        self, input_variables, device, mock_core_model, mock_sst_ds
    ):

        dx = CBottleInfill(mock_core_model, mock_sst_ds, input_variables).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        # With AMIP time range
        time = np.array([datetime(2020, 1, 1, 1), datetime(2021, 1, 1, 1)])
        lead_time = np.array([timedelta(hours=1)])

        x = torch.randn(
            1, time.shape[0], lead_time.shape[0], input_variables.shape[0], 721, 1440
        ).to(device)
        coords = OrderedDict(
            {
                "ensemble": np.array([1]),
                "time": time,
                "lead_time": lead_time,
                "variable": input_variables,
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        out, out_coords = dx(x, coords)

        # Outside of AMIP time range
        coords["time"] = np.array([datetime(2023, 1, 1, 1), datetime(2002, 2, 2, 2)])

        if "sst" in input_variables:
            out, coords = dx(x, coords)
        else:
            with pytest.raises(ValueError):
                out, coords = dx(x, coords)

    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_cbottle_infill_invariant_inputs(
        self, device, mock_core_model, mock_sst_ds
    ):
        # Checks a few invariant inputs that should produce the same result
        input_variables = np.array(["u10m", "v10m"])
        dx = CBottleInfill(mock_core_model, mock_sst_ds, input_variables).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        time = np.array([datetime(1995, 8, 2, 3, 12)])
        lead_time = np.array([timedelta(hours=6)])

        x = torch.randn(
            1, time.shape[0], lead_time.shape[0], input_variables.shape[0], 721, 1440
        ).to(device)
        coords = OrderedDict(
            {
                "ensemble": np.array([1]),
                "time": time,
                "lead_time": lead_time,
                "variable": input_variables,
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        torch.manual_seed(0)
        dx.set_seed(0)
        out0, out_coords = dx(x, coords)

        # Adjust time and lead time dim so data is at same timestamp
        coords["time"] = np.array([datetime(1995, 8, 2, 9, 12)])
        coords["lead_time"] = np.array([timedelta(hours=0)])
        torch.manual_seed(0)
        dx.set_seed(0)
        out1, out_coords = dx(x, coords)

        # Permute variables
        input_variables = np.array(["v10m", "u10m"])
        dx = CBottleInfill(mock_core_model, mock_sst_ds, input_variables).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        coords["variable"] = input_variables
        x = torch.flip(x, dims=[-3])
        torch.manual_seed(0)
        dx.set_seed(0)
        out2, out_coords = dx(x, coords)

        assert torch.allclose(out0, out1)
        assert torch.allclose(out0, out2)


@pytest.mark.ci_cache
@pytest.mark.slow
@pytest.mark.timeout(30)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_cbottle_package(device, model_cache_context):
    # Test the cached model package
    # Only cuda supported
    input_variables = np.array(["tpf"])
    with model_cache_context():
        package = CBottleInfill.load_default_package()
        dx = CBottleInfill.load_model(package, input_variables=input_variables).to(
            device
        )

    time = np.array([datetime(2020, 1, 1, 1), datetime(2021, 1, 1, 1)])
    lead_time = np.array([timedelta(hours=1)])
    x = torch.zeros(
        1, time.shape[0], lead_time.shape[0], input_variables.shape[0], 721, 1440
    ).to(device)
    coords = OrderedDict(
        {
            "ensemble": np.array([1]),
            "time": time,
            "lead_time": lead_time,
            "variable": input_variables,
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )

    out, out_coords = dx(x, coords)

    assert out.shape == torch.Size(
        [
            1,
            time.shape[0],
            lead_time.shape[0],
            out_coords["variable"].shape[0],
            721,
            1440,
        ]
    )
    assert np.all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    assert np.all(out_coords["time"] == time)
    assert np.all(out_coords["lead_time"] == lead_time)
    handshake_dim(out_coords, "lon", -1)
    handshake_dim(out_coords, "lat", -2)
    handshake_dim(out_coords, "variable", -3)
    handshake_dim(out_coords, "lead_time", -4)
    handshake_dim(out_coords, "time", -5)

    # Check physical ranges
    vidx = np.where(out_coords["variable"] == "sic")[0]
    assert (out[:, :, :, vidx] >= -0.1).all() and (out[:, :, :, vidx] <= 1.1).all()
    vidx = np.where(out_coords["variable"] == "u10m")[0]
    assert (out[:, :, :, vidx] >= -40).all() and (out[:, :, :, vidx] <= 40).all()
    vidx = np.where(out_coords["variable"] == "t2m")[0]
    assert (out[:, :, :, vidx] >= 184).all() and (out[:, :, :, vidx] <= 330).all()
