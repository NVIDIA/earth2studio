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
    from cbottle.datasets import base
    from cbottle.inference import MixtureOfExpertsDenoiser
except ImportError:
    pytest.skip("cbottle dependencies not installed", allow_module_level=True)

from earth2studio.models.dx import CBottleTCGuidance
from earth2studio.utils import handshake_dim


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
    model2 = cbottle.models.get_model(model_config)
    return MixtureOfExpertsDenoiser(
        [model1, model2],
        (100.0, 10.0),
        batch_info=base.BatchInfo(CBottleTCGuidance.output_variables),
    )


@pytest.fixture(scope="class")
def mock_classifier_model() -> torch.nn.Module:
    # Real model checkpoint has
    # {"model_channels": 192, "label_dim": 1024, "out_channels": 45, "condition_channels": 1}
    model_config = cbottle.config.models.ModelConfigV1()
    model_config.model_channels = 4
    model_config.label_dim = 1024
    model_config.out_channels = 45
    model_config.condition_channels = 1
    model_config.level = 2
    model_config.enable_classifier = True
    return cbottle.models.get_model(model_config)


class TestCBottleTCMock:
    @pytest.mark.parametrize(
        "lat_coords,lon_coords",
        [
            (torch.tensor([30.0]), torch.tensor([120.0])),  # Single point
            (torch.tensor([30.0, 45.0]), torch.tensor([120.0, -80.0])),  # Two points
            (torch.tensor([-30.0]), torch.tensor([0.0])),  # Southern hemisphere
        ],
    )
    @pytest.mark.parametrize(
        "times", [[datetime(1990, 1, 1)], [datetime(1990, 1, 1), datetime(1990, 1, 2)]]
    )
    def test_create_guidance_tensor(self, lat_coords, lon_coords, times):
        """Test guidance tensor creation with different coordinate combinations"""
        guidance, coords = CBottleTCGuidance.create_guidance_tensor(
            lat_coords, lon_coords, times
        )

        assert guidance.shape == (len(times), 1, 1, 721, 1440)
        assert guidance.dtype == torch.float32
        assert torch.sum(guidance) == len(times) * len(
            lat_coords
        )  # One point per coordinate pair

        assert "time" in coords
        assert "lead_time" in coords
        assert "variable" in coords
        assert "lat" in coords
        assert "lon" in coords
        assert coords["variable"] == ["tc_guidance"]

    @pytest.mark.parametrize(
        "x,time",
        [
            (torch.zeros(1, 1, 1, 1, 721, 1440), np.array([datetime(2020, 1, 1)])),
            (
                torch.zeros(1, 2, 1, 1, 721, 1440),
                np.array([datetime(2000, 1, 2, 3, 4, 5), datetime(1980, 8, 1)]),
            ),
            (
                torch.zeros(2, 2, 1, 1, 721, 1440),
                np.array([datetime(2000, 1, 2, 3, 4, 5), datetime(1980, 8, 1)]),
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_tc_forward(
        self, x, time, device, mock_core_model, mock_classifier_model, mock_sst_ds
    ):
        dx = CBottleTCGuidance(mock_core_model, mock_classifier_model, mock_sst_ds).to(
            device
        )
        dx.sampler_steps = 2  # Speed up sampler
        dx.batch_size = 2

        coords = OrderedDict(
            {
                "batch": np.arange(x.shape[0]),
                "time": time,
                "lead_time": np.array([timedelta(hours=0)]),
                "variable": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )
        x = x.to(device)
        out, out_coords = dx(x, coords)

        assert out.shape == torch.Size(
            [x.shape[0], x.shape[1], x.shape[2], 45, 721, 1440]
        )
        assert np.all(out_coords["variable"] == dx.output_coords(coords)["variable"])
        handshake_dim(out_coords, "lon", 5)
        handshake_dim(out_coords, "lat", 4)
        handshake_dim(out_coords, "variable", 3)
        handshake_dim(out_coords, "lead_time", 2)
        handshake_dim(out_coords, "time", 1)
        handshake_dim(out_coords, "batch", 0)

    @pytest.mark.parametrize(
        "x,time",
        [
            (torch.zeros(1, 1, 1, 768), np.array([datetime(2020, 1, 1)])),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_tc_forward_hpx(
        self, x, time, device, mock_core_model, mock_classifier_model, mock_sst_ds
    ):
        dx = CBottleTCGuidance(
            mock_core_model, mock_classifier_model, mock_sst_ds, lat_lon=False
        ).to(device)
        dx.sampler_steps = 2  # Speed up sampler
        dx.batch_size = 2

        coords = OrderedDict(
            {
                "time": time,
                "lead_time": np.array([timedelta(hours=0)]),
                "variable": dx.input_coords()["variable"],
                "hpx": np.arange(x.shape[-1]),
            }
        )
        x = x.to(device)
        out, out_coords = dx(x, coords)

        assert out.shape == torch.Size([x.shape[0], x.shape[1], 45, 49152])
        assert np.all(out_coords["variable"] == dx.output_coords(coords)["variable"])
        handshake_dim(out_coords, "hpx", 3)
        handshake_dim(out_coords, "variable", 2)
        handshake_dim(out_coords, "lead_time", 1)
        handshake_dim(out_coords, "time", 0)

    def test_validate_sst_time_valid(
        self, mock_core_model, mock_classifier_model, mock_sst_ds
    ):
        valid_times = [
            datetime(1950, 6, 15),
            datetime(2000, 12, 31),
            datetime(2022, 12, 15),
        ]
        dx = CBottleTCGuidance(mock_core_model, mock_classifier_model, mock_sst_ds)
        # Should not raise any exceptions
        dx._validate_sst_time(valid_times)

        invalid_times = [datetime(1939, 12, 31)]
        with pytest.raises(ValueError):
            dx._validate_sst_time(invalid_times)

        invalid_times = [datetime(2022, 12, 16, 12)]
        with pytest.raises(ValueError):
            dx._validate_sst_time(invalid_times)


@pytest.mark.ci_cache
@pytest.mark.slow
@pytest.mark.timeout(30)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_cbottle_tc_package(device, model_cache_context):
    # Test the cached model package
    # Only cuda used here to speed things up, but CPU also works
    with model_cache_context():
        package = CBottleTCGuidance.load_default_package()
        dx = CBottleTCGuidance.load_model(package).to(device)

    # Guidance over florida
    lat = 27
    lon = -82
    time = np.array(
        [datetime(2000, 8, 9, 10), datetime(2005, 10, 11, 12)], dtype=np.datetime64
    )
    guidance, coords = CBottleTCGuidance.create_guidance_tensor(
        torch.tensor([lat]),
        torch.tensor([lon]),
        time,
    )
    guidance = guidance.to(device)

    out, out_coords = dx(guidance, coords)
    assert out.shape == torch.Size(
        [
            out_coords["time"].shape[0],
            out_coords["lead_time"].shape[0],
            out_coords["variable"].shape[0],
            721,
            1440,
        ]
    )
    assert np.all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    assert np.all(out_coords["time"] == time)
    handshake_dim(out_coords, "lon", -1)
    handshake_dim(out_coords, "lat", -2)
    handshake_dim(out_coords, "variable", -3)
    handshake_dim(out_coords, "lead_time", -4)
    handshake_dim(out_coords, "time", -5)

    # For a physical sanity check, see if tcwv is high at the guidance location
    vidx = np.where(out_coords["variable"] == "tcwv")[0]
    lat_idx = 4 * (90 - lat)
    lon_idx = 4 * (360 + lon)
    assert (out[:, :, vidx, lat_idx, lon_idx] >= 60).all()
