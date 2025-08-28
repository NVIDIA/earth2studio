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
from datetime import datetime

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

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import CBottleVideo
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
    # {"model_channels": 256, "label_dim": 1024, "out_channels": 45, "condition_channels": 47}
    model_config = cbottle.config.models.ModelConfigV1()
    model_config.model_channels = 64
    model_config.label_dim = 1024
    model_config.out_channels = 45
    model_config.time_length = 12
    model_config.num_groups = 16
    model_config.condition_channels = 47
    model_config.level = 2
    model1 = cbottle.models.get_model(model_config)
    return MixtureOfExpertsDenoiser(
        [model1],
        (),
        batch_info=base.BatchInfo(CBottleVideo.VARIABLES),
    )


class TestCBottleVideoMock:

    @pytest.mark.parametrize(
        "x,time",
        [
            (
                torch.zeros(1, 1, 1, 45, 721, 1440),
                np.array([datetime(2020, 1, 1)], dtype=np.datetime64),
            ),
            (
                torch.full((1, 2, 1, 45, 721, 1440), fill_value=torch.nan),
                np.array(
                    [datetime(2000, 1, 2, 3, 4, 5), datetime(1980, 8, 1)],
                    dtype=np.datetime64,
                ),
            ),
            (
                torch.zeros(2, 2, 1, 45, 721, 1440),
                np.array(
                    [datetime(2000, 1, 2, 3, 4, 5), datetime(1980, 8, 1)],
                    dtype=np.datetime64,
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "device", ["cuda:0"]
    )  # , "cpu" takes too long, it should work but skipping
    def test_cbottle_video_forward(self, x, time, device, mock_core_model, mock_sst_ds):
        px = CBottleVideo(mock_core_model, mock_sst_ds).to(device)
        px.sampler_steps = 2  # Speed up sampler

        coords = px.input_coords()
        coords["batch"] = np.arange(x.shape[0])
        coords["time"] = time

        x = x.to(device)
        out, out_coords = px(x, coords)

        assert out.shape == torch.Size(
            [x.shape[0], x.shape[1], x.shape[2], 45, 721, 1440]
        )
        assert np.all(out_coords["variable"] == px.output_coords(coords)["variable"])
        handshake_dim(out_coords, "lon", 5)
        handshake_dim(out_coords, "lat", 4)
        handshake_dim(out_coords, "variable", 3)
        handshake_dim(out_coords, "lead_time", 2)
        handshake_dim(out_coords, "time", 1)
        handshake_dim(out_coords, "batch", 0)

    @pytest.mark.parametrize(
        "x,time",
        [
            (
                torch.zeros(1, 1, 1, 45, 49152),
                np.array([datetime(2020, 1, 1)], dtype=np.datetime64),
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_cbottle_video_hpx_forward(
        self, x, time, device, mock_core_model, mock_sst_ds
    ):
        px = CBottleVideo(mock_core_model, mock_sst_ds, lat_lon=False).to(device)
        px.sampler_steps = 2  # Speed up sampler

        coords = px.input_coords()
        coords["batch"] = np.arange(x.shape[0])
        coords["time"] = time

        x = x.to(device)
        out, out_coords = px(x, coords)

        assert out.shape == torch.Size([x.shape[0], x.shape[1], x.shape[2], 45, 49152])
        assert np.all(out_coords["variable"] == px.output_coords(coords)["variable"])
        handshake_dim(out_coords, "hpx", 4)
        handshake_dim(out_coords, "variable", 3)
        handshake_dim(out_coords, "lead_time", 2)
        handshake_dim(out_coords, "time", 1)
        handshake_dim(out_coords, "batch", 0)

    @pytest.mark.parametrize(
        "ensemble",
        [1, 2],
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_cbottle_video_iter(self, ensemble, device, mock_core_model, mock_sst_ds):
        time = np.array([np.datetime64("1993-04-05T00:00")])
        # Spoof model
        px = CBottleVideo(mock_core_model, mock_sst_ds).to(device)
        px.sampler_steps = 2  # Speed up sampler
        # Initialize Data Source
        dc = px.input_coords()
        del dc["batch"]
        del dc["time"]
        del dc["lead_time"]
        del dc["variable"]
        r = Random(dc)

        # Get Data and convert to tensor, coords
        lead_time = px.input_coords()["lead_time"]
        variable = px.input_coords()["variable"]
        x, coords = fetch_data(r, time, variable, lead_time, device=device)

        # Add ensemble to front
        x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
        coords.update({"ensemble": np.arange(ensemble)})
        coords.move_to_end("ensemble", last=False)

        p_iter = px.create_iterator(x, coords)

        # Get generator
        for i, (out, out_coords) in enumerate(p_iter):
            assert len(out.shape) == 6
            assert out.shape == torch.Size([ensemble, len(time), 1, 45, 721, 1440])
            assert (
                out_coords["variable"] == px.output_coords(coords)["variable"]
            ).all()
            assert (out_coords["ensemble"] == np.arange(ensemble)).all()
            assert (out_coords["time"] == time).all()
            assert out_coords["lead_time"] == np.timedelta64(6 * i, "h")
            # Single forward is 12 steps so need to test more
            if i > 16:
                break

    @pytest.mark.parametrize(
        "dc",
        [
            OrderedDict({"lat": np.random.randn(721)}),
            OrderedDict({"lat": np.random.randn(721), "phoo": np.random.randn(1440)}),
            OrderedDict({"lat": np.random.randn(721), "lon": np.random.randn(1)}),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_aurora_exceptions(self, dc, device, mock_core_model, mock_sst_ds):
        time = np.array([np.datetime64("1993-04-05T00:00")])
        px = CBottleVideo(mock_core_model, mock_sst_ds).to(device)

        # Initialize Data Source
        r = Random(dc)

        # Get Data and convert to tensor, coords
        lead_time = px.input_coords()["lead_time"]
        variable = px.input_coords()["variable"]
        x, coords = fetch_data(r, time, variable, lead_time, device=device)

        with pytest.raises((KeyError, ValueError)):
            px(x, coords)


@pytest.fixture(scope="function")
def model(model_cache_context) -> CBottleVideo:
    # Test only on cuda device
    with model_cache_context():
        package = CBottleVideo.load_default_package()
        p = CBottleVideo.load_model(package)
        return p


@pytest.mark.ci_cache
@pytest.mark.timeout(360)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aurora_package(model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package FCN
    px = model.to(device)
    px.sampler_steps = 2

    dc = px.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = px.input_coords()["lead_time"]
    variable = px.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = px(x, coords)

    assert out.shape == torch.Size([len(time), 1, 45, 721, 1440])
    assert (out_coords["variable"] == px.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
