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
    cbottle = None

from earth2studio.models.conformance import (
    ContractException,
    check_prognostic_contract,
)
from earth2studio.models.px import CBottleVideo
from earth2studio.utils import handshake_dim
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch


@pytest.fixture(autouse=True)
def offline_video(monkeypatch):
    if cbottle is not None:
        return

    def initialize(
        self, core, sst, lat_lon=True, dataset_modality=1, seed=None, **kwargs
    ):
        torch.nn.Module.__init__(self)
        self.sst, self.lat_lon, self.seed = sst, lat_lon, seed
        self.dataset_modality = dataset_modality
        self._time_length = 12
        self._time_step = np.timedelta64(6, "h")
        self.register_buffer("device_buffer", torch.empty(0))

    def forward(self, x, times):
        gen = (
            torch.Generator(device=x.device).manual_seed(self.seed)
            if self.seed is not None
            else None
        )
        noise = torch.rand((), device=x.device, generator=gen)
        return torch.nan_to_num(x).expand(-1, 12, *x.shape[2:]) + noise

    monkeypatch.setattr(CBottleVideo, "__init__", initialize)
    monkeypatch.setattr(CBottleVideo, "_forward", forward)


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
    if cbottle is None:
        return torch.nn.Identity()
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
        "x,time,dataset_modality",
        [
            (
                torch.zeros(1, 1, 1, 45, 721, 1440),
                np.array([datetime(2020, 1, 1)], dtype=np.datetime64),
                1,
            ),
            (
                torch.full((1, 2, 1, 45, 721, 1440), fill_value=torch.nan),
                np.array(
                    [datetime(2000, 1, 2, 3, 4, 5), datetime(1980, 8, 1)],
                    dtype=np.datetime64,
                ),
                0,
            ),
            (
                torch.zeros(2, 2, 1, 45, 721, 1440),
                np.array(
                    [datetime(2000, 1, 2, 3, 4, 5), datetime(1980, 8, 1)],
                    dtype=np.datetime64,
                ),
                0,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "device", ["cuda:0"]
    )  # , "cpu" takes too long, it should work but skipping
    def test_cbottle_video_forward(
        self, x, time, dataset_modality, device, mock_core_model, mock_sst_ds
    ):
        px = CBottleVideo(
            mock_core_model, mock_sst_ds, dataset_modality=dataset_modality
        ).to(device)
        px.sampler_steps = 2  # Speed up sampler

        coords = coord_array_like(
            px.input_coords(), {"batch": np.arange(x.shape[0]), "time": time}
        )
        coords.attrs["user"] = {"notes": ["input"]}
        coords.encoding["user"] = {"notes": ["input"]}
        coords.time.attrs["user"] = {"notes": ["input"]}
        planned = px.output_coords(coords)
        assert planned.data.nbytes == 0
        planned.attrs["user"]["notes"].append("output")
        planned.encoding["user"]["notes"].append("output")
        planned.time.attrs["user"]["notes"].append("output")
        assert coords.attrs["user"]["notes"] == ["input"]
        assert coords.encoding["user"]["notes"] == ["input"]
        assert coords.time.attrs["user"]["notes"] == ["input"]

        x = x.to(device)
        out = px(from_torch(x, coords))
        out_coords = {k: out.coords[k].values for k in out.dims}

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

        coords = coord_array_like(
            px.input_coords(), {"batch": np.arange(x.shape[0]), "time": time}
        )

        x = x.to(device)
        out = px(from_torch(x, coords))
        out_coords = {k: out.coords[k].values for k in out.dims}

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
        coords = coord_array_like(
            px.input_coords(), {"batch": np.arange(ensemble), "time": time}
        ).rename(batch="ensemble")
        x = from_torch(torch.zeros(coords.shape), coords)
        x.name = "conditioning"
        x.attrs["user"] = {"history": ["initial"]}
        x.encoding["user"] = {"history": ["initial"]}
        x = x.assign_coords(marker=1)
        original = x.copy(deep=True)
        calls = []
        px.front_hook = lambda a: calls.append(a.dims) or a

        def rear(a):
            a.attrs["user"]["history"].append("forecast")
            a.encoding["user"]["history"].append("forecast")
            return a.drop_vars("marker") if "marker" in a.coords else a

        px.rear_hook = rear
        p_iter = px.create_iterator(x)
        initial = next(p_iter)
        xr.testing.assert_identical(initial, original)
        assert calls == []

        # Get generator
        for i, out in enumerate(p_iter, 1):
            out_coords = {k: v.values for k, v in out.coords.items()}
            assert len(out.shape) == 6
            assert out.shape == torch.Size([ensemble, len(time), 1, 45, 721, 1440])
            assert (
                out_coords["variable"] == px.output_coords(coords)["variable"]
            ).all()
            assert (out_coords["ensemble"] == np.arange(ensemble)).all()
            assert (out_coords["time"] == time).all()
            assert out_coords["lead_time"] == np.timedelta64(6 * i, "h")
            assert "marker" not in out.coords
            assert len(out.attrs["user"]["history"]) == i + 1
            assert len(out.encoding["user"]["history"]) == i + 1
            if i == 1:
                retained = out
                retained_copy = out.copy(deep=True)
            # Single forward is 12 steps so need to test more
            if i > 16:
                break
        xr.testing.assert_identical(x, original)
        xr.testing.assert_identical(initial, original)
        xr.testing.assert_identical(retained, retained_copy)
        assert retained.encoding == retained_copy.encoding
        assert calls == [x.dims, x.dims]

    @pytest.mark.parametrize(
        "dc",
        [
            OrderedDict({"lat": np.random.randn(721)}),
            OrderedDict({"lat": np.random.randn(721), "phoo": np.random.randn(1440)}),
            OrderedDict({"lat": np.random.randn(721), "lon": np.random.randn(1)}),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_video_exceptions(self, dc, device, mock_core_model, mock_sst_ds):
        time = np.array([np.datetime64("1993-04-05T00:00")])
        px = CBottleVideo(mock_core_model, mock_sst_ds).to(device)

        # Initialize Data Source

        # Get Data and convert to tensor, coords
        lead_time = px.input_coords()["lead_time"]
        variable = px.input_coords()["variable"]
        coords = OrderedDict(
            time=time, lead_time=lead_time.values, variable=variable.values, **dc
        )
        x = xr.DataArray(
            np.zeros(tuple(len(v) for v in coords.values()), dtype=np.float32),
            dims=tuple(coords),
            coords=coords,
        )

        with pytest.raises((KeyError, ValueError)):
            px(x)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_video_conformance(self, device, mock_core_model, mock_sst_ds):
        """Check the mock CBottleVideo model against the Earth2Studio model contract.

        CBottleVideo does not currently declare `stochastic` or implement
        `set_rng()` (see dev/spec/MODEL_CONTRACT_SPEC.md's Migration table: it
        already passes a seed straight to the core model's sample() call, so only
        the declaration and set_rng() entry point are missing). Until that lands
        the checker takes the undeclared stochasticity at face value: two
        rollouts from one input disagree while the model declares
        stochastic=False, which is P13. Pinned here until the wrapper declares
        stochastic and implements set_rng().
        """
        px = CBottleVideo(mock_core_model, mock_sst_ds).to(device)
        px.sampler_steps = 2  # Speed up sampler
        with pytest.raises(ContractException) as exc_info:
            check_prognostic_contract(px, nsteps=1, device=device)
        assert exc_info.value.violations == [
            "P13: repeated runs with the same input and seed disagree"
        ]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_cbottle_video_package(device):
    package = CBottleVideo.load_default_package()
    model = CBottleVideo.load_model(package)

    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package FCN
    px = model.to(device)
    px.sampler_steps = 2

    coords = coord_array_like(px.input_coords(), {"batch": [0], "time": time}).isel(
        batch=0, drop=True
    )
    x = from_torch(torch.randn(coords.shape, device=device), coords)
    out = px(x)
    out_coords = {k: out.coords[k].values for k in out.dims}

    assert out.shape == torch.Size([len(time), 1, 45, 721, 1440])
    assert (out_coords["variable"] == px.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
