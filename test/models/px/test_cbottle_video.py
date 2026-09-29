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
    check_prognostic_contract,
)
from earth2studio.models.px import CBottleVideo
from earth2studio.models.rng import seeded
from earth2studio.utils import handshake_dim
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


@pytest.fixture(autouse=True)
def offline_video(monkeypatch):
    if cbottle is not None:
        return

    def initialize(self, core, sst, lat_lon=True, dataset_modality=1, **kwargs):
        torch.nn.Module.__init__(self)
        self.sst, self.lat_lon = sst, lat_lon
        self.dataset_modality = dataset_modality
        self._time_length = 12
        self._time_step = np.timedelta64(6, "h")
        self.register_buffer("device_buffer", torch.empty(0))

    @seeded
    def forward(self, x, times):
        noise = torch.rand((), device=x.device)
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
                torch.zeros(1, 2, 1, 45, 721, 1440),
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
        assert planned.attrs["user"] == coords.attrs["user"]
        assert planned.time.attrs["user"] == coords.time.attrs["user"]
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
        px = CBottleVideo(mock_core_model, mock_sst_ds).to(device)
        px.sampler_steps = 2  # Speed up sampler
        check_prognostic_contract(px, nsteps=1, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cbottle_video_selects_frame_before_unbatch(monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    class SmallVideo(CBottleVideo):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self._time_length = 12
            self._time_step = np.timedelta64(6, "h")
            self.register_buffer("device_buffer", torch.empty(0))
            self.calls = 0

        def input_coords(self):
            return coord_array(
                ("batch", "time", "lead_time", "variable", "hpx"),
                {
                    "lead_time": [np.timedelta64(0, "h")],
                    "variable": ["t2m", "u10m"],
                    "hpx": [0, 1, 2],
                },
                dynamic=("batch", "time"),
            )

        def _forward(self, x, times):
            self.calls += 1
            # Match the real core's non-contiguous [batch, lead, variable, hpx] layout.
            frames = torch.arange(12, device=x.device, dtype=x.dtype)
            return (x.transpose(1, 2) + frames[None, None, :, None]).transpose(1, 2)

    model = SmallVideo().to(device)
    coords = coord_array_like(
        model.input_coords(),
        {
            "batch": [10, 20],
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        },
    ).rename(batch="ensemble")
    x = from_torch(
        torch.arange(24, dtype=torch.float64, device=device).reshape(coords.shape),
        coords,
    )
    x.name = "conditioning"
    x.attrs["user"] = {"notes": ["input"]}
    x.encoding["user"] = {"notes": ["input"]}
    x.time.attrs["user"] = {"notes": ["time"]}
    original = x.copy(deep=True)
    expected = model._advance(x).isel(lead_time=slice(0, 1)).copy(deep=True)
    model.calls = 0

    unbatched_frames = []
    accessor = type(x.e2s)
    unbatch = accessor.unbatch

    def record_unbatch(self, *args, **kwargs):
        result = unbatch(self, *args, **kwargs)
        unbatched_frames.append(result.sizes["lead_time"])
        return result

    monkeypatch.setattr(accessor, "unbatch", record_unbatch)
    out = model(x)
    assert unbatched_frames == [1]
    assert model.calls == 1
    xr.testing.assert_identical(out.e2s.as_numpy(), expected.e2s.as_numpy())
    assert out.encoding == expected.encoding

    unbatched_frames.clear()
    iterator = model.create_iterator(x)
    next(iterator)
    for step in range(1, 13):
        frame = next(iterator)
        np.testing.assert_array_equal(
            frame.e2s.as_numpy().values, original.e2s.as_numpy().values + step
        )
        assert frame.lead_time.values == np.timedelta64(6 * step, "h")
    iterator.close()
    assert unbatched_frames == [11, 11]
    assert model.calls == 3
    out.data[...] = -1
    out.attrs["user"]["notes"].append("changed")
    xr.testing.assert_identical(x.e2s.as_numpy(), original.e2s.as_numpy())
    assert x.encoding == original.encoding


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
