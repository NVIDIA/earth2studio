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

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from earth2studio.models.da.healda import (
    ALL_SENSORS,
    E2S_CHANNELS,
    HealDA,
)

try:
    import cupy as cp
except ImportError:
    cp = None

# ---------- Constants ----------

NVAR = len(E2S_CHANNELS)  # 74
NPIX = 48
IN_CHANNELS = 2
TIME_LENGTH = 1
NLAT = 5
NLON = 10


# ---------- Mock neural network ----------


class PhooHealDAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = IN_CHANNELS
        self.out_channels = NVAR
        self.npix = NPIX
        self.time_length = TIME_LENGTH

    def forward(
        self,
        x,
        t,
        obs,
        float_metadata,
        pix,
        local_channel,
        local_platform,
        obs_type,
        offsets,
        second_of_day,
        day_of_year,
        class_labels,
    ):
        batch = x.shape[0]
        return torch.randn(
            batch,
            self.out_channels,
            self.time_length,
            self.npix,
            device=x.device,
        )


class MockGrid:
    def ang2pix(self, lon, lat):
        return torch.zeros(lon.shape[0], dtype=torch.long, device=lon.device)


class MockRegridder:
    def __init__(self, nlat, nlon):
        self.nlat = nlat
        self.nlon = nlon

    def __call__(self, x):
        # x: [batch, var, npix] → [batch, var, lat, lon]
        return torch.randn(
            *x.shape[:-1], self.nlat, self.nlon, dtype=x.dtype, device=x.device
        )


def _build_sensor_stats():
    return {
        "conv": {
            "means": np.zeros(8, dtype=np.float32),
            "stds": np.ones(8, dtype=np.float32),
            "raw_to_local": np.arange(9, dtype=int),
        },
        "atms": {
            "means": np.zeros(22, dtype=np.float32),
            "stds": np.ones(22, dtype=np.float32),
            "raw_to_local": np.arange(23, dtype=int),
        },
    }


def _build_model(device="cpu", lat_lon=False):
    with patch("earth2studio.models.da.healda.earth2grid") as mock_e2g:
        mock_e2g.healpix.Grid.return_value = MockGrid()
        mock_e2g.healpix.HEALPIX_PAD_XY = 0
        if lat_lon:
            mock_e2g.get_regridder.return_value = MockRegridder(NLAT, NLON)
        model = HealDA(
            model=PhooHealDAModel(),
            condition=torch.zeros(1, IN_CHANNELS, TIME_LENGTH, NPIX),
            era5_mean=torch.zeros(1, NVAR, 1, 1),
            era5_std=torch.ones(1, NVAR, 1, 1),
            sensor_stats=_build_sensor_stats(),
            lat_lon=lat_lon,
            output_resolution=(NLAT, NLON),
        )
    model._grid = MockGrid()
    return model.to(device)


def _build_raw_conv_df(n_obs=10, request_time=None):
    """Raw input format matching conv_schema (time, variable, type, elev, pres in Pa)."""
    if request_time is None:
        request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    df = pd.DataFrame(
        {
            "time": np.tile(request_time, n_obs)[:n_obs],
            "lat": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "lon": np.random.uniform(0, 360, n_obs).astype(np.float32),
            "observation": np.random.uniform(200, 300, n_obs).astype(np.float32),
            "variable": "t",
            "type": "0",
            "elev": np.full(n_obs, 100.0, dtype=np.float32),
            "pres": np.full(n_obs, 50000.0, dtype=np.float32),
        }
    )
    df.attrs = {"request_time": request_time}
    return df


def _build_raw_sat_df(n_obs=10, request_time=None, sensor="atms"):
    """Raw input format matching sat_schema."""
    if request_time is None:
        request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    t = request_time[0].astype("datetime64[ns]")
    df = pd.DataFrame(
        {
            "time": np.full(n_obs, t),
            "lat": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "lon": np.random.uniform(0, 360, n_obs).astype(np.float32),
            "observation": np.random.uniform(200, 300, n_obs).astype(np.float32),
            "variable": sensor,
            "sensor_index": np.ones(n_obs, dtype=np.uint16),
            "satellite": "n20",
            "scan_angle": np.zeros(n_obs, dtype=np.float32),
            "satellite_za": np.full(n_obs, 30.0, dtype=np.float32),
            "solza": np.full(n_obs, 45.0, dtype=np.float32),
        }
    )
    df.attrs = {"request_time": request_time}
    return df


def _mock_forward(inputs):
    return torch.randn(
        inputs["condition"].shape[0], NVAR, 1, NPIX, device=inputs["condition"].device
    )


@pytest.mark.parametrize(
    "request_time",
    [
        np.array([np.datetime64("2024-01-01T12:00:00")]),
        np.array(
            [np.datetime64("2024-01-01T12:00:00"), np.datetime64("2024-01-01T18:00:00")]
        ),
        np.array(
            [
                np.datetime64("2024-01-01T06:00:00"),
                np.datetime64("2024-01-01T12:00:00"),
                np.datetime64("2024-01-01T18:00:00"),
            ]
        ),
    ],
)
def test_build_model_inputs(request_time):
    model = _build_model()
    n_times = len(request_time)

    # Build one raw conv DF per request time with observations at that time
    parts = []
    for i, t in enumerate(request_time):
        df = _build_raw_conv_df(10 + i, request_time)
        df["time"] = t.astype("datetime64[ns]")
        parts.append(df)
    raw_conv = pd.concat(parts, ignore_index=True)
    raw_conv.attrs = {"request_time": request_time}

    obs_dict = model.filter_and_normalize(raw_conv, None, request_time)
    assert set(obs_dict.keys()) == set(ALL_SENSORS)

    inputs = model.build_input(obs_dict, request_time)
    total_obs = sum(
        len(df) for time_list in obs_dict.values() for df in time_list if df is not None
    )
    assert "obs" in inputs
    assert "float_metadata" in inputs
    assert "pix" in inputs
    assert "offsets" in inputs
    assert inputs["obs"].shape[0] == total_obs
    assert inputs["float_metadata"].shape[1] == 28
    assert inputs["offsets"].shape == (len(ALL_SENSORS), n_times, 1)
    assert inputs["second_of_day"].shape == (n_times, 1)
    assert inputs["day_of_year"].shape == (n_times, 1)
    assert inputs["condition"].shape[0] == n_times
    # Sensor monotonicity: offsets[s,:,:] <= offsets[s+1,:,:]
    for s in range(len(ALL_SENSORS) - 1):
        assert (inputs["offsets"][s, :, :] <= inputs["offsets"][s + 1, :, :]).all()
    # Time monotonicity within each sensor
    for s in range(len(ALL_SENSORS)):
        for t in range(n_times - 1):
            assert inputs["offsets"][s, t, 0] <= inputs["offsets"][s, t + 1, 0]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "request_time",
    [
        np.array([np.datetime64("2024-01-01T12:00:00")]),
        np.array(
            [np.datetime64("2024-01-01T12:00:00"), np.datetime64("2024-01-01T18:00:00")]
        ),
    ],
)
@pytest.mark.parametrize("lat_lon", [False, True])
def test_healda_call(device, request_time, lat_lon):
    model = _build_model(device=device, lat_lon=lat_lon)
    df = _build_raw_conv_df(15, request_time)

    with patch.object(model, "_forward", _mock_forward):
        out = model(df)

    n_times = len(request_time)
    assert isinstance(out, xr.DataArray)
    if lat_lon:
        assert out.dims == ("time", "variable", "lat", "lon")
        assert out.shape == (n_times, NVAR, NLAT, NLON)
    else:
        assert out.dims == ("time", "variable", "npix")
        assert out.shape == (n_times, NVAR, NPIX)
    assert np.all(out.coords["time"].values == request_time)


def test_healda_call_missing_request_time():
    model = _build_model()
    df = _build_raw_conv_df(5)
    df.attrs = {}  # No request_time → should raise
    with pytest.raises(ValueError, match="request_time"):
        model(df)


def test_healda_call_empty_obs():
    model = _build_model()
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    df = _build_raw_conv_df(5, request_time)
    df["observation"] = 999.0  # Will be filtered out

    out = model(df)
    assert isinstance(out, xr.DataArray)
    assert out.shape == (1, NVAR, NPIX)
    assert np.all(np.isnan(out.values))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_healda_independent_calls_with_optional_observations(device: str) -> None:
    model = _build_model(device=device)
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    conv_df = _build_raw_conv_df(10, request_time)
    conv_df["observation"] = 250.0
    sat_time = request_time + np.timedelta64(1, "D")
    sat_df = _build_raw_sat_df(10, sat_time)
    sat_df["observation"] = 270.0
    original_conv = conv_df.copy(deep=True)
    original_sat = sat_df.copy(deep=True)
    simultaneous_sat = sat_df.copy(deep=True)
    simultaneous_sat["time"] = request_time[0]
    simultaneous_sat.attrs = {"request_time": request_time}

    def observation_mean(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return inputs["obs"].mean().expand(1, NVAR, 1, NPIX).clone()

    with patch.object(model, "_forward", side_effect=observation_mean) as forward:
        first = model(conv_df, None)
        first_values = first.e2s.to_torch()[0].clone()
        for conv, sat, time, expected, count in [
            (None, sat_df, sat_time, 270.0, 10),
            (conv_df, simultaneous_sat, request_time, 260.0, 20),
            (conv_df, None, request_time, 250.0, 10),
        ]:
            out = model(conv, sat)
            assert out.shape == (1, NVAR, NPIX)
            np.testing.assert_array_equal(out.time, time)
            values = out.e2s.to_torch()[0]
            assert values.device == torch.device(device)
            torch.testing.assert_close(values, torch.full_like(values, expected))
            assert forward.call_args.args[0]["obs"].numel() == count

        with pytest.raises(ValueError, match="At least one"):
            model(None, None)
        torch.testing.assert_close(model(conv_df).e2s.to_torch()[0], first_values)

    torch.testing.assert_close(first_values, torch.full_like(first_values, 250.0))
    torch.testing.assert_close(first.e2s.to_torch()[0], first_values)
    pd.testing.assert_frame_equal(conv_df, original_conv)
    pd.testing.assert_frame_equal(sat_df, original_sat)
    np.testing.assert_array_equal(conv_df.attrs["request_time"], request_time)
    np.testing.assert_array_equal(sat_df.attrs["request_time"], sat_time)


def test_healda_input_coords():
    model = _build_model()
    conv_schema, sat_schema = model.input_coords()
    assert "time" in conv_schema
    assert "lat" in conv_schema
    assert "observation" in conv_schema
    assert "variable" in conv_schema
    assert "time" in sat_schema
    assert "lat" in sat_schema
    assert "observation" in sat_schema
    assert "sensor_index" in sat_schema


def test_healda_output_coords():
    model = _build_model()
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    coords = model.output_coords(*model.input_coords(), request_time=request_time)
    assert "time" in coords.coords
    assert "variable" in coords.coords
    assert "npix" in coords.coords
    assert len(coords["variable"]) == NVAR
    assert len(coords["npix"]) == NPIX


@pytest.fixture(scope="function")
def healda_model() -> HealDA:
    package = HealDA.load_default_package()
    return HealDA.load_model(package)


@pytest.mark.package
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_healda_package(device, healda_model):
    model = healda_model.to(device)
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    df = _build_raw_conv_df(20, request_time)

    out = model(df)

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("time", "variable", "npix")
    assert out.shape[1] == NVAR
    assert np.all(out.coords["time"].values == request_time)
