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
    _compute_unified_metadata,
    _fourier_features,
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


def _build_model(device="cpu"):
    with patch("earth2studio.models.da.healda.earth2grid") as mock_e2g:
        mock_e2g.healpix.Grid.return_value = MockGrid()
        mock_e2g.healpix.HEALPIX_PAD_XY = 0
        model = HealDA(
            model=PhooHealDAModel(),
            condition=torch.zeros(1, IN_CHANNELS, TIME_LENGTH, NPIX),
            era5_mean=torch.zeros(1, NVAR, 1, 1),
            era5_std=torch.ones(1, NVAR, 1, 1),
            sensor_stats=_build_sensor_stats(),
        )
    model._grid = MockGrid()
    return model.to(device)


def _build_conv_obs_df(n_obs=10, request_time=None):
    """Internal (post-prep_conv) format for testing _filter_and_normalize etc."""
    if request_time is None:
        request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    t_ns = request_time[0].astype("datetime64[ns]").view(np.int64)
    return pd.DataFrame(
        {
            "lat": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "lon": np.random.uniform(0, 360, n_obs).astype(np.float32),
            "obs_time_ns": np.full(n_obs, t_ns, dtype=np.int64),
            "observation": np.random.uniform(200, 300, n_obs).astype(np.float64),
            "local_channel": np.full(n_obs, 5, dtype=np.int32),  # "t" channel
            "local_platform": np.zeros(n_obs, dtype=np.int64),
            "sensor": "conv",
            "obs_type": np.zeros(n_obs, dtype=np.int32),
            "height": np.full(n_obs, 100.0, dtype=np.float32),
            "pressure": np.full(n_obs, 500.0, dtype=np.float32),
            "scan_angle": np.full(n_obs, np.nan, dtype=np.float32),
            "sat_zenith_angle": np.full(n_obs, np.nan, dtype=np.float32),
            "sol_zenith_angle": np.full(n_obs, np.nan, dtype=np.float32),
        }
    )


def _build_raw_conv_df(n_obs=10, request_time=None):
    """Raw input format matching conv_schema (time, variable, type, elev, pres)."""
    if request_time is None:
        request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    t = request_time[0].astype("datetime64[ns]")
    df = pd.DataFrame(
        {
            "time": np.full(n_obs, t),
            "lat": np.random.uniform(-90, 90, n_obs).astype(np.float32),
            "lon": np.random.uniform(0, 360, n_obs).astype(np.float32),
            "observation": np.random.uniform(200, 300, n_obs).astype(np.float32),
            "variable": "t",
            "type": "0",
            "elev": np.full(n_obs, 100.0, dtype=np.float32),
            "pres": np.full(n_obs, 500.0, dtype=np.float32),
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
            "channel_index": np.ones(n_obs, dtype=np.uint16),
            "satellite": "n20",
            "scan_angle": np.zeros(n_obs, dtype=np.float32),
            "satellite_za": np.full(n_obs, 30.0, dtype=np.float32),
            "solza": np.full(n_obs, 45.0, dtype=np.float32),
        }
    )
    df.attrs = {"request_time": request_time}
    return df


def _mock_forward(inputs, device):
    return torch.randn(1, NVAR, TIME_LENGTH, NPIX, device=device)


def test_fourier_features():
    x = torch.tensor([0.0, 0.5, 1.0])
    out = _fourier_features(x, 3)
    assert out.shape == (3, 6)  # 3 freqs * 2 (sin + cos)


def test_compute_unified_metadata():
    n = 5
    device = torch.device("cpu")
    target = torch.full((n,), 1704067200, dtype=torch.int64, device=device)
    lon = torch.rand(n, device=device) * 360
    time_ns = torch.full((n,), 1704067200 * 10**9, dtype=torch.int64, device=device)
    height = torch.tensor([100, 200, float("nan"), 500, float("nan")], device=device)
    pressure = torch.tensor([1000, float("nan"), 500, 300, float("nan")], device=device)
    scan = torch.full((n,), float("nan"), device=device)
    sat_zen = torch.full((n,), float("nan"), device=device)
    sol_zen = torch.full((n,), float("nan"), device=device)

    out = _compute_unified_metadata(
        target,
        lon=lon,
        time=time_ns,
        height=height,
        pressure=pressure,
        scan_angle=scan,
        sat_zenith_angle=sat_zen,
        sol_zenith_angle=sol_zen,
    )
    assert out.shape == (n, 28)
    assert torch.isfinite(out).all()


def test_build_channel_stats():
    model = _build_model()
    stats = model._channel_stats
    assert "sensor" in stats.columns
    assert "local_channel" in stats.columns
    assert "mean" in stats.columns
    assert "std" in stats.columns
    # conv has 8 channels, atms has 22
    assert len(stats) == 30


def test_filter_and_normalize():
    model = _build_model()
    df = _build_conv_obs_df(20)
    result = model._filter_and_normalize(df)
    assert len(result) > 0
    assert "mean" not in result.columns
    assert "std" not in result.columns
    # Observations should be z-score normalized (not raw values)
    assert result["observation"].dtype == np.float32


def test_filter_and_normalize_empty():
    model = _build_model()
    # All observations outside valid range for "t" channel (150-350)
    df = _build_conv_obs_df(5)
    df["observation"] = 999.0
    result = model._filter_and_normalize(df)
    assert len(result) == 0


def test_build_model_inputs():
    model = _build_model()
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    df = _build_conv_obs_df(10, request_time)
    filtered = model._filter_and_normalize(df)

    sensor_order = pd.CategoricalDtype(categories=ALL_SENSORS, ordered=True)
    filtered["sensor"] = filtered["sensor"].astype(sensor_order)
    filtered = filtered.sort_values("sensor", kind="stable").reset_index(drop=True)

    inputs = model._build_model_inputs(
        filtered, pd.Timestamp("2024-01-01T12:00:00"), torch.device("cpu")
    )
    assert "obs" in inputs
    assert "float_metadata" in inputs
    assert "pix" in inputs
    assert "offsets" in inputs
    assert inputs["float_metadata"].shape[1] == 28
    assert inputs["offsets"].shape[0] == len(ALL_SENSORS)


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
def test_healda_call(device):
    model = _build_model(device=device)
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    df = _build_raw_conv_df(15, request_time)

    with patch.object(model, "_forward", _mock_forward):
        out = model(df)

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("time", "variable", "npix")
    assert out.shape == (1, NVAR, NPIX)
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


def test_healda_generator():
    model = _build_model()
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    conv_df = _build_raw_conv_df(10, request_time)
    sat_df = _build_raw_sat_df(10, request_time)

    gen = model.create_generator()
    result = gen.send(None)
    assert result is None

    # conv only
    with patch.object(model, "_forward", _mock_forward):
        da = gen.send((conv_df, None))
    assert isinstance(da, xr.DataArray)
    assert da.shape == (1, NVAR, NPIX)

    # sat only
    with patch.object(model, "_forward", _mock_forward):
        da = gen.send((None, sat_df))
    assert isinstance(da, xr.DataArray)
    assert da.shape == (1, NVAR, NPIX)

    # both
    with patch.object(model, "_forward", _mock_forward):
        da = gen.send((conv_df, sat_df))
    assert isinstance(da, xr.DataArray)
    assert da.shape == (1, NVAR, NPIX)

    with pytest.raises(ValueError, match="At least one"):
        gen.send((None, None))

    gen.close()


def test_healda_init_coords():
    model = _build_model()
    assert model.init_coords() is None


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
    assert "channel_index" in sat_schema


def test_healda_output_coords():
    model = _build_model()
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    (coords,) = model.output_coords(model.input_coords(), request_time=request_time)
    assert "time" in coords
    assert "variable" in coords
    assert "npix" in coords
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
