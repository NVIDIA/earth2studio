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
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from earth2studio.data import Random, RandomDataFrame, fetch_data, fetch_dataframe
from earth2studio.models.da.sda_stormcast import StormCastSDA

try:
    import cupy as cp
except ImportError:
    cp = None


# ---------- Mock neural networks ----------


class PhooRegressionModel(torch.nn.Module):
    def __init__(self, out_vars=3):
        super().__init__()
        self.out_vars = out_vars

    def forward(self, x):
        return x[:, : self.out_vars, :, :]


class PhooSDADiffusionModel(torch.nn.Module):
    def forward(self, x, t, condition=None):
        return x


# ---------- Constants ----------

Y_START, Y_END = 32, 128
X_START, X_END = 32, 64
NVAR = 3
NVAR_COND = 5
SAMPLER_ARGS = {
    "num_steps": 2,
    "sigma_min": 0.002,
    "sigma_max": 88.0,
    "rho": 7.0,
    "S_churn": 0.0,
    "S_min": 0.0,
    "S_max": float("inf"),
    "S_noise": 1.0,
}


# ---------- Helpers ----------


def _build_model(device="cpu"):
    regression = PhooRegressionModel(out_vars=NVAR)
    diffusion = PhooSDADiffusionModel()

    ny = Y_END - Y_START
    nx = X_END - X_START
    variables = np.array(["u%02d" % i for i in range(NVAR)])
    means = torch.zeros(1, NVAR, 1, 1)
    stds = torch.ones(1, NVAR, 1, 1)
    invariants = torch.randn(1, 2, ny, nx)
    conditioning_means = torch.randn(1, NVAR_COND, 1, 1)
    conditioning_stds = torch.randn(1, NVAR_COND, 1, 1).abs() + 0.1
    conditioning_variables = np.array(["c%02d" % i for i in range(NVAR_COND)])

    return StormCastSDA(
        regression,
        diffusion,
        means,
        stds,
        invariants,
        hrrr_lat_lim=(Y_START, Y_END),
        hrrr_lon_lim=(X_START, X_END),
        variables=variables,
        conditioning_means=conditioning_means,
        conditioning_stds=conditioning_stds,
        conditioning_variables=conditioning_variables,
        sampler_args=SAMPLER_ARGS,
    ).to(device)


def _build_input_da(model, time, device="cpu", conditioning=False):
    dc = OrderedDict([("hrrr_y", model.hrrr_y), ("hrrr_x", model.hrrr_x)])
    r = Random(dc)
    x = fetch_data(
        r,
        time,
        model.conditioning_variables if conditioning else model.variables,
        lead_time=np.array([np.timedelta64(0, "h")]),
        device=device,
    )
    return x.assign_coords(
        lat=(["hrrr_y", "hrrr_x"], model.lat),
        lon=(["hrrr_y", "hrrr_x"], model.lon),
    )


def _build_obs_source(model, n_obs=10):
    grid_lat, grid_lon = model.lat, model.lon
    _state: dict = {}

    def lat_gen():
        y = np.random.randint(5, grid_lat.shape[0] - 5)
        x = np.random.randint(5, grid_lat.shape[1] - 5)
        _state["y"], _state["x"] = y, x
        return float(grid_lat[y, x])

    def lon_gen():
        return float(grid_lon[_state["y"], _state["x"]])

    return RandomDataFrame(
        n_obs=n_obs,
        field_generators={"lat": lat_gen, "lon": lon_gen},
    )


def _mock_forward(x, conditioning, y_obs, mask):
    return torch.zeros_like(x)


# ---------- Unit tests: _points_in_polygon ----------


def test_points_in_polygon_square():
    polygon = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float64)
    inside = np.array([[0.5, 0.5], [0.1, 0.1], [0.9, 0.9]], dtype=np.float64)
    outside = np.array([[-1, -1], [2, 2], [0.5, 1.5]], dtype=np.float64)
    points = np.vstack([inside, outside])

    result = StormCastSDA._points_in_polygon(points, polygon)

    assert result[:3].all()
    assert not result[3:].any()


def test_points_in_polygon_triangle():
    polygon = np.array([[0, 0], [2, 0], [1, 2]], dtype=np.float64)
    inside = np.array([[1, 0.5]], dtype=np.float64)
    outside = np.array([[3, 3], [-1, 0]], dtype=np.float64)
    points = np.vstack([inside, outside])

    result = StormCastSDA._points_in_polygon(points, polygon)

    assert result[0]
    assert not result[1:].any()


# ---------- Unit tests: _build_obs_tensors ----------


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
def test_build_obs_tensors(device):
    model = _build_model(device=device)
    time = np.array([np.datetime64("2020-01-01T00:00")])
    obs_source = _build_obs_source(model, n_obs=5)
    obs_df = fetch_dataframe(obs_source, time, model.variables[:2])
    ny, nx = model.lat.shape

    y_obs, mask = model._build_obs_tensors(obs_df, time[0], model.device)

    assert y_obs.shape == (1, NVAR, ny, nx)
    assert mask.shape == (1, NVAR, ny, nx)
    assert mask.sum() > 0
    assert y_obs.device == model.device
    assert mask.device == model.device


def test_build_obs_tensors_none():
    model = _build_model()
    time = np.array([np.datetime64("2020-01-01T00:00")])
    ny, nx = model.lat.shape

    y_obs, mask = model._build_obs_tensors(None, time[0], model.device)

    assert y_obs.shape == (1, NVAR, ny, nx)
    assert (mask == 0).all()
    assert (y_obs == 0).all()


def test_build_obs_tensors_outside_grid():
    model = _build_model()
    time = np.array([np.datetime64("2020-01-01T00:00")])

    # RandomDataFrame with lat/lon fixed far outside the HRRR grid
    obs_source = RandomDataFrame(
        n_obs=5,
        field_generators={
            "lat": lambda: 0.0,
            "lon": lambda: 10.0,
        },
    )
    obs_df = fetch_dataframe(obs_source, time, [str(model.variables[0])])

    y_obs, mask = model._build_obs_tensors(obs_df, time[0], model.device)

    assert (mask == 0).all()


def test_build_obs_tensors_averages_duplicates():
    model = _build_model()
    time = np.array([np.datetime64("2020-01-01T00:00")])
    ny, nx = model.lat.shape

    mid_y, mid_x = ny // 2, nx // 2
    pt_lat = float(model.lat[mid_y, mid_x])
    pt_lon = float(model.lon[mid_y, mid_x])
    var_name = str(model.variables[0])

    # Three observations at the exact same location and variable
    obs_df = pd.DataFrame(
        {
            "time": pd.to_datetime([time[0]] * 3),
            "lat": [pt_lat] * 3,
            "lon": [pt_lon] * 3,
            "variable": [var_name] * 3,
            "observation": [3.0, 9.0, 30.0],
        }
    )

    y_obs, mask = model._build_obs_tensors(obs_df, time[0], model.device)

    assert mask.sum() == 1
    assert torch.isclose(y_obs[mask == 1], torch.tensor(14.0)).all()

    obs_df = pd.DataFrame(
        {
            "time": pd.to_datetime([time[0]] * 3),
            "lat": [0, pt_lat, pt_lat],
            "lon": [0, pt_lon, pt_lon],
            "variable": [var_name] * 3,
            "observation": [3.0, 9.0, 30.0],
        }
    )

    y_obs, mask = model._build_obs_tensors(obs_df, time[0], model.device)
    assert mask.sum() == 1
    assert torch.isclose(y_obs[mask == 1], torch.tensor(19.5)).all()


# ---------- Test: __call__ ----------


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2020-04-05T00:00")]),
        np.array(
            [np.datetime64("2020-10-11T12:00"), np.datetime64("2020-06-04T00:00")]
        ),
    ],
)
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
def test_stormcast_sda_call(time, device):
    model = _build_model(device=device)
    state = model.input_coords()
    conditioning, obs = model.forcing_coords()
    assert isinstance(state, xr.DataArray)
    assert conditioning.dims == state.dims
    np.testing.assert_array_equal(
        conditioning["variable"], model.conditioning_variables
    )
    np.testing.assert_array_equal(obs["variable"], model.variables)
    assert model.output_coords(state).lead_time.values[0] == np.timedelta64(1, "h")
    x = _build_input_da(model, time, device=device)
    original, _ = x.e2s.to_torch()
    original = original.clone()
    c = _build_input_da(model, time, device=device, conditioning=True)
    obs_source = _build_obs_source(model, n_obs=10)
    obs_df = fetch_dataframe(obs_source, time, model.variables[:2])
    ny, nx = model.lat.shape

    with patch.object(model, "_forward", _mock_forward):
        out = model(x, c, obs_df)

    assert out.shape == (len(time), 1, NVAR, ny, nx)
    assert set(out.dims) == {"time", "lead_time", "variable", "hrrr_y", "hrrr_x"}
    assert np.all(out.coords["time"].values == time)
    assert out.coords["lead_time"].values[0] == np.timedelta64(1, "h")

    # Without observations
    with patch.object(model, "_forward", _mock_forward):
        out_none = model(x, c, None)

    assert out_none.shape == out.shape
    torch.testing.assert_close(x.e2s.to_torch()[0], original)
    assert out.e2s.to_torch()[0].device == model.device


# ---------- Test: create_generator ----------


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
def test_stormcast_sda_generator(device):
    model = _build_model(device=device)
    time = np.array([np.datetime64("2020-04-05T00:00")])
    x = _build_input_da(model, time, device=device)
    c = _build_input_da(model, time, device=device, conditioning=True)
    obs_source = _build_obs_source(model, n_obs=10)
    obs_df = fetch_dataframe(obs_source, time, model.variables[:2])
    ny, nx = model.lat.shape

    seen = []

    def forward(x, conditioning, y_obs, mask):
        seen.append(conditioning.clone())
        return torch.zeros_like(x)

    with patch.object(model, "_forward", forward):
        gen = model.create_generator(x)

        # Prime without advancing the model or applying forcing.
        state = next(gen)
        assert state is x
        assert not seen
        state = gen.send((c, obs_df))
        assert state.shape == (len(time), 1, NVAR, ny, nx)
        assert state.coords["lead_time"].values[0] == np.timedelta64(1, "h")
        torch.testing.assert_close(seen[0], c.e2s.to_torch()[0][0])

        # Update conditioning to the current state window; observations are optional.
        c = (c + 1).assign_coords(lead_time=state.lead_time)
        state = gen.send((c, None))
        assert state.shape == (len(time), 1, NVAR, ny, nx)
        assert state.coords["lead_time"].values[0] == np.timedelta64(2, "h")
        torch.testing.assert_close(seen[1], c.e2s.to_torch()[0][0])

        gen.close()


# ---------- Test: exceptions ----------


@pytest.mark.parametrize("coord", ["time", "lead_time", "variable", "lat", "lon"])
def test_stormcast_sda_exceptions(coord):
    model = _build_model()
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x = _build_input_da(model, time)
    c = _build_input_da(model, time, conditioning=True)
    values = c[coord].values.copy()
    if coord == "variable":
        values[0] = "bad"
    elif coord in ("time", "lead_time"):
        values += np.timedelta64(1, "h")
    else:
        values += 1
    c = c.assign_coords({coord: (c[coord].dims, values)})
    with pytest.raises(ValueError):
        model(x, c)
    gen = model.create_generator(x)
    next(gen)
    with pytest.raises(ValueError, match="forcing_coords"):
        gen.send(None)


# ---------- Test: package loading ----------


@pytest.fixture(scope="function")
def sda_model() -> StormCastSDA:
    package = StormCastSDA.load_default_package()
    return StormCastSDA.load_model(package)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_stormcast_sda_package(device, sda_model):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2020-04-05T00:00")])

    model = sda_model.to(device)

    model.sampler_args = SAMPLER_ARGS
    x = _build_input_da(model, time, device=device)
    c = _build_input_da(model, time, device=device, conditioning=True)
    out = model(x, c, None)

    assert out.shape == (1, 1, 99, 512, 640)
    assert set(out.dims) == {"time", "lead_time", "variable", "hrrr_y", "hrrr_x"}
    assert np.all(out.coords["time"].values == time)
    assert out.coords["lead_time"].values[0] == np.timedelta64(1, "h")
