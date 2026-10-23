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
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

import earth2studio.models.da.sda_stormcast as stormcast_module
from earth2studio.data import HRRR, Random, RandomDataFrame, fetch_data, fetch_dataframe
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


@pytest.fixture(autouse=True)
def mock_hrrr_grid(monkeypatch, request):
    if request.node.get_closest_marker("package"):
        return
    lat, lon = np.meshgrid(
        np.linspace(30, 40, Y_END), np.linspace(250, 260, X_END), indexing="ij"
    )
    monkeypatch.setattr(HRRR, "grid", lambda: (lat, lon))


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
    with pytest.raises(ValueError):
        next(model.create_iterator(x, c))


@pytest.fixture
def stochastic_model(monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA missing")
    model = _build_model(device)

    def sample(denoiser, latents, **kwargs):
        # Exercise _forward's real latent draw and a sampler-side global draw,
        # including CPU noise when the model is on CUDA.
        return 0.01 * (latents + torch.randn_like(latents) + torch.rand(()).item())

    monkeypatch.setattr(stormcast_module, "sample", sample)
    return model


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcast_sda_iter(stochastic_model, device):
    model = stochastic_model
    time = np.array([np.datetime64("2020-04-05T00:00")])
    x = _build_input_da(model, time, device)
    x.name = "background"
    x.attrs["source"] = "mock"
    c = _build_input_da(model, time, device, conditioning=True)
    original = x.copy(deep=True)
    original_c = c.copy(deep=True)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None

    model.set_rng(42)
    y1, s1 = model.initialize(x, c)
    saved_y1 = y1.copy(deep=True)
    saved_s1 = {key: value.clone() for key, value in s1.items()}
    c1 = (c + 1).assign_coords(lead_time=y1.lead_time)
    y2, s2 = model.step(y1, c1, None, s1)
    c2 = (c + 2).assign_coords(lead_time=y2.lead_time)
    y3, s3 = model.step(y2, c2, None, state=s2)
    assert y3.lead_time.values[0] == np.timedelta64(3, "h")

    # A saved pair is a complete continuation, including sampler RNG position.
    buffer = BytesIO()
    torch.save((y1, s1), buffer)
    buffer.seek(0)
    restored_y, restored_s = torch.load(buffer, weights_only=False)
    model.set_rng(99)
    other_y, other_s = model.initialize(x, c)
    assert not torch.equal(other_y.e2s.to_torch()[0], y1.e2s.to_torch()[0])
    model.step(other_y, c1, None, other_s)
    replay_y2, replay_s2 = model.step(restored_y, c1, None, restored_s)
    replay_y3, replay_s3 = model.step(replay_y2, c2, None, replay_s2)
    for actual, expected in [(replay_y2, y2), (replay_y3, y3)]:
        torch.testing.assert_close(
            actual.e2s.to_torch()[0], expected.e2s.to_torch()[0], rtol=0, atol=0
        )
        xr.testing.assert_identical(
            actual.coords.to_dataset(), expected.coords.to_dataset()
        )
    for key in s1:
        assert torch.equal(s1[key], saved_s1[key])
        assert torch.equal(s2[key], replay_s2[key])
        assert torch.equal(s3[key], replay_s3[key])
        assert s1[key].data_ptr() != s2[key].data_ptr()

    # Call, initialization, and iteration agree under the same seed.
    model.set_rng(42)
    called = model(x, c)
    torch.testing.assert_close(
        called.e2s.to_torch()[0], y1.e2s.to_torch()[0], rtol=0, atol=0
    )
    model.set_rng(42)
    iterator = model.create_iterator(x, c)
    first = next(iterator)
    first_saved = first.copy(deep=True)
    second = iterator.send((c1, None))
    third = iterator.send((c2, None))
    iterator.close()
    for actual, expected in [
        (first, y1),
        (second, y2),
        (third, y3),
        (first, first_saved),
        (y1, saved_y1),
        (x, original),
        (c, original_c),
    ]:
        torch.testing.assert_close(
            actual.e2s.to_torch()[0], expected.e2s.to_torch()[0], rtol=0, atol=0
        )
        xr.testing.assert_identical(
            actual.coords.to_dataset(), expected.coords.to_dataset()
        )
        assert actual.attrs == expected.attrs
        assert actual.name == expected.name
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcast_sda_deterministic_seed(stochastic_model, device):
    model = stochastic_model
    time = np.array([np.datetime64("2020-04-05T00:00")])
    x = _build_input_da(model, time, device)
    c = _build_input_da(model, time, device, conditioning=True)
    model.set_rng(42, reset=False)
    first = model(x, c)
    model.set_rng(99, reset=False)
    second = model(x, c)
    assert not torch.equal(first.e2s.to_torch()[0], second.e2s.to_torch()[0])
    model.set_rng(42)
    torch.testing.assert_close(
        model(x, c).e2s.to_torch()[0], first.e2s.to_torch()[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        model(x, c).e2s.to_torch()[0], second.e2s.to_torch()[0], rtol=0, atol=0
    )


def test_stormcast_sda_iter_observations():
    model = _build_model()
    time = np.array([np.datetime64("2020-04-05T00:00")])
    x = _build_input_da(model, time)
    c = _build_input_da(model, time, conditioning=True)
    obs = pd.DataFrame(
        {
            "time": [time[0] + np.timedelta64(i, "h") for i in (1, 2)],
            "lat": [model.lat[10, 10]] * 2,
            "lon": [model.lon[10, 10]] * 2,
            "variable": [model.variables[0]] * 2,
            "observation": [5.0, 9.0],
        }
    )
    original = obs.copy(deep=True)

    def forward(x, conditioning, y_obs, mask):
        return x + conditioning[:, :NVAR] + mask * y_obs

    with patch.object(model, "_forward", forward):
        y1, state = model.initialize(x, c, obs)
        expected = x.values + c.values[:, :, :NVAR]
        expected[0, 0, 0, 10, 10] += 5.0
        np.testing.assert_allclose(y1.values, expected)
        c1 = c.assign_coords(lead_time=y1.lead_time)
        y2, _ = model.step(y1, c1, obs, state)
        expected = y1.values + c.values[:, :, :NVAR]
        expected[0, 0, 0, 10, 10] += 9.0
        np.testing.assert_allclose(y2.values, expected)
        without_obs, _ = model.step(y1, c1, None, state)
        empty_obs, _ = model.step(y1, c1, obs.iloc[:0], state)
        np.testing.assert_allclose(
            without_obs.values, y1.values + c.values[:, :, :NVAR]
        )
        xr.testing.assert_identical(empty_obs, without_obs)
        xr.testing.assert_identical(model(x, c, obs), y1)
        iterator = model.create_iterator(x, c, obs)
        xr.testing.assert_identical(next(iterator), y1)
        xr.testing.assert_identical(iterator.send((c1, obs)), y2)
        iterator.close()
    pd.testing.assert_frame_equal(obs, original)


@pytest.mark.parametrize("forcing", [None, (None,), (None, None)])
def test_stormcast_sda_iter_exceptions(forcing):
    model = _build_model()
    time = np.array([np.datetime64("2020-04-05T00:00")])
    x = _build_input_da(model, time)
    c = _build_input_da(model, time, conditioning=True)
    with patch.object(model, "_forward", _mock_forward):
        iterator = model.create_iterator(x, c)
        next(iterator)
        with pytest.raises(ValueError, match="forcing_coords|conditioning"):
            iterator.send(forcing)


def test_stormcast_sda_step_exceptions():
    model = _build_model()
    time = np.array([np.datetime64("2020-04-05T00:00")])
    x = _build_input_da(model, time)
    c = _build_input_da(model, time, conditioning=True)
    with pytest.raises(ValueError, match="conditioning"):
        model.initialize(x, None)
    with patch.object(model, "_forward", _mock_forward):
        y, state = model.initialize(x, c)
    with pytest.raises(ValueError, match="conditioning"):
        model.step(y, None, None, state)
    with pytest.raises(ValueError):
        model.step(y, c, None, state)  # stale conditioning
    with pytest.raises(ValueError, match="RNG snapshots"):
        model.step(y, c.assign_coords(lead_time=y.lead_time), None, {})


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcast_sda_rng_exception_isolation(stochastic_model, device):
    model = stochastic_model
    time = np.array([np.datetime64("2020-04-05T00:00")])
    x = _build_input_da(model, time, device)
    c = _build_input_da(model, time, device, conditioning=True)
    model.set_rng(42)
    y, state = model.initialize(x, c)
    saved_state = {key: value.clone() for key, value in state.items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None

    def fail(denoiser, latents, **kwargs):
        torch.randn_like(latents)
        torch.rand(())
        raise RuntimeError("sampler failed")

    with patch.object(stormcast_module, "sample", fail):
        with pytest.raises(RuntimeError, match="sampler failed"):
            model.step(y, c.assign_coords(lead_time=y.lead_time), None, state)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    for key in state:
        assert torch.equal(state[key], saved_state[key])


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
