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

"""Tests for the StormCastEurope prognostic model."""

import json
import os
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.px.stormcasteurope import (
    StormCastEurope,
    _interp_levels_to_height,
)

H, W = 4, 6
STATE = ["T_2M", "TOT_PRECIP", "CLCT"]  # checkpoint COSMO channel names
CANONICAL = ["t2m", "tp", "tcc"]  # canonical Earth2Studio names
TRANSFORMS = {
    "TOT_PRECIP": {"transform": "log_eps", "eps": 1e-4},
    "CLCT": {"transform": "logit_eps_percent", "eps": 1e-3, "scale": 100},
}
# Shared hub-height wind config: 5-channel state with 10 m and 50 m wind levels.
WIND_STATE = ["T_2M", "U_L40", "V_L40", "U_L39", "V_L39"]
WIND_LEVELS = {
    "elevation_invariant": "elevation_norm",
    "levels": [
        {"u": "U_L40", "v": "V_L40", "a": 10.0, "b": 0.0},
        {"u": "U_L39", "v": "V_L39", "a": 50.0, "b": 0.0},
    ],
}


class MockERA5:
    """Regular-grid ERA5 source covering the test input grid."""

    def __init__(self) -> None:
        self.lat = np.linspace(40.0, 60.0, 21)
        self.lon = np.linspace(0.0, 20.0, 21)
        self.calls: list = []  # (requested times, requested variables) per call

    def __call__(self, time, variable):  # noqa: ANN001
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        self.calls.append((time.copy(), list(variable)))
        data = np.zeros(
            (len(time), len(variable), len(self.lat), len(self.lon)), dtype=np.float32
        )
        return xr.DataArray(
            data,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": self.lat,
                "lon": self.lon,
            },
        )


class MockForecastERA5:
    """Forecast source with [time, variable, lead_time, lat, lon] dimensions.

    Each variable has a distinct constant value so axis reordering can be checked.
    """

    def __init__(self) -> None:
        self.lat = np.linspace(40.0, 60.0, 21)
        self.lon = np.linspace(0.0, 20.0, 21)

    def __call__(self, time, lead_time, variable):  # noqa: ANN001
        time = np.atleast_1d(time)
        lead_time = np.atleast_1d(lead_time)
        variable = np.atleast_1d(variable)
        data = np.zeros(
            (
                len(time),
                len(variable),
                len(lead_time),
                len(self.lat),
                len(self.lon),
            ),
            dtype=np.float32,
        )
        for vi in range(len(variable)):
            data[:, vi] = float(vi + 1)  # var 0 -> 1.0, var 1 -> 2.0
        return xr.DataArray(
            data,
            dims=["time", "variable", "lead_time", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lead_time": lead_time,
                "lat": self.lat,
                "lon": self.lon,
            },
        )


# Mock DiT with the nested attributes used by load_model and set_domain.
class _Tok:
    patch_size = (2, 2)


class _Detok:
    pass


class _Attn:
    # load_model uses this kernel to set the per-side minimum sub-domain size.
    attn_kernel = 1


class _Block:
    def __init__(self) -> None:
        self.attention = _Attn()


class _Inner:
    def __init__(self) -> None:
        self.tokenizer = _Tok()
        self.detokenizer = _Detok()
        self.attn_kwargs_forward: dict = {}
        self.blocks = [_Block()]


class _Mid:
    def __init__(self) -> None:
        self.model = _Inner()


class PhooNet(torch.nn.Module):
    """Mock diffusion network returning zeros or the model-space previous state."""

    def __init__(self, identity: bool = False, n_state: int = 0) -> None:
        super().__init__()
        self.model = _Mid()
        self.identity = identity
        self.n_state = n_state
        self.seen_prev: torch.Tensor | None = None

    def forward(self, x, sigma, condition=None, attn_kwargs=None):  # noqa: ANN001
        if not self.identity:
            return torch.zeros_like(x)
        cc = condition["cond_concat"] if hasattr(condition, "keys") else condition
        prev = cc[:, -self.n_state :]
        self.seen_prev = prev.detach().clone()
        return prev.to(x.dtype)


def _build(**overrides) -> StormCastEurope:
    lat_input = torch.linspace(45.0, 55.0, 11)
    lon_input = torch.linspace(5.0, 15.0, 11)
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    kw = dict(
        state_variables=STATE,
        era5_variables=["u10m", "t2m"],
        diffusion_model=PhooNet(),
        resolution="rea6",
        lat_input_grid=lat_input,
        lon_input_grid=lon_input,
        lat_output_grid=47.0 + yy,
        lon_output_grid=7.0 + xx,
        era5_center=torch.zeros(2),
        era5_scale=torch.ones(2),
        state_center=torch.zeros(3),
        state_scale=torch.ones(3),
        static_invariants=OrderedDict(
            elevation_norm=torch.zeros(H, W), land_fraction=torch.ones(H, W) * 0.5
        ),
        pre_invariant_variables=[
            "sin_lat",
            "cos_lat",
            "sin_lon",
            "cos_lon",
            "elevation_norm",
        ],
        post_invariant_variables=["land_fraction"],
        channel_transforms=TRANSFORMS,
        conditioning_data_source=MockERA5(),
        number_of_steps=4,
    )
    kw.update(overrides)
    return StormCastEurope(**kw)


def _coords() -> OrderedDict:
    return OrderedDict(
        batch=np.empty(0),
        time=np.array([np.datetime64("2016-06-15T10:00:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(CANONICAL),
        rea_y=np.arange(H),
        rea_x=np.arange(W),
    )


def _state() -> torch.Tensor:
    x = torch.zeros(1, 1, 1, len(STATE), H, W)
    x[:, :, :, 0] = 280.0  # t2m (K)
    x[:, :, :, 1] = 0.002  # tp (m) == 2 mm
    x[:, :, :, 2] = 0.5  # tcc (fraction)
    return x


def test_stormcasteurope_call_wiring() -> None:
    # Check unit decoding, model-space conditioning, and canonical encoding.
    net = PhooNet(identity=True, n_state=len(STATE))
    model = _build(diffusion_model=net, number_of_steps=8)
    x = _state()
    out, _ = model(x, _coords())
    # De-normalizing the conditioned state recovers the decoded physical input.
    seen_phys = model._state_from_model(net.seen_prev)
    decoded = model._decode_units(x.clone())[:, 0, 0]
    torch.testing.assert_close(seen_phys, decoded, rtol=1e-4, atol=1e-4)
    # The identity rollout preserves the canonical input tensor.
    torch.testing.assert_close(out, x, rtol=1e-3, atol=1e-4)


def test_stormcasteurope_background_layout() -> None:
    # Expected order: ERA5 | position/elevation | cos_zenith | land_fraction.
    model = _build()
    lat2d, lon2d = model.lat_output_grid, model.lon_output_grid
    hin, win = model.lat_input_numpy.shape[0], model.lon_input_numpy.shape[0]
    era5 = torch.zeros(2, hin, win)
    era5[0], era5[1] = 5.0, 9.0  # constant ERA5 fields (center=0, scale=1 -> unchanged)
    valid = datetime(2016, 6, 15, 12, tzinfo=timezone.utc)
    bg = model._background(era5, valid, lat2d, lon2d)
    assert bg.shape[1] == 9
    # ERA5 block first, unchanged by the identity normalization
    assert bg[0, 0].mean().item() == pytest.approx(5.0)
    assert bg[0, 1].mean().item() == pytest.approx(9.0)
    torch.testing.assert_close(bg[0, 2:6], model._position_channels(lat2d, lon2d))
    # last pre-invariant (elevation_norm = 0) sits at index 6, before cos_zenith
    assert bg[0, 6].abs().max().item() == pytest.approx(0.0)
    # cos_zenith occupies index 7 (after the pre-invariants, before the post block)
    cz = torch.as_tensor(
        model._cos_zenith(valid, lat2d.cpu().numpy(), lon2d.cpu().numpy()),
        dtype=torch.float32,
    )
    torch.testing.assert_close(bg[0, 7], cz)
    # post-invariant (land_fraction = 0.5) comes after cos_zenith
    assert bg[0, 8].mean().item() == pytest.approx(0.5)


def test_stormcasteurope_transforms_explicit() -> None:
    # Check explicit expected values, including the asinh transform omitted by the
    # default test configuration.
    transforms = {
        "TOT_PRECIP": {"transform": "asinh", "eps": 0.5},
        "CLCT": {"transform": "log_eps", "eps": 1e-3},
    }
    center = torch.tensor([280.0, 0.5, 1.0])
    scale = torch.tensor([10.0, 2.0, 0.5])
    model = _build(
        state_center=center, state_scale=scale, channel_transforms=transforms
    )

    x = torch.zeros(1, len(STATE), H, W)
    x[:, 0], x[:, 1], x[:, 2] = 290.0, 1.5, 0.7  # internal COSMO units
    model_state = model._state_to_model(x)
    # T_2M: no transform, just z-score -> (290-280)/10 = 1.0
    assert model_state[0, 0].mean().item() == pytest.approx(1.0)
    # TOT_PRECIP: asinh(1.5/0.5) then z-score -> (asinh(3)-0.5)/2
    assert model_state[0, 1].mean().item() == pytest.approx(
        (np.arcsinh(3.0) - 0.5) / 2.0, rel=1e-5
    )
    # CLCT: log1p(0.7/1e-3) then z-score -> (log1p(700)-1)/0.5
    assert model_state[0, 2].mean().item() == pytest.approx(
        (np.log1p(700.0) - 1.0) / 0.5, rel=1e-5
    )
    # The inverse recovers the physical input within floating-point tolerance.
    recovered = model._state_from_model(model_state)
    torch.testing.assert_close(recovered, x, rtol=1e-4, atol=1e-4)


def test_stormcasteurope_iter() -> None:
    model = _build()
    leads = []
    for i, (xo, co) in enumerate(model.create_iterator(_state(), _coords())):
        leads.append(int(co["lead_time"][0] / np.timedelta64(1, "h")))
        assert tuple(xo.shape) == (1, 1, 1, len(STATE), H, W)
        assert torch.isfinite(xo).all()
        if i == 2:
            break
    assert leads == [0, 1, 2]  # initial condition first, then 1 h, 2 h


def test_stormcasteurope_units() -> None:
    model = _build()
    # Coordinates use canonical names; tp and tcc require unit conversion.
    assert list(model.input_coords()["variable"]) == CANONICAL

    x = _state()
    internal = model._decode_units(x.clone())
    assert internal[0, 0, 0, 1, 0, 0].item() == pytest.approx(2.0)  # 0.002 m -> 2 mm
    assert internal[0, 0, 0, 2, 0, 0].item() == pytest.approx(50.0)  # 0.5 -> 50 %
    # Encoding reverses decoding within floating-point tolerance.
    torch.testing.assert_close(model._encode_units(internal.clone()), x)


def test_stormcasteurope_rejects_nonfinite_input() -> None:
    model = _build()
    x = _state()
    x[0, 0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        model(x, _coords())


def test_stormcasteurope_constraints() -> None:
    # An identity transition isolates the physical output constraint.
    net = PhooNet(identity=True, n_state=len(STATE))
    base = _build(diffusion_model=net, number_of_steps=8)
    assert base._has_constraints is False
    out0, _ = base(_state(), _coords())
    assert out0[0, 0, 0, 0].max().item() == pytest.approx(280.0, abs=1e-2)  # unchanged

    net2 = PhooNet(identity=True, n_state=len(STATE))
    model = _build(
        diffusion_model=net2,
        number_of_steps=8,
        constraints={"bounds": {"T_2M": {"max": 250.0}}},
    )
    assert model._has_constraints is True
    out1, _ = model(_state(), _coords())
    assert out1[0, 0, 0, 0].max().item() == pytest.approx(250.0, abs=1e-2)  # clamped


def test_stormcasteurope_solar_gate() -> None:
    # The shortwave solar gate attenuates radiation at night; other channels remain
    # unchanged.
    state_variables = ["T_2M", "ASWDIR_S", "ASWDIFD_S"]
    model = _build(
        state_variables=state_variables,
        state_center=torch.zeros(len(state_variables)),
        state_scale=torch.ones(len(state_variables)),
        channel_transforms={},
        constraints={
            "sza_gate": {
                "channels": {
                    "ASWDIR_S": {"threshold": 0.01},
                    "ASWDIFD_S": {"threshold": 0.01},
                }
            }
        },
    )
    # Night: the gate closes and zeros both shortwave channels.
    night_state = torch.full((1, len(state_variables), H, W), 500.0)
    night = datetime(2016, 1, 1, 0, 0, tzinfo=timezone.utc)  # deep winter night
    model._apply_constraints(
        night_state, night, model.lat_output_numpy, model.lon_output_numpy
    )
    assert night_state[0, 1].abs().max().item() < 1.0  # ASWDIR_S gated to ~0
    assert night_state[0, 2].abs().max().item() < 1.0  # ASWDIFD_S gated to ~0
    assert night_state[0, 0].min().item() == pytest.approx(500.0)  # T_2M unchanged

    # Day: the gate is fully open, so shortwave radiation passes through unchanged
    # (a gate stuck at ~0 or an inverted sign would fail here).
    day_state = torch.full((1, len(state_variables), H, W), 500.0)
    noon = datetime(2016, 6, 15, 12, 0, tzinfo=timezone.utc)  # high summer sun
    model._apply_constraints(
        day_state, noon, model.lat_output_numpy, model.lon_output_numpy
    )
    assert day_state[0, 1].min().item() == pytest.approx(500.0)  # ASWDIR_S preserved
    assert day_state[0, 2].min().item() == pytest.approx(500.0)  # ASWDIFD_S preserved


def test_stormcasteurope_constraint_validation() -> None:
    # Reject invalid bounds and incomplete shortwave constraints.
    shortwave_config = {
        "state_variables": ["T_2M", "ASWDIR_S"],
        "state_center": torch.zeros(2),
        "state_scale": torch.ones(2),
        "channel_transforms": {},
    }
    with pytest.raises(ValueError, match="min .* > max"):
        _build(constraints={"bounds": {"T_2M": {"min": 340.0, "max": 180.0}}})
    with pytest.raises(ValueError, match="not finite"):
        _build(constraints={"bounds": {"T_2M": {"max": float("inf")}}})
    # A known channel with a non-dict bounds entry is a packaging error, not ignored.
    with pytest.raises(ValueError, match="must be a dict"):
        _build(constraints={"bounds": {"T_2M": [180.0, 340.0]}})
    # An unsupported bounds mode (e.g. a typo) must raise, not silently clamp.
    with pytest.raises(ValueError, match="unsupported"):
        _build(constraints={"bounds": {"T_2M": {"min": 180.0, "mode": "clmp"}}})
    # Reject non-finite solar-gate parameters.
    with pytest.raises(ValueError, match="threshold must be finite"):
        _build(
            **shortwave_config,
            constraints={"sza_gate": {"channels": {"ASWDIR_S": {"threshold": np.nan}}}},
        )
    with pytest.raises(ValueError, match="half_width must be finite"):
        _build(
            **shortwave_config,
            constraints={
                "sza_gate": {
                    "half_width": np.nan,
                    "channels": {"ASWDIR_S": {"threshold": 0.01}},
                }
            },
        )
    # Every shortwave state channel requires a solar gate when constraints are set.
    with pytest.raises(ValueError, match="sza_gate must cover"):
        _build(
            **shortwave_config,
            constraints={"bounds": {"T_2M": {"min": 180.0, "max": 340.0}}},
        )
    # Omitting constraints permits direct, unconstrained construction.
    _build(**shortwave_config)


def _hub_build(**overrides) -> StormCastEurope:  # noqa: ANN003
    kw = dict(
        state_variables=WIND_STATE,
        state_center=torch.zeros(len(WIND_STATE)),
        state_scale=torch.ones(len(WIND_STATE)),
        channel_transforms={},
        wind_levels=WIND_LEVELS,
    )
    kw.update(overrides)
    return _build(**kw)


def test_stormcasteurope_hub_validation() -> None:
    # Reject duplicate output labels, invalid heights, and state-name collisions.
    with pytest.raises(ValueError, match="duplicate output labels"):
        _hub_build(hub_heights=[30.0, 30.0])
    with pytest.raises(ValueError, match="finite and positive"):
        _hub_build(hub_heights=[float("inf")])
    # A 10 m hub collides with the canonical names for U_10M and V_10M.
    state_variables = ["U_10M", "V_10M", "U_L40", "V_L40", "U_L39", "V_L39"]
    with pytest.raises(ValueError, match="collide with existing state"):
        _hub_build(
            state_variables=state_variables,
            state_center=torch.zeros(len(state_variables)),
            state_scale=torch.ones(len(state_variables)),
            hub_heights=[10.0],
        )


def test_stormcasteurope_hub_wind() -> None:
    # Interpolate the 10 m and 50 m wind levels to a 30 m hub height.
    model = _hub_build(hub_heights=[30.0])
    n_state = len(WIND_STATE)
    input_variables = list(model.input_coords()["variable"])
    assert "u30m" not in input_variables and "v30m" not in input_variables

    physical_state = torch.zeros(1, n_state, H, W)
    # Wind components at the 10 m and 50 m levels.
    physical_state[:, 1], physical_state[:, 3] = 4.0, 8.0
    physical_state[:, 2], physical_state[:, 4] = 1.0, 3.0
    derived = model._derive_hub_wind(physical_state)
    assert derived.shape == (1, n_state + 2, H, W)
    # 30 m is halfway between 10 and 50 -> weight 0.5
    assert derived[0, n_state].mean().item() == pytest.approx(6.0)
    assert derived[0, n_state + 1].mean().item() == pytest.approx(2.0)

    # __call__ appends hub-height wind to the state output.
    state = torch.zeros(1, 1, 1, n_state, H, W)
    coords = OrderedDict(
        batch=np.empty(0),
        time=np.array([np.datetime64("2016-06-15T10:00:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(input_variables),
        rea_y=np.arange(H),
        rea_x=np.arange(W),
    )
    output, output_coords = model(state, coords)
    assert output.shape == (1, 1, 1, n_state + 2, H, W)
    assert list(output_coords["variable"][-2:]) == ["u30m", "v30m"]
    assert torch.isfinite(output).all()


def test_interp_levels_to_height_linear_and_log() -> None:
    # Compare linear-height and log-height interpolation.
    values = torch.tensor([1.0, 4.0]).view(2, 1, 1)  # [K, H, W]
    heights = torch.tensor([10.0, 40.0]).view(2, 1, 1)
    # linear at 25 m (halfway) -> 1 + 3*0.5 = 2.5
    linear = _interp_levels_to_height(values, heights, 25.0, method="linear")
    assert linear.item() == pytest.approx(2.5)
    # log at 20 m -> weight (ln20-ln10)/(ln40-ln10), value 1 + 3*w
    log_weight = (np.log(20) - np.log(10)) / (np.log(40) - np.log(10))
    logarithmic = _interp_levels_to_height(values, heights, 20.0, method="log")
    assert logarithmic.item() == pytest.approx(1.0 + 3.0 * log_weight)
    # outside the profile clamps to the nearest level (no extrapolation)
    assert _interp_levels_to_height(values, heights, 5.0).item() == pytest.approx(1.0)
    assert _interp_levels_to_height(values, heights, 99.0).item() == pytest.approx(4.0)


def test_stormcasteurope_hub_wind_terrain() -> None:
    # Terrain shifts the upper wind level from 50 m to 60 m.
    wind_levels = {
        "elevation_invariant": "elevation_norm",
        "levels": [
            {"u": "U_L40", "v": "V_L40", "a": 10.0, "b": 0.0},
            {"u": "U_L39", "v": "V_L39", "a": 50.0, "b": 10.0},
        ],
    }
    model = _hub_build(
        hub_heights=[35.0],
        wind_levels=wind_levels,
        static_invariants=OrderedDict(
            elevation_norm=torch.ones(H, W),
            land_fraction=torch.full((H, W), 0.5),
        ),
    )
    n_state = len(WIND_STATE)
    physical_state = torch.zeros(1, n_state, H, W)
    physical_state[:, 1], physical_state[:, 3] = 4.0, 8.0
    physical_state[:, 2], physical_state[:, 4] = 1.0, 5.0
    derived = model._derive_hub_wind(physical_state)
    # At 35 m, the interpolation weight between 10 m and 60 m is 0.5.
    assert derived[0, n_state].mean().item() == pytest.approx(6.0)
    assert derived[0, n_state + 1].mean().item() == pytest.approx(3.0)


def test_stormcasteurope_set_domain() -> None:
    # set_domain returns a working model on the selected sub-domain.
    model = _build()  # grid lat 47-50, lon 7-12 (4x6)
    subdomain = model.set_domain(48.0, 49.5, 8.0, 11.0)
    height, width = subdomain.lat_output_numpy.shape
    assert (height, width) == (len(subdomain.rea_y), len(subdomain.rea_x))
    assert height < H or width < W  # strictly smaller than the parent
    assert list(subdomain.input_coords()["variable"]) == CANONICAL
    # A bounding box outside the footprint raises.
    with pytest.raises(ValueError, match="not fully inside"):
        model.set_domain(10.0, 12.0, 8.0, 11.0)

    # The sub-domain produces a finite rollout step on its own grid.
    state = torch.zeros(1, 1, 1, len(STATE), height, width)
    state[:, :, :, 0], state[:, :, :, 1], state[:, :, :, 2] = 280.0, 0.002, 0.5
    coords = OrderedDict(
        batch=np.empty(0),
        time=np.array([np.datetime64("2016-06-15T10:00:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(CANONICAL),
        rea_y=subdomain.rea_y,
        rea_x=subdomain.rea_x,
    )
    output, output_coords = subdomain(state, coords)
    assert tuple(output.shape) == (1, 1, 1, len(STATE), height, width)
    assert torch.isfinite(output).all()
    assert output_coords["lead_time"][0] == np.timedelta64(1, "h")


def test_stormcasteurope_set_domain_preserves_config() -> None:
    # The sub-domain must preserve the parent's inference configuration.
    model = _build(
        era5_center=torch.tensor([1.0, 2.0]),
        era5_scale=torch.tensor([3.0, 4.0]),
        state_center=torch.tensor([5.0, 6.0, 7.0]),
        state_scale=torch.tensor([8.0, 9.0, 10.0]),
        number_of_steps=6,
        sigma_min=0.01,
        sigma_max=700.0,
        rho=8.0,
        solver="euler",
        physical_clamp=False,
        amp=True,
        constraints={"bounds": {"T_2M": {"min": 200.0, "max": 320.0}}},
    )
    model._patch_size, model._min_domain_cells = 2, 0
    subdomain = model.set_domain(48.0, 49.5, 8.0, 11.0)
    assert subdomain.number_of_steps == 6
    assert (subdomain.sigma_min, subdomain.sigma_max, subdomain.rho) == (
        0.01,
        700.0,
        8.0,
    )
    assert subdomain.solver == "euler"
    assert subdomain.physical_clamp is False and subdomain.amp is True
    assert subdomain._constraints == model._constraints
    assert subdomain.diffusion_model is model.diffusion_model
    assert subdomain._channel_transforms == model._channel_transforms
    torch.testing.assert_close(
        subdomain.era5_center.flatten(), model.era5_center.flatten()
    )
    torch.testing.assert_close(
        subdomain.era5_scale.flatten(), model.era5_scale.flatten()
    )
    torch.testing.assert_close(
        subdomain.state_center.flatten(), model.state_center.flatten()
    )
    torch.testing.assert_close(
        subdomain.state_scale.flatten(), model.state_scale.flatten()
    )
    assert subdomain._patch_size == 2 and subdomain._min_domain_cells == 0


def test_stormcasteurope_domain_guards() -> None:
    # Reject malformed set_domain arguments up front with clear messages.
    model = _build()
    with pytest.raises(ValueError, match="lat_min <= lat_max"):
        model.set_domain(50.0, 48.0, 8.0, 11.0)
    with pytest.raises(ValueError, match="must be finite"):
        model.set_domain(float("nan"), 49.0, 8.0, 11.0)
    with pytest.raises(ValueError, match="margin_deg must be > 0"):
        model.set_domain(48.0, 49.0, 8.0, 11.0, margin_deg=0.0)

    # Set the patch and kernel limits directly for the synthetic network.
    model._patch_size = 2
    model._min_domain_cells = 6  # wider than a small bounding box can supply
    with pytest.raises(ValueError, match="per-side minimum"):
        model.set_domain(48.0, 48.4, 8.0, 8.4)

    # Force patch alignment to remove requested cells and check that it raises.
    shrink_model = _build()
    shrink_model._patch_size = 2
    shrink_model._min_domain_cells = 0
    shrink_model._snap_to_patch = lambda lo, hi, n: (lo, hi - 2)
    with pytest.raises(ValueError, match="not divisible by patch_size"):
        shrink_model.set_domain(48.0, 49.5, 8.0, 11.0)


def test_stormcasteurope_hub_subdomain_iter() -> None:
    # A sub-domain rollout retains the derived hub-height wind components.
    model = _hub_build(hub_heights=[30.0])
    subdomain = model.set_domain(48.0, 49.5, 8.0, 11.0)
    height, width = subdomain.lat_output_numpy.shape
    n_state = len(WIND_STATE)
    n_output = n_state + 2

    state = torch.zeros(1, 1, 1, n_state, height, width)
    coords = OrderedDict(
        batch=np.empty(0),
        time=np.array([np.datetime64("2016-06-15T10:00:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(list(subdomain.input_coords()["variable"])),
        rea_y=subdomain.rea_y,
        rea_x=subdomain.rea_x,
    )
    iterator = subdomain.create_iterator(state, coords)
    initial, initial_coords = next(iterator)
    assert initial.shape == (1, 1, 1, n_output, height, width)
    assert list(initial_coords["variable"][-2:]) == ["u30m", "v30m"]
    step, step_coords = next(iterator)
    assert step.shape == (1, 1, 1, n_output, height, width)
    assert torch.isfinite(step).all()
    assert step_coords["lead_time"][0] == np.timedelta64(1, "h")


def test_stormcasteurope_rear_hook_hub_consistency() -> None:
    # Hub-height wind must reflect the state after the rear hook.
    model = _hub_build(hub_heights=[30.0])

    def rear(
        state: torch.Tensor, coords: OrderedDict
    ) -> tuple[torch.Tensor, OrderedDict]:
        # Set known values at 10 m and 50 m; the 30 m interpolation weight is 0.5.
        state = state.clone()
        state[:, :, :, 1], state[:, :, :, 3] = 0.0, 20.0
        state[:, :, :, 2], state[:, :, :, 4] = 0.0, 4.0
        return state, coords

    model.rear_hook = rear
    n_state = len(WIND_STATE)
    state = torch.zeros(1, 1, 1, n_state, H, W)
    coords = OrderedDict(
        batch=np.empty(0),
        time=np.array([np.datetime64("2016-06-15T10:00:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(list(model.input_coords()["variable"])),
        rea_y=np.arange(H),
        rea_x=np.arange(W),
    )
    iterator = model.create_iterator(state, coords)
    next(iterator)  # initial condition; hook not yet applied
    step, _ = next(iterator)
    # State winds use hook values, and hub-height wind is recomputed from them.
    assert step[0, 0, 0, 1].mean().item() == pytest.approx(0.0)  # U_L40
    assert step[0, 0, 0, 3].mean().item() == pytest.approx(20.0)  # U_L39
    assert step[0, 0, 0, n_state].mean().item() == pytest.approx(10.0)  # u30m
    assert step[0, 0, 0, n_state + 1].mean().item() == pytest.approx(2.0)  # v30m


def test_stormcasteurope_reorders_forecast_source_dimensions() -> None:
    # Reorder ForecastSource dimensions before indexing time and lead time.
    model = _build(conditioning_data_source=MockForecastERA5())
    coords = _coords()
    coords["batch"] = np.array([0])  # materialize one batch element for a direct call
    era5 = model._get_conditioning(coords, torch.device("cpu"))
    # The variable axis follows era5_variables order: u10m=1 and t2m=2.
    assert tuple(era5.shape[:4]) == (1, 1, 1, len(model.era5_variables))
    assert era5[0, 0, 0, 0].mean().item() == pytest.approx(1.0)  # u10m
    assert era5[0, 0, 0, 1].mean().item() == pytest.approx(2.0)  # t2m


def test_stormcasteurope_rejects_misaligned_conditioning_grid() -> None:
    model = _build()
    coords = _coords()
    coords["batch"] = np.array([0])
    lead_time = coords["lead_time"] + model.time_step
    misaligned_lat = model.lat_input_numpy.copy()
    misaligned_lat[0] += 0.01
    conditioning = torch.zeros(
        1,
        1,
        len(model.era5_variables),
        len(model.lat_input_numpy),
        len(model.lon_input_numpy),
    )
    conditioning_coords = OrderedDict(
        time=coords["time"],
        lead_time=lead_time,
        variable=np.array(model.era5_variables),
        _lat=misaligned_lat,
        _lon=model.lon_input_numpy,
    )
    with (
        patch(
            "earth2studio.models.px.stormcasteurope.fetch_data",
            return_value=(conditioning, conditioning_coords),
        ),
        pytest.raises(RuntimeError, match="latitude coordinates"),
    ):
        model._get_conditioning(coords, torch.device("cpu"))


def test_stormcasteurope_target_time_conditioning() -> None:
    # Fetch ERA5 at time + lead_time + one hour for each lead.
    source = MockERA5()
    model = _build(conditioning_data_source=source)
    coords = _coords()
    coords["lead_time"] = np.array([np.timedelta64(0, "h"), np.timedelta64(1, "h")])
    state = torch.zeros(1, 1, len(coords["lead_time"]), len(STATE), H, W)
    state[:, :, :, 0], state[:, :, :, 1], state[:, :, :, 2] = 280.0, 0.002, 0.5
    model(state, coords)
    initial_time = coords["time"][0]
    requested_times = {
        np.datetime64(time, "h") for call in source.calls for time in call[0]
    }
    expected_times = {
        np.datetime64(initial_time + np.timedelta64(hour, "h"), "h") for hour in (1, 2)
    }
    requested_variables = {variable for call in source.calls for variable in call[1]}
    assert requested_times == expected_times
    assert requested_variables == set(model.era5_variables)


def test_stormcasteurope_basic_validation() -> None:
    with pytest.raises(ValueError, match="resolution"):
        _build(resolution="rea9")
    with pytest.raises(ValueError, match="number_of_steps"):
        _build(number_of_steps=1)
    # Public input variable names must be unique.
    with pytest.raises(ValueError, match="duplicate Earth2Studio"):
        _build(
            state_variables=["U_10M", "U_10M"],
            state_center=torch.zeros(2),
            state_scale=torch.ones(2),
            channel_transforms={},
        )
    # Non-canonical variable names fail coordinate validation.
    model = _build()
    bad_coords = _coords()
    bad_coords["variable"] = np.array(["T_2M", "TOT_PRECIP", "CLCT"])
    with pytest.raises(ValueError):
        model.output_coords(bad_coords)
    # Reject a COSMO output grid outside the ERA5 input footprint.
    with pytest.raises(ValueError, match="inside the ERA5 input grid"):
        _build(lat_output_grid=torch.full((H, W), 40.0))
    # Reject an output grid that is not divisible by the DiT patch size.
    with pytest.raises(ValueError, match="divisible"):
        _build()._rebind_latent(4, 5)
    # Reject a static invariant whose shape differs from the output grid.
    with pytest.raises(ValueError, match="!= the COSMO output grid"):
        _build(
            static_invariants=OrderedDict(
                elevation_norm=torch.zeros(H + 1, W),  # wrong shape
                land_fraction=torch.full((H, W), 0.5),
            )
        )


def test_stormcasteurope_rejects_invalid_configuration_and_coords() -> None:
    with pytest.raises(ValueError, match="finite values"):
        _build(lat_input_grid=torch.tensor([45.0, float("inf")]))

    invalid_transforms = [
        ({"transform": "bad_log_eps", "eps": 1e-4}, "unsupported transform"),
        ({"transform": "log_eps", "eps": 0.0}, "eps must be finite and > 0"),
        ({"transform": "asinh", "eps": float("inf")}, "eps must be finite and > 0"),
        ({"transform": "logit_eps", "eps": 0.5}, "0 < eps < 0.5"),
        (
            {"transform": "logit_eps", "eps": 1e-3, "scale": 0.0},
            "scale must be finite and > 0",
        ),
    ]
    for spec, message in invalid_transforms:
        with pytest.raises(ValueError, match=message):
            _build(channel_transforms={"TOT_PRECIP": spec})

    # Empty pre-invariants omit required position channels.
    with pytest.raises(ValueError, match="non-empty"):
        _build(pre_invariant_variables=[])
    # Reject a missing position channel.
    with pytest.raises(ValueError, match="missing position channels"):
        _build(
            pre_invariant_variables=["sin_lat", "cos_lat", "sin_lon", "elevation_norm"]
        )
    # Reject position channels outside the trained order.
    with pytest.raises(ValueError, match="trained order"):
        _build(pre_invariant_variables=["cos_lat", "sin_lat", "sin_lon", "cos_lon"])

    # Reject a missing post-invariant at construction.
    with pytest.raises(ValueError, match="missing non-position invariant"):
        _build(post_invariant_variables=["z0_lu"])  # not in static_invariants

    # Reject malformed output grids.
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    nan_grid = (47.0 + yy).clone()
    nan_grid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _build(lat_output_grid=nan_grid)
    with pytest.raises(ValueError, match="2D curvilinear"):
        _build(lat_output_grid=torch.linspace(47.0, 50.0, H))  # 1D
    with pytest.raises(ValueError, match="non-empty"):
        _build(
            lat_output_grid=torch.zeros(0, W), lon_output_grid=torch.zeros(0, W)
        )  # empty 2D grid
    with pytest.raises(ValueError, match="same shape"):
        _build(lon_output_grid=torch.zeros(H, W + 1))  # shape mismatch vs lat grid

    model = _build()
    # Reject a tensor whose lead dimension differs from its coordinates.
    coords = _coords()
    coords["lead_time"] = np.array([np.timedelta64(0, "h"), np.timedelta64(1, "h")])
    with pytest.raises(ValueError, match="does not match coords"):
        model(_state(), coords)  # x has lead dim 1, coords says 2
    # Reversed rea_y values fail coordinate validation.
    reversed_coords = _coords()
    reversed_coords["rea_y"] = reversed_coords["rea_y"][::-1].copy()
    with pytest.raises(ValueError):
        model(_state(), reversed_coords)
    # Mismatched grid lengths fail coordinate validation.
    wrong_size_coords = _coords()
    wrong_size_coords["rea_y"], wrong_size_coords["rea_x"] = np.arange(W), np.arange(H)
    with pytest.raises(ValueError):
        model(_state(), wrong_size_coords)
    # A missing time dimension is rejected during coordinate validation.
    bad_time = OrderedDict(("t" if k == "time" else k, v) for k, v in _coords().items())
    with pytest.raises(KeyError, match="Required dimension time"):
        model._validate_coords(bad_time)
    # Device mismatch is only observable with a second device.
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="device"):
            model(_state().cuda(), _coords())


# ── package loading ──────────────────────────────────────────────────────────


def _create_test_package(  # noqa: ANN001
    tmp_path,
    extended: bool = True,
    inv_arrays: dict | None = None,
    norm_stats: dict | None = None,
    native_offset: tuple = (1, 1),
    native_shape: tuple | None = None,
    constraints: bool | dict = True,
    arch_meta: dict | None = None,
) -> Package:
    """Build a synthetic on-disk package matching the ``load_model`` layout.

    ``extended`` writes physical invariants on an extended grid
    (``invariants_ext.nc`` + ``invariants_norm_stats.json`` + an ``invariants``
    metadata block). ``extended=False`` writes pre-normalized invariants. The
    remaining arguments allow tests to provide values or invalid metadata.
    """
    resolution_dir = tmp_path / "rea6"
    resolution_dir.mkdir(parents=True, exist_ok=True)

    lat_in = np.linspace(45.0, 55.0, 11, dtype=np.float32)
    lon_in = np.linspace(5.0, 15.0, 11, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    lat_out = (47.0 + yy).astype(np.float32)  # strictly inside the input grid
    lon_out = (7.0 + xx).astype(np.float32)
    xr.Dataset(
        {
            "lat_input": (("lat_in",), lat_in),
            "lon_input": (("lon_in",), lon_in),
            "lat_output": (("y", "x"), lat_out),
            "lon_output": (("y", "x"), lon_out),
        }
    ).to_netcdf(resolution_dir / "grids.nc")

    rng = np.random.default_rng(0)
    invariant_metadata = None
    if extended:
        # Store physical invariants on an extended grid.
        extended_height, extended_width = H + 2, W + 2
        if inv_arrays is None:
            inv_arrays = {
                "elevation": rng.standard_normal(
                    (extended_height, extended_width)
                ).astype(np.float32),
                "land_fraction": rng.random((extended_height, extended_width)).astype(
                    np.float32
                ),
            }
        if norm_stats is None:
            norm_stats = {"elevation": {"method": "zscore", "mean": 100.0, "std": 50.0}}
        # Align the native crop with the output grid in grids.nc.
        extended_height, extended_width = next(iter(inv_arrays.values())).shape
        i0, j0 = native_offset
        eyy, exx = np.meshgrid(
            np.arange(extended_height), np.arange(extended_width), indexing="ij"
        )
        ext_lat = (47.0 + (eyy - i0)).astype(np.float32)
        ext_lon = (7.0 + (exx - j0)).astype(np.float32)
        ext_vars = {k: (("y", "x"), v) for k, v in inv_arrays.items()}
        ext_vars["lat"] = (("y", "x"), ext_lat)
        ext_vars["lon"] = (("y", "x"), ext_lon)
        xr.Dataset(ext_vars).to_netcdf(resolution_dir / "invariants_ext.nc")
        (resolution_dir / "invariants_norm_stats.json").write_text(
            json.dumps({"channels": norm_stats})
        )
        invariant_metadata = {
            "file": "invariants_ext.nc",
            "norm_stats_file": "invariants_norm_stats.json",
            "channels": list(inv_arrays.keys()),
            "native_offset": list(native_offset),
            "native_shape": (
                list(native_shape) if native_shape is not None else [H, W]
            ),
        }
    else:
        xr.Dataset(
            {
                "elevation_norm": (
                    ("y", "x"),
                    rng.standard_normal((H, W)).astype(np.float32),
                ),
                "land_fraction": (
                    ("y", "x"),
                    rng.random((H, W)).astype(np.float32),
                ),
            }
        ).to_netcdf(resolution_dir / "invariants.nc")

    stats = {
        "era5": {v: {"mean": 0.0, "std": 1.0} for v in ["u10m", "t2m"]},
        "state": {v: {"mean": 0.0, "std": 1.0} for v in STATE},
    }
    (resolution_dir / "stats.json").write_text(json.dumps(stats))

    meta = {
        "era5_variables": ["u10m", "t2m"],
        "state_variables": STATE,
        "pre_invariant_variables": [
            "sin_lat",
            "cos_lat",
            "sin_lon",
            "cos_lon",
            "elevation_norm",
        ],
        "post_invariant_variables": ["land_fraction"],
        "channel_transforms": TRANSFORMS,
        "sampler": {
            "num_steps": 4,
            "sigma_min": 0.002,
            "sigma_max": 800.0,
            "rho": 7.0,
            "solver": "euler",
            "physical_clamp": False,
            "amp": True,
        },
        "checkpoints": {"rea6": "ckpt.mdlus"},
    }
    if constraints:
        # Packages include physical constraints; a dict overrides the test default.
        meta["constraints"] = (
            constraints
            if isinstance(constraints, dict)
            else {"bounds": {"T_2M": {"min": 180.0, "max": 340.0}}}
        )
    if invariant_metadata is not None:
        meta["invariants"] = invariant_metadata
    if arch_meta is not None:
        meta["diffusion"] = arch_meta
    (resolution_dir / "metadata.json").write_text(json.dumps(meta))
    (resolution_dir / "ckpt.mdlus").write_bytes(b"")  # from_checkpoint is mocked
    # Minimal root config.json (package contract; load_model resolves it).
    (tmp_path / "config.json").write_text(json.dumps({"name": "stormcast-europe"}))
    return Package(str(tmp_path))


def test_stormcasteurope_load_default_package() -> None:
    # Default package points at the public HuggingFace weights repo (lazy; no fetch).
    package = StormCastEurope.load_default_package()
    assert "nvidia/stormcast-cosmo-era5" in package.root


@pytest.mark.parametrize("extended", [True, False])
@patch("earth2studio.models.px.stormcasteurope.EDMPreconditioner")
def test_stormcasteurope_load_model_assembles(
    mock_edm, tmp_path, extended
) -> None:  # noqa: ANN001
    # Load synthetic packages with extended and pre-normalized invariants.
    mock_edm.from_checkpoint.return_value = PhooNet()
    package = _create_test_package(tmp_path, extended=extended)

    model = StormCastEurope.load_model(package, device="cpu", resolution="rea6")

    assert model.state_variables == STATE
    assert list(model.input_coords()["variable"]) == CANONICAL
    assert model.number_of_steps == 4 and model.solver == "euler"
    # Sampler flags physical_clamp and amp are read from package metadata.
    assert model.physical_clamp is False and model.amp is True
    # Packaged constraints cannot be disabled.
    assert model._has_constraints is True
    assert model._bound_lo and model._bound_up  # T_2M min/max parsed
    assert model._static_names == ["elevation_norm", "land_fraction"]
    height, width = model.lat_output_numpy.shape
    # Extended invariants are cropped to the native output grid.
    assert model.static_invariants.shape == (2, height, width)
    if extended:
        # The extended grid + invariants are retained for set_domain, and the native
        # crop matches the output grid.
        assert model._ext_lat_numpy is not None and model._ext_lon_numpy is not None
        assert model._ext_static_numpy is not None
        assert model._ext_static_numpy.shape[1:] == model._ext_lat_numpy.shape
        i0, j0 = 1, 1  # _create_test_package native_offset default
        crop = model._ext_lat_numpy[i0 : i0 + height, j0 : j0 + width]
        assert np.allclose(crop, model.lat_output_numpy, atol=1e-4)
    else:
        assert model._ext_lat_numpy is None and model._ext_static_numpy is None
    # Per-side minimum sub-domain size is attn_kernel * patch_size (1 * 2).
    assert model._patch_size == 2
    assert model._min_domain_cells == 2
    mock_edm.from_checkpoint.assert_called_once()


def test_stormcasteurope_requires_constraints(tmp_path) -> None:  # noqa: ANN001
    package = _create_test_package(tmp_path, constraints=False)
    with pytest.raises(ValueError, match="no 'constraints' block"):
        StormCastEurope.load_model(package, device="cpu", resolution="rea6")


@patch("earth2studio.models.px.stormcasteurope.EDMPreconditioner")
def test_stormcasteurope_native_offset_misalignment(
    mock_edm, tmp_path
) -> None:  # noqa: ANN001
    # Shift the native crop while preserving its shape.
    mock_edm.from_checkpoint.return_value = PhooNet()
    package = _create_test_package(tmp_path)
    metadata_path = tmp_path / "rea6" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["invariants"]["native_offset"] = [0, 0]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mis-registered"):
        StormCastEurope.load_model(package, device="cpu", resolution="rea6")


@patch("earth2studio.models.px.stormcasteurope.EDMPreconditioner")
def test_stormcasteurope_set_domain_extended(
    mock_edm, tmp_path
) -> None:  # noqa: ANN001
    # Extended-grid crops warn when they reach beyond the native footprint.
    mock_edm.from_checkpoint.return_value = PhooNet()
    extended_height, extended_width = H + 3, W + 3
    invariant_arrays = {
        "elevation": np.zeros((extended_height, extended_width), dtype=np.float32),
        "land_fraction": np.full(
            (extended_height, extended_width), 0.5, dtype=np.float32
        ),
    }
    package = _create_test_package(
        tmp_path,
        inv_arrays=invariant_arrays,
        norm_stats={"elevation": {"method": "zscore", "mean": 0.0, "std": 1.0}},
    )
    model = StormCastEurope.load_model(
        package,
        device="cpu",
        resolution="rea6",
        conditioning_data_source=MockERA5(),
    )
    assert model._ext_lat_numpy is not None and model._ext_static_numpy is not None
    native_lat_max = 50.0
    assert float(model.lat_output_numpy.max()) == pytest.approx(native_lat_max)
    # Isolate the extended-crop path from the NATTEN min-size / patch-snap guards.
    model._patch_size, model._min_domain_cells = 1, 0

    # A crop inside the native footprint must not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        native_subdomain = model.set_domain(47.5, 49.5, 8.0, 11.0)
    assert float(native_subdomain.lat_output_numpy.max()) <= native_lat_max + 1e-4

    # A requested margin cell must warn and remain in the crop.
    with pytest.warns(UserWarning, match="beyond the validated"):
        margin_subdomain = model.set_domain(48.0, 51.0, 8.0, 11.0)
    assert float(margin_subdomain.lat_output_numpy.max()) > native_lat_max

    # A native bounding box can expand into the extended margin during patch
    # alignment. The warning must therefore check the final grid.
    model._patch_size, model._min_domain_cells = 4, 0
    with pytest.warns(UserWarning, match="beyond the validated"):
        snapped_subdomain = model.set_domain(48.0, 49.0, 8.0, 11.0)
    assert float(snapped_subdomain.lat_output_numpy.max()) > native_lat_max


@patch("earth2studio.models.px.stormcasteurope.EDMPreconditioner")
def test_stormcasteurope_requires_kernel(mock_edm, tmp_path) -> None:  # noqa: ANN001
    network = PhooNet()
    for block in network.model.model.blocks:
        block.attention = object()  # no attn_kernel attribute
    mock_edm.from_checkpoint.return_value = network
    package = _create_test_package(tmp_path)
    with pytest.raises(ValueError, match="attn_kernel"):
        StormCastEurope.load_model(package, device="cpu", resolution="rea6")


@patch("earth2studio.models.px.stormcasteurope.EDMPreconditioner")
def test_stormcasteurope_kernel_consistency(mock_edm, tmp_path) -> None:  # noqa: ANN001
    # All attention blocks must use the same NATTEN kernel.
    net = PhooNet()
    blk = _Block()
    blk.attention = _Attn()
    blk.attention.attn_kernel = 5  # differs from the other block
    net.model.model.blocks = [_Block(), blk]
    mock_edm.from_checkpoint.return_value = net
    pkg = _create_test_package(tmp_path)
    with pytest.raises(ValueError, match="inconsistent NATTEN kernels"):
        StormCastEurope.load_model(pkg, device="cpu", resolution="rea6")


@patch("earth2studio.models.px.stormcasteurope.EDMPreconditioner")
def test_stormcasteurope_rejects_constraints_for_unknown_channels(
    mock_edm, tmp_path
) -> None:  # noqa: ANN001
    mock_edm.from_checkpoint.return_value = PhooNet()
    package = _create_test_package(
        tmp_path, constraints={"bounds": {"NOT_A_STATE_VAR": {"min": 0.0, "max": 1.0}}}
    )
    with pytest.raises(ValueError, match="no bounds or solar gate that applies"):
        StormCastEurope.load_model(package, device="cpu", resolution="rea6")


@patch("earth2studio.models.px.stormcasteurope.EDMPreconditioner")
def test_stormcasteurope_rejects_kernel_metadata_mismatch(
    mock_edm, tmp_path
) -> None:  # noqa: ANN001
    # The mock network and metadata report different kernels.
    mock_edm.from_checkpoint.return_value = PhooNet()
    package = _create_test_package(tmp_path, arch_meta={"attn_kernel_size": 5})
    with pytest.raises(ValueError, match="disagrees with the loaded DiT"):
        StormCastEurope.load_model(package, device="cpu", resolution="rea6")


@patch("earth2studio.models.px.stormcasteurope.EDMPreconditioner")
def test_stormcasteurope_loads_extended_invariant_values(
    mock_edm, tmp_path
) -> None:  # noqa: ANN001
    # Verify z-score and identity normalization after the native crop.
    mock_edm.from_checkpoint.return_value = PhooNet()
    extended_height, extended_width = H + 2, W + 2
    elevation = np.arange(extended_height * extended_width, dtype=np.float32).reshape(
        extended_height, extended_width
    )
    land_fraction = (elevation / 100.0).astype(np.float32)
    mean, std, offset = 100.0, 50.0, (1, 1)
    package = _create_test_package(
        tmp_path,
        inv_arrays={"elevation": elevation, "land_fraction": land_fraction},
        norm_stats={"elevation": {"method": "zscore", "mean": mean, "std": std}},
        native_offset=offset,
    )
    model = StormCastEurope.load_model(package, device="cpu", resolution="rea6")

    row_offset, column_offset = offset
    expected_elevation = ((elevation - mean) / std)[
        row_offset : row_offset + H, column_offset : column_offset + W
    ]
    expected_land_fraction = land_fraction[
        row_offset : row_offset + H, column_offset : column_offset + W
    ]
    torch.testing.assert_close(
        model.static_invariants[0], torch.as_tensor(expected_elevation)
    )
    torch.testing.assert_close(
        model.static_invariants[1], torch.as_tensor(expected_land_fraction)
    )


@patch("earth2studio.models.px.stormcasteurope.EDMPreconditioner")
def test_stormcasteurope_extended_invariant_errors(
    mock_edm, tmp_path
) -> None:  # noqa: ANN001
    # Reject invalid extended-invariant metadata.
    mock_edm.from_checkpoint.return_value = PhooNet()

    # (1) elevation_norm without z-score statistics
    with pytest.raises(ValueError, match="no zscore stats"):
        StormCastEurope.load_model(
            _create_test_package(tmp_path / "a", norm_stats={}),
            device="cpu",
            resolution="rea6",
        )
    # (2) an unsupported normalization method
    with pytest.raises(ValueError, match="unknown normalization method"):
        StormCastEurope.load_model(
            _create_test_package(
                tmp_path / "b",
                norm_stats={
                    "elevation": {"method": "zscore", "mean": 0.0, "std": 1.0},
                    "land_fraction": {"method": "minmax"},
                },
            ),
            device="cpu",
            resolution="rea6",
        )
    # (3) native_shape that does not match the output grid
    with pytest.raises(ValueError, match="native_shape"):
        StormCastEurope.load_model(
            _create_test_package(tmp_path / "c", native_shape=(H + 1, W)),
            device="cpu",
            resolution="rea6",
        )
    # (4) a z-score entry missing mean or std
    with pytest.raises(ValueError, match="missing 'mean'"):
        StormCastEurope.load_model(
            _create_test_package(
                tmp_path / "d", norm_stats={"elevation": {"method": "zscore"}}
            ),
            device="cpu",
            resolution="rea6",
        )


@pytest.mark.package
@pytest.mark.parametrize("resolution", ["rea6", "rea2"])
def test_stormcasteurope_package(resolution) -> None:  # noqa: ANN001
    # Load and run both resolutions from a package in COSMO_REA_AR_PACKAGE.
    package_path = os.environ.get("COSMO_REA_AR_PACKAGE")
    if not package_path:
        pytest.skip("set COSMO_REA_AR_PACKAGE to a built package dir to run")

    device = "cuda:0" if torch.cuda.is_available() else None
    model = StormCastEurope.load_model(
        Package(package_path), device=device, resolution=resolution
    )
    # The checkpoint must reconstruct RoPE-NATTEN attention.
    network = model.diffusion_model.model.model
    attention_classes = {type(block.attention).__name__ for block in network.blocks}
    assert attention_classes == {"RopeNatten2DSelfAttention"}, attention_classes
    attention_kernels = {int(block.attention.attn_kernel) for block in network.blocks}
    assert len(attention_kernels) == 1
    attention_kernel = next(iter(attention_kernels))
    if device is None:
        pytest.skip("NATTEN forward requires CUDA; checkpoint load verified on CPU")

    # Synthetic ERA5 covering the package's input footprint, so __call__ runs.
    latitude = np.linspace(
        float(model.lat_input_numpy.min()) - 0.5,
        float(model.lat_input_numpy.max()) + 0.5,
        32,
    )
    longitude = np.linspace(
        float(model.lon_input_numpy.min()) - 0.5,
        float(model.lon_input_numpy.max()) + 0.5,
        32,
    )
    rng = np.random.default_rng(0)

    class _GridERA5:
        def __call__(self, time, variable):  # noqa: ANN001
            time, variable = np.atleast_1d(time), np.atleast_1d(variable)
            data = rng.standard_normal(
                (len(time), len(variable), len(latitude), len(longitude))
            ).astype(np.float32)
            return xr.DataArray(
                data,
                dims=["time", "variable", "lat", "lon"],
                coords={
                    "time": time,
                    "variable": variable,
                    "lat": latitude,
                    "lon": longitude,
                },
            )

    model.conditioning_data_source = _GridERA5()
    height, width = model.lat_output_numpy.shape
    coords = OrderedDict(
        batch=np.empty(0),
        time=np.array([np.datetime64("2016-06-15T10:00:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=model.input_coords()["variable"],
        rea_y=np.arange(height),
        rea_x=np.arange(width),
    )
    state = torch.randn(
        1, 1, 1, len(model.state_variables), height, width, device=device
    )
    output, output_coords = model(state, coords)
    assert tuple(output.shape) == (
        1,
        1,
        1,
        len(model.state_variables),
        height,
        width,
    )
    assert torch.isfinite(output).all()
    assert output_coords["lead_time"][0] == np.timedelta64(1, "h")

    # The minimum sub-domain size reflects the NATTEN kernel.
    assert model._min_domain_cells == attention_kernel * model._patch_size

    # Run inference on a central European sub-domain.
    subdomain = model.set_domain(48.0, 52.0, 6.0, 12.0)
    sub_height, sub_width = subdomain.lat_output_numpy.shape
    assert min(sub_height, sub_width) >= subdomain._min_domain_cells
    assert sub_height < height or sub_width < width
    subdomain_coords = OrderedDict(
        batch=np.empty(0),
        time=np.array([np.datetime64("2016-06-15T10:00:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=subdomain.input_coords()["variable"],
        rea_y=subdomain.rea_y,
        rea_x=subdomain.rea_x,
    )
    subdomain_state = torch.randn(
        1,
        1,
        1,
        len(subdomain.state_variables),
        sub_height,
        sub_width,
        device=device,
    )
    subdomain_output, _ = subdomain(subdomain_state, subdomain_coords)
    assert tuple(subdomain_output.shape) == (
        1,
        1,
        1,
        len(subdomain.state_variables),
        sub_height,
        sub_width,
    )
    assert torch.isfinite(subdomain_output).all()
