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
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.data.camulator import CAMULATOR_GRID_LAT, CAMULATOR_GRID_LON
from earth2studio.models.nn.camulator import CamulatorNet
from earth2studio.models.nn.camulator_physics import (
    GRAVITY,
    column_integral,
    grid_area,
    weighted_sum,
)
from earth2studio.models.px.camulator import (
    _N_OUT,
    _N_STATE,
    _TRACERS,
    DIAGNOSTIC_VARIABLES,
    OUTPUT_VARIABLES,
    PROGNOSTIC_VARIABLES,
    CAMulator,
)
from earth2studio.utils import handshake_dim

H, W = len(CAMULATOR_GRID_LAT), len(CAMULATOR_GRID_LON)
CUDA = pytest.param(
    "cuda:0",
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
)


class PhooCamulatorNet(torch.nn.Module):
    """Stand-in for the CrossFormer core: maps the 136-channel input
    ``(batch, 136, 1, lat, lon)`` to the 147-channel output by passing the 130
    prognostic channels through and filling the 17 diagnostics with a constant."""

    def __init__(self, diag_value: float = 1.0e-3):
        super().__init__()
        self.diag_value = diag_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diag = torch.full(
            (x.shape[0], len(DIAGNOSTIC_VARIABLES), *x.shape[2:]),
            self.diag_value,
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x[:, :_N_STATE], diag], dim=1)


def build_model(
    core: torch.nn.Module,
    conservation_fixers: bool = False,
    wind_filter: bool = False,
) -> CAMulator:
    """CAMulator wrapper with identity normalization, synthetic hybrid
    coefficients and a random forcing source on the model grid."""
    # Model (south-to-north) orientation, like the shipped statics
    lat2d, lon2d = np.meshgrid(
        CAMULATOR_GRID_LAT[::-1], CAMULATOR_GRID_LON, indexing="ij"
    )
    return CAMulator(
        core,
        center=torch.zeros(_N_OUT, H, W),
        scale=torch.ones(_N_OUT, H, W),
        forcing_center=torch.zeros(4),
        forcing_scale=torch.ones(4),
        tracer_center=torch.zeros(len(_TRACERS)),
        tracer_scale=torch.ones(len(_TRACERS)),
        statics=torch.zeros(2, H, W),
        hyai=torch.linspace(0.0, 400.0, 33),
        hybi=torch.linspace(0.0, 1.0, 33) ** 2,
        area=grid_area(
            torch.tensor(lat2d, dtype=torch.float32),
            torch.tensor(lon2d, dtype=torch.float32),
        ),
        phis=torch.zeros(H, W),
        forcing_data_source=Random(
            {"lat": CAMULATOR_GRID_LAT, "lon": CAMULATOR_GRID_LON}
        ),
        conservation_fixers=conservation_fixers,
        wind_filter=wind_filter,
    )


def random_input(
    model: CAMulator, time: np.ndarray, device: str
) -> tuple[torch.Tensor, OrderedDict]:
    dc = model.input_coords()
    for key in ("batch", "time", "lead_time", "variable"):
        del dc[key]
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    return fetch_data(Random(dc), time, variable, lead_time, device=device)


def physical_state(device: str) -> torch.Tensor:
    """A plausible physical atmospheric state ``(1, 1, 1, 130, lat, lon)`` in
    Earth2Studio orientation, for exercising the conservation fixers."""
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(1, 1, 1, _N_STATE, H, W)
    x[..., 0:32, :, :] = 3.0 * torch.randn(32, H, W, generator=g)  # u
    x[..., 32:64, :, :] = 3.0 * torch.randn(32, H, W, generator=g)  # v
    x[..., 64:96, :, :] = 250.0 + 30.0 * torch.rand(32, H, W, generator=g)  # t
    x[..., 96:128, :, :] = 1.0e-3 + 1.0e-4 * torch.rand(32, H, W, generator=g)  # qtot
    x[..., 128, :, :] = 1.0e5 + 100.0 * torch.randn(H, W, generator=g)  # sp
    x[..., 129, :, :] = 288.0  # t2m
    return x.to(device)


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2001-01-01T00:00")]),
        np.array(
            [np.datetime64("1999-10-11T12:00"), np.datetime64("2001-06-04T00:00")]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_camulator_call(time, device):
    torch.cuda.empty_cache()
    p = build_model(PhooCamulatorNet()).to(device)
    x, coords = random_input(p, time, device)
    x_ref = x.clone()

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert torch.equal(x, x_ref), "input tensor must not be modified in place"
    assert out.shape == (len(time), 1, _N_OUT, H, W)
    assert torch.isfinite(out).all()
    # Identity stats and a pass-through core: prognostic output equals the input,
    # except that the tracer fixer clips total water at zero
    expected = x.clone()
    expected[:, :, 96:128].clamp_(min=0.0)
    torch.testing.assert_close(out[:, :, :_N_STATE], expected)
    assert (out_coords["variable"] == np.array(OUTPUT_VARIABLES)).all()
    assert (out_coords["time"] == time).all()
    np.testing.assert_array_equal(
        out_coords["lead_time"], coords["lead_time"] + np.timedelta64(6, "h")
    )
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
    assert out_coords["lat"][0] == 90
    assert out_coords["lat"][-1] == -90
    np.testing.assert_array_equal(out_coords["lat"], CAMULATOR_GRID_LAT)
    np.testing.assert_array_equal(out_coords["lon"], CAMULATOR_GRID_LON)


@pytest.mark.parametrize("ensemble", [1, 2])
@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_camulator_iter(ensemble, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2001-01-01T00:00")])
    p = build_model(PhooCamulatorNet()).to(device)
    x, coords = random_input(p, time, device)

    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)
    assert isinstance(p_iter, Iterable)

    # Step 0: initial condition in the output schema, diagnostics NaN-filled
    out, out_coords = next(p_iter)
    assert out.shape == (ensemble, 1, 1, _N_OUT, H, W)
    assert out_coords["lead_time"][0] == np.timedelta64(0, "h")
    assert (out_coords["variable"] == np.array(OUTPUT_VARIABLES)).all()
    torch.testing.assert_close(out[:, :, :, :_N_STATE], x)
    assert torch.isnan(out[:, :, :, _N_STATE:]).all()

    for i, (out, out_coords) in enumerate(p_iter):
        assert out.shape == (ensemble, 1, 1, _N_OUT, H, W)
        assert torch.isfinite(out).all()
        assert (out_coords["variable"] == np.array(OUTPUT_VARIABLES)).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")
        np.testing.assert_array_equal(out_coords["lat"], CAMULATOR_GRID_LAT)
        np.testing.assert_array_equal(out_coords["lon"], CAMULATOR_GRID_LON)
        if i > 1:
            break


@pytest.mark.parametrize(
    "coords",
    [
        # Wrong variable name
        OrderedDict(
            {
                "batch": np.array([0]),
                "time": np.array([np.datetime64("2001-01-01T00:00")]),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["wrong_var"] * _N_STATE),
                "lat": CAMULATOR_GRID_LAT,
                "lon": CAMULATOR_GRID_LON,
            }
        ),
        # Wrong latitude orientation (model exposes north to south)
        OrderedDict(
            {
                "batch": np.array([0]),
                "time": np.array([np.datetime64("2001-01-01T00:00")]),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(PROGNOSTIC_VARIABLES),
                "lat": CAMULATOR_GRID_LAT[::-1].copy(),
                "lon": CAMULATOR_GRID_LON,
            }
        ),
        # More than one input lead time
        OrderedDict(
            {
                "batch": np.array([0]),
                "time": np.array([np.datetime64("2001-01-01T00:00")]),
                "lead_time": np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")]),
                "variable": np.array(PROGNOSTIC_VARIABLES),
                "lat": CAMULATOR_GRID_LAT,
                "lon": CAMULATOR_GRID_LON,
            }
        ),
        # Wrong dimension order (lat/lon swapped)
        OrderedDict(
            {
                "batch": np.array([0]),
                "time": np.array([np.datetime64("2001-01-01T00:00")]),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(PROGNOSTIC_VARIABLES),
                "lon": CAMULATOR_GRID_LON,
                "lat": CAMULATOR_GRID_LAT,
            }
        ),
    ],
)
def test_camulator_exceptions(coords):
    p = build_model(PhooCamulatorNet())
    shape = [len(v) for v in coords.values()]
    x = torch.randn(*shape)
    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_camulator_postprocess(device):
    """Wind filter and conservation fixers stay finite on a physical state and
    the global dry-air mass is conserved by the mass fixer."""
    torch.cuda.empty_cache()
    p = build_model(PhooCamulatorNet(), conservation_fixers=True, wind_filter=True)
    p = p.to(device)
    x = physical_state(device)
    coords = p.input_coords()
    coords["batch"] = np.array([0])
    coords["time"] = np.array([np.datetime64("2001-01-01T00:00")])

    out, out_coords = p(x, coords)
    assert out.shape == (1, 1, 1, _N_OUT, H, W)
    assert torch.isfinite(out).all()

    # Dry-air mass of the output equals that of the input (model orientation)
    def dry_mass(state: torch.Tensor) -> torch.Tensor:
        s = torch.flip(state[0, 0, 0], dims=(-2,))
        sp, q = s[128][None], s[96:128][None]
        return weighted_sum(
            column_integral(1 - q, sp, p.hyai, p.hybi) / GRAVITY, p.area
        )

    torch.testing.assert_close(
        dry_mass(out), dry_mass(x), rtol=1.0e-5, atol=0.0, check_dtype=False
    )


@pytest.mark.parametrize("device", [CUDA])
def test_camulator_net(device):
    """The vendored CrossFormer runs at the CAMulator grid with reduced widths."""
    torch.cuda.empty_cache()
    net = CamulatorNet(
        dim=(8, 16, 32, 64), depth=(1, 1, 1, 1), dim_head=8, use_spectral_norm=False
    ).to(device)
    net.eval()
    x = torch.randn(2, 136, 1, H, W, device=device)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, _N_OUT, 1, H, W)
    assert torch.isfinite(y).all()


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_camulator_package(device):
    torch.cuda.empty_cache()
    p = CAMulator.load_model(CAMulator.load_default_package()).to(device)

    # Random fields are not a valid atmosphere for the real checkpoint; use the
    # normalization center (normalized zero state) as a neutral finite input.
    x = torch.flip(p.center[:_N_STATE], dims=(-2,))[None, None, None]
    coords = p.input_coords()
    coords["batch"] = np.array([0])
    coords["time"] = np.array([np.datetime64("1981-01-01T00:00")])

    out, out_coords = p(x, coords)

    assert out.shape == (1, 1, 1, _N_OUT, H, W)
    assert torch.isfinite(out).all()
    sp = out[0, 0, 0, OUTPUT_VARIABLES.index("sp")]
    assert 40_000 < sp.min() and sp.max() < 110_000
    assert (out_coords["variable"] == np.array(OUTPUT_VARIABLES)).all()
    assert out_coords["lead_time"][0] == np.timedelta64(6, "h")
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
