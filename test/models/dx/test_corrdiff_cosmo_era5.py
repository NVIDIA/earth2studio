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

"""Unit tests for the CorrDiffCosmoEra5 diagnostic wrapper.

These are construction-based (no real package / no GPU forward): they exercise
the metadata-driven plumbing — per-mode parsing, the output->lexicon coord
mapping, the inline-constraint parsing, and crucially that ``set_domain()``
preserves the full per-resolution/per-mode config.
"""

import json
import os
import types
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.dx.corrdiff_cosmo_era5 import CorrDiffCosmoEra5


class PhooNet(torch.nn.Module):
    """Stand-in for the DiT; construction-only (forward not exercised)."""

    def forward(self, x, *args, **kwargs):  # pragma: no cover - construction-only
        return x


class PhooRegDiT(torch.nn.Module):
    """Structural stand-in for the bare upstream regression DiT."""

    def __init__(self, n_out: int):
        super().__init__()
        self.n_out = n_out
        self.tokenizer = types.SimpleNamespace(patch_size=(2, 2))
        self.attn_kwargs_forward: dict = {}
        self.detokenizer = types.SimpleNamespace(h_patches=None, w_patches=None)

    def forward(self, x, t, condition=None):
        return x[:, : self.n_out]


class PhooDiffusionDiT(torch.nn.Module):
    """Stand-in for EDMPreconditioner(ConcatConditionWrapper(DiT))."""

    def __init__(self, n_out: int, gain: float = 0.0):
        super().__init__()
        self.n_out = n_out
        self.gain = gain
        inner = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(patch_size=(2, 2)),
            attn_kwargs_forward={},
            detokenizer=types.SimpleNamespace(h_patches=None, w_patches=None),
        )
        self.model = types.SimpleNamespace(model=inner)

    def forward(self, x, sigma, condition=None):
        return self.gain * x[:, : self.n_out]


OUTPUT_VARIABLES = ["U_10M", "T_2M", "TOT_PRECIP", "CLCT", "ASWDIR_S", "TKE_L40"]
ERA5_VARIABLES = ["u10m", "v10m", "t2m"]
PRE_INV = [
    "sin_lat",
    "cos_lat",
    "sin_lon",
    "cos_lon",
    "elevation_norm",
    "land_fraction",
]
POST_INV = ["z0_lu_norm"]
CHANNEL_TRANSFORMS = {
    "TOT_PRECIP": {"transform": "log_eps", "eps": 1e-5},
    "ASWDIR_S": {"transform": "log_eps", "eps": 1.0},
    "CLCT": {"transform": "logit_eps_percent", "eps": 0.01, "scale": 100.0},
}
CONSTRAINTS = {
    "bounds": {
        "TOT_PRECIP": {"min": 0.0, "max": None},
        "ASWDIR_S": {"min": 0.0, "max": None},
        "CLCT": {"min": 0.0, "max": 100.0, "mode": "clamp"},
        "T_2M": {"min": 0.0, "max": None},
    },
    "sza_gate": {"half_width": 0.05, "channels": {"ASWDIR_S": {"threshold": 0.03}}},
}


def _build(**overrides):
    """Construct a small synthetic CorrDiffCosmoEra5 (mean mode)."""
    n_e, n_o = len(ERA5_VARIABLES), len(OUTPUT_VARIABLES)
    lat_in = np.arange(45.0, 56.0, 1.0, dtype=np.float32)  # 11
    lon_in = np.arange(5.0, 17.0, 1.0, dtype=np.float32)  # 12
    lat2d, lon2d = np.meshgrid(
        np.linspace(47.0, 53.0, 8, dtype=np.float32),
        np.linspace(7.0, 14.0, 8, dtype=np.float32),
        indexing="ij",
    )
    H, W = lat2d.shape
    g = torch.Generator().manual_seed(0)
    static = OrderedDict(
        (n, torch.rand(H, W, generator=g))
        for n in (*PRE_INV, *POST_INV)
        if n not in ("sin_lat", "cos_lat", "sin_lon", "cos_lon")
    )
    kwargs = dict(
        era5_variables=ERA5_VARIABLES,
        output_variables=OUTPUT_VARIABLES,
        regression_model=PhooNet(),
        diffusion_model=None,
        resolution="rea6",
        mode="mean",
        lat_input_grid=torch.tensor(lat_in),
        lon_input_grid=torch.tensor(lon_in),
        lat_output_grid=torch.tensor(lat2d),
        lon_output_grid=torch.tensor(lon2d),
        era5_center=torch.zeros(n_e),
        era5_scale=torch.ones(n_e),
        out_center=torch.zeros(n_o),
        out_scale=torch.ones(n_o),
        static_invariants=static,
        pre_invariant_variables=PRE_INV,
        post_invariant_variables=POST_INV,
        channel_transforms=CHANNEL_TRANSFORMS,
        constraints=CONSTRAINTS,
        number_of_steps=18,
        sigma_max=800.0,
    )
    kwargs.update(overrides)
    return CorrDiffCosmoEra5(**kwargs)


def _build_hub(**overrides):
    """A model set up to derive hub-height wind."""
    ov = ["U_10M", "V_10M", "U_L40", "V_L40", "U_L39", "V_L39", "T_2M"]
    n = len(ov)
    wind_levels = {
        "elevation_invariant": "elevation_norm",
        "levels": [
            {"u": "U_L40", "v": "V_L40", "a": 10.0, "b": 0.0},
            {"u": "U_L39", "v": "V_L39", "a": 35.0, "b": 0.0},
        ],
    }
    kw = dict(
        output_variables=ov,
        out_center=torch.zeros(n),
        out_scale=torch.ones(n),
        channel_transforms={},
        constraints={},
        wind_levels=wind_levels,
        hub_heights=[35.0],
    )
    kw.update(overrides)
    return _build(**kw)


def _diffusion_coords(dx):
    """A single-frame (x, coords) pair for a full ``dx(x, coords)`` call."""
    ic = dx.input_coords()
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([np.datetime64("2021-07-14T12:00")]),
        variable=np.array(ERA5_VARIABLES),
        lat=ic["lat"],
        lon=ic["lon"],
    )
    x = torch.randn(1, 1, len(ERA5_VARIABLES), len(ic["lat"]), len(ic["lon"]))
    return x, coords


# ── Core end-to-end tests ───────────────────────────────────────────────────


def test_corrdiff_cosmo_era5_call():
    """Full ``dx(x, coords)`` for mean mode with a mock DiT, covering the 6-D
    output assembly, batch/time loop, sample-expand, physical-space constraints,
    and the lexicon output_coords contract."""
    dx = _build(regression_model=PhooRegDiT(len(OUTPUT_VARIABLES)), number_of_samples=2)
    ic = dx.input_coords()
    coords = OrderedDict(
        {
            "batch": np.array([0]),
            "time": np.array(
                [
                    np.datetime64("2021-07-14T12:00"),  # day
                    np.datetime64("2021-07-14T00:00"),  # night
                ]
            ),
            "variable": np.array(ERA5_VARIABLES),
            "lat": ic["lat"],
            "lon": ic["lon"],
        }
    )
    x = torch.randn(1, 2, len(ERA5_VARIABLES), len(ic["lat"]), len(ic["lon"]))
    out, out_coords = dx(x, coords)

    H, W = dx.lat_output_numpy.shape
    assert out.shape == (1, 2, 2, len(OUTPUT_VARIABLES), H, W)
    assert torch.isfinite(out).all()
    assert list(out_coords["variable"]) == [
        "u10m",
        "t2m",
        "tp",
        "tcc",
        "aswdir_s",
        "tke_l40",
    ]
    # deterministic mean replicates across the sample-expand axis
    assert torch.allclose(out[0, 0], out[0, 1])
    # constraints applied in physical space (TOT_PRECIP clamped nonneg)
    assert out[0, 0, 0, OUTPUT_VARIABLES.index("TOT_PRECIP")].min() >= -1e-6
    # per-time valid_times indexing: ASWDIR_S solar-gated
    sw = OUTPUT_VARIABLES.index("ASWDIR_S")
    assert out[0, 0, 0, sw].abs().max() > 1e-3
    assert out[0, 0, 1, sw].abs().max() < 1e-3


@pytest.mark.parametrize("number_of_samples", [1, 3])
def test_corrdiff_cosmo_era5_samples(number_of_samples):
    """Generative sample dim + diffusion __call__ plumbing with a mock EDM net."""
    ov = OUTPUT_VARIABLES
    dx = _build(
        mode="diffusion",
        regression_model=None,
        diffusion_model=PhooDiffusionDiT(len(ov), gain=0.5),
        number_of_samples=number_of_samples,
        seed=0,
        channel_transforms={},
        constraints={},
    )
    x, coords = _diffusion_coords(dx)
    out, oc = dx(x, coords)
    H, W = dx.lat_output_numpy.shape
    assert "sample" in oc and len(oc["sample"]) == number_of_samples
    assert out.shape == (1, number_of_samples, 1, len(ov), H, W)
    assert torch.isfinite(out).all()
    if number_of_samples > 1:
        assert not torch.allclose(out[0, 0], out[0, 1])


def test_diffusion_euler_differs_from_heun():
    """``solver="euler"`` runs and produces a different result than ``"heun"``."""
    ov = OUTPUT_VARIABLES

    def _build_solver(solver):
        return _build(
            mode="diffusion",
            regression_model=None,
            diffusion_model=PhooDiffusionDiT(len(ov), gain=0.5),
            number_of_samples=1,
            seed=0,
            solver=solver,
            channel_transforms={},
            constraints={},
        )

    dx_heun = _build_solver("heun")
    dx_euler = _build_solver("euler")
    x, coords = _diffusion_coords(dx_heun)
    out_heun, _ = dx_heun(x, coords)
    out_euler, _ = dx_euler(x, coords)
    H, W = dx_heun.lat_output_numpy.shape
    assert out_euler.shape == (1, 1, 1, len(ov), H, W)
    assert torch.isfinite(out_euler).all()
    assert not torch.allclose(out_heun, out_euler)


def test_call_end_to_end_with_hub():
    """Full ``dx(x, coords)`` with hub heights through the real ``__call__`` path."""
    ov = ["U_10M", "V_10M", "U_L40", "V_L40", "U_L39", "V_L39", "T_2M"]
    dx = _build_hub(regression_model=PhooRegDiT(len(ov)))
    ic = dx.input_coords()
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([np.datetime64("2021-07-14T12:00")]),
        variable=np.array(ERA5_VARIABLES),
        lat=ic["lat"],
        lon=ic["lon"],
    )
    x = torch.randn(1, 1, len(ERA5_VARIABLES), len(ic["lat"]), len(ic["lon"]))
    out, oc = dx(x, coords)
    H, W = dx.lat_output_numpy.shape
    assert out.shape == (1, dx.number_of_samples, 1, len(ov) + 2, H, W)
    assert list(oc["variable"])[-2:] == ["u35m", "v35m"]
    assert torch.isfinite(out).all()


# ── set_domain tests ────────────────────────────────────────────────────────


def test_set_domain_preserves_config():
    """set_domain() must carry through EVERY per-mode/per-resolution field."""
    dx = _build(
        number_of_steps=12,
        sigma_min=0.01,
        sigma_max=500.0,
        rho=5.0,
        solver="euler",
        number_of_samples=3,
        amp=True,
        physical_clamp=False,
        seed=42,
    )
    cr = dx.set_domain(lat_min=48.0, lat_max=52.0, lon_min=8.0, lon_max=13.0)

    assert cr._has_constraints == dx._has_constraints
    assert sorted(cr._bound_lo) == sorted(dx._bound_lo)
    assert cr._bound_up == dx._bound_up
    assert cr._sza_gate == dx._sza_gate
    assert cr._log_eps_idx == dx._log_eps_idx
    assert cr._logit_idx == dx._logit_idx
    assert cr._asinh_idx == dx._asinh_idx
    for attr in (
        "number_of_steps",
        "sigma_min",
        "sigma_max",
        "rho",
        "solver",
        "mode",
        "resolution",
        "number_of_samples",
        "amp",
        "physical_clamp",
        "seed",
    ):
        assert getattr(cr, attr) == getattr(dx, attr), attr
    assert cr.output_variables == dx.output_variables
    assert list(cr._output_coord_variables) == list(dx._output_coord_variables)
    assert cr._output_unit_scale == dx._output_unit_scale
    assert torch.equal(cr.out_center, dx.out_center)
    assert torch.equal(cr.out_scale, dx.out_scale)
    assert cr.regression_model is dx.regression_model
    assert cr.lat_output_numpy.shape[0] <= dx.lat_output_numpy.shape[0]
    assert cr._static_names == dx._static_names
    assert cr.static_invariants.shape[-2:] == cr.lat_output_grid.shape


def test_set_domain_halo_clamps_at_grid_edge():
    """When the halo would extend past the grid edge it is clamped."""
    dx = _build()
    with pytest.warns(UserWarning, match="clamped at the grid edge"):
        cr = dx.set_domain(47.5, 50.0, 9.0, 11.0, halo=2)
    assert cr._halo == (1, 2, 2, 2)


def test_set_domain_snaps_run_grid_to_patch():
    """For a real DiT (patch_size>1) set_domain snaps the RUN grid to a multiple
    of patch_size."""
    dx = _build()
    dx._patch_size = 2
    cr = dx.set_domain(49.5, 51.3, 9.5, 11.5)
    rg = tuple(cr.lat_output_grid.shape)
    assert rg[0] % 2 == 0 and rg[1] % 2 == 0
    top, bot, left, right = cr._halo
    bb = np.asarray(cr.output_coords(cr.input_coords())["lat"]).shape
    assert (bb[0] + top + bot, bb[1] + left + right) == rg


def _attach_extended_grid(
    dx: CorrDiffCosmoEra5, static: np.ndarray | None = None
) -> CorrDiffCosmoEra5:
    lat_e, lon_e = np.meshgrid(
        np.linspace(45.5, 54.5, 12, dtype=np.float32),
        np.linspace(5.5, 15.5, 12, dtype=np.float32),
        indexing="ij",
    )
    dx._ext_lat_numpy = lat_e
    dx._ext_lon_numpy = lon_e
    dx._ext_static_numpy = (
        static
        if static is not None
        else np.zeros((len(dx._static_names), 12, 12), dtype=np.float32)
    )
    return dx


def test_set_domain_extended_footprint():
    """set_domain uses extended grid for OOD margins."""
    inside = _attach_extended_grid(_build())
    sub = inside.set_domain(49.0, 51.0, 9.0, 11.0)
    assert isinstance(sub, CorrDiffCosmoEra5)
    assert sub.lat_output_grid.shape[0] < inside.lat_output_grid.shape[0]
    margin = _attach_extended_grid(_build())
    sub2 = margin.set_domain(53.3, 54.0, 9.0, 11.0)
    assert isinstance(sub2, CorrDiffCosmoEra5)
    assert sub2.lat_output_numpy.max() > 53.0


def test_set_domain_extended_state_guards():
    """Extended grid state guards fire when arrays are incomplete."""
    dx = _attach_extended_grid(_build())
    dx._ext_lon_numpy = None
    with pytest.raises(RuntimeError, match="extended grid arrays are not set"):
        dx.set_domain(49.0, 51.0, 9.0, 11.0)
    dx2 = _attach_extended_grid(_build())
    dx2._ext_static_numpy = None
    with pytest.raises(RuntimeError, match="extended static array is not set"):
        dx2.set_domain(49.0, 51.0, 9.0, 11.0)


def test_forward_core_halo_trims_border():
    """Halo-expanded run grid is trimmed back to the bbox in the output."""
    dx = _build(regression_model=PhooRegDiT(len(OUTPUT_VARIABLES)))
    plain = dx.set_domain(48.0, 52.0, 8.0, 13.0, halo=0)
    haloed = dx.set_domain(48.0, 52.0, 8.0, 13.0, halo=1)
    assert haloed._halo != (0, 0, 0, 0)
    Hp, Wp = plain.lat_output_numpy.shape

    ic = haloed.input_coords()
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([np.datetime64("2021-07-14T12:00")]),
        variable=np.array(ERA5_VARIABLES),
        lat=ic["lat"],
        lon=ic["lon"],
    )
    x = torch.randn(1, 1, len(ERA5_VARIABLES), len(ic["lat"]), len(ic["lon"]))
    out, _ = haloed(x, coords)
    assert out.shape[-2:] == (Hp, Wp)
    assert torch.isfinite(out).all()


# ── Transform / postprocess tests ──────────────────────────────────────────


def test_transform_round_trip_incl_asinh():
    """postprocess_output inverts each channel transform correctly."""
    import math

    ov = ["CH_LOG", "CH_LOGIT", "CH_ASINH"]
    n = len(ov)
    ct = {
        "CH_LOG": {"transform": "log_eps", "eps": 1.0},
        "CH_LOGIT": {"transform": "logit_eps_percent", "eps": 0.01, "scale": 100.0},
        "CH_ASINH": {"transform": "asinh", "eps": 2.0},
    }
    dx = _build(
        output_variables=ov,
        out_center=torch.tensor([0.5, -0.2, -0.4]),
        out_scale=torch.tensor([2.0, 1.5, 0.5]),
        channel_transforms=ct,
        constraints={},
        physical_clamp=True,
    )
    H, W = dx.lat_output_numpy.shape
    x = torch.zeros(1, n, H, W)
    x[0, 0] = 0.7
    x[0, 1] = 0.3
    x[0, 2] = -0.5
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    exp_log = 1.0 * (math.exp(0.7 * 2.0 + 0.5) - 1)
    exp_logit = ((1 / (1 + math.exp(-(0.3 * 1.5 - 0.2))) - 0.01) / 0.98) * 100.0
    exp_asinh = 2.0 * math.sinh(-0.5 * 0.5 - 0.4)
    assert torch.allclose(out[0, 0], torch.full((H, W), exp_log), atol=1e-4)
    assert torch.allclose(out[0, 1], torch.full((H, W), exp_logit), atol=1e-4)
    assert torch.allclose(out[0, 2], torch.full((H, W), exp_asinh), atol=1e-4)


def test_postprocess_physical_clamp_disabled():
    """physical_clamp=False skips bounds and solar gate."""
    dx = _build(physical_clamp=False)
    ov = OUTPUT_VARIABLES
    H, W = dx.lat_output_numpy.shape
    lat2d, lon2d = dx.lat_output_numpy, dx.lon_output_numpy
    x = torch.zeros(1, len(ov), H, W)
    x[0, ov.index("TOT_PRECIP")] = -3.0
    x[0, ov.index("ASWDIR_S")] = 2.0
    out = dx.postprocess_output(x, datetime(2021, 1, 1, 0), lat2d, lon2d)[0].numpy()
    assert np.isfinite(out).all()
    assert out[ov.index("TOT_PRECIP")].min() < 0.0
    assert out[ov.index("ASWDIR_S")].max() > 1e-3


def test_nonfinite_input_warns():
    """A NaN in the ERA5 input is flagged; check_inputs=False suppresses it."""
    dx = _build()
    era5 = torch.zeros(
        len(ERA5_VARIABLES), len(dx.lat_input_numpy), len(dx.lon_input_numpy)
    )
    era5[0] = float("nan")
    with pytest.warns(UserWarning, match="non-finite"):
        dx.preprocess_input(
            era5, datetime(2021, 7, 14, 12), dx.lat_output_grid, dx.lon_output_grid
        )
    dx.check_inputs = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dx.preprocess_input(
            era5, datetime(2021, 7, 14, 12), dx.lat_output_grid, dx.lon_output_grid
        )
    assert not any("non-finite" in str(w.message) for w in caught)


# ── Hub-height wind tests ───────────────────────────────────────────────────


def test_hub_height_elevation_dependent_heights():
    """b_L != 0: per-level heights track the elevation invariant."""
    ov = ["U_10M", "V_10M", "U_L40", "V_L40", "U_L39", "V_L39", "T_2M"]
    n, H, W = len(ov), 8, 8
    static = OrderedDict(
        elevation_norm=torch.full((H, W), 2.0),
        land_fraction=torch.zeros(H, W),
        z0_lu_norm=torch.zeros(H, W),
    )
    wind_levels = {
        "elevation_invariant": "elevation_norm",
        "levels": [
            {"u": "U_L40", "v": "V_L40", "a": 10.0, "b": 0.0},
            {"u": "U_L39", "v": "V_L39", "a": 30.0, "b": 2.5},  # 30 + 2.5*2 = 35
        ],
    }
    dx = _build(
        output_variables=ov,
        out_center=torch.zeros(n),
        out_scale=torch.ones(n),
        channel_transforms={},
        constraints={},
        wind_levels=wind_levels,
        hub_heights=[22.5],
        static_invariants=static,
    )
    x = torch.zeros(1, n, H, W)
    x[0, ov.index("U_L39")] = 20.0
    x[0, ov.index("V_L39")] = 40.0
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    # w = (22.5-10)/(35-10) = 0.5
    assert torch.allclose(out[0, n], torch.full((H, W), 10.0), atol=1e-4)
    assert torch.allclose(out[0, n + 1], torch.full((H, W), 20.0), atol=1e-4)


def test_hub_interp_log_method():
    """The 'log' interpolation works in ln(height)."""
    target = float((10.0 * 35.0) ** 0.5)  # ~18.71 m, geometric midpoint
    dx = _build_hub(hub_interp="log", hub_heights=[target])
    ov = dx.output_variables
    H, W = dx.lat_output_numpy.shape
    x = torch.zeros(1, len(ov), H, W)
    x[0, ov.index("U_L40")] = 0.0
    x[0, ov.index("U_L39")] = 10.0
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    assert torch.allclose(out[0, len(ov)], torch.full((H, W), 5.0), atol=1e-3)


# ── Constructor / validation tests ─────────────────────────────────────────


def test_construct_rejects_invalid_mode_model_and_resolution():
    """Invalid mode/model/resolution combos rejected at construction."""
    with pytest.raises(ValueError, match="mode='mean' requires regression_model"):
        _build(mode="mean", regression_model=None)
    with pytest.raises(ValueError, match="mode='diffusion' requires diffusion_model"):
        _build(mode="diffusion", regression_model=None, diffusion_model=None)
    with pytest.raises(ValueError, match="mode must be 'mean' or 'diffusion'"):
        _build(mode="both")
    with pytest.raises(ValueError, match="resolution must be one of"):
        _build(resolution="rea3")


def test_constraint_parsing_ignores_unknown_and_scalar_gate():
    """Unknown channels/non-dict bounds skipped; bare scalar gate gets default hw."""
    cons = {
        "bounds": {
            "NOT_A_CHANNEL": {"min": 0.0},
            "T_2M": "not-a-dict",
        },
        "sza_gate": {
            "channels": {
                "NOT_SHORTWAVE": {"threshold": 0.02},
                "ASWDIR_S": 0.04,
            }
        },
    }
    dx = _build(constraints=cons)
    idx = OUTPUT_VARIABLES.index("ASWDIR_S")
    assert (idx, 0.04, 0.05) in dx._sza_gate
    assert dx._bound_lo == [] and dx._bound_up == []


def test_forward_methods_require_loaded_model():
    """Calling the wrong forward for the mode raises clearly."""
    diff = _build(
        mode="diffusion",
        regression_model=None,
        diffusion_model=PhooDiffusionDiT(len(OUTPUT_VARIABLES)),
    )
    with pytest.raises(RuntimeError, match="regression_model is not loaded"):
        diff._regression_forward(torch.zeros(1, len(ERA5_VARIABLES), 8, 8))
    mean = _build()
    with pytest.raises(RuntimeError, match="diffusion_model is not loaded"):
        mean._denoise(torch.zeros(1, len(ERA5_VARIABLES), 8, 8), None)


def test_corrdiff_cosmo_era5_exceptions():
    """The public dx(x, coords) rejects invalid input coords."""
    dx = _build(regression_model=PhooRegDiT(len(OUTPUT_VARIABLES)))
    ic = dx.input_coords()
    x = torch.randn(1, 1, len(ERA5_VARIABLES), len(ic["lat"]), len(ic["lon"]))
    t = np.array([np.datetime64("2021-07-14T12:00")])
    bad_var = OrderedDict(
        batch=np.array([0]),
        time=t,
        variable=np.array(["bogus"] * len(ERA5_VARIABLES)),
        lat=ic["lat"],
        lon=ic["lon"],
    )
    with pytest.raises((KeyError, ValueError), match="required dim variable"):
        dx(x, bad_var)
    bad_order = OrderedDict(
        batch=np.array([0]),
        time=t,
        variable=np.array(ERA5_VARIABLES),
        lon=ic["lon"],
        lat=ic["lat"],
    )
    with pytest.raises((KeyError, ValueError), match="index -2"):
        dx(x, bad_order)


# ── load_model assembly tests (synthetic local package) ─────────────────────

_INV_NAMES = ["elevation_norm", "land_fraction", "z0_lu_norm"]


def _write_grids(res_dir: Path) -> tuple[int, int]:
    lat_in = np.arange(45.0, 56.0, 1.0, dtype=np.float32)
    lon_in = np.arange(5.0, 17.0, 1.0, dtype=np.float32)
    lat2d, lon2d = np.meshgrid(
        np.linspace(47.0, 53.0, 8, dtype=np.float32),
        np.linspace(7.0, 14.0, 8, dtype=np.float32),
        indexing="ij",
    )
    xr.Dataset(
        {
            "lat_input": (("lat_in",), lat_in),
            "lon_input": (("lon_in",), lon_in),
            "lat_output": (("y", "x"), lat2d),
            "lon_output": (("y", "x"), lon2d),
        }
    ).to_netcdf(res_dir / "grids.nc")
    return lat2d.shape


def _base_metadata() -> dict[str, Any]:
    return {
        "era5_variables": ERA5_VARIABLES,
        "output_variables": OUTPUT_VARIABLES,
        "pre_invariant_variables": PRE_INV,
        "post_invariant_variables": POST_INV,
        "channel_transforms": CHANNEL_TRANSFORMS,
        "constraints": CONSTRAINTS,
        "number_of_samples": 2,
        "sampler": {
            "num_steps": 20,
            "sigma_min": 0.001,
            "sigma_max": 700.0,
            "rho": 8.0,
            "solver": "euler",
        },
        "checkpoints": {
            "rea6_mean": "ckpt_mean.mdlus",
            "rea6_diffusion": "ckpt_diff.mdlus",
        },
        "regression": {"patch_size": 2, "attn_kernel_size": 3},
        "diffusion": {"patch_size": 4, "attn_kernel_size": 2},
    }


def _write_stats(
    res_dir: Path,
    *,
    drop_output: str | None = None,
    bad_std: bool = False,
    alt_output_key: str | None = None,
) -> None:
    era5 = {v: {"mean": 0.0, "std": 1.0} for v in ERA5_VARIABLES}
    out = {v: {"mean": 0.0, "std": 1.0} for v in OUTPUT_VARIABLES}
    if drop_output is not None:
        del out[drop_output]
    if bad_std:
        out[OUTPUT_VARIABLES[0]]["std"] = 0.0
    payload = {"era5": era5, "output": out}
    if alt_output_key is not None:
        payload[alt_output_key] = {
            v: {"mean": 5.0, "std": 2.0} for v in OUTPUT_VARIABLES
        }
    (res_dir / "stats.json").write_text(json.dumps(payload))


def _write_fallback_invariants(res_dir: Path, shape: tuple[int, int]) -> None:
    H, W = shape
    g = np.random.default_rng(0)
    xr.Dataset(
        {
            n: (("y", "x"), g.standard_normal((H, W)).astype(np.float32))
            for n in _INV_NAMES
        }
    ).to_netcdf(res_dir / "invariants.nc")


def _write_extended_invariants(
    res_dir: Path, *, bad_std: bool = False
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    lat_e, lon_e = np.meshgrid(
        np.linspace(45.5, 54.5, 12, dtype=np.float32),
        np.linspace(5.5, 15.5, 12, dtype=np.float32),
        indexing="ij",
    )
    g = np.random.default_rng(1)
    raw = {
        "elevation": g.standard_normal((12, 12)).astype(np.float32),
        "land_fraction": g.random((12, 12)).astype(np.float32),
        "z0_lu": g.random((12, 12)).astype(np.float32),
    }
    data = {
        "lat": (("y", "x"), lat_e),
        "lon": (("y", "x"), lon_e),
        **{name: (("y", "x"), arr) for name, arr in raw.items()},
    }
    xr.Dataset(data).to_netcdf(res_dir / "invariants_ext.nc")
    norm = {
        "channels": {
            "elevation": {"method": "zscore", "mean": 100.0, "std": 50.0},
            "z0_lu": {
                "method": "zscore",
                "mean": 0.1,
                "std": 0.0 if bad_std else 0.05,
            },
        }
    }
    (res_dir / "invariant_norm.json").write_text(json.dumps(norm))
    inv_meta = {
        "file": "invariants_ext.nc",
        "native_offset": [2, 2],
        "native_shape": [8, 8],
        "norm_stats_file": "invariant_norm.json",
    }
    return inv_meta, raw


def _make_package(
    tmp_path: Path,
    metadata: dict[str, Any],
    *,
    drop_output: str | None = None,
    bad_std: bool = False,
    extended: bool = False,
    alt_output_key: str | None = None,
) -> Package:
    res_dir = tmp_path / "rea6"
    res_dir.mkdir(parents=True, exist_ok=True)
    shape = _write_grids(res_dir)
    (res_dir / "metadata.json").write_text(json.dumps(metadata))
    _write_stats(
        res_dir, drop_output=drop_output, bad_std=bad_std, alt_output_key=alt_output_key
    )
    if not extended:
        _write_fallback_invariants(res_dir, shape)
    (res_dir / "ckpt_mean.mdlus").write_bytes(b"")
    (res_dir / "ckpt_diff.mdlus").write_bytes(b"")
    return Package(str(tmp_path))


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.EDMPreconditioner")
@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_assembles_mean(mock_dit, mock_edm, tmp_path):
    """load_model assembles a mean-mode model from a synthetic package."""
    mock_dit.from_checkpoint.return_value = PhooRegDiT(len(OUTPUT_VARIABLES))
    pkg = _make_package(tmp_path, _base_metadata())

    dx = CorrDiffCosmoEra5.load_model(pkg, device="cpu", mode="mean", resolution="rea6")

    assert isinstance(dx, CorrDiffCosmoEra5)
    assert dx.output_variables == OUTPUT_VARIABLES
    assert dx.regression_model is not None and dx.diffusion_model is None
    mock_dit.from_checkpoint.assert_called_once()
    mock_edm.from_checkpoint.assert_not_called()
    assert list(dx._output_coord_variables) == [
        "u10m",
        "t2m",
        "tp",
        "tcc",
        "aswdir_s",
        "tke_l40",
    ]
    assert dx.number_of_samples == 2
    assert dx.number_of_steps == 20 and dx.solver == "euler"
    assert dx._bound_lo and dx._bound_up and dx._sza_gate
    assert dx._patch_size == 2 and dx._min_domain_cells == 6
    assert dx._static_names == _INV_NAMES
    H, W = dx.lat_output_numpy.shape
    assert dx.static_invariants.shape == (len(_INV_NAMES), H, W)
    assert dx._ext_static_numpy is None


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.EDMPreconditioner")
@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_assembles_diffusion(mock_dit, mock_edm, tmp_path):
    """load_model assembles a diffusion-mode model from a synthetic package."""
    mock_edm.from_checkpoint.return_value = PhooDiffusionDiT(len(OUTPUT_VARIABLES))
    pkg = _make_package(tmp_path, _base_metadata())

    dx = CorrDiffCosmoEra5.load_model(pkg, mode="diffusion", resolution="rea6")

    assert isinstance(dx, CorrDiffCosmoEra5)
    assert dx.mode == "diffusion"
    assert dx.diffusion_model is not None and dx.regression_model is None
    mock_edm.from_checkpoint.assert_called_once()
    mock_dit.from_checkpoint.assert_not_called()
    assert dx._patch_size == 4 and dx._min_domain_cells == 8


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_per_mode_stats_and_transforms_override(mock_dit, tmp_path):
    """Per-mode stats and channel_transforms overrides are honoured."""
    mock_dit.from_checkpoint.return_value = PhooRegDiT(len(OUTPUT_VARIABLES))
    meta = _base_metadata()
    meta["modes"] = {"mean": {"stats": "output_mean", "channel_transforms": False}}
    pkg = _make_package(tmp_path, meta, alt_output_key="output_mean")

    dx = CorrDiffCosmoEra5.load_model(pkg, mode="mean", resolution="rea6")

    n = len(OUTPUT_VARIABLES)
    assert torch.allclose(dx.out_center, torch.full((n,), 5.0))
    assert torch.allclose(dx.out_scale, torch.full((n,), 2.0))
    assert dx._log_eps_idx == []
    assert dx._logit_idx == []


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.EDMPreconditioner")
@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_extended_invariants(mock_dit, mock_edm, tmp_path):
    """Extended invariants are loaded, z-scored, and sliced correctly."""
    mock_dit.from_checkpoint.return_value = PhooRegDiT(len(OUTPUT_VARIABLES))
    meta = _base_metadata()
    res_dir = tmp_path / "rea6"
    res_dir.mkdir(parents=True, exist_ok=True)
    meta["invariants"], raw = _write_extended_invariants(res_dir)
    pkg = _make_package(tmp_path, meta, extended=True)

    dx = CorrDiffCosmoEra5.load_model(pkg, mode="mean", resolution="rea6")

    assert dx._static_names == _INV_NAMES
    assert dx.static_invariants.shape == (len(_INV_NAMES), 8, 8)
    assert dx._ext_static_numpy is not None
    assert dx._ext_static_numpy.shape == (len(_INV_NAMES), 12, 12)
    ke = _INV_NAMES.index("elevation_norm")
    exp_elev = (raw["elevation"][2:10, 2:10] - 100.0) / 50.0
    assert np.allclose(dx.static_invariants[ke].cpu().numpy(), exp_elev)
    kl = _INV_NAMES.index("land_fraction")
    exp_land = raw["land_fraction"][2:10, 2:10]
    assert np.allclose(dx.static_invariants[kl].cpu().numpy(), exp_land)


# ── load_model error handling ───────────────────────────────────────────────


def test_load_model_invalid_selectors_rejected():
    """load_model validates resolution/mode up front."""
    with pytest.raises(ValueError, match="resolution must be one of"):
        CorrDiffCosmoEra5.load_model(None, resolution="rea3")
    with pytest.raises(ValueError, match="mode must be"):
        CorrDiffCosmoEra5.load_model(None, mode="both")


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_missing_checkpoint_entry_rejected(mock_dit, tmp_path):
    meta = _base_metadata()
    del meta["checkpoints"]["rea6_mean"]
    pkg = _make_package(tmp_path, meta)
    with pytest.raises(ValueError, match="no checkpoint entry 'rea6_mean'"):
        CorrDiffCosmoEra5.load_model(pkg, mode="mean", resolution="rea6")


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_from_checkpoint_failure_rejected(mock_dit, tmp_path):
    mock_dit.from_checkpoint.side_effect = RuntimeError("corrupt")
    pkg = _make_package(tmp_path, _base_metadata())
    with pytest.raises(ValueError, match="could not load the regression network"):
        CorrDiffCosmoEra5.load_model(pkg, mode="mean", resolution="rea6")


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_empty_metadata_rejected(mock_dit, tmp_path):
    res_dir = tmp_path / "rea6"
    res_dir.mkdir(parents=True, exist_ok=True)
    _write_grids(res_dir)
    (res_dir / "metadata.json").write_text("   ")
    with pytest.raises(ValueError, match="metadata.json is empty"):
        CorrDiffCosmoEra5.load_model(Package(str(tmp_path)), resolution="rea6")


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_modes_block_missing_mode_rejected(mock_dit, tmp_path):
    meta = _base_metadata()
    meta["modes"] = {"diffusion": {"stats": "output"}}
    pkg = _make_package(tmp_path, meta)
    with pytest.raises(ValueError, match="no entry for mode='mean'"):
        CorrDiffCosmoEra5.load_model(pkg, mode="mean", resolution="rea6")


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_missing_stats_rejected(mock_dit, tmp_path):
    mock_dit.from_checkpoint.return_value = PhooRegDiT(len(OUTPUT_VARIABLES))
    pkg = _make_package(tmp_path, _base_metadata(), drop_output=OUTPUT_VARIABLES[0])
    with pytest.raises(ValueError, match="missing normalization stats"):
        CorrDiffCosmoEra5.load_model(pkg, mode="mean", resolution="rea6")


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_invalid_stats_rejected(mock_dit, tmp_path):
    mock_dit.from_checkpoint.return_value = PhooRegDiT(len(OUTPUT_VARIABLES))
    pkg = _make_package(tmp_path, _base_metadata(), bad_std=True)
    with pytest.raises(ValueError, match="normalization stats are invalid"):
        CorrDiffCosmoEra5.load_model(pkg, mode="mean", resolution="rea6")


@patch("earth2studio.models.dx.corrdiff_cosmo_era5.DiT")
def test_load_model_invalid_invariant_zscore_rejected(mock_dit, tmp_path):
    mock_dit.from_checkpoint.return_value = PhooRegDiT(len(OUTPUT_VARIABLES))
    meta = _base_metadata()
    res_dir = tmp_path / "rea6"
    res_dir.mkdir(parents=True, exist_ok=True)
    meta["invariants"], _ = _write_extended_invariants(res_dir, bad_std=True)
    pkg = _make_package(tmp_path, meta, extended=True)
    with pytest.raises(ValueError, match="invalid z-score stats"):
        CorrDiffCosmoEra5.load_model(pkg, mode="mean", resolution="rea6")


# ── Package integration tests (real weights, GPU) ───────────────────────────


@pytest.mark.package
@pytest.mark.parametrize("resolution", ["rea6", "rea2"])
@pytest.mark.parametrize("mode", ["mean", "diffusion"])
def test_corrdiff_cosmo_era5_package(mode, resolution):
    """Load-and-run on a real local package (set ``$COSMO_REA_PACKAGE``)."""
    pkg_path = os.environ.get("COSMO_REA_PACKAGE")
    if not pkg_path:
        pytest.skip("set COSMO_REA_PACKAGE to a built package dir to run")

    device = "cuda:0" if torch.cuda.is_available() else None
    dx = CorrDiffCosmoEra5.load_model(
        Package(pkg_path), device=device, mode=mode, resolution=resolution
    )
    assert len(dx.output_variables) > 0
    if dx._constraints.get("bounds"):
        assert dx._bound_lo or dx._bound_up
    if dx._constraints.get("sza_gate"):
        assert dx._sza_gate
    if device is None:
        pytest.skip("NATTEN forward requires CUDA; checkpoint load verified on CPU")

    ic = dx.input_coords()
    torch.manual_seed(0)
    coords = OrderedDict(
        {
            "batch": np.array([0]),
            "time": np.array([np.datetime64("2021-07-14T12:00")]),
            "variable": ic["variable"],
            "lat": ic["lat"],
            "lon": ic["lon"],
        }
    )
    x = torch.randn(
        1, 1, len(ic["variable"]), len(ic["lat"]), len(ic["lon"]), device=device
    )
    out, oc = dx(x, coords)
    H, W = dx.lat_output_numpy.shape
    assert out.shape == (
        1,
        dx.number_of_samples,
        1,
        len(dx._output_coord_variables),
        H,
        W,
    )
    assert list(oc["variable"]) == list(dx._output_coord_variables)
    assert torch.isfinite(out).all()
    bounds = dx._constraints.get("bounds") or {}
    for ch, b in bounds.items():
        if ch not in dx.output_variables:
            continue
        i = dx.output_variables.index(ch)
        sl = out[0, :, 0, i]
        if b.get("min") is not None:
            assert sl.min().item() >= b["min"] - 1e-4, f"{ch} below min in {mode}"
        if b.get("max") is not None:
            assert sl.max().item() <= b["max"] + 1e-4, f"{ch} above max in {mode}"
