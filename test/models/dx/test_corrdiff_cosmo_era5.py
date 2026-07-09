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
preserves the full per-resolution/per-mode config: it re-lists the constructor
args by hand, so a field can be silently dropped if one is missed.
"""

import inspect
import math
import os
import types
import warnings
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pytest
import torch

from earth2studio.lexicon import CosmoLexicon
from earth2studio.models.dx.corrdiff_cosmo_era5 import CorrDiffCosmoEra5

# The real (un-mocked) DiT (diffusion transformer) needs upstream physicsnemo's
# natten2d_rope (PR #1731). On an older physicsnemo this symbol is absent -> the
# real-weight package tests below are skipped (the fast suite otherwise mocks the
# nets).
# TODO(cosmo): #1731 isn't in a physicsnemo release yet. Remove this skip (run the
# package tests unconditionally) once the physicsnemo pin is bumped to a release
# containing #1731.
try:
    from physicsnemo.nn.module.dit_layers import (  # noqa: F401
        RopeNatten2DSelfAttention,
    )

    _UPSTREAM_ROPE = True
except ImportError:
    _UPSTREAM_ROPE = False


class _MockNet(torch.nn.Module):
    """Stand-in for the DiT; construction-only (forward not exercised)."""

    def forward(self, x, *args, **kwargs):  # pragma: no cover - construction-only
        return x


class _MockRegDiT(torch.nn.Module):
    """Structural stand-in for the bare upstream regression DiT: exercises the
    _regression_forward path (latent rebind + single forward, constant t=0) without
    pulling in physicsnemo. The rebind targets live directly on the module (the
    regression model IS the DiT), and forward takes ``(x, t, condition)``."""

    def __init__(self, n_out: int):
        super().__init__()
        self.n_out = n_out
        self.tokenizer = types.SimpleNamespace(patch_size=(2, 2))
        self.attn_kwargs_forward: dict = {}
        self.detokenizer = types.SimpleNamespace(h_patches=None, w_patches=None)

    def forward(self, x, t, condition=None):
        return x[:, : self.n_out]


class _MockDiffusionDiT(torch.nn.Module):
    """Stand-in for EDMPreconditioner(ConcatConditionWrapper(DiT)): exposes the
    ``.model.model`` rebind target and a ``(x, sigma, condition)`` forward so the
    EDM/Karras Heun sampler in ``_denoise`` runs without physicsnemo/GPU.

    The denoised estimate is ``gain * x`` (the denoiser output ``D_x = gain*x``).
    With ``gain=0`` the
    EDM ODE telescopes to ``t_n = 0`` so the result is ~0 regardless of the seed,
    which pins the Karras schedule's telescoping to zero and exercises the Heun
    corrector path (with ``D_x=0`` the corrector slope equals the Euler slope, so
    this runs that path rather than verifying it). With
    ``gain != 0`` the result is a schedule-weighted multiple of the per-sample
    noise, so ensemble members differ (plumbing test). ``gain=1`` would make the
    ODE a no-op, hiding schedule bugs, so it is avoided."""

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


# Interior output names: a mix of lexicon-overlap (U_10M, T_2M), nonlinear
# transforms (TOT_PRECIP log_eps, CLCT logit), a shortwave channel that is both
# log_eps-transformed AND solar-zenith-angle (SZA)-gated (ASWDIR_S, exercising
# their interaction),
# and a COSMO-only channel kept raw (TKE_L40).
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
    # threshold (0.03) deliberately != the default half_width (0.05) so the parsed
    # gate tuple guards against a threshold/half-width swap.
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
    g = torch.Generator().manual_seed(0)  # deterministic synthetic invariants
    static = OrderedDict(
        (n, torch.rand(H, W, generator=g))
        for n in (*PRE_INV, *POST_INV)
        if n not in ("sin_lat", "cos_lat", "sin_lon", "cos_lon")
    )
    kwargs = dict(
        era5_variables=ERA5_VARIABLES,
        output_variables=OUTPUT_VARIABLES,
        regression_model=_MockNet(),
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


def test_output_coords_lexicon_mapping():
    """Overlap outputs map to Earth2Studio lexicon names at output_coords;
    COSMO-only stay raw; internal output_variables keep interior names; order is
    preserved."""
    dx = _build()
    coord = list(dx._output_coord_variables)
    # overlap vars -> canonical lexicon; COSMO-only -> lowercased interior name
    assert coord == ["u10m", "t2m", "tot_precip", "tcc", "aswdir_s", "tke_l40"]
    # internal indexing is unchanged (interior names)
    assert dx.output_variables == OUTPUT_VARIABLES
    # mapping comes from CosmoLexicon (name + unit scale)
    assert CosmoLexicon.to_e2studio("TD_2M") == ("d2m", 1.0)  # same-units canonical
    assert CosmoLexicon.to_e2studio("CLCT") == ("tcc", 0.01)  # % -> 0-1 fraction
    # both resolution spellings of the same field resolve to one E2S name
    assert CosmoLexicon.to_e2studio("U_10M")[0] == "u10m"  # REA6 spelling
    assert CosmoLexicon.to_e2studio("10U")[0] == "u10m"  # REA2 spelling
    # CLCT carries a % -> fraction value rescale at the right channel index
    assert dx._output_unit_scale == [(OUTPUT_VARIABLES.index("CLCT"), 0.01)]


def test_output_coords_requires_native_grid():
    """output_coords accepts ONLY the native ERA5 input grid. An input grid that
    differs from it -- either in VALUES (shifted off the footprint) or in SHAPE
    (different length) -- is refused with a clear error; for a sub-region the user
    must go through set_domain(), not pass a custom grid. Covers both branches of
    the _is_native_input check (value via allclose, shape)."""
    dx = _build()
    dx.output_coords(dx.input_coords())  # native -> ok

    # value mismatch: right shape, wrong values (shifted off the native footprint)
    shifted = OrderedDict(dx.input_coords())
    shifted["lat"] = np.asarray(shifted["lat"]) + 10.0
    with pytest.raises(ValueError, match="native input grid"):
        dx.output_coords(shifted)

    # shape mismatch: a different-length lat (still passes the dim handshakes, so
    # the failure is provably the native-grid shape check, not a handshake)
    wrong_shape = OrderedDict(dx.input_coords())
    wrong_shape["lat"] = np.asarray(wrong_shape["lat"])[:-1]
    with pytest.raises(ValueError, match="native input grid"):
        dx.output_coords(wrong_shape)


def test_constraint_parsing():
    """Constraints metadata parses into per-channel index lists: each min/max bound
    and the solar gate land at the correct output-channel index, a channel with a
    null max gets no upper-bound entry, and the gate stores (threshold, half_width)."""
    dx = _build()
    idx = {c: i for i, c in enumerate(OUTPUT_VARIABLES)}
    assert dx._has_constraints
    assert set(dx._bound_lo) == {
        (idx["TOT_PRECIP"], 0.0),
        (idx["ASWDIR_S"], 0.0),
        (idx["CLCT"], 0.0),
        (idx["T_2M"], 0.0),
    }
    assert dx._bound_up == [(idx["CLCT"], 100.0)]  # only CLCT has a max
    # (idx, threshold, half_width): distinct values (0.03 != 0.05) catch a swap
    assert dx._sza_gate == [(idx["ASWDIR_S"], 0.03, 0.05)]


def test_sigmoid_bound_rejected():
    """A two-sided 'sigmoid' bound only gives the right answer when applied in the
    model's normalized space at training time; applying it after de-norm (as
    inference post-processing) gives a different, wrong result, so it is rejected.
    A min/max 'clamp' is different: de-norm just rescales values and preserves
    their order, so clamping to the physical bounds in postprocess gives exactly
    the same result as the equivalent training-time clamp -- hence it is allowed."""
    bad = {"bounds": {"CLCT": {"min": 0.0, "max": 100.0, "mode": "sigmoid"}}}
    with pytest.raises(NotImplementedError, match="sigmoid"):
        _build(constraints=bad)


@pytest.mark.parametrize("hw", [0.0, -0.1])
def test_nonpositive_half_width_rejected(hw):
    """The solar-gate ramp half_width must be strictly positive: both 0 and a
    negative value are rejected at construction (the gate is ramp-only)."""
    bad = {
        "sza_gate": {"channels": {"ASWDIR_S": {"threshold": 0.05, "half_width": hw}}}
    }
    with pytest.raises(ValueError, match="half_width must be > 0"):
        _build(constraints=bad)


def test_ungated_shortwave_channel_rejected():
    """Once a solar gate is configured, every shortwave output must be covered by
    the gate; a shortwave channel left out is rejected at construction, rather
    than silently running without a night-time gate."""
    ov = ["ASWDIR_S", "ASWDIFD_S", "T_2M"]
    bad = {
        "sza_gate": {"channels": {"ASWDIR_S": {"threshold": 0.05}}}
    }  # misses ASWDIFD_S
    with pytest.raises(ValueError, match="must cover all shortwave outputs"):
        _build(
            output_variables=ov,
            out_center=torch.zeros(3),
            out_scale=torch.ones(3),
            channel_transforms={},
            constraints=bad,
        )


def test_set_domain_preserves_config():
    """set_domain() must carry through EVERY per-mode/per-resolution field: it
    re-lists the constructor args by hand, so a missed one would be silently
    dropped. Only the footprint changes."""
    # Non-default values on purpose: if these were left at the constructor
    # defaults, a field that set_domain forgot to pass would still come out as
    # that default (matching dx), so the loop below couldn't tell it was dropped.
    # Distinct values make a dropped field observable.
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

    # constraints + their parsed indices
    assert cr._has_constraints == dx._has_constraints
    assert sorted(cr._bound_lo) == sorted(dx._bound_lo)
    assert cr._bound_up == dx._bound_up
    assert cr._sza_gate == dx._sza_gate
    # transforms
    assert cr._log_eps_idx == dx._log_eps_idx
    assert cr._logit_idx == dx._logit_idx
    assert cr._asinh_idx == dx._asinh_idx
    # sampler + mode/resolution
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
    # output identity + lexicon coords
    assert cr.output_variables == dx.output_variables
    assert list(cr._output_coord_variables) == list(dx._output_coord_variables)
    assert cr._output_unit_scale == dx._output_unit_scale
    # normalization stats are domain-invariant (per-variable, not per-cell)
    assert torch.equal(cr.out_center, dx.out_center)
    assert torch.equal(cr.out_scale, dx.out_scale)
    assert torch.equal(cr.era5_center, dx.era5_center)
    assert torch.equal(cr.era5_scale, dx.era5_scale)
    # shared network objects (not reloaded)
    assert cr.regression_model is dx.regression_model
    # spatial footprint genuinely shrank and stays within the parent grid
    assert cr.lat_output_numpy.shape[0] <= dx.lat_output_numpy.shape[0]
    assert cr.lat_output_numpy.shape[1] <= dx.lat_output_numpy.shape[1]
    assert cr.lat_output_numpy.min() >= dx.lat_output_numpy.min() - 1e-5
    assert cr.lat_output_numpy.max() <= dx.lat_output_numpy.max() + 1e-5
    # invariants sliced consistently (names + shape track the restricted grid)
    assert cr._static_names == dx._static_names
    assert cr.static_invariants.shape[-2:] == cr.lat_output_grid.shape


def test_constructor_param_guard():
    """Guard against set_domain dropping a config field: if the constructor gains a
    new arg, this fails so the author is forced to thread it through set_domain()
    + extend its test."""
    params = set(inspect.signature(CorrDiffCosmoEra5.__init__).parameters) - {"self"}
    expected = {
        "era5_variables",
        "output_variables",
        "regression_model",
        "diffusion_model",
        "resolution",
        "mode",
        "lat_input_grid",
        "lon_input_grid",
        "lat_output_grid",
        "lon_output_grid",
        "era5_center",
        "era5_scale",
        "out_center",
        "out_scale",
        "static_invariants",
        "pre_invariant_variables",
        "post_invariant_variables",
        "channel_transforms",
        "constraints",
        "number_of_samples",
        "physical_clamp",
        "number_of_steps",
        "sigma_min",
        "sigma_max",
        "rho",
        "solver",
        "seed",
        "amp",
        "hub_heights",
        "hub_interp",
        "wind_levels",
    }
    assert params == expected, (
        "CorrDiffCosmoEra5.__init__ signature changed. If you added a config field, "
        "thread it through set_domain() and assert it in "
        "test_set_domain_preserves_config, then update this set."
    )


def test_set_domain_halo():
    """set_domain(halo=N) runs on an expanded grid but returns the trimmed bbox."""
    dx = _build()
    base = dx.set_domain(48.0, 52.0, 9.0, 13.0, halo=0)
    haloed = dx.set_domain(48.0, 52.0, 9.0, 13.0, halo=1)
    assert base._halo == (0, 0, 0, 0)
    assert haloed._halo == (1, 1, 1, 1)  # interior bbox -> full halo, no clamp
    # the model's RUN grid is expanded by the halo on each side...
    assert haloed.lat_output_grid.shape[0] == base.lat_output_grid.shape[0] + 2
    assert haloed.lat_output_grid.shape[1] == base.lat_output_grid.shape[1] + 2
    # ...but the REPORTED output grid is trimmed back to exactly the no-halo bbox
    oc_b = base.output_coords(base.input_coords())
    oc_h = haloed.output_coords(haloed.input_coords())
    assert np.asarray(oc_h["lat"]).shape == np.asarray(oc_b["lat"]).shape
    assert np.allclose(np.asarray(oc_h["lat"]), np.asarray(oc_b["lat"]))
    assert np.allclose(np.asarray(oc_h["lon"]), np.asarray(oc_b["lon"]))


def test_set_domain_halo_clamps_at_grid_edge():
    """When the halo would extend past the grid edge it is clamped there (real data
    runs out): that side's trim is smaller than requested and a warning is issued,
    rather than slicing out of bounds. The bbox starts at the 2nd grid row, so a
    halo of 2 can only expand 1 row upward before hitting the top edge."""
    dx = _build()
    with pytest.warns(UserWarning, match="clamped at the grid edge"):
        cr = dx.set_domain(47.5, 50.0, 9.0, 11.0, halo=2)
    assert cr._halo == (1, 2, 2, 2)  # top clamped to 1; other sides get the full 2


def test_set_domain_small_region():
    """Both models are DiT-RoPE (crop-size agnostic at the fixed resolution), so a
    sub-region just slices the small block: any size works -- no minimum size, no
    padding, and no halo by
    default."""
    dx = _build()  # mean
    cr = dx.set_domain(lat_min=49.5, lat_max=51.0, lon_min=9.5, lon_max=11.5)
    assert cr._halo == (0, 0, 0, 0)
    assert cr.lat_output_grid.shape[0] < 8 and cr.lat_output_grid.shape[1] < 8
    oc = cr.output_coords(cr.input_coords())
    assert np.asarray(oc["lat"]).shape == tuple(cr.lat_output_grid.shape)


def test_set_domain_snaps_run_grid_to_patch():
    """For a real DiT (patch_size>1) set_domain snaps the RUN grid to a multiple of
    patch_size — an odd extent would floor to extent-1 in the detokenizer and
    mismatch the allocated output — while the reported bbox stays exact (the
    snap rows are trimmed via the halo)."""
    dx = _build()
    dx._patch_size = 2  # emulate the DiT patch (synthetic default is 1 = no-op)
    cr = dx.set_domain(49.5, 51.3, 9.5, 11.5)  # selects an odd number of rows
    rg = tuple(cr.lat_output_grid.shape)
    assert rg[0] % 2 == 0 and rg[1] % 2 == 0  # run grid snapped to even
    top, bot, left, right = cr._halo
    bb = np.asarray(cr.output_coords(cr.input_coords())["lat"]).shape
    # reported bbox + trim == run grid
    assert (bb[0] + top + bot, bb[1] + left + right) == rg


def test_set_domain_negative_trim_rejected():
    """If the grid dimension isn't a multiple of patch_size, snapping a near-full
    window grows to the edge then shrinks below the requested bbox -> a negative
    halo trim. That is refused loudly, not silently returned as a smaller domain.
    (Unreachable as shipped: all native/extended dims are even and patch_size==2.)"""
    dx = _build()
    dx._patch_size = 5  # the 8-cell synthetic grid is not a multiple of 5
    # selects 6 of 8 rows/cols (strictly inside the footprint); the snap then can't
    # reach the next multiple of 5 within the 8-cell grid, so it shrinks.
    with pytest.raises(ValueError, match="not divisible by patch_size"):
        dx.set_domain(47.5, 52.5, 7.5, 13.5)


def test_set_domain_below_natten_minimum_raises():
    """A domain below attn_kernel*patch cells per side is refused with a clear
    error (the NATTEN neighborhood-attention kernel must fit the latent), not a
    cryptic error inside the net."""
    dx = _build()
    dx._patch_size, dx._min_domain_cells = 2, 6  # tiny stand-in for the real 46
    with pytest.raises(ValueError, match="minimum"):
        dx.set_domain(49.5, 51.3, 9.5, 11.5)


def test_regression_forward_dit():
    """The DiT regression forward rebinds the latent grid to the run size and
    does a single forward returning [1, n_out, H, W]."""
    n_out = len(OUTPUT_VARIABLES)
    dx = _build(regression_model=_MockRegDiT(n_out))
    out = dx._regression_forward(torch.zeros(1, 12, 16, 24))  # H=16, W=24, patch 2
    assert out.shape == (1, n_out, 16, 24)
    # rebind set latent_hw = (H/patch, W/patch) and the detokenizer patch counts
    assert dx.regression_model.attn_kwargs_forward["latent_hw"] == (8, 12)
    assert (
        dx.regression_model.detokenizer.h_patches,
        dx.regression_model.detokenizer.w_patches,
    ) == (8, 12)


def test_nonfinite_input_warns():
    """A NaN in the ERA5 input is flagged (it silently ruins the output otherwise),
    and the check_inputs toggle suppresses the warning."""
    dx = _build()
    era5 = torch.zeros(
        len(ERA5_VARIABLES), len(dx.lat_input_numpy), len(dx.lon_input_numpy)
    )
    era5[0] = float("nan")
    with pytest.warns(UserWarning, match="non-finite"):
        dx.preprocess_input(
            era5, datetime(2021, 7, 14, 12), dx.lat_output_grid, dx.lon_output_grid
        )
    # check_inputs=False suppresses the non-finite warning (targeted: an unrelated
    # warning in the path would not make this fail for the wrong reason)
    dx.check_inputs = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dx.preprocess_input(
            era5, datetime(2021, 7, 14, 12), dx.lat_output_grid, dx.lon_output_grid
        )
    assert not any("non-finite" in str(w.message) for w in caught)


def test_postprocess_physical_sane():
    """postprocess_output produces physically reasonable values: after de-norm,
    inverse transforms, bounds, and the CLCT %->fraction rescale, the output is
    finite, precip is >= 0, cloud cover is in [0, 1], and the solar gate passes
    direct shortwave by day but zeros it at night."""
    dx = _build()
    ov = OUTPUT_VARIABLES
    raw = (
        torch.randn(1, len(ov), 8, 8, generator=torch.Generator().manual_seed(0)) * 3.0
    )
    lat2d, lon2d = dx.lat_output_numpy, dx.lon_output_numpy
    # daytime
    out = dx.postprocess_output(raw.clone(), datetime(2021, 7, 14, 12), lat2d, lon2d)[
        0
    ].numpy()
    assert np.isfinite(out).all()
    assert out[ov.index("TOT_PRECIP")].min() >= -1e-6  # >=0 bound
    tcc = out[ov.index("CLCT")]
    assert tcc.min() >= -1e-6 and tcc.max() <= 1.0 + 1e-6  # tcc fraction (0-1)
    # summer noon: the gate is open, so direct shortwave passes (non-zero)
    assert out[ov.index("ASWDIR_S")].max() > 1e-3
    # winter midnight over central Europe -> SZA gate zeros direct shortwave
    out_n = dx.postprocess_output(raw.clone(), datetime(2021, 1, 1, 0), lat2d, lon2d)[
        0
    ].numpy()
    assert np.abs(out_n[ov.index("ASWDIR_S")]).max() < 1e-3


def _build_hub(**overrides):
    """A model set up to derive hub-height wind: two model wind levels -- L40 at
    10 m and L39 at 35 m above the ground -- and a request for wind at 35 m, which
    equals L39 exactly so the interpolation is easy to check. Heights are fixed
    here (b=0), so they don't depend on the random synthetic elevation -- that
    keeps the interpolation result deterministic; the terrain-following case
    (b!=0, height = a + b*elevation) is covered by a separate test. There is no
    separate 10 m reference level; the lowest model level (L40, ~10 m) is the
    near-surface node, as in the shipped packages."""
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


def test_hub_height_output_coords_and_derivation():
    """u{H}m, v{H}m are appended to output coords (in that order), and the
    derivation interpolates the level components to the hub height (here =35 m ->
    exactly the L39 level)."""
    dx = _build_hub()
    ov = dx.output_variables
    assert list(dx._output_coord_variables[-2:]) == ["u35m", "v35m"]
    H, W = dx.lat_output_numpy.shape
    x = torch.zeros(1, len(ov), H, W)
    x[0, ov.index("U_L39")] = 3.0
    x[0, ov.index("V_L39")] = 4.0  # (u,v)=(3,4) at 35 m
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    assert out.shape[1] == len(ov) + 2  # u{H}m, v{H}m appended
    assert torch.allclose(out[0, len(ov)], torch.full((H, W), 3.0), atol=1e-4)
    assert torch.allclose(out[0, len(ov) + 1], torch.full((H, W), 4.0), atol=1e-4)


def test_hub_components_compose_with_derived_ws():
    """The intended division of labour: the wrapper emits u{H}m/v{H}m components
    on-model, and the magnitude ws{H}m is composed by the stock DerivedWS. The
    integration CONTRACT is the naming handshake -- DerivedWS(levels=[H]) must
    consume EXACTLY the derived component names the wrapper appends. If the label
    format drifted (e.g. u35.0m, U35m, u_35m) the composition would break
    silently. Two heights, one decimal, guard the int and non-integer formats."""
    from earth2studio.models.dx import DerivedWS

    dx = _build_hub(hub_heights=[35.0, 22.5])
    emitted = list(dx._output_coord_variables)[-4:]  # u35m, v35m, u22.5m, v22.5m
    ws = DerivedWS(levels=["35m", "22.5m"])
    assert ws.in_variables == emitted  # DerivedWS consumes exactly what we emit
    assert ws.out_variables == ["ws35m", "ws22.5m"]


def test_hub_multi_height_interleave_order():
    """Two hub heights at once: the appended channels are interleaved u,v per
    height in hub_heights order (u35, v35, u22.5, v22.5). A single height has just
    one u,v pair, so there is no across-height ordering to get wrong; only >= 2
    heights expose a mix-up (e.g. all u's then all v's, or the heights swapped),
    pinned here by the four distinct expected values."""
    dx = _build_hub(hub_heights=[35.0, 22.5])
    ov = dx.output_variables
    assert list(dx._output_coord_variables[-4:]) == ["u35m", "v35m", "u22.5m", "v22.5m"]
    H, W = dx.lat_output_numpy.shape
    x = torch.zeros(1, len(ov), H, W)
    # L40 = 10 m, L39 = 35 m. At 35 m -> exactly L39; at 22.5 m -> midpoint.
    x[0, ov.index("U_L39")] = 1.0
    x[0, ov.index("V_L39")] = 2.0
    x[0, ov.index("U_L40")] = 9.0  # u22.5 = (9+1)/2 = 5
    x[0, ov.index("V_L40")] = 12.0  # v22.5 = (12+2)/2 = 7
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    expected = [1.0, 2.0, 5.0, 7.0]  # u35, v35, u22.5, v22.5
    for k, val in enumerate(expected):
        assert torch.allclose(out[0, len(ov) + k], torch.full((H, W), val), atol=1e-4)


def test_hub_heights_requires_wind_levels():
    """Requesting hub_heights without wind_levels metadata raises clearly."""
    with pytest.raises(ValueError, match="wind_levels"):
        _build(hub_heights=[100.0])


def test_set_domain_preserves_hub_heights():
    """Cropping a hub-enabled model with set_domain keeps the hub settings
    (heights, interp method, wind_levels) and rebuilds its internal lookups (the
    u/v channel-index maps + height coefficients) to match the parent."""
    dx = _build_hub()
    cr = dx.set_domain(48.0, 52.0, 8.0, 13.0)
    assert cr._hub_heights == [35.0]
    assert cr._hub_interp == dx._hub_interp
    assert cr._wind_levels == dx._wind_levels
    # the full rebuilt state (index maps + height coefficients) matches the parent
    assert cr._hub_u_idx.tolist() == dx._hub_u_idx.tolist()
    assert cr._hub_v_idx.tolist() == dx._hub_v_idx.tolist()
    assert cr._hub_a.tolist() == dx._hub_a.tolist()
    assert cr._hub_b.tolist() == dx._hub_b.tolist()
    assert list(cr._output_coord_variables)[-2:] == ["u35m", "v35m"]


def test_hub_height_elevation_dependent_heights():
    """b_L != 0: per-level heights track the elevation invariant (h = a + b*elev) --
    the most failure-prone path (elevation index/slice/normalization). The hub
    height is BETWEEN the levels, so the interpolation weight depends on the
    elevation-adjusted top height: a mishandled elevation gives the wrong weight.
    (A target ON a level, e.g. 35 m, would clamp and pass even with broken
    elevation, so it would not actually exercise the b*elev term.)"""
    ov = ["U_10M", "V_10M", "U_L40", "V_L40", "U_L39", "V_L39", "T_2M"]
    n, H, W = len(ov), 8, 8
    static = OrderedDict(
        elevation_norm=torch.full((H, W), 2.0),  # known normalized elevation
        land_fraction=torch.zeros(H, W),
        z0_lu_norm=torch.zeros(H, W),
    )
    wind_levels = {
        "elevation_invariant": "elevation_norm",
        "levels": [
            {"u": "U_L40", "v": "V_L40", "a": 10.0, "b": 0.0},  # height 10
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
        hub_heights=[22.5],  # midpoint of 10 and the elevation-adjusted 35
        static_invariants=static,
    )
    x = torch.zeros(1, n, H, W)
    x[0, ov.index("U_L39")] = 20.0
    x[0, ov.index("V_L39")] = 40.0  # L40 winds left at 0
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    # w = (22.5-10)/(35-10) = 0.5 -- only because elev=2 makes L39 height 35; a
    # broken elevation (height 30) gives w=0.625 -> 12.5/20, which these reject.
    assert torch.allclose(out[0, n], torch.full((H, W), 10.0), atol=1e-4)  # u22.5
    assert torch.allclose(out[0, n + 1], torch.full((H, W), 20.0), atol=1e-4)  # v22.5


def test_hub_interp_log_method():
    """The 'log' interpolation works in ln(height). The geometric mean of the two
    level heights, sqrt(10*35) ~= 18.7 m, is the midpoint in log space (weight 0.5),
    so a wind of 0 at 10 m and 10 at 35 m interpolates to 5. Linear interp would
    give ~3.5 there, so this both distinguishes and exercises the 'log' branch."""
    target = float((10.0 * 35.0) ** 0.5)  # ~18.71 m, geometric midpoint
    dx = _build_hub(hub_interp="log", hub_heights=[target])
    ov = dx.output_variables
    H, W = dx.lat_output_numpy.shape
    x = torch.zeros(1, len(ov), H, W)
    x[0, ov.index("U_L40")] = 0.0  # 10 m
    x[0, ov.index("U_L39")] = 10.0  # 35 m
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    assert torch.allclose(out[0, len(ov)], torch.full((H, W), 5.0), atol=1e-3)


def test_hub_levels_only_no_anchor_row():
    """wind_levels carries only model levels (no 10 m anchor): the index maps
    have exactly one row per level, with nothing prepended."""
    dx = _build_hub()
    assert dx._hub_u_idx.tolist() == [
        dx.output_variables.index("U_L40"),
        dx.output_variables.index("U_L39"),
    ]
    assert dx._hub_v_idx.tolist() == [
        dx.output_variables.index("V_L40"),
        dx.output_variables.index("V_L39"),
    ]
    assert len(dx._hub_u_idx) == 2  # one row per level, no anchor


def test_hub_extrapolation_warns():
    """A hub height outside the levels' resolvable band warns the user that
    values are clamped (held constant), not interpolated."""
    with pytest.warns(UserWarning, match="resolvable band"):
        _build_hub(hub_heights=[100.0])  # band is [10, 35] m; 100 > 35


def test_hub_clamps_below_lowest():
    """A hub height below the lowest level (~10 m) clamps to the lowest-level
    wind (no extrapolation/NaN) -- the lowest model level is the floor."""
    with pytest.warns(UserWarning, match="resolvable band"):
        dx = _build_hub(hub_heights=[5.0])  # below lowest level (10 m)
    ov = dx.output_variables
    H, W = dx.lat_output_numpy.shape
    x = torch.zeros(1, len(ov), H, W)
    x[0, ov.index("U_L40")] = 6.0
    x[0, ov.index("V_L40")] = 8.0  # (u,v)=(6,8) at the lowest level (10 m)
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    assert torch.isfinite(out[0, len(ov) : len(ov) + 2]).all()
    assert torch.allclose(out[0, len(ov)], torch.full((H, W), 6.0), atol=1e-4)  # u5m
    assert torch.allclose(
        out[0, len(ov) + 1], torch.full((H, W), 8.0), atol=1e-4
    )  # v5m


@pytest.mark.parametrize(
    "levels", [[], [{"u": "U_L40", "v": "V_L40", "a": 10.0, "b": 0.0}]]
)
def test_hub_fewer_than_two_levels_rejected(levels):
    """0 or 1 levels cannot interpolate (a single level is a constant field); both
    must fail the >=2 guard loudly at construction, not IndexError at inference."""
    with pytest.raises(ValueError, match="at least 2"):
        _build_hub(
            wind_levels={"elevation_invariant": "elevation_norm", "levels": levels}
        )


def test_hub_nonmonotonic_heights_rejected():
    """The monotonicity guard raises when per-pixel heights invert over the
    grid's elevation range. This guard is the entire safety basis for the
    levels-only (no-anchor) design, so it must be exercised directly."""
    static = OrderedDict(
        elevation_norm=torch.full((8, 8), 2.0),  # known normalized elevation
        land_fraction=torch.zeros(8, 8),
        z0_lu_norm=torch.zeros(8, 8),
    )
    # Sorted by nominal a, L40 (a=10) is "below" L39 (a=30), but b inverts it at
    # elev=2: h(L40)=10+15*2=40 > h(L39)=30+0*2=30.
    wl = {
        "elevation_invariant": "elevation_norm",
        "levels": [
            {"u": "U_L40", "v": "V_L40", "a": 10.0, "b": 15.0},
            {"u": "U_L39", "v": "V_L39", "a": 30.0, "b": 0.0},
        ],
    }
    with pytest.raises(ValueError, match="monotonic"):
        _build_hub(wind_levels=wl, static_invariants=static)


def test_hub_veer_interpolates_components_not_speed():
    """The wind direction changes with height (it "veers"): due east at 10 m, due
    north at 35 m. u and v are interpolated INDEPENDENTLY, so the speed of the
    interpolated components (sqrt(50) ~= 7.07) is NOT what interpolating the speed
    itself would give (10). This proves the model emits wind components, leaving
    the magnitude to a composed DerivedWS."""
    dx = _build_hub(hub_heights=[22.5])  # midpoint of 10 m and 35 m
    ov = dx.output_variables
    H, W = dx.lat_output_numpy.shape
    x = torch.zeros(1, len(ov), H, W)
    x[0, ov.index("U_L40")] = 10.0  # 10 m: due east (u=10, v=0), speed 10
    x[0, ov.index("V_L40")] = 0.0
    x[0, ov.index("U_L39")] = 0.0  # 35 m: due north (u=0, v=10), speed 10
    x[0, ov.index("V_L39")] = 10.0
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    u, v = out[0, len(ov)], out[0, len(ov) + 1]
    assert torch.allclose(u, torch.full((H, W), 5.0), atol=1e-4)  # component interp
    assert torch.allclose(v, torch.full((H, W), 5.0), atol=1e-4)
    # speed of interpolated components = sqrt(50) ~= 7.07, NOT the 10 you'd get
    # from interpolating speed -- the discriminating assertion.
    assert torch.allclose(
        torch.sqrt(u**2 + v**2), torch.full((H, W), 50.0**0.5), atol=1e-3
    )


def test_call_end_to_end_with_hub():
    """Full ``dx(x, coords)`` with hub heights through the real ``__call__`` path
    (mock DiT): the only non-skipped coverage of the output allocation width
    (trained + 2 derived u/v channels). A regression to ``len(output_variables)``
    in __call__ would raise here. Diffusion mode shares this path (``_derive_hub_
    wind`` runs in postprocess after sample assembly; allocation is mode-agnostic)."""
    ov = ["U_10M", "V_10M", "U_L40", "V_L40", "U_L39", "V_L39", "T_2M"]
    dx = _build_hub(regression_model=_MockRegDiT(len(ov)))
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


def test_hub_missing_channel_rejected():
    """wind_levels referencing a u/v channel absent from output_variables fails
    loudly at construction."""
    wl = {
        "elevation_invariant": "elevation_norm",
        "levels": [
            {"u": "U_NOPE", "v": "V_L40", "a": 10.0, "b": 0.0},
            {"u": "U_L39", "v": "V_L39", "a": 35.0, "b": 0.0},
        ],
    }
    with pytest.raises(ValueError, match=r"absent from output_variables.*U_NOPE"):
        _build_hub(wind_levels=wl)


def test_hub_elevation_invariant_must_be_static():
    """wind_levels.elevation_invariant must name a real static invariant."""
    wl = {
        "elevation_invariant": "not_a_real_field",
        "levels": [
            {"u": "U_L40", "v": "V_L40", "a": 10.0, "b": 0.0},
            {"u": "U_L39", "v": "V_L39", "a": 35.0, "b": 0.0},
        ],
    }
    with pytest.raises(ValueError, match=r"not_a_real_field.*static invariant"):
        _build_hub(wind_levels=wl)


@pytest.mark.parametrize("h", [0.0, -5.0])
def test_hub_heights_must_be_positive(h):
    """Non-positive hub heights are rejected (m above ground must be > 0): both the
    0 boundary and a negative value fail at construction."""
    with pytest.raises(ValueError, match="positive"):
        _build_hub(hub_heights=[h])


def test_hub_interp_invalid_rejected():
    """An unknown hub_interp method is rejected at construction."""
    with pytest.raises(ValueError, match=r"hub_interp must be.*powerlaw"):
        _build_hub(hub_interp="powerlaw")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cross-device co-location can only be exercised with a real device split",
)
def test_hub_buffers_share_static_invariants_device():
    """_setup_hub_wind must build the hub buffers on static_invariants.device
    (load_model builds static_invariants on the requested device; a default-CPU
    hub-buffer build would mismatch and crash at derive time). Put static on the
    GPU and assert the hub buffers land there too -- an all-CPU build can't expose
    this, so a real device split is required."""
    H, W = 8, 8
    static = OrderedDict(
        elevation_norm=torch.full((H, W), 2.0, device="cuda"),
        land_fraction=torch.zeros(H, W, device="cuda"),
        z0_lu_norm=torch.zeros(H, W, device="cuda"),
    )
    dx = _build_hub(static_invariants=static)
    dev = dx.static_invariants.device
    assert dev.type == "cuda"  # the split must be real, else the test is vacuous
    assert dx._hub_a.device == dev
    assert dx._hub_b.device == dev
    assert dx._hub_u_idx.device == dev
    assert dx._hub_v_idx.device == dev


def _diffusion_coords(dx):
    """A single-frame (x, coords) pair for a full ``dx(x, coords)`` call: one
    batch, one time, random ERA5 inputs on the native input grid. Used by the
    diffusion __call__ tests."""
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


def test_call_diffusion_ensemble():
    """Diffusion __call__ plumbing with a mock EDM net (GPU-free, gain=0.5 so the
    result is a schedule-weighted multiple of the per-sample noise): exercises the
    diffusion _forward, per-sample noise (seed+i), ensemble allocation, and the
    mode-agnostic postprocess. (Schedule correctness is pinned separately below.)"""
    ov = OUTPUT_VARIABLES
    dx = _build(
        mode="diffusion",
        regression_model=None,
        diffusion_model=_MockDiffusionDiT(len(ov), gain=0.5),
        number_of_samples=3,
        seed=0,
        channel_transforms={},
        constraints={},
    )
    x, coords = _diffusion_coords(dx)
    out, oc = dx(x, coords)
    H, W = dx.lat_output_numpy.shape
    assert out.shape == (1, 3, 1, len(ov), H, W)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[0, 0], out[0, 1])  # members differ (indep. noise)
    out2, _ = dx(x, coords)
    assert torch.allclose(out, out2)  # fixed seed -> reproducible


def test_diffusion_sampler_telescopes_to_zero():
    """With a perfect-denoiser mock (D_x = 0) the sampler must drive the sample to
    exactly 0: the schedule ends at t_n = 0, so the final Euler step is
    x + (0 - t_cur) * x / t_cur = 0, independent of the initial noise. This pins the
    t_n = 0 append (without it the sample lands at latents * sigma_min != 0) and that
    the loop runs end-to-end without NaN. Note: under D_x = 0 the intermediate steps
    telescope away and Heun degenerates to Euler (d_prime == d_cur), so this is a
    termination check -- not a schedule-shape (rho) or corrector check."""
    ov = OUTPUT_VARIABLES
    dx = _build(
        mode="diffusion",
        regression_model=None,
        diffusion_model=_MockDiffusionDiT(len(ov), gain=0.0),
        number_of_samples=1,
        seed=0,
        channel_transforms={},
        constraints={},
    )
    x, coords = _diffusion_coords(dx)
    out, _ = dx(x, coords)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


def test_diffusion_euler_differs_from_heun():
    """The ``solver="euler"`` (1st-order) branch runs and produces a DIFFERENT
    result than ``"heun"`` (2nd-order) for the same seed -- pinning that ``solver``
    actually switches the integrator (the euler path is otherwise unexercised).
    With the linear mock denoiser ``D_x = 0.5*x`` Heun's corrector slope differs
    from the Euler slope at every step but the last (the corrector is skipped at
    ``t_next = 0``), so the endpoints must differ."""
    ov = OUTPUT_VARIABLES

    def _build_solver(solver):
        return _build(
            mode="diffusion",
            regression_model=None,
            diffusion_model=_MockDiffusionDiT(len(ov), gain=0.5),
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
    # euler branch runs end-to-end: right shape, finite
    assert out_euler.shape == (1, 1, 1, len(ov), H, W)
    assert torch.isfinite(out_euler).all()
    # 1st-order != 2nd-order for the same seed (genuinely takes the euler branch)
    assert not torch.allclose(out_heun, out_euler)


def test_check_bounds_rejects_out_of_range_target():
    """Targets outside, or exactly on, any ERA5 input-grid edge are rejected
    (strict containment, both axes) so the interpolator never indexes past the
    array; an interior target passes. The on-edge cases pin the strict <=/>=
    comparisons; the interior positive control rules out an always-raise guard."""
    dx = _build()
    lat0, lat1 = float(dx.lat_input_numpy[0]), float(dx.lat_input_numpy[-1])
    lon0, lon1 = float(dx.lon_input_numpy[0]), float(dx.lon_input_numpy[-1])
    in_lat = np.array([[float(dx.lat_input_numpy[1])]], dtype=np.float32)
    in_lon = np.array([[float(dx.lon_input_numpy[1])]], dtype=np.float32)
    dx._check_bounds(in_lat, in_lon)  # interior target accepted (positive control)
    for bad_lat in (lat0 - 1.0, lat0, lat1, lat1 + 1.0):  # below / on / on / above
        with pytest.raises(ValueError, match="latitude"):
            dx._check_bounds(np.array([[bad_lat]], dtype=np.float32), in_lon)
    for bad_lon in (lon0 - 1.0, lon0, lon1, lon1 + 1.0):
        with pytest.raises(ValueError, match="longitude"):
            dx._check_bounds(in_lat, np.array([[bad_lon]], dtype=np.float32))


def test_transform_round_trip_incl_asinh():
    """postprocess_output applies de-normalize -> invert (in that order) and inverts
    each channel transform to its analytic physical value. Covers all three inverse
    transforms (log_eps, logit_eps_percent, asinh): non-trivial out_center/out_scale
    pin the de-norm-before-invert order (a swap would fail), and a negative asinh
    input checks the only signed inverse -- sinh must carry the sign and the
    nonnegativity clamp (physical_clamp=True) must skip it."""
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
    x[0, 0] = 0.7  # log_eps    -> de-norm y = 0.7*2.0 + 0.5  = 1.9
    x[0, 1] = 0.3  # logit      -> de-norm y = 0.3*1.5 - 0.2  = 0.25
    x[0, 2] = -0.5  # asinh     -> de-norm y = -0.5*0.5 - 0.4 = -0.65 (negative)
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    exp_log = 1.0 * (math.exp(0.7 * 2.0 + 0.5) - 1)
    exp_logit = ((1 / (1 + math.exp(-(0.3 * 1.5 - 0.2))) - 0.01) / 0.98) * 100.0
    exp_asinh = 2.0 * math.sinh(-0.5 * 0.5 - 0.4)
    assert exp_asinh < 0  # the signed-inverse case: must not be clamped to 0
    assert torch.allclose(out[0, 0], torch.full((H, W), exp_log), atol=1e-4)
    assert torch.allclose(out[0, 1], torch.full((H, W), exp_logit), atol=1e-4)
    assert torch.allclose(out[0, 2], torch.full((H, W), exp_asinh), atol=1e-4)


def test_bare_string_transform_rejected():
    """A bare-string channel transform (no explicit eps/scale) is refused loudly."""
    with pytest.raises(ValueError, match="must be a dict"):
        _build(channel_transforms={"T_2M": "log_eps"})


def test_unknown_transform_rejected():
    """An unrecognized transform name (a typo / unsupported op) is refused loudly,
    not silently skipped -- otherwise the forward transform would never be inverted
    and that channel's output would be wrong with no error."""
    with pytest.raises(ValueError, match="unsupported transform"):
        _build(channel_transforms={"T_2M": {"transform": "sqrt", "eps": 1.0}})


def test_invalid_sampler_params_rejected():
    """number_of_steps < 2 (0/0 in the EDM schedule) and an unknown solver are
    refused at construction, not silently producing NaN / a downgraded sampler."""
    with pytest.raises(ValueError, match="number_of_steps must be >= 2"):
        _build(number_of_steps=1)
    with pytest.raises(ValueError, match="solver must be"):
        _build(solver="bogus")


def test_physical_bounds_clamp_zscored_channel():
    """Metadata min/max bounds are enforced in PHYSICAL space at postprocess: a
    z-scored channel pushed below its min is clamped up to the min and one pushed
    above its max is clamped down to the max, while an unbounded channel is left
    untouched. postprocess is mode-independent
    (test_solar_gate_applies_identically_in_both_modes pins mean == diffusion), so
    this also covers nonnegativity for diffusion's z-scored channels
    (ALWU_S/ATHD_S/QV_2M/Q_L*) that their transforms don't bound."""
    ov = ["T_2M", "QV_2M", "TD_2M"]  # T_2M unbounded; QV_2M min 0; TD_2M max 10
    dx = _build(
        output_variables=ov,
        out_center=torch.zeros(3),
        out_scale=torch.ones(3),
        channel_transforms={},
        constraints={"bounds": {"QV_2M": {"min": 0.0}, "TD_2M": {"max": 10.0}}},
    )
    H, W = dx.lat_output_numpy.shape
    x = torch.empty(1, 3, H, W)
    x[0, 0] = -3.0  # T_2M unbounded
    x[0, 1] = -3.0  # QV_2M below min 0
    x[0, 2] = 50.0  # TD_2M above max 10
    out = dx.postprocess_output(
        x, datetime(2021, 7, 14, 12), dx.lat_output_numpy, dx.lon_output_numpy
    )
    assert torch.allclose(out[0, 0], torch.full((H, W), -3.0))  # unbounded: untouched
    assert torch.allclose(out[0, 1], torch.zeros(H, W))  # clamped TO min, not just >=0
    assert torch.allclose(out[0, 2], torch.full((H, W), 10.0))  # clamped TO max


def test_solar_gate_applies_identically_in_both_modes():
    """Unified night handling: with identical constraints, the mean and diffusion
    products run the SAME solar gate in physical space, so their postprocess
    outputs match -- even in the twilight band where the ramp partially attenuates. The input is POSITIVE and
    the time is low-sun on purpose: the result is then driven by the ramp multiply,
    not the >=0 clamp, so a mode-dependent gate would diverge here."""
    ov = OUTPUT_VARIABLES  # ASWDIR_S gated (th=0.05, hw=0.05 -> ramp band cos_z in [0,0.10])
    cons = {
        "sza_gate": {"half_width": 0.05, "channels": {"ASWDIR_S": {"threshold": 0.05}}}
    }
    common = dict(channel_transforms={}, constraints=cons)  # identity de-norm
    mean = _build(mode="mean", **common)
    diff = _build(
        mode="diffusion",
        regression_model=None,
        diffusion_model=_MockDiffusionDiT(len(ov)),
        **common,
    )
    assert diff._sza_gate == mean._sza_gate  # diffusion parses the same gate as mean
    H, W = mean.lat_output_numpy.shape
    lat2d, lon2d = mean.lat_output_numpy, mean.lon_output_numpy
    sw = ov.index("ASWDIR_S")
    xs = torch.full((1, len(ov), H, W), 5.0)  # positive -> gate ramp drives the value
    t_taper = datetime(2021, 7, 14, 4)  # low morning sun over the grid -> ramp band
    om = mean.postprocess_output(xs.clone(), t_taper, lat2d, lon2d)
    od = diff.postprocess_output(xs.clone(), t_taper, lat2d, lon2d)
    assert torch.allclose(om, od)  # mode-independent postprocess
    v = od[
        0, sw
    ]  # the ramp is genuinely active: some cells partially attenuated (0<v<5)
    assert ((v > 1e-3) & (v < 5.0 - 1e-3)).any()
    for m in (mean, diff):  # night -> zero in both modes
        night = m.postprocess_output(xs.clone(), datetime(2021, 1, 1, 0), lat2d, lon2d)
        assert night[0, sw].abs().max() < 1e-3


def test_load_default_package_placeholder(monkeypatch):
    """Placeholder hosting: with no DEFAULT_PACKAGE_URI, load_default_package
    raises NotImplementedError; setting the URI makes it return a Package (so
    enabling the hosted package later is a one-line change)."""
    import earth2studio.models.dx.corrdiff_cosmo_era5 as m
    from earth2studio.models.auto import Package

    monkeypatch.setattr(m, "DEFAULT_PACKAGE_URI", None)
    with pytest.raises(NotImplementedError):
        CorrDiffCosmoEra5.load_default_package()

    monkeypatch.setattr(
        m, "DEFAULT_PACKAGE_URI", "hf://nvidia/corrdiff-cosmo-era5@abc123"
    )
    pkg = CorrDiffCosmoEra5.load_default_package()
    assert isinstance(pkg, Package)


def test_load_model_invalid_selectors_rejected():
    """load_model validates resolution/mode up front -- before resolving any
    package file -- so a bad value fails with a clear message rather than a
    cryptic missing-file when resolving "<resolution>/metadata.json". (A None
    package is fine here: validation fires before the package is touched.)"""
    with pytest.raises(ValueError, match="resolution must be one of"):
        CorrDiffCosmoEra5.load_model(None, resolution="rea3")
    with pytest.raises(ValueError, match="mode must be"):
        CorrDiffCosmoEra5.load_model(None, mode="both")


def test_call_mean_end_to_end():
    """Full ``dx(x, coords)`` for mean mode with a mock DiT, covering what the
    construction-only tests never reach: the 6-D output assembly, the batch/time
    loop with per-time valid_times indexing (day vs night frames), the
    sample-expand broadcast (number_of_samples>1), physical-space postprocess
    constraints, and the lexicon ``output_coords`` contract."""
    dx = _build(
        regression_model=_MockRegDiT(len(OUTPUT_VARIABLES)), number_of_samples=2
    )
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
    # [batch, sample, time, variable, lat, lon]
    assert out.shape == (1, 2, 2, len(OUTPUT_VARIABLES), H, W)
    assert torch.isfinite(out).all()
    # the call surfaces the lexicon-mapped output names, not the interior names
    assert list(out_coords["variable"]) == [
        "u10m",
        "t2m",
        "tot_precip",
        "tcc",
        "aswdir_s",
        "tke_l40",
    ]
    # deterministic mean replicates across the sample-expand axis
    assert torch.allclose(out[0, 0], out[0, 1])
    # constraints applied in physical space (TOT_PRECIP clamped nonneg)
    assert out[0, 0, 0, OUTPUT_VARIABLES.index("TOT_PRECIP")].min() >= -1e-6
    # per-time valid_times indexing is real: ASWDIR_S is solar-gated, so the day
    # frame (t=0) is lit and the night frame (t=1) is gated to ~0 -- a misindexed
    # time would collapse both frames and break one of these.
    sw = OUTPUT_VARIABLES.index("ASWDIR_S")
    assert out[0, 0, 0, sw].abs().max() > 1e-3
    assert out[0, 0, 1, sw].abs().max() < 1e-3


@pytest.mark.package
@pytest.mark.parametrize("resolution", ["rea6", "rea2"])
@pytest.mark.parametrize("mode", ["mean", "diffusion"])
def test_corrdiff_cosmo_era5_package(mode, resolution):
    """Load-and-run on a real local package (set ``$COSMO_REA_PACKAGE`` to a
    built package dir). Loading validates the metadata-driven plumbing and a
    0/0 checkpoint load; the GPU forward exercises ``_forward``/``_denoise`` and
    the output contract. Skipped until the package is hosted (then switch to
    ``load_default_package``)."""
    pkg_path = os.environ.get("COSMO_REA_PACKAGE")
    if not pkg_path:
        pytest.skip("set COSMO_REA_PACKAGE to a built package dir to run")
    if not _UPSTREAM_ROPE:
        # load_model reconstructs the real DiT (from_checkpoint), which needs
        # natten2d_rope (PR #1731);
        # skip (don't fail) on a physicsnemo without it.
        # TODO(cosmo): remove once the physicsnemo pin includes #1731.
        pytest.skip("needs upstream physicsnemo natten2d_rope (PR #1731)")
    from earth2studio.models.auto import Package

    device = "cuda:0" if torch.cuda.is_available() else None
    dx = CorrDiffCosmoEra5.load_model(
        Package(pkg_path), device=device, mode=mode, resolution=resolution
    )
    assert len(dx.output_variables) > 0
    # load_model applies the FULL constraint set identically in BOTH modes
    # (CPU-checkable): bounds and the solar gate are parsed regardless of mode.
    if dx._constraints.get("bounds"):
        assert dx._bound_lo or dx._bound_up  # bounds parsed in both modes
    if dx._constraints.get("sza_gate"):
        assert dx._sza_gate  # solar gate parsed in both modes
    if device is None:
        pytest.skip("NATTEN forward requires CUDA; checkpoint load verified on CPU")

    # Drive the PUBLIC entry point dx(x, coords) on real weights + GPU -- this is
    # the only place a real checkpoint runs, so exercise the full plumbing
    # (_check_bounds, batch/time assembly, output_coords, lat/lon device handling)
    # rather than the private _forward. The input is deliberately synthetic N(0,1):
    # only the shape/finite/bounds contract is asserted, not output realism.
    ic = dx.input_coords()
    torch.manual_seed(0)  # deterministic input so the finite check can't flake
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
    # [batch, sample, time, variable, lat, lon]
    assert out.shape == (
        1,
        dx.number_of_samples,
        1,
        len(dx._output_coord_variables),
        H,
        W,
    )
    assert list(oc["variable"]) == list(dx._output_coord_variables)  # lexicon names
    assert torch.isfinite(out).all()
    # Physical-space metadata bounds are enforced in BOTH modes (diffusion too):
    # every channel with a finite min/max in metadata["constraints"]["bounds"] must
    # hold on the output, including z-scored nonneg channels that have no clamping
    # transform (ALWU_S/ATHD_S/QV_2M/Q_L*). Enforced via the default
    # physical_clamp=True that load_model uses.
    bounds = dx._constraints.get("bounds") or {}
    for ch, b in bounds.items():
        if ch not in dx.output_variables:
            continue
        i = dx.output_variables.index(ch)
        sl = out[0, :, 0, i]  # [sample, lat, lon]
        if b.get("min") is not None:
            assert sl.min().item() >= b["min"] - 1e-4, f"{ch} below min in {mode}"
        if b.get("max") is not None:
            assert sl.max().item() <= b["max"] + 1e-4, f"{ch} above max in {mode}"
