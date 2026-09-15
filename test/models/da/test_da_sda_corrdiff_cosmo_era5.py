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

"""Unit tests for the CorrDiffCosmoEra5SDA assimilation wrapper and its dx seam.

Construction-based (no real package / no GPU forward). They exercise the SDA
eligibility seam (``_identity_output_indices``), the dx ``_denoise`` score-predictor
seam, the sparse-obs -> normalized-space tensors, and the public DA contract
(coords, ``__call__``, ``create_generator``, ``load_model``).
"""

import os
import types
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from physicsnemo.diffusion.guidance import (
    DataConsistencyDPSGuidance,
    DPSScorePredictor,
)
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler

from earth2studio.models.auto import Package
from earth2studio.models.da.sda_corrdiff_cosmo_era5 import CorrDiffCosmoEra5SDA
from earth2studio.models.dx.corrdiff_cosmo_era5 import CorrDiffCosmoEra5
from earth2studio.utils.type import CoordSystem

# ── Mock networks ────────────────────────────────────────────────────────────


class PhooDiffusionDiT(torch.nn.Module):
    """Stand-in for EDMPreconditioner(ConcatConditionWrapper(DiT))."""

    def __init__(self, n_out: int, gain: float = 0.5):
        super().__init__()
        self.n_out = n_out
        self.gain = gain
        inner = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(patch_size=(2, 2)),
            attn_kwargs_forward={},
            detokenizer=types.SimpleNamespace(h_patches=None, w_patches=None),
        )
        self.model = types.SimpleNamespace(model=inner)

    def forward(self, x, sigma, condition=None, attn_kwargs=None):
        return self.gain * x[:, : self.n_out]


# ── Constants ────────────────────────────────────────────────────────────────

ERA5_VARIABLES = ["u10m", "v10m", "t2m"]
# Plain identity channels (U_10M/V_10M -> u10m/v10m, unit-scale 1) plus transformed
# ones (TOT_PRECIP log_eps -> tp, CLCT logit -> tcc) so eligibility has both kinds.
OUTPUT_VARIABLES = ["U_10M", "V_10M", "TOT_PRECIP", "CLCT"]
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
    "CLCT": {"transform": "logit_eps_percent", "eps": 0.01, "scale": 100.0},
}


# ── Builders ─────────────────────────────────────────────────────────────────


def _build(**overrides: Any) -> CorrDiffCosmoEra5:
    """Construct a small synthetic diffusion-mode CorrDiffCosmoEra5."""
    ov = list(overrides.pop("output_variables", OUTPUT_VARIABLES))
    # Copy so per-model mutation (e.g. the deny-by-default test) can't leak into
    # the module-level template shared across builds.
    ct = {
        k: dict(v)
        for k, v in overrides.pop("channel_transforms", CHANNEL_TRANSFORMS).items()
    }
    n_e, n_o = len(ERA5_VARIABLES), len(ov)
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
    kwargs: dict[str, Any] = dict(
        era5_variables=ERA5_VARIABLES,
        output_variables=ov,
        regression_model=None,
        diffusion_model=PhooDiffusionDiT(n_o, gain=0.5),
        resolution="rea6",
        mode="diffusion",
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
        channel_transforms=ct,
        constraints={},
        number_of_steps=18,
        number_of_samples=1,
        sigma_max=800.0,
    )
    kwargs.update(overrides)
    return CorrDiffCosmoEra5(**kwargs)


def _build_hub(**overrides: Any) -> CorrDiffCosmoEra5:
    """A model set up to derive hub-height wind (extra u/v{H}m output channels)."""
    ov = ["U_10M", "V_10M", "U_L40", "V_L40", "U_L39", "V_L39"]
    wind_levels = {
        "elevation_invariant": "elevation_norm",
        "levels": [
            {"u": "U_L40", "v": "V_L40", "a": 10.0, "b": 0.0},
            {"u": "U_L39", "v": "V_L39", "a": 35.0, "b": 0.0},
        ],
    }
    return _build(
        output_variables=ov,
        channel_transforms={},
        wind_levels=wind_levels,
        hub_heights=[35.0],
        **overrides,
    )


def _build_sda(**overrides: Any) -> CorrDiffCosmoEra5SDA:
    """Wrap a diffusion-mode CorrDiffCosmoEra5 in the SDA model."""
    model = overrides.pop("model", None)
    if model is None:
        model = _build()
    sda_kwargs: dict[str, Any] = dict(
        assimilate_variables=("u10m", "v10m"),
        number_of_samples=2,
        sampler_steps=18,
        sda_std_obs=0.5,
        sda_gamma=5e-5,
    )
    sda_kwargs.update(overrides)
    return CorrDiffCosmoEra5SDA(model, **sda_kwargs)


def _background(model: CorrDiffCosmoEra5) -> torch.Tensor:
    """A normalized network background [1, C_bg, H, W] for the ``_denoise`` seam."""
    g = torch.Generator().manual_seed(0)
    era5 = torch.randn(
        len(ERA5_VARIABLES),
        len(model.lat_input_numpy),
        len(model.lon_input_numpy),
        generator=g,
    )
    return model.preprocess_input(
        era5, datetime(2021, 7, 14, 12), model.lat_output_grid, model.lon_output_grid
    )


def _obs_df(
    sda: CorrDiffCosmoEra5SDA,
    cells: list[tuple[int, int]],
    variables: list[str],
    values: list[float],
    time: np.datetime64,
) -> pd.DataFrame:
    """Observations snapped to the given output-grid cells (lat/lon from the grid)."""
    lat, lon = sda._lat_np, sda._lon_np
    rows = [
        {
            "time": pd.Timestamp(time),
            "lat": float(lat[i, j]),
            "lon": float(lon[i, j]),
            "variable": v,
            "observation": val,
        }
        for (i, j), v, val in zip(cells, variables, values)
    ]
    return pd.DataFrame(rows)


def _era5_da(sda: CorrDiffCosmoEra5SDA, times: np.ndarray) -> xr.DataArray:
    """An ERA5 driving state (time, variable, lat, lon) on the native input grid."""
    model = sda.model
    lat = model.lat_input_numpy
    lon = model.lon_input_numpy
    data = (
        np.random.default_rng(0)
        .standard_normal((len(times), len(ERA5_VARIABLES), len(lat), len(lon)))
        .astype(np.float32)
    )
    return xr.DataArray(
        data=data,
        dims=["time", "variable", "lat", "lon"],
        coords={
            "time": times,
            "variable": np.array(ERA5_VARIABLES),
            "lat": lat,
            "lon": lon,
        },
    )


# ── A. Eligibility seam ──────────────────────────────────────────────────────


def test_identity_output_indices_excludes_transformed_and_scaled():
    model = _build()
    idx = model._identity_output_indices
    assert OUTPUT_VARIABLES.index("U_10M") in idx
    assert OUTPUT_VARIABLES.index("V_10M") in idx
    # transformed (and lexicon-scaled) channels are excluded
    assert OUTPUT_VARIABLES.index("TOT_PRECIP") not in idx
    assert OUTPUT_VARIABLES.index("CLCT") not in idx


def test_identity_output_indices_excludes_unit_scaled_without_transform():
    # CLCT carries a lexicon unit scale (0.01) but no transform here, so it is
    # excluded solely by the scale -- independent proof of the unit-scale rule.
    model = _build(output_variables=["U_10M", "CLCT"], channel_transforms={})
    idx = model._identity_output_indices
    assert not model._channel_transforms  # no transform involved
    assert 0 in idx  # U_10M: identity + unit scale 1.0
    assert 1 not in idx  # CLCT: unit scale 0.01 != 1.0


def test_out_idx_excludes_derived_hub_wind():
    # On a hub-wind model the SDA's obs->channel map must key only the trained,
    # eligible outputs and never the appended derived hub-height components -- else
    # observations would scatter into the wrong channel index. This tests the actual
    # consumer alignment, not merely that indices stay below len(output_variables).
    model = _build_hub()
    sda = CorrDiffCosmoEra5SDA(
        model,
        assimilate_variables=("u10m", "v10m"),
        number_of_samples=1,
        sampler_steps=2,
    )
    coord_vars = list(model.output_coords(model.input_coords())["variable"])
    derived = coord_vars[len(model.output_variables) :]
    assert derived  # the model really does append derived hub-height winds
    # the obs->channel map wires each trained wind to its ACTUAL index (a permuted
    # map would fail here) and excludes every appended derived channel
    assert sda._out_idx["u10m"] == 0 and sda._out_idx["v10m"] == 1
    assert all(d not in sda._out_idx for d in derived)


def test_identity_output_indices_deny_by_default():
    model = _build()
    i = model.output_variables.index("U_10M")
    assert i in model._identity_output_indices
    # a FUTURE unknown transform is auto-excluded via membership, not enumeration
    model._channel_transforms["U_10M"] = {"transform": "made_up"}
    assert i not in model._identity_output_indices


def test_construct_rejects_non_identity_assimilate_variable():
    with pytest.raises(ValueError, match="not an assimilable"):
        _build_sda(assimilate_variables=("tp",))


def test_model_level_channels_are_assimilable():
    # The model-level winds u3d_l47/v3d_l47 (distinct from the surface u10m/v10m used
    # elsewhere) must stay identity-eligible and wire to the right indices -- they are
    # a valid choice for assimilate_variables on REA2.
    model = _build(output_variables=["U3D_L47", "V3D_L47"], channel_transforms={})
    assert model._identity_output_indices == (0, 1)
    sda = CorrDiffCosmoEra5SDA(
        model,
        assimilate_variables=("u3d_l47", "v3d_l47"),
        number_of_samples=1,
        sampler_steps=2,
    )
    assert sda._out_idx == {"u3d_l47": 0, "v3d_l47": 1}


# ── B. _denoise seam (dx) ────────────────────────────────────────────────────


def test_denoise_factory_receives_x0_and_scheduler():
    model = _build(output_variables=["U_10M", "V_10M"], channel_transforms={})
    background = _background(model)
    n_out = len(model.output_variables)
    H, W = model.lat_output_numpy.shape
    captured: dict[str, Any] = {}

    def capture(
        x0_predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: EDMNoiseScheduler,
    ) -> DPSScorePredictor:
        captured["x0"] = x0_predictor
        captured["scheduler"] = scheduler
        mask = torch.zeros(1, n_out, H, W)
        y = torch.zeros_like(mask)
        std_y = torch.ones_like(mask)
        guidance = DataConsistencyDPSGuidance(
            mask,
            y,
            std_y,
            norm=2,
            gamma=5e-5,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
        )
        return DPSScorePredictor(
            x0_predictor=x0_predictor,
            x0_to_score_fn=scheduler.x0_to_score,
            guidances=guidance,
        )

    out = model._denoise(
        background, seed=0, score_predictor_factory=capture, num_steps=4
    )

    assert "x0" in captured and callable(captured["x0"])
    assert out.shape == (1, n_out, H, W)
    assert torch.isfinite(out).all()
    # invoke the captured x0-predictor: it must actually reflect the wrapped denoiser
    # (gain 0.5), not merely be *some* callable that a mis-wired closure would satisfy.
    z = torch.randn(1, n_out, H, W)
    xp = captured["x0"](z, captured["scheduler"].sigma(torch.tensor(1.0)))
    assert xp.shape == (1, n_out, H, W)
    assert torch.allclose(xp.float(), 0.5 * z, atol=1e-5)


def test_run_diffusion_guided_changes_analysis_under_no_grad():
    # DPS re-enables the gradients it needs internally, so the guided seam must run
    # (and stay finite) even under the caller's torch.no_grad(). A nonzero obs mask
    # must also *change* the analysis vs the unconstrained (empty-mask) sample at the
    # same seed -- proof the observations are actually consumed, which finiteness
    # alone cannot show. Convergence *toward* the obs value needs a faithful trained
    # denoiser (the linear mock diverges under guidance) and requires separate
    # real-model validation.
    sda = _identity_sda()
    model = sda.model
    background = _background(model)
    n_out = len(model.output_variables)
    H, W = model.lat_output_numpy.shape
    mask = torch.zeros(1, n_out, H, W)
    mask[0, 0, H // 2, W // 2] = 1.0
    y = torch.zeros_like(mask)
    y[0, 0, H // 2, W // 2] = 1.5
    std_y = torch.where(mask.bool(), torch.tensor(0.5), torch.tensor(1.0))
    with torch.no_grad():
        free = sda._run_diffusion(
            background, torch.zeros_like(y), torch.zeros_like(mask), std_y, seed=0
        )
        guided = sda._run_diffusion(background, y, mask, std_y, seed=0)
    assert guided.shape == (1, n_out, H, W)
    assert torch.isfinite(guided).all()
    assert not torch.allclose(free, guided)


# ── C. Construction / validation ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sda_std_obs": 0.0}, r"sda_std_obs.*must be > 0"),
        ({"sda_std_obs": -1.0}, r"sda_std_obs.*must be > 0"),
        ({"sda_std_obs": float("nan")}, r"sda_std_obs.*must be > 0"),
        ({"sda_std_obs": {"u10m": 0.5}}, "missing entries"),
        ({"sda_std_obs": {"u10m": 0.5, "v10m": 0.0}}, r"must be > 0"),
        ({"sda_std_obs": {"u10m": 0.5, "v10m": -1.0}}, r"must be > 0"),
        ({"sda_std_obs": {"u10m": 0.5, "v10m": float("inf")}}, r"must be > 0"),
        ({"sda_std_obs": {"u10m": 0.5, "v10m": 1.5, "t2m": 9.0}}, "non-assimilated"),
        ({"sda_gamma": -1e-6}, "sda_gamma must be a finite value"),
        ({"sda_gamma": float("inf")}, "sda_gamma must be a finite value"),
        ({"assimilate_variables": ()}, "assimilate_variables must be non-empty"),
        ({"number_of_samples": 0}, "number_of_samples must be >= 1"),
        ({"sampler_steps": 1}, "sampler_steps must be >= 2"),
    ],
)
def test_construct_rejects_invalid_params(kwargs: dict[str, Any], match: str):
    with pytest.raises(ValueError, match=match):
        _build_sda(**kwargs)


def test_construct_rejects_non_diffusion_model():
    mean = _build(
        mode="mean", regression_model=torch.nn.Identity(), diffusion_model=None
    )
    with pytest.raises(ValueError, match="diffusion-mode"):
        CorrDiffCosmoEra5SDA(mean, assimilate_variables=("u10m", "v10m"))


def test_construct_defaults_inherit_from_model():
    model = _build(number_of_samples=3, number_of_steps=7)
    sda = CorrDiffCosmoEra5SDA(
        model,
        assimilate_variables=("u10m", "v10m"),
        sda_std_obs={"u10m": 0.5, "v10m": 1.5},
    )
    assert sda.number_of_samples == 3
    assert sda.sampler_steps == 7
    assert sda.amp is False  # off by default, matching the wrapped downscaler
    # scalar broadcasts / a mapping is stored per variable (order-independent dict)
    assert sda._obs_std == {"u10m": 0.5, "v10m": 1.5}
    assert CorrDiffCosmoEra5SDA(
        model, assimilate_variables=("u10m", "v10m"), sda_std_obs=0.7, amp=True
    )._obs_std == {"u10m": 0.7, "v10m": 0.7}


# ── Coordinate contracts ─────────────────────────────────────────────────────


def test_coordinate_contracts():
    sda = _build_sda(number_of_samples=2, assimilate_variables=("u10m", "v10m"))
    model = sda.model
    # init_coords: the ERA5 driving state (the grid the downscaler conditions on)
    (ic,) = sda.init_coords()
    assert list(ic["variable"]) == ERA5_VARIABLES
    assert "lat" in ic and "lon" in ic
    # input_coords: the observation DataFrame schema
    (fs,) = sda.input_coords()
    assert set(fs.keys()) == {"time", "lat", "lon", "observation", "variable"}
    assert list(fs["variable"]) == ["u10m", "v10m"]
    # output_coords: analysis dims, 2D lat/lon, and the model's output vocabulary
    coords = OrderedDict(
        time=np.array([np.datetime64("2021-07-14T12:00")]),
        variable=np.array(ERA5_VARIABLES),
        lat=model.lat_input_numpy,
        lon=model.lon_input_numpy,
    )
    (oc,) = sda.output_coords((coords,))
    assert list(oc.keys()) == ["time", "sample", "variable", "y", "x", "lat", "lon"]
    assert len(oc["sample"]) == 2
    assert np.asarray(oc["lat"]).ndim == 2 and np.asarray(oc["lon"]).ndim == 2
    assert list(oc["variable"]) == list(model._output_coord_variables)


# ── _build_obs_tensors ───────────────────────────────────────────────────────


def test_build_obs_tensors_direct_mapping():
    model = _build(
        out_center=torch.tensor([2.0, 3.0, 0.0, 0.0]),
        out_scale=torch.tensor([4.0, 5.0, 1.0, 1.0]),
    )
    sda = CorrDiffCosmoEra5SDA(
        model, assimilate_variables=("u10m", "v10m"), sda_std_obs=0.5
    )
    t = np.datetime64("2021-07-14T12:00")
    i, j = 4, 4
    obs = _obs_df(sda, [(i, j)], ["u10m"], [10.0], t)

    y, mask, std_y = sda._build_obs_tensors(obs, t, sda.device)

    n_out = len(model.output_variables)
    H, W = model.lat_output_numpy.shape
    assert y.shape == (1, n_out, H, W)
    assert mask.shape == (1, n_out, H, W)
    assert std_y.shape == (1, n_out, H, W)
    # direct: y = (obs - center) / scale = (10 - 2) / 4 = 2.0
    assert mask[0, 0, i, j] == 1
    assert torch.isclose(y[0, 0, i, j], torch.tensor(2.0))
    # std_y = sda_std_obs / scale at the obs cell, 1 elsewhere
    assert torch.isclose(std_y[0, 0, i, j], torch.tensor(0.125))
    assert std_y[0, 0, 0, 0] == 1.0
    # v10m was not observed
    assert mask[0, 1].sum() == 0


def test_build_obs_tensors_per_variable_std():
    # A per-variable sda_std_obs mapping applies each variable's own std/scale at its
    # observed cells (0.5/4 for u10m, 2.0/5 for v10m). assimilate_variables is reversed
    # relative to the mapping to prove the std is keyed by name, not by order.
    model = _build(
        out_center=torch.tensor([2.0, 3.0, 0.0, 0.0]),
        out_scale=torch.tensor([4.0, 5.0, 1.0, 1.0]),
    )
    sda = CorrDiffCosmoEra5SDA(
        model,
        assimilate_variables=("v10m", "u10m"),
        sda_std_obs={"u10m": 0.5, "v10m": 2.0},
    )
    t = np.datetime64("2021-07-14T12:00")
    obs = _obs_df(sda, [(4, 4), (4, 4)], ["u10m", "v10m"], [10.0, 20.0], t)

    _, _, std_y = sda._build_obs_tensors(obs, t, sda.device)

    assert torch.isclose(std_y[0, 0, 4, 4], torch.tensor(0.5 / 4.0))  # u10m -> 0.125
    assert torch.isclose(std_y[0, 1, 4, 4], torch.tensor(2.0 / 5.0))  # v10m -> 0.4
    assert std_y[0, 0, 0, 0] == 1.0


@pytest.mark.parametrize(
    "make_obs",
    [
        lambda sda, t: None,
        lambda sda, t: _obs_df(
            sda,
            [(4, 4), (4, 4)],
            ["u10m", "v10m"],
            [1.0, 2.0],
            t + np.timedelta64(1, "h"),  # outside the 10-min time window
        ),
        lambda sda, t: pd.DataFrame(  # outside the spatial domain
            {
                "time": pd.to_datetime([t, t]),
                "lat": [0.0, 0.0],
                "lon": [100.0, 100.0],
                "variable": ["u10m", "v10m"],
                "observation": [1.0, 2.0],
            }
        ),
    ],
    ids=["none", "out_of_time", "out_of_domain"],
)
def test_build_obs_tensors_empty_guidance(
    make_obs: Callable[[CorrDiffCosmoEra5SDA, np.datetime64], pd.DataFrame | None],
):
    # No usable observation (missing, outside the time window, or outside the domain)
    # -> an all-zero mask with zero y and unit std_y: a free (unconstrained) analysis.
    sda = _build_sda()
    t = np.datetime64("2021-07-14T12:00")
    y, mask, std_y = sda._build_obs_tensors(make_obs(sda, t), t, sda.device)
    assert (mask == 0).all()
    assert (y == 0).all()
    assert (std_y == 1).all()


def test_build_obs_tensors_drops_nonfinite():
    sda = _build_sda()
    t = np.datetime64("2021-07-14T12:00")
    # a NaN observation value is dropped while a finite obs in another channel is kept
    obs = _obs_df(sda, [(4, 4), (3, 3)], ["u10m", "v10m"], [np.nan, 2.0], t)
    _, mask, _ = sda._build_obs_tensors(obs, t, sda.device)
    assert mask[0, 0].sum() == 0  # u10m: NaN value -> dropped
    assert mask[0, 1].sum() == 1  # v10m: finite -> kept
    # a NaN coordinate is likewise dropped (would otherwise snap to a wrong cell)
    obs_ll = _obs_df(sda, [(4, 4), (3, 3)], ["u10m", "v10m"], [1.0, 2.0], t)
    obs_ll.loc[0, "lat"] = np.nan
    _, mask_ll, _ = sda._build_obs_tensors(obs_ll, t, sda.device)
    assert mask_ll[0, 0].sum() == 0  # u10m: NaN lat -> dropped
    assert mask_ll[0, 1].sum() == 1


def test_build_obs_tensors_skips_degenerate_scale():
    # a zero (or non-finite) out_scale for an assimilated channel is dropped rather
    # than divided by, so no inf/nan from (obs-c)/s or sda_std_obs/s reaches DPS.
    model = _build(
        out_center=torch.tensor([2.0, 3.0, 0.0, 0.0]),
        out_scale=torch.tensor([0.0, 5.0, 1.0, 1.0]),  # u10m scale = 0 (degenerate)
    )
    sda = CorrDiffCosmoEra5SDA(model, assimilate_variables=("u10m", "v10m"))
    t = np.datetime64("2021-07-14T12:00")
    obs = _obs_df(sda, [(4, 4), (4, 4)], ["u10m", "v10m"], [10.0, 8.0], t)
    y, mask, std_y = sda._build_obs_tensors(obs, t, sda.device)
    assert mask[0, 0].sum() == 0  # u10m: degenerate scale -> dropped
    assert mask[0, 1].sum() == 1  # v10m: finite scale -> kept
    assert torch.isfinite(y).all() and torch.isfinite(std_y).all()


def test_build_obs_tensors_averages_duplicates():
    model = _build(
        out_center=torch.tensor([2.0, 3.0, 0.0, 0.0]),
        out_scale=torch.tensor([4.0, 5.0, 1.0, 1.0]),
    )
    sda = CorrDiffCosmoEra5SDA(model, assimilate_variables=("u10m",))
    t = np.datetime64("2021-07-14T12:00")
    i, j = 3, 3
    obs = _obs_df(sda, [(i, j), (i, j)], ["u10m", "u10m"], [8.0, 12.0], t)

    y, mask, _ = sda._build_obs_tensors(obs, t, sda.device)

    assert mask[0, 0].sum() == 1
    # mean of ((8-2)/4, (12-2)/4) = mean(1.5, 2.5) = 2.0
    assert torch.isclose(y[0, 0, i, j], torch.tensor(2.0))


def test_build_obs_tensors_missing_column_raises():
    sda = _build_sda()
    t = np.datetime64("2021-07-14T12:00")
    obs = _obs_df(sda, [(4, 4)], ["u10m"], [1.0], t).drop(columns=["observation"])
    with pytest.raises(ValueError, match="missing required fields"):
        sda._build_obs_tensors(obs, t, sda.device)


# ── __call__ / create_generator ──────────────────────────────────────────────


def _identity_sda() -> CorrDiffCosmoEra5SDA:
    """A fast identity-channel SDA (no transforms) for full forward paths."""
    model = _build(output_variables=["U_10M", "V_10M"], channel_transforms={})
    return CorrDiffCosmoEra5SDA(
        model,
        assimilate_variables=("u10m", "v10m"),
        number_of_samples=2,
        sampler_steps=2,
    )


def test_call_returns_analysis():
    sda = _identity_sda()
    times = np.array([np.datetime64("2021-07-14T12:00")])
    x = _era5_da(sda, times)
    obs = _obs_df(sda, [(4, 4), (4, 4)], ["u10m", "v10m"], [1.0, 2.0], times[0])

    out = sda(x, obs)

    H, W = sda.model.lat_output_numpy.shape
    n_var = len(sda.model._output_coord_variables)
    assert set(out.dims) == {"time", "sample", "variable", "y", "x"}
    assert out.shape == (1, 2, n_var, H, W)
    assert np.isfinite(out.values).all()
    assert isinstance(out.data, np.ndarray)  # CPU default -> NumPy-backed DataArray

    # obs=None yields a same-shape free analysis; a size-1 lead_time is squeezed
    x_lead = x.expand_dims(lead_time=[np.timedelta64(0, "h")])
    out_none = sda(x_lead, None)
    assert out_none.shape == out.shape


def test_call_rejects_multi_lead_time():
    sda = _build_sda()
    times = np.array([np.datetime64("2021-07-14T12:00")])
    x = _era5_da(sda, times).expand_dims(
        lead_time=[np.timedelta64(0, "h"), np.timedelta64(1, "h")]
    )
    with pytest.raises(ValueError, match="lead_time"):
        sda(x, None)


def _analysis_output_inputs(
    sda: CorrDiffCosmoEra5SDA,
) -> tuple[torch.Tensor, np.ndarray, CoordSystem]:
    """A dummy (out, times, oc) triple for exercising ``_to_output_dataarray``."""
    times = np.array([np.datetime64("2021-07-14T12:00")])
    xf = sda._era5_frames(_era5_da(sda, times))
    x_coords = OrderedDict({dim: xf.coords[dim].values for dim in xf.dims})
    (oc,) = sda.output_coords((x_coords,))
    H, W = sda.model.lat_output_numpy.shape
    n_var = len(sda.model._output_coord_variables)
    return torch.zeros(1, 1, n_var, H, W), times, oc


def test_to_output_dataarray_requires_cupy_on_cuda(monkeypatch):
    # On CUDA the analysis must be returned on-device (CuPy); with CuPy absent this
    # must raise rather than silently fall back to CPU NumPy (a contract violation).
    # Exercised on CPU by faking a CUDA device and a missing cupy import.
    sda = _identity_sda()
    out, times, oc = _analysis_output_inputs(sda)
    monkeypatch.setattr("earth2studio.models.da.sda_corrdiff_cosmo_era5.cp", None)
    monkeypatch.setattr(
        CorrDiffCosmoEra5SDA, "device", property(lambda self: torch.device("cuda"))
    )
    with pytest.raises(RuntimeError, match="requires CuPy"):
        sda._to_output_dataarray(out, times, oc)


def test_forward_trims_halo():
    # The SDA owns the halo/patch trim (dx postprocess_output does not apply it), so
    # a nonzero halo must crop the returned analysis to the reported output grid.
    sda = _identity_sda()
    m = sda.model
    n_out = len(m.output_variables)
    H, W = m.lat_output_numpy.shape
    era5 = torch.randn(
        len(ERA5_VARIABLES), len(m.lat_input_numpy), len(m.lon_input_numpy)
    )
    y = torch.zeros(1, n_out, H, W)
    mask = torch.zeros_like(y)
    std_y = torch.ones_like(y)
    valid = datetime(2021, 7, 14, 12)
    untrimmed = sda._forward(era5, valid, y, mask, std_y)
    m._halo = (1, 1, 1, 1)
    trimmed = sda._forward(era5, valid, y, mask, std_y)
    assert trimmed.shape[-2] == untrimmed.shape[-2] - 2
    assert trimmed.shape[-1] == untrimmed.shape[-1] - 2


def test_create_generator():
    sda = _identity_sda()
    t0 = np.datetime64("2021-07-14T12:00")
    t1 = np.datetime64("2021-07-14T13:00")
    times = np.array([t0, t1])
    x = _era5_da(sda, times)

    gen = sda.create_generator(x)
    assert next(gen) is None  # prime with no compute

    obs0 = _obs_df(sda, [(4, 4)], ["u10m"], [1.0], t0)
    a0 = gen.send(obs0)
    assert set(a0.dims) == {"time", "sample", "variable", "y", "x"}
    assert a0.sizes["time"] == 1
    assert a0.coords["time"].values[0] == t0

    a1 = gen.send(None)
    assert a1.sizes["time"] == 1
    assert a1.coords["time"].values[0] == t1
    gen.close()


# ── load_model (mocked DX loader) ────────────────────────────────────────────
# The DX suite already covers metadata / NetCDF grid / checkpoint loading and model
# construction, so these mock CorrDiffCosmoEra5.load_model and assert only the SDA's
# own wiring (mode/resolution request, wrapping, arg forwarding, set_domain).


def test_load_model_wraps_diffusion():
    # Checks the SDA's own wiring: mode/resolution request, wrapping, arg forwarding.
    model = _build(output_variables=["U_10M", "V_10M"], channel_transforms={})
    with patch.object(CorrDiffCosmoEra5, "load_model", return_value=model) as mock_load:
        sda = CorrDiffCosmoEra5SDA.load_model(
            MagicMock(),
            assimilate_variables=("u10m", "v10m"),
            resolution="rea6",
            number_of_samples=3,
            sampler_steps=5,
            sda_gamma=2e-4,
            sda_std_obs={"u10m": 0.5, "v10m": 1.5},
            amp=True,
        )
    assert isinstance(sda, CorrDiffCosmoEra5SDA)
    assert sda.model is model
    assert mock_load.call_args.kwargs["mode"] == "diffusion"
    assert mock_load.call_args.kwargs["resolution"] == "rea6"
    assert sda.assimilate_variables == ["u10m", "v10m"]
    # every SDA-specific arg must reach the constructor (a dropped/swapped kwarg would
    # otherwise fall silently to a default)
    assert sda.number_of_samples == 3
    assert sda.sampler_steps == 5
    assert sda.sda_gamma == 2e-4
    assert sda._obs_std == {"u10m": 0.5, "v10m": 1.5}
    assert sda.amp is True


def test_load_model_domain_crops():
    # domain= runs the real set_domain on the loaded model BEFORE wrapping, so the obs
    # grid tracks the cropped downscaler (a stale grid would misplace observations).
    full = _build()
    with patch.object(CorrDiffCosmoEra5, "load_model", return_value=full):
        sda = CorrDiffCosmoEra5SDA.load_model(
            MagicMock(),
            assimilate_variables=("u10m", "v10m"),
            resolution="rea6",
            domain=dict(lat_min=48.0, lat_max=52.0, lon_min=8.5, lon_max=12.5),
        )
    assert sda.model is not full  # set_domain returned a cropped copy
    assert sda.model.lat_output_numpy.shape[0] < full.lat_output_numpy.shape[0]
    assert sda._lat_np.shape == sda.model.lat_output_numpy.shape


def test_to_moves_wrapped_model():
    # A real device transition without CUDA: moving the wrapper to the meta device must
    # move the wrapped downscaler's registered buffers.
    sda = _build_sda()
    returned = sda.to("meta")
    assert returned is sda
    assert sda.device.type == "meta"
    assert sda.model.lat_output_grid.device.type == "meta"


def test_device_follows_wrapped_model():
    # device is derived from the wrapped model's grid buffer (single source of truth),
    # so it tracks the model even when the model is moved directly -- e.g. wrapping a
    # placed model, or sda.model.to(...) after construction. Uses meta (no GPU needed).
    sda = _build_sda()
    assert sda.device.type == "cpu"
    sda.model.to("meta")
    assert sda.device.type == "meta"


# ── Package integration test (real weights) ──────────────────────────────────


@pytest.mark.package
def test_da_sda_corrdiff_cosmo_era5_package():
    pkg_path = os.environ.get("COSMO_REA_PACKAGE")
    if not pkg_path:
        pytest.skip("set COSMO_REA_PACKAGE to a built package dir to run")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sda = CorrDiffCosmoEra5SDA.load_model(
        Package(pkg_path),
        assimilate_variables=("u10m", "v10m"),
        resolution="rea6",
        number_of_samples=1,
        sampler_steps=2,
    ).to(device)

    # checkpoint loading is verified above; the NATTEN forward requires CUDA
    if not torch.cuda.is_available():
        pytest.skip("NATTEN forward requires CUDA; checkpoint load verified on CPU")

    (ic,) = sda.init_coords()
    times = np.array([np.datetime64("2021-07-14T12:00")])
    lat, lon = np.asarray(ic["lat"]), np.asarray(ic["lon"])
    x = xr.DataArray(
        data=np.random.default_rng(0)
        .standard_normal((len(times), len(ic["variable"]), len(lat), len(lon)))
        .astype(np.float32),
        dims=["time", "variable", "lat", "lon"],
        coords={
            "time": times,
            "variable": np.asarray(ic["variable"]),
            "lat": lat,
            "lon": lon,
        },
    )
    obs = _obs_df(sda, [(4, 4), (5, 5)], ["u10m", "v10m"], [3.0, -2.0], times[0])

    out = sda(x, obs)

    assert set(out.dims) == {"time", "sample", "variable", "y", "x"}
    assert out.sizes["time"] == 1 and out.sizes["sample"] == 1
    # out.data is the raw backing array (cupy on GPU when da-cosmo is installed,
    # numpy on CPU); out.values would force an implicit numpy conversion that cupy
    # rejects with a TypeError before the .get() guard is ever reached.
    values = out.data
    if hasattr(values, "get"):
        values = values.get()
    assert np.isfinite(values).all()
