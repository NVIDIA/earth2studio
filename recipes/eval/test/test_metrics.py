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

"""Tests for the recipe-local xarray metrics (src/metrics.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr
from src.metrics import ensemble_variance, mse
from src.scoring import _is_statistic, validate_lead_time_chunking

from earth2studio.statistics import Metric, Statistic, rmse, variance
from earth2studio.statistics.weights import lat_weight
from earth2studio.utils.coords import coord_array

from .conftest import SMALL_LAT, SMALL_LON

LEAD_TIMES = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
VARIABLES = np.array(["t2m", "z500"])
ENSEMBLE = np.arange(4)


def _field(*, ensemble: bool = False, seed: int = 0) -> xr.DataArray:
    """Random forecast on the small grid, optionally with an ensemble axis."""
    dims = (["ensemble"] if ensemble else []) + ["lead_time", "variable", "lat", "lon"]
    coords = {
        "lead_time": LEAD_TIMES,
        "variable": VARIABLES,
        "lat": SMALL_LAT,
        "lon": SMALL_LON,
    }
    if ensemble:
        coords["ensemble"] = ENSEMBLE
    shape = [len(coords[d]) for d in dims]
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.standard_normal(shape).astype("float32"), dims=dims, coords=coords
    )


def _signature(*, ensemble: bool = False):
    da = _field(ensemble=ensemble)
    return coord_array(da.dims, {k: v.values for k, v in da.coords.items()})


# ---------------------------------------------------------------------------
# Protocol dispatch
# ---------------------------------------------------------------------------


class TestProtocols:
    def test_mse_is_metric(self):
        m = mse(["lat", "lon"])
        assert isinstance(m, Metric)
        assert not _is_statistic(m)

    def test_ensemble_variance_is_statistic(self):
        s = ensemble_variance("ensemble", ["lat", "lon"])
        assert isinstance(s, Statistic)
        assert _is_statistic(s)

    def test_library_classes(self):
        assert not _is_statistic(rmse(["lat", "lon"]))
        assert _is_statistic(variance(["lat", "lon"]))


# ---------------------------------------------------------------------------
# mse
# ---------------------------------------------------------------------------


class TestMSE:
    def test_output_coords_deterministic(self):
        out = mse(["lat", "lon"]).output_coords(_signature())
        assert tuple(out.dims) == ("lead_time", "variable")
        np.testing.assert_array_equal(out["lead_time"].values, LEAD_TIMES)

    def test_output_coords_with_ensemble(self):
        out = mse(["lat", "lon"], ensemble_dimension="ensemble").output_coords(
            _signature(ensemble=True)
        )
        assert tuple(out.dims) == ("lead_time", "variable")

    def test_output_coords_keeps_ensemble_without_reduction(self):
        out = mse(["lat", "lon"]).output_coords(_signature(ensemble=True))
        assert tuple(out.dims) == ("ensemble", "lead_time", "variable")

    def test_known_value(self):
        x = _field()
        y = x + 2.0
        out = mse(["lat", "lon"])(x, y)
        assert isinstance(out, xr.DataArray)
        assert tuple(out.dims) == ("lead_time", "variable")
        np.testing.assert_allclose(out.values, 4.0, rtol=1e-6)

    def test_no_sqrt(self):
        x, y = _field(seed=1), _field(seed=2)
        got = mse(["lat", "lon"])(x, y)
        root = rmse(["lat", "lon"])(x, y)
        np.testing.assert_allclose(got.values, root.values**2, rtol=1e-5)
        expected = ((x - y) ** 2).mean(["lat", "lon"])
        np.testing.assert_allclose(got.values, expected.values, rtol=1e-5)

    def test_ensemble_mean_before_mse(self):
        x = _field(ensemble=True)
        y = x.mean("ensemble") + 1.0
        out = mse(["lat", "lon"], ensemble_dimension="ensemble")(x, y)
        assert tuple(out.dims) == ("lead_time", "variable")
        np.testing.assert_allclose(out.values, 1.0, rtol=1e-5)

    def test_rejects_misaligned_observation(self):
        x = _field()
        y = _field().isel(lat=slice(0, 2))
        with pytest.raises(ValueError):
            mse(["lat", "lon"])(x, y)

    def test_with_lat_weights(self):
        x = _field()
        y = xr.zeros_like(x)
        w = lat_weight(torch.tensor(SMALL_LAT, dtype=torch.float32))
        weights = torch.ones(len(SMALL_LAT), len(SMALL_LON)) * w[:, None]
        out = mse(["lat", "lon"], weights=weights)(x, y)
        expected = (x**2 * w.numpy()[:, None]).sum(["lat", "lon"]) / (
            w.sum().item() * len(SMALL_LON)
        )
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-5)

    def test_reduction_dimensions_and_str(self):
        m = mse(["lat", "lon"])
        assert m.reduction_dimensions == ["lat", "lon"]
        assert str(m) == "lat_lon_mse"


# ---------------------------------------------------------------------------
# ensemble_variance
# ---------------------------------------------------------------------------


class TestEnsembleVariance:
    def test_output_coords(self):
        out = ensemble_variance("ensemble", ["lat", "lon"]).output_coords(
            _signature(ensemble=True)
        )
        assert tuple(out.dims) == ("lead_time", "variable")

    def test_reduction_dimensions_include_ensemble(self):
        s = ensemble_variance("ensemble", ["lat", "lon"])
        assert s.reduction_dimensions == ["ensemble", "lat", "lon"]

    def test_known_variance(self):
        # Members at base + k for k = 0..3: variance (N-1) of {0,1,2,3} = 5/3.
        base = _field()
        x = xr.concat(
            [base + float(k) for k in range(4)],
            dim=xr.DataArray(ENSEMBLE, dims="ensemble"),
        ).transpose("ensemble", ...)
        out = ensemble_variance("ensemble", ["lat", "lon"])(x)
        assert tuple(out.dims) == ("lead_time", "variable")
        np.testing.assert_allclose(out.values, 5.0 / 3.0, rtol=1e-5)

    def test_matches_xarray(self):
        x = _field(ensemble=True)
        out = ensemble_variance("ensemble", ["lat", "lon"])(x)
        expected = x.var("ensemble", ddof=1).mean(["lat", "lon"])
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-4)

    def test_with_weights(self):
        x = _field(ensemble=True)
        weights = torch.zeros(len(SMALL_LAT), len(SMALL_LON))
        weights[0] = 1.0  # only the first latitude row counts
        out = ensemble_variance("ensemble", ["lat", "lon"], weights=weights)(x)
        expected = x.var("ensemble", ddof=1).isel(lat=0).mean("lon")
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-4)


# ---------------------------------------------------------------------------
# Spread-skill round trip through the report aggregation
# ---------------------------------------------------------------------------


class TestSpreadSkillRoundTrip:
    def _scores(self, mse_vals, var_vals):
        n_times, n_lead = mse_vals.shape
        times = np.arange(n_times).astype("datetime64[D]").astype("datetime64[ns]")
        leads = np.arange(n_lead).astype("timedelta64[D]").astype("timedelta64[ns]")
        coords = {"time": times, "lead_time": leads}
        return xr.Dataset(
            {
                "ensemble_mean_mse__t2m": xr.DataArray(
                    mse_vals, dims=["time", "lead_time"], coords=coords
                ),
                "ensemble_variance__t2m": xr.DataArray(
                    var_vals, dims=["time", "lead_time"], coords=coords
                ),
            }
        )

    def test_known_ratio(self):
        from src.report import compute_spread_skill

        rng = np.random.default_rng(0)
        mse_vals = rng.uniform(1.0, 9.0, (20, 5))
        var_vals = rng.uniform(1.0, 9.0, (20, 5))
        ds = self._scores(mse_vals, var_vals)
        out = compute_spread_skill(
            ds, "ensemble_mean_mse", "ensemble_variance", ["t2m"]
        )
        np.testing.assert_allclose(out["rmse"]["t2m"].values, np.sqrt(mse_vals.mean(0)))
        np.testing.assert_allclose(
            out["spread"]["t2m"].values, np.sqrt(var_vals.mean(0))
        )
        np.testing.assert_allclose(
            out["ratio"]["t2m"].values,
            np.sqrt(var_vals.mean(0)) / np.sqrt(mse_vals.mean(0)),
        )

    def test_equal_mse_and_variance_gives_ratio_one(self):
        from src.report import compute_spread_skill

        vals = np.full((4, 3), 2.5)
        out = compute_spread_skill(
            self._scores(vals, vals), "ensemble_mean_mse", "ensemble_variance", ["t2m"]
        )
        np.testing.assert_allclose(out["ratio"]["t2m"].values, 1.0)

    def test_missing_arrays_produce_empty(self):
        from src.report import compute_spread_skill

        ds = self._scores(np.ones((2, 2)), np.ones((2, 2)))
        out = compute_spread_skill(
            ds, "ensemble_mean_mse", "ensemble_variance", ["z500"]
        )
        assert len(out["rmse"].data_vars) == 0


# ---------------------------------------------------------------------------
# Lead-time chunking compatibility
# ---------------------------------------------------------------------------


class TestLeadTimeChunkingCompat:
    def test_spatial_reductions_allow_chunking(self):
        metrics = {
            "mse": mse(["lat", "lon"]),
            "ensemble_variance": ensemble_variance("ensemble", ["lat", "lon"]),
        }
        validate_lead_time_chunking(metrics, chunk_size=2, n_lead_times=10)

    def test_lead_time_reduction_rejects_chunking(self):
        with pytest.raises(ValueError, match="lead_time"):
            validate_lead_time_chunking(
                {"mse": mse(["lead_time", "lat", "lon"])}, chunk_size=2, n_lead_times=10
            )
