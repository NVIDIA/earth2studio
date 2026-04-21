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

"""Tests for recipe-local metrics (src/metrics.py) and Statistic dispatch."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from src.metrics import ensemble_variance, mse
from src.scoring import _is_statistic

from earth2studio.utils.coords import CoordSystem

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)
LEAD_TIMES = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
VARIABLES = np.array(["t2m", "z500"])
ENSEMBLE = np.arange(4)


def _make_coords(*, ensemble: bool = False) -> CoordSystem:
    coords: CoordSystem = OrderedDict()
    if ensemble:
        coords["ensemble"] = ENSEMBLE
    coords["lead_time"] = LEAD_TIMES
    coords["variable"] = VARIABLES
    coords["lat"] = SMALL_LAT
    coords["lon"] = SMALL_LON
    return coords


def _make_tensor(coords: CoordSystem) -> torch.Tensor:
    shape = [len(v) for v in coords.values()]
    return torch.randn(shape)


# ---------------------------------------------------------------------------
# _is_statistic dispatch helper
# ---------------------------------------------------------------------------


class TestIsStatistic:
    def test_mse_is_metric(self):
        m = mse(reduction_dimensions=["lat", "lon"])
        assert not _is_statistic(m)

    def test_ensemble_variance_is_statistic(self):
        s = ensemble_variance(
            ensemble_dimension="ensemble",
            reduction_dimensions=["lat", "lon"],
        )
        assert _is_statistic(s)

    def test_e2s_rmse_is_metric(self):
        from earth2studio.statistics import rmse

        m = rmse(reduction_dimensions=["lat", "lon"])
        assert not _is_statistic(m)

    def test_e2s_variance_is_statistic(self):
        from earth2studio.statistics import variance

        s = variance(reduction_dimensions=["lat"])
        assert _is_statistic(s)


# ---------------------------------------------------------------------------
# mse
# ---------------------------------------------------------------------------


class TestMSE:
    def test_output_coords_deterministic(self):
        m = mse(reduction_dimensions=["lat", "lon"])
        coords = _make_coords(ensemble=False)
        out = m.output_coords(coords)
        assert "lead_time" in out
        assert "variable" in out
        assert "lat" not in out
        assert "lon" not in out

    def test_output_coords_with_ensemble(self):
        m = mse(
            reduction_dimensions=["lat", "lon"],
            ensemble_dimension="ensemble",
        )
        coords = _make_coords(ensemble=True)
        out = m.output_coords(coords)
        assert "ensemble" not in out
        assert "lead_time" in out
        assert "variable" in out
        assert "lat" not in out

    def test_basic_computation(self):
        """MSE of identical tensors should be zero."""
        m = mse(reduction_dimensions=["lat", "lon"])
        coords = _make_coords(ensemble=False)
        x = _make_tensor(coords)
        y = x.clone()
        result, out_coords = m(x, coords, y, coords)
        assert result.shape == (len(LEAD_TIMES), len(VARIABLES))
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=0)

    def test_known_value(self):
        """Hand-computed MSE for a simple case."""
        # 1x1 spatial grid, 1 lead time, 1 variable
        coords: CoordSystem = OrderedDict(
            lead_time=np.array([0], dtype="timedelta64[ns]"),
            variable=np.array(["t2m"]),
            lat=np.array([0.0]),
            lon=np.array([0.0]),
        )
        x = torch.tensor([[[[3.0]]]])  # (1, 1, 1, 1)
        y = torch.tensor([[[[1.0]]]])
        m = mse(reduction_dimensions=["lat", "lon"])
        result, _ = m(x, coords, y, coords)
        # MSE = (3-1)^2 = 4.0
        torch.testing.assert_close(result, torch.tensor([[4.0]]), atol=1e-6, rtol=0)

    def test_no_sqrt(self):
        """MSE should NOT take the square root (unlike RMSE)."""
        from earth2studio.statistics import rmse as e2s_rmse

        coords = _make_coords(ensemble=False)
        x = _make_tensor(coords)
        y = _make_tensor(coords)

        m = mse(reduction_dimensions=["lat", "lon"])
        rmse_metric = e2s_rmse(reduction_dimensions=["lat", "lon"])

        mse_val, _ = m(x, coords, y, coords)
        rmse_val, _ = rmse_metric(x, coords, y, coords)

        # RMSE = sqrt(MSE) → MSE = RMSE²
        torch.testing.assert_close(mse_val, rmse_val**2, atol=1e-5, rtol=1e-5)

    def test_ensemble_mean_before_mse(self):
        """With ensemble_dimension, should average members first."""
        coords = _make_coords(ensemble=True)
        # Shape: (4, 3, 2, 4, 8) = (ens, lt, var, lat, lon)
        torch.manual_seed(42)
        x = torch.randn([len(v) for v in coords.values()])

        obs_coords = _make_coords(ensemble=False)
        y = torch.randn([len(v) for v in obs_coords.values()])

        m = mse(
            reduction_dimensions=["lat", "lon"],
            ensemble_dimension="ensemble",
        )
        result, out_coords = m(x, coords, y, obs_coords)

        # Output should have no ensemble dimension.
        assert "ensemble" not in out_coords
        assert result.shape == (len(LEAD_TIMES), len(VARIABLES))

        # Verify: manually compute ensemble mean, then MSE.
        x_mean = x.mean(dim=0)  # (lt, var, lat, lon)
        diff_sq = (x_mean - y) ** 2
        expected = diff_sq.mean(dim=(-2, -1))  # average over lat, lon
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_with_weights(self):
        """Weighted MSE should differ from unweighted."""
        coords = _make_coords(ensemble=False)
        x = _make_tensor(coords)
        y = _make_tensor(coords)

        unweighted = mse(reduction_dimensions=["lat", "lon"])
        weights = torch.ones(len(SMALL_LAT), len(SMALL_LON))
        weights[0, :] = 10.0  # heavily weight first latitude
        weighted = mse(reduction_dimensions=["lat", "lon"], weights=weights)

        r_unw, _ = unweighted(x, coords, y, coords)
        r_w, _ = weighted(x, coords, y, coords)

        # Results should differ (unless pathologically equal, which is
        # astronomically unlikely with random data).
        assert not torch.allclose(r_unw, r_w)

    def test_reduction_dimensions_property(self):
        m = mse(reduction_dimensions=["lat", "lon"])
        assert m.reduction_dimensions == ["lat", "lon"]


# ---------------------------------------------------------------------------
# ensemble_variance
# ---------------------------------------------------------------------------


class TestEnsembleVariance:
    def test_output_coords(self):
        ev = ensemble_variance(
            ensemble_dimension="ensemble",
            reduction_dimensions=["lat", "lon"],
        )
        coords = _make_coords(ensemble=True)
        out = ev.output_coords(coords)
        assert "ensemble" not in out
        assert "lat" not in out
        assert "lon" not in out
        assert "lead_time" in out
        assert "variable" in out

    def test_basic_computation(self):
        """Constant ensemble should give zero variance."""
        ev = ensemble_variance(
            ensemble_dimension="ensemble",
            reduction_dimensions=["lat", "lon"],
        )
        coords = _make_coords(ensemble=True)
        # All ensemble members identical.
        single = torch.randn(
            1, len(LEAD_TIMES), len(VARIABLES), len(SMALL_LAT), len(SMALL_LON)
        )
        x = single.expand(len(ENSEMBLE), -1, -1, -1, -1).contiguous()
        result, out_coords = ev(x, coords)
        assert result.shape == (len(LEAD_TIMES), len(VARIABLES))
        torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-5, rtol=0)

    def test_known_variance(self):
        """Hand-computed variance for 2 members on a 1x1 grid."""
        coords: CoordSystem = OrderedDict(
            ensemble=np.arange(2),
            lead_time=np.array([0], dtype="timedelta64[ns]"),
            variable=np.array(["t2m"]),
            lat=np.array([0.0]),
            lon=np.array([0.0]),
        )
        # Members: 1.0 and 3.0 → mean=2, var(Bessel)=(1+1)/1=2.0
        x = torch.tensor([[[[[1.0]]]], [[[[3.0]]]]])
        ev = ensemble_variance(
            ensemble_dimension="ensemble",
            reduction_dimensions=["lat", "lon"],
        )
        result, _ = ev(x, coords)
        # Bessel-corrected variance: sum((xi - mean)^2) / (N-1)
        # = ((1-2)^2 + (3-2)^2) / 1 = 2.0
        torch.testing.assert_close(result, torch.tensor([[2.0]]), atol=1e-5, rtol=0)

    def test_statistic_protocol(self):
        """ensemble_variance follows the 2-arg Statistic protocol."""
        ev = ensemble_variance(
            ensemble_dimension="ensemble",
            reduction_dimensions=["lat", "lon"],
        )
        coords = _make_coords(ensemble=True)
        x = _make_tensor(coords)
        # Call with exactly 2 args (Statistic protocol).
        result, out_coords = ev(x, coords)
        assert result.shape == (len(LEAD_TIMES), len(VARIABLES))

    def test_reduction_dimensions_includes_ensemble(self):
        ev = ensemble_variance(
            ensemble_dimension="ensemble",
            reduction_dimensions=["lat", "lon"],
        )
        assert ev.reduction_dimensions == ["ensemble", "lat", "lon"]

    def test_with_weights(self):
        """Weighted spatial mean of variance should differ from unweighted."""
        coords = _make_coords(ensemble=True)
        x = _make_tensor(coords)

        unweighted = ensemble_variance(
            ensemble_dimension="ensemble",
            reduction_dimensions=["lat", "lon"],
        )
        weights = torch.ones(len(SMALL_LAT), len(SMALL_LON))
        weights[0, :] = 10.0
        weighted = ensemble_variance(
            ensemble_dimension="ensemble",
            reduction_dimensions=["lat", "lon"],
            weights=weights,
        )

        r_unw, _ = unweighted(x, coords)
        r_w, _ = weighted(x, coords)
        assert not torch.allclose(r_unw, r_w)


# ---------------------------------------------------------------------------
# Spread-skill round-trip
# ---------------------------------------------------------------------------


class TestSpreadSkillRoundTrip:
    def test_known_ratio(self):
        """Deterministic test: known MSE and variance → exact R."""
        import xarray as xr
        from src.report import compute_spread_skill

        n_times = 20
        n_lead = 5
        rng = np.random.default_rng(0)

        # Use varying-but-positive MSE and variance values so the
        # aggregation (mean over time, then sqrt) is non-trivial.
        mse_vals = rng.uniform(1.0, 9.0, (n_times, n_lead))
        var_vals = rng.uniform(1.0, 9.0, (n_times, n_lead))

        times = np.arange(n_times).astype("datetime64[D]").astype("datetime64[ns]")
        lead_times = (
            np.arange(n_lead).astype("timedelta64[D]").astype("timedelta64[ns]")
        )

        ds = xr.Dataset(
            {
                "ensemble_mean_mse__t2m": xr.DataArray(
                    mse_vals.astype("float64"),
                    dims=["time", "lead_time"],
                    coords={"time": times, "lead_time": lead_times},
                ),
                "ensemble_variance__t2m": xr.DataArray(
                    var_vals.astype("float64"),
                    dims=["time", "lead_time"],
                    coords={"time": times, "lead_time": lead_times},
                ),
            }
        )

        result = compute_spread_skill(
            ds,
            mse_metric="ensemble_mean_mse",
            variance_metric="ensemble_variance",
            variables=["t2m"],
        )

        # Verify the WB2 formulas:
        #   RMSE = sqrt(mean_t(MSE)), Spread = sqrt(mean_t(Var))
        #   R = Spread / RMSE
        expected_rmse = np.sqrt(mse_vals.mean(axis=0))
        expected_spread = np.sqrt(var_vals.mean(axis=0))
        expected_ratio = expected_spread / expected_rmse

        np.testing.assert_allclose(
            result["rmse"]["t2m"].values, expected_rmse, rtol=1e-5
        )
        np.testing.assert_allclose(
            result["spread"]["t2m"].values, expected_spread, rtol=1e-5
        )
        np.testing.assert_allclose(
            result["ratio"]["t2m"].values, expected_ratio, rtol=1e-5
        )

    def test_equal_mse_and_variance_gives_ratio_one(self):
        """When MSE == Variance everywhere, R should be exactly 1."""
        import xarray as xr
        from src.report import compute_spread_skill

        n_times = 10
        n_lead = 3
        vals = np.full((n_times, n_lead), 4.0)

        times = np.arange(n_times).astype("datetime64[D]").astype("datetime64[ns]")
        lead_times = (
            np.arange(n_lead).astype("timedelta64[D]").astype("timedelta64[ns]")
        )

        ds = xr.Dataset(
            {
                "ensemble_mean_mse__t2m": xr.DataArray(
                    vals,
                    dims=["time", "lead_time"],
                    coords={"time": times, "lead_time": lead_times},
                ),
                "ensemble_variance__t2m": xr.DataArray(
                    vals,
                    dims=["time", "lead_time"],
                    coords={"time": times, "lead_time": lead_times},
                ),
            }
        )

        result = compute_spread_skill(
            ds, "ensemble_mean_mse", "ensemble_variance", ["t2m"]
        )
        np.testing.assert_allclose(result["ratio"]["t2m"].values, 1.0, atol=1e-7)

    def test_missing_arrays_produces_empty(self):
        """When arrays are missing, result datasets should be empty."""
        import xarray as xr
        from src.report import compute_spread_skill

        ds = xr.Dataset()
        result = compute_spread_skill(
            ds,
            mse_metric="ensemble_mean_mse",
            variance_metric="ensemble_variance",
            variables=["t2m"],
        )
        assert len(result["rmse"].data_vars) == 0
        assert len(result["spread"].data_vars) == 0
        assert len(result["ratio"].data_vars) == 0


# ---------------------------------------------------------------------------
# Lead-time chunking compatibility
# ---------------------------------------------------------------------------


class TestLeadTimeChunkingCompat:
    def test_mse_allows_chunking(self):
        from src.scoring import validate_lead_time_chunking

        metrics = OrderedDict(
            {
                "mse": mse(reduction_dimensions=["lat", "lon"]),
            }
        )
        # Should not raise.
        validate_lead_time_chunking(metrics, 2, 10)

    def test_ensemble_variance_allows_chunking(self):
        from src.scoring import validate_lead_time_chunking

        metrics = OrderedDict(
            {
                "ev": ensemble_variance(
                    ensemble_dimension="ensemble",
                    reduction_dimensions=["lat", "lon"],
                ),
            }
        )
        # Should not raise.
        validate_lead_time_chunking(metrics, 2, 10)
