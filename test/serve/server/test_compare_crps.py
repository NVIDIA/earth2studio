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

"""Tests for the CRPS comparison script (serve/server/scripts/compare_crps.py).

Merged from the original test_compare.py and test_data_loading.py.
We import functions directly from the script via importlib since it lives
outside the normal package hierarchy.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

# ---------------------------------------------------------------------------
# Import compare_crps module from serve/server/scripts/
# ---------------------------------------------------------------------------
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "serve"
    / "server"
    / "scripts"
    / "compare_crps.py"
)
_spec = importlib.util.spec_from_file_location("compare_crps", _SCRIPT_PATH)
assert _spec is not None
compare_crps = importlib.util.module_from_spec(_spec)
sys.modules["compare_crps"] = compare_crps
_spec.loader.exec_module(compare_crps)  # type: ignore[union-attr]

assert_compatible = compare_crps.assert_compatible
report_results = compare_crps.report_results
spatial_coords_from_dataset = compare_crps.spatial_coords_from_dataset
build_lead_time_chunks = compare_crps.build_lead_time_chunks
load_prediction_chunk = compare_crps.load_prediction_chunk
load_verification_chunk = compare_crps.load_verification_chunk

# ---------------------------------------------------------------------------
# Helpers from test_compare.py
# ---------------------------------------------------------------------------

def _make_ds(
    times: list[str],
    lead_times_h: list[int],
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Build a minimal dataset with the given time and lead_time coords."""
    variables = variables or ["t2m"]
    lat = np.linspace(90, -90, 4)
    lon = np.linspace(0, 360, 8, endpoint=False)
    times_np = np.array(times, dtype="datetime64[ns]")
    lead_times_np = np.array([np.timedelta64(h, "h") for h in lead_times_h])
    rng = np.random.default_rng(0)

    data_vars = {}
    for var in variables:
        data_vars[var] = xr.DataArray(
            rng.standard_normal(
                (len(times_np), 2, len(lead_times_np), len(lat), len(lon))
            ).astype(np.float32),
            dims=["time", "ensemble", "lead_time", "lat", "lon"],
            coords={
                "time": times_np,
                "ensemble": np.arange(2),
                "lead_time": lead_times_np,
                "lat": lat,
                "lon": lon,
            },
        )
    return xr.Dataset(data_vars)

# ---- assert_compatible ----

class TestAssertCompatible:
    """Tests for assert_compatible."""

    def test_identical_datasets_pass(self):
        ds = _make_ds(["2024-01-01"], [0, 6, 12])
        assert_compatible(ds, ds)

    def test_no_overlapping_times_raises(self):
        ds_a = _make_ds(["2024-01-01"], [0, 6])
        ds_b = _make_ds(["2024-02-01"], [0, 6])
        with pytest.raises(ValueError, match="no overlapping IC times"):
            assert_compatible(ds_a, ds_b)

    def test_different_lead_times_raises(self):
        ds_a = _make_ds(["2024-01-01"], [0, 6])
        ds_b = _make_ds(["2024-01-01"], [0, 12])
        with pytest.raises(ValueError, match="different lead_time values"):
            assert_compatible(ds_a, ds_b)

    def test_partial_time_overlap_warns(self, capsys):
        ds_a = _make_ds(["2024-01-01", "2024-01-02"], [0, 6])
        ds_b = _make_ds(["2024-01-01"], [0, 6])
        assert_compatible(ds_a, ds_b)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

# ---- report_results ----

class TestReportResults:
    """Tests for report_results."""

    def test_identical_scores_all_pass(self):
        scores = np.array([[[2.0, 3.0]]]).astype(np.float32)
        lead_times = np.array([np.timedelta64(6, "h")])
        result = report_results(scores, scores, ["t2m", "z500"], lead_times, 0.01)
        assert result is True

    def test_large_diff_fails(self):
        scores_a = np.array([[[10.0]]]).astype(np.float32)
        scores_b = np.array([[[5.0]]]).astype(np.float32)
        lead_times = np.array([np.timedelta64(6, "h")])
        result = report_results(scores_a, scores_b, ["t2m"], lead_times, 0.01)
        assert result is False

    def test_within_threshold_passes(self):
        scores_a = np.array([[[10.0]]]).astype(np.float32)
        scores_b = np.array([[[10.05]]]).astype(np.float32)
        lead_times = np.array([np.timedelta64(6, "h")])
        # Relative diff = 0.05/10.05 ≈ 0.497%, well within 1%
        result = report_results(scores_a, scores_b, ["t2m"], lead_times, 0.01)
        assert result is True

    def test_both_near_zero_passes(self):
        scores = np.array([[[0.0]]]).astype(np.float32)
        lead_times = np.array([np.timedelta64(0, "h")])
        result = report_results(scores, scores, ["t2m"], lead_times, 0.01)
        assert result is True

    def test_report_shows_percentage(self, capsys):
        scores_a = np.array([[[4.0]]]).astype(np.float32)
        scores_b = np.array([[[4.2]]]).astype(np.float32)
        lead_times = np.array([np.timedelta64(6, "h")])
        report_results(scores_a, scores_b, ["t2m"], lead_times, 0.10)
        captured = capsys.readouterr()
        assert "Rel Diff%" in captured.out
        assert "%" in captured.out

# ---------------------------------------------------------------------------
# Helpers from test_data_loading.py
# ---------------------------------------------------------------------------

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)

def _make_prediction_ds(
    n_ensemble: int = 2,
    n_lead_times: int = 3,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Build a minimal in-memory prediction dataset."""
    variables = variables or ["t2m", "z500"]
    times = [np.datetime64("2024-01-01")]
    lead_times = np.array([np.timedelta64(6 * i, "h") for i in range(n_lead_times)])
    rng = np.random.default_rng(42)

    data_vars = {}
    for var in variables:
        data_vars[var] = xr.DataArray(
            rng.standard_normal(
                (1, n_ensemble, n_lead_times, len(SMALL_LAT), len(SMALL_LON))
            ).astype(np.float32),
            dims=["time", "ensemble", "lead_time", "lat", "lon"],
            coords={
                "time": times,
                "ensemble": np.arange(n_ensemble),
                "lead_time": lead_times,
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            },
        )
    return xr.Dataset(data_vars)

# ---- spatial_coords_from_dataset ----

class TestSpatialCoordsFromDataset:
    """Tests for spatial_coords_from_dataset."""

    def test_extracts_lat_lon(self):
        ds = _make_prediction_ds()
        coords = spatial_coords_from_dataset(ds)
        assert list(coords.keys()) == ["lat", "lon"]
        np.testing.assert_array_equal(coords["lat"], SMALL_LAT)
        np.testing.assert_array_equal(coords["lon"], SMALL_LON)

    def test_excludes_non_spatial_dims(self):
        ds = _make_prediction_ds()
        coords = spatial_coords_from_dataset(ds)
        for dim in ("time", "ensemble", "lead_time"):
            assert dim not in coords

# ---- build_lead_time_chunks ----

class TestBuildLeadTimeChunks:
    """Tests for build_lead_time_chunks."""

    def test_no_chunking_when_none(self):
        lt = np.arange(10)
        chunks = build_lead_time_chunks(lt, None)
        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], lt)

    def test_no_chunking_when_zero(self):
        lt = np.arange(10)
        chunks = build_lead_time_chunks(lt, 0)
        assert len(chunks) == 1

    def test_no_chunking_when_size_exceeds_length(self):
        lt = np.arange(5)
        chunks = build_lead_time_chunks(lt, 100)
        assert len(chunks) == 1

    def test_even_split(self):
        lt = np.arange(6)
        chunks = build_lead_time_chunks(lt, 3)
        assert len(chunks) == 2
        np.testing.assert_array_equal(chunks[0], [0, 1, 2])
        np.testing.assert_array_equal(chunks[1], [3, 4, 5])

    def test_uneven_split(self):
        lt = np.arange(7)
        chunks = build_lead_time_chunks(lt, 3)
        assert len(chunks) == 3
        assert len(chunks[-1]) == 1

    def test_chunk_size_one(self):
        lt = np.arange(4)
        chunks = build_lead_time_chunks(lt, 1)
        assert len(chunks) == 4

# ---- load_prediction_chunk ----

class TestLoadPredictionChunk:
    """Tests for load_prediction_chunk."""

    def test_returns_correct_shape_with_ensemble(self):
        ds = _make_prediction_ds(n_ensemble=3, n_lead_times=4)
        time = np.datetime64("2024-01-01")
        lead_times = ds.lead_time.values[:2]
        tensor, coords = load_prediction_chunk(
            ds, time, lead_times, ["t2m", "z500"], torch.device("cpu")
        )
        assert tensor.shape == (3, 2, 2, len(SMALL_LAT), len(SMALL_LON))
        assert list(coords.keys()) == [
            "ensemble",
            "lead_time",
            "variable",
            "lat",
            "lon",
        ]

    def test_returns_float32(self):
        ds = _make_prediction_ds()
        time = np.datetime64("2024-01-01")
        lead_times = ds.lead_time.values[:1]
        tensor, _ = load_prediction_chunk(
            ds, time, lead_times, ["t2m"], torch.device("cpu")
        )
        assert tensor.dtype == torch.float32

# ---- load_verification_chunk ----

class TestLoadVerificationChunk:
    """Tests for load_verification_chunk."""

    def test_returns_correct_shape(self):
        spatial_coords = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        lead_times = np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
        time = np.datetime64("2024-01-01")

        def fake_source(times, variables):
            """Return a DataArray matching the GFS source interface."""
            return xr.DataArray(
                np.ones(
                    (len(times), len(variables), len(SMALL_LAT), len(SMALL_LON)),
                    dtype=np.float32,
                ),
                dims=["time", "variable", "lat", "lon"],
                coords={
                    "time": times,
                    "variable": variables,
                    "lat": SMALL_LAT,
                    "lon": SMALL_LON,
                },
            )

        tensor, coords = load_verification_chunk(
            fake_source, time, lead_times, ["t2m"], spatial_coords, torch.device("cpu")
        )
        assert tensor.shape == (2, 1, len(SMALL_LAT), len(SMALL_LON))
        assert list(coords.keys()) == ["lead_time", "variable", "lat", "lon"]

    def test_aligns_mismatched_grid(self):
        """Verification grid is larger; sel(method='nearest') subsets it."""
        pred_lat = np.array([80.0, 40.0, 0.0, -40.0])
        pred_lon = np.array([0.0, 90.0, 180.0, 270.0])
        spatial_coords = OrderedDict({"lat": pred_lat, "lon": pred_lon})
        lead_times = np.array([np.timedelta64(0, "h")])
        time = np.datetime64("2024-01-01")

        verif_lat = np.linspace(90, -90, 721)
        verif_lon = np.linspace(0, 359.75, 1440)

        def big_source(times, variables):
            return xr.DataArray(
                np.ones(
                    (len(times), len(variables), len(verif_lat), len(verif_lon)),
                    dtype=np.float32,
                ),
                dims=["time", "variable", "lat", "lon"],
                coords={
                    "time": times,
                    "variable": variables,
                    "lat": verif_lat,
                    "lon": verif_lon,
                },
            )

        tensor, coords = load_verification_chunk(
            big_source, time, lead_times, ["t2m"], spatial_coords, torch.device("cpu")
        )
        assert tensor.shape == (1, 1, len(pred_lat), len(pred_lon))
