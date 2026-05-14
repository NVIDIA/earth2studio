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

"""Tests for the CRPS comparison script (compare.py)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# compare.py lives at recipes/eval/compare.py and is not a package —
# import it by manipulating sys.path so `from compare import ...` works.
_EVAL_DIR = str(Path(__file__).resolve().parents[1])
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

compare = importlib.import_module("compare")
assert_compatible = compare.assert_compatible
report_results = compare.report_results


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
