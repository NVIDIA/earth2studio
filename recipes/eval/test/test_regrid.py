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

"""Tests for the recipe-local regridder primitives.

Covers :class:`NearestNeighborRegridder` (wrapping
``earth2studio.utils.interp.NearestNeighborInterpolator``) and the
:class:`RegriddedSource` DataSource adapter.  The tests use small
synthetic lat/lon grids so they run without any network I/O or
checkpoint loads.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import torch
import xarray as xr
from src.regrid import NearestNeighborRegridder, RegriddedSource

# ---------------------------------------------------------------------------
# Small helper: build a source / target grid pair whose nearest-neighbor
# mapping is trivial (identity within a distance threshold).
# ---------------------------------------------------------------------------


def _regular_grid(
    n_lat: int, n_lon: int, lat0: float = 30.0, lon0: float = 260.0, d: float = 0.5
):
    """Return 2D lat/lon meshgrids for a regular n_lat x n_lon grid."""
    lats_1d = np.linspace(lat0, lat0 + d * (n_lat - 1), n_lat, dtype=np.float32)
    lons_1d = np.linspace(lon0, lon0 + d * (n_lon - 1), n_lon, dtype=np.float32)
    lat2d, lon2d = np.meshgrid(lats_1d, lons_1d, indexing="ij")
    return lat2d, lon2d


class TestNearestNeighborRegridderTargetCoords:
    def test_target_coords_default_dim_names(self):
        src_lat, src_lon = _regular_grid(4, 5)
        tgt_lat, tgt_lon = _regular_grid(4, 5)
        regridder = NearestNeighborRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=tgt_lat,
            target_lons=tgt_lon,
            target_y=np.arange(4),
            target_x=np.arange(5),
            max_dist_km=100.0,
        )
        coords = regridder.target_coords()
        assert list(coords.keys()) == ["y", "x"]
        np.testing.assert_array_equal(coords["y"], np.arange(4))
        np.testing.assert_array_equal(coords["x"], np.arange(5))

    def test_target_coords_custom_dim_names(self):
        src_lat, src_lon = _regular_grid(4, 5)
        regridder = NearestNeighborRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=src_lat,
            target_lons=src_lon,
            target_y=np.linspace(30.0, 31.5, 4),
            target_x=np.linspace(260.0, 262.0, 5),
            max_dist_km=100.0,
            target_dim_names=("lat", "lon"),
        )
        coords = regridder.target_coords()
        assert list(coords.keys()) == ["lat", "lon"]


class TestNearestNeighborRegridderApply:
    def test_identity_mapping_preserves_values(self):
        """Source == target → every target pixel picks up the same-location source."""
        src_lat, src_lon = _regular_grid(4, 5)
        regridder = NearestNeighborRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=src_lat,
            target_lons=src_lon,
            target_y=np.arange(4),
            target_x=np.arange(5),
            max_dist_km=100.0,
        )

        # Fill a tensor with a unique id per pixel so the gather is
        # trivially inspectable.
        values = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
        out = regridder.apply(values, spatial_dims=("lat", "lon"))
        assert out.shape == (4, 5)
        # Identity → output equals input.
        torch.testing.assert_close(out, values)

    def test_leading_dims_preserved(self):
        src_lat, src_lon = _regular_grid(4, 5)
        regridder = NearestNeighborRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=src_lat,
            target_lons=src_lon,
            target_y=np.arange(4),
            target_x=np.arange(5),
            max_dist_km=100.0,
        )
        values = torch.zeros(2, 3, 4, 5)
        out = regridder.apply(values, spatial_dims=("lat", "lon"))
        assert out.shape == (2, 3, 4, 5)

    def test_rejects_non_pair_spatial_dims(self):
        src_lat, src_lon = _regular_grid(4, 5)
        regridder = NearestNeighborRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=src_lat,
            target_lons=src_lon,
            target_y=np.arange(4),
            target_x=np.arange(5),
            max_dist_km=100.0,
        )
        values = torch.zeros(4, 5)
        with pytest.raises(ValueError, match="exactly two trailing spatial dims"):
            regridder.apply(values, spatial_dims=("lat",))

    def test_out_of_range_target_filled_with_nan(self):
        """A target point far from the source grid should become NaN."""
        src_lat, src_lon = _regular_grid(4, 5, lat0=30.0, lon0=260.0, d=0.5)
        # Target grid includes one pixel far from any source point.
        tgt_lat = np.array([[30.0, 30.0], [30.0, 80.0]], dtype=np.float32)
        tgt_lon = np.array([[260.0, 260.5], [260.5, 0.0]], dtype=np.float32)
        regridder = NearestNeighborRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=tgt_lat,
            target_lons=tgt_lon,
            target_y=np.arange(2),
            target_x=np.arange(2),
            max_dist_km=100.0,
        )
        values = torch.ones(4, 5, dtype=torch.float32)
        out = regridder.apply(values, spatial_dims=("lat", "lon"))
        assert torch.isnan(out[1, 1]), "far-away target should be NaN"
        assert not torch.isnan(out[0, 0])


class TestNearestNeighborRegridderDataArray:
    def test_apply_dataarray_swaps_spatial_dim_names(self):
        src_lat, src_lon = _regular_grid(4, 5)
        regridder = NearestNeighborRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=src_lat,
            target_lons=src_lon,
            target_y=np.arange(4),
            target_x=np.arange(5),
            max_dist_km=100.0,
        )
        data = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
        da = xr.DataArray(
            data,
            dims=("time", "lat", "lon"),
            coords={
                "time": np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[ns]"),
                "lat": np.arange(4),
                "lon": np.arange(5),
            },
        )
        out = regridder.apply_dataarray(da)
        assert out.dims == ("time", "y", "x")
        assert out.shape == (2, 4, 5)
        np.testing.assert_allclose(out.values, data)


# ---------------------------------------------------------------------------
# RegriddedSource
# ---------------------------------------------------------------------------


class _FakeSource:
    """Minimal DataSource that returns a DataArray on a 4x5 lat/lon grid."""

    def __init__(self, lat2d: np.ndarray, lon2d: np.ndarray) -> None:
        self._lat = lat2d
        self._lon = lon2d
        self.call_count = 0

    def __call__(self, time, variable):
        self.call_count += 1
        t = np.atleast_1d(time)
        v = np.atleast_1d(variable)
        h, w = self._lat.shape
        data = np.broadcast_to(
            np.arange(h * w, dtype=np.float32).reshape(1, 1, h, w),
            (len(t), len(v), h, w),
        ).copy()
        return xr.DataArray(
            data,
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": t,
                "variable": v,
                "lat": self._lat[:, 0],
                "lon": self._lon[0, :],
            },
        )

    async def fetch(self, time, variable):
        return self(time, variable)


class TestRegriddedSource:
    def test_regridded_source_applies_regridder(self):
        src_lat, src_lon = _regular_grid(4, 5)
        regridder = NearestNeighborRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=src_lat,
            target_lons=src_lon,
            target_y=np.arange(4),
            target_x=np.arange(5),
            max_dist_km=100.0,
        )
        raw_source = _FakeSource(src_lat, src_lon)
        wrapped = RegriddedSource(raw_source, regridder)

        out = wrapped(datetime(2023, 1, 1), ["varA"])
        assert raw_source.call_count == 1
        # Output spatial dims renamed from (lat, lon) to (y, x).
        assert out.dims == ("time", "variable", "y", "x")
        assert out.shape == (1, 1, 4, 5)

    def test_regridded_source_preserves_time_coord(self):
        src_lat, src_lon = _regular_grid(4, 5)
        regridder = NearestNeighborRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=src_lat,
            target_lons=src_lon,
            target_y=np.arange(4),
            target_x=np.arange(5),
            max_dist_km=100.0,
        )
        raw_source = _FakeSource(src_lat, src_lon)
        wrapped = RegriddedSource(raw_source, regridder)

        t = np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[ns]")
        out = wrapped(t, ["varA", "varB"])
        np.testing.assert_array_equal(out["time"].values, t)
        np.testing.assert_array_equal(out["variable"].values, ["varA", "varB"])
