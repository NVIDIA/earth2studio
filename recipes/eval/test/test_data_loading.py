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

"""Tests for the shared data-loading utilities (src/data_loading.py)."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import xarray as xr
from src.data_loading import (
    build_lead_time_chunks,
    load_prediction_chunk,
    load_verification_chunk,
    spatial_coords_from_dataset,
)

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
