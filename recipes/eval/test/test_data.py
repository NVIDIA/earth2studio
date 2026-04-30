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

"""Tests for PredownloadedSource (src/data.py)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from src.data import PredownloadedSource


def _create_zarr_store(path, times, variables, lat, lon):
    """Write a minimal zarr store with the predownload schema."""
    ds = xr.Dataset()
    for var in variables:
        data = np.random.default_rng(42).standard_normal(
            (len(times), len(lat), len(lon))
        )
        ds[var] = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": lat, "lon": lon},
        )
    ds.to_zarr(str(path))
    return ds


class TestPredownloadedSource:
    @pytest.fixture()
    def store_path(self, tmp_path):
        return tmp_path / "data.zarr"

    @pytest.fixture()
    def times(self):
        return np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
        )

    @pytest.fixture()
    def variables(self):
        return ["t2m", "z500"]

    @pytest.fixture()
    def lat(self):
        return np.array([90.0, 0.0, -90.0])

    @pytest.fixture()
    def lon(self):
        return np.array([0.0, 90.0, 180.0, 270.0])

    @pytest.fixture()
    def ds(self, store_path, times, variables, lat, lon):
        return _create_zarr_store(store_path, times, variables, lat, lon)

    @pytest.fixture()
    def source(self, ds, store_path):
        return PredownloadedSource(str(store_path))

    def test_call_returns_dataarray(self, source, times):
        result = source(times[:1], ["t2m"])
        assert isinstance(result, xr.DataArray)

    def test_select_single_time_and_variable(self, source):
        result = source(np.array(["2024-01-01"], dtype="datetime64[ns]"), ["t2m"])
        assert result.dims[0] == "time"
        assert result.dims[1] == "variable"
        assert len(result.time) == 1
        assert result.sizes["variable"] == 1

    def test_select_multiple_times(self, source, times):
        result = source(times, ["t2m", "z500"])
        assert len(result.time) == 3
        assert result.sizes["variable"] == 2

    def test_scalar_inputs(self, source):
        """Scalar time and variable are promoted to lists internally."""
        result = source(np.datetime64("2024-01-01", "ns"), "t2m")
        assert len(result.time) == 1
        assert result.sizes["variable"] == 1

    def test_data_values_match_written(self, source, ds):
        result = source(np.array(["2024-01-01"], dtype="datetime64[ns]"), ["t2m"])
        expected = ds["t2m"].sel(time="2024-01-01").values
        np.testing.assert_array_equal(
            result.sel(variable="t2m").values.squeeze(), expected
        )

    def test_async_fetch(self, source, times):
        import asyncio

        result = asyncio.run(source.fetch(times[:1], ["t2m"]))
        assert isinstance(result, xr.DataArray)

    def test_missing_time_raises(self, source):
        with pytest.raises(KeyError):
            source(np.array(["2099-01-01"], dtype="datetime64[ns]"), ["t2m"])

    def test_missing_variable_raises(self, source, times):
        with pytest.raises(KeyError):
            source(times[:1], ["nonexistent_var"])
