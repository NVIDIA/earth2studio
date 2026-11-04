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

"""Tests for the recipe data layer (src/data.py): predownloaded stores and
source resolution."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import OmegaConf
from src.data import (
    PredownloadedForecastSource,
    PredownloadedFrameSource,
    PredownloadedSource,
    frame_filename,
    frame_store_path,
    instantiate_source,
    is_forecast_source,
    open_predownloaded,
    resolve_source,
    store_name,
)

from earth2studio.data import Random, fetch_data
from earth2studio.io import ZarrBackend
from earth2studio.utils.coords import E2S_STATISTICS, coord_array

TIMES = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")
LEADS = np.array([0, 6], dtype="timedelta64[h]").astype("timedelta64[ns]")
VARIABLES = ["t2m", "z500"]
LAT = np.array([90.0, 0.0, -90.0])
LON = np.array([0.0, 90.0, 180.0, 270.0])


def _write_store(path, times=TIMES, variables=VARIABLES, lat=LAT, lon=LON):
    """Write a predownload-schema store (one array per variable on
    ``(time, lat, lon)``) through the library backend, as predownload does."""
    io = ZarrBackend(str(path))
    coords = {"time": times, "variable": np.array(variables), "lat": lat, "lon": lon}
    io.add_array(coord_array(("time", "variable", "lat", "lon"), coords))
    data = np.random.default_rng(42).standard_normal(
        (len(times), len(variables), len(lat), len(lon))
    )
    da = xr.DataArray(data.astype("float32"), dims=list(coords), coords=coords)
    io.write(da)
    io.close()
    return da


def _write_forecast_store(path, times=TIMES, leads=LEADS, variables=VARIABLES):
    """Write a forecast-source store (``(time, lead_time, lat, lon)`` arrays)."""
    io = ZarrBackend(str(path))
    coords = {
        "time": times,
        "lead_time": leads,
        "variable": np.array(variables),
        "lat": LAT,
        "lon": LON,
    }
    io.add_array(coord_array(("time", "lead_time", "variable", "lat", "lon"), coords))
    data = np.random.default_rng(7).standard_normal(
        (len(times), len(leads), len(variables), len(LAT), len(LON))
    )
    da = xr.DataArray(data.astype("float32"), dims=list(coords), coords=coords)
    io.write(da)
    io.close()
    return da


# ---------------------------------------------------------------------------
# PredownloadedSource
# ---------------------------------------------------------------------------


class TestPredownloadedSource:
    @pytest.fixture()
    def store_path(self, tmp_path):
        return tmp_path / "data.zarr"

    @pytest.fixture()
    def written(self, store_path):
        return _write_store(store_path)

    @pytest.fixture()
    def source(self, written, store_path):
        return PredownloadedSource(str(store_path))

    def test_missing_store_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PredownloadedSource(str(tmp_path / "nope.zarr"))

    def test_properties(self, source):
        assert sorted(source.variables) == sorted(VARIABLES)
        np.testing.assert_array_equal(source.times, TIMES)
        assert source.time_step == np.timedelta64(1, "D")

    def test_call_returns_dataarray(self, source):
        result = source(TIMES[:1], ["t2m"])
        assert isinstance(result, xr.DataArray)
        assert result.dims[:2] == ("time", "variable")
        assert result.sizes["time"] == 1 and result.sizes["variable"] == 1

    def test_select_multiple_times(self, source):
        result = source(TIMES, VARIABLES)
        assert result.sizes["time"] == 3 and result.sizes["variable"] == 2

    def test_scalar_inputs(self, source):
        result = source(np.datetime64("2024-01-01", "ns"), "t2m")
        assert result.sizes["time"] == 1 and result.sizes["variable"] == 1

    def test_values_match_written(self, source, written):
        result = source(TIMES[1:2], ["z500"])
        np.testing.assert_array_equal(
            result.values.squeeze(), written.sel(time=TIMES[1], variable="z500").values
        )

    def test_async_fetch(self, source):
        result = asyncio.run(source.fetch(TIMES[:1], ["t2m"]))
        assert isinstance(result, xr.DataArray)

    def test_missing_time_raises(self, source):
        with pytest.raises(KeyError, match="lacks some of the times"):
            source(np.array(["2099-01-01"], dtype="datetime64[ns]"), ["t2m"])

    def test_missing_variable_raises(self, source):
        with pytest.raises(KeyError, match="lacks variables"):
            source(TIMES[:1], ["nonexistent_var"])

    def test_works_with_fetch_data(self, source):
        from earth2studio.data import fetch_data

        da = fetch_data(
            source,
            time=TIMES[:1],
            variable=np.array(["t2m"]),
            lead_time=np.array([0, 24], dtype="timedelta64[h]"),
        )
        assert tuple(da.dims) == ("time", "lead_time", "variable", "lat", "lon")
        assert da.sizes["lead_time"] == 2


class TestQualifiedLabels:
    """Stored time statistics answer fetch_data's split requests."""

    LABELS = ["t2m:mean:0h:24h", "tp:mean:1h:25h"]

    @pytest.fixture()
    def store(self, tmp_path):
        written = _write_store(tmp_path / "data.zarr", variables=self.LABELS)
        return PredownloadedSource(str(tmp_path / "data.zarr")), written

    def test_window_length_is_the_time_step(self, store):
        source, _ = store
        assert source.time_step == np.timedelta64(24, "h")
        # Zarr lists arrays in store order, not request order.
        assert sorted(source.variables) == sorted(self.LABELS)

    def test_fetch_data_round_trips_the_reduced_fields(self, store):
        source, written = store
        da = fetch_data(
            source,
            time=TIMES[1:2],
            variable=np.array(self.LABELS),
            lead_time=np.array([-24, 0], dtype="timedelta64[h]"),
        )
        assert list(da["variable"].values) == self.LABELS
        for k, t in enumerate(TIMES[:2]):
            for label in self.LABELS:
                np.testing.assert_allclose(
                    da.sel(variable=label).isel(time=0, lead_time=k).values,
                    written.sel(time=t, variable=label).values,
                )
        statistics = da.attrs[E2S_STATISTICS]
        assert set(statistics) == set(self.LABELS)
        assert statistics["tp:mean:1h:25h"]["start_offset"] == np.timedelta64(1, "h")

    def test_base_variable_is_shifted_by_the_window_start(self, store):
        source, written = store
        asked = TIMES[:1] + np.timedelta64(1, "h")
        da = source(asked, ["tp"])
        np.testing.assert_array_equal(da["time"].values, asked)
        assert list(da["variable"].values) == ["tp"]
        np.testing.assert_allclose(
            da.values.squeeze(),
            written.sel(time=TIMES[0], variable="tp:mean:1h:25h").values,
        )
        assert E2S_STATISTICS not in da.attrs

    def test_direct_label_request_carries_statistics_metadata(self, store):
        source, written = store
        da = source(TIMES[:1], ["t2m:mean:0h:24h"])
        assert set(da.attrs[E2S_STATISTICS]) == {"t2m:mean:0h:24h"}
        np.testing.assert_allclose(
            da.values.squeeze(),
            written.sel(time=TIMES[0], variable="t2m:mean:0h:24h").values,
        )

    def test_two_statistics_of_one_variable_are_rejected(self, tmp_path):
        _write_store(
            tmp_path / "x.zarr", variables=["t2m:mean:0h:24h", "t2m:sum:0h:24h"]
        )
        with pytest.raises(ValueError, match="two statistics"):
            PredownloadedSource(str(tmp_path / "x.zarr"))

    def test_mixed_window_lengths_are_rejected(self, tmp_path):
        _write_store(
            tmp_path / "x.zarr", variables=["t2m:mean:0h:24h", "z500:mean:0h:6h"]
        )
        with pytest.raises(ValueError, match="mixes"):
            PredownloadedSource(str(tmp_path / "x.zarr"))


class TestPredownloadedForecastSource:
    @pytest.fixture()
    def store(self, tmp_path):
        path = tmp_path / "fx.zarr"
        return path, _write_forecast_store(path)

    def test_open_predownloaded_picks_the_kind(self, store, tmp_path):
        path, _ = store
        assert isinstance(open_predownloaded(str(path)), PredownloadedForecastSource)
        _write_store(tmp_path / "data.zarr")
        assert isinstance(
            open_predownloaded(str(tmp_path / "data.zarr")), PredownloadedSource
        )
        with pytest.raises(ValueError, match="forecast-source store"):
            PredownloadedSource(str(path))
        with pytest.raises(ValueError, match="data-source store"):
            PredownloadedForecastSource(str(tmp_path / "data.zarr"))
        with pytest.raises(FileNotFoundError):
            open_predownloaded(str(tmp_path / "nope.zarr"))

    def test_is_forecast_source(self, store, tmp_path):
        path, _ = store
        assert is_forecast_source(open_predownloaded(str(path)))
        _write_store(tmp_path / "data.zarr")
        assert not is_forecast_source(PredownloadedSource(str(tmp_path / "data.zarr")))
        assert not is_forecast_source(Random(domain_coords={"lat": LAT}))

    def test_fetch_data_uses_initial_time_and_lead(self, store):
        path, written = store
        source = PredownloadedForecastSource(str(path))
        np.testing.assert_array_equal(source.lead_times, LEADS)
        np.testing.assert_array_equal(source.times, TIMES)
        assert source.time_step == np.timedelta64(6, "h")
        da = fetch_data(
            source, time=TIMES[:1], variable=np.array(["z500"]), lead_time=LEADS
        )
        assert tuple(da.dims) == ("time", "lead_time", "variable", "lat", "lon")
        np.testing.assert_allclose(
            da.isel(time=0, lead_time=1, variable=0).values,
            written.sel(time=TIMES[0], lead_time=LEADS[1], variable="z500").values,
        )

    def test_missing_lead_raises(self, store):
        path, _ = store
        source = PredownloadedForecastSource(str(path))
        with pytest.raises(KeyError, match="lacks some of the times or lead times"):
            source(TIMES[:1], np.array([12], dtype="timedelta64[h]"), ["z500"])

    def test_async_fetch(self, store):
        path, _ = store
        source = PredownloadedForecastSource(str(path))
        result = asyncio.run(source.fetch(TIMES[:1], LEADS[:1], ["t2m"]))
        assert result.sizes["lead_time"] == 1


# ---------------------------------------------------------------------------
# resolve_source
# ---------------------------------------------------------------------------


class TestResolveSource:
    def _cfg(self, tmp_path):
        return OmegaConf.create({"output": {"path": str(tmp_path)}})

    def test_store_name(self):
        assert store_name("data") == "data.zarr"
        assert store_name("forcing0") == "forcing0.zarr"

    def test_instantiate_source(self):
        node = {"_target_": "earth2studio.data.Random", "domain_coords": {"lat": [0.0]}}
        assert isinstance(instantiate_source(node), Random)
        live = Random(domain_coords={"lat": np.array([0.0])})
        assert instantiate_source(live) is live
        assert instantiate_source(None) is None

    def test_byo_wins_over_store(self, tmp_path):
        _write_store(tmp_path / "data.zarr")
        byo = Random(domain_coords={"lat": np.array([0.0])})
        assert resolve_source(self._cfg(tmp_path), "data", byo=byo) is byo

    def test_predownloaded_store(self, tmp_path):
        _write_store(tmp_path / "verification.zarr")
        source = resolve_source(
            self._cfg(tmp_path), "verification", live={"_target_": "x"}
        )
        assert isinstance(source, PredownloadedSource)
        assert source.store_path == str(tmp_path / "verification.zarr")

    def test_predownloaded_forecast_store(self, tmp_path):
        _write_forecast_store(tmp_path / "fx.zarr")
        source = resolve_source(self._cfg(tmp_path), "fx", live={"_target_": "x"})
        assert isinstance(source, PredownloadedForecastSource)

    def test_live_when_store_not_preferred(self, tmp_path):
        _write_store(tmp_path / "data.zarr")
        live = Random(domain_coords={"lat": np.array([0.0])})
        assert (
            resolve_source(
                self._cfg(tmp_path), "data", live=live, prefer_predownloaded=False
            )
            is live
        )

    def test_live_node_instantiated(self, tmp_path):
        node = {"_target_": "earth2studio.data.Random", "domain_coords": {"lat": [0.0]}}
        assert isinstance(
            resolve_source(self._cfg(tmp_path), "data", live=node), Random
        )

    def test_nothing_gives_none(self, tmp_path):
        assert resolve_source(self._cfg(tmp_path), "data") is None


# ---------------------------------------------------------------------------
# Predownloaded observation frames
# ---------------------------------------------------------------------------


class TestFrameStore:
    def test_paths(self, tmp_path):
        assert frame_store_path(str(tmp_path), "obs") == str(tmp_path / "obs.parquet")
        assert (
            frame_filename(np.datetime64("2024-01-01T06:00"))
            == "20240101T060000.parquet"
        )

    @pytest.fixture()
    def frames(self, tmp_path):
        root = tmp_path / "obs.parquet"
        root.mkdir()
        for i, t in enumerate(TIMES[:2]):
            pd.DataFrame(
                {
                    "variable": ["t2m", "z500", "t2m"],
                    "value": [1.0 + i, 2.0 + i, 3.0 + i],
                    "lat": [0.0, 1.0, 2.0],
                }
            ).to_parquet(root / frame_filename(t))
        return root

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PredownloadedFrameSource(str(tmp_path / "nope.parquet"))

    def test_concatenates_times_in_order(self, frames):
        df = PredownloadedFrameSource(str(frames))(TIMES[:2], None)
        assert len(df) == 6
        assert df["value"].tolist() == [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]

    def test_variable_filter(self, frames):
        df = PredownloadedFrameSource(str(frames))(TIMES[:1], ["t2m"])
        assert df["variable"].tolist() == ["t2m", "t2m"]

    def test_fields_selection(self, frames):
        df = PredownloadedFrameSource(str(frames))(TIMES[:1], None, fields=["value"])
        assert list(df.columns) == ["value"]

    def test_missing_field_raises(self, frames):
        with pytest.raises(KeyError, match="lacks fields"):
            PredownloadedFrameSource(str(frames))(TIMES[:1], None, fields=["nope"])

    def test_missing_time_raises(self, frames):
        with pytest.raises(FileNotFoundError, match="re-run predownload"):
            PredownloadedFrameSource(str(frames))(TIMES[2:], None)
