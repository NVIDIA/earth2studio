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

"""Tests for PredownloadedSource, CompositeSource, ValidTimeForecastAdapter, CadenceRoundedSource."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from src.data import (
    CadenceRoundedSource,
    CompositeSource,
    PredownloadedSource,
    ValidTimeForecastAdapter,
)


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


# ---------------------------------------------------------------------------
# CompositeSource
# ---------------------------------------------------------------------------


def _create_yx_zarr_store(path, times, variables):
    """Write a zarr with (time, y, x) dims — matches StormScope predownload layout."""
    ds = xr.Dataset()
    y = np.arange(4)
    x = np.arange(5)
    for var in variables:
        data = np.random.default_rng(7).standard_normal((len(times), len(y), len(x)))
        ds[var] = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times, "y": y, "x": x},
        )
    ds.to_zarr(str(path))
    return ds


class TestCompositeSource:
    """CompositeSource dispatches variable requests across multiple sources.

    Fixtures create two small zarrs on a shared ``(time, y, x)`` grid but
    with disjoint variables — mirrors StormScope's
    ``data_goes.zarr`` / ``data_mrms.zarr`` layout.
    """

    @pytest.fixture()
    def times(self):
        return np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")

    @pytest.fixture()
    def stores(self, tmp_path, times):
        goes_path = tmp_path / "data_goes.zarr"
        mrms_path = tmp_path / "data_mrms.zarr"
        _create_yx_zarr_store(goes_path, times, ["abi01c", "abi02c"])
        _create_yx_zarr_store(mrms_path, times, ["refc"])
        return {"goes": str(goes_path), "mrms": str(mrms_path)}

    def test_from_predownloaded_stores_indexes_variables(self, stores):
        src = CompositeSource.from_predownloaded_stores(stores)
        # Variables from both stores are discoverable through the composite.
        _ = src(np.array(["2024-01-01"], dtype="datetime64[ns]"), ["abi01c"])
        _ = src(np.array(["2024-01-01"], dtype="datetime64[ns]"), ["refc"])

    def test_mixed_variable_request_dispatches_to_multiple_stores(self, stores, times):
        src = CompositeSource.from_predownloaded_stores(stores)
        result = src(times[:1], ["abi01c", "refc"])
        assert result.sizes["variable"] == 2
        # Caller-requested order is preserved.
        assert list(result.coords["variable"].values) == ["abi01c", "refc"]

    def test_variable_order_preserved_across_dispatch(self, stores, times):
        """Requesting [refc, abi01c, abi02c] returns variables in that order."""
        src = CompositeSource.from_predownloaded_stores(stores)
        result = src(times[:1], ["refc", "abi01c", "abi02c"])
        assert list(result.coords["variable"].values) == [
            "refc",
            "abi01c",
            "abi02c",
        ]

    def test_unknown_variable_raises(self, stores, times):
        src = CompositeSource.from_predownloaded_stores(stores)
        with pytest.raises(KeyError, match="not found in any"):
            src(times[:1], ["nonexistent"])

    def test_unknown_source_in_variable_index_raises(self, tmp_path, times):
        path = tmp_path / "data_goes.zarr"
        _create_yx_zarr_store(path, times, ["abi01c"])
        src = PredownloadedSource(str(path))
        with pytest.raises(ValueError, match="unknown sources"):
            CompositeSource({"goes": src}, variable_index={"abi01c": "typo"})


# ---------------------------------------------------------------------------
# ValidTimeForecastAdapter
# ---------------------------------------------------------------------------


class _FakeForecastSource:
    """Minimal ``ForecastSource``-shaped stub.

    Returns a deterministic DataArray shaped ``(time, lead_time, variable,
    lat, lon)`` so we can round-trip the adapter's coord-squeezing logic
    without network I/O.  Records each call so tests can assert on how
    the adapter forwarded the (init, lead) pair.
    """

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, time, lead_time, variable):
        # Normalize so the recorded tuple is easy to compare.
        times = list(time) if isinstance(time, (list, np.ndarray)) else [time]
        leads = (
            list(lead_time)
            if isinstance(lead_time, (list, np.ndarray))
            else [lead_time]
        )
        variables = (
            list(variable) if isinstance(variable, (list, np.ndarray)) else [variable]
        )
        self.calls.append((times, leads, variables))

        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 1.0, 2.0])
        shape = (len(times), len(leads), len(variables), lat.size, lon.size)
        # Encode (init, lead, var) into the payload so tests can recover
        # what was fetched.  init_int + lead_int*1e3 + var_idx.
        data = np.zeros(shape, dtype="float32")
        for ti, t in enumerate(times):
            for li, lt in enumerate(leads):
                for vi, _ in enumerate(variables):
                    data[ti, li, vi] = (
                        float(np.datetime64(t, "ns").astype("int64") % 1_000_000)
                        + float(np.timedelta64(lt, "ns").astype("int64") % 1_000) * 1e-3
                        + vi * 1e-6
                    )
        return xr.DataArray(
            data,
            dims=["time", "lead_time", "variable", "lat", "lon"],
            coords={
                "time": np.array(times, dtype="datetime64[ns]"),
                "lead_time": np.array(leads, dtype="timedelta64[ns]"),
                "variable": variables,
                "lat": lat,
                "lon": lon,
            },
        )


class TestValidTimeForecastAdapter:
    def _lookup(self):
        ic = np.datetime64("2023-12-05T12:00:00", "ns")
        return {
            ic + np.timedelta64(0, "m"): (ic, np.timedelta64(0, "m")),
            ic + np.timedelta64(60, "m"): (ic, np.timedelta64(60, "m")),
            ic + np.timedelta64(120, "m"): (ic, np.timedelta64(120, "m")),
        }

    def test_dispatches_to_forecast_source_by_lookup(self):
        src = _FakeForecastSource()
        lookup = self._lookup()
        adapter = ValidTimeForecastAdapter(src, lookup)

        ic = np.datetime64("2023-12-05T12:00:00", "ns")
        vt = ic + np.timedelta64(60, "m")
        out = adapter([vt], ["refc"])

        assert len(src.calls) == 1
        called_times, called_leads, called_vars = src.calls[0]
        # The adapter should have fetched at init=ic, lead=60min.
        assert np.datetime64(called_times[0], "ns") == ic
        assert np.timedelta64(called_leads[0], "m") == np.timedelta64(60, "m")
        assert called_vars == ["refc"]
        # And the returned DataArray is keyed by valid time (not init).
        assert list(out.dims) == ["time", "variable", "lat", "lon"]
        assert np.datetime64(out.time.values[0], "ns") == vt

    def test_multi_valid_time_concatenates(self):
        src = _FakeForecastSource()
        lookup = self._lookup()
        adapter = ValidTimeForecastAdapter(src, lookup)

        ic = np.datetime64("2023-12-05T12:00:00", "ns")
        times = [ic, ic + np.timedelta64(60, "m"), ic + np.timedelta64(120, "m")]
        out = adapter(times, ["refc"])

        assert len(src.calls) == 3
        assert out.sizes["time"] == 3
        assert [np.datetime64(t, "ns") for t in out.time.values] == times

    def test_unknown_valid_time_raises(self):
        src = _FakeForecastSource()
        adapter = ValidTimeForecastAdapter(src, self._lookup())
        with pytest.raises(KeyError, match="not in lookup"):
            adapter([np.datetime64("2024-01-01")], ["refc"])

    def test_lookup_keys_normalized_across_input_units(self):
        """Mixed-unit keys in the lookup (e.g. ``datetime64[s]``) still
        match ``datetime64[ns]`` queries via internal normalization."""
        import pandas as pd

        src = _FakeForecastSource()
        # Mix a python datetime and a numpy [s] as lookup keys.
        ic_dt = pd.Timestamp("2023-12-05T12:00:00").to_pydatetime()
        vt_s = np.datetime64("2023-12-05T13:00:00", "s")
        lookup = {vt_s: (ic_dt, np.timedelta64(60, "m"))}
        adapter = ValidTimeForecastAdapter(src, lookup)

        # Query with a datetime64[ns] → should hit the lookup.
        out = adapter([np.datetime64("2023-12-05T13:00:00", "ns")], ["refc"])
        assert out.sizes["time"] == 1
        assert len(src.calls) == 1


# ---------------------------------------------------------------------------
# CadenceRoundedSource
# ---------------------------------------------------------------------------


class _RecordingSource:
    """DataSource stub that records every ``(time, variable)`` call and
    returns a DataArray whose values encode the requested valid times.

    Used to verify which rounded times a wrapper actually forwards to
    the underlying source (dedup behavior, unit correctness).
    """

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, time, variable):
        times = np.asarray(time, dtype="datetime64[ns]")
        variables = np.atleast_1d(variable)
        self.calls.append((list(times), list(variables)))
        # Encode the valid time as a payload so the test can read it back.
        data = np.broadcast_to(
            times.astype("int64")[:, None, None, None].astype("float32"),
            (len(times), len(variables), 2, 3),
        ).copy()
        return xr.DataArray(
            data,
            dims=["time", "variable", "y", "x"],
            coords={
                "time": times,
                "variable": variables,
                "y": np.arange(2),
                "x": np.arange(3),
            },
        )

    async def fetch(self, time, variable):
        return self(time, variable)


class TestCadenceRoundedSource:
    """``CadenceRoundedSource`` rounds valid-time queries to a coarser
    native cadence and relabels the result so downstream consumers see
    the requested (pre-round) valid times."""

    def test_dedupes_requests_within_a_cadence_boundary(self):
        src = _RecordingSource()
        wrapped = CadenceRoundedSource(src, "1h")

        # Six 5-min requests all inside the 12:00 hour (so they all round
        # to 12:00 with nearest-hour rounding — ticks past 12:30 would
        # round up to 13:00).
        ic = np.datetime64("2023-12-05T12:00:00", "ns")
        times = [ic + np.timedelta64(k * 5, "m") for k in range(6)]
        out = wrapped(times, ["z500"])

        # One underlying fetch: the shared 12:00 bucket.
        assert len(src.calls) == 1
        called_times, _ = src.calls[0]
        assert len(called_times) == 1
        assert np.datetime64(called_times[0], "ns") == ic
        # Output preserves all six caller-requested timestamps.
        assert out.sizes["time"] == 6
        assert [np.datetime64(t, "ns") for t in out.time.values] == times

    def test_spans_multiple_cadence_boundaries(self):
        src = _RecordingSource()
        wrapped = CadenceRoundedSource(src, "1h")

        ic = np.datetime64("2023-12-05T12:00:00", "ns")
        times = [ic + np.timedelta64(k * 20, "m") for k in range(7)]
        # Rounded (nearest hour): 12:00, 12:00, 12:00, 13:00, 13:00, 13:00, 14:00.
        out = wrapped(times, ["z500"])

        assert len(src.calls) == 1
        called_times, _ = src.calls[0]
        # Three unique hours.
        assert len(called_times) == 3
        called_ints = sorted(
            int(np.datetime64(t, "ns").astype("int64")) for t in called_times
        )
        expected = sorted(
            int((ic + np.timedelta64(k, "h")).astype("int64")) for k in range(3)
        )
        assert called_ints == expected
        # Output still matches the caller's 7 requested times.
        assert out.sizes["time"] == 7
        assert [np.datetime64(t, "ns") for t in out.time.values] == times

    def test_nearest_rounding_goes_up_for_30min_tick(self):
        """At exactly 30 past the hour, nearest rounds up (standard
        tiebreaker for ``(x + c/2) // c``)."""
        src = _RecordingSource()
        wrapped = CadenceRoundedSource(src, "1h")
        t = np.datetime64("2023-12-05T12:30:00", "ns")
        _ = wrapped([t], ["z500"])
        called_times, _ = src.calls[0]
        assert np.datetime64(called_times[0], "ns") == np.datetime64(
            "2023-12-05T13:00:00", "ns"
        )

    def test_non_hour_cadence(self):
        src = _RecordingSource()
        wrapped = CadenceRoundedSource(src, "30min")
        ic = np.datetime64("2023-12-05T12:00:00", "ns")
        times = [ic + np.timedelta64(m, "m") for m in (5, 20, 40, 55)]
        # Rounded (nearest 30min): 12:00, 12:30, 12:30, 13:00.
        _ = wrapped(times, ["z500"])
        called_times, _ = src.calls[0]
        called_sorted = sorted(np.datetime64(t, "ns") for t in called_times)
        assert called_sorted == [
            np.datetime64("2023-12-05T12:00:00", "ns"),
            np.datetime64("2023-12-05T12:30:00", "ns"),
            np.datetime64("2023-12-05T13:00:00", "ns"),
        ]

    def test_non_positive_cadence_raises(self):
        src = _RecordingSource()
        with pytest.raises(ValueError, match="cadence must be positive"):
            CadenceRoundedSource(src, "0h")
