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

import weakref
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

from earth2studio.data import ARCO_ERA5, DailyMean, FuXiS2SERA5
from earth2studio.data.base import DataSource
from earth2studio.data.utils import fetch_data
from earth2studio.utils.type import TimeArray, VariableArray


class MockHourlySource:
    """Deterministic hourly data source used to test daily aggregation."""

    VARIABLE_OFFSETS = {
        "t2m": 0.0,
        "d2m": 100.0,
        "u10m": 200.0,
        "tp": 300.0,
        "ttr": 400.0,
    }

    def __init__(
        self,
        *,
        drop_last: bool = False,
        nan_time: datetime | None = None,
    ) -> None:
        self.drop_last = drop_last
        self.nan_time = nan_time
        self.call_history: list[tuple[list[datetime], list[str]]] = []

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        times = [time] if isinstance(time, datetime) else list(time)
        variables = [variable] if isinstance(variable, str) else list(variable)
        times = [
            (
                value.astype("datetime64[us]").astype(datetime)
                if isinstance(value, np.datetime64)
                else value
            )
            for value in times
        ]
        variables = [str(value) for value in variables]
        self.call_history.append((times, variables))

        if self.drop_last:
            times = times[:-1]

        epoch = datetime(2024, 1, 1)
        values: npt.NDArray[np.float64] = np.empty(
            (len(times), len(variables), 2, 3),
            dtype=np.float64,
        )
        for time_index, hourly_time in enumerate(times):
            hour = (hourly_time - epoch).total_seconds() / 3600
            for variable_index, variable in enumerate(variables):
                values[time_index, variable_index] = (
                    hour + self.VARIABLE_OFFSETS[variable]
                )

        if self.nan_time is not None and self.nan_time in times:
            time_index = times.index(self.nan_time)
            values[time_index, 0, 0, 0] = np.nan

        return xr.DataArray(
            values,
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": times,
                "variable": variables,
                "lat": np.array([1.5, 0.0]),
                "lon": np.array([0.0, 1.5, 3.0]),
            },
        )

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        return self(time, variable)

    def available(self, time: datetime | np.datetime64) -> bool:
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[us]").item()
        return time != self.nan_time


class MockERA5Source:
    """In-memory hourly source on either a valid or intentionally invalid grid."""

    def __init__(self, valid_grid: bool = True) -> None:
        self.valid_grid = valid_grid
        self.call_history: list[tuple[list[datetime], list[str]]] = []

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        times = [time] if isinstance(time, datetime) else list(time)
        variables = [variable] if isinstance(variable, str) else list(variable)
        times = [
            (
                value.astype("datetime64[us]").item()
                if isinstance(value, np.datetime64)
                else value
            )
            for value in times
        ]
        variables = [str(value) for value in variables]
        self.call_history.append((times, variables))

        if self.valid_grid:
            lat = FuXiS2SERA5.ERA5_LAT
            lon = FuXiS2SERA5.ERA5_LON
        else:
            lat = np.array([90.0, -90.0])
            lon = np.array([0.0, 180.0])

        grid = (
            np.arange(len(lat), dtype=np.float32)[:, None] * 10_000
            + np.arange(len(lon), dtype=np.float32)[None, :]
        )
        values = np.broadcast_to(
            grid[None, None],
            (len(times), len(variables), len(lat), len(lon)),
        )
        return xr.DataArray(
            values,
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": times,
                "variable": variables,
                "lat": lat,
                "lon": lon,
            },
        )

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        return self(time, variable)

    def available(self, time: datetime | np.datetime64) -> bool:
        return True


class MismatchedGridSource(MockHourlySource):
    """Hourly source that changes longitude coordinates after its first call."""

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        data = super().__call__(time, variable)
        if len(self.call_history) > 1:
            data = data.assign_coords(lon=data.lon + 0.25)
        return data


class MismatchedDimensionsSource(MockHourlySource):
    """Hourly source that changes spatial dimension names between calls."""

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        data = super().__call__(time, variable)
        if len(self.call_history) > 1:
            data = data.rename({"lat": "y", "lon": "x"})
        return data


class MismatchedAuxiliaryGridSource(MockHourlySource):
    """Hourly source that changes a curvilinear coordinate between calls."""

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        data = super().__call__(time, variable).rename({"lat": "y", "lon": "x"})
        lat = np.broadcast_to(np.asarray(data.y)[:, None], (2, 3)).copy()
        lon = np.broadcast_to(np.asarray(data.x)[None, :], (2, 3))
        if len(self.call_history) > 1:
            lat += 0.25
        return data.assign_coords(
            lat=(("y", "x"), lat),
            lon=(("y", "x"), lon),
        )


class ReleasingDailyMean(DailyMean):
    """Daily-mean adapter that checks completed batch buffers are released."""

    def __init__(self, datasource: DataSource, time_batch_size: int) -> None:
        super().__init__(datasource, time_batch_size=time_batch_size)
        self.previous_values: weakref.ReferenceType[np.ndarray] | None = None

    def _prepare_hourly_data(self, data: xr.DataArray) -> xr.DataArray:
        if self.previous_values is not None and self.previous_values() is not None:
            raise RuntimeError("previous hourly batch is still retained")

        self.previous_values = weakref.ref(data.values)
        return data


def test_daily_mean_is_datasource() -> None:
    assert isinstance(DailyMean(MockHourlySource()), DataSource)


def test_daily_mean_uses_variable_timestamp_conventions() -> None:
    source = MockHourlySource()
    daily_source = DailyMean(source, end_of_interval_variables=["tp", "ttr"])

    result = daily_source(datetime(2024, 1, 1), ["tp", "t2m"])

    assert result.dims == ("time", "variable", "lat", "lon")
    np.testing.assert_array_equal(result.time, [np.datetime64("2024-01-01")])
    np.testing.assert_array_equal(result.coords["variable"], ["tp", "t2m"])
    np.testing.assert_allclose(result.sel(variable="tp"), 312.5)
    np.testing.assert_allclose(result.sel(variable="t2m"), 11.5)

    instantaneous_times, instantaneous_variables = source.call_history[0]
    interval_times, interval_variables = source.call_history[1]
    assert instantaneous_variables == ["t2m"]
    assert instantaneous_times[0] == datetime(2024, 1, 1, 0)
    assert instantaneous_times[-1] == datetime(2024, 1, 1, 23)
    assert interval_variables == ["tp"]
    assert interval_times[0] == datetime(2024, 1, 1, 1)
    assert interval_times[-1] == datetime(2024, 1, 2, 0)


def test_daily_mean_preserves_requested_day_order() -> None:
    source = MockHourlySource()
    daily_source = DailyMean(source)
    days = [datetime(2024, 1, 2), datetime(2024, 1, 1)]

    result = daily_source(days, "t2m")

    np.testing.assert_array_equal(
        result.time,
        np.array(["2024-01-02", "2024-01-01"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(result[:, 0, 0, 0], [35.5, 11.5])


def test_daily_mean_batches_variables() -> None:
    source = MockHourlySource()
    daily_source = DailyMean(source, variable_batch_size=2)

    result = daily_source(
        datetime(2024, 1, 1),
        ["t2m", "d2m", "u10m"],
    )

    assert [variables for _, variables in source.call_history] == [
        ["t2m", "d2m"],
        ["u10m"],
    ]
    np.testing.assert_array_equal(
        result.coords["variable"],
        ["t2m", "d2m", "u10m"],
    )
    np.testing.assert_allclose(
        result.isel(time=0, lat=0, lon=0),
        [11.5, 111.5, 211.5],
    )


def test_daily_mean_batches_times_before_aggregation() -> None:
    source = MockHourlySource()
    daily_source = DailyMean(source, time_batch_size=5)

    result = daily_source(datetime(2024, 1, 1), "t2m")

    assert [len(times) for times, _ in source.call_history] == [5, 5, 5, 5, 4]
    assert source.call_history[0][0][0] == datetime(2024, 1, 1, 0)
    assert source.call_history[-1][0][-1] == datetime(2024, 1, 1, 23)
    np.testing.assert_allclose(result.sel(variable="t2m"), 11.5)


def test_daily_mean_validation_reuses_already_ordered_buffer() -> None:
    source = MockHourlySource()
    times = [datetime(2024, 1, 1, hour) for hour in range(5)]
    variables = ["t2m", "d2m"]
    data = source(times, variables)

    validated = DailyMean._validate_hourly_data(data, times, variables)

    assert validated is data


def test_daily_mean_releases_completed_hourly_batches() -> None:
    daily_source = ReleasingDailyMean(MockHourlySource(), time_batch_size=5)

    result = daily_source(datetime(2024, 1, 1), "t2m")

    np.testing.assert_allclose(result.sel(variable="t2m"), 11.5)
    assert daily_source.previous_values is not None
    assert daily_source.previous_values() is None


def test_daily_mean_batches_across_reverse_ordered_days() -> None:
    days = [datetime(2024, 1, 2), datetime(2024, 1, 1)]
    variables = ["t2m", "d2m"]

    batched = DailyMean(MockHourlySource(), time_batch_size=5)(days, variables)
    single_batch = DailyMean(MockHourlySource(), time_batch_size=48)(
        days,
        variables,
    )

    xr.testing.assert_allclose(batched, single_batch)
    np.testing.assert_array_equal(
        batched.time,
        np.array(["2024-01-02", "2024-01-01"], dtype="datetime64[ns]"),
    )


def test_daily_mean_rejects_inconsistent_batch_coordinates() -> None:
    daily_source = DailyMean(MismatchedGridSource(), time_batch_size=12)

    with pytest.raises(ValueError, match="coordinate schema"):
        daily_source(datetime(2024, 1, 1), "t2m")


@pytest.mark.parametrize(
    "source",
    [MismatchedDimensionsSource(), MismatchedAuxiliaryGridSource()],
)
def test_daily_mean_rejects_inconsistent_spatial_schema(
    source: DataSource,
) -> None:
    daily_source = DailyMean(source, time_batch_size=12)

    with pytest.raises(ValueError, match="coordinate schema"):
        daily_source(datetime(2024, 1, 1), "t2m")


def test_daily_mean_rejects_inconsistent_spatial_schema_between_days() -> None:
    daily_source = DailyMean(MismatchedDimensionsSource(), time_batch_size=24)

    with pytest.raises(ValueError, match="coordinate schema"):
        daily_source(
            [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "t2m",
        )


def test_daily_mean_rejects_inconsistent_spatial_schema_between_variables() -> None:
    daily_source = DailyMean(
        MismatchedDimensionsSource(),
        variable_batch_size=1,
        time_batch_size=24,
    )

    with pytest.raises(ValueError, match="coordinate schema"):
        daily_source(datetime(2024, 1, 1), ["t2m", "d2m"])


@pytest.mark.asyncio
async def test_daily_mean_async_fetch() -> None:
    source = MockHourlySource()
    daily_source = DailyMean(
        source,
        end_of_interval_variables="tp",
        time_batch_size=7,
    )

    result = await daily_source.fetch(datetime(2024, 1, 1), ["t2m", "tp"])

    np.testing.assert_allclose(result.sel(variable="t2m"), 11.5)
    np.testing.assert_allclose(result.sel(variable="tp"), 312.5)


@pytest.mark.asyncio
async def test_daily_mean_async_batches_across_days_and_preserves_nan() -> None:
    source = MockHourlySource(nan_time=datetime(2024, 1, 2, 3))
    daily_source = DailyMean(source, time_batch_size=5)

    result = await daily_source.fetch(
        [datetime(2024, 1, 2), datetime(2024, 1, 1)],
        "t2m",
    )

    assert np.isnan(result.isel(time=0, variable=0, lat=0, lon=0))
    np.testing.assert_allclose(result.isel(time=0, variable=0, lat=0, lon=1), 35.5)
    np.testing.assert_allclose(result.isel(time=1, variable=0, lat=0, lon=0), 11.5)


@pytest.mark.parametrize(
    ("end_of_interval_variables", "unavailable_time", "expected"),
    [
        (None, None, True),
        (None, datetime(2024, 1, 1, 23), False),
        (None, datetime(2024, 1, 2, 0), True),
        ("tp", datetime(2024, 1, 2, 0), False),
    ],
)
def test_daily_mean_available(
    end_of_interval_variables: str | None,
    unavailable_time: datetime | None,
    expected: bool,
) -> None:
    source = MockHourlySource(nan_time=unavailable_time)
    daily_source = DailyMean(source, end_of_interval_variables)

    assert daily_source.available(np.datetime64("2024-01-01")) is expected


def test_daily_mean_available_requires_supported_source() -> None:
    class SourceWithoutAvailable:
        def __call__(
            self,
            time: datetime | list[datetime] | TimeArray,
            variable: str | list[str] | VariableArray,
        ) -> xr.DataArray:
            raise NotImplementedError

    with pytest.raises(AttributeError, match="does not support availability"):
        DailyMean(SourceWithoutAvailable()).available(datetime(2024, 1, 1))


def test_daily_mean_does_not_ignore_missing_values() -> None:
    source = MockHourlySource(nan_time=datetime(2024, 1, 1, 12))
    daily_source = DailyMean(source, time_batch_size=5)

    result = daily_source(datetime(2024, 1, 1), "t2m")

    assert np.isnan(result.sel(variable="t2m")[0, 0, 0])
    np.testing.assert_allclose(result.sel(variable="t2m")[0, 0, 1], 11.5)


def test_daily_mean_rejects_missing_hour() -> None:
    source = MockHourlySource(drop_last=True)
    daily_source = DailyMean(source)

    with pytest.raises(ValueError, match="missing requested times or variables"):
        daily_source(datetime(2024, 1, 1), "t2m")


@pytest.mark.parametrize(
    ("time", "variable", "match"),
    [
        ([], "t2m", "time must contain at least one"),
        (datetime(2024, 1, 1), [], "variable must contain at least one"),
        (
            datetime(2024, 1, 1, 6),
            "t2m",
            "must be aligned to 00:00 UTC",
        ),
        (
            [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            "t2m",
            "time must not contain duplicate",
        ),
        (
            datetime(2024, 1, 1),
            ["t2m", "t2m"],
            "variable must not contain duplicate",
        ),
    ],
)
def test_daily_mean_rejects_invalid_requests(
    time: datetime | list[datetime],
    variable: str | list[str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        DailyMean(MockHourlySource())(time, variable)


def test_daily_mean_rejects_duplicate_interval_variables() -> None:
    with pytest.raises(
        ValueError,
        match="end_of_interval_variables must not contain duplicates",
    ):
        DailyMean(
            MockHourlySource(),
            end_of_interval_variables=["tp", "tp"],
        )


@pytest.mark.parametrize("variable_batch_size", [0, -1, 1.5])
def test_daily_mean_rejects_invalid_batch_size(
    variable_batch_size: int | float,
) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        DailyMean(
            MockHourlySource(),
            variable_batch_size=variable_batch_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("time_batch_size", [0, -1, 1.5])
def test_daily_mean_rejects_invalid_time_batch_size(
    time_batch_size: int | float,
) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        DailyMean(
            MockHourlySource(),
            time_batch_size=time_batch_size,  # type: ignore[arg-type]
        )


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(600)
def test_fuxi_s2s_era5_fetch_arco() -> None:
    source = FuXiS2SERA5(
        ARCO_ERA5(cache=False, verbose=False, async_timeout=600),
        variable_batch_size=1,
        time_batch_size=24,
    )

    result = source(datetime(2020, 6, 1), ["t2m", "tp"])

    assert result.dims == ("time", "variable", "lat", "lon")
    assert result.shape == (1, 2, 121, 240)
    assert bool(result.notnull().all())
    np.testing.assert_array_equal(result.lat, FuXiS2SERA5.FUXI_LAT)
    np.testing.assert_array_equal(result.lon, FuXiS2SERA5.FUXI_LON)
    assert float(result.sel(variable="t2m").mean()) == pytest.approx(
        281.3432,
        abs=0.001,
    )
    assert float(result.sel(variable="tp").mean()) == pytest.approx(
        9.52497e-5,
        abs=1.0e-8,
    )


def test_fuxi_s2s_era5_is_datasource_with_required_windows() -> None:
    source = MockERA5Source()
    fuxi_source = FuXiS2SERA5(source)

    assert isinstance(fuxi_source, DataSource)
    assert fuxi_source.end_of_interval_variables == frozenset({"tp", "ttr"})
    assert fuxi_source.variable_batch_size == 13
    assert fuxi_source.time_batch_size == 1


def test_fuxi_s2s_era5_point_samples_official_grid() -> None:
    source = MockERA5Source()
    fuxi_source = FuXiS2SERA5(source)

    result = fuxi_source(datetime(2024, 1, 1), "t2m")

    assert result.dims == ("time", "variable", "lat", "lon")
    assert result.shape == (1, 1, 121, 240)
    np.testing.assert_array_equal(result.lat, FuXiS2SERA5.FUXI_LAT)
    np.testing.assert_array_equal(result.lon, FuXiS2SERA5.FUXI_LON)
    assert result.isel(time=0, variable=0, lat=10, lon=20).item() == pytest.approx(
        60 * 10_000 + 120
    )
    assert len(source.call_history) == 24
    assert all(variables == ["t2m"] for _, variables in source.call_history)
    assert source.call_history[0][0] == [datetime(2024, 1, 1, 0)]
    assert source.call_history[-1][0] == [datetime(2024, 1, 1, 23)]


def test_fuxi_s2s_era5_materializes_sampled_grid() -> None:
    source = MockERA5Source()
    hourly = source(datetime(2024, 1, 1), "t2m")

    sampled = FuXiS2SERA5(source)._prepare_hourly_data(hourly)

    assert sampled.shape == (1, 1, 121, 240)
    assert not np.shares_memory(hourly.values, sampled.values)


def test_fuxi_s2s_era5_rejects_non_arco_grid() -> None:
    source = MockERA5Source(valid_grid=False)
    fuxi_source = FuXiS2SERA5(source)

    with pytest.raises(ValueError, match=r"shape \(721, 1440\)"):
        fuxi_source(datetime(2024, 1, 1), "t2m")


def test_fuxi_s2s_era5_canonicalizes_spatial_dimension_order() -> None:
    class SwappedSpatialSource(MockERA5Source):
        def __call__(
            self,
            time: datetime | list[datetime] | TimeArray,
            variable: str | list[str] | VariableArray,
        ) -> xr.DataArray:
            return (
                super()
                .__call__(time, variable)
                .transpose(
                    "time",
                    "variable",
                    "lon",
                    "lat",
                )
            )

    result = FuXiS2SERA5(SwappedSpatialSource())(
        datetime(2024, 1, 1),
        "t2m",
    )

    assert result.dims == ("time", "variable", "lat", "lon")


def test_fuxi_s2s_era5_rejects_extra_dimensions() -> None:
    class ExtraDimensionSource(MockERA5Source):
        def __call__(
            self,
            time: datetime | list[datetime] | TimeArray,
            variable: str | list[str] | VariableArray,
        ) -> xr.DataArray:
            return (
                super()
                .__call__(time, variable)
                .expand_dims(
                    ensemble=[0],
                    axis=2,
                )
            )

    with pytest.raises(ValueError, match="requires exactly the dimensions"):
        FuXiS2SERA5(ExtraDimensionSource())(datetime(2024, 1, 1), "t2m")


def test_fuxi_s2s_era5_fetch_data_handoff() -> None:
    source = FuXiS2SERA5(MockERA5Source())

    tensor, coords = fetch_data(
        source=source,
        time=np.array([np.datetime64("2024-01-02")]),
        variable=np.array(["t2m"]),
        lead_time=np.array([np.timedelta64(-1, "D"), np.timedelta64(0, "D")]),
    )

    assert tensor.shape == (1, 2, 1, 121, 240)
    np.testing.assert_array_equal(
        coords["lead_time"],
        np.array([np.timedelta64(-1, "D"), np.timedelta64(0, "D")]),
    )
    np.testing.assert_array_equal(coords["variable"], ["t2m"])
    np.testing.assert_array_equal(coords["lat"], FuXiS2SERA5.FUXI_LAT)
    np.testing.assert_array_equal(coords["lon"], FuXiS2SERA5.FUXI_LON)
