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

from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import xarray as xr

from earth2studio.data.base import DataSource
from earth2studio.data.utils import prep_data_inputs
from earth2studio.utils.type import TimeArray, VariableArray

_HOURS_PER_DAY = 24


class DailyMean:
    """Aggregate an hourly gridded data source into UTC daily means.

    The requested timestamps label UTC calendar days and must be at 00:00.
    Instantaneous variables use hourly samples from 00:00 through 23:00.
    Variables whose hourly values describe the preceding interval use samples
    from 01:00 through 00:00 of the following day.

    Parameters
    ----------
    datasource : DataSource
        Hourly gridded data source to aggregate.
    end_of_interval_variables : str | list[str] | VariableArray, optional
        Variables timestamped at the end of the represented hourly interval.
        ERA5 accumulated variables such as ``tp`` and ``ttr`` use this
        convention, by default None
    variable_batch_size : int, optional
        Maximum variables requested from the underlying hourly source at once.
        Small batches bound peak memory for high-resolution global sources, by
        default 1
    time_batch_size : int, optional
        Maximum hourly timestamps requested from the underlying source at once.
        Each batch is spatially prepared before the next batch is fetched, by
        default 24

    Note
    ----
    This adapter preserves the spatial grid and physical units returned by the
    underlying source. Regridding and model normalization are separate steps.

    Warning
    -------
    Each requested daily mean reads 24 hourly samples for every variable from
    the wrapped source. Use ``variable_batch_size`` and ``time_batch_size`` to
    balance request count and peak memory for large grids.

    Badges
    ------
    dataclass:analysis
    """

    def __init__(
        self,
        datasource: DataSource,
        end_of_interval_variables: str | list[str] | VariableArray | None = None,
        variable_batch_size: int = 1,
        time_batch_size: int = _HOURS_PER_DAY,
    ) -> None:
        self.datasource = datasource
        if not isinstance(variable_batch_size, int) or variable_batch_size < 1:
            raise ValueError("variable_batch_size must be a positive integer")
        if not isinstance(time_batch_size, int) or time_batch_size < 1:
            raise ValueError("time_batch_size must be a positive integer")
        self.variable_batch_size = variable_batch_size
        self.time_batch_size = time_batch_size
        if end_of_interval_variables is None:
            end_of_interval_variables = []
        _, variables = prep_data_inputs([], end_of_interval_variables)
        if len(variables) != len(set(variables)):
            raise ValueError("end_of_interval_variables must not contain duplicates")
        self.end_of_interval_variables = frozenset(variables)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return UTC daily means for requested dates and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            UTC calendar-day labels at 00:00.
        variable : str | list[str] | VariableArray
            Variables to aggregate.

        Returns
        -------
        xr.DataArray
            Daily means with dimensions ``(time, variable, ...)``.
        """
        time_list, variable_list = self._prepare_inputs(time, variable)
        return self._fetch_and_aggregate(time_list, variable_list)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Asynchronously return UTC daily means.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            UTC calendar-day labels at 00:00.
        variable : str | list[str] | VariableArray
            Variables to aggregate.

        Returns
        -------
        xr.DataArray
            Daily means with dimensions ``(time, variable, ...)``.

        Raises
        ------
        AttributeError
            If the wrapped data source does not provide async ``fetch``.
        """
        time_list, variable_list = self._prepare_inputs(time, variable)
        if not hasattr(self.datasource, "fetch"):
            raise AttributeError(
                f"Underlying datasource {type(self.datasource).__name__} does not "
                "support async fetch"
            )

        return await self._fetch_and_aggregate_async(
            time_list,
            variable_list,
        )

    def available(self, time: datetime | np.datetime64) -> bool:
        """Check whether every hourly sample needed for a day is available.

        Parameters
        ----------
        time : datetime | np.datetime64
            UTC calendar-day label at 00:00.

        Returns
        -------
        bool
            Whether all required hourly samples are available.

        Raises
        ------
        AttributeError
            If the wrapped data source does not provide ``available``.
        """
        time_list, _ = self._prepare_inputs(time, ["_availability"])
        if not hasattr(self.datasource, "available"):
            raise AttributeError(
                f"Underlying datasource {type(self.datasource).__name__} does not "
                "support availability checks"
            )

        last_hour = _HOURS_PER_DAY if self.end_of_interval_variables else 23
        hourly_times = [
            time_list[0] + timedelta(hours=hour) for hour in range(last_hour + 1)
        ]
        return all(
            self.datasource.available(hourly_time) for hourly_time in hourly_times
        )

    @staticmethod
    def _prepare_inputs(
        time: datetime | list[datetime] | np.datetime64 | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> tuple[list[datetime], list[str]]:
        time_list, variable_list = prep_data_inputs(time, variable)
        if not time_list:
            raise ValueError("time must contain at least one UTC calendar day")
        if not variable_list:
            raise ValueError("variable must contain at least one variable")
        if len(time_list) != len(set(time_list)):
            raise ValueError("time must not contain duplicate calendar days")
        if len(variable_list) != len(set(variable_list)):
            raise ValueError("variable must not contain duplicates")
        for daily_time in time_list:
            if daily_time != daily_time.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            ):
                raise ValueError(
                    f"Daily timestamp {daily_time} must be aligned to 00:00 UTC"
                )
        return time_list, variable_list

    def _split_variables(
        self,
        variable_list: list[str],
    ) -> tuple[list[str], list[str]]:
        instantaneous = [
            variable
            for variable in variable_list
            if variable not in self.end_of_interval_variables
        ]
        end_of_interval = [
            variable
            for variable in variable_list
            if variable in self.end_of_interval_variables
        ]
        return instantaneous, end_of_interval

    @staticmethod
    def _hourly_times(
        time_list: list[datetime],
        end_of_interval: bool,
    ) -> tuple[list[datetime], npt.NDArray[np.datetime64]]:
        first_hour = 1 if end_of_interval else 0
        hourly_times = [
            daily_time + timedelta(hours=hour)
            for daily_time in time_list
            for hour in range(first_hour, first_hour + _HOURS_PER_DAY)
        ]
        daily_labels: npt.NDArray[np.datetime64] = np.repeat(
            np.asarray(time_list, dtype="datetime64[ns]"),
            _HOURS_PER_DAY,
        )
        return hourly_times, daily_labels

    def _prepare_hourly_data(self, data: xr.DataArray) -> xr.DataArray:
        return data

    @staticmethod
    def _validate_hourly_data(
        data: xr.DataArray,
        hourly_times: list[datetime],
        variable_list: list[str],
    ) -> xr.DataArray:
        if data.dims[:2] != ("time", "variable"):
            raise ValueError(
                "Hourly datasource must return dimensions beginning with "
                "('time', 'variable')"
            )

        expected_times = np.asarray(hourly_times, dtype="datetime64[ns]")
        try:
            actual_times = np.asarray(
                data.coords["time"].values,
                dtype="datetime64[ns]",
            )
            actual_variables = np.asarray(data.coords["variable"].values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Hourly datasource response is missing requested times or variables"
            ) from error

        coordinates_match = np.array_equal(
            actual_times,
            expected_times,
        ) and np.array_equal(
            actual_variables,
            np.asarray(variable_list),
        )
        if not coordinates_match:
            try:
                data = data.sel(time=expected_times, variable=variable_list)
            except (KeyError, ValueError) as error:
                raise ValueError(
                    "Hourly datasource response is missing requested times or variables"
                ) from error

        if data.sizes["time"] != len(expected_times):
            raise ValueError(
                f"Hourly datasource returned {data.sizes['time']} samples; "
                f"expected {len(expected_times)}"
            )
        if data.sizes["variable"] != len(variable_list):
            raise ValueError(
                f"Hourly datasource returned {data.sizes['variable']} variables; "
                f"expected {len(variable_list)}"
            )

        return data

    @staticmethod
    def _validate_coordinate_schema(
        reference: xr.DataArray,
        candidate: xr.DataArray,
        ignored_dimensions: frozenset[str] = frozenset(),
    ) -> None:
        reference_dimensions = tuple(
            dimension
            for dimension in reference.dims
            if dimension not in ignored_dimensions
        )
        candidate_dimensions = tuple(
            dimension
            for dimension in candidate.dims
            if dimension not in ignored_dimensions
        )
        if reference_dimensions != candidate_dimensions:
            raise ValueError(
                "Hourly datasource returned inconsistent coordinate schema "
                f"between batches: dimensions {reference_dimensions} and "
                f"{candidate_dimensions} differ"
            )

        reference_coordinates = {
            name: coordinate
            for name, coordinate in reference.coords.items()
            if ignored_dimensions.isdisjoint(coordinate.dims)
        }
        candidate_coordinates = {
            name: coordinate
            for name, coordinate in candidate.coords.items()
            if ignored_dimensions.isdisjoint(coordinate.dims)
        }
        if reference_coordinates.keys() != candidate_coordinates.keys():
            raise ValueError(
                "Hourly datasource returned inconsistent coordinate schema "
                "between batches: coordinate names differ"
            )

        for name, reference_coordinate in reference_coordinates.items():
            candidate_coordinate = candidate_coordinates[name]
            if (
                reference_coordinate.dims != candidate_coordinate.dims
                or not reference_coordinate.equals(candidate_coordinate)
            ):
                raise ValueError(
                    "Hourly datasource returned inconsistent coordinate schema "
                    f"between batches: coordinate {name!r} differs"
                )

    @staticmethod
    def _accumulate_hourly_batch(
        data: xr.DataArray,
        daily_labels: npt.NDArray[np.datetime64],
        daily_sums: dict[np.datetime64, xr.DataArray],
        daily_counts: dict[np.datetime64, int],
    ) -> None:
        data = data.assign_coords(_daily_time=("time", daily_labels))
        partial_sums = data.groupby("_daily_time", squeeze=False).sum(
            dim="time",
            skipna=False,
        )

        labels, counts = np.unique(daily_labels, return_counts=True)
        for daily_label, count in zip(labels, counts):
            key = np.datetime64(daily_label, "ns")
            partial = partial_sums.sel(_daily_time=key, drop=True).copy(deep=True)
            if key in daily_sums:
                DailyMean._validate_coordinate_schema(daily_sums[key], partial)
                current, partial = xr.align(
                    daily_sums[key],
                    partial,
                    join="exact",
                    copy=False,
                )
                partial = current + partial
            daily_sums[key] = partial
            daily_counts[key] = daily_counts.get(key, 0) + int(count)

    @staticmethod
    def _finalize_daily_means(
        daily_sums: dict[np.datetime64, xr.DataArray],
        daily_counts: dict[np.datetime64, int],
        time_list: list[datetime],
        variable_list: list[str],
    ) -> xr.DataArray:
        requested_times = np.asarray(time_list, dtype="datetime64[ns]")
        means: list[xr.DataArray] = []
        for daily_time in requested_times:
            key = np.datetime64(daily_time, "ns")
            count = daily_counts.get(key, 0)
            if count != _HOURS_PER_DAY:
                raise ValueError(
                    f"Daily aggregation for {key} received {count} hourly samples; "
                    f"expected {_HOURS_PER_DAY}"
                )
            means.append(daily_sums[key] / count)

        for mean in means[1:]:
            DailyMean._validate_coordinate_schema(means[0], mean)
        daily = xr.concat(
            means,
            dim=xr.IndexVariable("time", requested_times),
            join="exact",
        )
        return daily.sel(variable=variable_list)

    @staticmethod
    def _concat_variable_groups(
        groups: list[xr.DataArray],
        variable_list: list[str],
    ) -> xr.DataArray:
        for group in groups[1:]:
            DailyMean._validate_coordinate_schema(
                groups[0],
                group,
                ignored_dimensions=frozenset({"variable"}),
            )
        daily = xr.concat(groups, dim="variable", join="exact")
        return daily.sel(variable=variable_list)

    def _fetch_hourly_batches(
        self,
        hourly_times: list[datetime],
        daily_labels: npt.NDArray[np.datetime64],
        time_list: list[datetime],
        variable_batch: list[str],
    ) -> xr.DataArray:
        daily_sums: dict[np.datetime64, xr.DataArray] = {}
        daily_counts: dict[np.datetime64, int] = {}
        for start in range(0, len(hourly_times), self.time_batch_size):
            time_batch = hourly_times[start : start + self.time_batch_size]
            data = self.datasource(time_batch, variable_batch)
            data = self._validate_hourly_data(data, time_batch, variable_batch)
            data = self._prepare_hourly_data(data)
            data = self._validate_hourly_data(data, time_batch, variable_batch)
            self._accumulate_hourly_batch(
                data,
                daily_labels[start : start + len(time_batch)],
                daily_sums,
                daily_counts,
            )
            del data
        return self._finalize_daily_means(
            daily_sums,
            daily_counts,
            time_list,
            variable_batch,
        )

    async def _fetch_hourly_batches_async(
        self,
        hourly_times: list[datetime],
        daily_labels: npt.NDArray[np.datetime64],
        time_list: list[datetime],
        variable_batch: list[str],
    ) -> xr.DataArray:
        daily_sums: dict[np.datetime64, xr.DataArray] = {}
        daily_counts: dict[np.datetime64, int] = {}
        for start in range(0, len(hourly_times), self.time_batch_size):
            time_batch = hourly_times[start : start + self.time_batch_size]
            data = await self.datasource.fetch(time_batch, variable_batch)
            data = self._validate_hourly_data(data, time_batch, variable_batch)
            data = self._prepare_hourly_data(data)
            data = self._validate_hourly_data(data, time_batch, variable_batch)
            self._accumulate_hourly_batch(
                data,
                daily_labels[start : start + len(time_batch)],
                daily_sums,
                daily_counts,
            )
            del data
        return self._finalize_daily_means(
            daily_sums,
            daily_counts,
            time_list,
            variable_batch,
        )

    def _fetch_and_aggregate(
        self,
        time_list: list[datetime],
        variable_list: list[str],
    ) -> xr.DataArray:
        instantaneous, end_of_interval = self._split_variables(variable_list)
        groups: list[xr.DataArray] = []
        for variables, interval_ending in (
            (instantaneous, False),
            (end_of_interval, True),
        ):
            for start in range(0, len(variables), self.variable_batch_size):
                variable_batch = variables[start : start + self.variable_batch_size]
                hourly_times, daily_labels = self._hourly_times(
                    time_list,
                    interval_ending,
                )
                groups.append(
                    self._fetch_hourly_batches(
                        hourly_times,
                        daily_labels,
                        time_list,
                        variable_batch,
                    )
                )

        return self._concat_variable_groups(groups, variable_list)

    async def _fetch_and_aggregate_async(
        self,
        time_list: list[datetime],
        variable_list: list[str],
    ) -> xr.DataArray:
        instantaneous, end_of_interval = self._split_variables(variable_list)
        groups: list[xr.DataArray] = []
        for variables, interval_ending in (
            (instantaneous, False),
            (end_of_interval, True),
        ):
            for start in range(0, len(variables), self.variable_batch_size):
                variable_batch = variables[start : start + self.variable_batch_size]
                hourly_times, daily_labels = self._hourly_times(
                    time_list,
                    interval_ending,
                )
                data = await self._fetch_hourly_batches_async(
                    hourly_times,
                    daily_labels,
                    time_list,
                    variable_batch,
                )
                groups.append(data)

        return self._concat_variable_groups(groups, variable_list)


class FuXiS2SERA5(DailyMean):
    """Prepare hourly 0.25-degree ERA5 data for FuXi-S2S.

    This adapter forms the UTC daily means required by FuXi-S2S and point
    samples every sixth ERA5 grid coordinate to produce the model's native
    1.5-degree ``121 x 240`` grid. ``tp`` and ``ttr`` use interval-ending
    samples from 01:00 through 00:00 of the following day.

    Parameters
    ----------
    datasource : DataSource
        Hourly ERA5 source on the global 0.25-degree ``721 x 1440`` grid, such
        as :class:`earth2studio.data.ARCO_ERA5`.
    variable_batch_size : int, optional
        Maximum variables requested from the hourly source at once, by default
        13. This aligns a full FuXi-S2S request with each 13-level pressure
        variable family.
    time_batch_size : int, optional
        Maximum hourly timestamps requested from the hourly source at once, by
        default 1. Each full-resolution field is point-sampled before the next
        batch to bound peak memory.

    Note
    ----
    Returned values remain in Earth2Studio physical units. FuXi-S2S model-unit
    conversion is performed by the prognostic model wrapper.

    For more information see:

    - https://github.com/google-research/arco-era5
    - https://github.com/tpys/FuXi-S2S

    Warning
    -------
    Requested timestamps label complete UTC calendar-day means, not forecast
    issue times. The mean labeled day ``D`` requires data through ``D + 1`` at
    00:00 UTC. For a strict forecast issued at day ``D`` 00:00 UTC, use
    ``D - 1`` as the Earth2Studio workflow time so the two inputs are ``D - 2``
    and ``D - 1``.

    A full 76-variable request reads 24 hourly samples per variable and input
    day. With public cloud ERA5 this is a large transfer; the validated
    two-day ARCO workflow can take tens of minutes. The default batches align
    with FuXi-S2S pressure-level families and keep only one full-resolution
    hour in memory at a time.

    Examples
    --------
    >>> from earth2studio.data import ARCO_ERA5, FuXiS2SERA5
    >>> data = FuXiS2SERA5(ARCO_ERA5())

    Badges
    ------
    region:global dataclass:reanalysis dataset:era5 product:wind product:precip product:temp product:atmos product:ocean
    """

    ERA5_LAT = np.linspace(90, -90, 721, endpoint=True)
    ERA5_LON = np.linspace(0, 360, 1440, endpoint=False)
    FUXI_LAT = np.linspace(90, -90, 121, endpoint=True)
    FUXI_LON = np.linspace(0, 360, 240, endpoint=False)

    def __init__(
        self,
        datasource: DataSource,
        variable_batch_size: int = 13,
        time_batch_size: int = 1,
    ) -> None:
        super().__init__(
            datasource,
            end_of_interval_variables=("tp", "ttr"),
            variable_batch_size=variable_batch_size,
            time_batch_size=time_batch_size,
        )

    def _prepare_hourly_data(self, data: xr.DataArray) -> xr.DataArray:
        required_dims = {"time", "variable", "lat", "lon"}
        if set(data.dims) != required_dims:
            raise ValueError(
                "FuXiS2SERA5 requires exactly the dimensions "
                "('time', 'variable', 'lat', 'lon')"
            )
        data = data.transpose("time", "variable", "lat", "lon")
        if data.sizes["lat"] != len(self.ERA5_LAT) or data.sizes["lon"] != len(
            self.ERA5_LON
        ):
            raise ValueError(
                "FuXiS2SERA5 requires the global 0.25-degree ERA5 grid with "
                "shape (721, 1440)"
            )
        if not np.allclose(data.lat.values, self.ERA5_LAT):
            raise ValueError(
                "FuXiS2SERA5 requires ERA5 latitude coordinates from 90 to -90"
            )
        if not np.allclose(data.lon.values, self.ERA5_LON):
            raise ValueError(
                "FuXiS2SERA5 requires ERA5 longitude coordinates from 0 to 359.75"
            )

        sampled = data.isel(
            lat=slice(None, None, 6),
            lon=slice(None, None, 6),
        ).copy(deep=True)
        return sampled.assign_coords(
            lat=self.FUXI_LAT,
            lon=self.FUXI_LON,
        )
