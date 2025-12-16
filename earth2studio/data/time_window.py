# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from collections.abc import Callable
from datetime import datetime, timedelta

import xarray as xr

from earth2studio.data.base import DataSource
from earth2studio.data.utils import prep_data_inputs
from earth2studio.utils.type import TimeArray, VariableArray


class TimeWindow:
    """Wrapper for datasources that fetches data at multiple time offsets.

    This wrapper takes an existing datasource and fetches data at multiple temporal
    offsets relative to the requested time(s). The variable dimension is expanded
    to include all offset versions with appropriate suffixes.

    The `group_by` parameter controls the primary (outer) grouping dimension.

    By default (group_by='variable'), variables are grouped together with their offsets:
    - Example: offsets=[-6h, 0h, +6h], suffixes=['_tm1', '_t', '_tp1'],
      variable=['t2m', 'u10m']
    - Output: ['t2m_tm1', 't2m_t', 't2m_tp1', 'u10m_tm1', 'u10m_t', 'u10m_tp1']

    The order of variables in the output matches the order they are requested.

    Parameters
    ----------
    datasource : DataSource
        The underlying datasource to wrap. Must implement the
        :class:`~earth2studio.data.DataSource` protocol (single time dimension).
        :class:`ForecastSource` is not supported.

        TimeWindow-specific requirements: The datasource must return variable
        names without time suffixes (e.g., "tas", "u10m"). The wrapper will
        add suffixes (e.g., "tas_t-1", "u10m_t+1") when fetching data at
        offset times.
    offsets : list[timedelta]
        List of time offsets to fetch. For example:
        - [timedelta(hours=-6), timedelta(hours=0), timedelta(hours=6)]
        - [timedelta(days=-1), timedelta(days=0), timedelta(days=1)]
    suffixes : list[str]
        List of suffixes to append to variable names. Must be same length as offsets.
        For example: ['_tm1', '_t', '_tp1'] or ['_prev', '_curr', '_next']
    group_by : str, optional
        Primary grouping dimension for variable ordering. Default: 'variable'

        Variables are always ordered in both dimensions - this parameter controls
        which is the primary (outer) vs secondary (inner) grouping:

        - 'variable': Group by variable first
          Output: [var1_off1, var1_off2, ..., var2_off1, var2_off2, ...]
          All temporal offsets for each variable are grouped together.

        - 'offset': Group by offset first
          Output: [var1_off1, var2_off1, ..., var1_off2, var2_off2, ...]
          All variables for each time offset are grouped together.

        Within each grouping, the order of variables matches the order requested
        by the user, and the order of offsets matches the order in the offsets list.
    time_fn : Callable[[datetime], datetime], optional
        Function to transform the base time before fetching data. Useful for
        normalizing request times to match data availability (e.g., always
        request at 12:00 for daily data). The original time is preserved in
        the output coordinates. Default: identity function (no transformation).

    Notes
    -----
    Models requiring temporal context (e.g., t-1, t, t+1) could alternatively call
    the datasource multiple times with offset datetimes. TimeWindow centralizes this
    logic so that: (1) it works with any DataSource, (2) variable naming with suffixes
    is handled consistently, and (3) the model receives a standard DataSource interface
    without needing offset logic.

    This wrapper only supports :class:`~earth2studio.data.DataSource` (single time
    dimension). It does not support :class:`ForecastSource` (which has both init time
    and lead time).

    The wrapper is transparent and does not add validation beyond what the underlying
    datasource provides. Different datasources handle missing data differently (e.g.,
    GFS logs warnings and returns NaN, while CMIP6 raises ValueError). The wrapper
    preserves this behavior.

    Examples
    --------
    >>> from earth2studio.data import GFS, TimeWindow
    >>> from datetime import datetime, timedelta
    >>>
    >>> # Create wrapper that fetches t-6h, t, and t+6h
    >>> gfs = GFS()
    >>> wrapped_gfs = TimeWindow(
    ...     datasource=gfs,
    ...     offsets=[timedelta(hours=-6), timedelta(hours=0), timedelta(hours=6)],
    ...     suffixes=['_tm1', '_t', '_tp1']
    ... )
    >>>
    >>> # Request data for a single time and multiple variables
    >>> data = wrapped_gfs(datetime(2024, 1, 1), ['t2m', 'u10m'])
    >>> # Returns data with 6 variables ordered as:
    >>> # ['t2m_tm1', 't2m_t', 't2m_tp1', 'u10m_tm1', 'u10m_t', 'u10m_tp1']
    """

    def __init__(
        self,
        datasource: DataSource,
        offsets: list[timedelta],
        suffixes: list[str],
        group_by: str = "variable",
        time_fn: Callable[[datetime], datetime] = lambda x: x,
    ):
        self.datasource = datasource
        self.offsets = offsets
        self.suffixes = suffixes
        self.group_by = group_by
        self.time_fn = time_fn

        if not offsets:
            raise ValueError("offsets must be a non-empty list")

        if not suffixes:
            raise ValueError("suffixes must be a non-empty list")

        if len(offsets) != len(suffixes):
            raise ValueError(
                f"offsets and suffixes must have the same length. "
                f"Got {len(offsets)} offsets and {len(suffixes)} suffixes."
            )

        if group_by not in ["variable", "offset"]:
            raise ValueError(
                f"group_by must be 'variable' or 'offset', got '{group_by}'"
            )

    def _expand_variables_with_suffixes(self, base_vars: list[str]) -> list[str]:
        """Return expanded variable names (base + suffix) in group_by order.

        Parameters
        ----------
        base_vars : list[str]
            List of base variable names (without suffixes).

        Returns
        -------
        list[str]
            Expanded variable names with suffixes in deterministic order.
        """
        if self.group_by == "variable":
            # Group by variable first: [v1_s1, v1_s2, ..., v2_s1, v2_s2, ...]
            return [f"{v}{s}" for v in base_vars for s in self.suffixes]
        # group_by == "offset"
        # Group by offset first: [v1_s1, v2_s1, ..., v1_s2, v2_s2, ...]
        return [f"{v}{s}" for s in self.suffixes for v in base_vars]

    def _process_offset_data(
        self,
        data: xr.DataArray,
        time_list: list[datetime],
        variable_list: list[str],
        suffix: str,
        offset: timedelta,
    ) -> xr.DataArray:
        """Process data for a single offset: validate, update time coords, and rename variables.

        Parameters
        ----------
        data : xr.DataArray
            Raw data from datasource
        time_list : list[datetime]
            Original requested times (not offset times)
        variable_list : list[str]
            List of requested variable names
        suffix : str
            Suffix to append to variable names
        offset : timedelta
            Time offset (for error messages)

        Returns
        -------
        xr.DataArray
            Processed data array
        """
        # Verify we got the data we expected
        if len(data.time) != len(time_list):
            raise ValueError(
                f"Datasource returned {len(data.time)} times but expected "
                f"{len(time_list)} for offset {offset}"
            )

        # Update time coordinates to match original request (not offset)
        # This ensures all offset versions align on the same time axis
        data = data.assign_coords(time=time_list)

        # Rename variables to include temporal suffix
        new_var_names = [f"{var}{suffix}" for var in variable_list]
        data = data.assign_coords(variable=new_var_names)

        return data

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve data at multiple temporal offsets.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (these are the "t" times, offsets are
            applied relative to these).
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return. The order
            of variables in the request is preserved in the output.

        Returns
        -------
        xr.DataArray
            Data array with expanded variable dimension. The time dimension matches
            the input, but the variable dimension is expanded by len(offsets).
            Variables are named with suffixes indicating their temporal offset.
            The order of output variables is deterministic and controlled by the
            `group_by` parameter (see constructor documentation).

        Raises
        ------
        ValueError
            If requested offset times are invalid or unavailable in the datasource.
            The error message includes context about which offset failed.
        KeyError
            If requested variables are not available in the datasource.
        FileNotFoundError
            If data files for the requested times are not found.
        RuntimeError
            For unexpected errors during data fetching.

        Note
        ----
        Error behavior depends on the underlying datasource. Some datasources (e.g., GFS)
        may log warnings and return NaN values instead of raising errors for missing data.
        The wrapper does not add additional validation.
        """
        # Keep original time for output coordinates
        time_list, variable_list = prep_data_inputs(time, variable)

        suffix_to_bases, output_order = self._prepare_variable_requests(variable_list)

        # Apply time_fn to get base time for fetching (but keep original time_list for output)
        fetch_time_list = [self.time_fn(t) for t in time_list]

        # Collect data for all offset times
        offset_data_arrays = []
        collected_names: list[str] = []

        for offset, suffix in zip(self.offsets, self.suffixes):
            base_vars = [str(b) for b in suffix_to_bases.get(suffix, [])]
            if not base_vars:
                continue

            # Compute offset times using transformed fetch times
            offset_times = [t + offset for t in fetch_time_list]

            try:
                # Fetch data from underlying datasource
                data = self.datasource(offset_times, base_vars)
            except (ValueError, KeyError, FileNotFoundError) as e:
                # Re-raise common datasource errors with additional context
                error_msg = (
                    f"Failed to fetch data for offset {offset} (suffix: {suffix}). "
                    f"Requested times: {offset_times}. "
                    f"Original error: {str(e)}"
                )
                raise type(e)(error_msg) from e
            except Exception as e:
                # Wrap unexpected errors as RuntimeError
                raise RuntimeError(
                    f"Unexpected error fetching data for offset {offset} (suffix: {suffix}). "
                    f"Requested times: {offset_times}. "
                    f"Original error: {str(e)}"
                ) from e

            # Process the fetched data
            data = self._process_offset_data(data, time_list, base_vars, suffix, offset)
            offset_data_arrays.append(data)
            collected_names.extend(f"{base}{suffix}" for base in base_vars)

        # Concatenate along variable dimension
        if not offset_data_arrays:
            raise ValueError(
                "No variables fetched; check requested variables and suffixes."
            )

        result = xr.concat(offset_data_arrays, dim="variable")

        # Use collected_names directly for ordering (they're already correct from _process_offset_data)
        ordered_vars = [var for var in output_order if var in collected_names]
        result = result.sel(variable=ordered_vars)

        return result

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async retrieve data at multiple temporal offsets.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (these are the "t" times, offsets are
            applied relative to these).
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return. The order
            of variables in the request is preserved in the output.

        Returns
        -------
        xr.DataArray
            Data array with expanded variable dimension. The time dimension matches
            the input, but the variable dimension is expanded by len(offsets).
            Variables are named with suffixes indicating their temporal offset.
            The order of output variables is deterministic and controlled by the
            `group_by` parameter (see constructor documentation).

        Raises
        ------
        ValueError
            If requested offset times are invalid or unavailable in the datasource.
            The error message includes context about which offset failed.
        KeyError
            If requested variables are not available in the datasource.
        FileNotFoundError
            If data files for the requested times are not found.
        RuntimeError
            For unexpected errors during data fetching.

        Note
        ----
        Error behavior depends on the underlying datasource. Some datasources (e.g., GFS)
        may log warnings and return NaN values instead of raising errors for missing data.
        The wrapper does not add additional validation.
        """
        time_list, variable_list = prep_data_inputs(time, variable)

        suffix_to_bases, output_order = self._prepare_variable_requests(variable_list)

        # Check if underlying datasource supports async
        if not hasattr(self.datasource, "fetch"):
            raise AttributeError(
                f"Underlying datasource {type(self.datasource).__name__} does not "
                "support async fetch"
            )

        # Apply time_fn to get base time for fetching (but keep original time_list for output)
        fetch_time_list = [self.time_fn(t) for t in time_list]

        # Collect data for all offset times
        offset_data_arrays = []
        collected_names: list[str] = []

        for offset, suffix in zip(self.offsets, self.suffixes):
            base_vars = [str(b) for b in suffix_to_bases.get(suffix, [])]
            if not base_vars:
                continue

            # Compute offset times using transformed fetch times
            offset_times = [t + offset for t in fetch_time_list]

            try:
                # Fetch data from underlying datasource
                data = await self.datasource.fetch(offset_times, base_vars)
            except (ValueError, KeyError, FileNotFoundError) as e:
                # Re-raise common datasource errors with additional context
                error_msg = (
                    f"Failed to fetch data for offset {offset} (suffix: {suffix}). "
                    f"Requested times: {offset_times}. "
                    f"Original error: {str(e)}"
                )
                raise type(e)(error_msg) from e
            except Exception as e:
                # Wrap unexpected errors as RuntimeError
                raise RuntimeError(
                    f"Unexpected error fetching data for offset {offset} (suffix: {suffix}). "
                    f"Requested times: {offset_times}. "
                    f"Original error: {str(e)}"
                ) from e

            # Process the fetched data
            data = self._process_offset_data(data, time_list, base_vars, suffix, offset)
            offset_data_arrays.append(data)
            collected_names.extend(f"{base}{suffix}" for base in base_vars)

        # Concatenate along variable dimension
        if not offset_data_arrays:
            raise ValueError(
                "No variables fetched; check requested variables and suffixes."
            )

        result = xr.concat(offset_data_arrays, dim="variable")

        # Use collected_names directly for ordering (they're already correct from _process_offset_data)
        ordered_vars = [var for var in output_order if var in collected_names]
        result = result.sel(variable=ordered_vars)

        return result

    def _prepare_variable_requests(
        self, variable_list: list[str]
    ) -> tuple[dict[str, list[str]], list[str]]:
        """Map requested variables to base variables per suffix and determine output order."""
        suffix_to_bases: dict[str, list[str]] = {suffix: [] for suffix in self.suffixes}
        suffix_seen: dict[str, set[str]] = {suffix: set() for suffix in self.suffixes}

        output_order: list[str] = []
        unsuffixed_vars: list[str] = []

        for var in variable_list:
            var_str = str(var)
            matched_suffix = None
            for suffix in self.suffixes:
                if var_str.endswith(suffix):
                    base = var_str[: -len(suffix)]
                    base = base if base else ""
                    if base not in suffix_seen[suffix]:
                        suffix_to_bases[suffix].append(base)
                        suffix_seen[suffix].add(base)
                    output_order.append(var_str)
                    matched_suffix = suffix
                    break
            if matched_suffix is None:
                unsuffixed_vars.append(var_str)

        if unsuffixed_vars:
            for base in unsuffixed_vars:
                for suffix in self.suffixes:
                    if base not in suffix_seen[suffix]:
                        suffix_to_bases[suffix].append(base)
                        suffix_seen[suffix].add(base)

            # Extend output_order based on group_by policy
            output_order.extend(self._expand_variables_with_suffixes(unsuffixed_vars))

        return suffix_to_bases, output_order
