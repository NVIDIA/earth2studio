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

import os
import shutil
import warnings
from datetime import datetime
from typing import Union, cast

import cftime
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

# Project-level imports come after stdlib/third-party
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import intake_esgf
    from scipy.interpolate import griddata
except ImportError:
    OptionalDependencyFailure("data")
    intake_esgf = None
    griddata = None

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.cmip6 import CMIP6Lexicon
from earth2studio.utils.type import TimeArray, VariableArray

# cftime concrete classes accepted (define after all imports to avoid E402)
CFDatetime = Union[
    cftime.DatetimeNoLeap,
    cftime.Datetime360Day,
    cftime.DatetimeGregorian,
]


@check_optional_dependencies()
class CMIP6:
    """CMIP6 data source for Earth2Studio.

    This class provides a thin convenience wrapper around the `intake-esgf` catalog that
    hosts the Coupled Model Inter-comparison Project Phase 6 (CMIP6) archive. This is
    meant to provide a seemless interface to the CMIP6 archive for Earth2Studio however
    the CMIP6 archive is very large there may be data that will break this interface.
    Currently this supports both atmospheric and oceanic datasets.

    Parameters
    ----------
    experiment_id : str
        CMIP6 experiment identifier (e.g. "historical", "ssp585")
    source_id : str
        CMIP6 model identifier (e.g. "MPI-ESM1-2-LR")
    table_id : str
        CMOR table describing variable realm/frequency ("Amon", "Omon", "SImon")
    variant_label : str
        Ensemble member / initial-condition label such as "r1i1p1f1".
    file_start : str, optional
        Optional filename prefix filters forwarded to ``ESGFCatalog.search`` to
        constrain the final dataset selection. Leave None to accept all, by default None
    file_end : str, optional
        Optional filename suffix filters forwarded to ``ESGFCatalog.search`` to
        constrain the final dataset selection. Leave None to accept all, by default None
    cache : bool, optional
        Cache data source on local memory, by default True. Multiple CMIP6 instances
        can safely share the same cache directory as intake-esgf automatically organizes
        files into detailed subdirectories by project, model, experiment, variant, variable,
        and version, preventing any conflicts.
    verbose : bool, optional
        Print download progress, by default True
    exact_time_match : bool, optional
        If True, raise an error when requested times don't match dataset times exactly.
        If False (default), use nearest neighbor time matching and issue a warning, by default False

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests. The source data used is provided in an
    unoptimized format.

    Note
    ----
    Additional information on the CMIP6 data repository can be referenced here:

    - https://esgf-node.llnl.gov/search/cmip6/

    The intake-esgf package is used to search the CMIP6 data repository and load the
    data into an xarray dataset. Additional information on the intake-esgf package can
    be referenced here:

    - https://intake-esgf.readthedocs.io/en/latest/

    Note
    ----
    By default, this data source will retrieve the closest time available using nearest
    neighbor matching. Depending on the experiment and temporal resolution, this may be
    significantly different than what was requested. Set `exact_time_match=True` to
    enforce exact time matching if precise timestamps are critical.
    """

    def __init__(
        self,
        experiment_id: str,
        source_id: str,
        table_id: str,
        variant_label: str,
        file_start: str | None = None,
        file_end: str | None = None,
        cache: bool = True,
        verbose: bool = True,
        exact_time_match: bool = False,
    ):
        self.experiment_id = experiment_id
        self.source_id = source_id
        self.table_id = table_id
        self.variant_label = variant_label
        self.file_start = file_start
        self.file_end = file_end
        self._cache = cache
        self._verbose = verbose
        self._exact_time_match = exact_time_match

        # Create catalog
        intake_esgf.conf.set(local_cache=self.cache)
        self.catalog = intake_esgf.ESGFCatalog()

        # Search for all available data (no variable_id filter) - metadata only
        self._search_catalog(
            self.catalog,
            self.experiment_id,
            self.source_id,
            self.table_id,
            self.variant_label,
            self.file_start,
            self.file_end,
        )
        # Extract available variables from catalog metadata (no data download)
        self.available_variables: set[str] = set(
            self.catalog.df["variable_id"].unique()
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Loaded data array
        """

        # Prepare data inputs
        time, variable = prep_data_inputs(time, variable)

        # Convert variable to CMIP6 IDs
        cmip6_ids_set: set[str] = {CMIP6Lexicon.get_item(v)[0][0] for v in variable}
        cmip6_variable_ids = list(cmip6_ids_set)

        # Validate that requested variables are available
        if cmip6_ids_set - self.available_variables:  # pragma: no cover
            raise IndexError(
                f"Variable(s) {cmip6_ids_set - self.available_variables} not found in CMIP6 dataset. "
                f"Available variables: {sorted(self.available_variables)}"
            )

        # Now download only the requested variables
        self._search_catalog(
            self.catalog,
            self.experiment_id,
            self.source_id,
            self.table_id,
            self.variant_label,
            self.file_start,
            self.file_end,
            variable_id=cmip6_variable_ids,
        )
        dsd = self.catalog.to_dataset_dict(
            prefer_streaming=False, add_measures=False
        )  # NOTE: it may be better to use streaming however this resulted in lots of errors

        # Get lat/lon and calendar from first dataset
        ds = dsd[list(dsd.keys())[0]]

        # Regular lat-lon grid → lat/lon 1-D coordinates exist
        if {"lat", "lon"} <= set(ds.coords):
            lat_1d = ds["lat"].values
            lon_1d = ds["lon"].values
            grid_shape = (len(lat_1d), len(lon_1d))
            da_dims_xy = ("lat", "lon")
            coord_dict: dict[str, object] = {
                "lat": lat_1d,
                "lon": lon_1d,
            }
        # Curvilinear ocean/ice grid → 2-D latitude/longitude coordinates
        elif {"latitude", "longitude"} <= set(ds.coords):
            lat2d = ds["latitude"].values
            lon2d = ds["longitude"].values
            # Use underlying dimension names (typically j,i or y,x)
            y_dim, x_dim = ds["latitude"].dims
            grid_shape = lat2d.shape  # (ny, nx)
            da_dims_xy = (y_dim, x_dim)
            coord_dict = {
                y_dim: ds[y_dim],
                x_dim: ds[x_dim],
                "_lat": (da_dims_xy, lat2d),
                "_lon": (da_dims_xy, lon2d),
            }
        else:  # pragma: no cover
            raise ValueError(
                "Unable to determine horizontal coordinates for CMIP6 dataset – expected 'lat/lon' or 'latitude/longitude'."
                f"\nDataset: {ds}"
            )

        # Find the available times in the first dataset.
        # Preserve original requested times for reference while sampling data.
        requested_times = self._convert_times_to_datetime(time)
        selection_times = self._convert_times_to_cftime(
            requested_times, ds.time.dt.calendar
        )

        # Select times based on matching mode
        if self._exact_time_match:
            # Try exact match
            try:
                ds_selected = ds.sel(time=selection_times)  # type: ignore[arg-type]
            except KeyError as e:
                # Find which times are missing
                available_times_cftime = ds.time.values
                available_times_dt = self._convert_times_to_datetime(
                    list(available_times_cftime)
                )
                missing_times = [
                    t for t in requested_times if t not in available_times_dt
                ]
                raise ValueError(
                    f"Exact time match required but the following timestamps were not found in the CMIP6 dataset: {missing_times}. "
                    f"Available times: {available_times_dt[:10]}... (showing first 10). "
                    f"Set exact_time_match=False to use nearest neighbor matching instead."
                ) from e
        else:
            # Use nearest neighbor matching (default behavior)
            ds_selected = ds.sel(time=selection_times, method="nearest")  # type: ignore[arg-type]

        # Extract selected times (common for both modes)
        selected_times = ds_selected.time.values  # cftime objects in dataset calendar
        selected_times_dt = self._convert_times_to_datetime(list(selected_times))

        # Warn if using nearest neighbor and times don't match exactly
        if not self._exact_time_match and not np.array_equal(
            np.asarray(requested_times, dtype=object),
            np.asarray(selected_times_dt, dtype=object),
        ):
            warnings.warn(
                "One or more requested timestamps were not found exactly in the CMIP6 dataset; "
                "nearest available snapshots have been substituted.",
                UserWarning,
            )

        # Use selected (actual available) times for data subsetting
        selection_times = selected_times

        # Subset time; rely on xarray's KeyError if a timestamp is missing
        for var_name, ds_var in dsd.items():
            dsd[var_name] = ds_var.sel(time=selection_times)  # type: ignore[arg-type]

        # Make data array
        da = xr.DataArray(
            data=np.empty((len(time), len(variable), *grid_shape), dtype=np.float32),
            dims=["time", "variable", *da_dims_xy],
            coords={
                "time": selected_times_dt,
                # Keep the originally requested timestamps alongside the actual sampled times
                "time_requested": ("time", requested_times),
                "variable": variable,
                **coord_dict,
            },
        )

        # Populate the array
        for var_idx, var in enumerate(
            tqdm(
                variable,
                total=len(variable),
                desc="Fetching CMIP6 variables",
                disable=(not self._verbose),
            )
        ):

            # Get variable, level, and modifier
            cmip6_entry = CMIP6Lexicon.get_item(var)
            (cmip6_var, level), modifier = cmip6_entry

            # Get data array
            ds_var = dsd[cmip6_var]
            data_arr = ds_var[cmip6_var]  # DataArray inside the dataset

            # Select pressure level if needed
            if level != -1:
                # Convert hPa → Pa to match dataset units
                target_pa = level * 100.0
                if "plev" not in data_arr.coords:  # pragma: no cover
                    raise ValueError(
                        f"Variable '{cmip6_var}' expected to have a 'plev' coordinate but none found"
                    )

                try:
                    data_arr = data_arr.sel(plev=target_pa)
                except KeyError as e:  # pragma: no cover
                    available = data_arr.plev.values / 100.0
                    raise ValueError(
                        f"Requested pressure level {level} hPa for variable '{cmip6_var}' not found."
                        f" Available levels: {available}"
                    ) from e

            # At this point data_arr dims should be (time, lat, lon)
            # Convert to numpy (trigger load) and apply modifier
            da.values[:, var_idx, :, :] = modifier(data_arr.values)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return da

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
        experiment_id: str,
        source_id: str,
        table_id: str,
        variant_label: str,
    ) -> bool:
        """Check if the requested *exact* timestamp exists in the ESGF archive.

        Parameters
        ----------
        time : datetime | np.datetime64
            Timestamp to test (UTC).
        experiment_id : str
            CMIP6 experiment identifier (e.g. "historical", "ssp585").
        source_id : str
            CMIP6 model identifier (e.g. "MPI-ESM1-2-LR").
        table_id : str
            CMOR table describing variable realm/frequency ("Amon", "Omon", "SImon" ...).
        variant_label : str
            Ensemble member / initial-condition label such as "r1i1p1f1".

        Warning
        -------
        This method downloads data from ESGF servers to verify the time coordinate
        range. It is not a lightweight metadata-only check.

        Notes
        -----
        The check performs an ESGF search and downloads at least one file to read
        its time coordinate bounds. If the target timestamp lies within the dataset's
        time span, `True` is returned. Otherwise returns `False`.
        """

        if isinstance(time, np.datetime64):  # np.datetime64 → datetime
            time = time.astype("datetime64[us]").astype(datetime)

        # Search for data
        try:
            cat = intake_esgf.ESGFCatalog()
            cat.search(
                experiment_id=experiment_id,
                source_id=source_id,
                table_id=table_id,
                variant_label=variant_label,
            )
        except Exception:
            return False

        # Just get the first record
        cat.df = cat.df.iloc[0:1]
        dsd = cat.to_dataset_dict(prefer_streaming=True, add_measures=False)
        ds = next(iter(dsd.values()))

        # Confirm timestamp lies inside at least one dataset’s time axis
        t0 = ds.time.min().item()
        t1 = ds.time.max().item()
        t0 = CMIP6._convert_times_to_datetime([t0])[0]
        t1 = CMIP6._convert_times_to_datetime([t1])[0]

        return t0 <= time <= t1

    @property
    def cache(self) -> str:
        """Get the appropriate cache location.

        Note
        ----
        Multiple CMIP6 instances can safely share the same base cache directory.
        The intake-esgf package automatically creates detailed subdirectories within
        the cache for each unique dataset, organized by project, institution, model,
        experiment, variant, table, variable, grid, and version. This prevents any
        cache conflicts between different CMIP6 sources.

        For example, files are cached in paths like:
        ``<cache>/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r1i1p2f1/day/tas/gn/v20190429/clt_day_CanESM5_ssp585_r1i1p2f1_gn_20150101-21001231.nc``

        Returns
        -------
        str
            Path to cache directory.
        """
        cache_location = os.path.join(datasource_cache_root(), "cmip6")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp")
        return cache_location

    @staticmethod
    def _search_catalog(
        catalog: "intake_esgf.ESGFCatalog",
        experiment_id: str,
        source_id: str,
        table_id: str,
        variant_label: str,
        file_start: str | None = None,
        file_end: str | None = None,
        variable_id: list[str] | None = None,
    ) -> None:
        """Search the ESGF catalog with consistent error handling.

        Parameters
        ----------
        catalog : intake_esgf.ESGFCatalog
            The catalog to search
        experiment_id : str
            CMIP6 experiment identifier
        source_id : str
            CMIP6 model identifier
        table_id : str
            CMOR table describing variable realm/frequency
        variant_label : str
            Ensemble member / initial-condition label
        file_start : str | None
            Optional filename prefix filter
        file_end : str | None
            Optional filename suffix filter
        variable_id : list[str] | None
            Optional list of variable IDs to filter by

        Raises
        ------
        ValueError
            If the search fails with context about the parameters
        """
        search_params: dict[str, str | list[str]] = {
            "experiment_id": experiment_id,
            "source_id": source_id,
            "table_id": table_id,
            "variant_label": variant_label,
        }

        if file_start is not None:
            search_params["file_start"] = file_start
        if file_end is not None:
            search_params["file_end"] = file_end
        if variable_id is not None:
            search_params["variable_id"] = variable_id

        try:
            catalog.search(**search_params)
        except Exception as e:
            error_msg = (
                f"Error searching for CMIP6 data: {e}\n"
                f"Experiment ID: {experiment_id}\n"
                f"Source ID: {source_id}\n"
                f"Table ID: {table_id}\n"
                f"Variant Label: {variant_label}"
            )
            if file_start:
                error_msg += f"\nFile Start: {file_start}"
            if file_end:
                error_msg += f"\nFile End: {file_end}"
            if variable_id:
                error_msg += f"\nVariable ID: {variable_id}"
            raise ValueError(error_msg) from e

    @staticmethod
    def _convert_times_to_cftime(
        raw_times: list[Union[datetime, np.datetime64, "CFDatetime"]],
        calendar: str,
    ) -> list[object]:
        """Convert python/NumPy datetimes to matching ``cftime`` objects.

        Parameters
        ----------
        raw_times : list
            List of datetime-like objects (``datetime``, ``numpy.datetime64`` or
            already-converted ``cftime`` objects).
        calendar : str
            The calendar type of the dataset.

        Returns
        -------
        List of times converted to the appropriate ``cftime`` class.
        """
        converted: list[object] = []
        for ts in raw_times:
            if isinstance(ts, np.datetime64):
                ts = ts.astype("datetime64[us]").astype(datetime)
            if isinstance(ts, datetime):
                y, m, d, H, Mi, S = (
                    ts.year,
                    ts.month,
                    ts.day,
                    ts.hour,
                    ts.minute,
                    ts.second,
                )
                if calendar in ("noleap", "365_day"):
                    converted.append(cftime.DatetimeNoLeap(y, m, d, H, Mi, S))
                elif calendar in ("360_day",):
                    converted.append(cftime.Datetime360Day(y, m, d, H, Mi, S))
                else:
                    converted.append(cftime.DatetimeGregorian(y, m, d, H, Mi, S))
            else:
                raise TypeError(f"Unsupported time type: {type(ts)}")

        return converted

    @staticmethod
    def _convert_times_to_datetime(
        raw_times: list[Union[datetime, np.datetime64, "CFDatetime"]]
    ) -> list[datetime]:
        """Convert a list of mixed time objects (cftime, numpy.datetime64, datetime)
        to Python ``datetime`` objects for xarray coordinate storage.

        Notes
        -----
        • cftime calendars that contain impossible Gregorian dates (e.g., 360-day
          calendar with day 30 for February) *can* still be represented by the
          standard ``datetime`` class because those dates are numerically valid.
          Only leap calendars with a 366th day of year 366 would fail; CMIP6
          native calendars avoid that case for monthly/daily data we target.
        """

        converted: list[datetime] = []
        for ts in raw_times:
            if isinstance(ts, np.datetime64):
                ts = ts.astype("datetime64[us]").astype(datetime)

            if isinstance(ts, datetime):
                converted.append(ts)
            elif ts.__class__.__module__.startswith("cftime"):
                ts_cf = cast(CFDatetime, ts)
                converted.append(
                    datetime(
                        ts_cf.year,
                        ts_cf.month,
                        ts_cf.day,
                        ts_cf.hour,
                        ts_cf.minute,
                        ts_cf.second,
                    )
                )
            else:
                raise TypeError(
                    f"Unsupported time type for datetime conversion: {type(ts)}"
                )

        return converted


@check_optional_dependencies()
class CMIP6MultiRealm:
    """CMIP6 data source for Earth2Studio with multiple realms.

    This class allows combining multiple CMIP6 data sources from different realms
    (e.g., atmosphere, ocean, sea ice) into a single unified interface. Variables
    are fetched from each source in the order provided, and data on different grids
    are automatically regridded to a common regular lat/lon grid.

    Parameters
    ----------
    cmip6_source_list : list[CMIP6]
        List of CMIP6 data sources to combine. Variables will be fetched from
        sources in the order they appear in the list. All sources must have the
        same `exact_time_match` setting.

    Raises
    ------
    ValueError
        If cmip6_source_list is empty or if sources have different exact_time_match settings.
    TypeError
        If any item in cmip6_source_list is not a CMIP6 instance.

    Note
    ----
    When multiple sources have different grids, curvilinear grids (e.g., from ocean
    or sea ice models) will be interpolated to the first regular lat/lon grid found
    using nearest-neighbor interpolation.

    All CMIP6 sources must be initialized with the same `exact_time_match` setting
    to ensure consistent time matching behavior across realms.
    """

    def __init__(self, cmip6_source_list: list[CMIP6]):
        if not cmip6_source_list:
            raise ValueError("cmip6_source_list cannot be empty")

        # Validate that all items are CMIP6 instances
        for i, source in enumerate(cmip6_source_list):
            if not isinstance(source, CMIP6):
                raise TypeError(
                    f"Item at index {i} in cmip6_source_list is not a CMIP6 instance. "
                    f"Got {type(source).__name__} instead."
                )

        # Validate that all sources have the same exact_time_match setting
        first_exact_time_match = cmip6_source_list[0]._exact_time_match
        for i, source in enumerate(cmip6_source_list[1:], start=1):
            if source._exact_time_match != first_exact_time_match:
                raise ValueError(
                    f"All CMIP6 sources must have the same exact_time_match setting. "
                    f"Source 0 has exact_time_match={first_exact_time_match}, "
                    f"but source {i} has exact_time_match={source._exact_time_match}."
                )

        self.cmip6_source_list = cmip6_source_list

        # Collect all available variables from all sources
        self.available_variables: set[str] = set()
        for source in cmip6_source_list:
            self.available_variables.update(source.available_variables)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve data from multiple CMIP6 sources and combine into single array.

        This method fetches the requested variables from the available CMIP6 sources,
        automatically regridding data from different grids to a common grid, and
        combines all variables into a single DataArray.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Variable(s) to retrieve. Each variable will be fetched from the first
            source in the list that has it available.

        Returns
        -------
        xr.DataArray
            Combined data array with dimensions (time, variable, lat, lon) or
            (time, variable, j, i) depending on the grid type.

        Raises
        ------
        ValueError
            If any requested variables are not found in any of the sources.
        NotImplementedError
            If all sources use curvilinear grids (at least one regular lat/lon grid required).

        Note
        ----
        Variables are retrieved from sources in the order they appear in
        cmip6_source_list. If multiple sources contain the same variable,
        only the first one will be used.

        Curvilinear grids (ocean/sea ice) are regridded to regular grids using
        nearest-neighbor interpolation to preserve data coverage near coastlines.
        At least one source with a regular lat/lon grid (typically atmospheric data)
        is required when combining multiple sources with different grids.
        """
        da_list = []
        var_done = []
        # get variables from the datasources in the order they are available
        for cmip6_source in self.cmip6_source_list:
            # get available variables
            # Get variable, level, and modifier
            var_available = []
            for v in variable:
                cmip6_entry = CMIP6Lexicon.get_item(v)
                (cmip6_var, _), _ = cmip6_entry
                if cmip6_var in cmip6_source.available_variables:
                    var_available.append(v)

            var_todo = [v for v in var_available if v not in var_done]

            if not var_todo:
                continue

            da_list.append(cmip6_source(time, var_todo))
            var_done.extend(var_todo)

        # Check if any variables were found
        if not da_list:
            raise ValueError(
                f"None of the requested variables {variable} were found in any of the provided CMIP6 sources"
            )

        # Check if ALL requested variables were found
        var_missing = [v for v in variable if v not in var_done]
        if var_missing:
            # Get CMIP6 variable IDs for the missing variables
            missing_cmip6_vars = []
            for v in var_missing:
                cmip6_entry = CMIP6Lexicon.get_item(v)
                (cmip6_var, _), _ = cmip6_entry
                missing_cmip6_vars.append(f"{v} (CMIP6: {cmip6_var})")

            raise ValueError(
                f"Variable(s) {missing_cmip6_vars} not found in any of the provided CMIP6 sources. "
                f"Found variables: {var_done}. "
                f"Available variables across all sources: {sorted(self.available_variables)}"
            )

        # Regrid all data arrays to a common grid if needed
        da_list = self._regrid_to_common_grid(da_list)

        # Combine all data arrays along variable dimension
        result = xr.concat(da_list, dim="variable")

        # Reorder variables to match the requested order
        result = result.sel(variable=variable)
        return result

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
        cmip6_source_list: list[CMIP6],
    ) -> bool:
        """Check if the requested timestamp is available in all sources.

        Parameters
        ----------
        time : datetime | np.datetime64
            Timestamp to test (UTC).
        cmip6_source_list : list[CMIP6]
            List of CMIP6 data sources to check.

        Returns
        -------
        bool
            True if the timestamp is available in all sources, False otherwise.

        Warning
        -------
        This method may download data from ESGF servers for each source to check
        time availability. For multiple sources, this can result in significant
        data transfer.

        Notes
        -----
        This method checks that ALL sources have data available at the requested time,
        since combining multi-realm data requires data from all sources. Each source
        is checked by downloading at least one file to verify the time coordinate range.
        """
        if not cmip6_source_list:
            return False

        # Check that ALL sources have data available at the requested time
        for source in cmip6_source_list:
            if not CMIP6.available(
                time,
                source.experiment_id,
                source.source_id,
                source.table_id,
                source.variant_label,
            ):
                return False

        return True

    @property
    def cache(self) -> list[str]:
        """Get cache locations from all CMIP6 sources.

        Returns
        -------
        list[str]
            List of cache directory paths, one for each source in cmip6_source_list.
            All sources share the same base cache directory (`<root>/cmip6/`), with
            sources that have `cache=False` using a `tmp/` subdirectory.

        Note
        ----
        Multiple CMIP6 sources can safely share the same base cache directory.
        The intake-esgf package automatically organizes files into detailed
        subdirectories within the cache, preventing any conflicts between different
        datasets.
        """
        return [source.cache for source in self.cmip6_source_list]

    def _regrid_to_common_grid(self, da_list: list[xr.DataArray]) -> list[xr.DataArray]:
        """Regrid all data arrays to a common regular lat/lon grid.

        Uses the first regular (non-curvilinear) grid as the target.
        Curvilinear grids (with _lat/_lon coords) are interpolated to this target
        using nearest-neighbor interpolation.

        Parameters
        ----------
        da_list : list[xr.DataArray]
            List of data arrays to regrid

        Returns
        -------
        list[xr.DataArray]
            List of regridded data arrays on common grid

        Raises
        ------
        NotImplementedError
            If all sources use curvilinear grids (no regular lat/lon grid found)
        """
        if len(da_list) == 1:
            return da_list

        # Find the target regular grid (first one with 'lat'/'lon' coords)
        target_idx = None
        for idx, da in enumerate(da_list):
            if "lat" in da.coords and "lon" in da.coords:
                target_idx = idx
                break

        if target_idx is None:
            raise NotImplementedError(
                "Regridding between multiple curvilinear grids is not yet supported. "
                "All CMIP6 sources use curvilinear grids. "
                "At least one source with a regular lat/lon grid "
                "is required to serve as the target grid for interpolation."
            )

        target_da = da_list[target_idx]
        target_lats = target_da["lat"].values
        target_lons = target_da["lon"].values

        # Regrid curvilinear datasets to target
        regridded_list = []
        for da in da_list:
            if "_lat" in da.coords and "_lon" in da.coords:
                # Curvilinear grid - needs interpolation
                regridded = self._interpolate_curvilinear_to_regular(
                    da, target_lats, target_lons, target_da
                )
                regridded_list.append(regridded)
            else:
                # Already regular grid
                regridded_list.append(da)

        return regridded_list

    def _interpolate_curvilinear_to_regular(
        self,
        da_curvilinear: xr.DataArray,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
        target_da: xr.DataArray,
    ) -> xr.DataArray:
        """Interpolate curvilinear grid data to a regular lat/lon grid.

        Uses nearest-neighbor interpolation to preserve data coverage near
        coastlines and avoid NaN expansion issues common with linear/cubic methods.

        Parameters
        ----------
        da_curvilinear : xr.DataArray
            Data array with curvilinear grid (_lat, _lon coords)
        target_lats : np.ndarray
            Target latitude coordinates
        target_lons : np.ndarray
            Target longitude coordinates
        target_da : xr.DataArray
            Target data array to match dimensions

        Returns
        -------
        xr.DataArray
            Regridded data array on regular grid
        """
        # Get source curvilinear coordinates
        source_lons = da_curvilinear["_lon"].values.flatten()
        source_lats = da_curvilinear["_lat"].values.flatten()
        points = np.column_stack((source_lats, source_lons))

        # Create target grid (note: meshgrid returns lon first, lat second)
        target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)

        # Prepare output array
        n_time, n_var = da_curvilinear.shape[:2]
        n_lat, n_lon = len(target_lats), len(target_lons)
        regridded_data = np.empty((n_time, n_var, n_lat, n_lon), dtype=np.float32)
        regridded_data[:] = np.nan

        # Interpolate each time/variable slice
        values = da_curvilinear.values
        for i_time in range(n_time):
            for i_var in range(n_var):
                slice_values = values[i_time, i_var, :, :].flatten()

                regridded_data[i_time, i_var, :, :] = griddata(
                    points,
                    slice_values,
                    (target_lat_grid, target_lon_grid),
                    method="nearest",
                )

        # Create new DataArray with target coordinates
        coords = target_da.coords.copy()
        coords["variable"] = da_curvilinear.coords["variable"]

        return xr.DataArray(regridded_data, dims=target_da.dims, coords=coords)
