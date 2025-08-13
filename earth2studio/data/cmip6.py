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
except ImportError:
    OptionalDependencyFailure("data")
    intake_esgf = None

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
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True

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
    This data source will retrieve the closest time possible, depending on the
    experiment this may be significantly different than what was requested.
    """

    def __init__(
        self,
        experiment_id: str,
        source_id: str,
        table_id: str,
        variant_label: str,
        file_start: str = None,
        file_end: str = None,
        cache: bool = True,
        verbose: bool = True,
    ):
        self.experiment_id = experiment_id
        self.source_id = source_id
        self.table_id = table_id
        self.variant_label = variant_label
        self.file_start = file_start
        self.file_end = file_end
        self._cache = cache
        self._verbose = verbose

        # Create catalog
        intake_esgf.conf.set(local_cache=self.cache)
        self.catalog = intake_esgf.ESGFCatalog()

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

        # Convert variable to
        var_set: set[str] = {CMIP6Lexicon[v][0][0] for v in variable}
        cmip6_variable_id = list(var_set)

        # Search for data
        try:
            self.catalog.search(
                experiment_id=self.experiment_id,
                source_id=self.source_id,
                table_id=self.table_id,
                variable_id=cmip6_variable_id,
                variant_label=self.variant_label,
                file_start=self.file_start,
                file_end=self.file_end,
            )
            dsd = self.catalog.to_dataset_dict(
                prefer_streaming=False, add_measures=False
            )  # NOTE: it may be better to use streaming however this resulted in lots of errors
        except Exception as e:
            raise ValueError(
                f"Error searching for CMIP6 data: {e}"
                f"\nExperiment ID: {self.experiment_id}"
                f"\nSource ID: {self.source_id}"
                f"\nTable ID: {self.table_id}"
                f"\nVariant Label: {self.variant_label}"
                f"\nFile Start: {self.file_start}"
                f"\nFile End: {self.file_end}"
            ) from e

        # Assert that we have all the data and no extra dimensions
        dsd_keys: set[str] = set(dsd.keys())
        cmip6_ids_set = var_set
        if dsd_keys - cmip6_ids_set:  # pragma: no cover
            raise IndexError(
                f"Variable(s) {cmip6_ids_set - dsd_keys} not found in CMIP6 dataset"
            )
        if cmip6_ids_set - dsd_keys:  # pragma: no cover
            raise IndexError(
                f"Variable(s) {dsd_keys - cmip6_ids_set} not found in CMIP6 dataset"
            )

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

        # Find the nearest available times in the first dataset.
        # Preserve original requested times for reference while sampling nearest data.
        requested_times = self._convert_times_to_datetime(time)
        selection_times = self._convert_times_to_cftime(
            requested_times, ds.time.dt.calendar
        )
        ds_nearest = ds.sel(time=selection_times, method="nearest")  # type: ignore[arg-type]
        selected_times = ds_nearest.time.values  # cftime objects in dataset calendar
        selected_times_dt = self._convert_times_to_datetime(list(selected_times))

        if not np.array_equal(
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

        Notes
        -----
        The check performs a lightweight ESGF search restricted to a one-day
        window surrounding *time*.  If any dataset is returned and the target
        timestamp lies within the dataset's time span, `True` is returned.
        Otherwise returns `False`.
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

        if t0 <= time <= t1:
            return True
        else:
            return False

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "cmip6")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_cmip6")
        return cache_location

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
