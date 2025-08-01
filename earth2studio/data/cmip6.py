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

from datetime import datetime

import cftime  # Calendar-aware datetimes
import numpy as np
import xarray as xr

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import intake_esgf  # type: ignore
except ImportError:  # pragma: no cover – handled gracefully
    OptionalDependencyFailure("data")
    intake_esgf = None  # type: ignore[assignment]

from earth2studio.data.utils import prep_data_inputs
from earth2studio.lexicon.cmip6 import CMIP6Lexicon
from earth2studio.utils.type import TimeArray, VariableArray


@check_optional_dependencies()
class CMIP6:
    """CMIP6 data source for Earth2Studio.

    This class provides a thin convenience wrapper around the
    `intake-esgf` catalog that hosts the Coupled Model Inter-comparison
    Project Phase 6 (CMIP6) archive. This is meant to provide a seemless
    interface to the CMIP6 archive for Earth2Studio however the CMIP6
    archive is very large there may be data that will break this interface.

    Parameters
    ----------
    experiment_id : str
        CMIP6 experiment identifier (e.g. ``"historical"``, ``"ssp585"``).
    source_id : str
        CMIP6 model identifier (e.g. ``"MPI-ESM1-2-LR"``).
    table_id : str
        CMOR table describing variable realm/frequency (``"Amon"``,
        ``"Omon"``, ``"SImon"`` …).
    variant_label : str
        Ensemble member / initial-condition label such as
        ``"r1i1p1f1"``.
    file_start, file_end : str | None, optional
        Optional filename prefix/suffix filters forwarded to
        ``ESGFCatalog.search`` to constrain the final dataset
        selection.  Leave ``None`` to accept all.

    Notes
    -----
    * Requires the *optional* dependency ``intake-esgf`` which is part
      of the ``earth2studio[data]`` extras group.
    * The catalog search is executed lazily inside
      :pymeth:`__call__`; the constructor itself performs only basic
      validation.

    Raises
    ------
    ImportError
        If ``intake-esgf`` is not installed.
    IndexError
        If one or more requested variables are missing from the
        selected dataset.
    ValueError
        If the requested timestamps are not available or if the grid
        type cannot be inferred.

    Examples
    --------
    >>> from datetime import datetime
    >>> from earth2studio.data.cmip6 import CMIP6
    >>> ds = CMIP6("historical", "MPI-ESM1-2-LR", "Amon", "r1i1p1f1")
    >>> da = ds(time=[datetime(2010, 1, 15)], variable=["t2m", "pr"])
    >>> da
    <xarray.DataArray (time: 1, variable: 2, lat: 192, lon: 288)>\n    ...
    """

    def __init__(
        self,
        experiment_id: str,
        source_id: str,
        table_id: str,
        variant_label: str,
        file_start: str = None,
        file_end: str = None,
    ):
        self.experiment_id = experiment_id
        self.source_id = source_id
        self.table_id = table_id
        self.variant_label = variant_label
        self.file_start = file_start
        self.file_end = file_end

        # Optional import not installed error
        if intake_esgf is None:  # pragma: no cover
            raise ImportError(
                "intake-esgf is not installed, install manually or using `pip install earth2studio[data]`"
            )

        # Create catalog
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
        self.catalog.search(
            experiment_id=self.experiment_id,
            source_id=self.source_id,
            table_id=self.table_id,
            variable_id=cmip6_variable_id,
            variant_label=self.variant_label,
        )
        dsd = self.catalog.to_dataset_dict(
            prefer_streaming=False, add_measures=False
        )  # NOTE: it may be better to use streaming however this can result in errors

        # Assert that we have all the data and no extra dimensions
        dsd_keys: set[str] = set(dsd.keys())
        cmip6_ids_set = var_set
        if dsd_keys - cmip6_ids_set:
            raise IndexError(
                f"Variable(s) {cmip6_ids_set - dsd_keys} not found in CMIP6 dataset"
            )
        if cmip6_ids_set - dsd_keys:
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
        else:
            raise ValueError(
                "Unable to determine horizontal coordinates for CMIP6 dataset – expected 'lat/lon' or 'latitude/longitude'."
            )

        # time conversion done above
        time = self._convert_times(time, ds.time.dt.calendar)

        # Subset time; rely on xarray's KeyError if a timestamp is missing
        requested_set = set(time)
        for var_name, ds_var in dsd.items():
            available_set = set(ds_var.time.values.tolist())  # type: ignore[arg-type]
            if not requested_set.issubset(available_set):
                raise ValueError(
                    f"One or more requested timestamps {time} not found in CMIP6 dataset '{var_name}'. CMIP6 dataset: {ds_var}"
                )

            dsd[var_name] = ds_var.sel(time=time)  # type: ignore[arg-type]

        # Make data array
        da = xr.DataArray(
            data=np.empty((len(time), len(variable), *grid_shape), dtype=np.float32),
            dims=["time", "variable", *da_dims_xy],
            coords={
                "time": time,
                "variable": variable,
                **coord_dict,
            },
        )

        # Populate the array
        for var_idx, var in enumerate(variable):

            # Get variable, level, and modifier with explicit typing for mypy
            cmip6_entry = CMIP6Lexicon[var]  # type: ignore[misc]
            (cmip6_var, level), modifier = cmip6_entry
            cmip6_var = str(cmip6_var)  # type: ignore[assignment]
            level = int(level)  # type: ignore[assignment]

            # Get data array
            ds_var = dsd[cmip6_var]
            data_arr = ds_var[cmip6_var]  # DataArray inside the dataset

            # Select pressure level if needed
            if level != -1:
                # Convert hPa → Pa to match dataset units
                target_pa = level * 100.0
                if "plev" not in data_arr.coords:
                    raise ValueError(
                        f"Variable '{cmip6_var}' expected to have a 'plev' coordinate but none found"
                    )

                try:
                    data_arr = data_arr.sel(plev=target_pa)
                except KeyError as e:
                    available = data_arr.plev.values / 100.0
                    raise ValueError(
                        f"Requested pressure level {level} hPa for variable '{cmip6_var}' not found."
                        f" Available levels: {available}"
                    ) from e

            # At this point data_arr dims should be (time, lat, lon)
            # Convert to numpy (trigger load) and apply modifier
            da.values[:, var_idx, :, :] = modifier(data_arr.values)

        return da

    @staticmethod
    def _convert_times(raw_times: list, calendar: str) -> list[object]:
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
            if ts.__class__.__module__.startswith("cftime"):
                converted.append(ts)
                continue

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
