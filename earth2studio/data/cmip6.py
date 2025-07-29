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
from datetime import datetime
from typing import Any

import xarray as xr
import numpy as np
from pandas import to_datetime
import intake_esgf

from earth2studio.data.utils import prep_data_inputs
from earth2studio.lexicon.cmip6 import CMIP6Lexicon
from earth2studio.utils.type import TimeArray, VariableArray


class CMIP6:
    """A CMIP6 data source.

    Parameters
    ----------
    experiment_id : str
        The experiment id.
    source_id : str
        The source id.
    table_id : str
        The table id.
    variant_label : str
        The variant label.
    file_start: str | None
        The start of the file name.
    file_end: str | None
        The end of the file name.
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
        cmip6_variable_id = set()
        for v in variable:
            cmip6_variable_id.add(CMIP6Lexicon[v][0][0])
        cmip6_variable_id = list(cmip6_variable_id)

        # Search for data
        self.catalog.search(
            experiment_id=self.experiment_id,
            source_id=self.source_id,
            table_id=self.table_id,
            variable_id=cmip6_variable_id,
            variant_label=self.variant_label,
        )
        dsd = self.catalog.to_dataset_dict(prefer_streaming=True, add_measures=False)

        # Assert that we have all the data and no extra dimensions
        dsd_keys = set(list(dsd.keys()))
        cmip6_variable_id = set(cmip6_variable_id)
        if dsd_keys - cmip6_variable_id:
            raise IndexError(
                f"Variable(s) {dsd_keys - cmip6_variable_id} not found in CMIP6 dataset"
            )
        if cmip6_variable_id - dsd_keys:
            raise IndexError(
                f"Variable(s) {cmip6_variable_id - dsd_keys} not found in CMIP6 dataset"
            )

        # Get lat/lon and calendar from first dataset
        ds = dsd[list(dsd.keys())[0]]
        lat = ds.lat.values
        lon = ds.lon.values

        # time conversion done above
        time = self._convert_times(time, ds.time.dt.calendar)

        # Subset time; rely on xarray's KeyError if a timestamp is missing
        for k, v in dsd.items():
            try:
                dsd[k] = v.sel(time=time)  # type: ignore[arg-type]
            except KeyError as e:
                raise ValueError(
                    f"One or more requested timestamps {time} not found in CMIP6 dataset '{k}'."
                ) from e

        # Make data array
        da = xr.DataArray(
            data=np.empty((len(time), len(variable), len(lat), len(lon)), dtype=np.float32),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": lat,
                "lon": lon,
            },
        )

        # Populate the array
        for var_idx, var in enumerate(variable):

            # Get variable, level, and modifier
            (cmip6_var, level), modifier = CMIP6Lexicon[var]

            # Get data array
            ds_var = dsd[cmip6_var]
            data_arr = ds_var[cmip6_var]  # DataArray inside the dataset

            # Select pressure level if needed
            if level != -1:
                # Convert hPa → Pa to match dataset units
                target_pa = level * 100.0
                if "plev" not in data_arr.coords:
                    raise ValueError(f"Variable '{cmip6_var}' expected to have a 'plev' coordinate but none found")

                try:
                    data_arr = data_arr.sel(plev=target_pa)
                except KeyError as e:
                    available = data_arr.plev.values / 100.0
                    raise ValueError(
                        f"Requested pressure level {level} hPa for variable '{cmip6_var}' not found."
                        f" Available levels: {available}" ) from e

            # At this point data_arr dims should be (time, lat, lon)
            # Convert to numpy (trigger load) and apply modifier
            da.values[:, var_idx, :, :] = modifier(data_arr.values)

        return da

    # ------------------------------------------------------------------
    # Helper utilities ---------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_times(raw_times: list, calendar: str):
        """Convert python/NumPy datetimes to matching ``cftime`` objects.

        Parameters
        ----------
        raw_times : list
            List of datetime-like objects (``datetime``, ``numpy.datetime64`` or
            already-converted ``cftime`` objects).
        calendar : str
            CF calendar name present in the dataset (e.g. ``noleap``,
            ``360_day``).

        Returns
        -------
        list
            List of times converted to the appropriate ``cftime`` class.
        """

        import cftime  # local import to avoid global hard dependency

        converted = []
        for ts in raw_times:
            # Already a cftime object – accept as-is
            if ts.__class__.__module__.startswith("cftime"):
                converted.append(ts)
                continue

            # Normalize numpy datetime64 to python datetime
            if isinstance(ts, np.datetime64):
                ts = ts.astype("datetime64[us]").astype(datetime)

            if isinstance(ts, datetime):
                y, m, d, H, Mi, S = ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second

                if calendar in ("noleap", "365_day"):
                    converted.append(cftime.DatetimeNoLeap(y, m, d, H, Mi, S))
                elif calendar in ("360_day",):
                    converted.append(cftime.Datetime360Day(y, m, d, H, Mi, S))
                else:
                    converted.append(cftime.DatetimeGregorian(y, m, d, H, Mi, S))
            else:
                raise TypeError(f"Unsupported time type: {type(ts)}")

        return converted

