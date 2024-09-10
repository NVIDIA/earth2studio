# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
import pathlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    prep_forecast_inputs,
)
from earth2studio.lexicon import HRRRLexicon
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class _HRRRBase:
    def __init__(self, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose

    def fetch_dataarray(
        self,
        time: list[datetime],
        lead_time: list[timedelta],
        variable: list[str],
    ) -> xr.DataArray:
        """Retrieve HRRR forecast data into a single Xarray data array

        Parameters
        ----------
        time : list[datetime]
            Timestamps to return data for (UTC).
        lead_time: list[timedelta]
            List of forecast lead times to fetch
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HRRR lexicon.

        Returns
        -------
        xr.DataArray
            HRRR weather data array
        """

        # Convert lead time timedeltas to indices
        lead_index = [int(delta_t.total_seconds() // 3600) for delta_t in lead_time]

        # Convert from Earth2Studio variable ID to HRRR id and modifier
        sfc_vars = {}
        prs_vars = {}
        for var in variable:
            try:
                hrrr_str, modifier = HRRRLexicon[var]
                hrrr_name = hrrr_str.split("::")
                if hrrr_name[0] == "sfc":
                    sfc_vars[var] = (f":{hrrr_name[1]}:{hrrr_name[2]}", modifier)
                else:
                    prs_vars[var] = (f":{hrrr_name[1]}:{hrrr_name[2]}", modifier)
            except KeyError:  # noqa: PERF203
                raise KeyError(f"variable id {var} not found in HRRR lexicon")

        # Import here to prevent prints
        from herbie import FastHerbie

        data_arrays = {}
        # Process surface and then pressure fields
        for product, var_dict in zip(["sfc", "prs"], [sfc_vars, prs_vars]):
            fh = FastHerbie(
                time,
                model="hrrr",
                product=product,
                fxx=lead_index,
                max_threads=8,
                save_dir=self.cache,
                verbose=False,
                priority=["aws", "google", "nomads"],
            )
            # TODO: MP
            for id, (hrrr_id, modifier) in tqdm(
                var_dict.items(),
                desc=f"Fetching HRRR {product} fields",
                disable=(not self._verbose),
            ):
                ds = fh.xarray(hrrr_id, verbose=False)
                if "gribfile_projection" in ds.data_vars:
                    ds = ds.drop("gribfile_projection")
                da = next(iter(ds.data_vars.values()))

                # Herbie squeezes dims in returned array, so expand if length is 1
                data = da.to_numpy()
                if len(lead_time) == 1:
                    data = data[None]
                if len(time) == 1:
                    data = data[:, None]
                # Add variable dimension
                data = data[:, :, None]
                # Transpose time and lead time
                data = np.transpose(data, (1, 0, 2, 3, 4))

                # Could initialize array ahead of time, need to improve
                data_arrays[id] = xr.DataArray(
                    data=data,
                    dims=["time", "lead_time", "variable", "hrrr_y", "hrrr_x"],
                    coords=dict(
                        hrrr_x=np.arange(da.coords["longitude"].shape[1]),
                        hrrr_y=np.arange(da.coords["longitude"].shape[0]),
                        lon=(["hrrr_y", "hrrr_x"], da.coords["longitude"].values),
                        lat=(["hrrr_y", "hrrr_x"], da.coords["latitude"].values),
                        time=time,
                        lead_time=lead_time,
                        variable=np.array([id]),
                    ),
                )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat([data_arrays[var] for var in variable], dim="variable")

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for HRRR

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for HRRR"
                )

            if time < datetime(year=2014, month=8, day=4, hour=1):
                raise ValueError(
                    f"Requested date time {time} needs to be after April 8th, 2014 1:00am for HRRR"
                )

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "hrrr")
        if not self._cache:
            if not DistributedManager.is_initialized():
                DistributedManager.initialize()
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the HRRR store

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to access

        Returns
        -------
        bool
            If date time is avaiable
        """
        if isinstance(time, np.datetime64):  # np.datetime64 -> datetime
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.utcfromtimestamp((time - _unix) / _ds)

        # Offline checks
        try:
            cls._validate_time([time])
        except ValueError:
            return False

        # Import here to prevent prints
        from herbie import FastHerbie

        if isinstance(time, np.datetime64):  # np.datetime64 -> datetime
            _unix = np.datetime64(0, "s")
            _ds = np.timedelta64(1, "s")
            time = datetime.utcfromtimestamp((time - _unix) / _ds)
        time, variable = prep_data_inputs(time, "t2m")

        fh = FastHerbie(time, model="hrrr", verbose=False)
        if len(fh.file_not_exists) > 0:
            return False
        else:
            return True


class HRRR(_HRRRBase):
    """High-Resolution Rapid Refresh (HRRR) data source provides hourly North-American
    weather analysis data developed by NOAA (used to initialize the HRRR forecast
    model). This data source is provided on a Lambert conformal 3km grid at 1-hour
    intervals. The spatial dimensionality of HRRR data is [1059, 1799].

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://rapidrefresh.noaa.gov/hrrr/
    - https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
    """

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve HRRR initial data to be used for initial conditions for the given
        time, variable information, and optional history.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HRRR lexicon.

        Returns
        -------
        xr.DataArray
            HRRR analysis data array
        """
        time, variable = prep_data_inputs(time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        data_array = self.fetch_dataarray(time, [timedelta(hours=0)], variable)
        return data_array.isel(lead_time=0).drop_vars("lead_time")


class HRRR_FX(_HRRRBase):
    """High-Resolution Rapid Refresh (HRRR) forecast source provides a North-American
    weather forecasts with hourly forecast runs developed by NOAA. This forecast source
    has hourly forecast steps up to a lead time of 48 hours. Data is provided on a
    Lambert conformal 3km grid at 1-hour intervals. The spatial dimensionality of HRRR
    data is [1059, 1799].

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.nco.ncep.noaa.gov/pmb/products/hrrr/
    - https://rapidrefresh.noaa.gov/hrrr/
    - https://console.cloud.google.com/marketplace/product/noaa-public/hrrr
    """

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve HRRR forecast data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the HRRR lexicon.

        Returns
        -------
        xr.DataArray
            HRRR forecast data array
        """
        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)
        self._validate_leadtime(lead_time)

        data_array = self.fetch_dataarray(time, lead_time, variable)
        return data_array

    @classmethod
    def _validate_leadtime(cls, lead_times: list[timedelta]) -> None:
        """Verify if lead time is valid for HRRR based on offline knowledge

        Parameters
        ----------
        lead_times : list[timedelta]
            list of lead times to fetch data
        """
        for delta in lead_times:
            if not delta.total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested lead time {delta} needs to be 1 hour interval for HRRR"
                )
            hours = int(delta.total_seconds() // 3600)
            if hours > 48 or hours < 0:
                raise ValueError(
                    f"Requested lead time {delta} can only be a max of 48 hours for HRRR"
                )
