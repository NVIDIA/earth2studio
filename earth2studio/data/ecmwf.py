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

import asyncio
import functools
import hashlib
import os
import pathlib
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import xarray as xr
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    prep_forecast_inputs,
)
from earth2studio.lexicon import AIFSLexicon, IFSLexicon
from earth2studio.lexicon.ecmwf import ECMWFOpenDataLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

try:
    import ecmwf.opendata as opendata
except ImportError:
    OptionalDependencyFailure("data")
    opendata = None

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@dataclass
class ECMWFOpenDataAsyncTask:
    """Small helper struct for Async tasks"""

    data_array_indices: tuple[int, int, int]
    time: datetime
    lead_time: timedelta
    variable: str
    levtype: str
    level: str | list[str]
    modifier: Callable


@check_optional_dependencies()
class ECMWFOpenDataSource(ABC):
    """Data provided through the ECMWF open data client, which includes the integrated
    forecast system (IFS) and artificial intelligence forecast system (AIFS) on an
    equirectangular grid at 0.25 degree resolution. Data for the most recent 4 days can
    be retrieved from ECMWF's servers (source `ecmwf`). Historical data is part of
    ECMWF's open data project on AWS (source `aws`).

    Parameters
    ----------
    source : str, optional
        Data source to fetch data from. For possible options refer to ECMWF's open data
        Python SDK, by default "aws".
    model: str, optional
        Model to fetch data for, by default "ifs".
    fc_type: str, optional
        Forecast type (e.g., deterministic, control, perturbed). For possible options
        refer to ECMWF's open data Python SDK, by default "fc".
    members: list of int, optional
        List of ensemble members. Set to None, empty list, or [0] for deterministic
        forecasts. By default "aws".
    cache : bool, optional
        Cache data source in local memory, by default True.
    verbose : bool, optional
        Print download progress, by default True.
    async_timeout: int, optional
        Time in seconds after which the download will be cancelled if not finished
        successfully, by default 600.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://github.com/ecmwf/ecmwf-opendata
    - https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts
    - https://registry.opendata.aws/ecmwf-forecasts/
    - https://console.cloud.google.com/storage/browser/ecmwf-open-data/
    """

    LAT = np.linspace(90, -90, 721)
    LON = np.linspace(0, 359.75, 1440)
    LEXICON: type[ECMWFOpenDataLexicon]

    def __init__(
        self,
        source: Literal["aws", "ecmwf", "azure"] = "aws",
        model: Literal["ifs", "aifs-single", "aifs-ens"] = "ifs",
        fc_type: Literal["fc", "cf", "pf"] = "fc",
        members: list[int] | None = None,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        # Optional import not installed error
        if opendata is None:
            raise ImportError(
                "ecmwf-opendata is not installed, install manually or using `pip install earth2studio[data]`"
            )

        if fc_type in ["fc", "cf"] and members is not None:
            raise ValueError(
                "Cannot provide ensemble members for forecast types fc, cf"
            )

        self.client = opendata.Client(source=source, model=model)
        self._fc_type = fc_type
        self._members = members

        self._cache = cache
        self._verbose = verbose
        self.async_timeout = async_timeout

        # Model name for caching and logging
        if model == "ifs":
            if fc_type == "fc":
                self._model = "IFS"
            else:
                self._model = "IFS-ENS"
        elif model == "aifs-single":
            self._model = "AIFS"
        else:
            self._model = "AIFS-ENS"

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve ECMWF data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the data lexicon.

        Returns
        -------
        xr.DataArray
            ECMWF weather data array
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        xr_array = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, lead_time, variable), timeout=self.async_timeout
            )
        )

        return xr_array

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        lead_time: timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to fetch.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the data lexicon.

        Returns
        -------
        xr.DataArray
            ECMWF weather data array.
        """
        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid. Negative lead times imply data needs to be
        # fetched from an earlier initialization cycle, so include those shifted
        # timestamps in the validation set as well.
        times_to_validate = set(time)
        for base_time in time:
            for delta in lead_time:
                if delta.total_seconds() < 0:
                    times_to_validate.add(base_time + delta)
        self._validate_time(sorted(times_to_validate))
        self._validate_leadtime(time, lead_time)

        # Pre-allocate full array (could be made more efficient)
        if self._members is None:
            xr_array = xr.DataArray(
                data=np.zeros(
                    (
                        len(time),
                        len(lead_time),
                        len(variable),
                        len(self.LAT),
                        len(self.LON),
                    )
                ),
                dims=["time", "lead_time", "variable", "lat", "lon"],
                coords={
                    "time": time,
                    "lead_time": lead_time,
                    "variable": variable,
                    "lat": self.LAT,
                    "lon": self.LON,
                },
            )
        else:
            xr_array = xr.DataArray(
                data=np.zeros(
                    (
                        len(time),
                        len(lead_time),
                        len(variable),
                        len(self._members),
                        len(self.LAT),
                        len(self.LON),
                    )
                ),
                dims=["time", "lead_time", "variable", "sample", "lat", "lon"],
                coords={
                    "time": time,
                    "lead_time": lead_time,
                    "variable": variable,
                    "sample": np.array(self._members),
                    "lat": self.LAT,
                    "lon": self.LON,
                },
            )

        async_tasks = await self._create_tasks(time, lead_time, variable)
        func_map = map(
            functools.partial(self.fetch_wrapper, xr_array=xr_array), async_tasks
        )

        await tqdm.gather(
            *func_map,
            desc=f"Fetching {self._model} data",
            disable=(not self._verbose),
        )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr_array

    async def _create_tasks(
        self,
        time: list[datetime],
        lead_time: list[timedelta],
        variable: list[str],
    ) -> list[ECMWFOpenDataAsyncTask]:
        """Create download tasks.

        Parameters
        ----------
        time : list[datetime]
            Timestamps to be downloaded (UTC).
        lead_time : list[datetime]
            Lead times to be downloaded.
        variable : list[str]
            List of variables to be downloaded.

        Returns
        -------
        list[ECMWFOpenDataAsyncTask]
            List of download tasks.
        """
        tasks: list[ECMWFOpenDataAsyncTask] = []

        for i, t in enumerate(time):
            for j, lt in enumerate(lead_time):
                for k, var in enumerate(variable):
                    try:
                        ifs_name, modifier = self.LEXICON[var]  # type: ignore[index]
                    except KeyError as e:
                        logger.error(f"Variable {var} not found in lexicon")
                        raise e

                    ifs_var, levtype, level = ifs_name.split("::")

                    req_time = t
                    req_lead = lt
                    if lt.total_seconds() < 0:
                        req_time = t + lt
                        req_lead = timedelta(0)

                    tasks.append(
                        ECMWFOpenDataAsyncTask(
                            data_array_indices=(i, j, k),
                            time=req_time,
                            lead_time=req_lead,
                            variable=ifs_var,
                            levtype=levtype,
                            level=level,
                            modifier=modifier,
                        )
                    )
        return tasks

    async def fetch_wrapper(
        self,
        task: ECMWFOpenDataAsyncTask,
        xr_array: xr.DataArray,
    ) -> None:
        """Small wrapper to pack arrays into the DataArray."""
        try:
            grib_file = await self._download_ifs_grib_cached(
                time=task.time,
                lead_time=task.lead_time,
                variable=task.variable,
                levtype=task.levtype,
                level=task.level,
            )

            # Open into xarray data-array
            # Provided [-180, 180], roll to [0, 360]
            type_map = {
                "pl": "isobaricInhPa",
                "sfc": "surface",
                "sl": "soilLayers",
                "ml": "generalVertical",
            }
            filter_keys = {"typeOfLevel": type_map.get(task.levtype, task.levtype)}
            # Pressure and soil levels require specifying the exact level requested
            if task.levtype in {"pl", "sl"}:
                level = task.level
                if isinstance(level, list):
                    # CFGRIB expects a single integer level per request
                    level = level[0]
                filter_keys["level"] = int(level)

            backend_kwargs = {"indexpath": "", "filter_by_keys": filter_keys}
            try:
                da = xr.open_dataarray(
                    grib_file,
                    engine="cfgrib",
                    backend_kwargs=backend_kwargs,
                )
            except ValueError as exc:
                if "contains no data variables" in str(exc):
                    # Retry without filtering for legacy files or unexpected level metadata
                    da = xr.open_dataarray(
                        grib_file,
                        engine="cfgrib",
                        backend_kwargs={"indexpath": ""},
                    )
                else:
                    raise

            da = da.roll(longitude=-len(self.LON) // 2, roll_coords=True)
            if self._members is not None:
                da = da.sel(number=self._members)  # reorder
            xr_array[task.data_array_indices] = task.modifier(da.values)
        except Exception as exc:
            logger.warning(
                f"Missing IFS field {task.variable}/{task.levtype}/{task.level} at "
                f"time={task.time}, lead={task.lead_time}: {exc}. Filling zeros."
            )
            xr_array[task.data_array_indices] = np.zeros(
                (len(self.LAT), len(self.LON)),
                dtype=xr_array.dtype,
            )

    async def _download_ifs_grib_cached(
        self,
        time: datetime,
        lead_time: timedelta,
        variable: str,
        levtype: str,
        level: str | list[str],
    ) -> str:
        """Download GRIB2 file to (temporary) cache."""
        if isinstance(level, str):
            level = [level]

        step_hours = int(lead_time.total_seconds() // 3600)

        hash_parts = [self._fc_type, time, lead_time, variable, levtype, *level]
        if self._members is not None:
            hash_parts.extend(self._members)  # type: ignore

        filename = hashlib.sha256(
            "_".join(str(x) for x in hash_parts).encode()
        ).hexdigest()
        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            request: dict[str, Any] = {
                "date": time,
                "type": self._fc_type,
                "param": variable,
                "levtype": levtype,
                "step": step_hours,
                "target": cache_path,
            }
            if levtype == "pl" or levtype == "sl":
                request["levelist"] = level
            if self._members is not None:
                request["number"] = self._members
            # Download
            await asyncio.to_thread(self.client.retrieve, **request)

        return cache_path

    @abstractmethod
    def _validate_time(self, times: list[datetime]) -> None:
        """Verify all times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        """
        ...

    @abstractmethod
    def _validate_leadtime(
        self, times: list[datetime], lead_times: list[timedelta]
    ) -> None:
        """Verify all lead times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        lead_times : list[timedelta]
            List of lead times to fetch data for.
        """
        ...

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_dir = self._model.lower()  # note that model is not part of cache hash
        cache_location = os.path.join(datasource_cache_root(), cache_dir)
        if not self._cache:
            cache_location = os.path.join(cache_location, f"tmp_{cache_dir}")
        return cache_location


class IFS(ECMWFOpenDataSource):
    """Integrated forecast system (IFS) HRES data on an equirectangular grid at 0.25
    degree resolution."""

    LEXICON = IFSLexicon

    def __init__(
        self,
        source: Literal["aws", "ecmwf", "azure"] = "aws",
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        super().__init__(
            source=source,
            model="ifs",
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify all times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        """
        validate_time(
            self._model, self.client.source, times, min_time=datetime(2024, 3, 1)
        )

    def _validate_leadtime(
        self,
        times: list[datetime],
        lead_times: list[timedelta],
    ) -> None:
        """Verify all lead times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        lead_times : list[timedelta]
            List of lead times to fetch data for.
        """
        validate_leadtime(self._model, lead_times, interval=3, max_lead_time=360)

        for delta in lead_times:
            hours = int(delta.total_seconds() // 3600)
            if any([time.hour in [6, 18] for time in times]) and hours > 144:
                # Shorter rollouts for forecasts starting at 06Z, 18Z
                raise ValueError(
                    f"Requested lead time {delta} can not be more than 144 hours for {self._model} starting at 06Z, 18Z"
                )
            if hours > 144 and not hours % 6 == 0:
                raise ValueError(
                    f"Requested lead time {delta} needs to be 6 hour interval for {self._model} after hour 144"
                )
            if hours < 0:
                continue
            if hours < 0:
                continue


class IFS_ENS(ECMWFOpenDataSource):
    """Integrated forecast system (IFS) ENS data on an equirectangular grid at 0.25
    degree resolution.

    The control member can only be requested separately from the perturbed members."""

    LEXICON = IFSLexicon

    def __init__(
        self,
        source: Literal["aws", "ecmwf", "azure"] = "aws",
        members: list[int] | None = None,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        fc_type: Literal["cf", "pf"]
        if members is None or len(members) == 0 or members == [0]:
            fc_type = "cf"  # control forecast
            members = None
        else:
            if 0 in members:
                raise ValueError(
                    "Please request control member (id 0) and perturbed members (id >0) separately"
                )
            fc_type = "pf"  # perturbed forecast

        super().__init__(
            source=source,
            model="ifs",
            fc_type=fc_type,
            members=members,
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify all times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        """
        validate_time(
            self._model, self.client.source, times, min_time=datetime(2024, 3, 1)
        )

    def _validate_leadtime(
        self,
        times: list[datetime],
        lead_times: list[timedelta],
    ) -> None:
        """Verify all lead times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        lead_times : list[timedelta]
            List of lead times to fetch data for.
        """
        validate_leadtime(self._model, lead_times, interval=3, max_lead_time=360)

        for delta in lead_times:
            hours = int(delta.total_seconds() // 3600)
            if any([time.hour in [6, 18] for time in times]) and hours > 144:
                # Shorter rollouts for forecasts starting at 06Z, 18Z
                raise ValueError(
                    f"Requested lead time {delta} can not be more than 144 hours for {self._model} starting at 06Z, 18Z"
                )
            if hours > 144 and not hours % 6 == 0:
                raise ValueError(
                    f"Requested lead time {delta} needs to be 6 hour interval for {self._model} after hour 144"
                )


@check_optional_dependencies()
import hashlib
import os
import pathlib
import shutil
from datetime import datetime

import numpy as np
import xarray as xr
from loguru import logger
from s3fs.core import S3FileSystem
from tqdm import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import IFSLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import TimeArray, VariableArray

try:
    import ecmwf.opendata as opendata
except ImportError:
    OptionalDependencyFailure("data")
    opendata = None

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@check_optional_dependencies()
class IFSOLD:
    """The integrated forecast system (IFS) initial state data source provided on an
    equirectangular grid. This data is part of ECMWF's open data project on AWS. This
    data source is provided on a 0.25 degree lat lon grid at 6-hour intervals for the
    most recent 4 days.
    Parameters
    ----------
    source : str, optional
        Data source to fetch data from. For possible options refer to ECMWF's open data
        Python SDK, by default "aws".
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
    This data source only fetches the initial state of control forecast of IFS and does
    not fetch an predicted time steps.
    Note
    ----
    Additional information on the data repository can be referenced here:
    - https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts
    - https://registry.opendata.aws/ecmwf-forecasts/
    - https://console.cloud.google.com/storage/browser/ecmwf-open-data/
    """

    IFS_BUCKET_NAME = "ecmwf-forecasts"
    IFS_LAT = np.linspace(90, -90, 721)
    IFS_LON = np.linspace(0, 359.75, 1440)

    def __init__(self, source: str = "aws", cache: bool = True, verbose: bool = True):
        # Optional import not installed error
        if opendata is None:
            raise ImportError(
                "ecmwf-opendata is not installed, install manually or using `pip install earth2studio[data]`"
            )

        self._cache = cache
        self._verbose = verbose
        self.client = opendata.Client(source=source)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.
        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in IFS lexicon.
        Returns
        -------
        xr.DataArray
            IFS weather data array
        """
        time, variable = prep_data_inputs(time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Fetch index file for requested time
        data_arrays = []
        for t0 in time:
            data_array = self.fetch_ifs_dataarray(t0, variable)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_ifs_dataarray(
        self,
        time: datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives IFS data array for given date time by fetching variable grib files
        using the ecmwf opendata package and combining grib files into a data array.
        Parameters
        ----------
        time : datetime
            Date time to fetch
        variables : list[str]
            list of atmosphric variables to fetch. Must be supported in IFS lexicon
        Returns
        -------
        xr.DataArray
            IFS data array for given date time
        """
        ifsda = xr.DataArray(
            data=np.empty((1, len(variables), len(self.IFS_LAT), len(self.IFS_LON))),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [time],
                "variable": variables,
                "lat": self.IFS_LAT,
                "lon": self.IFS_LON,
            },
        )

        # TODO: Add MP here, can further optimize by combining pressure levels
        # Not doing until tested.
        for i, variable in enumerate(
            tqdm(
                variables, desc=f"Fetching IFS for {time}", disable=(not self._verbose)
            )
        ):
            # Convert from Earth2Studio variable ID to GFS id and modifier
            try:
                ifs_name, modifier = IFSLexicon[variable]
            except KeyError as e:
                logger.error(f"Variable id {variable} not found in IFS lexicon")
                raise e

            variable, levtype, level = ifs_name.split("::")

            logger.debug(f"Fetching IFS grib file for variable: {variable} at {time}")
            grib_file = self._download_ifs_grib_cached(variable, levtype, level, time)
            # Open into xarray data-array
            # Provided [-180, 180], roll to [0, 360]
            da = xr.open_dataarray(
                grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
            ).roll(longitude=-len(self.IFS_LON) // 2, roll_coords=True)
            ifsda[0, i] = modifier(da.values)

        return ifsda

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for IFS
        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 6 hour interval for IFS"
                )

            if (datetime.now() - time).days > 4:
                raise ValueError(
                    f"Requested date time {time} needs to be within the past 4 days for IFS"
                )

            # if not self.available(time):
            #     raise ValueError(f"Requested date time {time} not available in IFS")

    def _download_ifs_grib_cached(
        self,
        variable: str,
        levtype: str,
        level: str | list[str],
        time: datetime,
    ) -> str:
        if isinstance(level, str):
            level = [level]

        sha = hashlib.sha256(f"{variable}_{levtype}_{'_'.join(level)}_{time}".encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            request = {
                "date": time,
                "type": "fc",
                "param": variable,
                # "levtype": levtype, # NOTE: Commenting this out fixes what seems to be a bug with Opendata API on soil levels
                "step": 0,  # Would change this for forecasts
                "target": cache_path,
            }
            if levtype == "pl" or levtype == "sl":  # Pressure levels or soil levels
                request["levelist"] = level
            # Download
            self.client.retrieve(**request)

        return cache_path

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "ifs_old")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_ifs_old")
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the IFS AWS data store
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

        fs = S3FileSystem(anon=True)

        file_name = f"{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}z/"
        s3_uri = f"s3://{cls.IFS_BUCKET_NAME}/{file_name}"
        exists = fs.exists(s3_uri)

        return exists

class AIFS(ECMWFOpenDataSource):
    """Artificial intelligence forecast system (AIFS) SINGLE data on an equirectangular
    grid at 0.25 degree resolution."""

    LEXICON = AIFSLexicon

    def __init__(
        self,
        source: Literal["aws", "ecmwf", "azure"] = "aws",
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        super().__init__(
            source=source,
            model="aifs-single",
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify all times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        """
        validate_time(
            self._model, self.client.source, times, min_time=datetime(2025, 7, 1, 6)
        )

    def _validate_leadtime(
        self, times: list[datetime], lead_times: list[timedelta]
    ) -> None:
        """Verify all lead times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        lead_times : list[timedelta]
            List of lead times to fetch data for.
        """
        validate_leadtime(self._model, lead_times, interval=6, max_lead_time=360)


class AIFS_ENS(ECMWFOpenDataSource):
    """Artificial intelligence forecast system (AIFS) ENS data on an equirectangular
    grid at 0.25 degree resolution.

    The control member can only be requested separately from the perturbed members."""

    LEXICON = AIFSLexicon

    def __init__(
        self,
        source: Literal["aws", "ecmwf", "azure"] = "aws",
        members: list[int] | None = None,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):
        fc_type: Literal["cf", "pf"]
        if members is None or len(members) == 0 or members == [0]:
            fc_type = "cf"  # control forecast
            members = None
        else:
            if 0 in members:
                raise ValueError(
                    "Please request control member (id 0) and perturbed members (id >0) separately"
                )
            fc_type = "pf"  # perturbed forecast

        super().__init__(
            source=source,
            model="aifs-ens",
            fc_type=fc_type,
            members=members,
            cache=cache,
            verbose=verbose,
            async_timeout=async_timeout,
        )

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify all times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        """
        validate_time(
            self._model, self.client.source, times, min_time=datetime(2025, 7, 1, 6)
        )

    def _validate_leadtime(
        self, times: list[datetime], lead_times: list[timedelta]
    ) -> None:
        """Verify all lead times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        lead_times : list[timedelta]
            List of lead times to fetch data for.
        """
        validate_leadtime(self._model, lead_times, interval=6, max_lead_time=360)


def validate_time(
    model: str,
    source: str,
    times: list[datetime],
    min_time: datetime,
) -> None:
    """Verify all times are valid based on offline knowledge.

    Parameters
    ----------
    model : str
        Model name.
    source : str
        ECMWF client source.
    times : list[datetime]
        List of date times to fetch data for.
    min_time : datetime
        Earliest available datetime.
    """
    for time in times:
        if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
            raise ValueError(
                f"Requested start time {time} needs to be 6-hour interval for {model}"
            )

        if time < min_time:
            raise ValueError(
                f"Requested start time {time} needs to be at least {min_time} for {model}"
            )

        if source == "ecmwf":
            if (datetime.now() - time).days > 4:
                raise ValueError(
                    f"Requested start time {time} needs to be within the past 4 days for {model} with source ECMWF"
                )


def validate_leadtime(
    model: str,
    lead_times: list[timedelta],
    interval: int,
    max_lead_time: int,
) -> None:
    """Verify all lead times are valid based on offline knowledge.

    Parameters
    ----------
    model : str
        Model name.
    lead_times : list[timedelta]
        List of lead times to fetch data for.
    interval : int
        Required lead time interval in hours.
    max_lead_time : timedelta
        Maximum available lead time in hours.
    """

    for delta in lead_times:
        # See https://github.com/ecmwf/ecmwf-opendata?tab=readme-ov-file#time-steps
        # But there seem to be some inaccuracies, e.g., HRES is available up to hour 360
        # See S3: aws s3 ls --no-sign-request s3://ecmwf-forecasts/20251016/06z/aifs-single/0p25/oper/
        hours = int(delta.total_seconds() // 3600)

        if not delta.total_seconds() % 3600 == 0 or not hours % interval == 0:
            raise ValueError(
                f"Requested lead time {delta} needs to be {interval}-hour interval for {model}"
            )

        if hours > max_lead_time:
            raise ValueError(
                f"Requested lead time {delta} cannot be more than {max_lead_time} hours for {model}"
            )
