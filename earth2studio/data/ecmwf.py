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

from earth2studio.data.utils import datasource_cache_root, prep_forecast_inputs
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

        # Make sure input time is valid
        self._validate_time(time)
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

                    tasks.append(
                        ECMWFOpenDataAsyncTask(
                            data_array_indices=(i, j, k),
                            time=t,
                            lead_time=lt,
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
        grib_file = await self._download_ifs_grib_cached(
            time=task.time,
            lead_time=task.lead_time,
            variable=task.variable,
            levtype=task.levtype,
            level=task.level,
        )
        # Open into xarray data-array
        # Provided [-180, 180], roll to [0, 360]
        da = xr.open_dataarray(
            grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
        ).roll(longitude=-len(self.LON) // 2, roll_coords=True)
        if self._members is not None:
            da = da.sel(number=self._members)  # reorder
        xr_array[task.data_array_indices] = task.modifier(da.values)

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
                "type": self._fc_type,  # "fc", "cf", "pf"
                "param": variable,
                # "levtype": levtype, # NOTE: Commenting this out fixes what seems to be a bug with Opendata API on soil levels
                "step": int(lead_time.total_seconds() // 3600),
                "target": cache_path,
            }
            if levtype == "pl" or levtype == "sl":  # Pressure levels or soil levels
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
