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

import hashlib
import multiprocessing as mp
import os
import pathlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from time import sleep
from typing import Callable

import cdsapi
import numpy as np
import xarray as xr
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2studio.data.utils import prep_data_inputs
from earth2studio.lexicon import CDSLexicon
from earth2studio.utils.type import TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

LOCAL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "earth2studio")


@dataclass
class CDSRequest:
    """CDS Request data class"""

    dataset: str
    time: datetime
    variable: str
    levels: list[str]
    modifiers: list[Callable]
    indices: list[int]
    ids: list[str]


class CDS:
    """The climate data source (CDS) serving ERA5 re-analysis data. This data soure
    requires users to have a CDS API access key which can be obtained for free on the
    CDS webpage.

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

    - https://cds.climate.copernicus.eu/cdsapp#!/home
    """

    MAX_BYTE_SIZE = 20000000

    CDS_LAT = np.linspace(90, -90, 721)
    CDS_LON = np.linspace(0, 359.75, 1440)

    def __init__(self, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose
        self.cds_client = cdsapi.Client(
            debug=False, quiet=True, wait_until_complete=False
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
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in CDS lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from CDS
        """
        time, variable = prep_data_inputs(time, variable)

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Fetch index file for requested time
        data_arrays = []
        for t0 in time:
            data_array = self.fetch_cds_dataarray(t0, variable)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_cds_dataarray(
        self,
        time: datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives CDS data array for given date time by fetching variable grib files
        using the cdsapi package and combining grib files into a single data array.

        Parameters
        ----------
        time : datetime
            Date time to fetch
        variables : list[str]
            list of atmosphric variables to fetch. Must be supported in CDS lexicon

        Returns
        -------
        xr.DataArray
            CDS data array for given date time
        """
        # Build requests for this time
        if isinstance(variables, str):
            variables = [variables]
        requests = self._build_requests(time, variables)
        pbar = tqdm(
            total=len(requests),
            desc=f"Fetching CDS for {time}",
            disable=(not self._verbose),
        )

        # Fetch process for getting data off CDS
        def _fetch_process(request: CDSRequest, rank: int, return_dict: dict) -> None:
            logger.info(
                f"Fetching CDS grib file for variable: {request.variable} at {request.time} with {len(request.levels)} levels"
            )
            grib_file = self._download_cds_grib_cached(
                request.time, request.dataset, request.variable, request.levels
            )
            return_dict[i] = grib_file

        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        for i, request in enumerate(requests):
            process = mp.Process(target=_fetch_process, args=(request, i, return_dict))
            processes.append(process)
            process.start()

        # wait for all processes to complete
        for process in processes:
            process.join()
            pbar.update(1)

        da = xr.DataArray(
            data=np.empty((1, len(variables), len(self.CDS_LAT), len(self.CDS_LON))),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [time],
                "variable": variables,
                "lat": self.CDS_LAT,
                "lon": self.CDS_LON,
            },
        )

        for i, request in enumerate(requests):
            # Open into xarray data-array
            grib = xr.open_dataarray(
                return_dict[i], engine="cfgrib", backend_kwargs={"indexpath": ""}
            )
            for level, index, modifier in zip(
                request.levels, request.indices, request.modifiers
            ):
                if len(grib.values.shape) == 2:
                    da[0, index] = modifier(grib.values)
                else:
                    da[0, index] = modifier(grib.sel(isobaricInhPa=float(level)).values)

        return da

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for CDS

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for CDS"
                )

            if time < datetime(year=1940, month=1, day=1):
                raise ValueError(
                    f"Requested date time {time} needs to be after January 1st, 1940 for CDS"
                )

            # if not self.available(time):
            #     raise ValueError(f"Requested date time {time} not available in CDS")

    def _build_requests(self, time: datetime, variables: list[str]) -> list[CDSRequest]:
        """Builds list of CDS request objects. Compiles different pressure levels for
        a given variable into a single request"""
        requests: dict[str, CDSRequest] = {}
        for i, variable in enumerate(variables):
            # Convert from Nvidia variable ID to CDS id and modifier
            try:
                cds_name, modifier = CDSLexicon[variable]
            except KeyError as e:
                logger.error(f"variable id {variable} not found in CDS lexicon")
                raise e

            dataset_name, cds_variable, level = cds_name.split("::")
            request_id = f"{dataset_name}-{cds_variable}"
            if request_id in requests:
                requests[request_id].levels.append(level)
                requests[request_id].modifiers.append(modifier)
                requests[request_id].indices.append(i)
                requests[request_id].ids.append(variable)
            else:
                requests[request_id] = CDSRequest(
                    dataset=dataset_name,
                    time=time,
                    variable=cds_variable,
                    levels=[level],
                    modifiers=[modifier],
                    indices=[i],
                    ids=[variable],
                )
        return list(requests.values())

    def _download_cds_grib_cached(
        self,
        time: datetime,
        dataset_name: str,
        variable: str,
        level: str | list[str],
    ) -> str:
        """Downloads grib file from CDS with requested field"""
        if isinstance(level, str):
            level = [level]

        sha = hashlib.sha256(
            f"{dataset_name}_{variable}_{'_'.join(level)}_{time}".encode()
        )
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            # Assemble request
            rbody = {
                "variable": variable,
                "product_type": "reanalysis",
                # "date": "2017-12-01/2017-12-02", (could do time range)
                "year": time.year,
                "month": time.month,
                "day": time.day,
                "time": time.strftime("%H:00"),
                "format": "grib",
            }
            if dataset_name == "reanalysis-era5-pressure-levels":
                rbody["pressure_level"] = level
            r = self.cds_client.retrieve(dataset_name, rbody)
            # Queue request
            while True:
                r.update()
                reply = r.reply
                logger.debug(
                    f"Request ID:{reply['request_id']}, state: {reply['state']}"
                )
                if reply["state"] == "completed":
                    break
                elif reply["state"] in ("queued", "running"):
                    logger.debug(f"Request ID: {reply['request_id']}, sleeping")
                    sleep(1.0)
                elif reply["state"] in ("failed",):
                    logger.error(
                        f"CDS request fail for: {dataset_name} {variable} {level} {time}"
                    )
                    logger.error(f"Message: {reply['error'].get('message')}")
                    logger.error(f"Reason: {reply['error'].get('reason')}")
                    raise Exception("%s." % (reply["error"].get("message")))
            # Download when ready
            r.download(cache_path)

        return cache_path

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(LOCAL_CACHE, "cds")
        if not self._cache:
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime | np.datetime64,
    ) -> bool:
        """Checks if given date time is avaliable in the CDS with the pressure level
        database

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

        client = cdsapi.Client(debug=False, quiet=True, wait_until_complete=False)
        # Assemble request
        r = client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "variable": "2m_temperature",
                "product_type": "reanalysis",
                "year": time.year,
                "month": time.month,
                "day": time.day,
                "time": time.strftime("%H:00"),
                "format": "grib",
            },
        )
        # Queue request
        while True:
            r.update()
            reply = r.reply
            logger.debug(f"Request ID:{reply['request_id']}, state: {reply['state']}")
            if reply["state"] == "completed":
                break
            elif reply["state"] in ("queued", "running"):
                logger.debug(f"Request ID: {reply['request_id']}, sleeping")
                sleep(0.5)
            elif reply["state"] in ("failed",):
                logger.error(f"CDS request fail for {time}")
                logger.error(f"Message: {reply['error'].get('message')}")
                logger.error(f"Reason: {reply['error'].get('reason')}")
                return False

        return True
