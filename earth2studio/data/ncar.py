# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import calendar
import hashlib
import multiprocessing
import os
import shutil
from datetime import date, datetime
from functools import partial

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from loguru import logger
from tqdm import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
)
from earth2studio.lexicon import NCAR_ERA5Lexicon
from earth2studio.utils.type import TimeArray, VariableArray

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class NCAR_ERA5:
    """ERA5 data provided by NSF NCAR via the AWS Open Data Sponsorship Program. ERA5
    is the fifth generation of the ECMWF global reanalysis and available on a 0.25
    degree WGS84 grid at hourly intervals spanning from 1940 to the present.

    Parameters
    ----------
    n_workers : int, optional
        Number of parallel workers, by default 8
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
    Additional resources:
    https://registry.opendata.aws/nsf-ncar-era5/
    """

    def __init__(self, n_workers: int = 8, cache: bool = True, verbose: bool = False):
        self._cache = cache
        self._verbose = verbose
        self._n_workers = n_workers

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve ERA5 data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the NCAR_ERA5 lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array
        """
        time, variable = prep_data_inputs(time, variable)
        self._validate_time(time)

        data_arrays: dict[str, xr.DataArray] = {}
        tasks = self._create_tasks(time, variable)
        logger.debug("Download tasks: {}", str(tasks))

        ctx = multiprocessing.get_context("spawn")  # s3fs requires spawn or forkserver
        fn = partial(self._fetch_dataarray, cache_path=self.cache, cache=self._cache)
        with ctx.Pool(self._n_workers) as p:
            for ename, arr in tqdm(
                p.imap_unordered(fn, tasks),
                "Step",
                len(tasks),
                disable=(not self._verbose),
            ):
                data_arrays.setdefault(ename, []).append(arr)

        # Concat time and variable dims
        array_list = [xr.concat(arrs, dim="time") for arrs in data_arrays.values()]
        res = xr.concat(array_list, dim="variable", combine_attrs="drop")
        res.name = None  # remove name, which is kept from one of the arrays
        res = res.transpose("time", "variable", ...)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return res.sel(time=time, variable=variable)  # reorder to match inputs

    @staticmethod
    def _create_tasks(times: list[datetime], variables: list[str]) -> list[dict]:
        """Create download tasks, each corresponding to one file on S3.

        Parameters
        ----------
        times : list[datetime]
            Timestamps to be downloaded (UTC).
        variables : list[str]
            List of variables to be downloaded.

        Returns
        -------
        list[dict]
            List of download tasks.
        """
        groups: dict[str, dict] = {}  # group pressure-level variables
        for var in set(variables):
            spec, _ = NCAR_ERA5Lexicon[var]
            lvl = spec.split(".")[3]  # surface/pressure-level
            ename = spec.split(".")[4].split("_")[-1]  # ECMWF name
            groups.setdefault(
                ename,
                {"var": var if lvl == "sfc" else var[0], "levels": [], "spec": spec},
            )
            if lvl == "pl":
                # Collect pressure levels
                groups[ename]["levels"].append(int(var[1:]))

        pattern = "s3://nsf-ncar-era5/e5.oper.an.{lvl}/{y}{m:02}/{spec}.{y}{m:02}{d:02}00_{y}{m:02}{dend:02}23.nc"

        tasks = []  # group tasks by S3 object
        times_by_day: dict[
            date, list[datetime]
        ] = {}  # pressure-level variables are in daily files
        times_by_month: dict[
            date, list[datetime]
        ] = {}  # surface-level variables are in monthly files
        for dt in times:
            times_by_day.setdefault(dt.date(), []).append(dt)
            times_by_month.setdefault(dt.date().replace(day=1), []).append(dt)
        for ename, group in groups.items():
            if len(group["levels"]) == 0:
                # Surface-level variable, monthly files
                for month, dts in times_by_month.items():
                    tasks.append(
                        {
                            "ename": ename,
                            "var": group["var"],  # Earth-2 variable specifier
                            "levels": [],
                            "dts": dts,
                            "s3pfx": pattern.format(
                                lvl="sfc",
                                y=month.year,
                                m=month.month,
                                spec=group["spec"],
                                d=1,
                                dend=calendar.monthrange(month.year, month.month)[-1],
                            ),
                        }
                    )
            else:
                # Pressure-level variable, daily files
                for day, dts in times_by_day.items():
                    tasks.append(
                        {
                            "ename": ename,
                            "var": group["var"],  # Earth-2 variable specifier
                            "levels": group["levels"],
                            "dts": dts,
                            "s3pfx": pattern.format(
                                lvl="pl",
                                y=day.year,
                                m=day.month,
                                spec=group["spec"],
                                d=day.day,
                                dend=day.day,
                            ),
                        }
                    )
        return tasks

    @staticmethod
    def _fetch_dataarray(task: dict, cache_path: str, cache: bool) -> xr.DataArray:
        """Retrieve ERA5 data for single group of times/variables.

        Parameters
        ----------
        task : dict
            Download task, specifying the variables, times, and S3 location.
        cache_path: str
            Locally cache directory
        cache: bool
            Cache data source on local memory

        Returns
        -------
        xr.DataArray
            ERA5 data for the given group of times/variables.
        """
        ename, var, levels, dts, s3pfx = (
            task["ename"],
            task["var"],
            task["levels"],
            task["dts"],
            task["s3pfx"],
        )

        fs = s3fs.S3FileSystem(anon=True)

        # Here we manually cache the data arrays, this is because fsspec caches the
        # entire HDF5 file by default which is large for this data, so instead we manually
        # cache the slice we need
        sha = hashlib.sha256(
            (str(ename) + str(var) + str(levels) + str(dts) + str(s3pfx)).encode()
        )
        filename = sha.hexdigest()
        cache_path = os.path.join(cache_path, filename)

        if os.path.exists(cache_path):
            ds = xr.open_dataarray(cache_path)
        else:
            with fs.open(s3pfx, "rb", block_size=1 * 1024 * 1024) as f:
                ds = xr.open_dataset(f, engine="h5netcdf", cache=False)

                if ename[0].isalpha():  # ECMWF names starting with a digit
                    xrname = ename.upper()
                else:
                    xrname = f"VAR_{ename.upper()}"

                if len(levels) == 0:
                    # Surface-level variable
                    ds = ds.sel(time=dts)[xrname]
                    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
                    ds = xr.concat([ds], pd.Index([var], name="variable"))
                    ds = ds.load()
                else:
                    # Pressure-level variable
                    ds = ds.sel(time=dts, level=[float(lvl) for lvl in levels])[xrname]
                    ds = ds.rename(latitude="lat", longitude="lon", level="variable")
                    ds["variable"] = np.array([f"{var}{lvl}" for lvl in levels])
                    ds = ds.load()

                # Save to cache, could be better optimized by not saving the coords
                # For some reason the default netcdf engine was giving errors
                if cache:
                    ds.to_netcdf(cache_path, engine="h5netcdf")

        return ename, ds

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify that date time is valid for ERA5 based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            Timestamps to be downloaded (UTC).
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for ERA5"
                )

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_path = os.path.join(datasource_cache_root(), "ncar_era5")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)
        return cache_path
