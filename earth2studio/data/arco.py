# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
import re
import shutil
import uuid
import warnings
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime

import numpy as np
import xarray as xr
import zarr
import zarr.abc.store
from loguru import logger

from earth2studio.data.utils import (
    DataRequest,
    _sync_async,
    compile_statistic_requests,
    datasource_cache_root,
    obstore_zarr_store,
    prep_data_inputs,
    run_data_requests,
)
from earth2studio.lexicon import ARCO_ERA5Lexicon
from earth2studio.lexicon.arco import ACCUMULATION_HOURS
from earth2studio.utils.type import TimeArray, VariableArray


class ARCO_ERA5:
    """Analysis-Ready, Cloud Optimized (ARCO) ERA5 is a reanalysis data source
    curated by Google. This data is stored in Zarr format and contains 31 surface
    variables, pressure level variables defined on 37 pressure levels, and model level
    variables defined on the 137 native ERA5 vertical levels, all on a 0.25 degree lat
    lon grid. Temporal resolution is 1 hour.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://cloud.google.com/storage/docs/public-datasets/era5

    Note
    ----
    Temporal statistics: ``mean``, ``sum``, ``min``, and ``max`` accept labels
    ``variable:method:window`` or ``variable:method:start:end``. Windows are
    left-closed/right-open, sampled hourly relative to the analysis time.
    ``t2m:mean:24h`` averages the preceding 24 samples, excluding that time.
    Accumulation timestamps mark interval ends; sums of overlapping multi-hour
    accumulations can double-count precipitation or radiation.

    Badges
    ------
    region:global dataclass:reanalysis dataset:era5 product:wind product:precip product:temp product:atmos provider:google
    """

    ARCO_LAT = np.linspace(90, -90, 721)
    ARCO_LON = np.linspace(0, 359.75, 1440)
    ARCO_PATH = "/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    ARCO_ML_PATH = "/gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
    ARCO_TIME_STOP = datetime(year=2025, month=12, day=31)
    lexicon = ARCO_ERA5Lexicon

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ):

        self._cache = cache
        self._verbose = verbose
        self._tmp_cache_hash: str | None = None

        self.zarr_group: zarr.core.group.AsyncGroup | None = None
        self.level_coords = None
        # Model-level store
        self.ml_zarr_group: zarr.core.group.AsyncGroup | None = None
        self.ml_level_coords = None

        self.async_timeout = async_timeout

    def _zarr_stores(self) -> tuple[zarr.abc.store.Store, zarr.abc.store.Store]:
        """Creates the pressure/surface and model-level zarr stores

        Returns
        -------
        tuple[zarr.abc.store.Store, zarr.abc.store.Store]
            Pressure/surface store and model-level store
        """
        cache_storage = self.cache if self._cache else None
        zstore = obstore_zarr_store(
            f"gs://{self.ARCO_PATH.lstrip('/')}",
            cache_storage=cache_storage,
            store_kwargs={"skip_signature": True},
        )
        ml_zstore = obstore_zarr_store(
            f"gs://{self.ARCO_ML_PATH.lstrip('/')}",
            cache_storage=cache_storage,
            store_kwargs={"skip_signature": True},
        )
        return zstore, ml_zstore

    async def _async_init(self) -> None:
        """Async initialization of zarr group"""
        zstore, ml_zstore = self._zarr_stores()

        # Pressure/surface store
        self.zarr_group = await zarr.api.asynchronous.open(store=zstore, mode="r")

        if "valid_time_stop" in self.zarr_group.attrs:
            ARCO_ERA5.ARCO_TIME_STOP = datetime.strptime(
                self.zarr_group.attrs["valid_time_stop"], "%Y-%m-%d"
            )

        self.level_coords = await (await self.zarr_group.get("level")).getitem(
            slice(None)
        )
        # Model-level store
        self.ml_zarr_group = await zarr.api.asynchronous.open(store=ml_zstore, mode="r")
        self.ml_level_coords = await (await self.ml_zarr_group.get("hybrid")).getitem(
            slice(None)
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the ARCO_ERA5Lexicon.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from ARCO_ERA5
        """

        try:
            xr_array = _sync_async(
                self.fetch, time, variable, timeout=self.async_timeout
            )
        finally:
            # Delete cache if needed
            if not self._cache:
                shutil.rmtree(self.cache, ignore_errors=True)

        return xr_array

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the ARCO_ERA5Lexicon. A variable may be qualified with
            a temporal statistic, such as ``"t2m:mean:24h"``; the hourly fields in
            the window are then fetched and reduced as they arrive.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from ARCO_ERA5
        """
        if self.zarr_group is None:
            await self._async_init()

        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        xr_array = xr.DataArray(
            data=np.full(
                (len(time), len(variable), len(self.ARCO_LAT), len(self.ARCO_LON)),
                np.nan,
            ),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": self.ARCO_LAT,
                "lon": self.ARCO_LON,
            },
        )

        requests, attrs, finalize = compile_statistic_requests(
            variable,
            time,
            np.timedelta64(1, "h"),
            xr_array.values,
            supported=self.lexicon.TIME_STATISTICS,
            source=type(self).__name__,
            verbose=self._verbose,
        )
        # Validate the extra times before reading anything
        self._validate_time(list({r.time for r in requests}))

        # Dispatch every consumer of a chunk together. A cache spanning requests
        # would retain a whole window of pressure-level chunks while waiting for
        # the next variable's consumers, despite the streaming reducers.
        chunks: dict[tuple[str, bool, int | None], list[DataRequest]] = defaultdict(
            list
        )
        chunk_requests: list[DataRequest] = []
        for request in requests:
            if request.variable in ACCUMULATION_HOURS:
                chunk_requests.append(request)
            else:
                chunks[self._chunk_key(request.time, request.variable)].append(request)

        def chunk_modifier(
            members: list[DataRequest], is_mdl: bool
        ) -> Callable[[np.ndarray], None]:
            def consume(chunk: np.ndarray) -> None:
                for member in members:
                    name, modifier = ARCO_ERA5Lexicon[member.variable]
                    sample = chunk
                    if chunk.ndim == 3:
                        levels = self.ml_level_coords if is_mdl else self.level_coords
                        sample = chunk[
                            np.searchsorted(levels, int(name.split("::")[1]))
                        ]
                    member.modifier(modifier(sample))

            return consume

        for (_, is_mdl, _), members in chunks.items():
            first = members[0]
            chunk_requests.append(
                DataRequest(
                    first.time,
                    first.lead_time,
                    first.variable,
                    chunk_modifier(members, is_mdl),
                )
            )

        async def fetch_sample(request: DataRequest) -> np.ndarray:
            if request.variable in ACCUMULATION_HOURS:
                return await self.fetch_array(request.time, request.variable)
            return await self._read_chunk(
                *self._chunk_key(request.time, request.variable)
            )

        await run_data_requests(
            chunk_requests,
            fetch_sample,
            desc="Fetching ARCO_ERA5 data",
            verbose=self._verbose,
            # Each atmospheric chunk contains every pressure/model level.
            async_workers=4,
        )
        finalize()
        xr_array.attrs.update(attrs)
        return xr_array

    def _chunk_key(self, time: datetime, variable: str) -> tuple[str, bool, int | None]:
        """Identify the zarr chunk holding a sample: (array, model level, time index)."""
        name = ARCO_ERA5Lexicon[variable][0].split("::")[0]  # type: ignore[misc]
        return (name, self._is_mdl_level(variable), self._get_time_index(time))

    async def _read_chunk(
        self, name: str, is_mdl: bool, time_index: int | None
    ) -> np.ndarray:
        """Read one zarr chunk: a whole hour of an array, or a static array."""
        group = self.ml_zarr_group if is_mdl else self.zarr_group
        zarr_array = await group.get(name)  # type: ignore[union-attr]
        if len(zarr_array.shape) == 2:
            return await zarr_array.getitem(slice(None))
        return await zarr_array.getitem(time_index)

    async def fetch_array(self, time: datetime, variable: str) -> np.ndarray:
        """Fetches requested array from remote store

        Parameters
        ----------
        time : datetime
            Time to fetch
        variable : str
            Variable to fetch

        Returns
        -------
        np.ndarray
            Data
        """
        if self.zarr_group is None or self.ml_zarr_group is None:
            raise ValueError("Zarr group is not initialized")
        # Get time index (vanilla zarr doesnt support date indices)
        time_index = self._get_time_index(time)
        logger.debug(
            f"Fetching ARCO zarr array for variable: {variable} at {time.isoformat()}"
        )
        try:
            arco_name, modifier = ARCO_ERA5Lexicon[variable]  # type: ignore[misc]
        except KeyError as e:
            logger.error(f"variable id {variable} not found in ARCO_ERA5Lexicon")
            raise e

        parts = arco_name.split("::")
        arco_variable, level = parts[0], (parts[1] if len(parts) > 1 else "")
        if self._is_mdl_level(variable):
            zarr_group = self.ml_zarr_group
            level_coords = self.ml_level_coords
        else:
            zarr_group = self.zarr_group
            level_coords = self.level_coords

        zarr_array = await zarr_group.get(arco_variable)
        shape = zarr_array.shape
        # Static variables
        if len(shape) == 2:
            data = await zarr_array.getitem(slice(None))
            output = modifier(data)
        # Surface variable
        elif len(shape) == 3:
            accumulation_hours = ACCUMULATION_HOURS.get(variable)
            # Sum hourly ARCO_ERA5 fields to support cumulative variables.
            if accumulation_hours is not None:
                start_index = time_index - accumulation_hours + 1
                if start_index < 0:
                    raise ValueError(
                        f"Cannot compute {accumulation_hours}-hour accumulation for {variable} at {time}"
                    )
                data = np.array(await zarr_array.getitem(start_index), copy=True)
                for index in range(start_index + 1, time_index + 1):
                    sample = await zarr_array.getitem(index)
                    np.add(data, sample, out=data)
                    del sample
                output = modifier(data)
            else:
                data = await zarr_array.getitem(time_index)
                output = modifier(data)
        # Atmospheric variable
        else:
            # Load levels coordinate system from Zarr store and check
            level_index = np.searchsorted(level_coords, int(level))
            data = await zarr_array.getitem((time_index, level_index))
            output = modifier(data)

        return output

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "arco_era5")
        if not self._cache:
            if self._tmp_cache_hash is None:
                # First access for temp cache: create a random suffix to avoid collisions
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_arco_era5_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for ARCO_ERA5.

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for ARCO_ERA5"
                )

            if time < datetime(year=1940, month=1, day=1):
                raise ValueError(
                    f"Requested date time {time} needs to be after January 1st, 1940 for ARCO_ERA5"
                )

            if time > cls.ARCO_TIME_STOP:
                raise ValueError(
                    f"Requested date time {time} needs to be on or before {cls.ARCO_TIME_STOP.strftime('%B %d, %Y')} for ARCO_ERA5"
                )

            # if not self.available(time):
            #     raise ValueError(f"Requested date time {time} not available in ARCO_ERA5")

    @classmethod
    def _get_time_index(cls, time: datetime) -> int:
        """Small little index converter to go from datetime to integer index.
        We don't need to do with with xarray, but since we are vanilla zarr for speed
        this conversion must be manual.

        Parameters
        ----------
        time : datetime
            Input date time

        Returns
        -------
        int
            Time coordinate index of ARCO_ERA5 data
        """
        start_date = datetime(year=1900, month=1, day=1)
        duration = time - start_date
        return int(divmod(duration.total_seconds(), 3600)[0])

    @classmethod
    def _is_mdl_level(cls, variable: str) -> bool:
        """Checks if given variable is a model level variable based on the lexicon pattern.

        Parameters
        ----------
        variable : str
            Variable to check

        Returns
        -------
        bool
            If variable is a model level variable
        """
        return bool(re.match(r"^[a-zA-Z0-9]+[0-9]+k$", variable))

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Checks if given date time is avaliable in the ARCO_ERA5 data source

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
            time = datetime.utcfromtimestamp(float((time - _unix) / _ds))

        # Offline checks
        try:
            cls._validate_time([time])
        except ValueError:
            return False

        gcstore = obstore_zarr_store(
            f"gs://{cls.ARCO_PATH.lstrip('/')}", store_kwargs={"skip_signature": True}
        )
        zarr_group = zarr.open(gcstore, mode="r")
        # Load time coordinate system from Zarr store and check
        time_index = cls._get_time_index(time)
        max_index = zarr_group["time"][-1]
        return time_index >= 0 and time_index <= max_index


class ARCO(ARCO_ERA5):
    """Deprecated alias for :class:`ARCO_ERA5`."""

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
    ) -> None:
        warnings.warn(
            "ARCO has been renamed to ARCO_ERA5 and will be removed in a future "
            "release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cache=cache, verbose=verbose, async_timeout=async_timeout)
