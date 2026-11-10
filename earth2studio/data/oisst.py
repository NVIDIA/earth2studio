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

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import netCDF4
import numpy as np
import obstore as obs
import torch
import xarray as xr

from earth2studio.data.utils import (
    _CleanupAfterLastCall,
    _sync_async,
    async_retry,
    datasource_cache_root,
    gather_with_concurrency,
    obstore_fetch_to_cache,
    obstore_store_from_url,
    prep_data_inputs,
    resolve_async_workers,
)
from earth2studio.utils.type import TimeArray, VariableArray


class OISST:
    """NOAA daily 0.25° Optimum Interpolation Sea Surface Temperature (OISST v2.1),
    read directly from NCEI.

    NCEI publishes a preliminary analysis for each day about one day later and
    replaces it with the final analysis about two weeks later. This source reads
    the final file when it exists and otherwise the preliminary one, so recent
    dates are available for real-time use. :class:`PlanetaryComputerOISST` serves
    the final product only.

    Values are in kelvin on the native 720 x 1440 cell-centred grid, with latitude
    from north to south and NaN over land. Subdaily request times select that
    day's analysis. The ``sst_date`` coordinate records the date of the analysis
    returned for each requested time.

    Parameters
    ----------
    lag_days : int, optional
        Days before the requested date to start searching, by default 0. For
        example, 1 uses the previous day's analysis, as for a forecast that may
        only use data available at its initial time.
    lookback_days : int, optional
        Number of consecutive days to try, newest first, starting ``lag_days``
        before the requested date, by default 1. Values above 1 fall back to an
        older analysis when the newest one is not yet published.
    cache : bool, optional
        Cache downloaded NetCDF files in the local filesystem cache, by default
        True
    verbose : bool, optional
        Show download progress, by default True
    async_timeout : int, optional
        Time in seconds after which a download is cancelled, by default 600
    async_workers : int | None, optional
        Maximum number of concurrent downloads, by default None
    retries : int, optional
        Retries per file for transient failures, by default 3
    device : torch.device | str, optional
        Device of the returned array, by default "cpu"

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.ncei.noaa.gov/products/optimum-interpolation-sst
    - https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/

    Badges
    ------
    region:global dataclass:analysis product:ocean provider:noaa
    """

    ROOT = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
    LAT = np.arange(89.875, -90, -0.25)
    LON = np.arange(0.125, 360, 0.25)

    def __init__(
        self,
        lag_days: int = 0,
        lookback_days: int = 1,
        cache: bool = True,
        verbose: bool = True,
        async_timeout: int = 600,
        async_workers: int | None = None,
        retries: int = 3,
        device: torch.device | str = "cpu",
    ) -> None:
        if lag_days < 0 or lookback_days < 1:
            raise ValueError("lag_days must be nonnegative and lookback_days positive")
        self.lag_days, self.lookback_days = lag_days, lookback_days
        self._cache, self._verbose = cache, verbose
        self.async_timeout, self.async_workers, self.retries = (
            async_timeout,
            async_workers,
            retries,
        )
        self.device = torch.device(device)
        self.store = obstore_store_from_url(self.ROOT)
        self._temporary = uuid4().hex
        # Overlapping calls share the temporary cache; the last one removes it
        self._calls = _CleanupAfterLastCall(self)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch daily SST; arguments and result match fetch."""
        return _sync_async(self.fetch, time, variable, timeout=self.async_timeout)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch SST in kelvin for the requested UTC dates.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Requested dates; subdaily times select the same daily product.
        variable : str | list[str] | VariableArray
            The supported variable is sst.

        Returns
        -------
        xr.DataArray
            Fields with time, variable, latitude and longitude dimensions.
        """
        with self._calls:
            return await self._fetch(time, variable)

    async def _fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        times, variables = prep_data_inputs(time, variable)
        if variables != ["sst"]:
            raise KeyError("OISST supplies only sst")
        self._validate_time(times)
        Path(self.cache).mkdir(parents=True, exist_ok=True)
        fields = await gather_with_concurrency(
            [self._fetch_date(t) for t in times],
            max_workers=resolve_async_workers(self.async_workers, len(times)),
            desc="Fetching OISST",
            verbose=self._verbose,
        )
        values, dates = zip(*fields)
        result = xr.DataArray(
            np.stack(values)[:, None],
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": times,
                "variable": variables,
                "lat": self.LAT,
                "lon": self.LON,
                "sst_date": ("time", np.asarray(dates, dtype="datetime64[ns]")),
            },
        )
        return result.e2s.to_backend("auto", device=self.device)

    async def _fetch_date(self, requested: datetime) -> tuple[np.ndarray, datetime]:
        for offset in range(self.lag_days, self.lag_days + self.lookback_days):
            date = requested.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=offset)
            # The final analysis replaces the preliminary one when published
            for suffix in ("", "_preliminary"):
                key = f"{date:%Y%m}/oisst-avhrr-v02r01.{date:%Y%m%d}{suffix}.nc"
                try:
                    path = await async_retry(
                        obstore_fetch_to_cache,
                        self.store,
                        key,
                        self.cache,
                        atomic=True,
                        retries=self.retries,
                        task_timeout=self.async_timeout,
                        # obstore errors derive from BaseError, not OSError
                        exceptions=(OSError, TimeoutError, obs.exceptions.BaseError),
                        no_retry=(FileNotFoundError, obs.exceptions.NotFoundError),
                    )
                except (FileNotFoundError, obs.exceptions.NotFoundError):
                    continue
                with netCDF4.Dataset(path) as data:
                    variable = data.variables["sst"]
                    if str(variable.units).lower() not in {
                        "celsius",
                        "degrees_celsius",
                        "degree_celsius",
                        "degc",
                    }:
                        raise ValueError(f"Unsupported OISST units: {variable.units}")
                    values = np.ma.filled(variable[0, 0], np.nan).astype(np.float32)
                    values += np.float32(273.15)
                    lat = np.asarray(data.variables["lat"][:])
                    lon = np.asarray(data.variables["lon"][:])
                order = np.argsort(lat)[::-1]
                if not np.allclose(lat[order], self.LAT) or not np.allclose(
                    lon, self.LON
                ):
                    raise ValueError("Unexpected OISST grid")
                return values[order], date
        raise FileNotFoundError(f"No OISST product for {requested:%Y-%m-%d}")

    @staticmethod
    def _validate_time(times: list[datetime]) -> None:
        if any(t < datetime(1981, 9, 1) for t in times):
            raise ValueError("OISST begins on 1981-09-01")

    def cleanup(self) -> None:
        """Remove temporary files when persistent caching is disabled."""
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

    @property
    def cache(self) -> str:
        """Directory containing NOAA product files."""
        root = Path(datasource_cache_root()) / "oisst"
        return str(root if self._cache else root / f"tmp_{self._temporary}")

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Return whether the date lies within the dataset's time range."""
        times, _ = prep_data_inputs(time, ["sst"])
        return (
            times[0] >= datetime(1981, 9, 1)
            and times[0].date() <= datetime.now(timezone.utc).date()
        )
