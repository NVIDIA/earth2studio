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

import asyncio
import hashlib
import os
import pathlib
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import sleep

import numpy as np
import xarray as xr
from loguru import logger

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_forecast_inputs,
)
from earth2studio.lexicon import CAMSLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

try:
    import cdsapi
except ImportError:
    OptionalDependencyFailure("data")
    cdsapi = None

# CAMS Global forecast available from 2015-01-01 onward
_CAMS_GLOBAL_MIN_TIME = datetime(2015, 1, 1)

_GLOBAL_DATASET = "cams-global-atmospheric-composition-forecasts"


@dataclass
class _CAMSVarInfo:
    e2s_name: str
    api_name: str
    nc_key: str
    dataset: str
    level: str
    index: int


def _resolve_variable(e2s_name: str, index: int) -> _CAMSVarInfo:
    cams_key, _ = CAMSLexicon[e2s_name]
    dataset, api_name, nc_key, level = cams_key.split("::")
    return _CAMSVarInfo(
        e2s_name=e2s_name,
        api_name=api_name,
        nc_key=nc_key,
        dataset=dataset,
        level=level,
        index=index,
    )


def _download_cams_netcdf(
    client: "cdsapi.Client",
    dataset: str,
    request_body: dict,
    cache_path: pathlib.Path,
    verbose: bool = True,
) -> pathlib.Path:
    if cache_path.is_file():
        return cache_path

    r = client.retrieve(dataset, request_body)
    while True:
        r.update()
        reply = r.reply
        if verbose:
            logger.debug(f"Request ID:{reply['request_id']}, state: {reply['state']}")
        if reply["state"] == "completed":
            break
        elif reply["state"] in ("queued", "running"):
            sleep(5.0)
        elif reply["state"] in ("failed",):
            raise RuntimeError(
                f"CAMS request failed for {dataset}: "
                + reply.get("error", {}).get("message", "unknown error")
            )
        else:
            sleep(2.0)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=cache_path.parent, suffix=".nc.tmp")
    try:
        os.close(tmp_fd)
        r.download(tmp_name)
        os.replace(tmp_name, cache_path)
    except Exception:
        pathlib.Path(tmp_name).unlink(missing_ok=True)
        raise
    return cache_path


def _extract_field(
    ds: xr.Dataset,
    nc_key: str,
    lead_time_hours: int | None = None,
) -> np.ndarray:
    if nc_key not in ds:
        raise ValueError(
            f"Variable '{nc_key}' not found in NetCDF. Available: {list(ds.data_vars)}"
        )
    field = ds[nc_key]
    non_spatial = [d for d in field.dims if d not in ("latitude", "longitude")]
    isel: dict[str, int] = {}
    for d in non_spatial:
        if d == "forecast_period" and lead_time_hours is not None:
            fp_vals = field.coords["forecast_period"].values.astype(float)
            target = float(lead_time_hours)
            nearest_idx = int(np.argmin(np.abs(fp_vals - target)))
            isel[d] = nearest_idx
        else:
            isel[d] = 0
    if isel:
        field = field.isel(isel)
    return field.values


def _validate_cams_time(times: list[datetime], min_time: datetime, name: str) -> None:
    for t in times:
        t_naive = t.replace(tzinfo=None) if t.tzinfo else t
        if t_naive < min_time:
            raise ValueError(
                f"Requested time {t} is before {name} availability "
                f"(earliest: {min_time})"
            )
        if t_naive.minute != 0 or t_naive.second != 0:
            raise ValueError(f"Requested time {t} must be on the hour for {name}")


def _validate_cams_leadtime(lead_times: list[timedelta], max_hours: int) -> None:
    for lt in lead_times:
        hours = int(lt.total_seconds() // 3600)
        if lt.total_seconds() % 3600 != 0:
            raise ValueError(f"Lead time {lt} must be a whole number of hours")
        if hours < 0 or hours > max_hours:
            raise ValueError(
                f"Lead time {lt} ({hours}h) outside valid range [0, {max_hours}]h"
            )


@check_optional_dependencies()
class CAMS_FX:
    """CAMS Global atmospheric composition forecast data source.

    Uses the ``cams-global-atmospheric-composition-forecasts`` dataset.
    Grid is 0.4 deg global, read dynamically from the downloaded NetCDF.

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
    Additional information on the data repository, registration, and authentication can
    be referenced here:

    - https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts
    - https://cds.climate.copernicus.eu/how-to-api

    Badges
    ------
    region:global dataclass:simulation product:airquality
    """

    MAX_LEAD_HOURS = 120

    def __init__(self, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose
        self._cds_client: "cdsapi.Client | None" = None

    @property
    def _client(self) -> "cdsapi.Client":
        if self._cds_client is None:
            if cdsapi is None:
                raise ImportError(
                    "cdsapi is required for CAMS_FX. "
                    "Install with: pip install 'earth2studio[data]'"
                )
            self._cds_client = cdsapi.Client(
                debug=False, quiet=True, wait_until_complete=False
            )
        return self._cds_client

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve CAMS Global forecast data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Forecast initialization times (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times.
        variable : str | list[str] | VariableArray
            Variables to return. Must be in CAMSLexicon.

        Returns
        -------
        xr.DataArray
            CAMS forecast data array with dims [time, lead_time, variable, lat, lon]
        """
        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)
        self._validate_time(time)
        self.cache.mkdir(parents=True, exist_ok=True)

        data_arrays = []
        for t0 in time:
            da = self._fetch_forecast(t0, lead_time, variable)
            data_arrays.append(da)

        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async retrieval of CAMS Global forecast data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Forecast initialization times (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times.
        variable : str | list[str] | VariableArray
            Variables to return. Must be in CAMSLexicon.

        Returns
        -------
        xr.DataArray
            CAMS forecast data array with dims [time, lead_time, variable, lat, lon]
        """
        return await asyncio.to_thread(self.__call__, time, lead_time, variable)

    @classmethod
    def available(
        cls,
        time: datetime | list[datetime],
    ) -> bool:
        """Check if CAMS Global forecast data is available for the requested times.

        Parameters
        ----------
        time : datetime | list[datetime]
            Timestamps to check availability for.

        Returns
        -------
        bool
            True if all requested times are within the valid range.
        """
        if isinstance(time, datetime):
            time = [time]
        return all(
            (t.replace(tzinfo=None) if t.tzinfo else t) >= _CAMS_GLOBAL_MIN_TIME
            for t in time
        )

    @staticmethod
    def _validate_time(times: list[datetime]) -> None:
        _validate_cams_time(times, _CAMS_GLOBAL_MIN_TIME, "CAMS Global forecast")

    def _fetch_forecast(
        self,
        time: datetime,
        lead_times: np.ndarray,
        variables: list[str],
    ) -> xr.DataArray:
        var_infos = []
        for i, v in enumerate(variables):
            info = _resolve_variable(v, i)
            if info.dataset != _GLOBAL_DATASET:
                raise ValueError(
                    f"CAMS_FX only supports Global dataset, got '{info.dataset}' "
                    f"for variable '{v}'."
                )
            var_infos.append(info)

        api_vars = list(dict.fromkeys(vi.api_name for vi in var_infos))
        lead_hours = [
            str(int(np.timedelta64(lt, "h").astype(int))) for lt in lead_times
        ]

        _validate_cams_leadtime(
            [timedelta(hours=int(h)) for h in lead_hours], self.MAX_LEAD_HOURS
        )

        nc_path = self._download_cached(time, api_vars, lead_hours)

        ds = xr.open_dataset(nc_path, decode_timedelta=False)
        lat = ds.latitude.values
        lon = ds.longitude.values

        da = xr.DataArray(
            data=np.empty((1, len(lead_times), len(variables), len(lat), len(lon))),
            dims=["time", "lead_time", "variable", "lat", "lon"],
            coords={
                "time": [time],
                "lead_time": lead_times,
                "variable": variables,
                "lat": lat,
                "lon": lon,
            },
        )

        for lt_idx, lt_h in enumerate(lead_hours):
            for info in var_infos:
                _, modifier = CAMSLexicon[info.e2s_name]
                da[0, lt_idx, info.index] = modifier(
                    _extract_field(
                        ds,
                        info.nc_key,
                        lead_time_hours=int(lt_h),
                    )
                )

        ds.close()
        return da

    def _download_cached(
        self,
        time: datetime,
        api_vars: list[str],
        lead_hours: list[str],
    ) -> pathlib.Path:
        date_str = time.strftime("%Y-%m-%d")
        sha = hashlib.sha256(
            f"cams_fx_{'_'.join(sorted(api_vars))}"
            f"_{'_'.join(sorted(lead_hours, key=int))}"
            f"_{date_str}_{time.hour:02d}".encode()
        )
        cache_path = self.cache / (sha.hexdigest() + ".nc")

        request_body: dict = {
            "variable": api_vars,
            "date": [f"{date_str}/{date_str}"],
            "type": ["forecast"],
            "time": [f"{time.hour:02d}:00"],
            "leadtime_hour": lead_hours,
            "data_format": "netcdf",
        }

        if self._verbose:
            logger.info(
                f"Fetching CAMS Global forecast for {date_str} "
                f"{time.hour:02d}:00 lead_hours={lead_hours} vars={api_vars}"
            )
        return _download_cams_netcdf(
            self._client, _GLOBAL_DATASET, request_body, cache_path, self._verbose
        )

    @property
    def cache(self) -> pathlib.Path:
        """Cache location."""
        cache_location = pathlib.Path(datasource_cache_root()) / "cams"
        if not self._cache:
            cache_location = cache_location / "tmp_cams_fx"
        return cache_location
