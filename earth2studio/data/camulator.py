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

import shutil
import uuid
from datetime import datetime
from typing import Literal

import numpy as np
import xarray as xr
from huggingface_hub import hf_hub_download
from loguru import logger

from earth2studio.data.utils import datasource_cache_dir, prep_data_inputs
from earth2studio.lexicon.camulator import CAMulatorLexicon
from earth2studio.utils.type import TimeArray, VariableArray

HF_REPO_ID = "willychap/camulator"
CAMULATOR_HF_REVISION = "4da83abd466aae4f7f39473c7f4bef83dd5a2ea0"

# 1 degree CESM finite-volume grid; Earth2Studio orientation (north to south)
CAMULATOR_GRID_LAT = np.linspace(90.0, -90.0, 192)
CAMULATOR_GRID_LON = np.linspace(0.0, 358.75, 288)

_FORCING_FILES = {
    "cyclic": "forcing_data/b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc",
    "transient": "forcing_data/b.e21.CREDIT_climate_branch_1980_2014.nc",
}


class CAMulatorForcing:
    """CAMulator prescribed forcing data source: TOA insolation (``mtdwswrf``),
    sea-surface temperature (``sst``), sea-ice fraction (``sic``) and CO2 volume
    mixing ratio (``global_mean_co2``, ppm) on the 1 degree CAMulator grid, as
    shipped with the CAMulator model on HuggingFace. Two records are available:
    a cyclic climatological year (default; every year of a rollout sees the same
    forcing) and the transient 1980-2014 record. Files are downloaded on demand
    and cached.

    The forcing files use a 365-day (no leap) calendar at 6-hourly resolution.
    Requested times are matched by month, day and hour (cyclic) or year, month,
    day and hour (transient); the 29th of February has no forcing and by default
    reuses the 28th.

    Parameters
    ----------
    mode : str, optional
        Either "cyclic" (climatological year) or "transient" (1980-2014), by
        default "cyclic"
    forcing_file : str, optional
        Local NetCDF file to read instead of downloading the shipped file. Must
        hold the CESM variables ``SOLIN``, ``SST``, ``ICEFRAC`` and
        ``co2vmr_3d`` with dimensions ``(time, latitude, longitude)`` on the
        CAMulator grid and a no-leap time coordinate, by default None
    leap_day : str, optional
        Handling of 29 February requests: "nearest" reuses the 28 February
        forcing (a warning is logged once), "raise" raises a ValueError, by
        default "nearest"
    cache : bool, optional
        Cache the downloaded forcing file on local disk, by default True
    verbose : bool, optional
        Log download progress, by default True

    Warning
    -------
    The forcing is served from single NetCDF files downloaded on first use: about
    1.3 GB for the cyclic year and 9.7 GB for the transient record. With
    ``cache=True`` the file lives in the HuggingFace hub cache (``HF_HOME``);
    with ``cache=False`` it is deleted after every call, which is unsuitable for
    model rollouts.

    Note
    ----
    Additional information on the forcing data can be found at:

    - https://huggingface.co/willychap/camulator
    - https://arxiv.org/abs/2504.06007

    Badges
    ------
    region:global dataclass:simulation product:atmos product:ocean provider:ncar
    """

    def __init__(
        self,
        mode: Literal["cyclic", "transient"] = "cyclic",
        forcing_file: str | None = None,
        leap_day: Literal["nearest", "raise"] = "nearest",
        cache: bool = True,
        verbose: bool = True,
    ):
        if mode not in _FORCING_FILES:
            raise ValueError("mode must be either 'cyclic' or 'transient'")
        if leap_day not in ("nearest", "raise"):
            raise ValueError("leap_day must be either 'nearest' or 'raise'")
        self._mode = mode
        self._forcing_file = forcing_file
        self._leap_day = leap_day
        self._cache = cache
        self._verbose = verbose
        self._tmp_cache_hash: str | None = None
        self._ds: xr.Dataset | None = None
        self._time_index: dict[tuple[int, ...], int] = {}
        self._warned_leap_day = False
        self.lat = CAMULATOR_GRID_LAT
        self.lon = CAMULATOR_GRID_LON

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC). Must fall on 00, 06, 12 or 18 UTC.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be forcing variables in the CAMulator lexicon.

        Returns
        -------
        xr.DataArray
            Forcing data array with dimensions ``[time, variable, lat, lon]``
        """
        try:
            result = self._fetch_array(time, variable)
        finally:
            if not self._cache:
                self._close()
                shutil.rmtree(self.cache, ignore_errors=True)
        return result

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to get data

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC). Must fall on 00, 06, 12 or 18 UTC.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be forcing variables in the CAMulator lexicon.

        Returns
        -------
        xr.DataArray
            Forcing data array with dimensions ``[time, variable, lat, lon]``
        """
        return self(time, variable)

    def _fetch_array(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        time_list, var_list = prep_data_inputs(time, variable)
        ds = self._open_dataset()
        indices = [self._index_of(t) for t in time_list]

        arrays = []
        for var in var_list:
            try:
                cesm_name, modifier = CAMulatorLexicon[var]
            except KeyError as e:
                raise KeyError(f"Unknown CAMulator variable id: {e}") from e
            if "::" in cesm_name or cesm_name not in ds:
                raise KeyError(f"Variable {var} is not provided by CAMulatorForcing")
            data = ds[cesm_name].isel(time=indices).values.astype(np.float32)
            data = modifier(data)
            # File latitude runs south to north; Earth2Studio is north to south
            arrays.append(np.ascontiguousarray(data[:, ::-1, :]))

        return xr.DataArray(
            data=np.stack(arrays, axis=1),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": np.array(time_list, dtype="datetime64[ns]"),
                "variable": np.array(var_list, dtype=object),
                "lat": self.lat,
                "lon": self.lon,
            },
        )

    def _index_of(self, time: datetime) -> int:
        """Index of the requested (Gregorian) time in the no-leap forcing record."""
        self._validate_time([time])
        month, day = time.month, time.day
        if (month, day) == (2, 29):
            if self._leap_day == "raise":
                raise ValueError(
                    f"{time} is a leap day; the CAMulator forcing uses a 365-day calendar"
                )
            if not self._warned_leap_day:
                logger.warning(
                    "CAMulator forcing uses a 365-day calendar; 29 February requests "
                    "reuse the 28 February forcing"
                )
                self._warned_leap_day = True
            day = 28

        key: tuple[int, ...]
        if self._mode == "cyclic":
            key = (month, day, time.hour)
        else:
            key = (time.year, month, day, time.hour)
        try:
            return self._time_index[key]
        except KeyError as e:
            raise ValueError(
                f"Time {time} is not available in the CAMulator {self._mode} forcing"
            ) from e

    def _open_dataset(self) -> xr.Dataset:
        if self._ds is not None:
            return self._ds
        if self._forcing_file is not None:
            path = self._forcing_file
        else:
            path = self._fetch_file(_FORCING_FILES[self._mode])
        self._ds = xr.open_dataset(
            path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
        )

        times = self._ds["time"].values
        for i, t in enumerate(times):
            if self._mode == "cyclic":
                self._time_index[(t.month, t.day, t.hour)] = i
            else:
                self._time_index[(t.year, t.month, t.day, t.hour)] = i
        return self._ds

    def _close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None
            self._time_index = {}

    def _fetch_file(self, filename: str) -> str:
        """Download a file from the CAMulator HuggingFace repository (pinned
        revision). With caching enabled the shared HuggingFace hub cache is used."""
        if self._verbose:
            logger.info("Fetching CAMulator forcing file: {}", filename)
        if self._cache:
            return hf_hub_download(HF_REPO_ID, filename, revision=CAMULATOR_HF_REVISION)
        return hf_hub_download(
            HF_REPO_ID, filename, revision=CAMULATOR_HF_REVISION, local_dir=self.cache
        )

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify that the requested times fall on the 6-hourly forcing grid.

        Parameters
        ----------
        times : list[datetime]
            Requested times

        Raises
        ------
        ValueError
            If a time is not at 00, 06, 12 or 18 UTC
        """
        for time in times:
            if time.hour % 6 or time.minute or time.second or time.microsecond:
                raise ValueError(
                    f"CAMulator forcing is 6-hourly (00/06/12/18 UTC); got {time}"
                )

    @property
    def cache(self) -> str:
        """Return the local cache path for downloaded files."""
        if not self._cache and self._tmp_cache_hash is None:
            self._tmp_cache_hash = uuid.uuid4().hex[:8]
        return datasource_cache_dir("camulator", self._cache, self._tmp_cache_hash)

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Checks if the given time is on the 6-hourly CAMulator forcing grid.
        Whether a particular date exists in the record depends on the ``mode`` of
        the instance (any date for the cyclic year, 1980-2014 for the transient
        record), which is checked when data is requested.

        Parameters
        ----------
        time : datetime | np.datetime64
            Time to check

        Returns
        -------
        bool
            True if the time falls on 00, 06, 12 or 18 UTC
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[us]").astype(datetime)
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True
