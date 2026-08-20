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

import numpy as np
import xarray as xr
from huggingface_hub import hf_hub_download
from loguru import logger

from earth2studio.data.utils import datasource_cache_dir
from earth2studio.lexicon.samudrace import SamudrACELexicon
from earth2studio.utils.type import TimeArray, VariableArray

HF_REPO_ID = "allenai/SamudrACE-CM4-piControl"
HF_REVISION = "48567c80676de7db51d270c8c0c5dd61f08c9f39"

# Initial-condition timestamps published in the HuggingFace repository. The
# SamudrACE checkpoint was trained on a GFDL CM4 preindustrial-control run, so
# these are CM4 model-year timestamps, not real calendar dates.
IC_TIMESTAMPS = (
    "0151-01-06T00:00:00",
    "0311-01-01T00:00:00",
    "0313-01-01T00:00:00",
    "0315-01-01T00:00:00",
    "0317-01-01T00:00:00",
    "0319-01-01T00:00:00",
)

# Forcing scenarios published in the HuggingFace repository. Scenario "0151"
# pairs with the 0151-01-06 initial condition; scenario "0311" pairs with the
# 0311/0313/0315/0317/0319 January 1 initial conditions.
FORCING_SCENARIOS = ("0151", "0311")


def _prep_time_inputs(time: datetime | list[datetime] | TimeArray) -> list[datetime]:
    """Normalize a data source time input to a list of datetimes.

    Handles CM4 model-year timestamps (e.g. year 151), which are outside the
    range representable by nanosecond-precision timestamps.

    Parameters
    ----------
    time : datetime | list[datetime] | TimeArray
        Datetime, list of datetimes, or array of np.datetime64.

    Returns
    -------
    list[datetime]
        List of datetimes.
    """
    if isinstance(time, datetime):
        return [time]
    if isinstance(time, np.datetime64):
        time = np.array([time])
    if isinstance(time, np.ndarray):
        return list(time.astype("datetime64[s]").astype(object))
    return list(time)


def _prep_variable_inputs(variable: str | list[str] | VariableArray) -> list[str]:
    """Normalize a data source variable input to a list of strings.

    Parameters
    ----------
    variable : str | list[str] | VariableArray
        String, list of strings, or array of strings.

    Returns
    -------
    list[str]
        List of variable names.
    """
    if isinstance(variable, str):
        return [variable]
    return [str(v) for v in variable]


def _to_fme_names(variables: list[str]) -> list[str]:
    """Map Earth2Studio variable names to FME names via the SamudrACE lexicon.

    Parameters
    ----------
    variables : list[str]
        Earth2Studio variable names.

    Returns
    -------
    list[str]
        FME variable names.

    Raises
    ------
    KeyError
        If a name is not in the SamudrACE lexicon.
    """
    return [SamudrACELexicon[v][0] for v in variables]


def _orient_north_to_south(da: xr.DataArray) -> xr.DataArray:
    """Order the latitude dimension north to south (Earth2Studio convention).

    Parameters
    ----------
    da : xr.DataArray
        Data array with a ``lat`` coordinate.

    Returns
    -------
    xr.DataArray
        Data array with descending latitude.
    """
    if da["lat"].values[0] < da["lat"].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
        # Materialize so the result is not a negative-stride view
        da = da.copy(data=np.ascontiguousarray(da.values))
    return da


class _SamudrACEBase:
    """Shared HuggingFace download/caching behavior for SamudrACE sources."""

    def __init__(self, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose
        self._tmp_cache_hash: str | None = None

    @property
    def cache(self) -> str:
        """Return the local cache path for downloaded files."""
        if not self._cache and self._tmp_cache_hash is None:
            # First access of the temp cache: give it a per-instance suffix, so
            # one source's clean-up cannot delete another's downloads
            self._tmp_cache_hash = uuid.uuid4().hex[:8]
        return datasource_cache_dir("samudrace", self._cache, self._tmp_cache_hash)

    def _fetch_file(self, filename: str) -> str:
        """Download a file from the SamudrACE HuggingFace repository.

        With caching enabled the shared HuggingFace hub cache is used, so
        files already present locally are not downloaded again.

        Parameters
        ----------
        filename : str
            Path of the file within the repository.

        Returns
        -------
        str
            Local filesystem path of the downloaded file.
        """
        if self._verbose:
            logger.info("Fetching SamudrACE file: {}", filename)
        if self._cache:
            return hf_hub_download(HF_REPO_ID, filename, revision=HF_REVISION)
        return hf_hub_download(
            HF_REPO_ID, filename, revision=HF_REVISION, local_dir=self.cache
        )

    def _clean_up(self) -> None:
        """Remove temporary downloads when caching is disabled."""
        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)


class SamudrACEData(_SamudrACEBase):
    """SamudrACE initial-condition data source.

    Provides the combined atmosphere and ocean initial-condition variables of
    the SamudrACE coupled emulator from the ``allenai/SamudrACE-CM4-piControl``
    HuggingFace repository, served with Earth2Studio variable names via the
    ``SamudrACELexicon``.

    The SamudrACE checkpoint was trained on a GFDL CM4 preindustrial-control
    run, so initial conditions exist only at the published CM4 model-year
    timestamps (e.g. ``0151-01-06T00:00:00``); requesting any other time
    raises an error. Because such model years are outside the range of
    nanosecond-precision timestamps, times should be provided as
    second-precision ``np.datetime64`` values or ``datetime`` objects, and the
    returned time coordinate is second precision.

    Parameters
    ----------
    cache : bool, optional
        Cache downloaded files locally (via the shared HuggingFace hub
        cache), by default True.
    verbose : bool, optional
        Log download progress, by default True.

    Warning
    -------
    Each initial condition consists of two NetCDF files of roughly 100 MB
    total that are downloaded on demand.

    Note
    ----
    For more information see the following references:

    - SamudrACE paper: https://arxiv.org/abs/2509.12490
    - HuggingFace repo: https://huggingface.co/allenai/SamudrACE-CM4-piControl

    Badges
    ------
    region:global dataclass:simulation product:temp product:atmos product:ocean
    """

    def __init__(self, cache: bool = True, verbose: bool = True):
        super().__init__(cache=cache, verbose=verbose)
        self._datasets: dict[tuple[str, str], xr.Dataset] = {}

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch initial-condition data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for. Every requested time must be one
            of the published initial-condition timestamps.
        variable : str | list[str] | VariableArray
            Earth2Studio variable names (see ``SamudrACELexicon``).

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]`` and
            latitude ordered north to south.
        """
        try:
            result = self._fetch_array(time, variable)
            # Materialize before the temporary downloads are removed below
            if not self._cache:
                result = result.load()
        finally:
            # Clean up on the error path too, so an interrupted download does
            # not leave a partial file behind
            if not self._cache:
                self._close_datasets()
                self._clean_up()

        return result

    def _fetch_array(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Assemble the requested initial-condition fields into one array.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Earth2Studio variable names.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``, lazy
            unless the caller loads it.
        """
        time_list = _prep_time_inputs(time)
        var_list = _prep_variable_inputs(variable)
        fme_list = _to_fme_names(var_list)

        timestamps = [self._validate_time(t) for t in time_list]

        arrays = []
        for timestamp in timestamps:
            atm_ds = self._open_component(timestamp, "atmosphere")
            ocean_ds = self._open_component(timestamp, "ocean")
            fields = []
            for e2s_name, fme_name in zip(var_list, fme_list):
                if fme_name in atm_ds:
                    da = atm_ds[fme_name]
                elif fme_name in ocean_ds:
                    da = ocean_ds[fme_name]
                else:
                    raise KeyError(
                        f"Variable '{e2s_name}' (FME name '{fme_name}') not "
                        f"found in the SamudrACE initial-condition files for "
                        f"timestamp '{timestamp}'"
                    )
                if "sample" in da.dims:
                    da = da.isel(sample=0)
                fields.append(da.transpose("lat", "lon"))
            arrays.append(xr.concat(fields, dim="variable"))

        result = xr.concat(arrays, dim="time")
        result = result.assign_coords(
            time=np.array(time_list, dtype="datetime64[s]"),
            variable=np.array(var_list, dtype=object),
        )
        result = result.transpose("time", "variable", "lat", "lon")
        return _orient_north_to_south(result)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async wrapper for :meth:`__call__`.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Earth2Studio variable names.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        return self(time, variable)

    def _validate_time(self, time: datetime) -> str:
        """Validate a requested time against the published IC timestamps.

        Parameters
        ----------
        time : datetime
            Requested time.

        Returns
        -------
        str
            The matching initial-condition timestamp identifier.

        Raises
        ------
        ValueError
            If the time is not one of the published IC timestamps.
        """
        timestamp = time.isoformat()
        if timestamp not in IC_TIMESTAMPS:
            raise ValueError(
                f"Requested time {timestamp} is not a published SamudrACE "
                f"initial-condition timestamp; valid timestamps are "
                f"{list(IC_TIMESTAMPS)}"
            )
        return timestamp

    def _open_component(self, timestamp: str, component: str) -> xr.Dataset:
        """Download and open one component's initial-condition dataset.

        Parameters
        ----------
        timestamp : str
            Initial-condition timestamp identifier.
        component : str
            Either ``"atmosphere"`` or ``"ocean"``.

        Returns
        -------
        xr.Dataset
            The component initial-condition dataset.
        """
        key = (timestamp, component)
        if key not in self._datasets:
            path = self._fetch_file(
                f"initial_conditions/{timestamp}/{component}/initial_condition.nc"
            )
            self._datasets[key] = xr.open_dataset(path, engine="netcdf4")
        return self._datasets[key]

    def _close_datasets(self) -> None:
        """Close the open initial-condition files and drop them."""
        for ds in self._datasets.values():
            ds.close()
        self._datasets = {}


class SamudrACEForcingData(_SamudrACEBase):
    """SamudrACE exogenous forcing data source.

    Provides the exogenous atmosphere forcing variables of the SamudrACE
    coupled emulator (e.g. downward shortwave radiation at the top of the
    atmosphere, land/lake fraction, surface height) from the
    ``allenai/SamudrACE-CM4-piControl`` HuggingFace repository, served with
    Earth2Studio variable names via the ``SamudrACELexicon``.

    Each forcing scenario NetCDF covers exactly one year of 6-hourly data on
    the no-leap CM4 model calendar (1460 timesteps). Time-varying fields are
    matched to requested times by month, day, and hour, ignoring the year, so
    the forcing series tiles annually for multi-year trajectories. Static
    fields are broadcast to every requested time. Requests that have no
    no-leap counterpart (February 29) or that fall off the 6-hourly grid
    raise an error.

    Parameters
    ----------
    scenario : str, optional
        Forcing scenario, either ``"0151"`` or ``"0311"``, by default
        ``"0311"``.
    cache : bool, optional
        Cache downloaded files locally (via the shared HuggingFace hub
        cache), by default True.
    verbose : bool, optional
        Log download progress, by default True.

    Warning
    -------
    Each forcing scenario is a NetCDF file of roughly 300 MB that is
    downloaded on demand.

    Note
    ----
    For more information see the following references:

    - SamudrACE paper: https://arxiv.org/abs/2509.12490
    - HuggingFace repo: https://huggingface.co/allenai/SamudrACE-CM4-piControl

    Badges
    ------
    region:global dataclass:simulation product:temp product:atmos
    """

    def __init__(
        self, scenario: str = "0311", cache: bool = True, verbose: bool = True
    ):
        super().__init__(cache=cache, verbose=verbose)
        if scenario not in FORCING_SCENARIOS:
            raise ValueError(
                f"scenario must be one of {list(FORCING_SCENARIOS)}, "
                f"got '{scenario}'"
            )
        self._scenario = scenario
        self._ds: xr.Dataset | None = None
        self._time_index: dict[tuple[int, int, int, int], int] = {}

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch forcing data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for. Time-varying fields are matched
            by month, day, and hour on the no-leap forcing calendar.
        variable : str | list[str] | VariableArray
            Earth2Studio variable names (see ``SamudrACELexicon``).

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]`` and
            latitude ordered north to south.
        """
        try:
            result = self._fetch_array(time, variable)
            # Materialize before the temporary downloads are removed below
            if not self._cache:
                result = result.load()
        finally:
            # Clean up on the error path too, so an interrupted download does
            # not leave a partial file behind
            if not self._cache:
                self._close_dataset()
                self._clean_up()

        return result

    def _fetch_array(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Assemble the requested forcing fields into one array.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Earth2Studio variable names.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``, lazy
            unless the caller loads it.
        """
        time_list = _prep_time_inputs(time)
        var_list = _prep_variable_inputs(variable)
        fme_list = _to_fme_names(var_list)

        ds = self._open_dataset()
        indices = [self._match_time(t) for t in time_list]

        fields = []
        for e2s_name, fme_name in zip(var_list, fme_list):
            if fme_name not in ds:
                raise KeyError(
                    f"Variable '{e2s_name}' (FME name '{fme_name}') not found "
                    f"in the SamudrACE forcing file for scenario "
                    f"'{self._scenario}'"
                )
            da = ds[fme_name]
            if "time" in da.dims:
                da = da.isel(time=indices).rename(time="window")
            else:
                da = da.expand_dims(window=len(time_list))
            fields.append(da.transpose("window", "lat", "lon"))

        result = xr.concat(fields, dim="variable").rename(window="time")
        result = result.assign_coords(
            time=np.array(time_list, dtype="datetime64[s]"),
            variable=np.array(var_list, dtype=object),
        )
        result = result.transpose("time", "variable", "lat", "lon")
        return _orient_north_to_south(result)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async wrapper for :meth:`__call__`.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Earth2Studio variable names.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        return self(time, variable)

    def _open_dataset(self) -> xr.Dataset:
        """Download and open the scenario forcing dataset.

        Returns
        -------
        xr.Dataset
            The forcing dataset, with a time index built for
            month/day/hour matching.
        """
        if self._ds is None:
            path = self._fetch_file(f"forcing_data/forcing_{self._scenario}.nc")
            self._ds = xr.open_dataset(
                path,
                engine="netcdf4",
                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            )
            self._time_index = {
                (t.month, t.day, t.hour, t.minute): i
                for i, t in enumerate(self._ds["time"].values)
            }
        return self._ds

    def _close_dataset(self) -> None:
        """Close the open forcing file and drop it."""
        if self._ds is not None:
            self._ds.close()
        self._ds = None
        self._time_index = {}

    def _match_time(self, time: datetime) -> int:
        """Match a requested time to the forcing time axis, ignoring year.

        Parameters
        ----------
        time : datetime
            Requested time.

        Returns
        -------
        int
            Index into the forcing time dimension.

        Raises
        ------
        ValueError
            If the time has no counterpart on the no-leap, 6-hourly forcing
            calendar (e.g. February 29 or an off-grid hour).
        """
        key = (time.month, time.day, time.hour, time.minute)
        if key not in self._time_index:
            raise ValueError(
                f"Requested time {time.isoformat()} has no counterpart on "
                f"the SamudrACE no-leap, 6-hourly forcing calendar "
                f"(scenario '{self._scenario}')"
            )
        return self._time_index[key]
