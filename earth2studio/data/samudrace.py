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
import shutil
from datetime import datetime

import numpy as np
import xarray as xr
from huggingface_hub import HfFileSystem
from loguru import logger

from earth2studio.data.ace2 import ACE_GRID_LAT, ACE_GRID_LON
from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.samudrace import SamudrACELexicon
from earth2studio.utils.type import TimeArray, VariableArray

# Available initial-condition timestamps and their HF paths
_IC_TIMESTAMPS: dict[str, str] = {
    "0151-01-06T00:00:00": "0151-01-06T00:00:00",
    "0311-01-01T00:00:00": "0311-01-01T00:00:00",
    "0313-01-01T00:00:00": "0313-01-01T00:00:00",
    "0315-01-01T00:00:00": "0315-01-01T00:00:00",
    "0317-01-01T00:00:00": "0317-01-01T00:00:00",
    "0319-01-01T00:00:00": "0319-01-01T00:00:00",
}


class SamudrACEData:
    """SamudrACE initial-condition data source.

    Provides combined atmosphere and ocean initial-condition variables from
    the SamudrACE HuggingFace repository.  Each initial condition has separate
    atmosphere and ocean NetCDF files that are downloaded, merged, and served
    as a single ``xr.DataArray`` with all 118 prognostic variables (38
    atmosphere + 80 ocean).

    The SamudrACE checkpoint was trained on a GFDL CM4 preindustrial-control
    run.  Initial conditions are identified by their CM4 model-year timestamp
    (e.g. ``"0311-01-01T00:00:00"``).  Because the model-year dates do not
    correspond to real calendar dates, the ``ic_timestamp`` parameter selects
    which IC snapshot to serve; the data are then returned for whatever
    ``time`` is requested (i.e. the same spatial fields are recycled).

    Parameters
    ----------
    ic_timestamp : str, optional
        Initial-condition timestamp identifier.  Must be one of the available
        timestamps in the repository, by default ``"0311-01-01T00:00:00"``.
    cache : bool, optional
        Cache downloaded files locally, by default True.
    verbose : bool, optional
        Print download progress, by default True.

    References
    ----------
    - SamudrACE paper: https://arxiv.org/abs/2509.12490
    - HuggingFace repo: https://huggingface.co/allenai/SamudrACE-CM4-piControl

    Badges
    ------
    region:global dataclass:reanalysis product:temp product:atmos product:ocean
    """

    _HF_REPO_ID = "allenai/SamudrACE-CM4-piControl"

    def __init__(
        self,
        ic_timestamp: str = "0311-01-01T00:00:00",
        cache: bool = True,
        verbose: bool = True,
    ):
        if ic_timestamp not in _IC_TIMESTAMPS:
            raise ValueError(
                f"ic_timestamp must be one of {list(_IC_TIMESTAMPS.keys())}, "
                f"got '{ic_timestamp}'"
            )
        self._ic_timestamp = ic_timestamp
        self._cache = cache
        self._verbose = verbose
        self.lat = ACE_GRID_LAT
        self.lon = ACE_GRID_LON
        self._hf_fs = HfFileSystem()
        self._atm_ds: xr.Dataset | None = None
        self._ocean_ds: xr.Dataset | None = None

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch initial-condition data.

        The same spatial fields are returned for every requested time (the IC
        snapshot is time-invariant).  Variables are mapped from Earth2Studio
        names to FME names via the ``SamudrACELexicon``.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).  The spatial fields are
            identical for all times; only the time coordinate differs.
        variable : str | list[str] | VariableArray
            Earth2Studio variable names.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        time_list, var_list_e2s = prep_data_inputs(time, variable)

        # Map E2S → FME variable names
        try:
            var_list_fme = [SamudrACELexicon[v][0] for v in var_list_e2s]
        except KeyError as e:
            raise KeyError(f"Unknown SamudrACE variable id: {e}") from e

        return self._fetch_initial_conditions(time_list, var_list_e2s, var_list_fme)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async wrapper for :meth:`__call__`.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Earth2Studio variable names.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        return self(time, variable)

    @property
    def cache(self) -> str:
        """Local cache directory for downloaded files."""
        root = os.path.join(datasource_cache_root(), "SamudrACE")
        if not self._cache:
            root = os.path.join(root, "tmp_SamudrACE")
        return root

    def _download_ic_file(self, component: str) -> str:
        """Download a single IC file (atmosphere or ocean) and return local path.

        Parameters
        ----------
        component : str
            Either ``"atmosphere"`` or ``"ocean"``.

        Returns
        -------
        str
            Local filesystem path to the downloaded NetCDF file.
        """
        hf_dir = _IC_TIMESTAMPS[self._ic_timestamp]
        filename = f"initial_conditions/{hf_dir}/{component}/initial_condition.nc"
        local_path = os.path.join(self.cache, filename)
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            hf_path = f"{self._HF_REPO_ID}/{filename}"
            if self._verbose:
                logger.info("Downloading SamudrACE {} IC: {}", component, hf_path)
            self._hf_fs.get_file(hf_path, local_path)
        return local_path

    def _open_ic_datasets(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Lazily download and open atmosphere and ocean IC datasets."""
        if self._atm_ds is None:
            atm_path = self._download_ic_file("atmosphere")
            self._atm_ds = xr.open_dataset(atm_path, engine="netcdf4")
        if self._ocean_ds is None:
            ocean_path = self._download_ic_file("ocean")
            self._ocean_ds = xr.open_dataset(ocean_path, engine="netcdf4")
        return self._atm_ds, self._ocean_ds

    def _fetch_initial_conditions(
        self,
        time_list: list[datetime],
        var_list_e2s: list[str],
        var_list_fme: list[str],
    ) -> xr.DataArray:
        """Load atmosphere + ocean IC variables and merge into a single DataArray.

        Each IC file has a ``sample`` dimension of size 1.  The spatial fields
        are broadcast identically to every requested time.
        """
        atm_ds, ocean_ds = self._open_ic_datasets()

        arrays: list[xr.DataArray] = []
        for fme_name, e2s_name in zip(var_list_fme, var_list_e2s):
            # Look in atmosphere first, then ocean
            if fme_name in atm_ds:
                da = atm_ds[fme_name]
            elif fme_name in ocean_ds:
                da = ocean_ds[fme_name]
            else:
                raise KeyError(
                    f"Variable '{fme_name}' (E2S: '{e2s_name}') not found in "
                    f"SamudrACE IC files for timestamp '{self._ic_timestamp}'"
                )

            # IC files have dims (sample, lat, lon); squeeze sample
            if "sample" in da.dims:
                da = da.isel(sample=0)

            # Ensure (lat, lon) ordering
            if "lat" in da.dims and "lon" in da.dims:
                da = da.transpose("lat", "lon")

            # Assign canonical grid coordinates
            da = da.assign_coords(lat=self.lat, lon=self.lon)

            # Broadcast to every requested time
            per_time = [da for _ in time_list]

            stacked = xr.concat(per_time, dim="time")
            stacked = stacked.assign_coords(
                time=np.array(time_list, dtype="datetime64[ns]")
            )
            arrays.append(stacked)

        result = xr.concat(arrays, dim="variable")
        result = result.assign_coords(variable=np.array(var_list_e2s, dtype=object))
        result = result.transpose("time", "variable", "lat", "lon")

        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        return result
