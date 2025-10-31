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

import os
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal

import numpy as np
import xarray as xr

from earth2studio.data.utils import prep_data_inputs
from earth2studio.lexicon.ace import ACELexicon
from earth2studio.models.auto import Package
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import TimeArray, VariableArray

try:
    # ACE2 uses F90 regular gaussian grid internally
    # https://confluence.ecmwf.int/display/OIFS/4.3+OpenIFS%3A+Horizontal+Resolution+and+Configurations
    # Compute gaussian grid latitudes using legendre polynomials
    from scipy.special import roots_legendre

    ACE_GRID_LAT = np.degrees(np.arcsin(roots_legendre(2 * 90)[0]))
    ACE_GRID_LON = np.linspace(0.5, 359.5, 4 * 90, endpoint=True)
except ImportError:
    OptionalDependencyFailure("ace2")
    ACE_GRID_LAT = None
    ACE_GRID_LON = None


@check_optional_dependencies("ace2")
class ACE2ERA5Data:
    """ACE2-ERA5 data source providing forcing or initial-conditions data.
    Files are downloaded on-demand and cached automatically, or loaded from a user-specified
    local directory when `local=True`. Data are served as-is; no transformations are applied.

    Provides all input variables described in the ACE2-ERA5 paper.

    Parameters
    ----------
    base_path : str
        HuggingFace repo ID or local directory path. Defaults to `allenai/ACE2-ERA5`.
    mode : str
        Either "forcing" or "initial_conditions". Controls which data tree and filenames are used.
        Defaults to "forcing".
    forcing_subdir : str, optional
        Subdirectory for forcing data. Defaults to `forcing_data` containing files `forcing_YYYY.nc`.
    ic_subdir : str, optional
        Subdirectory for initial conditions. Defaults to `initial_conditions` containing files `ic_YYYY.nc`.
    local : bool, optional
        If True, data are loaded from a local directory instead of downloaded from HuggingFace. Defaults to False.
    co2_fn : Callable[[Sequence[datetime]], np.ndarray] | None, optional
        Optional function returning CO2 concentration (ppm) for a given UTC datetime as a numpy array. Defaults to None.
        If provided, the global mean CO2 concentration from the source is ignored, and is computed using this function.
        The function must accept a list of datetimes as input, and return a numpy array of CO2 concentrations of the same length.

    References
    ----------
    - ACE2-ERA5 paper: https://arxiv.org/html/2411.11268v1
    """

    _IC_ALLOWED_YEARS = [1940, 1950, 1979, 2001, 2020]

    def __init__(
        self,
        base_path: str = "allenai/ACE2-ERA5",
        mode: Literal["forcing", "initial_conditions"] = "forcing",
        forcing_subdir: str = "forcing_data",
        ic_subdir: str = "initial_conditions",
        local: bool = False,
        co2_fn: Callable[[Sequence[datetime]], np.ndarray] | None = None,
    ):
        if mode not in ["forcing", "initial_conditions"]:
            raise ValueError("mode must be either 'forcing' or 'initial_conditions'")
        self._base_path = base_path
        self._mode = mode
        self._forcing_subdir = forcing_subdir
        self._ic_subdir = ic_subdir
        self._local = local
        self.lat = ACE_GRID_LAT
        self.lon = ACE_GRID_LON
        self._co2_fn = co2_fn

    def _validate_ic_times(self, time_list: list[datetime]) -> None:
        for t in time_list:
            if t.year not in self._IC_ALLOWED_YEARS:
                raise ValueError(
                    f"Initial condition time year {t.year} is not supported. Allowed years: {self._IC_ALLOWED_YEARS}"
                )
            if not (
                t.day == 1
                and t.hour == 0
                and t.minute == 0
                and t.second == 0
                and t.microsecond == 0
            ):
                raise ValueError(
                    "Initial condition times must be the first of each month at 00:00 UTC"
                )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        time_list, var_list_e2s = prep_data_inputs(time, variable)

        if self._mode == "initial_conditions":
            self._validate_ic_times(time_list)

        # Map requested Earth2Studio variable names to FME names present in files
        try:
            var_list_fme = [ACELexicon[v][0] for v in var_list_e2s]
        except KeyError as e:
            raise KeyError(f"Unknown ACE2ERA5 variable id: {e}") from e

        # Determine years to fetch
        years = sorted({t.year for t in time_list})
        paths: list[str] = []
        for y in years:
            if self._mode == "forcing":
                filename = f"{self._forcing_subdir}/forcing_{y}.nc"
                cache_name = "ace2era5"
            else:
                filename = f"{self._ic_subdir}/ic_{y}.nc"
                cache_name = "ace2era5"

            if self._local:
                path = os.path.join(self._base_path, filename)
            else:
                pkg = Package(
                    f"hf://{self._base_path}",
                    cache_options={
                        "cache_storage": Package.default_cache(cache_name),
                        "same_names": True,
                    },
                )
                path = pkg.resolve(filename)

            paths.append(path)

        # Open and concat across years
        dsets = [xr.open_dataset(p, engine="netcdf4") for p in paths]
        ds = xr.concat(dsets, dim="time") if len(dsets) > 1 else dsets[0]

        # Standardize lat/lon coord names
        if "latitude" in ds.coords or "longitude" in ds.coords:
            ds = ds.rename(
                {
                    k: v
                    for k, v in {"latitude": "lat", "longitude": "lon"}.items()
                    if k in ds.coords
                }
            )

        # Subset time and variables; select exact requested timestamps and order
        ds = ds.sel(time=time_list)
        ds = ds[var_list_fme]

        # Build output DataArray [time, variable, lat, lon] with E2S variable names order
        arrays = []
        for fme_name in var_list_fme:
            da = ds[fme_name]
            # Ensure dims ordered [time, lat, lon]
            if "time" in da.dims and "lat" in da.dims and "lon" in da.dims:
                da = da.transpose("time", "lat", "lon")
            elif "lat" in da.dims and "lon" in da.dims:
                da = (
                    da.expand_dims("time")
                    .assign_coords(time=time_list)
                    .transpose("time", "lat", "lon")
                )
            elif "time" in da.dims and self._mode == "forcing":
                # CO2 is time-only
                da = da.expand_dims({"lat": self.lat, "lon": self.lon})
                da = da.transpose("time", "lat", "lon")
                if self._co2_fn is not None:
                    co2_times = self._co2_fn(time_list)
                    da.values = co2_times[:, None, None] * np.ones(
                        (len(time_list), len(self.lat), len(self.lon)),
                        dtype=np.float32,
                    )
            else:
                raise ValueError(f"Unknown ACE2 variable dims: {da.dims}")
            arrays.append(da)
        stacked = xr.concat(arrays, dim="variable")
        stacked = stacked.assign_coords(variable=np.array(var_list_e2s, dtype=object))
        # Ensure canonical dim order
        stacked = stacked.transpose("time", "variable", "lat", "lon")

        # Use predefined lat/lon coords which are equivalent up to machine precision
        # Mitigates errors that would otherwise arise from checks in `handshake_coords`
        stacked = stacked.assign_coords(lat=self.lat, lon=self.lon)
        return stacked

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        return self(time, variable)
