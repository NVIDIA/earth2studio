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
import pathlib
import shutil
from datetime import datetime, timezone
from importlib.metadata import version

import fsspec
import gcsfs
import numpy as np
import xarray as xr
import zarr
from fsspec.implementations.cached import WholeFileCacheFileSystem

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.utils import handshake_dim
from earth2studio.utils.type import CoordSystem, TimeArray, VariableArray


class ARCORxBase:
    """Base class for ARCO (ERA5) prescriptive fields

    Parameters
    ----------
    id : str
        Earth-2 studio variable ID
    arco_id : str
        ARCO name of variable array in Zarr store
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    """

    ARCO_LAT = np.linspace(90, -90, 721)
    ARCO_LON = np.linspace(0, 359.75, 1440)

    def __init__(
        self,
        id: str,
        arco_id: str,
        cache: bool = True,
        verbose: bool = True,
    ):
        self.id = id
        self.arco_id = arco_id
        self._cache = cache
        self._verbose = verbose

        fs = gcsfs.GCSFileSystem(
            cache_timeout=-1,
            token="anon",  # noqa: S106 # nosec B106
            access="read_only",
            block_size=2**20,
        )

        if self._cache:
            cache_options = {
                "cache_storage": self.cache,
                "expiry_time": 31622400,  # 1 year
            }
            fs = WholeFileCacheFileSystem(fs=fs, **cache_options)

        # Check Zarr version and use appropriate method
        try:
            zarr_version = version("zarr")
            zarr_major_version = int(zarr_version.split(".")[0])
        except Exception:
            # Fallback to older method if version check fails
            zarr_major_version = 2  # Assume older version if we can't determine

        if zarr_major_version >= 3:
            # Zarr 3.0+ method
            zstore = zarr.storage.FsspecStore(
                fs,
                path="/gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
            )
            self.zarr_group = zarr.open(zstore, mode="r")
        else:
            # Legacy method for Zarr < 3.0
            # Use ARCO v2 over v3 if possible for faster loading (I think chunking is better in v2)
            fs_map = fsspec.FSMap(
                "gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
                fs,
            )
            self.zarr_group = zarr.open(fs_map, mode="r")

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch preciptive variable

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Preciptive variable will be repeated for each time.
        variable : str | list[str] | VariableArray
            Not relevant for this data-source

        Returns
        -------
        xr.DataArray
            Preciptive variable data array
        """
        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        lsmda = xr.DataArray(
            data=np.empty((len(time), 1, len(self.ARCO_LAT), len(self.ARCO_LON))),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": np.array([self.id]),
                "lat": self.ARCO_LAT,
                "lon": self.ARCO_LON,
            },
        )

        lsm_array = self.zarr_group[self.arco_id][:]
        lsm_array = np.repeat(
            lsm_array[np.newaxis, np.newaxis, :, :], len(time), axis=0
        )
        lsmda.values = lsm_array

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return lsmda

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), self.id)
        if not self._cache:
            cache_location = os.path.join(cache_location, f"tmp_{self.id}")
        return cache_location


class LandSeaMask(ARCORxBase):
    """Land-sea mask on a 0.25 degree equirectangular grid fetched from the ARCO v2
    datasource. This mask's cells are between [0,1] which denotes is the proportion of
    land, as opposed to ocean or inland waters (lakes, reservoirs, rivers and coastal
    waters), in that cell.

    Note
    ----
    For more information see:

    - https://codes.ecmwf.int/grib/param-db/172

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    """

    def __init__(self, cache: bool = True, verbose: bool = True):
        super().__init__(
            id="lsm",
            arco_id="land_sea_mask",
            cache=cache,
            verbose=verbose,
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray = "lsm",
    ) -> xr.DataArray:
        """Fetch ARCO land-sea mask

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Preciptive variable will be repeated for each time.
        variable : str | list[str] | VariableArray, optional
            Not relevant for this data-source, by default "lsm"

        Returns
        -------
        xr.DataArray
            Land-sea mask data array
        """
        return super().__call__(time, variable)


class SurfaceGeoPotential(ARCORxBase):
    """Surface geopotential on a 0.25 degree equirectangular grid fetched from the ARCO
    v2 datasource. This field is the gravitational potential energy of a unit mass
    relative to mean sea level. This is often referred to as the orography.

    Note
    ----
    For more information see:

    - https://codes.ecmwf.int/grib/param-db/129

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True
    """

    def __init__(self, cache: bool = True, verbose: bool = True):
        super().__init__(
            id="zsl",
            arco_id="geopotential_at_surface",
            cache=cache,
            verbose=verbose,
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray = "zsl",
    ) -> xr.DataArray:
        """Fetch ARCO orography field

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Preciptive variable will be repeated for each time.
        variable : str | list[str] | VariableArray, optional
            Not relevant for this data-source, by default "zsl"

        Returns
        -------
        xr.DataArray
            Orography data array
        """
        return super().__call__(time, variable)


class CosineSolarZenith:
    """Cosine of solar zenith angle. Use with caution needs verification.

    Note
    ----
    For more information see:

    - https://codes.ecmwf.int/grib/param-db/214001

    Parameters
    ----------
    domain_coords: OrderedDict[str, np.ndarray]
        Domain coordinates that the cos-zenith angle will be generated for, mush include
        [lat, lon] in the last two dimensions.
    """

    def __init__(self, domain_coords: CoordSystem):
        handshake_dim(domain_coords, "lat", -2)  # TODO: Generalize to any location
        handshake_dim(domain_coords, "lon", -1)
        self.domain_coords = domain_coords

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray = "",
    ) -> xr.DataArray:
        """Fetch Cosine zenith angle

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Times to generate cosine solar zenith angle
        variable : str | list[str] | VariableArray
            Not relevant for this data-source

        Returns
        -------
        xr.DataArray
            Cosine zenith angle data array
        """
        time, variable = prep_data_inputs(time, variable)
        try:
            from physicsnemo.utils.zenith_angle import cos_zenith_angle_from_timestamp
        except ImportError:
            raise ImportError(
                "nvidia-physicsnemo is required for this data source, which is not installed"
            )

        # For some reason physicsnemo function only works with float values
        # Hope this is correct
        data = cos_zenith_angle_from_timestamp(
            # https://docs.python.org/3/library/datetime.html#datetime.datetime.timestamp
            np.array(
                [t0.replace(tzinfo=timezone.utc).timestamp() for t0 in time]
            ).astype("float")[:, None, None],
            self.domain_coords["lat"][None, :, None],
            self.domain_coords["lon"][None, None, :],
        )[:, None, ...]
        # Expand other domain dimensions
        coords = {"time": time, "variable": np.array(["uvcossza"])}
        for key, values in self.domain_coords.items():
            coords[key] = values
            if key == "lat" or key == "lon":
                continue
            data = np.repeat(data[:, :, np.newaxis], values.shape[0], axis=2)

        da = xr.DataArray(data=data, dims=list(coords), coords=coords)
        return da
