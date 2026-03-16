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

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import shutil
import time as _time_module
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, TypeVar
from urllib.parse import urlparse
from uuid import uuid4

import nest_asyncio
import netCDF4
import numpy as np
import pygrib
import requests
import xarray as xr
from loguru import logger
from tqdm import tqdm

from earth2studio.data import GOES
from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.base import LexiconType
from earth2studio.lexicon.planetary_computer import (
    PlanetaryComputerECMWFOpenDataIFSLexicon,
    PlanetaryComputerGOESLexicon,
    PlanetaryComputerMODISFireLexicon,
    PlanetaryComputerOISSTLexicon,
    PlanetaryComputerSentinel3AODLexicon,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import TimeArray, VariableArray

try:
    import httpx
    import planetary_computer
    import rioxarray
    from pystac import Item
    from pystac_client import Client
except ImportError:
    OptionalDependencyFailure("data")
    httpx = None
    Client = None
    planetary_computer = None
    rioxarray = None
    Item = TypeVar("Item")  # type: ignore


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Resolved variable request against a lexicon."""

    index: int
    variable_id: str
    dataset_key: str
    modifier: Callable[[Any], Any]


@dataclass(slots=True)
class AssetPlan:
    """Plan describing an asset download and the variables it satisfies."""

    unsigned_href: str
    signed_href: str
    media_type: str | None
    local_path: pathlib.Path
    variables: list[VariableSpec]


@check_optional_dependencies()
class _PlanetaryComputerData:
    """Generic Microsoft Planetary Computer data source.

    The base class handles STAC searches, concurrent asset downloads, and conversion of
    MPC assets into Earth2Studio's standardized xarray interface. Subclasses configure
    the collection parameters and typically only need to implement
    :meth:`extract_variable_numpy`, which receives the :class:`AssetPlan`, opens the
    cached data file, and adapts the raw fields into numpy arrays. Advanced MPC
    products can still override :meth:`_prepare_asset_plans` when they require custom
    asset selection logic, but most data sources work out-of-the-box once spatial
    metadata and the lexicon are provided.

    Parameters
    ----------
    collection_id : str
        STAC collection identifier to search on the Planetary Computer.
    lexicon : LexiconType
        Lexicon mapping requested variable names to dataset keys and modifiers.
    asset_key : str, optional
        Item asset key that contains the requested variables, by default "netcdf".
        The available asset keys are listed in the item-level assets table at the
        bottom of the Planetary Computer overview page of each respective dataset.
    search_kwargs : Mapping[str, Any] | None, optional
        Additional keyword arguments forwarded to ``Client.search``, by default None
    search_tolerance : datetime.timedelta, optional
        Maximum time delta when locating the closest STAC item to the request time,
        by default 12 hours.
    data_dtype: type, optional
        Numpy dtype for the data array, by default np.float32.
    spatial_dims : Mapping[str, numpy.ndarray]
        Mapping of spatial dimension names to coordinate arrays defining the grid, by
        default None
    data_attrs : Mapping[str, Any] | None, optional
        Extra attributes copied onto the output :class:`xarray.DataArray`, by default None
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Whether to print progress information, by default True
    max_workers : int, optional
        Upper bound on concurrent download and processing tasks, by default 24
    request_timeout : int, optional
        Timeout (seconds) applied to individual HTTP requests, by default 60
    max_retries : int, optional
        Maximum retry attempts for transient network failures, by default 4
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Notes
    -----
    More information on the Microsoft Planetary Computer is available at
    https://planetarycomputer.microsoft.com/.
    """

    STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    DEFAULT_TIMEOUT = 60
    DEFAULT_RETRIES = 4
    DEFAULT_ASYNC_TIMEOUT = 600
    CHUNK_SIZE = 1 << 20
    USER_AGENT = "earth2studio-planetary-computer"

    def __init__(
        self,
        collection_id: str,
        lexicon: LexiconType,
        asset_key: str = "netcdf",
        search_kwargs: Mapping[str, Any] | None = None,
        search_tolerance: timedelta = timedelta(hours=0),
        data_dtype: type = np.float32,
        spatial_dims: Mapping[str, np.ndarray] | None = None,
        data_attrs: Mapping[str, Any] | None = None,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
        async_timeout: int = DEFAULT_ASYNC_TIMEOUT,
    ) -> None:

        self._collection_id = collection_id
        self._asset_key = asset_key
        self._lexicon = lexicon
        self._search_kwargs = dict(search_kwargs or {})
        self._search_tolerance = search_tolerance
        self._data_dtype = data_dtype
        if not spatial_dims:
            raise ValueError("At least one spatial dimension must be provided.")
        self._spatial_dim_names = tuple(spatial_dims.keys())
        self._spatial_coords = {
            dim: np.asarray(values, dtype=np.float32)
            for dim, values in spatial_dims.items()
        }
        self._spatial_shape = tuple(
            len(self._spatial_coords[dim]) for dim in self._spatial_dim_names
        )
        self._data_attrs = dict(data_attrs or {})
        self._cache = cache
        self._verbose = verbose
        self._max_workers = max_workers
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._async_timeout = async_timeout

        self._client: Client | None = None

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
            return. Must be in the ARCO lexicon.

        Returns
        -------
        xr.DataArray
            Data array from planetary computer
        """
        nest_asyncio.apply()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            asyncio.wait_for(self.fetch(time, variable), timeout=self._async_timeout)
        )
        if not self._cache:
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
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in the data source's lexicon.

        Returns
        -------
        xr.DataArray
            Plantary computer data array
        """
        times, variables = prep_data_inputs(time, variable)

        # Make sure input time is valid
        self._validate_time(times)

        # Normalize times and resolve variables
        normalized_times = [
            (
                t.replace(tzinfo=timezone.utc)
                if t.tzinfo is None
                else t.astimezone(timezone.utc)
            )
            for t in times
        ]
        specs = []
        for index, var in enumerate(variables):
            dataset_key, modifier = self._lexicon[var]
            specs.append(
                VariableSpec(
                    index=index,
                    variable_id=var,
                    dataset_key=dataset_key,
                    modifier=modifier,
                )
            )

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Create DataArray with appropriate dimensions
        coords: dict[str, Any] = {
            "time": np.array([np.datetime64(t) for t in normalized_times]),
            "variable": list(variables),
        }
        for dim_name in self._spatial_dim_names:
            coords[dim_name] = self._spatial_coords[dim_name]
        xr_array = xr.DataArray(
            data=np.zeros(
                (len(times), len(variables), *self._spatial_shape),
                dtype=self._data_dtype,
            ),
            dims=["time", "variable", *self._spatial_dim_names],
            coords=coords,
            attrs=dict(self._data_attrs),
        )

        # Create download tasks
        # Use a fancy tqdm progress bar too
        timeout = httpx.Timeout(self._request_timeout)
        limits = httpx.Limits(max_connections=self._max_workers)
        transport = httpx.AsyncHTTPTransport(
            limits=limits,
            retries=self._max_retries,
        )
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            semaphore = asyncio.Semaphore(self._max_workers)
            with tqdm(
                total=len(times) * len(variables),
                disable=not self._verbose,
                desc=f"Fetching msft-pc {self._collection_id}",
            ) as progress:
                tasks = [
                    asyncio.create_task(
                        self._fetch_data(
                            client=client,
                            semaphore=semaphore,
                            requested_time=normalized_times[index],
                            variables=specs,
                            xr_array=xr_array,
                            time_index=index,
                            progress=progress,
                        )
                    )
                    for index in range(len(times))
                ]
                if tasks:
                    await asyncio.gather(*tasks)

        return xr_array

    async def _fetch_data(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        requested_time: datetime,
        variables: Sequence[VariableSpec],
        xr_array: xr.DataArray,
        time_index: int,
        progress: tqdm,
    ) -> None:
        """Download all variables for a timestamp and assign them into the output
        data array.

        Parameters
        ----------
        client : httpx.AsyncClient
            Shared HTTP client used for asset requests.
        semaphore : asyncio.Semaphore
            Semaphore limiting concurrent download tasks.
        requested_time : datetime
            Timestamp currently being processed.
        variables : Sequence[VariableSpec]
            Variable specifications for the request.
        xr_array : xarray.DataArray
            Mutable output array receiving the stacked data.
        time_index : int
            Index into ``xr_array`` corresponding to ``requested_time``.
        progress : tqdm.tqdm
            Progress bar used to report completion.
        """
        # Find the closest STAC item within the configured tolerance.
        item = await asyncio.to_thread(self._locate_item, requested_time)
        # Map requested variables to the assets that serve them.
        asset_plans = self._prepare_asset_plans(item, variables)

        # Download any assets that are not yet cached locally.
        download_tasks = [
            self._downloaded_asset(client, semaphore, plan)
            for plan in asset_plans
            if not plan.local_path.exists()
        ]
        if download_tasks:
            await asyncio.gather(*download_tasks)

        # Extract each variable from its source asset and record completion.
        for plan in asset_plans:
            for spec in plan.variables:
                array = self.extract_variable_numpy(plan, spec, requested_time)
                xr_array[time_index, spec.index] = array

        progress.update(len(variables))

    def extract_variable_numpy(
        self,
        plan: AssetPlan,
        spec: VariableSpec,
        _target_time: datetime,
    ) -> np.ndarray:
        """Convert an asset payload into a numpy array for a requested variable. Should
        be implemented in sub-class

        Parameters
        ----------
        plan : AssetPlan
            Plan describing the cached asset that should be opened.
        spec : VariableSpec
            Variable specification including dataset key, modifier, and output index.
        target_time : datetime
            Timestamp associated with the current request, useful for disambiguation.

        Returns
        -------
        np.ndarray
            Numpy array shaped to match the configured spatial dimensions.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _prepare_asset_plans(
        self,
        item: Item,
        variables: Sequence[VariableSpec],
    ) -> list[AssetPlan]:
        """Create download plans for the STAC item that fulfil the requested variables.

        Parameters
        ----------
        item : pystac.Item
            STAC item returned by :meth:`_locate_item`, potentially carrying
            collection-specific metadata.
        variables : Sequence[VariableSpec]
            Variable specifications resolved from the lexicon for the current request.

        Returns
        -------
        list[AssetPlan]
            Plans describing which assets to download and which variables they satisfy.

        Notes
        -----
        The produced plans drive the remainder of the workflow: :meth:`_fetch_array`
        iterates over them, :meth:`_downloaded_asset` fetches cache misses, and
        :meth:`extract_variable_numpy` ultimately materializes the requested numpy
        arrays.

        This method might be overridden for collections that distribute variables across
        multiple files (for example, per-band HDF subsets). Currently, this method is
        not overridden for any collections though.
        """
        asset_key = self._asset_key
        if asset_key not in item.assets:
            raise KeyError(f"Asset '{asset_key}' not available in item {item.id}")
        asset = item.assets[asset_key]
        signed_asset = planetary_computer.sign(asset)  # type: ignore[union-attr]
        unsigned_href = asset.href
        local_path = self._local_asset_path(unsigned_href)
        return [
            AssetPlan(
                unsigned_href=unsigned_href,
                signed_href=signed_asset.href,
                media_type=asset.media_type,
                local_path=local_path,
                variables=list(variables),
            )
        ]

    async def _downloaded_asset(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        plan: AssetPlan,
    ) -> None:
        """Download asset from remote source"""
        # Ensure cache directories exist before writing any temp files.
        plan.local_path.parent.mkdir(parents=True, exist_ok=True)

        async with semaphore:
            temp_path = plan.local_path.with_suffix(".tmp")
            try:
                # Stream asset contents to a temporary file to guard against partial writes.
                async with client.stream(
                    "GET",
                    plan.signed_href,
                    headers={"User-Agent": self.USER_AGENT},
                ) as response:
                    response.raise_for_status()
                    with temp_path.open("wb") as file_handle:
                        async for chunk in response.aiter_bytes(self.CHUNK_SIZE):
                            if chunk:
                                file_handle.write(chunk)
                temp_path.replace(plan.local_path)
            except Exception as error:
                # Clean up partial downloads so future retries start cleanly.
                temp_path.unlink(missing_ok=True)
                plan.local_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to download asset {plan.signed_href}"
                ) from error

    def _locate_item(self, when: datetime) -> Item:
        """Locate the closest STAC item to ``when`` within the configured tolerance."""
        # Ensure the client is initialized
        if self._client is None:
            self._client = Client.open(self.STAC_API_URL)

        # Build a closed interval around the requested timestamp to search for items.
        if self._search_tolerance.total_seconds() > 0:
            start = (when - self._search_tolerance).isoformat()
            end = (when + self._search_tolerance).isoformat()
            datetime_param = f"{start}/{end}"
        else:
            datetime_param = when.isoformat()

        # Perform the search
        search = self._client.search(
            collections=[self._collection_id],
            datetime=datetime_param,
            **self._get_search_kwargs(),
        )

        # Return the first item
        items: list[Item] = list(search.items())
        if len(items) == 0:
            raise FileNotFoundError(
                f"No Planetary Computer item found for {when.isoformat()} "
                f"within ±{self._search_tolerance}"
            )
        return self._select_item(items, when)

    def _select_item(self, items: list[Item], when: datetime) -> Item:
        """Simply return the first item."""
        # Many but not all data sources have item.properties["datetime"], which can be used
        # for selection. OISST only has 'start_datetime' and 'end_datetime', for example.
        if len(items) > 1:
            logger.warning("Found more than one matching item, returning first match")
        return items[0]

    def _local_asset_path(self, href: str) -> pathlib.Path:
        """Resolve the cache path for a remote asset href."""
        # Use a hashed filename so long URLs map to stable cache entries.
        parsed = urlparse(href)
        suffix = pathlib.Path(parsed.path).suffix or ""
        filename = hashlib.sha256(parsed.path.encode()).hexdigest() + suffix
        return pathlib.Path(self.cache) / filename

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify all times are valid based on offline knowledge.
        The child class should override this method as needed.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        """
        pass

    def _get_search_kwargs(self) -> dict:
        """Get the asset search parameters for the PySTAC client."""
        return self._search_kwargs

    @property
    def cache(self) -> str:
        """Return appropriate cache location."""
        cache_root = os.path.join(datasource_cache_root(), "planetary_computer")
        if not self._cache:
            cache_root = os.path.join(cache_root, "tmp_planetary_computer")
        return cache_root


class PlanetaryComputerOISST(_PlanetaryComputerData):
    """Daily 0.25° NOAA Optimum Interpolation SST from Microsoft Planetary Computer.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Whether to print progress information, by default True
    max_workers : int, optional
        Upper bound on concurrent download and processing tasks, by default 24
    request_timeout : int, optional
        Timeout (seconds) applied to individual HTTP requests, by default 60
    max_retries : int, optional
        Maximum retry attempts for transient network failures, by default 4
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://planetarycomputer.microsoft.com/dataset/noaa-cdr-sea-surface-temperature-optimum-interpolation
    """

    COLLECTION_ID = "noaa-cdr-sea-surface-temperature-optimum-interpolation"
    ASSET_KEY = "netcdf"
    SEARCH_TOLERANCE = timedelta(hours=12)
    LAT_COORDS = np.linspace(-89.875, 89.875, 720, dtype=np.float32)
    LON_COORDS = np.linspace(0.125, 359.875, 1440, dtype=np.float32)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = _PlanetaryComputerData.DEFAULT_TIMEOUT,
        max_retries: int = _PlanetaryComputerData.DEFAULT_RETRIES,
        async_timeout: int = _PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT,
    ) -> None:
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.ASSET_KEY,
            lexicon=PlanetaryComputerOISSTLexicon,
            search_kwargs=None,
            search_tolerance=self.SEARCH_TOLERANCE,
            spatial_dims={
                "lat": self.LAT_COORDS,
                "lon": self.LON_COORDS,
            },
            cache=cache,
            verbose=verbose,
            max_workers=max_workers,
            request_timeout=request_timeout,
            max_retries=max_retries,
            async_timeout=async_timeout,
        )

    def extract_variable_numpy(
        self,
        plan: AssetPlan,
        spec: VariableSpec,
        _target_time: datetime,
    ) -> np.ndarray:
        """Extract an OISST variable as a numpy array.

        Parameters
        ----------
        plan : AssetPlan
            Plan containing the cached MPC asset to be opened.
        spec : VariableSpec
            Lexicon specification describing which field to read and modifier to apply.
        Returns
        -------
        numpy.ndarray
            Float32 array containing the requested OISST field.
        """
        with xr.open_dataset(plan.local_path, engine="h5netcdf") as dataset:
            field = dataset[spec.dataset_key].isel(time=0, zlev=0)
            values = np.asarray(field.values).astype(np.float32)
            result = np.asarray(spec.modifier(values), dtype=np.float32)
            return result


class PlanetaryComputerSentinel3AOD(_PlanetaryComputerData):
    """Sentinel-3 SYNERGY Level-2 aerosol optical depth and surface reflectance.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Whether to print progress information, by default True
    max_workers : int, optional
        Upper bound on concurrent download and processing tasks, by default 24
    request_timeout : int, optional
        Timeout (seconds) applied to individual HTTP requests, by default 60
    max_retries : int, optional
        Maximum retry attempts for transient network failures, by default 4
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://planetarycomputer.microsoft.com/dataset/sentinel-3-synergy-aod-l2-netcdf
    """

    COLLECTION_ID = "sentinel-3-synergy-aod-l2-netcdf"
    ASSET_KEY = "ntc-aod"
    SEARCH_TOLERANCE = timedelta(hours=12)
    ROW_COORDS = np.arange(4040, dtype=np.float32)
    COLUMN_COORDS = np.arange(324, dtype=np.float32)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = _PlanetaryComputerData.DEFAULT_TIMEOUT,
        max_retries: int = _PlanetaryComputerData.DEFAULT_RETRIES,
        async_timeout: int = _PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT,
    ) -> None:
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.ASSET_KEY,
            lexicon=PlanetaryComputerSentinel3AODLexicon,
            search_kwargs=None,
            search_tolerance=self.SEARCH_TOLERANCE,
            spatial_dims={
                "y": self.ROW_COORDS,
                "x": self.COLUMN_COORDS,
            },
            cache=cache,
            verbose=verbose,
            max_workers=max_workers,
            request_timeout=request_timeout,
            max_retries=max_retries,
            async_timeout=async_timeout,
        )

    def extract_variable_numpy(
        self,
        plan: AssetPlan,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        """Extract a Sentinel-3 field as a float32 numpy array.

        Parameters
        ----------
        plan : AssetPlan
            Plan describing the cached asset to open.
        spec : VariableSpec
            Variable specification detailing which field and modifier to apply.
        Returns
        -------
        numpy.ndarray
            Array shaped to ``(len(self.ROW_COORDS), len(self.COLUMN_COORDS))``.
        """
        with xr.open_dataset(plan.local_path, engine="h5netcdf") as dataset:
            field = dataset[spec.dataset_key]
            values = np.asarray(field.values).astype(np.float32)
            result = np.asarray(spec.modifier(values), dtype=np.float32)

        expected_shape = (len(self.ROW_COORDS), len(self.COLUMN_COORDS))

        # Data is sometimes 4000 x 324, but expected is 4040 x 324
        # So we need to pad the data to the expected shape with NaNs
        if result.shape != expected_shape:
            padded = np.full(expected_shape, np.nan, dtype=np.float32)
            rows = min(expected_shape[0], result.shape[0])
            cols = min(expected_shape[1], result.shape[1])
            padded[:rows, :cols] = result[:rows, :cols]
            return padded

        return result


class PlanetaryComputerMODISFire(_PlanetaryComputerData):
    """MODIS Thermal Anomalies/Fire Daily (FireMask, MaxFRP, QA).

    Parameters
    ----------
    tile : str
        The MODIS tile identifier (``hXXvYY``) to prioritize during STAC searches.
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Whether to print progress information, by default True
    max_workers : int, optional
        Upper bound on concurrent download and processing tasks, by default 24
    request_timeout : int, optional
        Timeout (seconds) applied to individual HTTP requests, by default 60
    max_retries : int, optional
        Maximum retry attempts for transient network failures, by default 4
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://planetarycomputer.microsoft.com/dataset/modis-14A1-061
    - https://modis-land.gsfc.nasa.gov/MODLAND_grid.html

    Warning
    -------
    Tile searches are best-effort. If no tile identifiers are provided (the default),
    the first available tile returned by the Planetary Computer search is used.
    """

    COLLECTION_ID = "modis-14A1-061"
    ASSET_KEY = "FireMask"
    SEARCH_TOLERANCE = timedelta(hours=12)
    TILE_SIZE = 1200
    PIXEL_SIZE_M = 926.625433138
    TILE_WIDTH_M = TILE_SIZE * PIXEL_SIZE_M
    SIN_EARTH_RADIUS = 6371007.181
    SIN_MIN_X = -20015109.354
    SIN_MAX_Y = 10007554.677
    VARIABLE_ASSETS = {
        "fire_mask": "FireMask",
        "max_frp": "MaxFRP",
        "qa": "QA",
    }

    def __init__(
        self,
        tile: str = "h35v10",
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = _PlanetaryComputerData.DEFAULT_TIMEOUT,
        max_retries: int = _PlanetaryComputerData.DEFAULT_RETRIES,
        async_timeout: int = _PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT,
    ) -> None:
        tile = tile.lower()
        tile_filter = {
            "filter": {
                "op": "iLike",
                "args": [
                    {"property": "id"},
                    f"%{tile}%",
                ],
            }
        }
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.ASSET_KEY,
            lexicon=PlanetaryComputerMODISFireLexicon,
            search_kwargs=tile_filter,
            search_tolerance=self.SEARCH_TOLERANCE,
            spatial_dims={
                "y": np.arange(self.TILE_SIZE, dtype=np.float32),
                "x": np.arange(self.TILE_SIZE, dtype=np.float32),
            },
            cache=cache,
            verbose=verbose,
            max_workers=max_workers,
            request_timeout=request_timeout,
            max_retries=max_retries,
            async_timeout=async_timeout,
        )

    @classmethod
    def grid(cls, tile: str = "h35v10") -> tuple[np.ndarray, np.ndarray]:
        """Return latitude/longitude grids (degrees) for the specified MODIS tile.

        Parameters
        ----------
        tile : str, optional
            MODIS sinusoidal tile identifier (``hXXvYY``) describing the desired grid,
            by default h35v10

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Latitude and longitude arrays shaped ``(1200, 1200)`` in degrees.
        """
        tile = tile.lower()
        if len(tile) != 6 or not tile.startswith("h") or tile[3] != "v":
            raise ValueError(
                "MODIS tile identifiers must look like hXXvYY (e.g., h35v10)."
            )
        h_idx = int(tile[1:3])
        v_idx = int(tile[4:])
        cols = (np.arange(cls.TILE_SIZE, dtype=np.float64) + 0.5) * cls.PIXEL_SIZE_M
        rows = (np.arange(cls.TILE_SIZE, dtype=np.float64) + 0.5) * cls.PIXEL_SIZE_M
        x0 = cls.SIN_MIN_X + h_idx * cls.TILE_WIDTH_M
        y0 = cls.SIN_MAX_Y - v_idx * cls.TILE_WIDTH_M
        x = np.broadcast_to(x0 + cols, (cls.TILE_SIZE, cls.TILE_SIZE))
        y = np.broadcast_to((y0 - rows)[:, None], (cls.TILE_SIZE, cls.TILE_SIZE))
        phi = y / cls.SIN_EARTH_RADIUS
        lat = np.degrees(phi)
        cos_phi = np.cos(phi)
        safe_cos = np.where(np.abs(cos_phi) < 1e-12, np.nan, cos_phi)
        lon = np.degrees(x / (cls.SIN_EARTH_RADIUS * safe_cos))
        return lat.astype(np.float32), lon.astype(np.float32)

    def extract_variable_numpy(
        self,
        plan: AssetPlan,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        """Extract a MODIS Fire variable for the requested day-of-year band.

        Parameters
        ----------
        plan : AssetPlan
            Plan describing the cached MODIS tile asset to open.
        spec : VariableSpec
            Variable specification describing which band to read and modifier to apply.
        target_time : datetime
            Timestamp used to determine the correct ``band`` index within the asset.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(1200, 1200)`` for the requested field.

        Raises
        ------
        ValueError
            If the requested date cannot be located in the MODIS asset metadata.
        """
        with rioxarray.open_rasterio(plan.local_path) as data_array:
            # Get "band" index for the target date
            day_text = data_array.attrs.get("DAYSOFYEAR")
            band_idx = -1
            target_date = target_time.date()
            for idx, token in enumerate(day_text.split(",")):
                token = token.strip()
                if datetime.fromisoformat(token).date() == target_date:
                    band_idx = idx
                    break

            # If no band found, raise an error
            # This should never happen, but just in case for debugging
            if band_idx == -1:
                raise ValueError(f"Date not found in {target_time.date()}")

            # Get data and apply modifier
            field = data_array.isel(band=band_idx, drop=True)
            values = np.asarray(field.values).astype(np.float32)
            result = np.asarray(spec.modifier(values), dtype=np.float32)
            return result


class PlanetaryComputerECMWFOpenDataIFS(_PlanetaryComputerData):
    """IFS analysis data from the ECMWF Open Data repository.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Whether to print progress information, by default True
    max_workers : int, optional
        Upper bound on concurrent download and processing tasks, by default 24
    request_timeout : int, optional
        Timeout (seconds) applied to individual HTTP requests, by default 60
    max_retries : int, optional
        Maximum retry attempts for transient network failures, by default 4
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://planetarycomputer.microsoft.com/dataset/ecmwf-forecast
    """

    COLLECTION_ID = "ecmwf-forecast"
    ASSET_KEY = "data"
    SEARCH_KWARGS = {
        "query": {
            "ecmwf:stream": {"in": ["oper", "scda"]},
            "ecmwf:type": {"eq": "fc"},
            "ecmwf:step": {"eq": "0h"},
            "ecmwf:resolution": {"eq": "0.25"},
        },
    }
    LATITUDE = np.linspace(90, -90, 721)
    LONGITUDE = np.linspace(0, 360, 1440, endpoint=False)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = _PlanetaryComputerData.DEFAULT_TIMEOUT,
        max_retries: int = _PlanetaryComputerData.DEFAULT_RETRIES,
        async_timeout: int = _PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT,
    ) -> None:
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.ASSET_KEY,
            lexicon=PlanetaryComputerECMWFOpenDataIFSLexicon,
            search_kwargs=self.SEARCH_KWARGS,
            data_dtype=np.float64,
            spatial_dims={
                "lat": self.LATITUDE,
                "lon": self.LONGITUDE,
            },
            cache=cache,
            verbose=verbose,
            max_workers=max_workers,
            request_timeout=request_timeout,
            max_retries=max_retries,
            async_timeout=async_timeout,
        )

    def extract_variable_numpy(
        self,
        plan: AssetPlan,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        """Extract an ECMWF Open Data field as a float32 numpy array.

        Parameters
        ----------
        plan : AssetPlan
            Plan describing the cached asset to open.
        spec : VariableSpec
            Variable specification detailing which field and modifier to apply.
        Returns
        -------
        numpy.ndarray
            Array shaped ``(721, 1440)``.
        """
        var, plev, lay = spec.dataset_key.split("::")
        gsel: dict[str, Any] = {"shortName": var}
        if plev:
            gsel["typeOfLevel"] = "isobaricInhPa"
            gsel["level"] = float(plev)
        if lay:
            gsel["typeOfLevel"] = "soilLayer"
            gsel["level"] = float(lay)
        try:
            grbidx = pygrib.index(str(plan.local_path), *list(gsel))
        except Exception as e:
            logger.error(f"Failed to open GRIB file {plan.local_path}")
            raise e
        try:
            selection = grbidx.select(**gsel)
            if len(selection) > 1:
                raise Exception("Selection contains more than one GRIB element")
            values = selection[0].values
            # Roll to prime meridian
            values = np.roll(values, -len(self.LONGITUDE) // 2, -1)
            values = spec.modifier(values)
        except Exception as e:
            logger.error(f"Failed to read GRIB file {plan.local_path}")
            raise e
        finally:
            grbidx.close()
        return values

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify all times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        """
        MIN_TIME = datetime(2024, 3, 1)
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 21600 == 0:
                raise ValueError(
                    f"Requested start time {time} needs to be 6-hour interval for IFS"
                )

            if time < MIN_TIME:
                raise ValueError(
                    f"Requested start time {time} needs to be at least {MIN_TIME} for IFS"
                )


class PlanetaryComputerGOES(_PlanetaryComputerData):
    """GOES-R ABI L2 Cloud and Moisture Imagery on Planetary Computer.

    Parameters
    ----------
    satellite : str, optional
        Which GOES satellite to use ('goes16', 'goes17', 'goes18', or 'goes19'), by default 'goes16'
    scan_mode : str, optional
        For ABI: Scan mode ('F' for Full Disk, 'C' for Continental US)
        Mesoscale data is currently not supported due to the changing scan position.
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Whether to print progress information, by default True
    max_workers : int, optional
        Upper bound on concurrent download and processing tasks, by default 24
    request_timeout : int, optional
        Timeout (seconds) applied to individual HTTP requests, by default 60
    max_retries : int, optional
        Maximum retry attempts for transient network failures, by default 4
    async_timeout : int, optional
        Time in sec after which download will be cancelled if not finished successfully,
        by default 600

    Note
    ----
    Please see ``earth2studio.data.goes.GOES`` for further details.
    This data source exposes the MCMIP products but not the full-resolution CMIP products.
    Additional information on the data repository can be referenced here:

    - https://planetarycomputer.microsoft.com/dataset/goes-cmi
    """

    COLLECTION_ID = "goes-cmi"
    ASSET_KEY = "MCMIP-nc"

    def __init__(
        self,
        satellite: str = "goes16",
        scan_mode: str = "F",
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = _PlanetaryComputerData.DEFAULT_TIMEOUT,
        max_retries: int = _PlanetaryComputerData.DEFAULT_RETRIES,
        async_timeout: int = _PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT,
    ) -> None:
        GOES._validate_satellite_scan_mode(satellite, scan_mode)
        if scan_mode == "F":
            y, x = GOES.FULL_DISK_YX
        else:
            y, x = GOES.CONTINENTAL_US_YX[satellite]
        scan_freq = GOES.SCAN_TIME_FREQUENCY[scan_mode]
        if satellite == "goes17":
            logger.warning(
                "GOES-17 data on Planetary Computer is incomplete, "
                "consider using 'earth2studio.data.goes.GOES' instead"
            )
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.ASSET_KEY,
            lexicon=PlanetaryComputerGOESLexicon,
            search_kwargs=None,
            search_tolerance=timedelta(seconds=scan_freq),
            data_dtype=np.float64,
            spatial_dims={
                "y": y,
                "x": x,
            },
            cache=cache,
            verbose=verbose,
            max_workers=max_workers,
            request_timeout=request_timeout,
            max_retries=max_retries,
            async_timeout=async_timeout,
        )
        self._satellite = satellite
        self._scan_mode = scan_mode
        self._lat, self._lon = GOES.grid(satellite=satellite, scan_mode=scan_mode)

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
            return. Must be in the data source's lexicon.

        Returns
        -------
        xr.DataArray
            Plantary computer data array
        """
        xr_array = await super().fetch(time, variable)
        xr_array = xr_array.assign_coords(
            {"_lat": (("y", "x"), self._lat), "_lon": (("y", "x"), self._lon)}
        )
        return xr_array

    def _select_item(self, items: list[Item], when: datetime) -> Item:
        """Return the temporally closest item."""
        if len(items) > 1:
            logger.warning("Found more than one matching item, returning closest match")
        dts = [datetime.fromisoformat(item.properties["datetime"]) for item in items]
        idx = min(range(len(dts)), key=lambda i: abs((dts[i] - when).total_seconds()))
        return items[idx]

    def _get_search_kwargs(self) -> dict:
        # Remap __init__ args, which are aligned with other GOES data source
        image_type = "FULL DISK" if self._scan_mode == "F" else "CONUS"
        satellite = "GOES-" + self._satellite[-2:]
        return {
            "query": {
                "platform": {"eq": satellite},
                "goes:image-type": {"eq": image_type},
            },
        }

    def extract_variable_numpy(
        self,
        plan: AssetPlan,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        """Extract a GOES-R MCMIP field as a float32 numpy array.

        Parameters
        ----------
        plan : AssetPlan
            Plan describing the cached asset to open.
        spec : VariableSpec
            Variable specification detailing which field and modifier to apply.
        Returns
        -------
        numpy.ndarray
            Array shaped ``(1500, 2500)`` for 'CONUS' or ``(5424, 5424)`` for 'FULL DISK' scan mode.
        """

        with netCDF4.Dataset(plan.local_path, mode="r") as ds:
            values = ds[spec.dataset_key][:].filled(np.nan)
            values = spec.modifier(values)

        return values

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify all times are valid based on offline knowledge.

        Parameters
        ----------
        times : list[datetime]
            List of date times to fetch data for.
        """
        scan_freq = GOES.SCAN_TIME_FREQUENCY[self._scan_mode]
        for time in times:
            # Check scan frequency interval
            if not (time - datetime(1900, 1, 1)).total_seconds() % scan_freq == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be {scan_freq} second interval for GOES with scan mode {self._scan_mode}"
                )

            start_date, end_date = GOES.GOES_HISTORY_RANGE[self._satellite]
            if time < start_date:
                raise ValueError(
                    f"Requested date time {time} is before {self._satellite} became operational ({start_date})"
                )
            if end_date and time > end_date:
                raise ValueError(
                    f"Requested date time {time} is after {self._satellite} was retired ({end_date})"
                )


class GeoCatalogClient:
    """Client for ingesting STAC features into a Planetary Computer GeoCatalog.

    Workflow-specific behavior (templates, tile settings, render options, step size,
    start-time parameter key) is loaded from JSON files in the caller-provided config
    directory. This class is independent of concrete workflows.

    Expected files per workflow (workflow_name is passed by the caller):
    - parameters-{workflow_name}.json: step_size_hours, start_time_parameter_key
    - template-collection-{suffix}.json, template-feature-{suffix}.json
    - tile-settings-{suffix}.json, render-options-{suffix}.json

    Parameters
    ----------
    workflow_name : str
        Workflow name used in filenames (e.g. "fcn3", "fcn3-stormscope-goes").
    config_dir : str | pathlib.Path
        Directory containing the GeoCatalog JSON config and template files.
    """

    APPLICATION_URL = "https://geocatalog.spatio.azure.com/"
    REQUESTS_TIMEOUT = 30
    CREATION_TIMEOUT = 300

    def __init__(
        self,
        workflow_name: str,
        config_dir: str | pathlib.Path,
    ) -> None:
        try:
            from azure.identity import DefaultAzureCredential as _DefaultAzureCredential
        except ImportError as e:
            raise ImportError(
                "GeoCatalogClient requires 'azure-identity'. "
                "Install with the serve extra or pip install azure-identity."
            ) from e
        self._DefaultAzureCredential = _DefaultAzureCredential
        self._workflow_name = workflow_name
        self._config_dir = pathlib.Path(config_dir)
        self._parameters: dict[str, Any] = {}
        self._load_parameters()
        self.headers: dict | None = None

    def _load_parameters(self) -> None:
        path = self._config_dir / f"parameters-{self._workflow_name}.json"
        with open(path) as f:
            self._parameters = json.load(f)

    def update_headers(self) -> None:
        """Refresh the Authorization header using a new Azure credential token."""
        credential = self._DefaultAzureCredential()
        token = credential.get_token(self.APPLICATION_URL)
        self.headers = {"Authorization": f"Bearer {token.token}"}

    def _get(self, url: str) -> Any:
        return requests.get(
            url,
            headers=self.headers,
            params={"api-version": "2025-04-30-preview"},
            timeout=self.REQUESTS_TIMEOUT,
        )

    def _post(self, url: str, body: dict | None = None) -> Any:
        return requests.post(
            url,
            json=body,
            headers=self.headers,
            params={"api-version": "2025-04-30-preview"},
            timeout=self.REQUESTS_TIMEOUT,
        )

    def _put(self, url: str, body: dict | None = None) -> Any:
        return requests.put(
            url,
            json=body,
            headers=self.headers,
            params={"api-version": "2025-04-30-preview"},
            timeout=self.REQUESTS_TIMEOUT,
        )

    def _create_element(self, url: str, stac_config: dict) -> None:
        response = self._post(url, body=stac_config)
        location = response.headers["location"]
        log = logging.getLogger("planetary_computer.geocatalog")
        log.info("Creating '%s'...", stac_config["id"])
        start = perf_counter()
        while True:
            if (perf_counter() - start) > self.CREATION_TIMEOUT:
                log.error("Creation of '%s' timed out", stac_config["id"])
                return
            response = self._get(location)
            status = response.json()["status"]
            log.info(status)
            if status not in {"Pending", "Running"}:
                break
            _time_module.sleep(5)
        if status == "Succeeded":
            log.info("Successfully created '%s'", stac_config["id"])
        else:
            log.error("Failed to create '%s': %s", stac_config["id"], response.text)

    def _get_collection_json(self, collection_id: str | None) -> dict:
        path = self._config_dir / f"template-collection-{self._workflow_name}.json"
        with open(path) as f:
            stac_config = json.load(f)
        if collection_id is None:
            stac_config["id"] = stac_config["id"].format(uuid=uuid4())
        else:
            stac_config["id"] = collection_id
        return stac_config

    def _get_feature_json(
        self,
        start_time: datetime,
        end_time: datetime,
        blob_url: str,
    ) -> dict:
        path = self._config_dir / f"template-feature-{self._workflow_name}.json"
        with open(path) as f:
            stac_config = json.load(f)
        iso_start = start_time.isoformat()
        iso_end = end_time.isoformat()
        stac_config["id"] = stac_config["id"].format(
            start_time=iso_start[:13], uuid=uuid4()
        )
        stac_config["properties"]["datetime"] = iso_start
        stac_config["properties"]["start_datetime"] = iso_start
        stac_config["properties"]["end_datetime"] = iso_end
        stac_config["assets"]["data"]["href"] = blob_url
        stac_config["assets"]["data"]["description"] = stac_config["assets"]["data"][
            "description"
        ].format(start_time=iso_start, end_time=iso_end)
        return stac_config

    def _update_tile_settings(self, geocatalog_url: str, collection_id: str) -> None:
        path = self._config_dir / f"tile-settings-{self._workflow_name}.json"
        with open(path) as f:
            tile_settings = json.load(f)
        response = self._put(
            f"{geocatalog_url}/stac/collections/{collection_id}/configurations/tile-settings",
            body=tile_settings,
        )
        if response.status_code not in {200, 201}:
            log = logging.getLogger("planetary_computer.geocatalog")
            log.error(
                "Could not update tile settings: Error %s - %s",
                response.status_code,
                response.text,
            )

    def _update_render_options(self, geocatalog_url: str, collection_id: str) -> None:
        path = self._config_dir / f"render-options-{self._workflow_name}.json"
        with open(path) as f:
            render_params = json.load(f)
        log = logging.getLogger("planetary_computer.geocatalog")
        for params in render_params:
            render_option = {
                "id": f"auto-{params['id']}",
                "name": params["id"],
                "type": "raster-tile",
                "options": (
                    f"assets=data&subdataset_name={params['id']}"
                    "&sel=time=2100-01-01&sel=ensemble=0&sel_method=nearest"
                    f"&rescale={params['scale'][0]},{params['scale'][1]}"
                    f"&colormap_name={params['cmap']}"
                ),
                "minZoom": 0,
            }
            response = self._post(
                f"{geocatalog_url}/stac/collections/{collection_id}/configurations/render-options",
                body=render_option,
            )
            if response.status_code not in {200, 201}:
                log.error(
                    "Could not update render options: Error %s - %s",
                    response.status_code,
                    response.text,
                )

    def _create_collection(
        self,
        geocatalog_url: str,
        collection_id: str | None,
    ) -> str:
        stac_config = self._get_collection_json(collection_id)
        self._create_element(
            url=f"{geocatalog_url}/stac/collections",
            stac_config=stac_config,
        )
        self._update_tile_settings(geocatalog_url, stac_config["id"])
        self._update_render_options(geocatalog_url, stac_config["id"])
        return stac_config["id"]

    def _ensure_collection_exists(self, geocatalog_url: str, collection_id: str) -> str:
        response = self._get(f"{geocatalog_url}/stac/collections/{collection_id}")
        if response.status_code == 200:
            return collection_id
        if response.status_code != 404:
            raise RuntimeError(
                f"Failed to retrieve collection: Error {response.status_code} - {response.text}"
            )
        return self._create_collection(geocatalog_url, collection_id)

    def _resolve_start_time(self, parameters: dict) -> datetime:
        key = self._parameters["start_time_parameter_key"]
        raw = parameters.get(key)
        if raw is None:
            raise ValueError(
                f"Missing {key!r} in parameters for workflow {self._workflow_name!r}"
            )
        if isinstance(raw, str):
            normalized = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        if isinstance(raw, datetime):
            return raw
        if hasattr(raw, "isoformat"):
            return datetime.fromisoformat(raw.isoformat())
        raise TypeError(f"start time must be str or datetime, got {type(raw)}")

    def create_feature(
        self,
        geocatalog_url: str,
        collection_id: str | None,
        parameters: dict,
        blob_url: str,
    ) -> tuple[str, str]:
        """Ingest a new STAC feature into the collection."""
        self.update_headers()
        if collection_id is None:
            collection_id = self._create_collection(geocatalog_url, None)
        else:
            self._ensure_collection_exists(geocatalog_url, collection_id)
        start_time = self._resolve_start_time(parameters)
        step_hours = self._parameters["step_size_hours"]
        end_time = start_time + timedelta(hours=step_hours)
        stac_config = self._get_feature_json(start_time, end_time, blob_url)
        self._create_element(
            url=f"{geocatalog_url}/stac/collections/{collection_id}/items",
            stac_config=stac_config,
        )
        return collection_id, stac_config["id"]
