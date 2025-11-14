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

from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import nest_asyncio
import numpy as np
import xarray as xr
from tqdm import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.base import LexiconType
from earth2studio.lexicon.planetary_computer import (
    MODISFireLexicon,
    OISSTLexicon,
    Sentinel3AODLexicon,
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
    from pystac_client import Client
except ImportError:
    OptionalDependencyFailure("data")
    httpx = None
    Client = None
    planetary_computer = None
    rioxarray = None


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
        Item asset key that contains the requested variables, by default "netcdf"
    search_kwargs : Mapping[str, Any] | None, optional
        Additional keyword arguments forwarded to ``Client.search``, by default None
    search_tolerance : datetime.timedelta, optional
        Maximum time delta when locating the closest STAC item to the request time,
        by default 12 hours.
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
        search_tolerance: timedelta = timedelta(hours=12),
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
                (len(times), len(variables), *self._spatial_shape), dtype=np.float32
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
                desc=f"Fetching {self._collection_id}",
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

        # Allocate the [variable, spatial…] stack that will be populated below.
        data_stack = np.zeros((len(variables), *self._spatial_shape), dtype=np.float32)

        # Extract each variable from its source asset and record completion.
        for plan in asset_plans:
            for spec in plan.variables:
                array = self.extract_variable_numpy(plan, spec, requested_time)
                data_stack[spec.index] = array

        xr_array[time_index] = data_stack
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
        item: Any,
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

    def _locate_item(self, when: datetime) -> Any:
        """Locate the closest STAC item to ``when`` within the configured tolerance."""
        # Ensure the client is initialized
        if self._client is None:
            self._client = Client.open(self.STAC_API_URL)

        # Build a closed interval around the requested timestamp to search for items.
        start = (when - self._search_tolerance).isoformat()
        end = (when + self._search_tolerance).isoformat()
        datetime_param = f"{start}/{end}"

        # Perform the search
        search = self._client.search(
            collections=[self._collection_id],
            datetime=datetime_param,
            limit=1,
            **self._search_kwargs,
        )

        # Return the first item
        try:
            return next(search.items())
        except StopIteration as error:
            raise FileNotFoundError(
                f"No Planetary Computer item found for {when.isoformat()} "
                f"within ±{self._search_tolerance}"
            ) from error

    def _local_asset_path(self, href: str) -> pathlib.Path:
        """Resolve the cache path for a remote asset href."""
        # Use a hashed filename so long URLs map to stable cache entries.
        parsed = urlparse(href)
        suffix = pathlib.Path(parsed.path).suffix or ""
        filename = hashlib.sha256(parsed.path.encode()).hexdigest() + suffix
        return pathlib.Path(self.cache) / filename

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
            lexicon=OISSTLexicon,
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
            lexicon=Sentinel3AODLexicon,
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
            asset_key="FireMask",
            lexicon=MODISFireLexicon,
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
