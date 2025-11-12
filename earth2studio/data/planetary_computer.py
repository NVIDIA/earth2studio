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
import contextlib
import hashlib
import os
import pathlib
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
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

    key: str
    unsigned_href: str
    signed_href: str
    media_type: str | None
    local_path: pathlib.Path
    variables: list[VariableSpec]


@check_optional_dependencies()
class PlanetaryComputerData:
    """Generic Microsoft Planetary Computer data source.

    The base class handles STAC searches, concurrent asset downloads, and conversion of
    MPC assets into Earth2Studio's standardized xarray interface. Subclasses configure
    the collection parameters and can override helper hooks for bespoke behaviour. In
    most situations you only need to provide collection metadata, but advanced MPC
    products may require overriding :meth:`prepare_asset_plans`, :meth:`select_assets`,
    and :meth:`extract_variable_numpy`. Together these hooks control how STAC items are
    mapped to local cache entries, how MPC assets are grouped per request, and how raw
    files are decoded into :class:`numpy.ndarray` payloads ready for stacking.

    Parameters
    ----------
    collection_id : str
        STAC collection identifier to search on the Planetary Computer.
    lexicon : type[LexiconType]
        Lexicon mapping requested variable names to dataset keys and modifiers.
    asset_key : str, default="netcdf"
        Item asset key that contains the requested variables.
    search_kwargs : Mapping[str, Any] | None, optional
        Additional keyword arguments forwarded to ``Client.search``.
    search_tolerance : datetime.timedelta, default=12 hours
        Maximum time delta when locating the closest STAC item to the request time.
    time_coordinate : str, default="time"
        Name of the time dimension within returned :class:`xarray.DataArray` objects.
    spatial_dims : Mapping[str, numpy.ndarray]
        Mapping of spatial dimension names to coordinate arrays defining the grid.
    data_attrs : Mapping[str, Any] | None, optional
        Extra attributes copied onto the output :class:`xarray.DataArray`.
    cache : bool, default=True
        Whether to persist downloaded assets between calls.
    verbose : bool, default=True
        Controls progress reporting via :mod:`tqdm`.
    max_workers : int, default=24
        Upper bound on concurrent download and processing tasks.
    request_timeout : int, default=60
        Timeout (seconds) applied to individual HTTP requests.
    max_retries : int, default=4
        Maximum retry attempts for transient network failures.
    async_timeout : int, default=600
        Overall timeout (seconds) for the asynchronous ``fetch`` workflow.

    Notes
    -----
    More information on the Microsoft Planetary Computer is available at
    https://planetarycomputer.microsoft.com/.

    Currently only netcdf and geotiff data sources are supported.
    """

    STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    NETCDF_SUFFIXES = {".nc", ".nc4", ".cdf"}
    GEOTIFF_SUFFIXES = {".tif", ".tiff"}
    LEVEL_DIMS = {"zlev", "lev", "level", "depth"}
    OPTIONAL_SINGLETON_DIMS = {"band"}

    DEFAULT_TIMEOUT = 60
    DEFAULT_RETRIES = 4
    DEFAULT_ASYNC_TIMEOUT = 600
    CHUNK_SIZE = 1 << 20
    USER_AGENT = "earth2studio-planetary-computer"

    def __init__(
        self,
        collection_id: str,
        lexicon: type[LexiconType],
        asset_key: str = "netcdf",
        search_kwargs: Mapping[str, Any] | None = None,
        search_tolerance: timedelta = timedelta(hours=12),
        time_coordinate: str = "time",
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
        self._time_coordinate = time_coordinate
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
        """Fetch data synchronously, mirroring :class:`~earth2studio.data.base.DataSource`.

        Parameters
        ----------
        time : datetime or array-like
            Requested timestamps, either a scalar ``datetime`` or sequence-like object.
        variable : str or array-like
            One or more variable identifiers to resolve through the configured lexicon.

        Returns
        -------
        xarray.DataArray
            Array with dimensions ``(time, variable, spatial...)`` containing the
            stacked Planetary Computer fields.
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
        """Fetch data asynchronously with concurrency-friendly downloads.

        Parameters
        ----------
        time : datetime or array-like
            Requested timestamps, either a scalar ``datetime`` or sequence-like object.
        variable : str or array-like
            One or more variable identifiers to resolve via the lexicon.

        Returns
        -------
        xarray.DataArray
            Array containing the requested variables across the provided timestamps.
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
        retry_config: int | Any
        if hasattr(httpx, "Retry"):
            retry_config = httpx.Retry(
                max_attempts=self._max_retries,
                backoff_factor=1.0,
                allowed_methods={"GET"},
            )
        else:
            # httpx<0.29 exposes a simple integer `retries` knob.
            retry_config = self._max_retries
        transport = httpx.AsyncHTTPTransport(
            retries=retry_config,
            limits=limits,
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
                        self._fetch_wrapper(
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

    async def _fetch_wrapper(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        requested_time: datetime,
        variables: Sequence[VariableSpec],
        xr_array: xr.DataArray,
        time_index: int,
        progress: tqdm,
    ) -> None:
        """Download all variables for a timestamp and assign them into the output array.

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
        # Fetch the full variable stack for this timestamp and write it into the output.
        stacked = await self._fetch_array(client, semaphore, requested_time, variables)
        if stacked.shape != (len(variables), *self._spatial_shape):
            raise ValueError(
                f"Unexpected data shape {stacked.shape} for {requested_time.isoformat()} "
                f"in collection {self._collection_id}; expected {(len(variables), *self._spatial_shape)}."
            )
        xr_array[time_index] = stacked
        progress.update(len(variables))

    async def _fetch_array(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        requested_time: datetime,
        variables: Sequence[VariableSpec],
    ) -> np.ndarray:
        """Fetch all requested variables for a timestamp and return stacked numpy data.

        Parameters
        ----------
        client : httpx.AsyncClient
            HTTP client for downloading assets.
        semaphore : asyncio.Semaphore
            Concurrency guard for download tasks.
        requested_time : datetime
            Timestamp being serviced.
        variables : Sequence[VariableSpec]
            Variable specifications required for the timestamp.

        Returns
        -------
        numpy.ndarray
            Array shaped ``(len(variables), *spatial_shape)`` containing the data.
        """
        # Find the closest STAC item within the configured tolerance.
        item = await asyncio.to_thread(self._locate_item, requested_time)
        # Map requested variables to the assets that serve them.
        asset_plans = self.prepare_asset_plans(item, variables)

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
            with self._open_asset(plan.local_path, plan.media_type) as asset_data:
                for spec in plan.variables:
                    array = self.extract_variable_numpy(
                        asset_data, spec, requested_time
                    )
                    data_stack[spec.index] = array

        return data_stack

    # ------------------------------------------------------------------
    # Methods to override for MPC collections
    # * prepare_asset_plans
    # * select_assets
    # * extract_variable_numpy
    # --------------------------------------------------------
    def prepare_asset_plans(
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
        """
        # Build a mapping of asset key -> variables satisfied by that asset.
        asset_map = self.select_assets(item, variables)
        plans: list[AssetPlan] = []

        for asset_key, specs in asset_map.items():
            if asset_key not in item.assets:
                raise KeyError(f"Asset '{asset_key}' not available in item {item.id}")
            asset = item.assets[asset_key]
            # Planetary Computer returns unsigned HREFs; sign them before download.
            signed_asset = planetary_computer.sign(asset)  # type: ignore[union-attr]
            unsigned_href = asset.href
            local_path = self._local_asset_path(unsigned_href)
            plans.append(
                AssetPlan(
                    key=asset_key,
                    unsigned_href=unsigned_href,
                    signed_href=signed_asset.href,
                    media_type=asset.media_type,
                    local_path=local_path,
                    variables=list(specs),
                )
            )
        return plans

    def select_assets(
        self,
        item: Any,
        variables: Sequence[VariableSpec],
    ) -> Mapping[str, Sequence[VariableSpec]]:
        """Return a mapping between STAC asset keys and variable specifications.

        Parameters
        ----------
        item : pystac.Item
            STAC item that holds asset metadata for the requested timestamp.
        variables : Sequence[VariableSpec]
            Variable specifications that must be sourced from the item.

        Returns
        -------
        Mapping[str, Sequence[VariableSpec]]
            Dictionary keyed by asset identifiers with the variables each asset serves.

        Notes
        -----
        The default implementation assumes all variables reside in ``self._asset_key``.
        Override this method for collections that distribute variables across multiple
        files (for example, per-band HDF subsets).
        """
        return {self._asset_key: variables}

    def extract_variable_numpy(
        self,
        asset_data: xr.Dataset | xr.DataArray,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        """Convert an asset payload into a numpy array for a requested variable.

        Parameters
        ----------
        asset_data : xarray.Dataset or xarray.DataArray
            Opened asset contents containing the raw Planetary Computer fields.
        spec : VariableSpec
            Variable specification including dataset key, modifier, and output index.
        target_time : datetime
            Timestamp associated with the current request, useful for disambiguation.

        Returns
        -------
        numpy.ndarray
            Float32 array shaped to match the configured spatial dimensions.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    async def _downloaded_asset(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        plan: AssetPlan,
    ) -> None:
        """Ensure the asset described by ``plan`` is downloaded into the cache.

        Parameters
        ----------
        client : httpx.AsyncClient
            HTTP client used for streaming the asset.
        semaphore : asyncio.Semaphore
            Semaphore limiting concurrent transfers.
        plan : AssetPlan
            Download plan describing the remote/signed URLs and cache path.
        """
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

    def _locate_item(self, when: datetime):
        """Locate the closest STAC item to ``when`` within the configured tolerance.

        Parameters
        ----------
        when : datetime
            Target timestamp for which to retrieve data.

        Returns
        -------
        pystac.Item
            First STAC item returned by the search query.

        Raises
        ------
        FileNotFoundError
            If no item is found within ``self._search_tolerance``.
        """
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
        """Resolve the cache path for a remote asset href.

        Parameters
        ----------
        href : str
            Remote asset URL obtained from the STAC item.

        Returns
        -------
        pathlib.Path
            Path inside the datasource cache where the asset should be stored.
        """
        # Use a hashed filename so long URLs map to stable cache entries.
        parsed = urlparse(href)
        suffix = pathlib.Path(parsed.path).suffix or ""
        filename = hashlib.sha256(parsed.path.encode()).hexdigest() + suffix
        return pathlib.Path(self.cache) / filename

    @contextlib.contextmanager
    def _open_asset(
        self,
        local_path: pathlib.Path,
        media_type: str | None,
    ) -> Iterator[xr.Dataset | xr.DataArray]:
        """Open a cached asset as an :mod:`xarray` dataset or data array.

        Parameters
        ----------
        local_path : pathlib.Path
            Location of the cached asset on disk.
        media_type : str or None
            Optional media type hint from the STAC metadata.

        Returns
        -------
        Iterator[xarray.Dataset | xarray.DataArray]
            Context-managed iterator yielding the opened dataset/array.

        Raises
        ------
        NotImplementedError
            If the asset format is neither NetCDF nor GeoTIFF.
        """
        suffix = local_path.suffix.lower()

        # Prefer NetCDF when the suffix or media type indicates an HDF/NetCDF payload.
        if suffix in self.NETCDF_SUFFIXES or (
            media_type and "netcdf" in media_type.lower()
        ):
            dataset = xr.open_dataset(local_path, engine="h5netcdf")
            yield dataset
            dataset.close()
            return

        # Otherwise treat the asset as a GeoTIFF using rioxarray utilities.
        if suffix in self.GEOTIFF_SUFFIXES or (
            media_type and "tiff" in media_type.lower()
        ):
            data_array = rioxarray.open_rasterio(local_path)
            yield data_array
            data_array.close()
            return

        raise NotImplementedError(
            f"Unsupported asset format for file '{local_path}'. "
            "Only NetCDF and GeoTIFF assets are currently supported."
        )

    @property
    def cache(self) -> str:
        """Return cache root for Planetary Computer assets.

        Returns
        -------
        str
            Filesystem path used to store cached Planetary Computer assets.
        """
        cache_root = os.path.join(datasource_cache_root(), "planetary_computer")
        if not self._cache:
            cache_root = os.path.join(cache_root, "tmp")
        return cache_root


class PlanetaryComputerOISST(PlanetaryComputerData):
    """Daily 0.25° NOAA Optimum Interpolation SST from Microsoft Planetary Computer.

    Parameters
    ----------
    cache : bool, default=True
        Whether to persist downloaded assets between invocations.
    verbose : bool, default=True
        Enable progress bar output while fetching data.
    max_workers : int, default=24
        Maximum number of concurrent download/extraction tasks.
    request_timeout : int, default=PlanetaryComputerData.DEFAULT_TIMEOUT
        Timeout (seconds) for individual HTTP requests.
    max_retries : int, default=PlanetaryComputerData.DEFAULT_RETRIES
        Number of retry attempts for transient network failures.
    async_timeout : int, default=PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT
        Overall timeout (seconds) for the asynchronous fetch workflow.

    Notes
    -----
    Dataset reference: https://planetarycomputer.microsoft.com/dataset/noaa-cdr-sea-surface-temperature-optimum-interpolation
    """

    COLLECTION_ID = "noaa-cdr-sea-surface-temperature-optimum-interpolation"
    ASSET_KEY = "netcdf"
    TIME_COORDINATE = "time"
    SEARCH_TOLERANCE = timedelta(hours=12)
    LAT_COORDS = np.linspace(-89.875, 89.875, 720, dtype=np.float32)
    LON_COORDS = np.linspace(0.125, 359.875, 1440, dtype=np.float32)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = PlanetaryComputerData.DEFAULT_TIMEOUT,
        max_retries: int = PlanetaryComputerData.DEFAULT_RETRIES,
        async_timeout: int = PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT,
    ) -> None:
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.ASSET_KEY,
            lexicon=OISSTLexicon,
            search_kwargs=None,
            search_tolerance=self.SEARCH_TOLERANCE,
            time_coordinate=self.TIME_COORDINATE,
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
        asset_data: xr.Dataset | xr.DataArray,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        """Extract an OISST variable as a numpy array.

        Parameters
        ----------
        asset_data : xarray.Dataset or xarray.DataArray
            NetCDF payload opened from the cached MPC asset.
        spec : VariableSpec
            Lexicon specification describing which field to read and modifier to apply.
        target_time : datetime
            Timestamp associated with the current download (unused here but required).

        Returns
        -------
        numpy.ndarray
            Float32 array containing the requested OISST field.
        """
        field = asset_data[spec.dataset_key].isel(time=0, zlev=0)
        values = np.asarray(field.values).astype(np.float32)
        result = np.asarray(spec.modifier(values), dtype=np.float32)
        return result


class PlanetaryComputerSentinel3AOD(PlanetaryComputerData):
    """Sentinel-3 SYNERGY Level-2 aerosol optical depth and surface reflectance.

    Parameters
    ----------
    cache : bool, default=True
        Whether to persist downloaded assets locally.
    verbose : bool, default=True
        Enable progress reporting while fetching.
    max_workers : int, default=24
        Maximum concurrency for download/extraction tasks.
    request_timeout : int, default=PlanetaryComputerData.DEFAULT_TIMEOUT
        Timeout (seconds) for HTTP requests to the STAC API/assets.
    max_retries : int, default=PlanetaryComputerData.DEFAULT_RETRIES
        Number of retry attempts for transient failures.
    async_timeout : int, default=PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT
        Overall timeout (seconds) for the asynchronous fetch workflow.

    Notes
    -----
    Dataset reference: https://planetarycomputer.microsoft.com/dataset/sentinel-3-synergy-aod-l2-netcdf
    """

    COLLECTION_ID = "sentinel-3-synergy-aod-l2-netcdf"
    ASSET_KEY = "ntc-aod"
    TIME_COORDINATE = "time"
    SEARCH_TOLERANCE = timedelta(hours=12)
    ROW_COORDS = np.arange(4040, dtype=np.float32)
    COLUMN_COORDS = np.arange(324, dtype=np.float32)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = PlanetaryComputerData.DEFAULT_TIMEOUT,
        max_retries: int = PlanetaryComputerData.DEFAULT_RETRIES,
        async_timeout: int = PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT,
    ) -> None:
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.ASSET_KEY,
            lexicon=Sentinel3AODLexicon,
            search_kwargs=None,
            search_tolerance=self.SEARCH_TOLERANCE,
            time_coordinate=self.TIME_COORDINATE,
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
        asset_data: xr.Dataset | xr.DataArray,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        """Extract a Sentinel-3 field as a float32 numpy array.

        Parameters
        ----------
        asset_data : xarray.Dataset or xarray.DataArray
            Dataset returned by :func:`xarray.open_dataset`/``rioxarray``.
        spec : VariableSpec
            Variable specification detailing which field and modifier to apply.
        target_time : datetime
            Timestamp associated with the current request (unused).

        Returns
        -------
        numpy.ndarray
            Array shaped to ``(len(self.ROW_COORDS), len(self.COLUMN_COORDS))``.
        """
        field = asset_data[spec.dataset_key]
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


class PlanetaryComputerMODISFire(PlanetaryComputerData):
    """MODIS Thermal Anomalies/Fire Daily (FireMask, MaxFRP, QA).

    Parameters
    ----------
    cache : bool, default=True
        Whether to retain downloaded assets between calls.
    verbose : bool, default=True
        Enable progress reporting while downloading.
    max_workers : int, default=24
        Maximum concurrency for download and extraction tasks.
    request_timeout : int, default=PlanetaryComputerData.DEFAULT_TIMEOUT
        Timeout (seconds) for individual HTTP requests.
    max_retries : int, default=PlanetaryComputerData.DEFAULT_RETRIES
        Number of retry attempts for transient HTTP failures.
    async_timeout : int, default=PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT
        Overall timeout (seconds) for the asynchronous fetch workflow.

    Notes
    -----
    Specific tile selection (e.g., ``hXXvYY``) is not currently supported; the first
    available tile returned by the Planetary Computer search is used.

    Dataset reference: https://planetarycomputer.microsoft.com/dataset/modis-14A1-061
    """

    COLLECTION_ID = "modis-14A1-061"
    TIME_COORDINATE = "time"
    SEARCH_TOLERANCE = timedelta(hours=12)
    VARIABLE_ASSETS = {
        "fire_mask": "FireMask",
        "max_frp": "MaxFRP",
        "qa": "QA",
    }

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = PlanetaryComputerData.DEFAULT_TIMEOUT,
        max_retries: int = PlanetaryComputerData.DEFAULT_RETRIES,
        async_timeout: int = PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT,
    ) -> None:
        super().__init__(
            self.COLLECTION_ID,
            asset_key="FireMask",
            lexicon=MODISFireLexicon,
            search_kwargs=None,
            search_tolerance=self.SEARCH_TOLERANCE,
            time_coordinate=self.TIME_COORDINATE,
            spatial_dims={
                "y": np.arange(1200, dtype=np.float32),
                "x": np.arange(1200, dtype=np.float32),
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
        asset_data: xr.DataArray,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        """Extract a MODIS Fire variable for the requested day-of-year band.

        Parameters
        ----------
        asset_data : xarray.DataArray
            Rasterio-backed array representing the MODIS Fire tiled product.
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
        # Get "band" index for the target date
        day_text = asset_data.attrs.get("DAYSOFYEAR")
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
        field = asset_data.isel(band=band_idx, drop=True)
        values = np.asarray(field.values).astype(np.float32)
        result = np.asarray(spec.modifier(values), dtype=np.float32)
        return result
