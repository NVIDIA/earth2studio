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
from typing import Any, Type
from urllib.parse import urlparse

import nest_asyncio
import numpy as np
import xarray as xr
from tqdm import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.base import LexiconType
from earth2studio.lexicon.modis_fire import MODISFireLexicon
from earth2studio.lexicon.oisst import OISSTLexicon
from earth2studio.lexicon.sentinel3_aod import Sentinel3AODLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import TimeArray, VariableArray

try:  # pragma: no cover - exercised in integration tests
    import httpx
    import planetary_computer
    import rioxarray
    from pystac_client import Client
except ImportError:  # pragma: no cover - handled by optional dependency guard
    OptionalDependencyFailure("data")
    httpx = None  # type: ignore[assignment]
    Client = None  # type: ignore[assignment]
    planetary_computer = None  # type: ignore[assignment]
    rioxarray = None  # type: ignore[assignment]


Modifier = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Resolved variable request against a lexicon."""

    index: int
    variable_id: str
    dataset_key: str
    modifier: Modifier


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

    The base class handles STAC searches, concurrent asset downloads and conversion of
    MPC assets into Earth2Studio's standardized xarray interface. Subclasses configure
    the collection parameters and can override helper hooks for bespoke behaviour.
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
        *,
        lexicon: type[LexiconType],
        asset_key: str = "netcdf",
        search_kwargs: Mapping[str, Any] | None = None,
        search_tolerance: timedelta = timedelta(hours=12),
        time_coordinate: str = "time",
        spatial_dims: Mapping[str, np.ndarray],
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
        """Synchronous entry point matching :class:`~earth2studio.data.base.DataSource`."""
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
        """Async entry point matching :class:`~earth2studio.data.base.DataSource`."""
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
        """Wrapper to download variables for a timestamp and assign into the output array."""
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
        """Fetch all requested variables for a timestamp and return stacked numpy data."""
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
    # Methods to override for atypical MPC collections
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
            The STAC item returned by :meth:`_locate_item`.  Sub-classes may expect
            collection-specific properties (e.g., per-tile assets or bespoke metadata).
        variables : Sequence[VariableSpec]
            Resolved variable requests for the current timestamp.  Each entry records
            the lexicon dataset key, modifier callable and output slice index.

        Returns
        -------
        list[AssetPlan]
            A list of :class:`AssetPlan` instances describing every remote asset that
            must be fetched along with the variables each asset can satisfy.  The
            default implementation assumes a 1:many relationship between a single
            asset key and all variables, but subclasses can override
            :meth:`select_assets` or this method entirely when collections expose
            multiple files (e.g., per-band HDF subsets).

        Notes
        -----
        The plans produced here drive the rest of the download workflow:
        :meth:`_fetch_array` will iterate the plans, invoke
        :meth:`_downloaded_asset` for cache misses, and finally extract variables via
        :meth:`extract_variable_numpy`.  Custom data sources should override this
        method when they need to attach additional metadata (e.g., per-variable media
        types) or when asset selection depends on runtime inputs beyond the lexicon.
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
        """Return mapping of asset key -> variables satisfied by that asset.

        Parameters
        ----------
        item : pystac.Item
            The STAC item to inspect for available assets.
        variables : Sequence[VariableSpec]
            The resolved variables that need to be sourced from the item.

        Returns
        -------
        Mapping[str, Sequence[VariableSpec]]
            Dictionary keyed by asset identifier (as used in ``item.assets``) with the
            list of variable specifications that can be served by that asset.

        Notes
        -----
        The base implementation routes every variable through ``self._asset_key``,
        which matches collections that bundle all requested fields in a single NetCDF
        or GeoTIFF.  Collections with per-variable assets (for example, MODIS Fire
        where FireMask, MaxFRP, and QA live in distinct files) should override this
        method to return a more granular mapping while reusing the rest of the
        download pipeline.
        """
        return {self._asset_key: variables}

    def extract_variable_numpy(
        self,
        asset_data: xr.Dataset | xr.DataArray,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
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
            # data_array.rio.close()
            return

        raise NotImplementedError(
            f"Unsupported asset format for file '{local_path}'. "
            "Only NetCDF and GeoTIFF assets are currently supported."
        )

    @property
    def cache(self) -> str:
        """Return cache root for Planetary Computer assets."""
        cache_root = os.path.join(datasource_cache_root(), "planetary_computer")
        if not self._cache:
            cache_root = os.path.join(cache_root, "tmp")
        return cache_root


class PlanetaryComputerOISST(PlanetaryComputerData):
    """Daily 0.25° NOAA Optimum Interpolation SST from Microsoft Planetary Computer."""

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
        field = asset_data[spec.dataset_key].isel(time=0, zlev=0)
        values = np.asarray(field.values, dtype=np.float32)
        result = np.asarray(spec.modifier(values), dtype=np.float32)
        return result


class PlanetaryComputerSentinel3AOD(PlanetaryComputerData):
    """Sentinel-3 SYNERGY Level-2 aerosol optical depth and surface reflectance."""

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
        field = asset_data[spec.dataset_key]
        values = np.asarray(field.values, dtype=np.float32)
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
    """MODIS Thermal Anomalies/Fire Daily (FireMask, MaxFRP, QA)."""

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
        tile: str | None = None,
        cache: bool = True,
        verbose: bool = True,
        max_workers: int = 24,
        request_timeout: int = PlanetaryComputerData.DEFAULT_TIMEOUT,
        max_retries: int = PlanetaryComputerData.DEFAULT_RETRIES,
        async_timeout: int = PlanetaryComputerData.DEFAULT_ASYNC_TIMEOUT,
    ) -> None:
        search_kwargs = None
        self._tile_id = tile.lower() if tile else None
        if self._tile_id:
            if (
                len(self._tile_id) != 6
                or not self._tile_id.startswith("h")
                or self._tile_id[3] != "v"
            ):
                raise ValueError(
                    "MODIS tile identifiers must look like hXXvYY (e.g., h35v10)."
                )
            search_kwargs = {"ids": [f"*{self._tile_id}*"]}
        super().__init__(
            self.COLLECTION_ID,
            asset_key="FireMask",
            lexicon=MODISFireLexicon,
            search_kwargs=search_kwargs,
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
        values = np.asarray(field.values, dtype=np.float32)
        result = np.asarray(spec.modifier(values), dtype=np.float32)
        return result
