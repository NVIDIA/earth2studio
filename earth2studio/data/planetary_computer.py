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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator, Mapping, Sequence, Type
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
from earth2studio.utils.imports import OptionalDependencyFailure, check_optional_dependencies
from earth2studio.utils.type import TimeArray, VariableArray

try:  # pragma: no cover - exercised in integration tests
    import httpx
    from pystac_client import Client
    import planetary_computer
    import rioxarray
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
        lexicon: Type[LexiconType],
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
            dim: np.asarray(values, dtype=np.float32) for dim, values in spatial_dims.items()
        }
        self._spatial_shape = tuple(len(self._spatial_coords[dim]) for dim in self._spatial_dim_names)
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
            t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)
            for t in times
        ]
        specs = [self._resolve_variable(index, var) for index, var in enumerate(variables)]

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
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            semaphore = asyncio.Semaphore(self._max_workers)
            with tqdm(
                total=len(times) * len(variables),
                disable=not self._verbose,
                desc=f"Fetching {self._collection_id}",
            ) as progress:
                tasks = [
                    asyncio.create_task(
                        self._fetch_and_assign(
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

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------
    async def _fetch_and_assign(
        self,
        *,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        requested_time: datetime,
        variables: Sequence[VariableSpec],
        xr_array: xr.DataArray,
        time_index: int,
        progress: tqdm,
    ) -> None:
        """Download assets for a single timestamp and populate the output array."""
        stacked = await self._fetch_time_payload(
            client, semaphore, requested_time, variables
        )
        if stacked.shape != (len(variables), *self._spatial_shape):
            raise ValueError(
                f"Unexpected data shape {stacked.shape} for {requested_time.isoformat()} "
                f"in collection {self._collection_id}; expected {(len(variables), *self._spatial_shape)}."
            )
        xr_array[time_index] = stacked
        progress.update(len(variables))

    async def _fetch_time_payload(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        requested_time: datetime,
        variables: Sequence[VariableSpec],
    ) -> np.ndarray:
        """Fetch all requested variables for a timestamp and return stacked numpy data."""
        item = await asyncio.to_thread(self._locate_item, requested_time)
        asset_plans = self._prepare_asset_plans(item, variables)

        download_tasks = [
            self._ensure_asset_downloaded(client, semaphore, plan)
            for plan in asset_plans
            if not plan.local_path.exists()
        ]
        if download_tasks:
            await asyncio.gather(*download_tasks)

        data_stack = np.zeros(
            (len(variables), *self._spatial_shape), dtype=np.float32
        )
        filled = [False] * len(variables)

        for plan in asset_plans:
            with self._open_asset(plan.local_path, plan.media_type) as asset_data:
                for spec in plan.variables:
                    array = self._extract_variable_numpy(
                        asset_data, spec, requested_time
                    )
                    data_stack[spec.index] = array
                    filled[spec.index] = True

        if not all(filled):
            missing = [
                variables[idx].variable_id
                for idx, present in enumerate(filled)
                if not present
            ]
            raise ValueError(
                f"Failed to extract variables {missing} for {requested_time.isoformat()}"
            )

        return data_stack

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def _prepare_asset_plans(
        self,
        item: Any,
        variables: Sequence[VariableSpec],
    ) -> list[AssetPlan]:
        asset_map = self._select_assets(item, variables)
        plans: list[AssetPlan] = []

        for asset_key, specs in asset_map.items():
            if asset_key not in item.assets:
                raise KeyError(f"Asset '{asset_key}' not available in item {item.id}")
            asset = item.assets[asset_key]
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

    def _select_assets(
        self,
        item: Any,
        variables: Sequence[VariableSpec],
    ) -> Mapping[str, Sequence[VariableSpec]]:
        """Return mapping of asset key -> variables satisfied by that asset."""
        return {self._asset_key: variables}

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------
    async def _ensure_asset_downloaded(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        plan: AssetPlan,
    ) -> None:
        plan.local_path.parent.mkdir(parents=True, exist_ok=True)

        async with semaphore:
            backoff = 1.5
            attempt = 0
            while True:
                try:
                    async with client.stream(
                        "GET",
                        plan.signed_href,
                        headers={"User-Agent": self.USER_AGENT},
                    ) as response:
                        response.raise_for_status()
                        temp_path = plan.local_path.with_suffix(".tmp")
                        with temp_path.open("wb") as file_handle:
                            async for chunk in response.aiter_bytes(self.CHUNK_SIZE):
                                if chunk:
                                    file_handle.write(chunk)
                        temp_path.replace(plan.local_path)
                    return
                except Exception as error:
                    attempt += 1
                    if attempt >= self._max_retries:
                        if plan.local_path.exists():
                            plan.local_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Failed to download asset {plan.signed_href}"
                        ) from error
                    await asyncio.sleep(backoff)
                    backoff *= 2

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------
    def _extract_variable_numpy(
        self,
        asset_data: xr.Dataset | xr.DataArray,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        data_array = self._select_variable(asset_data, spec)
        if self._time_coordinate in data_array.dims and self._time_coordinate != "time":
            data_array = data_array.rename({self._time_coordinate: "time"})

        if "time" in data_array.dims and data_array.sizes.get("time", 0) > 1:
            data_array = data_array.sel(time=np.datetime64(target_time), method="nearest")
        if "time" in data_array.dims:
            data_array = data_array.isel(time=0, drop=True)

        for dim in list(data_array.dims):
            if dim in self.LEVEL_DIMS | self.OPTIONAL_SINGLETON_DIMS and data_array.sizes.get(dim, 0) == 1:
                data_array = data_array.squeeze(dim, drop=True)

        missing_dims = [dim for dim in self._spatial_dim_names if dim not in data_array.dims]
        if missing_dims:
            raise KeyError(
                f"Variable '{spec.variable_id}' missing expected spatial dimensions "
                f"{missing_dims} in collection {self._collection_id}"
            )

        data_array = data_array.transpose(*self._spatial_dim_names)
        array = np.asarray(spec.modifier(data_array.values), dtype=np.float32)

        if array.shape != self._spatial_shape:
            raise ValueError(
                f"Modifier for '{spec.variable_id}' returned shape {array.shape}, "
                f"expected {self._spatial_shape}"
            )

        return array

    # ------------------------------------------------------------------
    # STAC helpers
    # ------------------------------------------------------------------
    def _locate_item(self, when: datetime):
        if self._client is None:
            self._client = Client.open(self.STAC_API_URL)
        start = (when - self._search_tolerance).isoformat()
        end = (when + self._search_tolerance).isoformat()
        datetime_param = f"{start}/{end}"

        search = self._client.search(
            collections=[self._collection_id],
            datetime=datetime_param,
            limit=1,
            **self._search_kwargs,
        )
        try:
            return next(search.items())
        except StopIteration as error:
            raise FileNotFoundError(
                f"No Planetary Computer item found for {when.isoformat()} "
                f"within ±{self._search_tolerance}"
            ) from error

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _resolve_variable(self, variable: str) -> VariableSpec:
        try:
            dataset_key, modifier = self._lexicon[variable]
        except KeyError as error:
            raise KeyError(
                f"Variable '{variable}' not supported by {self._lexicon.__name__}"
            ) from error
        return VariableSpec(variable, dataset_key, modifier)

    def _local_asset_path(self, href: str) -> pathlib.Path:
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

        if suffix in self.NETCDF_SUFFIXES or (
            media_type and "netcdf" in media_type.lower()
        ):
            dataset = xr.open_dataset(local_path)
            yield dataset
            dataset.close()
            return

        if suffix in self.GEOTIFF_SUFFIXES or (
            media_type and "tiff" in media_type.lower()
        ):
            data_array = rioxarray.open_rasterio(local_path)
            yield data_array
            close_method = getattr(data_array, "close", None)
            if callable(close_method):
                close_method()
            with contextlib.suppress(AttributeError):
                data_array.rio.close()
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


# ----------------------------------------------------------------------
# Concrete data sources
# ----------------------------------------------------------------------


class PlanetaryComputerOISST(PlanetaryComputerData):
    """Daily 0.25° NOAA Optimum Interpolation SST from Microsoft Planetary Computer."""

    COLLECTION_ID = "noaa-cdr-sea-surface-temperature-optimum-interpolation"
    ASSET_KEY = "netcdf"
    TIME_COORDINATE = "time"
    SEARCH_TOLERANCE = timedelta(hours=12)

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
                "lat": np.linspace(-89.875, 89.875, 720, dtype=np.float32),
                "lon": np.linspace(0.125, 359.875, 1440, dtype=np.float32),
            },
            cache=cache,
            verbose=verbose,
            max_workers=max_workers,
            request_timeout=request_timeout,
            max_retries=max_retries,
            async_timeout=async_timeout,
        )


class PlanetaryComputerSentinel3AOD(PlanetaryComputerData):
    """Sentinel-3 SYNERGY Level-2 aerosol optical depth and surface reflectance."""

    COLLECTION_ID = "sentinel-3-synergy-aod-l2-netcdf"
    ASSET_KEY = "ntc-aod"
    TIME_COORDINATE = "time"
    SEARCH_TOLERANCE = timedelta(hours=12)

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
                "latitude": np.linspace(-90.0, 90.0, 1800, dtype=np.float32),
                "longitude": np.linspace(-180.0, 180.0, 3600, dtype=np.float32),
            },
            cache=cache,
            verbose=verbose,
            max_workers=max_workers,
            request_timeout=request_timeout,
            max_retries=max_retries,
            async_timeout=async_timeout,
        )

    def _extract_variable_numpy(
        self,
        asset_data: xr.Dataset | xr.DataArray,
        spec: VariableSpec,
        target_time: datetime,
    ) -> np.ndarray:
        array = super()._extract_variable_numpy(asset_data, spec, target_time)
        return array.astype(np.float32, copy=False)


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
                "y": np.arange(2400, dtype=np.float32),
                "x": np.arange(2400, dtype=np.float32),
            },
            cache=cache,
            verbose=verbose,
            max_workers=max_workers,
            request_timeout=request_timeout,
            max_retries=max_retries,
            async_timeout=async_timeout,
        )

    def _select_assets(
        self,
        item: Any,
        variables: Sequence[VariableSpec],
    ) -> Mapping[str, Sequence[VariableSpec]]:
        asset_map: dict[str, list[VariableSpec]] = {}
        for spec in variables:
            asset_key = self.VARIABLE_ASSETS.get(spec.variable_id)
            if asset_key is None:
                raise KeyError(
                    f"Asset mapping not defined for MODIS Fire variable '{spec.variable_id}'"
                )
            asset_map.setdefault(asset_key, []).append(spec)
        return asset_map

    def _select_variable(
        self,
        asset_data: xr.Dataset | xr.DataArray,
        spec: VariableSpec,
    ) -> xr.DataArray:
        data_array = super()._select_variable(asset_data, spec)
        if "band" in data_array.dims and data_array.sizes.get("band", 0) == 1:
            data_array = data_array.squeeze("band", drop=True)
        data_array.name = spec.dataset_key
        return data_array
