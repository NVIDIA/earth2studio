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

import hashlib
import os
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Type
from urllib.parse import urlparse

import numpy as np
import requests
import xarray as xr

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon.base import LexiconType
from earth2studio.lexicon.modis_fire import MODISFireLexicon
from earth2studio.lexicon.oisst import OISSTLexicon
from earth2studio.lexicon.sentinel3_aod import Sentinel3AODLexicon
from earth2studio.utils.imports import OptionalDependencyFailure, check_optional_dependencies
from earth2studio.utils.type import TimeArray, VariableArray

try:  # pragma: no cover - exercised in integration tests
    from pystac_client import Client
    import planetary_computer
except ImportError:  # pragma: no cover - handled by optional dependency guard
    OptionalDependencyFailure("data")
    Client = None  # type: ignore[assignment]
    planetary_computer = None  # type: ignore[assignment]


Modifier = Callable[[Any], Any]


@check_optional_dependencies()
class PlanetaryComputerData:
    """Generic data source that streams assets from Microsoft Planetary Computer.

    This helper wraps the Planetary Computer STAC API and downloads assets matching
    a collection/search query into the standard Earth2Studio xarray interface. It is
    intentionally lightweight: you provide the STAC collection, which asset within
    each item should be read (e.g. ``"netcdf"`` or ``"cog"``), and optionally a lexicon
    describing how Earth2Studio variable names map onto dataset variables. Assets are
    signed with short-lived SAS tokens via :mod:`planetary_computer` prior to download.

    Parameters
    ----------
    collection_id : str
        STAC collection identifier to query (e.g. ``"noaa-cdr-sea-surface-temperature-optimum-interpolation"``).
        asset_key : str, optional
        Asset key within each item to download. The default ``"netcdf"`` accesses the
        OISST NetCDF files. Any asset readable by :func:`xarray.open_dataset` is
        supported, by default ``"netcdf"``.
    lexicon : type[LexiconType]
        Lexicon providing the mapping between Earth2Studio variable identifiers and
        dataset variable names/modifiers. All Planetary Computer data sources require a
        lexicon to translate requested variables to dataset fields.
    search_kwargs : dict[str, Any] | None, optional
        Extra arguments forwarded to :meth:`pystac_client.Client.search`. This can
        include spatial filters (``bbox``), STAC property filters, etc., by default None.
    search_tolerance : timedelta, optional
        +/- window added to each requested time when querying STAC (``datetime`` search
        parameter). Daily collections often store a single timestamp; default 12 hours
        ensures a match within the same UTC day, by default ``timedelta(hours=12)``.
    time_coordinate : str, optional
        Name of the temporal dimension in downloaded assets. If found, it is renamed to
        ``"time"`` and overwritten with the requested timestamp, by default ``"time"``.
    cache : bool, optional
        Cache downloaded assets on disk under ``~/.cache/earth2studio/planetary_computer``.
        When ``False`` files are deleted after use, by default ``True``.
    verbose : bool, optional
        Emit log messages when downloading assets, by default ``True``.

    Notes
    -----
    This class focuses on gridded NetCDF assets. Cloud-Optimised GeoTIFF assets can be
    accessed by setting ``asset_key`` appropriately and providing a modifier that uses
    :func:`xarray.open_dataset` compatible readers such as ``rioxarray.open_rasterio``.
    """

    STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def __init__(
        self,
        collection_id: str,
        *,
        asset_key: str = "netcdf",
        lexicon: Type[LexiconType],
        search_kwargs: dict[str, Any] | None = None,
        search_tolerance: timedelta = timedelta(hours=12),
        time_coordinate: str = "time",
        cache: bool = True,
        verbose: bool = True,
    ) -> None:

        self._collection_id = collection_id
        self._asset_key = asset_key
        self._lexicon = lexicon
        self._search_kwargs = search_kwargs or {}
        self._search_tolerance = search_tolerance
        self._time_coordinate = time_coordinate
        self._cache = cache
        self._verbose = verbose

        self._client: Client | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        times, variables = prep_data_inputs(time, variable)

        datasets: list[xr.DataArray] = []
        for requested_time in times:
            time_array = self._fetch_time(requested_time, variables)
            datasets.append(time_array)

        result = xr.concat(datasets, dim="time")
        return result

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async interface matching :class:`~earth2studio.data.base.DataSource`."""
        import asyncio

        return await asyncio.to_thread(self.__call__, time, variable)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fetch_time(self, requested_time: datetime, variables: list[str]) -> xr.DataArray:
        normalized_time = self._to_datetime(requested_time)
        item = self._locate_item(normalized_time)
        asset = self._extract_asset(item)

        signed_asset = planetary_computer.sign(asset)  # type: ignore[union-attr]
        local_path = self._download_asset(signed_asset.href)

        try:
            ds = self._open_dataset(local_path)
        except Exception as error:  # pragma: no cover - xarray handles errors
            if not self._cache:
                self._cleanup_file(local_path)
            raise error

        variable_arrays = []
        for var in variables:
            dataset_key, modifier = self._resolve_variable(var)
            if dataset_key not in ds:
                raise KeyError(
                    f"Variable '{dataset_key}' not present in dataset for {normalized_time.isoformat()}"
                )
            da = ds[dataset_key]
            da = modifier(da)
            for singleton_dim in [
                dim
                for dim in da.dims
                if da.sizes.get(dim, 0) == 1 and dim in {"zlev", "lev", "level", "depth"}
            ]:
                da = da.squeeze(singleton_dim, drop=True)
            da = self._prepare_time_dimension(da, normalized_time)
            da = da.expand_dims(dim={"variable": [var]})
            variable_arrays.append(da)

        time_data = xr.concat(variable_arrays, dim="variable")
        # grid dims follow underlying dataset order; ensure 'time' leading for consistency
        existing_dims = [dim for dim in time_data.dims if dim not in {"time", "variable"}]
        time_data = time_data.transpose("time", "variable", *existing_dims)
        if not self._cache:
            self._cleanup_file(local_path)
        return time_data.astype(np.float32)

    def _prepare_time_dimension(self, da: xr.DataArray, target_time: datetime) -> xr.DataArray:
        if self._time_coordinate in da.dims:
            da = da.rename({self._time_coordinate: "time"})
            da = da.isel(time=0, drop=True) if da.sizes.get("time", 1) == 1 else da
        if "time" not in da.dims:
            da = da.expand_dims(time=[np.datetime64(target_time)])
        else:
            da = da.assign_coords(time=[np.datetime64(target_time)])
            da = da.isel(time=slice(0, 1))
        return da

    def _locate_item(self, when: datetime):
        client = self._ensure_client()
        start = (when - self._search_tolerance).isoformat()
        end = (when + self._search_tolerance).isoformat()
        datetime_param = f"{start}/{end}"

        search = client.search(
            collections=[self._collection_id],
            datetime=datetime_param,
            limit=1,
            **self._search_kwargs,
        )
        try:
            item = next(search.items())
        except StopIteration as error:
            raise FileNotFoundError(
                f"No Planetary Computer item found for {when.isoformat()} within ±{self._search_tolerance}"
            ) from error
        return item

    def _extract_asset(self, item, asset_key: str | None = None):
        key = asset_key or self._asset_key
        if key not in item.assets:
            raise KeyError(f"Asset '{key}' not available in item {item.id}")
        return item.assets[key]

    def _download_asset(self, href: str) -> pathlib.Path:
        parsed = urlparse(href)
        suffix = pathlib.Path(parsed.path).suffix or ""
        filename = hashlib.sha256(href.encode()).hexdigest() + suffix
        local_path = pathlib.Path(self.cache) / filename

        if local_path.exists():
            return local_path

        if self._verbose:
            print(f"Downloading Planetary Computer asset {href}")

        response = requests.get(
            href,
            stream=True,
            timeout=60,
            headers={"User-Agent": "earth2studio-planetary-computer"},
        )
        response.raise_for_status()

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    file_handle.write(chunk)
        return local_path

    def _cleanup_file(self, path: pathlib.Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best-effort cleanup
            return
        parent = path.parent
        root = pathlib.Path(self.cache)
        while parent != root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def _resolve_variable(self, variable: str) -> tuple[str, Modifier]:
        try:
            dataset_key, modifier = self._lexicon[variable]
        except KeyError as error:
            raise KeyError(
                f"Variable '{variable}' not supported by {self._lexicon.__name__}"
            ) from error
        return dataset_key, modifier

    @staticmethod
    def _identity(array: xr.DataArray) -> xr.DataArray:
        return array

    @staticmethod
    def _to_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _open_dataset(self, path: pathlib.Path) -> xr.Dataset:
        return xr.open_dataset(path)

    def _ensure_client(self) -> Client:
        if self._client is None:
            self._client = Client.open(self.STAC_API_URL)
        return self._client

    @property
    def cache(self) -> str:
        """Return cache root for Planetary Computer assets."""
        cache_root = os.path.join(datasource_cache_root(), "planetary_computer")
        if not self._cache:
            cache_root = os.path.join(cache_root, "tmp")
        os.makedirs(cache_root, exist_ok=True)
        return cache_root


class PlanetaryComputerOISST(PlanetaryComputerData):
    """Daily 0.25° NOAA Optimum Interpolation SST from Microsoft Planetary Computer.

    Parameters
    ----------
    cache : bool, optional
        Cache downloaded NetCDF files locally. Defaults to ``True``.
    verbose : bool, optional
        Print download information while fetching assets. Defaults to ``True``.

    Notes
    -----
    Data origin: `NOAA CDR Sea Surface Temperature Optimum Interpolation`
    <https://planetarycomputer.microsoft.com/dataset/noaa-cdr-sea-surface-temperature-optimum-interpolation#overview>`_.
    """

    COLLECTION_ID = "noaa-cdr-sea-surface-temperature-optimum-interpolation"
    ASSET_KEY = "netcdf"
    TIME_COORDINATE = "time"
    SEARCH_TOLERANCE = timedelta(hours=12)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.ASSET_KEY,
            lexicon=OISSTLexicon,
            search_kwargs=None,
            search_tolerance=self.SEARCH_TOLERANCE,
            time_coordinate=self.TIME_COORDINATE,
            cache=cache,
            verbose=verbose,
        )


class PlanetaryComputerSentinel3AOD(PlanetaryComputerData):
    """Sentinel-3 SYNERGY Level-2 aerosol optical depth and surface reflectance.

    Parameters
    ----------
    cache : bool, optional
        Cache downloaded NetCDF files locally. Defaults to ``True``.
    verbose : bool, optional
        Print download information while fetching assets. Defaults to ``True``.

    Notes
    -----
    Data origin: `Sentinel-3 SYNERGY Aerosol Optical Depth`
    <https://planetarycomputer.microsoft.com/dataset/sentinel-3-synergy-aod-l2-netcdf>`_.
    """

    COLLECTION_ID = "sentinel-3-synergy-aod-l2-netcdf"
    ASSET_KEY = "ntc-aod"
    TIME_COORDINATE = "time"
    SEARCH_TOLERANCE = timedelta(hours=12)

    def __init__(
        self,
        cache: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.ASSET_KEY,
            lexicon=Sentinel3AODLexicon,
            search_kwargs=None,
            search_tolerance=self.SEARCH_TOLERANCE,
            time_coordinate=self.TIME_COORDINATE,
            cache=cache,
            verbose=verbose,
        )

    def _open_dataset(self, path: pathlib.Path) -> xr.Dataset:
        ds = xr.open_dataset(path)
        for coord in ("latitude", "longitude"):
            if coord in ds and ds[coord].dtype != np.float32:
                ds[coord] = ds[coord].astype(np.float32)
        return ds


class PlanetaryComputerMODISFire(PlanetaryComputerData):
    """MODIS Thermal Anomalies/Fire Daily (FireMask, MaxFRP, QA).

    Parameters
    ----------
    cache : bool, optional
        Cache downloaded assets locally. Defaults to ``True``.
    verbose : bool, optional
        Print download information while fetching assets. Defaults to ``True``.

    Notes
    -----
    Data origin: `MODIS Thermal Anomalies/Fire Daily (MOD14A1/MYD14A1)`
    <https://planetarycomputer.microsoft.com/dataset/modis-14A1-061>`_.
    """

    COLLECTION_ID = "modis-14A1-061"
    PRIMARY_ASSET = "FireMask"
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
    ) -> None:
        super().__init__(
            self.COLLECTION_ID,
            asset_key=self.PRIMARY_ASSET,
            lexicon=MODISFireLexicon,
            search_kwargs=None,
            search_tolerance=self.SEARCH_TOLERANCE,
            time_coordinate=self.TIME_COORDINATE,
            cache=cache,
            verbose=verbose,
        )

    def _open_dataset(self, path: pathlib.Path) -> xr.Dataset:  # pragma: no cover
        raise NotImplementedError(
            "PlanetaryComputerMODISFire loads per-variable assets and does not use the base opener"
        )

    def _fetch_time(self, requested_time: datetime, variables: list[str]) -> xr.DataArray:
        normalized_time = self._to_datetime(requested_time)
        item = self._locate_item(normalized_time)

        from collections import defaultdict

        asset_requirements: dict[str, list[tuple[str, str, Modifier]]] = defaultdict(list)
        for var in variables:
            dataset_key, modifier = self._resolve_variable(var)
            asset_key = self.VARIABLE_ASSETS.get(var)
            if asset_key is None:
                raise KeyError(
                    f"Asset mapping not defined for MODIS Fire variable '{var}'"
                )
            asset_requirements[asset_key].append((var, dataset_key, modifier))

        per_variable_arrays: list[xr.DataArray] = []
        for asset_key, requests_for_asset in asset_requirements.items():
            asset = self._extract_asset(item, asset_key)
            signed_asset = planetary_computer.sign(asset)  # type: ignore[union-attr]
            local_path = self._download_asset(signed_asset.href)

            try:
                import rioxarray  # noqa: PLC0415

                data = rioxarray.open_rasterio(local_path)
            except Exception as error:  # pragma: no cover
                if not self._cache:
                    self._cleanup_file(local_path)
                raise error

            for var, dataset_key, modifier in requests_for_asset:
                da = data
                if "band" in da.dims and da.sizes.get("band") == 1:
                    da = da.squeeze("band", drop=True)
                da = da.astype(np.float32)
                da.name = dataset_key
                da = modifier(da)
                da = da.to_dataset(name=dataset_key)[dataset_key]
                da = da.expand_dims(dim={"variable": [var]})
                da = self._prepare_time_dimension(da, normalized_time)
                per_variable_arrays.append(da)

            if not self._cache:
                self._cleanup_file(local_path)

        result = xr.concat(per_variable_arrays, dim="variable")
        existing_dims = [dim for dim in result.dims if dim not in {"time", "variable"}]
        result = result.transpose("time", "variable", *existing_dims)
        return result.astype(np.float32)

    @staticmethod
    def _identity(array: xr.DataArray) -> xr.DataArray:
        return array

    @staticmethod
    def _to_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
