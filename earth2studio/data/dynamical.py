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

import json
import os
import urllib.request
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm

from earth2studio.data.utils import (
    _sync_async,
    datasource_cache_root,
    ensure_utc,
    prep_forecast_inputs,
)
from earth2studio.lexicon import DynamicalLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray

try:
    import icechunk
except ImportError:
    OptionalDependencyFailure("data")
    icechunk = None  # type: ignore[assignment]

# Earth2Studio usually uses 0-360 ascending longitude and 90 -> -90 descending
# latitude. Regional domains that cross Greenwich stay in their native
# contiguous order to avoid splitting the grid across the array edge.


def _fetch_json(url: str) -> dict[str, Any]:
    """Fetch and parse a JSON document from the dynamical.org STAC catalog.

    A custom User-Agent header is required; the dynamical.org server rejects the
    default urllib agent with HTTP 403.

    Parameters
    ----------
    url : str
        URL of the JSON document to fetch.

    Returns
    -------
    dict[str, Any]
        Parsed JSON document.
    """
    request = urllib.request.Request(  # noqa: S310 # https only
        url, headers={"User-Agent": "earth2studio"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read())


@check_optional_dependencies()
class _DynamicalBase:
    """Shared infrastructure for dynamical.org STAC data sources.

    Resolves a dynamical.org STAC collection to its Icechunk repository,
    opens it lazily with xarray, and maps Earth2Studio variable ids to the
    collection's variables (via :class:`DynamicalLexicon` plus native
    pass-through), applying lexicon-defined unit conversions to the
    Earth2Studio convention.
    """

    STAC_CATALOG_URL = "https://stac.dynamical.org/catalog.json"
    # Name of the temporal STAC dimension used for time validation and
    # availability checks (overridden by forecast collections).
    _TIME_DIMENSION = "time"

    def __init__(
        self,
        collection: str,
        member: int = 0,
        cache: bool = True,
        verbose: bool = True,
    ) -> None:
        self.collection = collection
        self._member = member
        self._cache = cache
        self._verbose = verbose

        self._ds: xr.Dataset | None = None
        self._cube_variables: dict[str, dict[str, Any]] = {}
        self._cube_dimensions: dict[str, dict[str, Any]] = {}
        # Normalized grid coordinates (computed once on open).
        self._lat: np.ndarray = np.array([])
        self._lon: np.ndarray = np.array([])
        self._x: np.ndarray = np.array([])
        self._y: np.ndarray = np.array([])
        self._source_spatial_dims: tuple[str, str] = ("latitude", "longitude")
        self._output_spatial_dims: tuple[str, str] = ("lat", "lon")

    def _resolve_collection_url(self) -> str:
        """Resolve the STAC collection.json URL from the root catalog.

        Returns
        -------
        str
            Absolute URL of the requested collection.

        Raises
        ------
        ValueError
            If the collection id is not present in the catalog.
        """
        catalog = _fetch_json(self.STAC_CATALOG_URL)
        children = {}
        for link in catalog.get("links", []):
            if link.get("rel") != "child":
                continue
            href = urljoin(self.STAC_CATALOG_URL, link["href"])
            # Collection id is the path segment preceding ``collection.json``.
            collection_id = urlparse(href).path.rstrip("/").split("/")[-2]
            children[collection_id] = href
        if self.collection not in children:
            available = ", ".join(sorted(children))
            raise ValueError(
                f"Unknown dynamical.org collection {self.collection!r}. "
                f"Available collections: {available}"
            )
        return children[self.collection]

    def _open(self) -> xr.Dataset:
        """Lazily fetch the STAC collection and open its Icechunk repository.

        Returns
        -------
        xr.Dataset
            The grid-normalized, lazily opened dataset.
        """
        if self._ds is not None:
            return self._ds

        collection = _fetch_json(self._resolve_collection_url())
        cube_variables = collection.get("cube:variables", {})
        self._cube_variables = cube_variables
        self._cube_dimensions = collection.get("cube:dimensions", {})

        dims = self._cube_dimensions
        is_regular_latlon = "latitude" in dims and "longitude" in dims
        is_projected_xy = "x" in dims and "y" in dims
        if not is_regular_latlon and not is_projected_xy:
            raise ValueError(
                f"dynamical.org collection {self.collection!r} is not on a supported "
                f"regular latitude/longitude or projected x/y grid (dimensions: "
                f"{sorted(dims)})."
            )

        assets = collection.get("assets", {})
        if "icechunk" not in assets:
            raise ValueError(
                f"dynamical.org collection {self.collection!r} has no 'icechunk' asset"
            )
        asset = assets["icechunk"]
        if "href" not in asset:
            raise KeyError(
                f"dynamical.org collection {self.collection!r} icechunk asset is "
                f"missing required 'href' key in STAC metadata"
            )
        href = asset["href"]
        storage_options = asset.get("xarray:storage_options", {})
        region = storage_options.get("client_kwargs", {}).get("region_name")
        virtual_containers = asset.get("icechunk:virtual_chunk_containers") or []

        ds = self._open_icechunk(
            href, region=region, virtual_containers=virtual_containers
        )
        ds = self._setup_grid(ds)
        self._cube_variables = {
            name: cube_variables.get(name, {})
            for name in ds.data_vars
            if name in cube_variables
        }
        self._ds = ds
        return self._ds

    def _open_icechunk(
        self,
        href: str,
        region: str | None = None,
        virtual_containers: list[dict[str, Any]] | None = None,
    ) -> xr.Dataset:
        """Open the Icechunk repository described by a STAC asset.

        Parameters
        ----------
        href : str
            S3 URL of the Icechunk repository (e.g. ``s3://bucket/prefix``).
        region : str | None, optional
            AWS region for the S3 bucket. If None, icechunk will attempt
            auto-detection.
        virtual_containers : list[dict[str, Any]] | None, optional
            List of virtual chunk container entries from the STAC asset; each
            entry should have a ``url_prefix`` key.

        Returns
        -------
        xr.Dataset
            Lazily opened dataset backed by the Icechunk session store.
        """
        parsed = urlparse(href)
        if parsed.scheme != "s3":
            raise ValueError(
                f"dynamical.org collection {self.collection!r} icechunk asset href "
                f"is not an s3 url: {href!r}"
            )
        # region may be None if the STAC asset does not advertise it; icechunk
        # will attempt to auto-detect the AWS region in that case.
        storage = icechunk.s3_storage(
            bucket=parsed.netloc,
            prefix=parsed.path.lstrip("/"),
            region=region,
            anonymous=True,
        )

        # Authorize anonymous access to any referenced virtual chunk containers.
        prefixes = [
            entry["url_prefix"]
            for entry in (virtual_containers or [])
            if "url_prefix" in entry
        ]
        authorize = (
            icechunk.containers_credentials(
                {p: icechunk.s3_anonymous_credentials() for p in prefixes}
            )
            if prefixes
            else None
        )

        repo = icechunk.Repository.open(
            storage, authorize_virtual_chunk_access=authorize
        )
        session = repo.readonly_session("main")
        # ``chunks=None`` avoids dask and lets zarr read only the requested chunks
        # on indexing (the dynamical.org-recommended access pattern). Icechunk
        # manages its own metadata, so zarr consolidated metadata does not apply.
        return xr.open_zarr(session.store, consolidated=False, chunks=None)

    def _setup_grid(self, ds: xr.Dataset) -> xr.Dataset:
        """Normalize the dataset grid to Earth2Studio's convention.

        Regular latitude/longitude grids are sorted north-to-south and, when
        possible without splitting a regional domain, to [0, 360) ascending
        longitude. Projected x/y grids preserve their native projection axes and
        expose 2D latitude/longitude coordinates.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset opened from dynamical.org.

        Returns
        -------
        xr.Dataset
            Dataset with normalized grid coordinates.
        """
        if "latitude" in ds.dims and "longitude" in ds.dims:
            ds = ds.sortby("latitude", ascending=False)
            lon = np.asarray(ds["longitude"].values)
            if self._should_wrap_longitude(lon):
                ds = ds.assign_coords(longitude=(lon % 360))
                ds = ds.sortby("longitude", ascending=True)
            self._lat = np.asarray(ds["latitude"].values)
            self._lon = np.asarray(ds["longitude"].values)
            self._source_spatial_dims = ("latitude", "longitude")
            self._output_spatial_dims = ("lat", "lon")
            return ds

        if {"x", "y"}.issubset(ds.dims) and {"latitude", "longitude"}.issubset(
            ds.coords
        ):
            lon = np.asarray(ds["longitude"].values)
            if self._should_wrap_longitude(lon):
                ds = ds.assign_coords(longitude=(ds["longitude"].dims, lon % 360))
            self._lat = np.asarray(ds["latitude"].values)
            self._lon = np.asarray(ds["longitude"].values)
            self._x = np.asarray(ds["x"].values)
            self._y = np.asarray(ds["y"].values)
            self._source_spatial_dims = ("y", "x")
            self._output_spatial_dims = ("y", "x")
            return ds

        raise ValueError(
            f"dynamical.org collection {self.collection!r} opened with unsupported "
            f"coordinates {sorted(ds.coords)} and dimensions {sorted(ds.dims)}."
        )

    @staticmethod
    def _should_wrap_longitude(lon: np.ndarray) -> bool:
        """Return whether longitude can be safely normalized to [0, 360)."""
        lon_min = float(np.nanmin(lon))
        lon_max = float(np.nanmax(lon))
        crosses_greenwich = lon_min < 0.0 < lon_max
        global_like = (lon_max - lon_min) >= 300.0
        return global_like or not crosses_greenwich

    def _spatial_shape(self) -> tuple[int, ...]:
        """Return the output spatial shape."""
        return (
            self._lat.shape if self._lat.ndim == 2 else (len(self._lat), len(self._lon))
        )

    def _spatial_coords(self) -> dict[str, Any]:
        """Return output spatial coordinates for the current grid."""
        if self._lat.ndim == 2:
            return {
                "y": self._y,
                "x": self._x,
                "lat": (self._output_spatial_dims, self._lat),
                "lon": (self._output_spatial_dims, self._lon),
                "_lat": (self._output_spatial_dims, self._lat),
                "_lon": (self._output_spatial_dims, self._lon),
            }
        return {"lat": self._lat, "lon": self._lon}

    def _unsupported_extra_dims(self, da: xr.DataArray) -> list[str]:
        """Return dimensions that cannot be collapsed into the standard output."""
        supported = {self._TIME_DIMENSION, "lead_time", *self._source_spatial_dims}
        return [
            dim for dim in da.dims if dim not in supported and dim != "ensemble_member"
        ]

    def _resolve_variable(self, variable: str) -> tuple[str, Callable]:
        """Resolve an Earth2Studio variable id to a collection variable and modifier.

        Parameters
        ----------
        variable : str
            Earth2Studio variable id, or a native dynamical.org variable name.

        Returns
        -------
        tuple[str, Callable]
            The dynamical.org variable name and a unit-conversion modifier.

        Raises
        ------
        KeyError
            If the variable is neither in the lexicon nor a native variable of
            this collection.
        """
        if variable in DynamicalLexicon.VOCAB:
            dynamical_name, modifier = DynamicalLexicon.get_item(variable)
        elif variable in self._cube_variables:
            # Native pass-through for variables not in the lexicon.
            dynamical_name = variable

            def modifier(x: np.ndarray) -> np.ndarray:
                return x

        else:
            available = ", ".join(sorted(self._cube_variables))
            raise KeyError(
                f"Variable {variable!r} not found in dynamical.org lexicon or in "
                f"collection {self.collection!r}. Available variables: {available}"
            )

        if dynamical_name not in self._cube_variables and variable == "tpf":
            if "precipitation_rate_surface" in self._cube_variables:
                dynamical_name = "precipitation_rate_surface"

        if dynamical_name not in self._cube_variables:
            available = ", ".join(sorted(self._cube_variables))
            raise KeyError(
                f"Variable {variable!r} (-> {dynamical_name!r}) is not available in "
                f"collection {self.collection!r}. Available variables: {available}"
            )

        return dynamical_name, modifier

    @property
    def cache(self) -> str:
        """Cache location (Icechunk fetches chunks lazily; kept for API parity)."""
        return os.path.join(datasource_cache_root(), "dynamical")

    def _time_extent(self, dimension: str) -> tuple[datetime | None, datetime | None]:
        """Return the (start, end) extent of a temporal STAC dimension."""
        extent = self._cube_dimensions.get(dimension, {}).get("extent", [None, None])

        def _parse(value: str | None) -> datetime | None:
            if value is None:
                return None
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return ensure_utc(dt)

        return (
            _parse(extent[0] if len(extent) > 0 else None),
            _parse(extent[1] if len(extent) > 1 else None),
        )

    def _validate_time(self, times: list[datetime], dimension: str) -> None:
        """Validate requested times against the collection's temporal extent.

        dynamical.org collections advertise an open-ended STAC extent (they are
        continuously updated), so the upper bound is taken from the last
        coordinate actually present in the opened store rather than the (absent)
        STAC end. This keeps :meth:`available` honest and turns out-of-range
        requests into a clear error instead of an opaque xarray ``KeyError``.

        Parameters
        ----------
        times : list[datetime]
            Requested times.
        dimension : str
            Name of the temporal STAC dimension (``time`` or ``init_time``).
        """
        times = [ensure_utc(t) for t in times]
        start, end = self._time_extent(dimension)
        if end is None and self._ds is not None and dimension in self._ds.coords:
            end = self._ds[dimension].values.max().astype("datetime64[us]").item()
        for time in times:
            if start is not None and time < start:
                raise ValueError(
                    f"Requested time {time} is before the start of collection "
                    f"{self.collection!r} ({start})"
                )
            if end is not None and time > end:
                raise ValueError(
                    f"Requested time {time} is after the end of collection "
                    f"{self.collection!r} ({end})"
                )

    def available(self, time: datetime | np.datetime64) -> bool:
        """Check if a given time is available in this dynamical.org collection.

        Unlike most Earth2Studio sources, availability is collection-dependent
        (each collection advertises its own temporal extent in its STAC
        metadata), so this is an instance method rather than a classmethod: it
        opens the collection and checks the requested time against the extent of
        the collection's temporal dimension (``time`` for analysis,
        ``init_time`` for forecast collections).

        Parameters
        ----------
        time : datetime | np.datetime64
            Time to check. For forecast collections this is the forecast
            initialization time.

        Returns
        -------
        bool
            Whether the time falls within the collection's temporal extent.
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        self._open()
        try:
            self._validate_time([time], self._TIME_DIMENSION)
        except ValueError:
            return False
        return True

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve forecast data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Forecast initialization timestamps to return data for (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to return.
        variable : str | list[str] | VariableArray
            Variable(s) to return, in the dynamical.org lexicon or native to the
            collection.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, lead_time, variable, ...]``.
        """
        xr_array = _sync_async(self.fetch, time, lead_time, variable)
        return xr_array

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to retrieve forecast data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Forecast initialization timestamps to return data for (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to return.
        variable : str | list[str] | VariableArray
            Variable(s) to return, in the dynamical.org lexicon or native to the
            collection.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, lead_time, variable, ...]``.
        """
        ds = self._open()
        times, lead_times, variables = prep_forecast_inputs(time, lead_time, variable)
        self._validate_time(times, self._TIME_DIMENSION)

        times_np = np.array(times, dtype="datetime64[ns]")
        leads_np = np.array(lead_times, dtype="timedelta64[ns]")
        has_lead_time = "lead_time" in ds.dims
        spatial_shape = self._spatial_shape()
        coords: dict[str, Any] = {
            "time": times_np,
            "lead_time": leads_np,
            "variable": variables,
            **self._spatial_coords(),
        }
        xr_array = xr.DataArray(
            data=np.empty(
                (len(times), len(lead_times), len(variables), *spatial_shape),
                dtype=np.float32,
            ),
            dims=["time", "lead_time", "variable", *self._output_spatial_dims],
            coords=coords,
        )

        for j, var in enumerate(
            tqdm(
                variables,
                desc=f"Fetching dynamical.org {self.collection} data",
                disable=(not self._verbose),
            )
        ):
            dynamical_name, modifier = self._resolve_variable(var)
            logger.debug(f"Fetching dynamical.org variable {var} ({dynamical_name})")
            if has_lead_time:
                da = ds[dynamical_name].sel(
                    {self._TIME_DIMENSION: times_np, "lead_time": leads_np}
                )
                if "ensemble_member" in da.dims:
                    da = da.isel(ensemble_member=self._member)
                extra_dims = self._unsupported_extra_dims(da)
                if extra_dims:
                    raise ValueError(
                        f"Variable {var!r} in collection {self.collection!r} has "
                        f"unsupported extra dimensions: {extra_dims}"
                    )
                da = da.transpose(
                    self._TIME_DIMENSION, "lead_time", *self._source_spatial_dims
                )
            else:
                da = ds[dynamical_name].sel({self._TIME_DIMENSION: times_np})
                extra_dims = self._unsupported_extra_dims(da)
                if extra_dims:
                    raise ValueError(
                        f"Variable {var!r} in collection {self.collection!r} has "
                        f"unsupported extra dimensions: {extra_dims}"
                    )
                da = da.transpose(self._TIME_DIMENSION, *self._source_spatial_dims)
                # Add lead_time axis for uniform output shape
                da = da.expand_dims("lead_time", axis=1)
            xr_array[:, :, j] = modifier(np.asarray(da.values, dtype=np.float32))

        return xr_array


class _DynamicalAnalysis(_DynamicalBase):
    """Shared synchronous and async entry points for analysis collections."""

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve analysis data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable(s) to return, in the dynamical.org lexicon or native to the
            collection.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, ...]``.
        """
        xr_array = _sync_async(self.fetch, time, variable)
        return xr_array

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to retrieve analysis data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable(s) to return, in the dynamical.org lexicon or native to the
            collection.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, ...]``.
        """
        result = await super().fetch(time, timedelta(0), variable)
        return result.isel(lead_time=0, drop=True)


class DynamicalAIFS(_DynamicalAnalysis):
    """ECMWF AIFS Single analysis view from the dynamical.org catalog.

    Lead-time-zero view of the deterministic ECMWF Artificial Intelligence
    Forecasting System (AIFS) forecast archive, on a global 0.25 degree regular
    latitude/longitude grid.

    Parameters
    ----------
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/ecmwf-aifs-single-forecast/
    - https://stac.dynamical.org/ecmwf-aifs-single-forecast/collection.json

    Badges
    ------
    region:global dataclass:analysis product:wind product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(self, cache: bool = True, verbose: bool = True) -> None:
        super().__init__("ecmwf-aifs-single-forecast", cache=cache, verbose=verbose)


class DynamicalAIFS_ENS(_DynamicalAnalysis):
    """ECMWF AIFS ENS analysis view from the dynamical.org catalog.

    Lead-time-zero view of the ECMWF Artificial Intelligence Forecasting System
    ensemble forecast archive, on a global 0.25 degree regular
    latitude/longitude grid. A single ensemble member is selected via
    ``member``.

    Parameters
    ----------
    member : int, optional
        Ensemble member index to select, by default 0 (control member).
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/ecmwf-aifs-ens-forecast/
    - https://stac.dynamical.org/ecmwf-aifs-ens-forecast/collection.json

    Badges
    ------
    region:global dataclass:analysis product:wind product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(
        self, member: int = 0, cache: bool = True, verbose: bool = True
    ) -> None:
        super().__init__(
            "ecmwf-aifs-ens-forecast", member=member, cache=cache, verbose=verbose
        )


class DynamicalGFS(_DynamicalBase):
    """NOAA GFS analysis from the dynamical.org catalog.

    Best-estimate analysis (dimensions ``[time, lat, lon]``) built from the
    first hours of successive NOAA Global Forecast System runs, on a global
    0.25 degree regular latitude/longitude grid.

    Parameters
    ----------
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/noaa-gfs-analysis/
    - https://stac.dynamical.org/noaa-gfs-analysis/collection.json

    Badges
    ------
    region:global dataclass:analysis product:wind product:temp product:atmos
    """

    def __init__(self, cache: bool = True, verbose: bool = True) -> None:
        super().__init__("noaa-gfs-analysis", cache=cache, verbose=verbose)

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve analysis data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable(s) to return, in the dynamical.org lexicon or native to the
            collection.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        xr_array = _sync_async(self.fetch, time, variable)
        return xr_array

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to retrieve analysis data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable(s) to return, in the dynamical.org lexicon or native to the
            collection.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        result = await super().fetch(time, timedelta(0), variable)
        return result.isel(lead_time=0, drop=True)


class DynamicalGEFS(_DynamicalBase):
    """NOAA GEFS analysis from the dynamical.org catalog.

    Best-estimate analysis (dimensions ``[time, lat, lon]``) built from the
    first hours of successive NOAA Global Ensemble Forecast System runs, on a
    global 0.25 degree regular latitude/longitude grid.

    Parameters
    ----------
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/noaa-gefs-analysis/
    - https://stac.dynamical.org/noaa-gefs-analysis/collection.json

    Badges
    ------
    region:global dataclass:analysis product:wind product:temp product:atmos
    """

    def __init__(self, cache: bool = True, verbose: bool = True) -> None:
        super().__init__("noaa-gefs-analysis", cache=cache, verbose=verbose)

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve analysis data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable(s) to return, in the dynamical.org lexicon or native to the
            collection.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        xr_array = _sync_async(self.fetch, time, variable)
        return xr_array

    async def fetch(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async function to retrieve analysis data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            Variable(s) to return, in the dynamical.org lexicon or native to the
            collection.

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        result = await super().fetch(time, timedelta(0), variable)
        return result.isel(lead_time=0, drop=True)


class DynamicalHRRR(_DynamicalAnalysis):
    """NOAA HRRR analysis from the dynamical.org catalog.

    Best-estimate analysis from the NOAA High-Resolution Rapid Refresh model,
    on the native 3 km Lambert conformal CONUS grid. Data are returned on
    projection dimensions ``[y, x]`` with 2D ``lat``/``lon`` coordinates.

    Parameters
    ----------
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/noaa-hrrr-analysis/
    - https://stac.dynamical.org/noaa-hrrr-analysis/collection.json

    Badges
    ------
    region:na dataclass:analysis product:wind product:precip product:temp product:atmos
    """

    def __init__(self, cache: bool = True, verbose: bool = True) -> None:
        super().__init__("noaa-hrrr-analysis", cache=cache, verbose=verbose)


class DynamicalMRMS(_DynamicalAnalysis):
    """NOAA MRMS CONUS analysis from the dynamical.org catalog.

    Hourly NOAA Multi-Radar/Multi-Sensor precipitation analyses on a CONUS 0.01
    degree regular latitude/longitude grid.

    Parameters
    ----------
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/noaa-mrms-conus-analysis-hourly/
    - https://stac.dynamical.org/noaa-mrms-conus-analysis-hourly/collection.json

    Badges
    ------
    region:na dataclass:analysis product:precip product:radar
    """

    def __init__(self, cache: bool = True, verbose: bool = True) -> None:
        super().__init__(
            "noaa-mrms-conus-analysis-hourly", cache=cache, verbose=verbose
        )


class DynamicalGFS_FX(_DynamicalBase):
    """NOAA GFS forecast from the dynamical.org catalog.

    Deterministic NOAA Global Forecast System forecasts (dimensions
    ``[time, lead_time, lat, lon]``) on a global 0.25 degree regular
    latitude/longitude grid.

    Parameters
    ----------
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/noaa-gfs-forecast/
    - https://stac.dynamical.org/noaa-gfs-forecast/collection.json

    Badges
    ------
    region:global dataclass:simulation product:wind product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(self, cache: bool = True, verbose: bool = True) -> None:
        super().__init__("noaa-gfs-forecast", cache=cache, verbose=verbose)


class DynamicalHRRR_FX(_DynamicalBase):
    """NOAA HRRR 48-hour virtual forecast from the dynamical.org catalog.

    NOAA High-Resolution Rapid Refresh forecasts on the native 3 km Lambert
    conformal CONUS grid. Data are returned on projection dimensions
    ``[y, x]`` with 2D ``lat``/``lon`` coordinates.

    Parameters
    ----------
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour-virtual/
    - https://stac.dynamical.org/noaa-hrrr-forecast-48-hour-virtual/collection.json

    Badges
    ------
    region:na dataclass:simulation product:wind product:precip product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(self, cache: bool = True, verbose: bool = True) -> None:
        super().__init__(
            "noaa-hrrr-forecast-48-hour-virtual", cache=cache, verbose=verbose
        )


class DynamicalICON_EU_FX(_DynamicalBase):
    """DWD ICON-EU 5-day forecast from the dynamical.org catalog.

    Deutscher Wetterdienst ICON-EU forecasts on a regional Europe 0.0625 degree
    regular latitude/longitude grid, out to 5 days.

    Parameters
    ----------
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/dwd-icon-eu-forecast-5-day/
    - https://stac.dynamical.org/dwd-icon-eu-forecast-5-day/collection.json

    Badges
    ------
    region:eu dataclass:simulation product:wind product:precip product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(self, cache: bool = True, verbose: bool = True) -> None:
        super().__init__("dwd-icon-eu-forecast-5-day", cache=cache, verbose=verbose)


class DynamicalGEFS_FX(_DynamicalBase):
    """NOAA GEFS (35 day) ensemble forecast from the dynamical.org catalog.

    NOAA Global Ensemble Forecast System forecasts (dimensions
    ``[time, lead_time, lat, lon]``) on a global regular latitude/longitude
    grid, out to 35 days from the 00 UTC initialization. A single ensemble
    member is selected via ``member``.

    Parameters
    ----------
    member : int, optional
        Ensemble member index to select, by default 0 (control member).
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/noaa-gefs-forecast-35-day/
    - https://stac.dynamical.org/noaa-gefs-forecast-35-day/collection.json

    Badges
    ------
    region:global dataclass:simulation product:wind product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(
        self, member: int = 0, cache: bool = True, verbose: bool = True
    ) -> None:
        super().__init__(
            "noaa-gefs-forecast-35-day", member=member, cache=cache, verbose=verbose
        )


class DynamicalIFS_ENS(_DynamicalAnalysis):
    """ECMWF IFS ENS analysis view from the dynamical.org catalog.

    Lead-time-zero view of the ECMWF Integrated Forecasting System ensemble
    forecast archive, on a global 0.25 degree regular latitude/longitude grid.
    A single ensemble member is selected via ``member``.

    Parameters
    ----------
    member : int, optional
        Ensemble member index to select, by default 0 (control member).
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/
    - https://stac.dynamical.org/ecmwf-ifs-ens-forecast-15-day-0-25-degree/collection.json

    Badges
    ------
    region:global dataclass:analysis product:wind product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(
        self, member: int = 0, cache: bool = True, verbose: bool = True
    ) -> None:
        super().__init__(
            "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
            member=member,
            cache=cache,
            verbose=verbose,
        )


class DynamicalIFS_ENS_FX(_DynamicalBase):
    """ECMWF IFS ENS (15 day, 0.25 degree) ensemble forecast from dynamical.org.

    ECMWF Integrated Forecasting System ensemble forecasts (dimensions
    ``[time, lead_time, lat, lon]``) on a global 0.25 degree regular
    latitude/longitude grid, out to 15 days from the 00 UTC initialization. A
    single ensemble member is selected via ``member``.

    Parameters
    ----------
    member : int, optional
        Ensemble member index to select, by default 0 (control member).
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/ecmwf-ifs-ens-forecast-15-day-0-25-degree/
    - https://stac.dynamical.org/ecmwf-ifs-ens-forecast-15-day-0-25-degree/collection.json

    Badges
    ------
    region:global dataclass:simulation product:wind product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(
        self, member: int = 0, cache: bool = True, verbose: bool = True
    ) -> None:
        super().__init__(
            "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
            member=member,
            cache=cache,
            verbose=verbose,
        )


class DynamicalAIFS_FX(_DynamicalBase):
    """ECMWF AIFS Single forecast from the dynamical.org catalog.

    Deterministic ECMWF Artificial Intelligence Forecasting System (AIFS)
    forecasts (dimensions ``[time, lead_time, lat, lon]``) on a global 0.25
    degree regular latitude/longitude grid, out to 15 days at a 6 hourly step.

    Parameters
    ----------
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/ecmwf-aifs-single-forecast/
    - https://stac.dynamical.org/ecmwf-aifs-single-forecast/collection.json

    Badges
    ------
    region:global dataclass:simulation product:wind product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(self, cache: bool = True, verbose: bool = True) -> None:
        super().__init__("ecmwf-aifs-single-forecast", cache=cache, verbose=verbose)


class DynamicalAIFSENS_FX(_DynamicalBase):
    """ECMWF AIFS ENS ensemble forecast from the dynamical.org catalog.

    Ensemble ECMWF Artificial Intelligence Forecasting System (AIFS ENS)
    forecasts (dimensions ``[time, lead_time, lat, lon]``) on a global 0.25
    degree regular latitude/longitude grid, out to 15 days at a 6 hourly step.
    A single ensemble member is selected via ``member``.

    Parameters
    ----------
    member : int, optional
        Ensemble member index to select, by default 0 (control member).
    cache : bool, optional
        Retained for API parity; Icechunk reads chunks lazily on demand rather
        than caching whole files locally, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of
    data to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://dynamical.org/catalog/ecmwf-aifs-ens-forecast/
    - https://stac.dynamical.org/ecmwf-aifs-ens-forecast/collection.json

    Badges
    ------
    region:global dataclass:simulation product:wind product:temp product:atmos
    """

    _TIME_DIMENSION = "init_time"

    def __init__(
        self, member: int = 0, cache: bool = True, verbose: bool = True
    ) -> None:
        super().__init__(
            "ecmwf-aifs-ens-forecast", member=member, cache=cache, verbose=verbose
        )
