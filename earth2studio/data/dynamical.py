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
from tqdm.asyncio import tqdm

from earth2studio.data.utils import (
    _sync_async,
    datasource_cache_root,
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

# Earth2Studio uses 0-360 ascending longitude and 90 -> -90 descending latitude.
# dynamical.org serves -180..180 ascending longitude, so coordinates are
# normalized on open.


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
    pass-through), applying STAC-``unit``-driven conversions to the Earth2Studio
    convention.
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
        # Grid normalization (computed once on open): index arrays that reorder
        # latitude to descending (90 -> -90) and longitude to [0, 360) ascending.
        self._lat: np.ndarray = np.array([])
        self._lon: np.ndarray = np.array([])
        self._lat_idx: np.ndarray = np.array([], dtype=int)
        self._lon_idx: np.ndarray = np.array([], dtype=int)

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
        self._cube_variables = collection.get("cube:variables", {})
        self._cube_dimensions = collection.get("cube:dimensions", {})

        dims = self._cube_dimensions
        if "latitude" not in dims or "longitude" not in dims:
            raise ValueError(
                f"dynamical.org collection {self.collection!r} is not on a regular "
                f"latitude/longitude grid (dimensions: {sorted(dims)}). Projected "
                "datasets (e.g. HRRR, MRMS) are not supported; regrid externally."
            )

        assets = collection.get("assets", {})
        if "icechunk" not in assets:
            raise ValueError(
                f"dynamical.org collection {self.collection!r} has no 'icechunk' asset"
            )
        ds = self._open_icechunk(assets["icechunk"])
        self._setup_grid(ds)
        self._ds = ds
        return self._ds

    def _open_icechunk(self, asset: dict[str, Any]) -> xr.Dataset:
        """Open the Icechunk repository described by a STAC asset.

        Parameters
        ----------
        asset : dict[str, Any]
            The collection's ``icechunk`` STAC asset.

        Returns
        -------
        xr.Dataset
            Lazily opened dataset backed by the Icechunk session store.
        """
        href = asset["href"]
        parsed = urlparse(href)
        if parsed.scheme != "s3":
            raise ValueError(
                f"dynamical.org collection {self.collection!r} icechunk asset href "
                f"is not an s3 url: {href!r}"
            )
        storage_options = asset.get("xarray:storage_options", {})
        region = storage_options.get("client_kwargs", {}).get("region_name")
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
            for entry in asset.get("icechunk:virtual_chunk_containers", []) or []
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

    def _setup_grid(self, ds: xr.Dataset) -> None:
        """Compute index arrays that normalize to Earth2Studio's grid convention.

        Earth2Studio uses latitude descending (90 -> -90) and longitude in
        [0, 360) ascending. Reordering is applied as cheap numpy indexing on the
        small fetched arrays (see :meth:`_reorder`) rather than ``sortby`` on the
        lazy dataset, which would force reading the entire store.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset opened from dynamical.org.
        """
        lat = np.asarray(ds["latitude"].values)
        lon = np.asarray(ds["longitude"].values)
        self._lat_idx = np.argsort(-lat, kind="stable")
        # TODO(regional): the unconditional 0-360 wrap tears a meridian-crossing
        # regional domain (e.g. DWD ICON-EU, lon -23.5..62.5) into two spatially
        # discontiguous halves. Only global collections are supported today; when
        # adding regional ones, make this wrap conditional on global coverage and
        # keep the native contiguous order otherwise.
        lon_mod = lon % 360
        self._lon_idx = np.argsort(lon_mod, kind="stable")
        self._lat = lat[self._lat_idx]
        self._lon = lon_mod[self._lon_idx]

    def _reorder(self, arr: np.ndarray) -> np.ndarray:
        """Reorder the trailing (latitude, longitude) axes to E2Studio convention.

        Parameters
        ----------
        arr : np.ndarray
            Array whose last two axes are (latitude, longitude) in the store's
            native order.

        Returns
        -------
        np.ndarray
            Array with latitude descending and longitude in [0, 360) ascending.
        """
        return arr[..., self._lat_idx, :][..., self._lon_idx]

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
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(
                tzinfo=None
            )

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

    def _coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the normalized (latitude, longitude) coordinate arrays."""
        return self._lat, self._lon

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
            Data array with dimensions ``[time, lead_time, variable, lat, lon]``.
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
            Data array with dimensions ``[time, lead_time, variable, lat, lon]``.
        """
        ds = self._open()
        times, lead_times, variables = prep_forecast_inputs(time, lead_time, variable)
        self._validate_time(times, self._TIME_DIMENSION)

        lat, lon = self._coords()
        times_np = np.array(times, dtype="datetime64[ns]")
        leads_np = np.array(lead_times, dtype="timedelta64[ns]")
        has_lead_time = "lead_time" in ds.dims
        xr_array = xr.DataArray(
            data=np.empty(
                (len(times), len(lead_times), len(variables), len(lat), len(lon)),
                dtype=np.float32,
            ),
            dims=["time", "lead_time", "variable", "lat", "lon"],
            coords={
                "time": times_np,
                "lead_time": leads_np,
                "variable": variables,
                "lat": lat,
                "lon": lon,
            },
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
                da = da.transpose(
                    self._TIME_DIMENSION, "lead_time", "latitude", "longitude"
                )
            else:
                da = ds[dynamical_name].sel({self._TIME_DIMENSION: times_np})
                da = da.transpose(self._TIME_DIMENSION, "latitude", "longitude")
                # Add lead_time axis for uniform output shape
                da = da.expand_dims("lead_time", axis=1)
            xr_array[:, :, j] = modifier(self._reorder(np.asarray(da.values)))

        return xr_array


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


class DynamicalIFSENS_FX(_DynamicalBase):
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
