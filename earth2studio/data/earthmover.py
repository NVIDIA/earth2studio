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
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import NoReturn, cast

import numpy as np
import xarray as xr
from loguru import logger

from earth2studio.data.utils import (
    _sync_async,
    datasource_cache_root,
    prep_data_inputs,
    prep_forecast_inputs,
)
from earth2studio.lexicon.base import LexiconType
from earth2studio.lexicon.earthmover import (
    EarthMoverERA5Lexicon,
    EarthMoverIFSInitialConditionLexicon,
    EarthMoverIFSLexicon,
    VariableSpec,
    make_modifier,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import (
    LeadTimeArray,
    TimeArray,
    VariableArray,
)

try:
    import arraylake
except ImportError:
    OptionalDependencyFailure("data")
    arraylake = None  # type: ignore[assignment]


# Candidate coordinate names for the various physical axes. Marketplace datasets
# follow CF, so axis attributes are preferred, with these names as a fallback.
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_TIME_NAMES = ("time", "valid_time", "init_time", "forecast_reference_time")
_LEAD_NAMES = ("lead_time", "step", "prediction_timedelta", "forecast_period")
_VERTICAL_NAMES = (
    "pressure_level",
    "isobaricInhPa",
    "isobaric",
    "level",
    "plev",
    "lev",
)
_SOIL_NAMES = ("soil_level", "soilLayer", "soil_layer", "depthBelowLandLayer")
_API_KEY_ENV_VAR = "EARTHMOVER_API_KEY"  # noqa: S105


@dataclass
class _Resolved:
    """Result of resolving an Earth2Studio variable against a repository."""

    spec: VariableSpec
    dataset: xr.Dataset
    var_name: str
    level_selection: dict[str, float]
    modifier: Callable
    rule: str  # which metadata rule matched (for diagnostics)


class _EarthMoverBase:
    """Shared Earthmover Arraylake connection and resolution logic."""

    LEXICON: LexiconType = EarthMoverIFSLexicon

    def __init__(
        self,
        repo: str,
        group: str | list[str] | None = None,
        branch: str = "main",
        client: arraylake.AsyncClient | None = None,
        cache: bool = True,
        verbose: bool = True,
        marketplace_url: str | None = None,
    ) -> None:
        self._repo_name = repo
        if group is None or isinstance(group, str):
            self._groups: list[str | None] = [group]
        else:
            self._groups = list(group)
        self._branch = branch
        self._client = client
        self._cache = cache
        self._verbose = verbose
        self._marketplace_url = marketplace_url

        # Populated lazily on first connect.
        self._datasets: list[xr.Dataset] | None = None
        # repo variable name -> (dataset, variable metadata)
        self._index: dict[str, list[tuple[xr.Dataset, xr.DataArray]]] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _make_client(self) -> arraylake.AsyncClient:
        """Create an Arraylake client from an injected client or env API key.

        Returns
        -------
        arraylake.AsyncClient
            Authenticated client. Uses an injected ``client`` first, then the
            API key stored in ``EARTHMOVER_API_KEY``.
        """
        if self._client is not None:
            return self._client
        api_key = os.environ.get(_API_KEY_ENV_VAR)
        if api_key:
            return arraylake.AsyncClient(token=api_key)
        marketplace = (
            f" Subscribe on {self._marketplace_url}." if self._marketplace_url else ""
        )
        raise ValueError(
            f"Set {_API_KEY_ENV_VAR} with an Earthmover / Arraylake API key "
            f"before accessing repo '{self._repo_name}'.{marketplace}"
        )

    async def _connect(self) -> None:
        """Open the repository's group(s) and build the resolution index."""
        if self._datasets is not None:
            return

        client = self._make_client()
        try:
            repo = await client.get_repo(self._repo_name)
            session = repo.readonly_session(branch=self._branch)
        except Exception as err:  # noqa: BLE001 - re-raised with guidance below
            self._raise_access_error(err)

        datasets: list[xr.Dataset] = []
        for group in self._groups:
            try:
                ds = await asyncio.to_thread(
                    xr.open_zarr,
                    session.store,
                    group=group,
                    decode_timedelta=True,
                )
            except Exception as err:  # noqa: BLE001
                self._raise_access_error(err, group=group)
            datasets.append(ds)

        self._datasets = datasets
        self._build_index()

    def _raise_access_error(self, err: Exception, group: str | None = None) -> NoReturn:
        """Translate access failures into actionable guidance.

        Permission / not-found errors on a Marketplace repo usually mean the user
        has not yet created a subscription on the listing page.
        """
        msg = str(err).lower()
        loc = f"{self._repo_name}" + (f" (group {group!r})" if group else "")
        if any(k in msg for k in ("403", "forbidden", "permission", "not authorized")):
            raise PermissionError(
                f"Access to Arraylake repo '{loc}' was denied. If this is an "
                "Earthmover Marketplace dataset, you must first create a "
                "subscription on the dataset's listing page "
                "(https://www.earthmover.io/marketplace), then ensure you are "
                f"authenticated by setting {_API_KEY_ENV_VAR}."
            ) from err
        if any(k in msg for k in ("404", "not found", "does not exist")):
            raise ValueError(
                f"Arraylake repo '{loc}' was not found. Check the 'org/repo' name "
                "and (for Marketplace data) that your subscription is active."
            ) from err
        raise err

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """Index every data variable across opened groups by its native name."""
        self._index = {}
        for ds in self._datasets or []:
            for name, da in ds.data_vars.items():
                self._index.setdefault(str(name), []).append((ds, da))

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_coord(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
        """Find a coordinate/dimension by candidate names (case-insensitive)."""
        lookup = {str(c).lower(): str(c) for c in ds.variables}
        for n in names:
            if n.lower() in lookup:
                return lookup[n.lower()]
        return None

    @staticmethod
    def _find_dim_coord(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
        """Find a 1-D coordinate that can be used as an index dimension."""
        lookup = {str(c).lower(): str(c) for c in ds.variables}
        for n in names:
            coord = lookup.get(n.lower())
            if coord is not None and coord in ds.sizes and ds[coord].ndim == 1:
                return coord
        return None

    @classmethod
    def _vertical_coord(cls, da: xr.DataArray) -> str | None:
        """Return the pressure coordinate on ``da``, if any.

        Detected via CF ``axis='Z'`` / ``standard_name='air_pressure'`` first,
        falling back to common coordinate names.
        """
        for coord in da.coords:
            attrs = da[coord].attrs
            if attrs.get("axis") == "Z" or attrs.get("standard_name") == "air_pressure":
                return str(coord)
        for coord in da.coords:
            if str(coord).lower() in _VERTICAL_NAMES:
                return str(coord)
        return None

    @classmethod
    def _soil_coord(cls, da: xr.DataArray) -> str | None:
        """Return the soil-level coordinate on ``da``, if any."""
        for coord in da.coords:
            name = str(coord)
            if name in _SOIL_NAMES or name.lower() in _SOIL_NAMES:
                return name
        return None

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def _candidate_rule(self, da: xr.DataArray, spec: VariableSpec) -> str | None:
        """Return the matching rule name for ``da`` against ``spec``, else None.

        Priority: ECMWF ``GRIB_paramId`` -> ``GRIB_shortName`` / cfVarName /
        variable name -> CF ``standard_name``.
        """
        attrs = da.attrs
        param_id = attrs.get("GRIB_paramId")
        if param_id is not None and spec.param_id is not None:
            try:
                if int(param_id) == spec.param_id:
                    return "paramId"
            except (TypeError, ValueError):
                pass
        source_names = {
            str(attrs.get("GRIB_shortName", "")),
            str(attrs.get("GRIB_cfVarName", "")),
            str(da.name),
        }
        target_names = {spec.short_name, *spec.aliases}
        if source_names & target_names:
            return "shortName"
        sn = attrs.get("standard_name", "")
        if spec.standard_name and sn == spec.standard_name:
            return "standardName"
        return None

    def _resolve(self, variable: str) -> _Resolved:
        """Resolve an Earth2Studio variable id to a repository variable.

        Parameters
        ----------
        variable : str
            Earth2Studio variable id (e.g. ``t2m``, ``z500``).

        Returns
        -------
        _Resolved
            The matched dataset, variable name, level selection and unit modifier.

        Raises
        ------
        ValueError
            If ``variable`` is unknown to Earth2Studio, or no repository variable
        satisfies its metadata descriptor.
        """
        try:
            specs = cast(dict[str, VariableSpec], getattr(self.LEXICON, "SPECS"))
            spec = specs[variable]
        except KeyError:
            raise ValueError(
                f"'{variable}' is not a known Earth2Studio variable id."
            ) from None

        want_pressure_level = spec.level_type == "isobaric"
        want_soil_level = spec.level_type == "soil" and spec.level is not None
        # Collect candidates by rule priority across all groups.
        ranked: list[tuple[int, _Resolved]] = []
        rule_order = {"paramId": 0, "shortName": 1, "standardName": 2}
        for ds, da in (item for items in self._index.values() for item in items):
            rule = self._candidate_rule(da, spec)
            if rule is None:
                continue
            vcoord = self._vertical_coord(da)
            scoord = self._soil_coord(da)
            if want_pressure_level and vcoord is None:
                continue
            if not want_pressure_level and vcoord is not None and ds[vcoord].size > 1:
                # A multi-level field cannot satisfy a surface request.
                continue

            level_selection: dict[str, float] = {}
            if want_pressure_level:
                if vcoord is None or spec.level is None:
                    continue
                sel = self._level_value(ds[vcoord], spec.level)
                if sel is None:
                    continue
                level_selection[vcoord] = sel
            elif want_soil_level and scoord is not None:
                if spec.level is None:
                    continue
                sel = self._level_value(ds[scoord], spec.level)
                if sel is None:
                    continue
                level_selection[scoord] = sel

            resolved = _Resolved(
                spec=spec,
                dataset=ds,
                var_name=str(da.name),
                level_selection=level_selection,
                modifier=make_modifier(spec, da.attrs.get("units")),
                rule=rule,
            )
            ranked.append((rule_order[rule], resolved))

        if not ranked:
            raise ValueError(self._unresolved_message(variable, spec))

        ranked.sort(key=lambda t: t[0])
        best_tier = ranked[0][0]
        best = [r for tier, r in ranked if tier == best_tier]
        distinct = {r.var_name for r in best}
        if len(distinct) > 1:
            # Several repo variables match equally well (typically by
            # standard_name only, which cannot disambiguate e.g. wind height).
            # Refuse to guess rather than return the wrong field.
            raise ValueError(
                f"Earth2Studio variable '{variable}' is ambiguous in repo "
                f"'{self._repo_name}': it matched {sorted(distinct)} equally by "
                f"'{best[0].rule}'. Add a distinguishing GRIB_paramId / "
                "GRIB_shortName attribute or a height/level coordinate to the "
                "dataset to resolve it unambiguously."
            )
        return best[0]

    @staticmethod
    def _level_value(coord: xr.DataArray, level_hpa: float) -> float | None:
        """Return the coordinate value matching ``level_hpa`` (unit-aware), or None.

        Handles pressure coordinates stored in hPa or Pa.
        """
        values = np.asarray(coord.values, dtype="float64")
        unit = str(coord.attrs.get("units", "")).strip().lower()
        target = level_hpa * 100.0 if unit in {"pa", "pascal", "pascals"} else level_hpa
        match = np.isclose(values, target, rtol=1e-3, atol=1e-6)
        if not match.any():
            return None
        return float(values[match][0])

    def _unresolved_message(self, variable: str, spec: VariableSpec) -> str:
        """Build a diagnostic error listing what was searched and what exists."""
        available = sorted(self._index.keys())
        preview = ", ".join(available[:40])
        if len(available) > 40:
            preview += ", ..."
        return (
            f"Could not resolve Earth2Studio variable '{variable}' in repo "
            f"'{self._repo_name}'. Searched by "
            f"GRIB_paramId={spec.param_id or '(unknown)'}, "
            f"GRIB_shortName/cfVarName/name={sorted({spec.short_name, *spec.aliases})}, "
            "and "
            f"standard_name='{spec.standard_name or '(none)'}'"
            + (
                f" at pressure level {int(spec.level)} hPa"
                if spec.level_type == "isobaric" and spec.level is not None
                else ""
            )
            + f". Available repo variables: {preview}. To make this dataset "
            "resolvable, ensure variables carry GRIB_paramId / GRIB_shortName "
            "or CF standard_name attributes."
        )

    @staticmethod
    def _standardize_longitude(da: xr.DataArray) -> xr.DataArray:
        """Roll native -180..180 longitudes to Earth2Studio's 0..360 convention."""
        lon = np.asarray(da.lon.values, dtype="float64")
        if lon.size and np.isclose(lon[0], -180.0):
            da = da.roll(lon=-(lon.size // 2), roll_coords=True)
            da = da.assign_coords(lon=np.mod(np.asarray(da.lon.values), 360.0))
        return da

    # ------------------------------------------------------------------
    # Per-variable fetch (shared by analysis & forecast)
    # ------------------------------------------------------------------
    def _select_variable(
        self,
        resolved: _Resolved,
        time_sel: np.ndarray,
        extra_sel: dict[str, np.ndarray] | None = None,
    ) -> xr.DataArray:
        """Select, normalize coords and apply the unit modifier for one variable.

        Returns a DataArray with horizontal dims renamed to ``lat``/``lon`` and the
        time (and optional lead-time) axis selected.
        """
        ds = resolved.dataset
        da = ds[resolved.var_name]

        lat = self._find_coord(ds, _LAT_NAMES)
        lon = self._find_coord(ds, _LON_NAMES)
        if lat is None or lon is None:
            raise ValueError(
                f"Repo '{self._repo_name}' variable '{resolved.var_name}' has no "
                "recognizable latitude/longitude coordinates."
            )
        if ds[lat].ndim != 1 or ds[lon].ndim != 1:
            raise ValueError(
                f"Repo '{self._repo_name}' uses a non-regular (projected) grid "
                f"(latitude/longitude are {ds[lat].ndim}-D). Earth2Studio data "
                "sources require a regular 1-D lat/lon grid; regrid the dataset "
                "before use."
            )

        time_coord = self._find_dim_coord(ds, _TIME_NAMES)
        if time_coord is None:
            raise ValueError(
                f"Repo '{self._repo_name}' has no recognizable time coordinate."
            )

        selection: dict[str, np.ndarray | float] = {time_coord: time_sel}
        selection.update(resolved.level_selection)
        if extra_sel:
            selection.update(extra_sel)
        da = da.sel(selection)

        # Apply metadata-driven unit conversion.
        da = resolved.modifier(da)

        # Drop the (now scalar) level coordinate and rename horizontal axes.
        for coord in list(resolved.level_selection):
            if coord in da.coords:
                da = da.drop_vars(coord)
        rename = {lat: "lat", lon: "lon", time_coord: "time"}
        da = da.rename({k: v for k, v in rename.items() if k != v})
        da = self._standardize_longitude(da)
        return da

    # ------------------------------------------------------------------
    # Cache property (Arraylake/Icechunk manages remote caching internally)
    # ------------------------------------------------------------------
    @property
    def cache(self) -> str:
        """Local cache location (Arraylake reads lazily via Icechunk)."""
        cache_location = os.path.join(datasource_cache_root(), "earthmover")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp")
        return cache_location

    async def _available(self, time: datetime | np.datetime64) -> bool:
        """Check whether ``time`` is present in the repository's time axis.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check.

        Returns
        -------
        bool
            True if the timestamp is present in the repository's time coordinate.

        Note
        ----
        Unlike most Earth2Studio sources this is an *instance* method, because
        availability depends on the specific repository being wrapped.
        """
        await self._connect()
        t64 = np.datetime64(time)
        for ds in self._datasets or []:
            time_coord = self._find_dim_coord(ds, _TIME_NAMES)
            if time_coord is None:
                continue
            if (ds[time_coord].values.astype("datetime64[ns]") == t64).any():
                return True
        return False

    def available(self, time: datetime | np.datetime64) -> bool:
        """Synchronous wrapper for :meth:`_available`."""
        return _sync_async(self._available, time)

    async def _validate_times(self, time_list: list[datetime]) -> None:
        """Raise if any requested time is absent from the repository."""
        missing = [t for t in time_list if not await self._available(t)]
        if missing:
            raise ValueError(
                f"Requested times not available in '{self._repo_name}': {missing}"
            )


@check_optional_dependencies()
class EarthMoverERA5(_EarthMoverBase):
    """Earthmover ERA5 0.25 degree reanalysis data source.

    Parameters
    ----------
    repo : str, optional
        Arraylake repository name as ``org/repo``. When omitted, derives the
        repo from ``EARTHMOVER_ORGANIZATION`` as
        ``<org>/era5-subscription``, by default None.
    branch : str, optional
        Repository branch to read, by default "main".
    client : arraylake.AsyncClient, optional
        Pre-authenticated Arraylake async client. When omitted, this data source
        uses the API key stored in ``EARTHMOVER_API_KEY``, by default None.
    cache : bool, optional
        Retained for API compatibility; Arraylake reads lazily via Icechunk, by
        default True.
    verbose : bool, optional
        Print progress, by default True.

    Warning
    -------
    This is a remote data source and can download a large amount of data for large
    requests.

    Note
    ----
    Configure Earthmover access before using this data source:

    - Set the ``EARTHMOVER_API_KEY`` environment variable to an Earthmover /
      Arraylake API key, unless passing a pre-authenticated ``client``.
    - Pass ``repo="org/repo"`` to open a specific Arraylake repository.
    - If ``repo`` is omitted, set the ``EARTHMOVER_ORGANIZATION`` environment
      variable. The default repository is
      ``<EARTHMOVER_ORGANIZATION>/era5-subscription``.

    The data source opens the ``single/spatial`` and ``pressure/spatial``
    groups from the ERA5 Marketplace repository. Arraylake-backed Earthmover
    sources require Python 3.12 or newer.

    Additional information on the data repository can be referenced here:

    - https://app.earthmover.io/marketplace/6a19bcfe9aa6e97720a2fad2

    Badges
    ------
    region:global dataclass:reanalysis product:wind product:precip product:temp product:atmos product:solar
    """

    LEXICON = EarthMoverERA5Lexicon
    MARKETPLACE_URL = "https://app.earthmover.io/marketplace/6a19bcfe9aa6e97720a2fad2"
    ORG_ENV_VAR = "EARTHMOVER_ORGANIZATION"
    SUBSCRIPTION_REPO_NAME = "era5-subscription"
    DEFAULT_BRANCH = "main"
    TIME_START = datetime(year=1940, month=1, day=1)
    VARIABLES = tuple(EarthMoverERA5Lexicon.VOCAB)

    def __init__(
        self,
        repo: str | None = None,
        branch: str = DEFAULT_BRANCH,
        client: arraylake.AsyncClient | None = None,
        cache: bool = True,
        verbose: bool = True,
    ) -> None:
        repo_name = repo
        if repo_name is None:
            org_name = os.environ.get(self.ORG_ENV_VAR)
            if org_name:
                repo_name = f"{org_name}/{self.SUBSCRIPTION_REPO_NAME}"
        if repo_name is None:
            raise ValueError(
                f"Pass repo='org/repo' or set {self.ORG_ENV_VAR} to derive "
                f"'<org>/{self.SUBSCRIPTION_REPO_NAME}'. Listing: "
                f"{self.MARKETPLACE_URL}"
            )
        super().__init__(
            repo_name,
            group=["single/spatial", "pressure/spatial"],
            branch=branch,
            client=client,
            cache=cache,
            verbose=verbose,
            marketplace_url=self.MARKETPLACE_URL,
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve ERA5 data for times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Analysis timestamps (UTC).
        variable : str | list[str] | VariableArray
            Earth2Studio variable id(s).

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        return _sync_async(self.fetch, time, variable)

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for ERA5 based on offline knowledge."""
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for ERA5"
                )

            if time < cls.TIME_START:
                raise ValueError(
                    f"Requested date time {time} needs to be after January 1st, 1940 "
                    "for ERA5"
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Checks if given date time is available in the ERA5 data source.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to access.

        Returns
        -------
        bool
            If date time is available.
        """
        time_list, _ = prep_data_inputs(time, "")
        try:
            cls._validate_time(time_list)
        except ValueError:
            return False
        return True

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve ERA5 data for times and variables asynchronously.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Analysis timestamps (UTC).
        variable : str | list[str] | VariableArray
            Earth2Studio variable id(s).

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        await self._connect()

        time_sel = np.array(time_list, dtype="datetime64[ns]")

        arrays = []
        for v in variable_list:
            resolved = self._resolve(v)
            if self._verbose:
                logger.debug(
                    f"{v} -> {self._repo_name}:{resolved.var_name} "
                    f"(matched by {resolved.rule})"
                )
            da = self._select_variable(resolved, time_sel)
            arrays.append(da.transpose("time", "lat", "lon"))

        out = xr.concat(arrays, dim="variable")
        out = out.assign_coords(variable=variable_list).transpose(
            "time", "variable", "lat", "lon"
        )
        return await asyncio.to_thread(out.load)


@check_optional_dependencies()
class EarthMoverBrightBandIFS(_EarthMoverBase):
    """Brightband ECMWF IFS 0.1 degree (10km) initial-condition data source on
    Earthmover Arraylake.

    Parameters
    ----------
    repo : str, optional
        Arraylake repository name as ``org/repo``. When omitted, derives the
        repo from ``EARTHMOVER_ORGANIZATION`` as
        ``<org>/ecmwf-ifs-initial-conditions-open-subscription``, by default None.
    branch : str, optional
        Repository branch to read, by default "main".
    client : arraylake.AsyncClient, optional
        Pre-authenticated Arraylake async client. When omitted, this data source
        uses the API key stored in ``EARTHMOVER_API_KEY``, by default None.
    cache : bool, optional
        Retained for API compatibility; Arraylake reads lazily via Icechunk, by
        default True.
    verbose : bool, optional
        Print progress, by default True.

    Warning
    -------
    This is a remote data source and can download a large amount of data for large
    requests.

    Note
    ----
    Configure Earthmover access before using this data source:

    - Set the ``EARTHMOVER_API_KEY`` environment variable to an Earthmover /
      Arraylake API key, unless passing a pre-authenticated ``client``.
    - Pass ``repo="org/repo"`` to open a specific Arraylake repository.
    - If ``repo`` is omitted, set the ``EARTHMOVER_ORGANIZATION`` environment
      variable. The default repository is
      ``<EARTHMOVER_ORGANIZATION>/ecmwf-ifs-initial-conditions-open-subscription``.

    Arraylake-backed Earthmover sources require Python 3.12 or newer.

    Additional information on the data repository can be referenced here:

    - https://app.earthmover.io/marketplace/697162921880507a6587c31b

    Badges
    ------
    region:global dataclass:analysis product:wind product:temp product:atmos product:ocean product:land
    """

    MARKETPLACE_URL = "https://app.earthmover.io/marketplace/697162921880507a6587c31b"
    DEFAULT_BRANCH = "main"
    ORG_ENV_VAR = "EARTHMOVER_ORGANIZATION"
    SUBSCRIPTION_REPO_NAME = "ecmwf-ifs-initial-conditions-open-subscription"
    LEXICON = EarthMoverIFSInitialConditionLexicon
    VARIABLES = tuple(EarthMoverIFSInitialConditionLexicon.VOCAB)

    def __init__(
        self,
        repo: str | None = None,
        branch: str = DEFAULT_BRANCH,
        client: arraylake.AsyncClient | None = None,
        cache: bool = True,
        verbose: bool = True,
    ) -> None:
        repo_name = repo
        if repo_name is None:
            org_name = os.environ.get(self.ORG_ENV_VAR)
            if org_name:
                repo_name = f"{org_name}/{self.SUBSCRIPTION_REPO_NAME}"
        if repo_name is None:
            raise ValueError(
                f"Pass repo='org/repo' or set {self.ORG_ENV_VAR} to derive "
                f"'<org>/{self.SUBSCRIPTION_REPO_NAME}'. Listing: "
                f"{self.MARKETPLACE_URL}"
            )
        super().__init__(
            repo_name,
            group=None,
            branch=branch,
            client=client,
            cache=cache,
            verbose=verbose,
            marketplace_url=self.MARKETPLACE_URL,
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve IFS analysis data for times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Analysis timestamps (UTC).
        variable : str | list[str] | VariableArray
            Earth2Studio variable id(s).

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        return _sync_async(self.fetch, time, variable)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve IFS analysis data for times and variables asynchronously.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Analysis timestamps (UTC).
        variable : str | list[str] | VariableArray
            Earth2Studio variable id(s).

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, variable, lat, lon]``.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        await self._connect()
        await self._validate_times(time_list)

        time_sel = np.array(time_list, dtype="datetime64[ns]")

        arrays = []
        for v in variable_list:
            resolved = self._resolve(v)
            if self._verbose:
                logger.debug(
                    f"{v} -> {self._repo_name}:{resolved.var_name} "
                    f"(matched by {resolved.rule})"
                )
            da = self._select_variable(resolved, time_sel)
            squeeze_dims = [
                dim
                for dim, size in da.sizes.items()
                if dim not in ("time", "lat", "lon") and size == 1
            ]
            if squeeze_dims:
                da = da.squeeze(dim=squeeze_dims, drop=True)
            arrays.append(da.transpose("time", "lat", "lon"))

        out = xr.concat(arrays, dim="variable")
        out = out.assign_coords(variable=variable_list).transpose(
            "time", "variable", "lat", "lon"
        )
        return await asyncio.to_thread(out.load)


@check_optional_dependencies()
class EarthMoverBrightBandIFS_FX(_EarthMoverBase):
    """Brightband ECMWF IFS 0.1 degree (10km) 15-day forecast data source on Earthmover
    Arraylake.

    Parameters
    ----------
    repo : str, optional
        Arraylake repository name as ``org/repo``. When omitted, derives the
        repo from ``EARTHMOVER_ORGANIZATION`` as
        ``<org>/ecmwf-ifs-15-day-forecast-open-subscription``, by default None.
    branch : str, optional
        Repository branch to read, by default "main".
    client : arraylake.AsyncClient, optional
        Pre-authenticated Arraylake async client. When omitted, this data source
        uses the API key stored in ``EARTHMOVER_API_KEY``, by default None.
    cache : bool, optional
        Retained for API compatibility; Arraylake reads lazily via Icechunk, by
        default True.
    verbose : bool, optional
        Print progress, by default True.

    Warning
    -------
    This is a remote data source and can download a large amount of data for large
    requests.

    Note
    ----
    Configure Earthmover access before using this data source:

    - Set the ``EARTHMOVER_API_KEY`` environment variable to an Earthmover /
      Arraylake API key, unless passing a pre-authenticated ``client``.
    - Pass ``repo="org/repo"`` to open a specific Arraylake repository.
    - If ``repo`` is omitted, set the ``EARTHMOVER_ORGANIZATION`` environment
      variable. The default repository is
      ``<EARTHMOVER_ORGANIZATION>/ecmwf-ifs-15-day-forecast-open-subscription``.

    Arraylake-backed Earthmover sources require Python 3.12 or newer.

    Additional information on the data repository can be referenced here:

    - https://app.earthmover.io/marketplace/6971be98fc964a0d0fb66e04

    Badges
    ------
    region:global dataclass:forecast product:wind product:precip product:temp product:atmos
    """

    MARKETPLACE_URL = "https://app.earthmover.io/marketplace/6971be98fc964a0d0fb66e04"
    DEFAULT_BRANCH = "main"
    ORG_ENV_VAR = "EARTHMOVER_ORGANIZATION"
    SUBSCRIPTION_REPO_NAME = "ecmwf-ifs-15-day-forecast-open-subscription"
    LEXICON = EarthMoverIFSLexicon
    DATASET_VARIABLES = (
        "100u",
        "100v",
        "10u",
        "10v",
        "2d",
        "2t",
        "cp",
        "fdir",
        "hcc",
        "lcc",
        "mcc",
        "msl",
        "sd",
        "ssrd",
        "tp",
    )
    VARIABLES = tuple(EarthMoverIFSLexicon.VOCAB)

    def __init__(
        self,
        repo: str | None = None,
        branch: str = DEFAULT_BRANCH,
        client: arraylake.AsyncClient | None = None,
        cache: bool = True,
        verbose: bool = True,
    ) -> None:
        repo_name = repo
        if repo_name is None:
            org_name = os.environ.get(self.ORG_ENV_VAR)
            if org_name:
                repo_name = f"{org_name}/{self.SUBSCRIPTION_REPO_NAME}"
        if repo_name is None:
            raise ValueError(
                f"Pass repo='org/repo' or set {self.ORG_ENV_VAR} to derive "
                f"'<org>/{self.SUBSCRIPTION_REPO_NAME}'. Listing: "
                f"{self.MARKETPLACE_URL}"
            )
        super().__init__(
            repo_name,
            group=None,
            branch=branch,
            client=client,
            cache=cache,
            verbose=verbose,
            marketplace_url=self.MARKETPLACE_URL,
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve IFS forecast data for init times, lead times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Forecast initialization timestamps (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times.
        variable : str | list[str] | VariableArray
            Earth2Studio variable id(s).

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, lead_time, variable, lat, lon]``.
        """
        return _sync_async(self.fetch, time, lead_time, variable)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve IFS forecast data asynchronously.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Forecast initialization timestamps (UTC).
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times.
        variable : str | list[str] | VariableArray
            Earth2Studio variable id(s).

        Returns
        -------
        xr.DataArray
            Data array with dimensions ``[time, lead_time, variable, lat, lon]``.
        """
        time_list, lead_list, variable_list = prep_forecast_inputs(
            time, lead_time, variable
        )
        await self._connect()
        await self._validate_times(time_list)

        time_sel = np.array(time_list, dtype="datetime64[ns]")
        lead_sel = np.array(lead_list, dtype="timedelta64[ns]")

        arrays = []
        for v in variable_list:
            resolved = self._resolve(v)
            lead_coord = self._find_coord(resolved.dataset, _LEAD_NAMES)
            if lead_coord is None:
                raise ValueError(
                    f"Repo '{self._repo_name}' has no lead-time/step coordinate."
                )
            if self._verbose:
                logger.debug(
                    f"{v} -> {self._repo_name}:{resolved.var_name} "
                    f"(matched by {resolved.rule})"
                )
            da = self._select_variable(
                resolved, time_sel, extra_sel={lead_coord: lead_sel}
            )
            da = da.rename({lead_coord: "lead_time"})
            arrays.append(da.transpose("time", "lead_time", "lat", "lon"))

        out = xr.concat(arrays, dim="variable")
        out = out.assign_coords(variable=variable_list).transpose(
            "time", "lead_time", "variable", "lat", "lon"
        )
        return await asyncio.to_thread(out.load)
