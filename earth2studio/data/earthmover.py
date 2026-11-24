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
import re
from collections.abc import Callable, Hashable
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
from earth2studio.grids import with_grid_metadata
from earth2studio.lexicon.base import LexiconType
from earth2studio.lexicon.earthmover import (
    EarthMoverCustomLexicon,
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
_HEIGHT_NAMES = ("height", "heightaboveground", "height_above_ground", "altitude")
_API_KEY_ENV_VAR = "EARTHMOVER_API_KEY"  # noqa: S105


@dataclass
class _Resolved:
    """Result of resolving an Earth2Studio variable against a repository."""

    spec: VariableSpec
    dataset: xr.Dataset
    var_name: str
    level_selection: dict[str, float]
    modifier: Callable
    rule: str  # which rule matched (for diagnostics)


class _ArraylakeStore:
    """Opens the groups of one Earthmover Arraylake repository as datasets.

    Parameters
    ----------
    repo : str
        Arraylake repository name as ``org/repo``.
    group : str | list[str] | None
        Zarr group(s) to open. None opens the repository root.
    branch : str
        Repository branch to read.
    client : arraylake.AsyncClient | None
        Pre-authenticated client. When None the API key stored in
        ``EARTHMOVER_API_KEY`` is used.
    marketplace_url : str | None
        Marketplace listing mentioned in authentication errors.
    """

    def __init__(
        self,
        repo: str,
        group: str | list[str] | None,
        branch: str,
        client: arraylake.AsyncClient | None,
        marketplace_url: str | None,
    ) -> None:
        self.repo = repo
        if group is None or isinstance(group, str):
            self.groups: list[str | None] = [group]
        else:
            self.groups = list(group)
            if not self.groups:
                raise ValueError(
                    "group must name at least one group, or be None for the "
                    "repository root."
                )
        self._branch = branch
        self._client = client
        self._marketplace_url = marketplace_url

    def make_client(self) -> arraylake.AsyncClient:
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
            f"before accessing repo '{self.repo}'.{marketplace}"
        )

    async def open(self) -> list[xr.Dataset]:
        """Open every configured group lazily.

        Returns
        -------
        list[xr.Dataset]
            One lazily loaded dataset per configured group, in order.
        """
        client = self.make_client()
        try:
            repo = await client.get_repo(self.repo)
            session = repo.readonly_session(branch=self._branch)
        except Exception as err:  # noqa: BLE001 - re-raised with guidance below
            self._raise_access_error(err)

        datasets: list[xr.Dataset] = []
        for group in self.groups:
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
        return datasets

    def _raise_access_error(self, err: Exception, group: str | None = None) -> NoReturn:
        """Translate access failures into actionable guidance.

        Permission / not-found errors on a Marketplace repo usually mean the user
        has not yet created a subscription on the listing page.
        """
        msg = str(err).lower()
        loc = f"{self.repo}" + (f" (group {group!r})" if group else "")
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


class _DatasetResolver:
    """Resolves Earth2Studio variables against CF / GRIB style datasets.

    Works on already opened ``xarray`` datasets only and has no knowledge of
    where they come from.

    Parameters
    ----------
    datasets : list[xr.Dataset]
        Datasets to resolve variables in, e.g. one per repository group.
    lexicon : LexiconType
        Lexicon providing the variable descriptors (``SPECS``).
    name : str
        Name of the data, used in error messages.
    labels : list[str | None] | None, optional
        Label of each dataset (its group), used in error messages, by default
        None.
    strict : bool, optional
        Strict resolution as used for the Marketplace presets: only lexicon
        variables resolved through metadata. When False, a dataset variable
        named exactly like the requested id takes precedence and ids outside
        the lexicon are allowed, by default True.
    keep_extra_dims : bool, optional
        Keep dimensions other than time, lead time and lat/lon (e.g. ensemble
        members) in the output instead of raising, by default False.
    verbose : bool, optional
        Log how each variable was resolved, by default True.
    """

    def __init__(
        self,
        datasets: list[xr.Dataset],
        lexicon: LexiconType,
        name: str,
        labels: list[str | None] | None = None,
        strict: bool = True,
        keep_extra_dims: bool = False,
        verbose: bool = True,
    ) -> None:
        self._datasets = datasets
        self._specs = cast(dict[str, VariableSpec], getattr(lexicon, "SPECS"))
        self._name = name
        self._strict = strict
        self._keep_extra_dims = keep_extra_dims
        self._verbose = verbose

        for ds, label in zip(datasets, labels or [None] * len(datasets)):
            self._check_dataset(ds, label)
        # dataset variable name -> (dataset, variable metadata), across datasets
        self._index: dict[str, list[tuple[xr.Dataset, xr.DataArray]]] = {}
        for ds in datasets:
            for var_name, da in ds.data_vars.items():
                self._index.setdefault(str(var_name), []).append((ds, da))

    def _check_dataset(self, ds: xr.Dataset, label: str | None) -> None:
        """Fail early, naming the offending coordinate, on unsupported layouts.

        Raises
        ------
        ValueError
            If the time coordinate is not ``datetime64``, latitude/longitude
            are not one-dimensional, or latitude is not monotonic.
        """
        loc = f"'{self._name}'" + (f" (group {label!r})" if label else "")
        time_coord = self._find_dim_coord(ds, _TIME_NAMES)
        if time_coord is not None and not np.issubdtype(
            ds[time_coord].dtype, np.datetime64
        ):
            raise ValueError(
                f"Repo {loc} time coordinate '{time_coord}' has dtype "
                f"{ds[time_coord].dtype}, not datetime64. Non-standard calendars "
                "can be converted with a preprocess function, e.g. "
                "preprocess=lambda ds: ds.convert_calendar('standard', "
                "use_cftime=False); a 360-day calendar also needs "
                "align_on='date'."
            )
        lead_coord = self._find_dim_coord(ds, _LEAD_NAMES)
        if lead_coord is not None and not np.issubdtype(
            ds[lead_coord].dtype, np.timedelta64
        ):
            raise ValueError(
                f"Repo {loc} lead-time coordinate '{lead_coord}' has dtype "
                f"{ds[lead_coord].dtype}, not timedelta64. Decode it in a "
                "preprocess function, e.g. preprocess=lambda ds: "
                f"ds.assign_coords({lead_coord}=pd.to_timedelta("
                f"ds['{lead_coord}'].values, unit='h'))."
            )
        for names in (_LAT_NAMES, _LON_NAMES):
            coord = self._find_coord(ds, names)
            if coord is None:
                continue
            if ds[coord].ndim != 1:
                raise ValueError(
                    f"Repo {loc} uses a non-regular (projected) grid "
                    f"(coordinate '{coord}' is {ds[coord].ndim}-D). Earth2Studio "
                    "data sources require a regular 1-D lat/lon grid; regrid the "
                    "dataset before use."
                )
            if names is _LON_NAMES:
                # Longitude may legitimately wrap (e.g. 0..180, -179..-1).
                continue
            steps = np.diff(np.asarray(ds[coord].values, dtype="float64"))
            if steps.size and not ((steps > 0).all() or (steps < 0).all()):
                raise ValueError(
                    f"Repo {loc} coordinate '{coord}' is not strictly monotonic. "
                    "Sort it with a preprocess function, e.g. "
                    f"preprocess=lambda ds: ds.sortby('{coord}')."
                )

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

    @staticmethod
    def _is_height(da: xr.DataArray, coord: Hashable) -> bool:
        """Whether ``coord`` is a height-above-ground coordinate of ``da``."""
        attrs = da[coord].attrs if coord in da.coords else {}
        return str(coord).lower() in _HEIGHT_NAMES or attrs.get("standard_name") in (
            "height",
            "altitude",
        )

    @classmethod
    def _height_coord(cls, da: xr.DataArray) -> str | None:
        """Return the height-above-ground coordinate on ``da``, if any."""
        for coord in da.coords:
            if cls._is_height(da, coord) and da[coord].ndim <= 1:
                return str(coord)
        return None

    @staticmethod
    def _implied_height(variable: str) -> float | None:
        """Height in metres implied by an id such as ``t2m`` or ``u100m``."""
        match = re.fullmatch(r"[a-z]+?(\d+)m", variable)
        return float(match.group(1)) if match else None

    @classmethod
    def _unselected_levels(
        cls, da: xr.DataArray, level_selection: dict[str, float]
    ) -> list[str]:
        """Multi-level vertical dimensions of ``da`` not covered by a selection.

        Pressure, soil and height dimensions hold different physical fields
        per level, so they must be reduced to one level and are never returned
        as an extra dimension.
        """
        vertical = {cls._vertical_coord(da), cls._soil_coord(da)}
        vertical.update(str(dim) for dim in da.dims if cls._is_height(da, dim))
        return [
            str(dim)
            for dim in da.dims
            if dim in vertical and da.sizes[dim] > 1 and dim not in level_selection
        ]

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

    def resolve(self, variable: str) -> _Resolved:
        """Resolve an Earth2Studio variable id to a dataset variable.

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
            If ``variable`` is unknown to Earth2Studio, or no dataset variable
            satisfies its metadata descriptor.
        """
        if not self._strict and variable in self._index:
            return self._resolve_named(variable)
        try:
            spec = self._specs[variable]
        except KeyError:
            hint = (
                ""
                if self._strict
                else f" To use a repository variable Earth2Studio does not know, "
                f"name it '{variable}' in a preprocess function."
            )
            raise ValueError(
                f"'{variable}' is not a known Earth2Studio variable id.{hint}"
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

            height = self._implied_height(variable)
            hcoord = self._height_coord(da)
            if height is not None and hcoord is not None:
                # ``standard_name`` alone cannot tell a 10 m from a 100 m wind,
                # so a height coordinate on the variable must agree with the id.
                heights = np.atleast_1d(ds[hcoord].values).astype("float64")
                match = np.isclose(heights, height)
                if hcoord in da.dims and heights.size > 1:
                    if not match.any():
                        continue
                    level_selection[hcoord] = float(heights[match][0])
                elif not self._strict and rule == "standardName" and not match.any():
                    # A scalar height applies to every variable of a dataset,
                    # so this can also reject a correct field. That fails
                    # loudly, and naming the variable in preprocess resolves it.
                    continue

            if self._unselected_levels(da, level_selection):
                # e.g. a field on several heights cannot satisfy a variable
                # that does not name one of them.
                continue

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
            # Several dataset variables match equally well (typically by
            # standard_name only, which cannot disambiguate e.g. wind height).
            # Refuse to guess rather than return the wrong field.
            raise ValueError(
                f"Earth2Studio variable '{variable}' is ambiguous in repo "
                f"'{self._name}': it matched {sorted(distinct)} equally by "
                f"'{best[0].rule}'. Add a distinguishing GRIB_paramId / "
                "GRIB_shortName attribute or a height/level coordinate to the "
                "dataset to resolve it unambiguously." + self._preprocess_hint(variable)
            )
        return best[0]

    def _resolve_named(self, variable: str) -> _Resolved:
        """Resolve a dataset variable named exactly like the requested id.

        The name is an explicit choice (typically made in a preprocess
        function), so metadata rules are skipped. Lexicon variables still get
        their level selected and their units converted.

        Raises
        ------
        ValueError
            If the name exists in several datasets, or the variable keeps
            several levels that cannot be reduced automatically.
        """
        items = self._index[variable]
        if len(items) > 1:
            raise ValueError(
                f"Variable '{variable}' exists in {len(items)} groups of repo "
                f"'{self._name}'; list only the group that should provide it."
            )
        ds, da = items[0]
        spec = self._specs.get(variable) or VariableSpec(
            e2s=variable,
            short_name=variable,
            param_id=None,
            standard_name="",
            level_type="surface",
            level=None,
            canonical_units="",
        )

        level_selection: dict[str, float] = {}
        if spec.level is not None and spec.level_type in ("isobaric", "soil"):
            coord = (
                self._vertical_coord(da)
                if spec.level_type == "isobaric"
                else self._soil_coord(da)
            )
            if coord is not None and coord in da.dims and da.sizes[coord] > 1:
                sel = self._level_value(ds[coord], spec.level)
                if sel is not None:
                    level_selection[coord] = sel
        unselected = self._unselected_levels(da, level_selection)
        if unselected:
            raise ValueError(
                f"Variable '{variable}' in repo '{self._name}' has several levels "
                f"along {unselected}. Select one in a preprocess function, e.g. "
                f"ds.assign({variable}=ds['{variable}'].sel({unselected[0]}=<value>))."
            )
        return _Resolved(
            spec=spec,
            dataset=ds,
            var_name=variable,
            level_selection=level_selection,
            modifier=make_modifier(spec, da.attrs.get("units")),
            rule="name",
        )

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

    def _preprocess_hint(self, variable: str) -> str:
        """Advice on resolving ``variable`` explicitly, when that is allowed."""
        if self._strict:
            return ""
        return (
            " Alternatively provide it under the requested name in a preprocess "
            f"function, e.g. preprocess=lambda ds: ds.assign({variable}="
            "ds['<repo variable>'].sel(<level coordinate>=<value>))."
        )

    def _unresolved_message(self, variable: str, spec: VariableSpec) -> str:
        """Build a diagnostic error listing what was searched and what exists."""
        available = sorted(self._index.keys())
        preview = ", ".join(available[:40])
        if len(available) > 40:
            preview += ", ..."
        return (
            f"Could not resolve Earth2Studio variable '{variable}' in repo "
            f"'{self._name}'. Searched by "
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
            "or CF standard_name attributes." + self._preprocess_hint(variable)
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    @staticmethod
    def _standardize_longitude(da: xr.DataArray) -> xr.DataArray:
        """Map negative longitudes to Earth2Studio's 0..360 convention.

        A global axis with negative longitudes is reordered to be ascending.
        A regional axis keeps its stored order, so a region crossing the prime
        meridian stays contiguous. An axis without negative values is returned
        as stored.
        """
        lon = np.asarray(da.lon.values)
        if lon.size == 0 or not (lon < 0).any():
            return da
        wrapped = np.mod(lon, 360.0)
        # A stored "zero" carrying float noise (e.g. -1e-14) wraps to ~360.
        wrapped[np.isclose(wrapped, 360.0, rtol=0.0, atol=1e-6)] = 0.0
        da = da.assign_coords(lon=wrapped.astype(lon.dtype))
        if lon.size > 1 and not (np.diff(wrapped) > 0).all():
            spacing = np.abs(np.diff(lon.astype("float64"))).max()
            if float(lon.max()) - float(lon.min()) + spacing >= 360.0 - 1e-6:
                da = da.sortby("lon")
        return da

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
                f"Repo '{self._name}' variable '{resolved.var_name}' has no "
                "recognizable latitude/longitude coordinates."
            )

        time_coord = self._find_dim_coord(ds, _TIME_NAMES)
        if time_coord is None:
            raise ValueError(
                f"Repo '{self._name}' has no recognizable time coordinate."
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
        rename = {k: v for k, v in rename.items() if k != v}
        # Auxiliary coordinates already using a target name (e.g. a 2-D valid
        # ``time`` next to an ``init_time`` dimension) would block the rename.
        da = da.drop_vars([v for v in rename.values() if v in da.coords])
        da = da.rename(rename)
        da = self._standardize_longitude(da)
        return da

    @staticmethod
    def _is_forecast_step(da: xr.DataArray, dim: Hashable) -> bool:
        """Whether a size-1 dimension is a non-zero forecast lead time.

        A single zero lead (analysis stored on a lead axis) can be dropped
        safely; a single non-zero lead is forecast data and must not be
        returned as an analysis.
        """
        if str(dim).lower() not in _LEAD_NAMES or dim not in da.coords:
            return False
        value = da[dim].values[0]
        return bool(
            np.issubdtype(value.dtype, np.timedelta64)
            and value != np.timedelta64(0, "ns")
        )

    def prepare_variable(
        self,
        variable: str,
        time_sel: np.ndarray,
        lead_sel: np.ndarray | None = None,
    ) -> xr.DataArray:
        """Resolve and select one variable as ``[time, (lead_time), lat, lon]``.

        Parameters
        ----------
        variable : str
            Earth2Studio variable id.
        time_sel : np.ndarray
            Requested (initialization) times.
        lead_sel : np.ndarray | None, optional
            Requested lead times. When given the variable must have a
            lead-time dimension, by default None.

        Returns
        -------
        xr.DataArray
            Lazy array with dimensions ``[time, lat, lon]`` or, with
            ``lead_sel``, ``[time, lead_time, lat, lon]``. Extra dimensions,
            when kept, sit between these and ``lat``.

        Raises
        ------
        ValueError
            If a lead time is requested from a variable without a lead-time
            dimension, a requested lead time is missing, or the variable has
            dimensions that cannot be reduced to the expected ones.
        """
        resolved = self.resolve(variable)
        dims = ["time", "lat", "lon"]
        extra_sel: dict[str, np.ndarray] | None = None
        lead_coord: str | None = None
        if lead_sel is not None:
            lead_coord = self._find_dim_coord(resolved.dataset, _LEAD_NAMES)
            if (
                lead_coord is None
                or lead_coord not in resolved.dataset[resolved.var_name].dims
            ):
                raise ValueError(
                    f"Repo '{self._name}' variable '{resolved.var_name}' "
                    "has no lead-time/step dimension. Use an analysis source "
                    "such as EarthMoverCustomRepo for repositories without a "
                    "lead-time dimension."
                )
            leads = resolved.dataset[lead_coord].values.astype("timedelta64[ns]")
            missing = [lead for lead in lead_sel if lead not in leads]
            if missing:
                raise ValueError(
                    f"Requested lead times not available in '{self._name}': "
                    f"{missing}"
                )
            extra_sel = {lead_coord: lead_sel}
            dims.insert(1, "lead_time")
        if self._verbose:
            logger.debug(
                f"{variable} -> {self._name}:{resolved.var_name} "
                f"(matched by {resolved.rule})"
            )

        da = self._select_variable(resolved, time_sel, extra_sel=extra_sel)
        if lead_coord is not None and lead_coord != "lead_time":
            da = da.rename({lead_coord: "lead_time"})

        # Size-one level and zero-lead dimensions carry no information. Other
        # size-one dimensions (e.g. a single ensemble member) are only dropped
        # when extra dimensions are not kept, so that members of different
        # variables are never mixed.
        levels = {self._vertical_coord(da), self._soil_coord(da)}
        squeeze_dims = [
            dim
            for dim, size in da.sizes.items()
            if dim not in dims
            and size == 1
            and not self._is_forecast_step(da, dim)
            and (
                not self._keep_extra_dims
                or dim in levels
                or self._is_height(da, dim)
                or str(dim).lower() in _LEAD_NAMES
            )
        ]
        if squeeze_dims:
            da = da.squeeze(dim=squeeze_dims, drop=True)
        extra_dims = [str(dim) for dim in da.dims if dim not in dims]
        lead_dims = [d for d in extra_dims if d.lower() in _LEAD_NAMES]
        if lead_dims and lead_sel is None:
            raise ValueError(
                f"Repo '{self._name}' variable '{resolved.var_name}' has "
                f"lead-time dimension(s) {lead_dims}. This looks like a forecast "
                "repository; use a forecast source such as "
                "EarthMoverCustomRepo_FX."
            )
        if "variable" in extra_dims:
            raise ValueError(
                f"Repo '{self._name}' variable '{resolved.var_name}' has a "
                "dimension named 'variable', which is reserved for the output. "
                "Rename it in a preprocess function."
            )
        if extra_dims and not self._keep_extra_dims:
            hint = (
                ""
                if self._strict
                else " Select one entry in a preprocess function, e.g. "
                f"preprocess=lambda ds: ds.isel({extra_dims[0]}=0), or pass "
                "keep_extra_dims=True."
            )
            raise ValueError(
                f"Repo '{self._name}' variable '{resolved.var_name}' has "
                f"unsupported dimension(s) {extra_dims}; expected {dims}.{hint}"
            )
        return da.transpose(*dims[:-2], *extra_dims, "lat", "lon")

    def concat_variables(
        self, arrays: list[xr.DataArray], variables: list[str]
    ) -> xr.DataArray:
        """Stack per-variable arrays along ``variable``, requiring equal coords.

        Variables lacking an extra dimension that others have are broadcast
        along it. Extra dimensions follow the order of the first variable
        that has them.
        """
        first = arrays[0]
        ordered = [first]
        for arr in arrays[1:]:
            # Equal coordinates stored in a different order (e.g. one group
            # north-to-south, another south-to-north) are reordered, not
            # rejected. Anything else is left for the exact join to refuse.
            for dim in arr.dims:
                if dim not in first.indexes or dim not in arr.indexes:
                    continue
                ref, idx = first.indexes[dim], arr.indexes[dim]
                if (
                    not ref.equals(idx)
                    and ref.is_unique
                    and len(ref) == len(idx)
                    and ref.sort_values().equals(idx.sort_values())
                ):
                    arr = arr.sel({dim: ref})
            ordered.append(arr)
        try:
            if len({arr.dims for arr in ordered}) > 1:
                ordered = list(xr.broadcast(*xr.align(*ordered, join="exact")))
            out = xr.concat(ordered, dim="variable", join="exact")
        except ValueError as err:
            raise ValueError(
                f"Variables {variables} in repo '{self._name}' do not share "
                "the same coordinates. All requested variables, including those "
                "from different groups, must be on one latitude/longitude grid "
                "and agree on any extra dimension such as ensemble members."
            ) from err
        return out.assign_coords(variable=variables)

    @property
    def has_time_axis(self) -> bool:
        """Whether any dataset has a recognizable time dimension."""
        return any(
            self._find_dim_coord(ds, _TIME_NAMES) is not None for ds in self._datasets
        )

    def has_time(self, time: datetime | np.datetime64) -> bool:
        """Whether ``time`` is on the time axis of every dataset that has one.

        Datasets without a time dimension (e.g. static fields) do not
        constrain availability.
        """
        t64 = np.datetime64(time)
        has_time_coord = False
        for ds in self._datasets:
            time_coord = self._find_dim_coord(ds, _TIME_NAMES)
            if time_coord is None:
                continue
            has_time_coord = True
            if not (ds[time_coord].values.astype("datetime64[ns]") == t64).any():
                return False
        return has_time_coord


class _EarthMoverSource:
    """Shared plumbing of the Earthmover Arraylake data sources.

    Composes an :py:class:`_ArraylakeStore`, which opens the repository, with
    a :py:class:`_DatasetResolver`, which maps Earth2Studio variables onto the
    opened datasets. Subclasses choose the lexicon and pass their policy
    (``strict``, ``keep_extra_dims``, ``grid``) to the constructor.
    """

    LEXICON: LexiconType = EarthMoverIFSLexicon
    ORG_ENV_VAR = "EARTHMOVER_ORGANIZATION"
    SUBSCRIPTION_REPO_NAME: str
    MARKETPLACE_URL: str | None = None

    @classmethod
    def _derive_repo_name(cls, repo: str | None) -> str:
        """Resolve an explicit repo or derive one from the configured organization."""
        if repo is not None:
            return repo

        org_name = os.environ.get(cls.ORG_ENV_VAR)
        if org_name:
            return f"{org_name}/{cls.SUBSCRIPTION_REPO_NAME}"

        raise ValueError(
            f"Pass repo='org/repo' or set {cls.ORG_ENV_VAR} to derive "
            f"'<org>/{cls.SUBSCRIPTION_REPO_NAME}'. Listing: "
            f"{cls.MARKETPLACE_URL}"
        )

    def __init__(
        self,
        repo: str,
        group: str | list[str] | None = None,
        branch: str = "main",
        client: arraylake.AsyncClient | None = None,
        cache: bool = True,
        verbose: bool = True,
        marketplace_url: str | None = None,
        grid: str | None = None,
        preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
        strict: bool = True,
        keep_extra_dims: bool = False,
    ) -> None:
        self._store = _ArraylakeStore(repo, group, branch, client, marketplace_url)
        self._cache = cache
        self._verbose = verbose
        # Registered grid the repository is known to sit on, if any. None
        # annotates the output with its actual lat/lon axes.
        self._grid = grid
        self._preprocess = preprocess
        self._strict = strict
        self._keep_extra_dims = keep_extra_dims
        # Built lazily on first connect.
        self._resolver: _DatasetResolver | None = None

    @property
    def _repo_name(self) -> str:
        """Arraylake repository name as ``org/repo``."""
        return self._store.repo

    def _make_client(self) -> arraylake.AsyncClient:
        """Create an Arraylake client from an injected client or env API key."""
        return self._store.make_client()

    async def _connect(self) -> _DatasetResolver:
        """Open the repository's group(s) and build the variable resolver."""
        if self._resolver is None:
            datasets = await self._store.open()
            if self._preprocess is not None:
                datasets = [
                    await asyncio.to_thread(self._preprocess, ds) for ds in datasets
                ]
                for ds in datasets:
                    if not isinstance(ds, xr.Dataset):
                        raise TypeError(
                            "preprocess must return an xarray.Dataset, got "
                            f"{type(ds).__name__}."
                        )
            self._resolver = _DatasetResolver(
                datasets,
                self.LEXICON,
                name=self._repo_name,
                labels=self._store.groups,
                strict=self._strict,
                keep_extra_dims=self._keep_extra_dims,
                verbose=self._verbose,
            )
        return self._resolver

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

    def _validate_time(self, times: list[datetime]) -> None:
        """Offline time validation hook for subclasses."""
        pass

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
        resolver = await self._connect()
        return resolver.has_time(time)

    def available(self, time: datetime | np.datetime64) -> bool:
        """Synchronous wrapper for :meth:`_available`."""
        time_list, _ = prep_data_inputs(time, "")
        try:
            self._validate_time(time_list)
        except ValueError:
            return False
        return _sync_async(self._available, time)

    async def _validate_times(self, time_list: list[datetime]) -> None:
        """Raise if any requested time is absent from the repository."""
        resolver = await self._connect()
        if not resolver.has_time_axis:
            raise ValueError(
                f"Repo '{self._repo_name}' has no recognizable time coordinate in "
                f"group(s) {self._store.groups}."
            )
        missing = [t for t in time_list if not resolver.has_time(t)]
        if missing:
            raise ValueError(
                f"Requested times not available in '{self._repo_name}': {missing}"
            )


class _EarthMoverAnalysis(_EarthMoverSource):
    """Analysis (``DataSource``) interface shared by Earthmover sources."""

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve data for times and variables.

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
        """Retrieve data for times and variables asynchronously.

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
        resolver = await self._connect()
        await self._validate_times(time_list)

        time_sel = np.array(time_list, dtype="datetime64[ns]")
        arrays = [resolver.prepare_variable(v, time_sel) for v in variable_list]

        out = resolver.concat_variables(arrays, variable_list).transpose(
            "time", ..., "variable", "lat", "lon"
        )
        return with_grid_metadata(await asyncio.to_thread(out.load), self._grid)


class _EarthMoverForecast(_EarthMoverSource):
    """Forecast (``ForecastSource``) interface shared by Earthmover sources."""

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve forecast data for init times, lead times and variables.

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
        """Retrieve forecast data asynchronously.

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
        self._validate_time(time_list)
        resolver = await self._connect()
        await self._validate_times(time_list)

        time_sel = np.array(time_list, dtype="datetime64[ns]")
        lead_sel = np.array(lead_list, dtype="timedelta64[ns]")
        arrays = [
            resolver.prepare_variable(v, time_sel, lead_sel=lead_sel)
            for v in variable_list
        ]

        out = resolver.concat_variables(arrays, variable_list).transpose(
            "time", "lead_time", ..., "variable", "lat", "lon"
        )
        return with_grid_metadata(await asyncio.to_thread(out.load), self._grid)


@check_optional_dependencies()
class EarthMoverCustomRepo(_EarthMoverAnalysis):
    """Analysis data source for a custom or private Earthmover Arraylake
    repository.

    Use this class for repositories that are not one of the Marketplace presets
    (:py:class:`EarthMoverERA5`, :py:class:`EarthMoverBrightBandIFS`). The
    group layout is given explicitly and only the requested times are checked
    against the repository's time coordinate; no dataset-specific validation
    is applied. For repositories with a lead-time dimension use
    :py:class:`EarthMoverCustomRepo_FX`.

    Parameters
    ----------
    repo : str
        Arraylake repository name as ``org/repo``.
    group : str | list[str] | None, optional
        Zarr group, or groups, to open inside the repository. Variables are
        resolved across all listed groups. None opens the repository root, by
        default None.
    branch : str, optional
        Repository branch to read, by default "main".
    preprocess : Callable[[xr.Dataset], xr.Dataset], optional
        Function applied lazily to each opened group before variables are
        resolved, like ``preprocess`` in ``xarray.open_mfdataset``. Use it to
        name or derive variables, rename coordinates, select an ensemble
        member, sort latitude or convert calendars, by default None.
    keep_extra_dims : bool, optional
        Keep dimensions other than time, level, latitude and longitude, such as
        ensemble members, in front of ``variable``, e.g.
        ``[time, realization, variable, lat, lon]``. Such output does not follow
        the data source protocol shape, by default False.
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
    The repository, after ``preprocess``, must provide a regular
    one-dimensional latitude/longitude grid and a ``datetime64`` time
    coordinate (``time``, ``valid_time``, ``init_time`` or
    ``forecast_reference_time``). Variables are resolved in this order:

    1. A repository variable named exactly like the requested id. This is the
       way to choose a variable explicitly: name or derive it in ``preprocess``,
       e.g. ``lambda ds: ds.assign(u10m=ds["uas"].sel(height=10.0))``. Ids that
       are not Earth2Studio variables are allowed this way and are returned
       without unit conversion.
    2. ``GRIB_paramId``, then ``GRIB_shortName`` / ``GRIB_cfVarName`` /
       variable name, then CF ``standard_name`` metadata. Pressure-level
       variables are selected from a pressure coordinate in hPa or Pa.

    A variable on several heights or levels is never guessed; an error asks
    for an explicit choice.

    All requested variables must share one grid, also across groups, and a
    requested time must exist in every listed group that has a time dimension.
    Latitude is returned in the order it is stored. Negative longitudes are
    mapped to ``[0, 360)``; a global axis is reordered to be ascending.
    Convert non-standard calendars in ``preprocess``, e.g.
    ``lambda ds: ds.convert_calendar("standard", use_cftime=False)``.

    Dimensions other than time, level, latitude and longitude are dropped when
    they have size one. Larger ones raise an error unless
    ``keep_extra_dims=True``; select a single entry in ``preprocess``, e.g.
    ``lambda ds: ds.sel(realization="r1i1p1f1")``, to use the source with
    Earth2Studio models and workflows.

    Set the ``EARTHMOVER_API_KEY`` environment variable to an Earthmover /
    Arraylake API key, unless passing a pre-authenticated ``client``.
    Arraylake-backed Earthmover sources require Python 3.12 or newer.

    Badges
    ------
    region:global dataclass:analysis provider:earthmover
    """

    LEXICON = EarthMoverCustomLexicon

    def __init__(
        self,
        repo: str,
        group: str | list[str] | None = None,
        branch: str = "main",
        preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
        keep_extra_dims: bool = False,
        client: arraylake.AsyncClient | None = None,
        cache: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            repo,
            group=group,
            branch=branch,
            client=client,
            cache=cache,
            verbose=verbose,
            preprocess=preprocess,
            strict=False,
            keep_extra_dims=keep_extra_dims,
        )


@check_optional_dependencies()
class EarthMoverCustomRepo_FX(_EarthMoverForecast):
    """Forecast data source for a custom or private Earthmover Arraylake
    repository.

    Use this class for forecast repositories that are not one of the
    Marketplace presets (:py:class:`EarthMoverBrightBandIFS_FX`). The group
    layout is given explicitly and only the requested initialization and lead
    times are checked against the repository's coordinates. For repositories
    without a lead-time dimension use :py:class:`EarthMoverCustomRepo`.

    Parameters
    ----------
    repo : str
        Arraylake repository name as ``org/repo``.
    group : str | list[str] | None, optional
        Zarr group, or groups, to open inside the repository. Variables are
        resolved across all listed groups. None opens the repository root, by
        default None.
    branch : str, optional
        Repository branch to read, by default "main".
    preprocess : Callable[[xr.Dataset], xr.Dataset], optional
        Function applied lazily to each opened group before variables are
        resolved, like ``preprocess`` in ``xarray.open_mfdataset``. Use it to
        name or derive variables, rename coordinates, select an ensemble
        member, sort latitude or convert calendars, by default None.
    keep_extra_dims : bool, optional
        Keep dimensions other than time, lead time, level, latitude and
        longitude, such as ensemble members, in front of ``variable``, e.g.
        ``[time, lead_time, realization, variable, lat, lon]``. Such output
        does not follow the forecast source protocol shape, by default False.
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
    The repository, after ``preprocess``, must provide a regular
    one-dimensional latitude/longitude grid, a ``datetime64`` initialization
    time coordinate (``time``, ``valid_time``, ``init_time`` or
    ``forecast_reference_time``) and a lead-time dimension (``lead_time``,
    ``step``, ``prediction_timedelta`` or ``forecast_period``). Variables are
    resolved in this order:

    1. A repository variable named exactly like the requested id. This is the
       way to choose a variable explicitly: name or derive it in ``preprocess``,
       e.g. ``lambda ds: ds.assign(u10m=ds["uas"].sel(height=10.0))``. Ids that
       are not Earth2Studio variables are allowed this way and are returned
       without unit conversion.
    2. ``GRIB_paramId``, then ``GRIB_shortName`` / ``GRIB_cfVarName`` /
       variable name, then CF ``standard_name`` metadata. Pressure-level
       variables are selected from a pressure coordinate in hPa or Pa.

    A variable on several heights or levels is never guessed; an error asks
    for an explicit choice.

    All requested variables must share one grid, also across groups, and a
    requested time must exist in every listed group that has a time dimension.
    Latitude is returned in the order it is stored. Negative longitudes are
    mapped to ``[0, 360)``; a global axis is reordered to be ascending.
    Convert non-standard calendars in ``preprocess``, e.g.
    ``lambda ds: ds.convert_calendar("standard", use_cftime=False)``.

    Dimensions other than time, lead time, level, latitude and longitude are
    dropped when they have size one. Larger ones raise an error unless
    ``keep_extra_dims=True``; select a single entry in ``preprocess``, e.g.
    ``lambda ds: ds.sel(realization="r1i1p1f1")``, to use the source with
    Earth2Studio models and workflows.

    Set the ``EARTHMOVER_API_KEY`` environment variable to an Earthmover /
    Arraylake API key, unless passing a pre-authenticated ``client``.
    Arraylake-backed Earthmover sources require Python 3.12 or newer.

    Badges
    ------
    region:global dataclass:simulation provider:earthmover
    """

    LEXICON = EarthMoverCustomLexicon

    def __init__(
        self,
        repo: str,
        group: str | list[str] | None = None,
        branch: str = "main",
        preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
        keep_extra_dims: bool = False,
        client: arraylake.AsyncClient | None = None,
        cache: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            repo,
            group=group,
            branch=branch,
            client=client,
            cache=cache,
            verbose=verbose,
            preprocess=preprocess,
            strict=False,
            keep_extra_dims=keep_extra_dims,
        )


@check_optional_dependencies()
class EarthMoverERA5(_EarthMoverAnalysis):
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

    For a walkthrough of subscribing to a Marketplace listing, authenticating,
    fetching data and writing output back to Arraylake, see the blog post
    [Access Earthmover Marketplace Data with Earth2Studio](https://nvidia.github.io/earth2studio/blog/2026/10/05/earthmover-marketplace-data/).

    Additional information on the data repository can be referenced here:

    - https://app.earthmover.io/marketplace/6a19bcfe9aa6e97720a2fad2

    Badges
    ------
    region:global dataclass:reanalysis dataset:era5 product:wind product:precip product:temp product:atmos product:solar provider:earthmover
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
        super().__init__(
            self._derive_repo_name(repo),
            group=["single/spatial", "pressure/spatial"],
            branch=branch,
            client=client,
            cache=cache,
            verbose=verbose,
            marketplace_url=self.MARKETPLACE_URL,
            grid="latlon-0.25deg",
        )

    def _validate_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for ERA5 based on offline knowledge."""
        for time in times:
            if not (time - datetime(1900, 1, 1)).total_seconds() % 3600 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for ERA5"
                )

            if time < self.TIME_START:
                raise ValueError(
                    f"Requested date time {time} needs to be on or after January 1st, "
                    "1940 for ERA5"
                )


@check_optional_dependencies()
class EarthMoverBrightBandIFS(_EarthMoverAnalysis):
    """Brightband ECMWF IFS 0.25 degree initial-condition data source on
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

    For a walkthrough of subscribing to a Marketplace listing, authenticating,
    fetching data and writing output back to Arraylake, see the blog post
    [Access Earthmover Marketplace Data with Earth2Studio](https://nvidia.github.io/earth2studio/blog/2026/10/05/earthmover-marketplace-data/).

    Additional information on the data repository can be referenced here:

    - https://app.earthmover.io/marketplace/697162921880507a6587c31b

    Badges
    ------
    region:global dataclass:analysis dataset:ifs product:wind product:temp product:atmos product:ocean product:land provider:earthmover
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
        super().__init__(
            self._derive_repo_name(repo),
            group=None,
            branch=branch,
            client=client,
            cache=cache,
            verbose=verbose,
            marketplace_url=self.MARKETPLACE_URL,
            grid="latlon-0.25deg",
        )


@check_optional_dependencies()
class EarthMoverBrightBandIFS_FX(_EarthMoverForecast):
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

    For a walkthrough of subscribing to a Marketplace listing, authenticating,
    fetching data and writing output back to Arraylake, see the blog post
    [Access Earthmover Marketplace Data with Earth2Studio](https://nvidia.github.io/earth2studio/blog/2026/10/05/earthmover-marketplace-data/).

    Additional information on the data repository can be referenced here:

    - https://app.earthmover.io/marketplace/6971be98fc964a0d0fb66e04

    Badges
    ------
    region:global dataclass:simulation dataset:ifs product:wind product:precip product:temp product:atmos provider:earthmover
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
        super().__init__(
            self._derive_repo_name(repo),
            group=None,
            branch=branch,
            client=client,
            cache=cache,
            verbose=verbose,
            marketplace_url=self.MARKETPLACE_URL,
            grid="latlon-0.25deg",
        )
