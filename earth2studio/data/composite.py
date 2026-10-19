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


"""Data and forecast sources that route variables to component sources."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.data.utils import _is_forecast_source, _offset_times
from earth2studio.utils.coords import E2S_STATISTICS
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray


class _Composite:
    """Variable routing, device and cadence checks shared by the composites."""

    def __init__(
        self,
        sources: Mapping[str, DataSource | ForecastSource],
        variables: Mapping[str, Sequence[str]],
    ) -> None:
        unknown = set(variables).difference(sources)
        if unknown:
            raise ValueError(f"Variables name unknown sources: {sorted(unknown)}")
        routes: dict[str, str] = {}
        for name, provided in variables.items():
            for variable in map(str, np.atleast_1d(provided)):
                if ":" in variable:
                    raise ValueError(
                        f"Declare base variable names; {variable!r} is qualified"
                    )
                if variable in routes:
                    raise ValueError(
                        f"Variable {variable!r} is provided by both "
                        f"{routes[variable]!r} and {name!r}"
                    )
                routes[variable] = name
        devices = {_device(source) for source in sources.values()}
        if len(devices) > 1:
            raise ValueError(
                f"Components must return fields on one device, not {sorted(map(str, devices))}"
            )
        self._device = devices.pop() if devices else torch.device("cpu")
        self._sources = dict(sources)
        self._routes = routes

    @property
    def sources(self) -> dict[str, DataSource | ForecastSource]:
        """Named component sources."""
        return dict(self._sources)

    @property
    def variables(self) -> dict[str, tuple[str, ...]]:
        """Variables provided by each named component."""
        provided: dict[str, list[str]] = {name: [] for name in self._sources}
        for variable, name in self._routes.items():
            provided[name].append(variable)
        return {name: tuple(names) for name, names in provided.items()}

    @property
    def device(self) -> torch.device:
        """Device shared by every component, CPU for sources without one."""
        return self._device

    @property
    def time_step(self) -> np.timedelta64 | None:
        """Cadence shared by every component, or None when they differ."""
        steps = [
            getattr(source, "time_step", None) for source in self._sources.values()
        ]
        if not steps or any(step is None for step in steps):
            return None
        first = steps[0]
        return first if all(step == first for step in steps) else None

    def _plan(self, variable: Any) -> tuple[list[str], dict[str, list[str]]]:
        requested = [str(name) for name in np.atleast_1d(variable)]
        # Route qualified labels by base name; the component handles the statistic.
        unknown = [name for name in requested if _base(name) not in self._routes]
        if unknown:
            raise KeyError(
                f"Variables {unknown} are not provided by any component "
                f"(known: {sorted(self._routes)})"
            )
        groups: dict[str, list[str]] = {}
        for name in dict.fromkeys(requested):
            groups.setdefault(self._routes[_base(name)], []).append(name)
        return requested, groups


class CompositeSource(_Composite):
    """Data source assembled from analysis sources that provide disjoint variables.

    Each requested variable is fetched from the one component that provides it, and
    the parts are joined along ``variable`` in request order. Labels qualified with a
    temporal statistic, such as ``"t2m:mean:24h"``, route by their base name and are
    forwarded unchanged to the component providing it. Components must return
    fields on the same coordinates apart from ``variable``; wrap components on other
    grids in :class:`~earth2studio.data.RegriddedSource` with a shared target grid.
    Use :class:`CompositeForecastSource` to include forecast sources.

    Parameters
    ----------
    sources : Mapping[str, DataSource]
        Named analysis component sources, on one ``device``.
    variables : Mapping[str, Sequence[str]]
        Base variable names provided by each named component, without statistic
        qualifiers. A variable may be provided by only one component.

    Raises
    ------
    ValueError
        If ``variables`` names an unknown component, assigns a variable twice or
        contains a qualified label, or components return fields on different
        devices.
    TypeError
        If a component is a forecast source.

    Examples
    --------
    >>> source = CompositeSource(
    ...     {"radar": RegriddedSource(MRMS(), radar_regridder),
    ...      "lightning": RegriddedSource(GOESGLMGrid(), glm_regridder)},
    ...     {"radar": ["refc"], "lightning": ["flashcount"]},
    ... )
    >>> field = source(time, ["refc", "flashcount"])
    """

    def __init__(
        self,
        sources: Mapping[str, DataSource],
        variables: Mapping[str, Sequence[str]],
    ) -> None:
        forecast = [name for name, s in sources.items() if _is_forecast_source(s)]
        if forecast:
            raise TypeError(
                f"Components {forecast} are forecast sources; "
                "use CompositeForecastSource"
            )
        super().__init__(sources, variables)

    def __call__(self, time: TimeArray, variable: VariableArray) -> xr.DataArray:
        """Fetch each component's variables and join them in request order.

        Parameters
        ----------
        time : TimeArray
            Timestamps to return data for (UTC).
        variable : VariableArray
            Variables to return, optionally qualified with a temporal statistic.

        Returns
        -------
        xr.DataArray
            Requested variables, in request order.
        """
        requested, groups = self._plan(variable)
        parts = [
            self._sources[name](time, np.array(names))  # type: ignore[call-arg]
            for name, names in groups.items()
        ]
        return _join(parts, requested)

    async def fetch(self, time: TimeArray, variable: VariableArray) -> xr.DataArray:
        """Asynchronously fetch every component concurrently.

        Components without an async ``fetch`` run in a worker thread.

        Parameters
        ----------
        time : TimeArray
            Timestamps to return data for (UTC).
        variable : VariableArray
            Variables to return, optionally qualified with a temporal statistic.

        Returns
        -------
        xr.DataArray
            Requested variables, in request order.
        """
        requested, groups = self._plan(variable)
        parts = await asyncio.gather(
            *(
                _fetch(self._sources[name], time, np.array(names))
                for name, names in groups.items()
            )
        )
        return _join(list(parts), requested)


class CompositeForecastSource(_Composite):
    """Forecast source assembled from components that provide disjoint variables.

    Routes and joins variables as :class:`CompositeSource` does. Forecast components
    receive each request unchanged. Analysis components are fetched at the valid
    times ``time + lead_time``, and their fields are labelled by the requested times
    and lead times, so observations can join forecasts.

    Parameters
    ----------
    sources : Mapping[str, DataSource | ForecastSource]
        Named analysis or forecast component sources, on one ``device``.
    variables : Mapping[str, Sequence[str]]
        Base variable names provided by each named component, without statistic
        qualifiers. A variable may be provided by only one component.

    Raises
    ------
    ValueError
        If ``variables`` names an unknown component, assigns a variable twice or
        contains a qualified label, or components return fields on different
        devices.

    Examples
    --------
    >>> source = CompositeForecastSource(
    ...     {"radar": RegriddedSource(MRMS(), radar_regridder),
    ...      "conditioning": RegriddedSource(GFS_FX(), gfs_regridder)},
    ...     {"radar": ["refc"], "conditioning": ["u10m", "v10m"]},
    ... )
    >>> field = source(time, lead_time, ["refc", "u10m", "v10m"])
    """

    def __init__(
        self,
        sources: Mapping[str, DataSource | ForecastSource],
        variables: Mapping[str, Sequence[str]],
    ) -> None:
        super().__init__(sources, variables)
        self._analysis = {
            name for name, source in sources.items() if not _is_forecast_source(source)
        }

    def __call__(
        self, time: TimeArray, lead_time: LeadTimeArray, variable: VariableArray
    ) -> xr.DataArray:
        """Fetch each component's variables and join them in request order.

        Parameters
        ----------
        time : TimeArray
            Initialization timestamps (UTC).
        lead_time : LeadTimeArray
            Lead times to return for each initialization.
        variable : VariableArray
            Variables to return, optionally qualified with a temporal statistic.

        Returns
        -------
        xr.DataArray
            Requested variables, in request order.
        """
        requested, groups = self._plan(variable)
        parts = []
        for name, names in groups.items():
            source = self._sources[name]
            if name in self._analysis:
                fields = [
                    source(valid, np.array(names))  # type: ignore[call-arg]
                    for valid in _valid_times(time, lead_time)
                ]
                parts.append(_label_leads(fields, time, lead_time))
            else:
                parts.append(source(time, lead_time, np.array(names)))  # type: ignore[call-arg]
        return _join(parts, requested)

    async def fetch(
        self, time: TimeArray, lead_time: LeadTimeArray, variable: VariableArray
    ) -> xr.DataArray:
        """Asynchronously fetch every component concurrently.

        Components without an async ``fetch`` run in a worker thread.

        Parameters
        ----------
        time : TimeArray
            Initialization timestamps (UTC).
        lead_time : LeadTimeArray
            Lead times to return for each initialization.
        variable : VariableArray
            Variables to return, optionally qualified with a temporal statistic.

        Returns
        -------
        xr.DataArray
            Requested variables, in request order.
        """
        requested, groups = self._plan(variable)

        async def _component(name: str, names: list[str]) -> xr.DataArray:
            source = self._sources[name]
            if name not in self._analysis:
                return await _fetch(source, time, lead_time, np.array(names))
            fields = await asyncio.gather(
                *(
                    _fetch(source, valid, np.array(names))
                    for valid in _valid_times(time, lead_time)
                )
            )
            return _label_leads(list(fields), time, lead_time)

        parts = await asyncio.gather(
            *(_component(name, names) for name, names in groups.items())
        )
        return _join(list(parts), requested)


def _device(source: Any) -> torch.device:
    """Device a source returns fields on, with the CUDA index made explicit."""
    device = torch.device(getattr(source, "device", "cpu"))
    if device.type == "cuda" and device.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        device = torch.device("cuda", index)
    return device


def _fetch(source: Any, *args: Any) -> Awaitable[xr.DataArray]:
    """Await a component's async ``fetch``, or run its call in a worker thread."""
    fetch: Callable[..., Awaitable[xr.DataArray]] | None = getattr(
        source, "fetch", None
    )
    if fetch is not None:
        return fetch(*args)
    return asyncio.to_thread(source, *args)


def _valid_times(time: TimeArray, lead_time: LeadTimeArray) -> list[TimeArray]:
    """Valid times of each lead time, for fetching analysis components."""
    time = np.asarray(time)
    return [_offset_times(time, lead) for lead in np.atleast_1d(lead_time)]


def _label_leads(
    fields: list[xr.DataArray], time: TimeArray, lead_time: LeadTimeArray
) -> xr.DataArray:
    """Label analysis fields fetched at valid times by initialization and lead."""
    time = np.asarray(time)
    leads = []
    for lead, valid, field in zip(
        np.atleast_1d(lead_time), _valid_times(time, lead_time), fields
    ):
        field = field.sel(time=valid).assign_coords(time=time)
        leads.append(field.expand_dims(lead_time=[lead], axis=1))
    return xr.concat(leads, "lead_time")


def _base(label: str) -> str:
    """Base variable name of a possibly qualified label."""
    return label.partition(":")[0]


def _join(parts: list[xr.DataArray], requested: list[str]) -> xr.DataArray:
    """Join component fields along ``variable``."""
    if len(parts) > 1:
        dims = parts[0].dims
        if any(set(part.dims) != set(dims) for part in parts):
            raise ValueError("Component fields must share dimensions")
        parts = [part.transpose(*dims) for part in parts]
        try:
            field = xr.concat(
                parts, "variable", join="exact", coords="minimal", compat="equals"
            )
        except ValueError as error:
            raise ValueError(
                "Component fields must share coordinates other than variable; "
                "regrid components onto a common grid"
            ) from error
    else:
        field = parts[0]
    field = field.sel(variable=requested)
    # Concatenation keeps only the first part's attributes; merge statistics, which
    # are recorded per variable, from every part.
    statistics = {}
    for part in parts:
        statistics.update(part.attrs.get(E2S_STATISTICS, {}))
    field.attrs = dict(field.attrs)
    field.attrs.pop(E2S_STATISTICS, None)
    statistics = {name: statistics[name] for name in requested if name in statistics}
    if statistics:
        field.attrs[E2S_STATISTICS] = statistics
    return field
