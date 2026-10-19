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

"""Data source that routes variables to component sources."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.data.utils import _is_forecast_source, _offset_times
from earth2studio.utils.coords import E2S_STATISTICS


class CompositeSource:
    """Data or forecast source assembled from components that provide disjoint
    variables.

    Each requested variable is fetched from the one component that provides it, and
    the parts are joined along ``variable`` in request order. Labels qualified with a
    temporal statistic, such as ``"t2m:mean:24h"``, route by their base name and are
    forwarded unchanged to the component providing it. Components must
    return fields on the same coordinates apart from ``variable``; wrap components
    on other grids in :class:`~earth2studio.data.RegriddedSource` with a shared
    target grid.

    A composite of analysis sources is an analysis source. With any forecast
    component it is a forecast source, and fetches its analysis components at the
    valid times ``time + lead_time``, labelled by the requested times and leads.

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
    >>> source = CompositeSource(
    ...     {"radar": RegriddedSource(MRMS(), radar_regridder),
    ...      "lightning": RegriddedSource(GOESGLMGrid(), glm_regridder)},
    ...     {"radar": ["refc"], "lightning": ["flashcount"]},
    ... )
    >>> field = source(time, ["refc", "flashcount"])
    """

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
        self._analysis = {
            name for name, source in sources.items() if not _is_forecast_source(source)
        }
        self._forecast = len(self._analysis) < len(sources)
        devices = {
            torch.device(getattr(source, "device", "cpu"))
            for source in sources.values()
        }
        if len(devices) > 1:
            raise ValueError(
                f"Components must return fields on one device, not {sorted(map(str, devices))}"
            )
        self._device = devices.pop() if devices else torch.device("cpu")
        self._sources = MappingProxyType(dict(sources))
        self._routes = routes

    @property
    def sources(self) -> Mapping[str, DataSource | ForecastSource]:
        """Named component sources."""
        return self._sources

    @property
    def variables(self) -> Mapping[str, tuple[str, ...]]:
        """Variables provided by each named component."""
        provided: dict[str, list[str]] = {name: [] for name in self._sources}
        for variable, name in self._routes.items():
            provided[name].append(variable)
        return MappingProxyType({k: tuple(v) for k, v in provided.items()})

    @property
    def is_forecast(self) -> bool:
        """Whether any component is a forecast source."""
        return self._forecast

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

    def _bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        names = (
            ("time", "lead_time", "variable")
            if self._forecast
            else ("time", "variable")
        )
        if len(args) > len(names):
            raise TypeError(f"Expected at most {len(names)} positional arguments")
        request = dict(zip(names, args))
        for key, value in kwargs.items():
            if key not in names or key in request:
                raise TypeError(f"Unexpected or repeated argument {key!r}")
            request[key] = value
        missing = set(names).difference(request)
        if missing:
            raise TypeError(f"Missing arguments: {sorted(missing)}")
        return request

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

    def __call__(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Fetch each component's variables and join them in request order.

        Parameters
        ----------
        *args, **kwargs : Any
            ``(time, variable)`` for an analysis composite or
            ``(time, lead_time, variable)`` for a forecast composite.

        Returns
        -------
        xr.DataArray
            Requested variables, in request order.
        """
        request = self._bind(args, kwargs)
        requested, groups = self._plan(request["variable"])
        parts = []
        for name, names in groups.items():
            calls = self._calls(name, request, names)
            fields = [self._sources[name](*call) for call in calls]
            parts.append(self._assemble(name, request, fields))
        return _join(parts, requested)

    async def fetch(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Asynchronously fetch every component concurrently.

        Components without an async ``fetch`` run in a worker thread.

        Parameters
        ----------
        *args, **kwargs : Any
            ``(time, variable)`` for an analysis composite or
            ``(time, lead_time, variable)`` for a forecast composite.

        Returns
        -------
        xr.DataArray
            Requested variables, in request order.
        """
        request = self._bind(args, kwargs)
        requested, groups = self._plan(request["variable"])

        async def _call(source: Any, call: tuple[Any, ...]) -> xr.DataArray:
            if hasattr(source, "fetch"):
                return await source.fetch(*call)
            return await asyncio.to_thread(source, *call)

        async def _component(name: str, names: list[str]) -> xr.DataArray:
            calls = self._calls(name, request, names)
            fields = await asyncio.gather(
                *(_call(self._sources[name], call) for call in calls)
            )
            return self._assemble(name, request, list(fields))

        parts = await asyncio.gather(
            *(_component(name, names) for name, names in groups.items())
        )
        return _join(list(parts), requested)

    def _calls(
        self, name: str, request: dict[str, Any], names: list[str]
    ) -> list[tuple[Any, ...]]:
        """Component call arguments; one call per lead for adapted analyses."""
        variable = np.array(names)
        if not self._forecast:
            return [(request["time"], variable)]
        if name not in self._analysis:
            return [(request["time"], request["lead_time"], variable)]
        time = np.asarray(request["time"])
        return [
            (_offset_times(time, lead), variable)
            for lead in np.atleast_1d(request["lead_time"])
        ]

    def _assemble(
        self, name: str, request: dict[str, Any], fields: list[xr.DataArray]
    ) -> xr.DataArray:
        """Label adapted analysis fields by requested time and lead time."""
        if not self._forecast or name not in self._analysis:
            return fields[0]
        time = np.asarray(request["time"])
        leads = []
        for lead, field in zip(np.atleast_1d(request["lead_time"]), fields):
            field = field.sel(time=_offset_times(time, lead)).assign_coords(time=time)
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
