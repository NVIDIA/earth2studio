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
from earth2studio.data.utils import _is_forecast_source

try:
    import cupy as cp
except ImportError:
    cp = None


class CompositeSource:
    """Data or forecast source assembled from components that provide disjoint
    variables.

    Each requested variable is fetched from the one component that provides it, and
    the parts are joined along ``variable`` in request order. Components must
    return fields on the same coordinates apart from ``variable``; wrap components
    on other grids in :class:`~earth2studio.data.RegriddedSource` with a shared
    target grid.

    Parameters
    ----------
    sources : Mapping[str, DataSource | ForecastSource]
        Named component sources, all analysis sources or all forecast sources.
    variables : Mapping[str, Sequence[str]]
        Variables provided by each named component. A variable may be provided by
        only one component.

    Raises
    ------
    ValueError
        If ``variables`` names an unknown component or assigns a variable twice.
    TypeError
        If analysis and forecast components are mixed.

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
                if variable in routes:
                    raise ValueError(
                        f"Variable {variable!r} is provided by both "
                        f"{routes[variable]!r} and {name!r}"
                    )
                routes[variable] = name
        kinds = {_is_forecast_source(source) for source in sources.values()}
        if len(kinds) > 1:
            raise TypeError("Components must all be analysis or all forecast sources")
        self._forecast = kinds.pop() if kinds else False
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
        """Whether the components are forecast sources."""
        return self._forecast

    @property
    def device(self) -> torch.device:
        """Device joined fields are returned on: the first non-CPU component's."""
        for source in self._sources.values():
            device = torch.device(getattr(source, "device", "cpu"))
            if device.type != "cpu":
                return device
        return torch.device("cpu")

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
        unknown = [name for name in requested if name not in self._routes]
        if unknown:
            raise KeyError(
                f"Variables {unknown} are not provided by any component "
                f"(known: {sorted(self._routes)})"
            )
        groups: dict[str, list[str]] = {}
        for name in dict.fromkeys(requested):
            groups.setdefault(self._routes[name], []).append(name)
        return requested, groups

    def __call__(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Fetch each component's variables and join them in request order.

        Parameters
        ----------
        *args, **kwargs : Any
            ``(time, variable)`` for analysis components or
            ``(time, lead_time, variable)`` for forecast components.

        Returns
        -------
        xr.DataArray
            Requested variables, in request order.
        """
        request = self._bind(args, kwargs)
        requested, groups = self._plan(request["variable"])
        parts = [
            self._sources[name](*self._call_args(request, names))
            for name, names in groups.items()
        ]
        return _join(parts, requested)

    async def fetch(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Asynchronously fetch every component concurrently.

        Components without an async ``fetch`` run in a worker thread.

        Parameters
        ----------
        *args, **kwargs : Any
            ``(time, variable)`` for analysis components or
            ``(time, lead_time, variable)`` for forecast components.

        Returns
        -------
        xr.DataArray
            Requested variables, in request order.
        """
        request = self._bind(args, kwargs)
        requested, groups = self._plan(request["variable"])

        async def _component(name: str, names: list[str]) -> xr.DataArray:
            source = self._sources[name]
            call_args = self._call_args(request, names)
            if hasattr(source, "fetch"):
                return await source.fetch(*call_args)
            return await asyncio.to_thread(source, *call_args)

        parts = await asyncio.gather(
            *(_component(name, names) for name, names in groups.items())
        )
        return _join(list(parts), requested)

    def _call_args(self, request: dict[str, Any], names: list[str]) -> tuple[Any, ...]:
        variable = np.array(names)
        if self._forecast:
            return request["time"], request["lead_time"], variable
        return request["time"], variable


def _join(parts: list[xr.DataArray], requested: list[str]) -> xr.DataArray:
    """Join component fields along ``variable`` on a common backing and device."""
    parts = _common_backing(parts)
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
    return field.sel(variable=requested)


def _common_backing(parts: list[xr.DataArray]) -> list[xr.DataArray]:
    """Move NumPy parts onto the device of the first CuPy part, if any."""
    if cp is None:
        return parts
    devices = [
        part.data.device.id for part in parts if isinstance(part.data, cp.ndarray)
    ]
    if not devices:
        return parts
    with cp.cuda.Device(devices[0]):
        return [
            (
                part
                if isinstance(part.data, cp.ndarray)
                and part.data.device.id == devices[0]
                else part.copy(data=cp.array(part.data))
            )
            for part in parts
        ]
