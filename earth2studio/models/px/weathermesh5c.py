# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Generator, Hashable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.data import IFS_ENS
from earth2studio.data.oisst import OISST
from earth2studio.grids import LatLonGrid
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from weathermesh5c import RuntimeBundle
    from weathermesh5c import load_model as load_core_model
    from weathermesh5c.inference import (
        OPEN_LEVELS,
        PRESSURE_INPUTS,
        SURFACE_INPUTS,
        SURFACE_VARIABLES,
        Inference,
        OpenIFSFields,
    )
except ImportError:
    OptionalDependencyFailure("weathermesh5c")
    Inference = OpenIFSFields = None  # type: ignore[assignment, misc]
    OPEN_LEVELS = PRESSURE_INPUTS = SURFACE_INPUTS = SURFACE_VARIABLES = None  # type: ignore[assignment]

HF_REVISION = "260377d3534a098a5a3369adab309f897a351bc3"
STEP = np.timedelta64(3, "h")


def _load_core(package: Package) -> torch.nn.Module:
    manifest = Path(package.resolve("manifest.json"))
    return load_core_model(RuntimeBundle(manifest.parent, resolve=package.resolve))


@dataclass(frozen=True)
class _State:
    latent: torch.Tensor
    seed: int
    position: int
    coordinates: dict[Hashable, np.ndarray]


def _variables(
    available: tuple[str, ...], selected: Sequence[str] | None
) -> tuple[str, ...]:
    variables = available if selected is None else tuple(selected)
    if (
        not variables
        or len(set(variables)) != len(variables)
        or not set(variables) <= set(available)
    ):
        raise ValueError("Output variables must be a nonempty unique supported subset")
    return variables


def _inputs(runtime: Inference, lead: int = 0) -> tuple[xr.DataArray, xr.DataArray]:
    lat = runtime.input_latitudes
    if len(lat) == 720 and lat[0] == 90:
        lat = np.append(lat, -90)
    base = {
        "time": np.array([], dtype="datetime64[ns]"),
        "lead_time": np.array([lead], dtype="timedelta64[h]"),
    }
    dims = ("batch", "time", "lead_time", "variable", "lat", "lon")
    ifs = coord_array(
        dims,
        {**base, "variable": list(runtime.input_variables)},
        dynamic=("batch", "time"),
        grid=LatLonGrid(lat, runtime.input_longitudes),
    )
    sst = coord_array(
        dims,
        {**base, "variable": ["sst"]},
        dynamic=("batch", "time"),
        grid=LatLonGrid(np.arange(89.875, -90, -0.25), np.arange(0.125, 360, 0.25)),
    )
    return ifs, sst


def _validate(
    x: xr.DataArray, signature: xr.DataArray, *, allow_shift: bool = True
) -> None:
    handshake_time(x, "lead_time")
    if allow_shift:
        x = x.assign_coords(
            lead_time=x.lead_time.values
            - x.lead_time.values[-1]
            + signature.lead_time.values[-1]
        )
    handshake_dataarray(x, signature)
    handshake_time(x, allow_dynamic=True)


def _layout(
    runtime: Inference, x: xr.DataArray, variables: tuple[str, ...], lead: np.ndarray
) -> xr.DataArray:
    prefix = x.dims[:-4]
    return coord_array(
        (*prefix, "lead_time", "variable", "lat", "lon"),
        {
            **{name: x.coords[name] for name in prefix},
            "lead_time": lead,
            "variable": list(variables),
        },
        grid=LatLonGrid(runtime.output_latitudes, runtime.output_longitudes),
        dynamic=tuple(x.attrs.get("earth2studio_dynamic_dims", ())),
    )


def _backgrounds(
    runtime: Inference, x: xr.DataArray, sst: xr.DataArray
) -> Iterator[np.ndarray]:
    handshake_nonempty(x)
    xs, ss = _inputs(runtime)
    _validate(x, xs)
    _validate(sst, ss)
    if x.dims[:-4] != sst.dims[:-4] or any(
        not np.array_equal(x[d], sst[d]) for d in x.dims[:-4]
    ):
        raise ValueError("IFS and SST leading coordinates must match")
    if not np.array_equal(x.lead_time, sst.lead_time):
        raise ValueError("IFS and SST valid times must match")
    selected = x.sel(lat=runtime.input_latitudes, lon=runtime.input_longitudes)
    data = selected.e2s.as_numpy().values.reshape(
        -1,
        len(runtime.input_variables),
        len(runtime.input_latitudes),
        len(runtime.input_longitudes),
    )
    oceans = sst.e2s.as_numpy().values.reshape(-1, sst.sizes["lat"], sst.sizes["lon"])
    for index, values in enumerate(data):
        columns = dict(
            zip(runtime.input_variables, values.astype(np.float32, copy=False))
        )
        fields = OpenIFSFields(
            {
                parameter: {level: columns[f"{name}{level}"] for level in OPEN_LEVELS}
                for name, parameter in PRESSURE_INPUTS.items()
            },
            {parameter: columns[name] for name, parameter in SURFACE_INPUTS.items()},
        )
        ocean = runtime.prepare_sst(oceans[index], sst.lat.values, sst.lon.values)
        yield runtime.prepare_background(fields, ocean)


def _times(x: xr.DataArray) -> list[datetime]:
    axis = x.dims.index("time")
    prefix = x.dims[:-4]
    shape = tuple(x.sizes[name] for name in prefix)
    times = np.broadcast_to(
        x.time.values.reshape(
            tuple(x.sizes["time"] if i == axis else 1 for i in range(len(prefix)))
        ),
        shape,
    ).reshape(-1)
    return [
        datetime.fromtimestamp(
            int((t + x.lead_time.values[-1]).astype("datetime64[s]").astype(np.int64)),
            timezone.utc,
        )
        for t in times
    ]


def _decode(
    runtime: Inference,
    latent: torch.Tensor,
    layout: xr.DataArray,
    variables: tuple[str, ...],
    seed: int = 0,
    position: int = 0,
) -> tuple[xr.DataArray, _State]:
    pressure = any(name in runtime.pressure_variables for name in variables)
    available = runtime.variables if pressure else SURFACE_VARIABLES
    values = runtime.decode(latent, pressure=pressure)[
        :, [available.index(v) for v in variables]
    ]
    values = values.reshape(layout.shape)
    tensor = torch.from_numpy(values).to(runtime.device)
    y = from_torch(tensor, layout)
    coordinates = {dim: y[dim].values.copy() for dim in y.dims[:-3]}
    return y, _State(latent, seed, position, coordinates)


def _step(
    runtime: Inference,
    y: xr.DataArray,
    state: _State,
    variables: tuple[str, ...],
    fork_hour: int | None = None,
) -> tuple[xr.DataArray, _State]:
    if not isinstance(state, _State):
        raise ValueError("Expected the matching WeatherMesh continuation state")
    expected = _layout(runtime, y, variables, y.lead_time.values)
    handshake_dataarray(y, expected)
    if tuple(state.coordinates) != y.dims[:-3] or any(
        not np.array_equal(y[dim], saved) for dim, saved in state.coordinates.items()
    ):
        raise ValueError(
            "Edited forecast coordinates do not match the native latent state"
        )
    latent = runtime.step(state.latent)
    lead = y.lead_time.values + STEP
    position = state.position
    if fork_hour is not None and int(lead[-1] / np.timedelta64(1, "h")) == fork_hour:
        latent = runtime.fork(latent, hour=fork_hour, seed=state.seed + position)
        position += 1
    return _decode(
        runtime,
        latent,
        coord_array_like(y, {"lead_time": lead}),
        variables,
        state.seed,
        position,
    )


def _hook(hook: Callable, y: xr.DataArray) -> xr.DataArray:
    if hook is PrognosticMixin._default_hook:
        return y
    before = y.e2s.as_numpy().copy(deep=True)
    result = hook(y)
    after = result.e2s.as_numpy()
    if not before.coords.identical(after.coords) or (
        before.dims != after.dims
        or before.dtype != after.dtype
        or before.values.tobytes() != after.values.tobytes()
    ):
        raise ValueError("Edited forecast fields cannot update the native latent state")
    return result


def _iterator(model: Any, *inputs: Any) -> Generator[xr.DataArray]:
    y, state = model.initialize(*inputs)
    while True:
        y = _hook(model.rear_hook, y)
        forcing = yield y
        if forcing is not None:
            raise ValueError("WeatherMesh does not accept step forcing")
        y = _hook(model.front_hook, y)
        y, state = model.step(y, state)


@check_optional_dependencies()
class WeatherMesh5C(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """WindBorne's global weather forecast model.

    WeatherMesh-5c starts from ECMWF IFS ensemble fields and NOAA OISST sea
    surface temperature at the initialization time. Every three hours it returns
    surface and pressure-level fields, such as temperature, wind, and precipitation,
    on a global 0.25° latitude-longitude grid (720 × 1440). For observation-based
    initialization and learned ensemble branching, use
    [`AIDAWeatherMesh5C`][earth2studio.models.da.AIDAWeatherMesh5C].
    Decoded fields are converted to physical units on the CPU at every output step.

    Parameters
    ----------
    core_model : torch.nn.Module
        WeatherMesh-5c network.
    variables : Sequence[str] | None, optional
        Output subset, by default all pressure and surface fields

    Note
    ----
    For an ensemble, set `source_member` on
    [`CoupledRunner`][earth2studio.coupling.CoupledRunner] to choose the IFS member
    for each output member. See the [WeatherMesh-5c release notes](
    https://windbornesystems.com/blog/weathermesh-5c-release-notes), the
    [model card](https://huggingface.co/windborne/weathermesh-5c), and the
    [runtime](https://github.com/windborne/weathermesh-5c).

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos
    year:2025 provider:windborne backend:pytorch gpu:40gb
    """

    def __init__(
        self, core_model: torch.nn.Module, variables: Sequence[str] | None = None
    ) -> None:
        super().__init__()
        self._runtime = Inference(core_model)
        self.variables = _variables(self._runtime.variables, variables)

    @property
    def device(self) -> torch.device:
        """Device holding the numerical runtime."""
        return self._runtime.device

    def input_coords(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Declare physical IFS and cell-centred SST input slots.

        Returns
        -------
        tuple[CoordinateSystem, CoordinateSystem]
            IFS and SST at relative lead zero, with dynamic batch/time axes.
        """
        return _inputs(self._runtime)

    def output_coords(self, x: xr.DataArray, sst: xr.DataArray) -> xr.DataArray:
        """Plan the first forecast on the decoder grid.

        Parameters
        ----------
        x, sst : CoordinateSystem
            Physical IFS and SST signatures.

        Returns
        -------
        CoordinateSystem
            Complete forecast three hours after the final input lead.
        """
        xs, ss = self.input_coords()
        _validate(x, xs)
        _validate(sst, ss)
        return _layout(self._runtime, x, self.variables, x.lead_time.values + STEP)

    def default_sources(self) -> tuple[IFS_ENS, OISST]:
        """Recommend IFS member one and the most recent prior-day OISST.

        Returns
        -------
        tuple[DataSource, DataSource]
            The IFS ensemble background (Google Cloud mirror) and OISST sea
            surface temperature.
        """
        return IFS_ENS(source="google", member=1), OISST(lag_days=1, lookback_days=7)

    def __call__(self, x: xr.DataArray, sst: xr.DataArray) -> xr.DataArray:
        """Compute the first complete forecast without iterator hooks.

        Parameters
        ----------
        x, sst : xr.DataArray
            Physical IFS and SST fields.

        Returns
        -------
        xr.DataArray
            Forecast at the final input lead plus three hours.
        """
        return self.initialize(x, sst)[0]

    @torch.inference_mode()
    def initialize(
        self, x: xr.DataArray, sst: xr.DataArray
    ) -> tuple[xr.DataArray, _State]:
        """Encode initial fields and compute the first forecast.

        Parameters
        ----------
        x, sst : xr.DataArray
            Physical IFS and SST fields.

        Returns
        -------
        tuple[xr.DataArray, _State]
            First forecast and private latent continuation state.
        """
        layout = self.output_coords(x, sst)
        latents = [
            self._runtime.step(self._runtime.encode(values, time))
            for values, time in zip(
                _backgrounds(self._runtime, x, sst), _times(x), strict=True
            )
        ]
        return _decode(self._runtime, torch.cat(latents), layout, self.variables)

    def step(self, y: xr.DataArray, state: _State) -> tuple[xr.DataArray, _State]:
        """Advance three hours without modifying the supplied output or state.

        Parameters
        ----------
        y : xr.DataArray
            Previous forecast; its coordinates must match the supplied state.
        state : _State
            Complete latent continuation state.

        Returns
        -------
        tuple[xr.DataArray, _State]
            Next forecast and state, computed from the latent without applying hooks.
        """
        return _step(self._runtime, y, state, self.variables)

    def create_iterator(
        self, x: xr.DataArray, sst: xr.DataArray
    ) -> Generator[xr.DataArray]:
        """Yield complete three-hour forecasts with iterator-only hooks.

        Parameters
        ----------
        x, sst : xr.DataArray
            Physical initialization fields.

        Yields
        ------
        xr.DataArray
            Forecasts beginning at +3h. Rear hooks run before each yield;
            front hooks run before each step. Custom hooks that edit fields or
            coordinates raise ValueError.
        """
        yield from _iterator(self, x, sst)

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default WeatherMesh-5c/AIDA package."""
        return Package(
            f"hf://windborne/weathermesh-5c@{HF_REVISION}",
            cache_options={
                "cache_storage": Package.default_cache("weathermesh5c"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls, package: Package, variables: Sequence[str] | None = None
    ) -> WeatherMesh5C:
        """Load model assets directly from the standard package cache.

        Parameters
        ----------
        package : Package
            Local or remote checkpoint and asset package.
        variables : Sequence[str] | None, optional
            Forecast fields, by default all supported fields

        Returns
        -------
        WeatherMesh5C
            Model loaded on CPU in evaluation mode.
        """
        return cls(_load_core(package), variables=variables).eval()
