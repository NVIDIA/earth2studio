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
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.grids import (
    CurvilinearGrid,
    GridDefinition,
    HEALPixGrid,
    LatLonGrid,
    ProjectedGrid,
    resolve_grid,
)
from earth2studio.utils.coordinate import CoordinateSystem, coord_array
from earth2studio.utils.type import CoordSystem

_TIME_STATISTICS = {
    "cp06": "sum:6h",
    "ro06": "sum:6h",
    "sf06": "sum:6h",
    "ssrd06": "sum:6h",
    "strd06": "sum:6h",
    "tp06": "sum:6h",
    "tp12": "sum:12h",
    "sf1h": "sum:1h",
    "ssrd1h": "sum:1h",
    "tp1h": "sum:1h",
    "ttr1h": "sum:1h",
    "ttr-3h": "sum:3h",
}
_PUBLIC_VARIABLES = {
    "cp06": "cp",
    "ro06": "ro",
    "sf06": "sf",
    "ssrd06": "ssrd",
    "strd06": "strd",
    "tp06": "tp",
    "tp12": "tp",
    "sf1h": "sf",
    "ssrd1h": "ssrd",
    "tp1h": "tp",
    "ttr1h": "ttr",
    "ttr-3h": "ttr",
}


def _numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _same_grid(first: GridDefinition, second: GridDefinition) -> bool:
    return first.dims == second.dims and all(
        np.array_equal(
            first.coords(only_index=True)[dimension],
            second.coords(only_index=True)[dimension],
        )
        for dimension in first.dims
    )


def _spatial_grid(model: Any, coords: CoordSystem) -> str | GridDefinition | None:
    explicit = getattr(model, "_grid", None)
    if explicit is not None:
        return explicit
    if "lat" in coords and "lon" in coords:
        latlon_grid = LatLonGrid(_numpy(coords["lat"]), _numpy(coords["lon"]))
        for name in ("latlon025", "fcn1"):
            if _same_grid(latlon_grid, resolve_grid(name)):
                return name
        return latlon_grid
    if {"face", "height", "width"}.issubset(coords):
        nside = len(coords["height"])
        return HEALPixGrid(level=int(np.log2(nside)), ordering="xy", layout="face")
    if "hpx" in coords:
        nside = int(np.sqrt(len(coords["hpx"]) / 12))
        return HEALPixGrid(level=int(np.log2(nside)), ordering="nested")
    if {"hrrr_y", "hrrr_x"}.issubset(coords):
        registered = resolve_grid("hrrr")
        projected_grid = ProjectedGrid(
            _numpy(coords["hrrr_y"]), _numpy(coords["hrrr_x"]), registered.crs
        )
        return "hrrr" if _same_grid(projected_grid, registered) else projected_grid
    if {"y", "x"}.issubset(coords):
        shape = (len(coords["y"]), len(coords["x"]))
        for latitude, longitude in (
            (getattr(model, "lat", None), getattr(model, "lon", None)),
            (getattr(model, "latitudes", None), getattr(model, "longitudes", None)),
        ):
            if latitude is None or longitude is None:
                continue
            latitude_array = np.squeeze(_numpy(latitude))
            longitude_array = np.squeeze(_numpy(longitude))
            if latitude_array.shape == longitude_array.shape == shape:
                return CurvilinearGrid(
                    latitude_array,
                    longitude_array,
                    _numpy(coords["y"]),
                    _numpy(coords["x"]),
                )
    return None


def _public_coords(coords: CoordSystem) -> CoordSystem:
    return OrderedDict(
        (
            "y" if key == "hrrr_y" else "x" if key == "hrrr_x" else key,
            (
                np.asarray(
                    [_PUBLIC_VARIABLES.get(str(item), str(item)) for item in value]
                )
                if key == "variable"
                else value
            ),
        )
        for key, value in coords.items()
    )


def _signature(model: Any, coords: CoordSystem) -> xr.DataArray:
    public = _public_coords(coords)
    dynamic: list[str] = []
    for dimension, values in public.items():
        if _numpy(values).size:
            break
        dynamic.append(dimension)
    tensor_variables = np.asarray(coords.get("variable", ())).astype(str)
    statistics = {
        _PUBLIC_VARIABLES[variable]: _TIME_STATISTICS[variable]
        for variable in tensor_variables
        if variable in _TIME_STATISTICS
    }
    variables = np.asarray(public.get("variable", ())).astype(str)
    if len(set(variables)) != len(variables):
        raise ValueError("A coordinate signature cannot repeat a public variable")
    grid = _spatial_grid(model, coords)
    definition = resolve_grid(grid) if isinstance(grid, str) else grid
    coordinates: dict[Hashable, Any] = {
        dimension: values
        for dimension, values in public.items()
        if definition is None or dimension not in definition.dims
    }
    return coord_array(
        tuple(public),
        coordinates,
        dynamic=dynamic,
        grid=grid,
        statistics=statistics,
    )


def _tensor_coords(reference: CoordSystem, array: xr.DataArray) -> CoordSystem:
    aliases = {"y": "hrrr_y", "x": "hrrr_x"} if "hrrr_y" in reference else {}
    return OrderedDict(
        (
            aliases.get(str(dimension), str(dimension)),
            (
                reference[aliases[str(dimension)]]
                if str(dimension) in aliases
                else (
                    reference["variable"]
                    if dimension == "variable" and "variable" in reference
                    else (
                        np.asarray(array.coords[dimension])
                        if dimension in array.coords
                        else np.arange(array.sizes[dimension])
                    )
                )
            ),
        )
        for dimension in array.dims
    )


class PrognosticMixin:
    """Provide DataArray coordinate declarations and tensor execution hooks."""

    def input_coords(self) -> CoordinateSystem:
        """Return ordered allocation-free input signatures."""
        return (_signature(self, self._input_tensor_coords()),)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Return ordered output signatures for one forecast step."""
        if len(input_coords) != 1:
            raise ValueError(f"Expected 1 DataArray, received {len(input_coords)}")
        coords = _tensor_coords(self._input_tensor_coords(), input_coords[0])
        return (_signature(self, self._output_tensor_coords(coords)),)

    def _input_tensor_coords(self) -> CoordSystem:
        raise NotImplementedError

    def _output_tensor_coords(self, input_coords: CoordSystem) -> CoordSystem:
        raise NotImplementedError

    def _default_hook(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        return x, coords

    front_hook = _default_hook
    rear_hook = _default_hook
