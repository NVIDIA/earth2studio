# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spatial grid definitions and the process-local grid registry."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import xarray as xr
from pyproj import CRS

from earth2studio.grids.base import (
    E2S_CRS,
    E2S_GRID_ID,
    GridDefinition,
    GridTopology,
)
from earth2studio.grids.curvilinear import CurvilinearGrid
from earth2studio.grids.healpix import (
    HEALPixGrid,
    HEALPixLayout,
    HEALPixOrdering,
    HEALPixOrigin,
)
from earth2studio.grids.latlon import LatLonGrid
from earth2studio.grids.point import PointGrid
from earth2studio.grids.projected import ProjectedGrid

_GRID_REGISTRY: dict[str, GridDefinition] = {}
_GRID_ALIASES: dict[str, str] = {}


def _is_crs(value: str) -> bool:
    try:
        CRS.from_user_input(value)
    except Exception:
        return False
    return True


def _validate_definition(definition: GridDefinition) -> None:
    if not isinstance(definition, GridDefinition):
        raise TypeError("definition must implement GridDefinition")
    if not definition.dims or len(set(definition.dims)) != len(definition.dims):
        raise ValueError("Grid dimensions must be nonempty and unique")
    if len(definition.shape) != len(definition.dims) or any(
        size <= 0 for size in definition.shape
    ):
        raise ValueError("Grid shape must positively size every dimension")
    coordinates = definition.coords(only_index=True)
    for dimension, size in zip(definition.dims, definition.shape, strict=True):
        if dimension not in coordinates or coordinates[dimension].dims != (dimension,):
            raise ValueError(f"Grid must define a 1D '{dimension}' index coordinate")
        if coordinates[dimension].size != size:
            raise ValueError(f"Grid index coordinate '{dimension}' has the wrong size")
    try:
        json.dumps(definition.to_metadata())
    except TypeError as error:
        raise TypeError("Grid metadata must be JSON serializable") from error
    if not definition.fingerprint():
        raise ValueError("Grid fingerprint must not be empty")


def register_grid(
    name: str, definition: GridDefinition, *, aliases: Sequence[str] = ()
) -> None:
    """Register a complete grid definition under a name and optional aliases."""
    canonical = str(name)
    grid_aliases = tuple(str(alias) for alias in aliases)
    if not canonical:
        raise ValueError("Grid name must not be empty")
    if len(set(grid_aliases)) != len(grid_aliases) or canonical in grid_aliases:
        raise ValueError("Grid aliases must be unique and differ from the grid name")
    _validate_definition(definition)
    incoming = {canonical, *grid_aliases}
    for key in incoming:
        if _is_crs(key):
            raise ValueError(f"Grid name or alias '{key}' is already a valid CRS input")
        existing = key if key in _GRID_REGISTRY else _GRID_ALIASES.get(key)
        if existing is not None:
            current_aliases = {
                alias for alias, target in _GRID_ALIASES.items() if target == existing
            }
            if (
                existing == canonical
                and _GRID_REGISTRY[existing].fingerprint() == definition.fingerprint()
                and set(grid_aliases) == current_aliases
            ):
                return
            raise ValueError(
                f"Grid name or alias conflicts with registered grid '{existing}'"
            )
    _GRID_REGISTRY[canonical] = definition
    _GRID_ALIASES.update({alias: canonical for alias in grid_aliases})


def list_grids() -> tuple[str, ...]:
    """List canonical grid names in registration order."""
    return tuple(_GRID_REGISTRY)


def resolve_grid(grid: str) -> GridDefinition:
    """Resolve a canonical grid name or alias."""
    canonical = grid if grid in _GRID_REGISTRY else _GRID_ALIASES.get(grid)
    if canonical is not None:
        return _GRID_REGISTRY[canonical]
    if _is_crs(grid):
        raise ValueError(
            "A CRS does not define grid dimensions or geometry; create a grid "
            "definition and register it before use"
        )
    raise ValueError(f"Unknown Earth2Studio grid '{grid}'")


def infer_grid(array: xr.DataArray | xr.Dataset) -> GridDefinition:
    """Infer a grid from explicit metadata or standard Xarray coordinates."""
    grid_id = array.attrs.get(E2S_GRID_ID)
    if grid_id is not None:
        definition = resolve_grid(grid_id)
        spatial_dims = tuple(
            dimension for dimension in array.dims if dimension in definition.dims
        )
        if spatial_dims == definition.dims and all(
            array.sizes[dimension] == size
            for dimension, size in zip(definition.dims, definition.shape, strict=True)
        ):
            return definition

    coordinates = array.coords
    if "y" in coordinates and "x" in coordinates and E2S_CRS in array.attrs:
        return ProjectedGrid(
            np.asarray(coordinates["y"]),
            np.asarray(coordinates["x"]),
            array.attrs[E2S_CRS],
        )
    if "lat" in coordinates and "lon" in coordinates:
        latitude = coordinates["lat"]
        longitude = coordinates["lon"]
        if latitude.dims == ("lat",) and longitude.dims == ("lon",):
            return LatLonGrid(np.asarray(latitude), np.asarray(longitude))
        if latitude.dims == longitude.dims == ("x",):
            x = np.asarray(coordinates["x"]) if "x" in coordinates else None
            return PointGrid(np.asarray(latitude), np.asarray(longitude), x)
        if latitude.dims == longitude.dims == ("y", "x"):
            y = np.asarray(coordinates["y"]) if "y" in coordinates else None
            x = np.asarray(coordinates["x"]) if "x" in coordinates else None
            return CurvilinearGrid(np.asarray(latitude), np.asarray(longitude), y, x)
        raise ValueError("Latitude and longitude coordinates have an unsupported layout")
    raise ValueError(
        "Cannot infer grid geometry; provide lat/lon coordinates or y/x coordinates "
        f"with the '{E2S_CRS}' attribute"
    )


register_grid(
    "latlon-0.25deg",
    LatLonGrid(
        latitude=np.arange(90.0, -90.25, -0.25),
        longitude=np.arange(0.0, 360.0, 0.25),
    ),
    aliases=("latlon025",),
)
register_grid(
    "latlon-0.25deg-south-pole-excluded",
    LatLonGrid(
        latitude=np.arange(90.0, -90.0, -0.25),
        longitude=np.arange(0.0, 360.0, 0.25),
    ),
    aliases=("fcn1", "fcn1-global-0.25deg"),
)
register_grid(
    "hrrr-conus-3km",
    ProjectedGrid(
        y=-1587306.1525566636 + 3000.0 * np.arange(1059),
        x=-2697520.1425219304 + 3000.0 * np.arange(1799),
        coordinate_reference_system=(
            "+proj=lcc +lon_0=262.5 +lat_0=38.5 +lat_1=38.5 "
            "+lat_2=38.5 +R=6371229 +units=m +type=crs"
        ),
    ),
    aliases=("hrrr",),
)
register_grid(
    "healpix-l6-nested",
    HEALPixGrid(level=6, ordering="nested"),
    aliases=("hpx6",),
)

__all__ = [
    "E2S_CRS",
    "E2S_GRID_ID",
    "CurvilinearGrid",
    "GridDefinition",
    "GridTopology",
    "HEALPixGrid",
    "HEALPixLayout",
    "HEALPixOrdering",
    "HEALPixOrigin",
    "LatLonGrid",
    "PointGrid",
    "ProjectedGrid",
    "infer_grid",
    "list_grids",
    "register_grid",
    "resolve_grid",
]
