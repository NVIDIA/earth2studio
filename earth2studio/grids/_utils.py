# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared grid implementation helpers."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS, Transformer

if TYPE_CHECKING:
    from earth2studio.grids.base import GridDefinition


def array(values: Any, ndim: int, name: str) -> NDArray[Any]:
    """Create an immutable, nonempty array."""
    result = np.asarray(values)
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}D array")
    result = result.copy()
    result.setflags(write=False)
    return result


def coordinate_hash(*arrays: NDArray[Any]) -> str:
    """Hash coordinate values, dtypes, and shapes."""
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def metadata(definition: GridDefinition, **details: Any) -> dict[str, Any]:
    """Build a JSON-serializable grid description."""
    result: dict[str, Any] = {
        "type": type(definition).__name__,
        "dims": list(definition.dims),
        "shape": list(definition.shape),
        "topology": definition.topology,
    }
    if definition.crs is not None:
        result["crs"] = definition.crs.to_string()
    result.update(details)
    return result


def geographic_subset_indexers(
    definition: GridDefinition, coordinates: xr.Coordinates, **selection: Any
) -> dict[str, Any]:
    """Translate geographic bounds into positional indexers."""
    unknown = set(selection) - {"bounds", "bounds_crs"}
    if unknown:
        raise ValueError(f"Unsupported grid subset options: {sorted(unknown)}")
    if "bounds_crs" in selection and "bounds" not in selection:
        raise ValueError("Grid subset bounds_crs requires bounds")
    if "bounds" not in selection:
        return {}

    bounds = selection["bounds"]
    if len(bounds) != 4:
        raise ValueError("Bounds must contain (min_x, min_y, max_x, max_y)")
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    if min_y > max_y:
        raise ValueError("Bounds minimum y must not exceed maximum y")

    geographic = (
        xr.Coordinates({"lat": coordinates["lat"], "lon": coordinates["lon"]})
        if "lat" in coordinates and "lon" in coordinates
        else definition.coords(
            {
                dimension: np.asarray(coordinates[dimension])
                for dimension in definition.dims
            }
        )
    )
    latitude, longitude = xr.broadcast(geographic["lat"], geographic["lon"])
    latitude_values = np.asarray(latitude)
    longitude_values = np.asarray(longitude)
    target_crs = CRS.from_user_input(selection.get("bounds_crs", "OGC:CRS84"))
    if target_crs.is_geographic:
        x_values = (
            np.mod(longitude_values, 360)
            if min_x >= 0 and max_x >= 0 and (min_x > 180 or max_x > 180)
            else np.mod(longitude_values + 180, 360) - 180
        )
        x_mask = (
            (x_values >= min_x) & (x_values <= max_x)
            if min_x <= max_x
            else (x_values >= min_x) | (x_values <= max_x)
        )
        y_values = latitude_values
    else:
        x_values, y_values = Transformer.from_crs(
            CRS.from_epsg(4326), target_crs, always_xy=True
        ).transform(longitude_values, latitude_values)
        x_mask = (x_values >= min_x) & (x_values <= max_x)
    mask = x_mask & (y_values >= min_y) & (y_values <= max_y)
    if not np.any(mask):
        raise ValueError("Grid subset bounds do not contain any cell centers")

    indexers: dict[str, Any] = {}
    for dimension in definition.dims:
        axis = latitude.dims.index(dimension)
        other_axes = tuple(index for index in range(mask.ndim) if index != axis)
        selected = np.flatnonzero(np.any(mask, axis=other_axes) if other_axes else mask)
        indexers[dimension] = (
            slice(int(selected[0]), int(selected[-1]) + 1)
            if np.all(np.diff(selected) == 1)
            else selected
        )
    return indexers
