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

"""Allocation-free Xarray coordinate signatures."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike

from earth2studio.grids import (
    E2S_CRS,
    E2S_GRID,
    E2S_GRID_ID,
    GridDefinition,
    resolve_grid,
)
from earth2studio.utils.time_statistics import time_statistic_metadata

CoordinateSystem = tuple[xr.DataArray, ...]

E2S_DYNAMIC_DIMS = "earth2studio_dynamic_dims"
E2S_KIND = "earth2studio_kind"
E2S_SCHEMA_VERSION = "earth2studio_schema_version"
E2S_STATISTICS = "earth2studio_statistics"


class _CoordinateArray:
    __array_priority__ = 100

    def __init__(self, shape: Sequence[int], dtype: DTypeLike) -> None:
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return 0

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise TypeError("Coordinate arrays do not contain field values")

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        return NotImplemented

    def __array_ufunc__(
        self, ufunc: Any, method: str, *args: Any, **kwargs: Any
    ) -> Any:
        return NotImplemented

    def __getitem__(self, key: Any) -> _CoordinateArray:
        key = getattr(key, "tuple", key)
        items = key if isinstance(key, tuple) else (key,)
        shape: list[int] = []
        axis = 0
        for item in items:
            if item is Ellipsis:
                count = self.ndim - len(items) + 1
                shape.extend(self.shape[axis : axis + count])
                axis += count
            elif item is None:
                shape.append(1)
            elif isinstance(item, slice):
                shape.append(len(range(*item.indices(self.shape[axis]))))
                axis += 1
            elif isinstance(item, (int, np.integer)):
                axis += 1
            elif isinstance(item, np.ndarray) and item.ndim == 1:
                shape.append(
                    int(np.count_nonzero(item))
                    if item.dtype == bool
                    else int(item.size)
                )
                axis += 1
            else:
                raise TypeError("Unsupported coordinate-array indexer")
        shape.extend(self.shape[axis:])
        return type(self)(shape, self.dtype)

    def transpose(self, axes: Sequence[int] | None = None) -> _CoordinateArray:
        axes = tuple(reversed(range(self.ndim))) if axes is None else tuple(axes)
        return type(self)(tuple(self.shape[axis] for axis in axes), self.dtype)


def _coordinate_sizes(coordinates: Mapping[Hashable, Any]) -> dict[Hashable, int]:
    sizes: dict[Hashable, int] = {}
    for name, value in coordinates.items():
        if isinstance(value, (xr.DataArray, xr.Variable)):
            dimensions, shape = value.dims, value.shape
        elif isinstance(value, tuple) and len(value) >= 2:
            dimensions = (value[0],) if isinstance(value[0], str) else tuple(value[0])
            shape = np.asarray(value[1]).shape
        else:
            array = np.asarray(value)
            dimensions, shape = (
                ((name,), array.shape) if array.ndim == 1 else ((), array.shape)
            )
        if len(dimensions) != len(shape):
            raise ValueError(f"Coordinate '{name}' dimensions do not match its shape")
        for dimension, size in zip(dimensions, shape, strict=True):
            if dimension in sizes and sizes[dimension] != size:
                raise ValueError(f"Coordinates disagree on size of '{dimension}'")
            sizes[dimension] = int(size)
    return sizes


def coord_array(
    dims: Sequence[Hashable],
    coords: Mapping[Hashable, Any] | None = None,
    *,
    dynamic: Sequence[Hashable] = (),
    sizes: Mapping[Hashable, int] | None = None,
    grid: str | GridDefinition | None = None,
    statistics: Mapping[str, str] | None = None,
    dtype: DTypeLike = np.float32,
    name: Hashable | None = None,
    attrs: Mapping[Hashable, Any] | None = None,
) -> xr.DataArray:
    """Create an allocation-free Earth2Studio coordinate signature."""
    dimensions = tuple(dims)
    dynamic_dims = tuple(dynamic)
    coordinates = dict(coords or {})
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Dimensions must be unique")
    if not set(dynamic_dims).issubset(dimensions):
        raise ValueError("Dynamic dimensions must be present in dims")
    if dimensions[: len(dynamic_dims)] != dynamic_dims:
        raise ValueError("Dynamic dimensions must lead the coordinate signature")

    candidates = dict(sizes or {})
    definition = resolve_grid(grid) if isinstance(grid, str) else grid
    if definition is not None:
        missing = set(definition.dims) - set(dimensions)
        if missing:
            raise ValueError(
                f"Grid dimensions are missing from dims: {sorted(missing)}"
            )
        candidates.update(zip(definition.dims, definition.shape, strict=True))
        grid_coords = definition.coords(
            only_index=definition.topology not in {"curvilinear", "points"}
        )
        for coordinate, value in grid_coords.items():
            coordinates.setdefault(coordinate, value)

    coordinate_sizes = _coordinate_sizes(coordinates)
    resolved_sizes: dict[Hashable, int] = {}
    for dimension in dimensions:
        coordinate_size = coordinate_sizes.get(dimension)
        declared_size = candidates.get(dimension)
        if coordinate_size is not None and declared_size not in {None, coordinate_size}:
            raise ValueError(f"Coordinate and declared size differ for '{dimension}'")
        size = coordinate_size if coordinate_size is not None else declared_size
        if dimension in dynamic_dims:
            if size not in {None, 0}:
                raise ValueError(f"Dynamic dimension '{dimension}' must have size zero")
            size = 0
        if size is None:
            raise ValueError(f"Missing size for dimension '{dimension}'")
        resolved_sizes[dimension] = int(size)

    metadata = dict(attrs or {})
    metadata.update(
        {
            E2S_KIND: "coordinate_array",
            E2S_SCHEMA_VERSION: 1,
            E2S_DYNAMIC_DIMS: dynamic_dims,
        }
    )
    if definition is not None:
        metadata[E2S_GRID] = definition.attrs
        if definition.crs is not None:
            metadata[E2S_CRS] = definition.crs.to_string()
    if isinstance(grid, str):
        metadata[E2S_GRID_ID] = grid
    array = xr.DataArray(
        _CoordinateArray(
            tuple(resolved_sizes[dimension] for dimension in dimensions), dtype
        ),
        dims=dimensions,
        coords=coordinates,
        name=name,
        attrs=metadata,
    )
    if statistics:
        if "variable" not in array.coords:
            raise ValueError("Statistics require a variable coordinate")
        variables = set(np.asarray(array.coords["variable"]).astype(str))
        missing = set(statistics) - variables
        if missing:
            raise ValueError(
                f"Statistics reference unknown variables: {sorted(missing)}"
            )
        array.attrs[E2S_STATISTICS] = {
            variable: time_statistic_metadata(modifier)
            for variable, modifier in statistics.items()
        }
    return array


def handshake_dataarray(array: xr.DataArray, signature: xr.DataArray) -> None:
    """Validate ordered dimensions, sizes, and labels against a signature."""
    dynamic = tuple(signature.attrs.get(E2S_DYNAMIC_DIMS, ()))
    if tuple(signature.dims[: len(dynamic)]) != dynamic:
        raise ValueError("Dynamic dimensions must lead the coordinate signature")
    fixed = signature.dims[len(dynamic) :]
    trailing = tuple(array.dims[-len(fixed) :]) if fixed else ()
    if trailing != fixed:
        raise ValueError(f"Expected trailing dimensions {fixed}, got {array.dims}")
    for dimension in fixed:
        if array.sizes[dimension] != signature.sizes[dimension]:
            raise ValueError(f"Dimension '{dimension}' has the wrong size")
        if dimension in signature.coords:
            if dimension not in array.coords:
                raise ValueError(f"Coordinate '{dimension}' is missing")
            if not np.array_equal(array.coords[dimension], signature.coords[dimension]):
                raise ValueError(f"Coordinate '{dimension}' does not match")
    for key, label in ((E2S_GRID, "grid"), (E2S_STATISTICS, "statistics")):
        expected = signature.attrs.get(key)
        if expected is not None and array.attrs.get(key) != expected:
            raise ValueError(f"DataArray {label} metadata does not match")


def handshake_dataarrays(
    arrays: Sequence[xr.DataArray], signatures: Sequence[xr.DataArray]
) -> None:
    """Validate an ordered collection of DataArrays against model signatures."""
    if len(arrays) != len(signatures):
        raise ValueError(
            f"Expected {len(signatures)} DataArrays, received {len(arrays)}"
        )
    for array, signature in zip(arrays, signatures, strict=True):
        handshake_dataarray(array, signature)


def statistics_from_metadata(array: xr.DataArray | None) -> dict[str, str]:
    """Return compact temporal-statistic modifiers from a signature."""
    if array is None:
        return {}
    return {
        variable: details["modifier"]
        for variable, details in array.attrs.get(E2S_STATISTICS, {}).items()
    }
