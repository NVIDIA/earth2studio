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

import warnings
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

import numpy as np
import torch
import xarray as xr
from numpy.typing import DTypeLike

from earth2studio.grids import E2S_CRS, E2S_GRID_ID, GridDefinition, resolve_grid

try:
    import cupy as cp
except ImportError:
    cp = None

from earth2studio.utils.time_statistics import time_statistic_metadata
from earth2studio.utils.type import CoordinateSystem, CoordSystem

E2S_DYNAMIC_DIMS = "earth2studio_dynamic_dims"
E2S_KIND = "earth2studio_kind"
E2S_SCHEMA_VERSION = "earth2studio_schema_version"
E2S_STATISTICS = "earth2studio_statistics"


class _CoordinateArray:
    """Shape/dtype-only backing array for allocation-free coordinate signatures.

    Indexing and transposing update the shape without allocating field values;
    converting this placeholder to a NumPy array is intentionally unsupported.
    """

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
    dynamic: Sequence[Hashable] = (),
    sizes: Mapping[Hashable, int] | None = None,
    grid: str | GridDefinition | None = None,
    grid_dims: Mapping[str, str] | None = None,
    dtype: DTypeLike = np.float32,
    name: Hashable | None = None,
    attrs: Mapping[Hashable, Any] | None = None,
) -> CoordinateSystem:
    """Create a coordinate signature from dimensions and optional information.

    The returned CoordinateSystem's backing array stores only shape and dtype;
    it does not allocate memory for field values. Coordinate arrays themselves
    still occupy memory.

    Parameters
    ----------
    dims : sequence of hashable
        Ordered dimensions, including any dynamic leading dimensions.
    coords : mapping, optional
        Xarray-compatible dimension and auxiliary coordinates.
    dynamic : sequence of hashable, optional
        Leading wildcard dimensions, each with size zero.
    sizes : mapping, optional
        Sizes for dimensions without coordinates.
    grid : str or GridDefinition, optional
        Registered grid or definition supplying spatial coordinates and metadata.
        Projected grids also supply geographic coordinates; HEALPix grids supply
        only index coordinates.
    grid_dims : mapping, optional
        Rename grid dimensions to model dimensions, for example
        ``{"y": "hrrr_y", "x": "hrrr_x"}``. Explicit coordinates use model names.
    dtype : dtype-like, optional
        Declared field dtype; no field values are allocated.
    name : hashable, optional
        DataArray name.
    attrs : mapping, optional
        Additional metadata. Grid and signature metadata take precedence.

    Notes
    -----
    Declare temporal statistics in variable labels, for example ``tp:sum:6h``.
    Statistics metadata is derived from those labels, not supplied independently.

    Returns
    -------
    CoordinateSystem
        DataArray with coordinates and a shape/dtype-only backing array.
    """
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
    renamed = dict(grid_dims or {})
    if renamed and (definition is None or not set(renamed).issubset(definition.dims)):
        raise ValueError("grid_dims must map dimensions of the supplied grid")
    if definition is not None:
        spatial_dims = tuple(renamed.get(dim, dim) for dim in definition.dims)
        if len(set(spatial_dims)) != len(spatial_dims):
            raise ValueError("Renamed grid dimensions must be unique")
        missing = set(spatial_dims) - set(dimensions)
        if missing:
            raise ValueError(
                f"Grid dimensions are missing from dims: {sorted(missing)}"
            )
        for dim, grid_size in zip(spatial_dims, definition.shape, strict=True):
            if dim in candidates and candidates[dim] != grid_size:
                raise ValueError(f"Grid and declared size differ for '{dim}'")
            candidates[dim] = grid_size
        grid_coords = definition.coords(
            only_index=definition.topology not in {"curvilinear", "points", "projected"}
        )
        for coordinate, value in grid_coords.items():
            coordinates.setdefault(
                renamed.get(coordinate, coordinate),
                xr.Variable(
                    tuple(renamed.get(dim, dim) for dim in value.dims),
                    value.data,
                    attrs=value.attrs,
                ),
            )

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
    metadata.pop(E2S_STATISTICS, None)
    metadata.update(
        {
            E2S_KIND: "coordinate_array",
            E2S_SCHEMA_VERSION: 1,
            E2S_DYNAMIC_DIMS: dynamic_dims,
        }
    )
    if definition is not None:
        metadata.update(definition.attrs)
        metadata["dims"] = list(spatial_dims)
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
    if "variable" in array.coords:
        if array.coords["variable"].dims != ("variable",):
            raise ValueError("Variable must be a one-dimensional coordinate")
        variables = tuple(np.asarray(array.coords["variable"]).astype(str))
        if len(set(variables)) != len(variables):
            raise ValueError("Variable coordinates must be unique")
        declarations = {}
        for variable in variables:
            base, separator, modifier = variable.partition(":")
            if not base:
                raise ValueError("Variable source name must not be empty")
            if separator:
                declarations[variable] = time_statistic_metadata(modifier)
        if declarations:
            array.attrs[E2S_STATISTICS] = declarations
    return array


def coord_array_like(
    array: xr.DataArray,
    coords: Mapping[Hashable, Any] | None = None,
) -> CoordinateSystem:
    """Build a fresh coordinate signature from an array without copying field data.

    Parameters
    ----------
    array : xr.DataArray
        Concrete data or a coordinate signature to describe.
    coords : mapping, optional
        Coordinate replacements. Replacing a dimension drops its dependent
        coordinates and infers its new size. Dimension order is preserved.
        Grid geometry cannot be replaced; use ``coord_array(grid=...)`` with a
        new definition when changing spatial coordinates.
        Resolve dynamic dimensions together or from right to left so remaining
        wildcards stay a leading prefix. Resolving only ``batch`` while leaving
        a following ``time`` dynamic raises ``ValueError``.
        Temporal statistics are derived from the resulting variable labels.

    Returns
    -------
    CoordinateSystem
        Allocation-free signature preserving name, dtype, metadata, and unaffected
        coordinates. Nonzero dimensions are never declared dynamic.
    """
    replacements = dict(coords or {})
    changed_dims = set(replacements).intersection(array.dims)
    spatial_dims = set(array.attrs.get("dims", ()))
    for key in replacements:
        if key in spatial_dims or (
            key in array.coords and spatial_dims.intersection(array.coords[key].dims)
        ):
            raise ValueError(
                "Use coord_array with a new grid to replace spatial coordinates"
            )
    coordinates = {
        key: value.variable
        for key, value in array.coords.items()
        if not changed_dims.intersection(value.dims)
    }
    coordinates.update(replacements)
    sizes = {dim: size for dim, size in array.sizes.items() if dim not in changed_dims}
    for dim, size in _coordinate_sizes(coordinates).items():
        if dim in sizes and sizes[dim] != size:
            raise ValueError(f"Coordinate and existing size differ for '{dim}'")
        sizes[dim] = size
    dynamic = tuple(
        dim
        for dim in array.attrs.get(E2S_DYNAMIC_DIMS, ())
        if dim in sizes and sizes[dim] == 0
    )
    output = coord_array(
        array.dims,
        coordinates,
        sizes=sizes,
        dynamic=dynamic,
        dtype=array.dtype,
        name=array.name,
        attrs=array.attrs,
    )
    return output


def handshake_dataarray(
    array: xr.DataArray,
    signature: xr.DataArray | None = None,
    *,
    relative_lead_time: bool = False,
    runtime: bool = False,
) -> None:
    """Validate input coordinates without accessing field data.

    Parameters
    ----------
    array : xr.DataArray
        Input field or allocation-free coordinate declaration.
    signature : xr.DataArray, optional
        Required fixed trailing dimensions, coordinates and grid metadata.
        Omit to check only input structure and temporal coordinates.
    relative_lead_time : bool, optional
        Compare lead times relative to their final value, by default False
    runtime : bool, optional
        Require concrete nonempty axes before execution, by default False

    Notes
    -----
    Leading axes need not have labels. Temporal axes require explicit finite,
    one-dimensional labels, except declared zero-sized dynamic planning axes.
    Comparisons ignore auxiliaries attached to individual coordinate DataArrays.
    """
    if not isinstance(array, xr.DataArray):
        raise TypeError("Expected a DataArray")
    runtime = runtime or not isinstance(array.data, _CoordinateArray)
    declared = tuple(
        dim
        for dim in array.attrs.get(E2S_DYNAMIC_DIMS, ())
        if array.sizes.get(dim) == 0
    )
    if array.dims[: len(declared)] != declared:
        raise ValueError("Input dynamic dimensions must be a leading prefix")
    for dim, size in array.sizes.items():
        if size == 0 and (runtime or dim not in declared):
            raise ValueError(f"Dimension '{dim}' must be nonempty")
    temporal = {dim for dim in ("time", "lead_time") if dim in array.coords}
    if signature is not None:
        temporal.update(
            dim
            for dim in ("time", "lead_time")
            if dim in signature.dims and dim in array.dims
        )
    for dim in temporal:
        handshake_time(
            array, dim, allow_dynamic=not runtime, dimension=dim in array.dims
        )
    if signature is None:
        return
    dynamic = tuple(signature.attrs.get(E2S_DYNAMIC_DIMS, ()))
    if tuple(signature.dims[: len(dynamic)]) != dynamic:
        raise ValueError("Dynamic dimensions must lead the coordinate signature")
    fixed = signature.dims[len(dynamic) :]
    trailing = tuple(array.dims[-len(fixed) :]) if fixed else ()
    if trailing != fixed:
        raise ValueError(f"Expected trailing dimensions {fixed}, got {array.dims}")
    for index, dimension in enumerate(fixed, start=-len(fixed)):
        handshake_dim(array, dimension, index)
        handshake_size(array, dimension, signature.sizes[dimension])
    lead = None
    if relative_lead_time:
        handshake_time(array, "lead_time")
        lead = np.asarray(array.coords["lead_time"])
    for name, coordinate in signature.coords.items():
        if set(coordinate.dims).intersection(dynamic):
            continue
        actual = array
        if name == "lead_time" and lead is not None:
            actual = array.assign_coords(lead_time=lead - lead[-1])
        try:
            handshake_coords(actual, signature, name)
        except KeyError as error:
            raise ValueError(str(error)) from error
    keys = [
        key for key in (E2S_GRID_ID, E2S_CRS, E2S_STATISTICS) if key in signature.attrs
    ]
    if signature.attrs.get("type") == "HEALPixGrid":
        keys.extend(
            (
                "type",
                "topology",
                "dims",
                "shape",
                "level",
                "nside",
                "ordering",
                "layout",
                "origin",
                "clockwise",
                "crs",
                E2S_CRS,
                E2S_GRID_ID,
            )
        )
    handshake_metadata(array, signature, keys)


def handshake_metadata(
    array: xr.DataArray, target: xr.DataArray, keys: Sequence[Hashable]
) -> None:
    """Require matching named attributes, including explicitly absent attributes.

    Parameters
    ----------
    array, target : xr.DataArray
        Input and reference arrays. Field values are never inspected.
    keys : sequence of hashable
        Metadata keys to compare; missing keys compare as None.
    """
    for key in keys:
        if not np.array_equal(
            np.asarray(array.attrs.get(key)), np.asarray(target.attrs.get(key))
        ):
            raise ValueError(f"DataArray metadata {key!r} does not match")


def handshake_time(
    input_coords: CoordSystem | xr.DataArray,
    required_dim: str = "time",
    *,
    allow_dynamic: bool = False,
    dimension: bool = True,
    step: np.timedelta64 | None = None,
    minimum: np.timedelta64 | np.datetime64 | None = None,
    maximum: np.timedelta64 | np.datetime64 | None = None,
) -> None:
    """Validate explicit nonempty one-dimensional finite temporal labels.

    Parameters
    ----------
    input_coords : CoordSystem or xr.DataArray
        Input coordinates; only coordinate labels are inspected.
    required_dim : str, optional
        ``time`` requires datetimes; other names require timedeltas.
    allow_dynamic : bool, optional
        Permit an explicitly declared zero-sized planning axis, by default False
    dimension : bool, optional
        Require a one-dimensional dimension coordinate, by default True.
        False permits scalar or auxiliary validity-time coordinates.
    step : np.timedelta64, optional
        Require labels aligned to this interval (datetimes relative to the epoch).
    minimum : np.timedelta64 or np.datetime64, optional
        Inclusive minimum permitted label.
    maximum : np.timedelta64 or np.datetime64, optional
        Exclusive maximum permitted label.
    """
    coordinates: Any = input_coords
    if isinstance(input_coords, xr.DataArray):
        coordinates = input_coords.coords
        if (
            allow_dynamic
            and required_dim in input_coords.attrs.get(E2S_DYNAMIC_DIMS, ())
            and input_coords.sizes.get(required_dim) == 0
        ):
            return
    if required_dim not in coordinates:
        raise ValueError(f"{required_dim} coordinate is required")
    coordinate = coordinates[required_dim]
    values = np.asarray(coordinate)
    kind = "M" if required_dim == "time" else "m"
    if (
        not values.size
        or values.dtype.kind != kind
        or (
            dimension
            and (
                values.ndim != 1
                or (
                    isinstance(coordinate, xr.DataArray)
                    and coordinate.dims != (required_dim,)
                )
            )
        )
        or np.isnat(values).any()
    ):
        raise ValueError(
            f"{required_dim} must contain nonempty finite {'datetimes' if kind == 'M' else 'timedeltas'}"
        )
    offsets = values - np.datetime64("1970-01-01") if kind == "M" else values
    if step is not None and np.any(offsets % step != np.timedelta64(0, "ns")):
        raise ValueError(f"{required_dim} must align to {step}")
    if minimum is not None and np.any(values < minimum):
        raise ValueError(f"{required_dim} must be at least {minimum}")
    if maximum is not None and np.any(values >= maximum):
        raise ValueError(f"{required_dim} must be before {maximum}")


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


def handshake_dim(
    input_coords: CoordSystem | xr.DataArray,
    required_dim: Hashable | tuple[Hashable, ...],
    required_index: int | None = None,
) -> None:
    """Simple check to see if coordinate system has a dimension in a particular index

    Parameters
    ----------
    input_coords : CoordSystem or xr.DataArray
        Input coordinate system to validate
    required_dim : str or tuple of hashable
        Required dimension, or the complete ordered dimension tuple.
    required_index : int, optional
        Required index of dimension if needed, by default None

    Raises
    ------
    KeyError
        If required dimension is not found in the input coordinate system
    ValueError
        If the required index is outside the dimensionality of the input coordinate system
    ValueError
        If dimension is not in the required index
    """

    input_dims = list(
        input_coords.dims
        if isinstance(input_coords, xr.DataArray)
        else input_coords.keys()
    )
    if isinstance(required_dim, tuple):
        if tuple(input_dims) != required_dim:
            raise ValueError(
                f"Expected dimensions {required_dim}, got {tuple(input_dims)}"
            )
        return
    if required_dim not in input_dims:
        raise KeyError(
            f"Required dimension {required_dim} not found in input coordinates"
        )

    if required_index is None:
        return

    try:
        input_dims[required_index]
    except IndexError:
        raise ValueError(
            f"Required index {required_index} outside dimensionality of input coordinate system of {len(input_dims)}"
        )

    if input_dims[required_index] != required_dim:
        raise ValueError(
            f"Required dimension {required_dim} not found in the required index {required_index} in dim list {input_dims}"
        )


def handshake_coords(
    input_coords: CoordSystem | xr.DataArray,
    target_coords: CoordSystem | xr.DataArray,
    required_dim: Hashable | Sequence[Hashable],
    *,
    subset: bool = False,
) -> None:
    """Simple check to see if the required dimensions have the same coordinate system

    Parameters
    ----------
    input_coords : CoordSystem or xr.DataArray
        Input coordinate system to validate
    target_coords : CoordSystem or xr.DataArray
        Target coordinate system
    required_dim : hashable or sequence of hashable
        Required dimension(s) (name of coordinate)
    subset : bool, optional
        Require target labels to be present in input, by default False.
        Used before explicit model-specific subsetting.
    Raises
    ------
    KeyError
        If required dim is not present in coordinate systems
    ValueError
        If coordinates of required dimensions don't match
    """
    if isinstance(required_dim, str) or not isinstance(required_dim, Sequence):
        required_dim = [required_dim]

    actual_coords: Any = (
        input_coords.coords if isinstance(input_coords, xr.DataArray) else input_coords
    )
    expected_coords: Any = (
        target_coords.coords
        if isinstance(target_coords, xr.DataArray)
        else target_coords
    )

    for _required_dim in required_dim:
        if _required_dim not in actual_coords:
            raise KeyError(
                f"Required dimension {_required_dim} not found in input coordinates"
            )

        if _required_dim not in expected_coords:
            raise KeyError(
                f"Required dimension {_required_dim} not found in target coordinates"
            )

        if subset:
            if not np.isin(
                expected_coords[_required_dim], actual_coords[_required_dim]
            ).all():
                raise ValueError(f"Required {_required_dim} labels are missing")
            continue
        if actual_coords[_required_dim].shape != expected_coords[_required_dim].shape:
            raise ValueError(
                f"Coordinate systems for required dim {_required_dim} are not the same"
            )

        actual, expected = actual_coords[_required_dim], expected_coords[_required_dim]
        if (
            isinstance(actual, xr.DataArray)
            and isinstance(expected, xr.DataArray)
            and actual.dims != expected.dims
        ):
            raise ValueError(f"Coordinate dimensions for {_required_dim} do not match")
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"Coordinate systems for required dim {_required_dim} are not the same"
            )


def handshake_size(
    input_coords: CoordSystem | xr.DataArray,
    required_dim: Hashable,
    required_size: int,
) -> None:
    """Simple check to see if a coordinate system of a given dimension is a required
    size

    Parameters
    ----------
    input_coords : CoordSystem or xr.DataArray
        Input coordinate system to validate
    required_dim : str
        Required dimension (name of coordinate)
    required_size : int
        Required coordinate system size

    Raises
    ------
    KeyError
        If required dim is not present in input coordinate system
    ValueError
        If required dimension is not of required size

    Note
    ----
    Presently assumes coordinate system of given dimension is 1D
    """

    handshake_dim(input_coords, required_dim)
    size = (
        input_coords.sizes[required_dim]
        if isinstance(input_coords, xr.DataArray)
        else input_coords[str(required_dim)].shape[0]
    )
    if size != required_size:
        raise ValueError(
            f"Coordinate size for required dim {required_dim} is not of size {required_size}"
        )


def map_coords(
    x: torch.Tensor,
    input_coords: CoordSystem,
    output_coords: CoordSystem,
    method: Literal["nearest"] = "nearest",
) -> tuple[torch.Tensor, CoordSystem]:
    """A basic interpolation util to map between coordinate systems with common
    dimensions. Namely, `output_coords` should consist of keys are present in
    `input_coords`. Note that `output_coords` do not need have all the dimensions of the
    `input_coords`. Does not support more advanced interpolation, such as between a regular
    and curvilinear grid. For such use-cases, use `fetch_data` or `prep_data_array` from
    `data/utils`.

    Parameters
    ----------
    x : torch.Tensor
        Input data to map
    input_coords : CoordSystem
        Respective input coordinate system
    output_coords : CoordSystem
        Target output coordinates to map.
    method : Literal[&quot;nearest&quot;], optional
        Method to use for mapping numeric coordinates, by default "nearest"

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Mapped data and coordinate system.

    Warning
    -------
    Use this function with caution. Only certain coordinate transformations are
    supported / tested. Consider doing complex transforms manually in the inference
    pipeline.

    Raises
    ------
    KeyError:
        If output coordinate has a dimension not in the input coordinate
    ValueError
        If value in non-numeric output coordinate is not in input coordinate
        If asked to interpolate between 2D lat/lon (curvilinear) coordinates
    """
    mapped_coords = input_coords.copy()

    for key, value in output_coords.items():
        if key in [
            "batch",
            "time",
            "lead_time",
        ]:  # TODO: Need better solution, time is numeric
            continue

        # Handling np.empty(0) (free coordinate system)
        if len(value) == 0:
            continue

        if key not in input_coords:
            raise KeyError(f"Output coordinate dim {key} not found in input coords")

        outc = value
        inc = mapped_coords[key]
        dim = list(input_coords).index(key)

        if np.all(np.isin(outc, inc)):
            if inc.shape[0] == outc.shape[0] and np.all(inc == outc):
                # skip interpolation if input and output coords are identical
                continue

            if key in ["lat", "lon"] and len(inc.shape) > 1 or len(outc.shape) > 1:
                # Guard against 2D lat/lon grids (curvilinear case)
                raise ValueError(f"Coordinate dim {key} in input or mapped coords is \
                        two-dimensional; please use fetch_data or \
                        prep_data_array to regrid/interpolate first.")

            # Roll condition
            first_element = outc[0]
            shift_amount = np.where(inc == first_element)[0][0]
            if np.array_equal(np.roll(inc, shift_amount), outc):
                x = torch.roll(x, shifts=shift_amount, dims=dim)
                mapped_coords[key] = outc
                continue

            # Slice condition
            indx = np.where(inc == outc[0])[0][0]
            inc_slice = inc[indx : indx + outc.shape[0]]
            if inc_slice.shape[0] == outc.shape[0] and np.all(inc_slice == outc):
                # Min here, to deal when coords have extra meta-data
                # TODO: Improve this method / outright remove
                x_slice = [slice(None)] * min([len(input_coords), x.ndim])
                x_slice[dim] = slice(indx, indx + outc.shape[0])
                x = x[x_slice]
                mapped_coords[key] = outc
                continue

            # Generic fall back
            if True:
                # sort inputs and outputs before np.isin
                indx_inc = inc.argsort()
                indx_outc = outc.argsort()
                indx_rev_outc = indx_outc.argsort()
                indx = np.where(
                    np.isin(inc[indx_inc], outc[indx_outc], assume_unique=True)
                )[0]

                # undo sorting
                indx = indx_inc[indx][indx_rev_outc]

                if len(indx) != len(value):
                    raise ValueError(
                        f"Output coord dim {key} contains values not present in input"
                    )

                mapped_coords[key] = outc
                x = torch.index_select(
                    x, dim, torch.tensor(indx, dtype=torch.int32, device=x.device)
                )
                continue

        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(
                f"For non-numeric coordinate, {key}, all values of output coords must be in the input coordinates. "
                + f"Some elements of {outc} are not in {inc}."
            )

        if method == "nearest":
            # Method = nearest
            c1 = np.repeat(inc[:, np.newaxis], outc.shape[0], axis=1)
            c2 = np.repeat(outc[np.newaxis, :], inc.shape[0], axis=0)
            c = np.abs(c1 - c2)

            idx = np.argmin(c, axis=0)

            x = torch.index_select(
                x, dim, torch.tensor(idx, dtype=torch.int32, device=x.device)
            )
            mapped_coords[key] = outc
            continue
        else:
            raise ValueError(f"Map method {method} not supported")

    # Only keep the first x.ndim keys from mapped_coords
    # TODO: Remove this when proper support for dim
    mapped_coords = OrderedDict(list(mapped_coords.items())[: x.ndim])
    return x, mapped_coords


def map_coords_xr(
    x: xr.DataArray,
    output_coords: CoordSystem,
    method: Literal["nearest"] = "nearest",
) -> xr.DataArray:
    """Map xarray DataArray to target coordinate system using selection or interpolation.

    Maps an input DataArray to match the coordinates specified in output_coords by
    selecting or interpolating along dimensions. Supports both numpy and cupy-backed
    DataArrays. Empty coordinate arrays are ignored, and warnings are issued for
    missing coordinate keys.

    Parameters
    ----------
    x : xr.DataArray
        Input DataArray to map. May be backed by numpy or cupy arrays.
    output_coords : CoordSystem
        Target coordinate system containing a subset of coordinates present in x.
        Dimensions not in output_coords are preserved from the input.
    method : Literal["nearest"], optional
        Interpolation method for numeric coordinates, by default "nearest"

    Returns
    -------
    xr.DataArray
        Mapped DataArray with coordinates matching output_coords where specified.
        Preserves all dimensions and coordinates not specified in output_coords.

    Raises
    ------
    KeyError
        If output coordinate dimension is not found in input DataArray
    ValueError
        If non-numeric coordinate values are not present in input coordinates
        If interpolation method is not supported
    """
    result = x.copy()

    # Build selection/interpolation dictionary
    sel_dict = {}
    interp_dict = {}

    for key, value in output_coords.items():
        # Ignore batch dimension
        if key == "batch":
            continue

        # Skip empty arrays (free coordinate system)
        if len(value) == 0:
            continue

        # Check if dimension exists in input
        if key not in result.dims and key not in result.coords:
            warnings.warn(
                f"Coordinate key '{key}' not found in input DataArray. "
                f"Available dims: {list(result.dims)}, "
                f"Available coords: {list(result.coords.keys())}"
            )
            continue

        # Get coordinate values from input DataArray
        if key in result.coords:
            coord_values = result.coords[key]
        elif key in result.dims:
            coord_values = result[key]
        else:
            continue  # Should not happen due to check above

        coord_array = (
            coord_values.values if hasattr(coord_values, "values") else coord_values
        )

        # Check if coordinate types are compatible
        # Check for datetime/timedelta first (these are not numeric for comparison purposes)
        is_datetime = np.issubdtype(value.dtype, np.datetime64) or np.issubdtype(
            coord_array.dtype, np.datetime64
        )
        is_timedelta = np.issubdtype(value.dtype, np.timedelta64) or np.issubdtype(
            coord_array.dtype, np.timedelta64
        )
        # Only treat as numeric if both are numeric AND neither is datetime/timedelta
        is_numeric = (
            not is_datetime
            and not is_timedelta
            and np.issubdtype(value.dtype, np.number)
            and np.issubdtype(coord_array.dtype, np.number)
        )

        # Check if all output values are in input (exact match)
        if is_numeric:
            # Numeric coordinate: check if values match exactly
            if len(value) == len(coord_array) and np.allclose(
                value, coord_array, equal_nan=True
            ):
                continue  # No change needed, exact match

            # Check if all values are present in input (can use selection)
            if np.all(np.isin(value, coord_array)):
                sel_dict[key] = value
            else:
                # Need interpolation for values not in input
                # xarray's interp uses coordinate names and handles dimension mapping
                interp_dict[key] = xr.DataArray(value, dims=[key])
        elif is_datetime or is_timedelta:
            # Datetime/timedelta coordinate: use direct equality comparison
            if len(value) == len(coord_array) and np.array_equal(value, coord_array):
                continue  # No change needed, exact match

            # Check if all values are present in input (can use selection)
            if np.all(np.isin(value, coord_array)):
                sel_dict[key] = value
            else:
                # Need interpolation for datetime/timedelta values not in input
                # xarray's interp uses coordinate names and handles dimension mapping
                interp_dict[key] = xr.DataArray(value, dims=[key])
        else:
            # Non-numeric coordinate: must use selection, all values must be present
            if not np.all(np.isin(value, coord_array)):
                raise ValueError(
                    f"For non-numeric coordinate '{key}', all values of output coords "
                    f"must be in the input coordinates. Some elements of {value} are "
                    f"not in {coord_array}."
                )
            sel_dict[key] = value

    # Apply selection first (exact matches)
    if sel_dict:
        result = result.sel(sel_dict)

    # Apply nearest-neighbor interpolation per dimension using torch
    if interp_dict:
        if method != "nearest":
            raise ValueError(f"Interpolation method '{method}' not supported")

        data = result.data
        is_cupy = cp is not None and isinstance(data, cp.ndarray)
        dims = list(result.dims)
        new_coords = dict(result.coords)

        for key, target_da in interp_dict.items():
            dim_idx = dims.index(key)
            src_raw = np.asarray(result.coords[key].values)
            tgt_raw = np.asarray(target_da.values)

            sort_order = np.argsort(src_raw)
            src_sorted = src_raw[sort_order]
            idx_sorted = np.searchsorted(src_sorted, tgt_raw)
            idx_sorted = np.clip(idx_sorted, 1, len(src_sorted) - 1)
            left = np.abs(tgt_raw - src_sorted[idx_sorted - 1])
            right = np.abs(tgt_raw - src_sorted[idx_sorted])
            idx_sorted = np.where(left <= right, idx_sorted - 1, idx_sorted)

            # Map back to original unsorted indices
            idx = sort_order[idx_sorted]

            # Index into the data array along this dimension
            if is_cupy:
                idx_arr = cp.asarray(idx)
                data = cp.take(data, idx_arr, axis=dim_idx)
            else:
                data = np.take(data, idx, axis=dim_idx)

            # Update the interpolated dimension coordinate
            new_coords[key] = (key, tgt_raw)

            # Re-index any non-dimension coordinates that depend on this dim
            for cname, cval in list(new_coords.items()):
                if cname == key:
                    continue
                if isinstance(cval, xr.Variable):
                    c_dims = cval.dims
                    c_data = cval.values
                elif isinstance(cval, xr.DataArray):
                    c_dims = cval.dims
                    c_data = cval.values
                elif isinstance(cval, tuple) and len(cval) == 2:
                    c_dims, c_data = cval
                    if isinstance(c_dims, str):
                        c_dims = (c_dims,)
                else:
                    continue

                if key in c_dims:
                    ax = list(c_dims).index(key)
                    c_data = np.take(np.asarray(c_data), idx, axis=ax)
                    new_coords[cname] = (c_dims, c_data)

        result = xr.DataArray(data=data, dims=dims, coords=new_coords)

    return result


def split_coords(
    x: torch.Tensor, coords: CoordSystem, dim: str = "variable"
) -> tuple[list[torch.Tensor], CoordSystem, np.ndarray]:
    """
    A utility function to split a dimension from a (x,coords) pair and convert it into
    a list of tensors, a CoordSystem, and the dimension that extract from coords.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    coords : CoordSystem
        Coordinates referring to the dimensions of x
    dim : str
        Name of the dimension in coords to split along

    Returns
    -------
    list[torch.Tensor]
        List of tensors extracted by splitting the extracted dimension from coords.
    CoordSystem
        The updated coord system with the extracted dimension removed.
    np.ndarray
        The values of the dimension extracted from the coordinate system.
    """

    if dim not in coords:
        raise ValueError(f"dim {dim} is not in coords: {list(coords)}.")

    reduced_coords = coords.copy()
    dim_index = list(reduced_coords).index(dim)
    values = reduced_coords.pop(dim)
    xs = [xi.squeeze(dim_index) for xi in x.split(1, dim=dim_index)]
    return xs, reduced_coords, values


def convert_multidim_to_singledim(
    coords: CoordSystem, return_mapping: bool = False
) -> tuple[CoordSystem, dict[str, list[str]]]:
    """Converts a set of coordinates from a complex coordinate system, which has some
    coordinates with multidimensional arrays, into a simple coordinate system
    containing only one-dimensional arrays.

    This conversion is done by creating individual indexes that are enumerations of
    the complex coordinate.

    Assumptions
    -----------
    This code assumes that if a coordinate entry is n-dimensional, then the following
    (n-1) coordinate entries in the ordered dictionary have the same shape and represent
    the same multidimensional grid.

    Example
    -------

    Suppose we have lat/lon coordinates represented by 2-dimensional grids.
    ```python
    lat = np.linspace(0, 1, 10)
    lon = np.linspace(0, 1, 20)
    LON, LAT = np.meshgrid(lat, lon)
    c = CoordSystem({"lat": LAT, "lon": LON})

    c1, m = convert_multidim_to_singledim(c)
    ```

    `c1` has 2 keys - `x1` and `x2`, with shapes `(10,)` and `(20,)` respectively.

    Parameters
    ----------
    coords : CoordSystem
        CoordSystem to convert.

    Returns
    -------
    CoordSystem
        Converted coordinate system where each coordinate is 1-dimensional.
    dict[str, list[str]]
        Mapping of multidimensional coordinates to a list of their 1-dimensional
        enumerations.
    """

    adjusted_coords = {}
    mapping: dict[str, list[str]] = {}

    items = list(coords.items())
    i = 0
    while i < len(items):
        item = items[i]
        k, v = item
        # Temp fix: TODO: REMOVE
        if k.startswith("_"):
            i += 1
            continue

        ndim = v.ndim
        if v.ndim < 2:
            adjusted_coords[k] = v
            i += 1
        else:
            s = v.shape
            mapping[k] = []

            for j in range(ndim):
                if i + j > len(items) - 1:
                    raise ValueError(
                        "Assumed that if an n-dimensional coordinate exists, "
                        "then there will be exactly n coordinates with the same shape."
                    )

                k1, v1 = items[i + j]
                if v1.shape != s:
                    raise ValueError(
                        "Assumed that if an n-dimensional coordinate exists, "
                        "then there will be exactly n coordinates with the same shape."
                    )

                adjusted_coords["i" + k1] = np.arange(s[j])
                mapping[k].append("i" + k1)
                mapping[k1] = mapping[k]

            i += j + 1

    return CoordSystem(adjusted_coords), mapping


def tile_coords(
    x: torch.Tensor, coords: CoordSystem, target_coords: CoordSystem
) -> tuple[torch.Tensor, CoordSystem]:
    """Tile tensor x to match dimensions in target_coords that don't exist in coords.

    This function tiles the input tensor to match leading dimensions from target_coords
    that are not present in coords. Dimensions that exist in both coords and target_coords
    are ignored in target_coords and use the values from coords instead.

    Parameters
    ----------
    x : torch.Tensor
        Source tensor to be tiled
    coords : CoordSystem
        Coordinate system for x tensor
    target_coords : CoordSystem
        Target coordinate system. Dimensions that exist in coords are ignored.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing the tiled tensor and updated coordinate system

    Examples
    --------
    Tiling a tensor to match additional dimensions from target_coords.

    ```python
    from collections import OrderedDict

    import numpy as np
    import torch

    from earth2studio.utils.coords import tile_coords

    x = torch.randn(3, 4)
    coords = OrderedDict(
        {
            "variable": np.array(["a", "b", "c"]),
            "time": np.array([0, 1, 2, 3]),
        }
    )

    target_coords = OrderedDict(
        {
            "batch": np.array([0, 1]),
            "ensemble": np.array([0, 1, 2]),
            "variable": np.array(["x", "y"]),  # Ignored, uses coords value
            "time": np.array([10, 20, 30, 40]),  # Ignored, uses coords value
        }
    )

    # Tile x to match batch and ensemble dimensions.
    x_tiled, out_coords = tile_coords(x, coords, target_coords)
    print(x_tiled.shape)
    print(list(out_coords.keys()))
    ```

    ```text
    torch.Size([2, 3, 3, 4])
    ['batch', 'ensemble', 'variable', 'time']
    ```
    """
    coords_keys = set(coords.keys())
    leading_dims = OrderedDict()
    common_dims = []

    for key, val in target_coords.items():
        if key not in coords_keys:
            leading_dims[key] = val
        else:
            common_dims.append(key)

    # Validate that all common dims are at the end of target_coords
    if common_dims:
        target_keys = list(target_coords.keys())
        coords_keys_list = list(coords.keys())
        if target_keys[-len(common_dims) :] != coords_keys_list:
            raise ValueError(
                f"All common dimensions must appear at the end of target_coords. "
                f"Common dimensions: {coords_keys_list}, "
                f"target_coords trailing keys: {target_keys[-len(common_dims):]}, "
                f"target_coords order: {target_keys}"
            )

    n_lead = len(leading_dims)

    out_coords = deepcopy(leading_dims)
    for key, val in coords.items():
        out_coords[key] = val

    # add leading size-1 dims so tile has one rep per dimension, then tile to match target
    reps = [len(dim) for dim in leading_dims.values()] + [1] * len(x.shape)
    x_tiled = x.view(*([1] * n_lead), *x.shape).tile(reps)

    return x_tiled, out_coords


def cat_coords(
    tensors: tuple[torch.Tensor, ...],
    coords: tuple[CoordSystem, ...],
    dim: str = "variable",
) -> tuple[torch.Tensor, CoordSystem]:
    """
    concatenate data along coordinate dimension.

    Parameters
    ----------
    tensors : tuple[torch.Tensor, ...]
        Tuple of input tensors to concatenate
    coords : tuple[CoordSystem, ...]
        Tuple of OrderedDicts representing coordinate systems for each tensor
    dim : str
        name of dimension along which to concatenate

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict from
        concatenated data.

    Raises
    ------
    ValueError
        If tensors and coords have different lengths
        If input tensors have different dimension names
        If non-concatenation dimensions don't match across inputs
    KeyError
        If concatenation dimension is not found in all coordinate systems
    """
    if len(tensors) != len(coords):
        raise ValueError(
            f"tensors and coords must have the same length, got {len(tensors)} tensors and {len(coords)} coords"
        )

    if len(tensors) == 0:
        raise ValueError("at least one tensor and coord must be provided")

    if len(tensors) == 1:
        return tensors[0], deepcopy(coords[0])

    # Use first coord as reference
    ref_coord = coords[0]

    # make sure cat dim is present in all tensors
    handshake_dim(ref_coord, dim)

    # make sure all tensors have the same dimension names
    ref_keys = list(ref_coord.keys())
    for i, coord in enumerate(coords[1:], start=1):
        handshake_dim(coord, dim)
        if not list(coord.keys()) == ref_keys:
            raise ValueError(
                f"all input tensors must have the same dimension names. "
                f"Tensor 0 has {ref_keys}, tensor {i} has {list(coord.keys())}"
            )

    # make sure all the other dimensions are of equal length
    other_dims = list(ref_keys)
    other_dims.remove(dim)
    for i, coord in enumerate(coords[1:], start=1):
        handshake_coords(ref_coord, coord, other_dims)

    # assemble output coords
    coz = deepcopy(ref_coord)
    coz[dim] = np.concatenate([coord[dim] for coord in coords])

    # concatenate tensors
    dim_index = list(coz).index(dim)
    zz = torch.cat(tensors, dim=dim_index)

    return zz, coz
