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

"""Shared helpers for DataArray IO backends and output planning."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from earth2studio.utils.coords import (
    E2S_DYNAMIC_DIMS,
    E2S_KIND,
    E2S_SCHEMA_VERSION,
    E2S_STATISTICS,
    coord_array,
)
from earth2studio.utils.type import CoordinateSystem

# Attributes determined by the signature or its variable labels; never stored
_DROPPED_ATTRS = (E2S_KIND, E2S_SCHEMA_VERSION, E2S_DYNAMIC_DIMS, E2S_STATISTICS)

Indexer: TypeAlias = slice | np.ndarray


@dataclass(frozen=True)
class ArrayPlan:
    """Storage an IO template requires.

    Attributes
    ----------
    names : tuple[str, ...]
        Stored array names, one per ``variable`` label or the template name.
    dims : tuple[Hashable, ...]
        Dimensions of each stored array, in template order without ``variable``.
    shape : tuple[int, ...]
        Size of each dimension in ``dims``.
    coords : dict[Hashable, xr.Variable]
        Dimension and auxiliary coordinates missing from the store, with their
        attributes.
    dtype : np.dtype
        Field dtype.
    fill_value : Any
        Value of unwritten positions: NaN for inexact dtypes, otherwise zero.
    attrs : dict[Hashable, Any]
        Attributes to persist on every stored array.
    """

    names: tuple[str, ...]
    dims: tuple[Hashable, ...]
    shape: tuple[int, ...]
    coords: dict[Hashable, xr.Variable]
    dtype: np.dtype
    fill_value: Any
    attrs: dict[Hashable, Any]


def output_template(
    signature: xr.DataArray,
    leading: Mapping[str, ArrayLike],
    coords: Mapping[str, ArrayLike] | None = None,
) -> CoordinateSystem:
    """Concretize a planning signature into an IO template.

    Parameters
    ----------
    signature : xr.DataArray
        Coordinate signature, typically ``model.output_coords(...)``. Its dynamic
        leading dimensions are replaced; a signature without any gains ``leading``
        as new leading dimensions.
    leading : Mapping[str, ArrayLike]
        Concrete leading dimensions and labels, in order, such as ``ensemble`` and
        ``time``.
    coords : Mapping[str, ArrayLike], optional
        Replacement labels for fixed, non-spatial dimensions, such as the run's
        ``lead_time`` extent. Replacing a dimension drops its dependent auxiliary
        coordinates.

    Returns
    -------
    CoordinateSystem
        Allocation-free template with no dynamic dimensions, preserving grid
        metadata, auxiliary coordinates, dtype, name and attributes.

    Raises
    ------
    ValueError
        If a leading dimension duplicates a fixed dimension, a replacement names an
        unknown or spatial dimension, or a resulting dimension is empty.
    """
    dynamic = tuple(signature.attrs.get(E2S_DYNAMIC_DIMS, ()))
    fixed = tuple(dim for dim in signature.dims if dim not in dynamic)
    replacements = dict(coords or {})
    duplicated = set(leading).intersection(fixed)
    if duplicated:
        raise ValueError(
            f"Leading dimensions are already fixed in the signature: "
            f"{sorted(duplicated)}; replace their labels with coords instead"
        )
    unknown = set(replacements).difference(fixed)
    if unknown:
        raise ValueError(f"Cannot replace unknown dimensions: {sorted(unknown)}")
    spatial_dims = set(signature.attrs.get("dims", ()))
    for key in replacements:
        if key in spatial_dims or spatial_dims.intersection(signature[key].dims):
            raise ValueError(
                "Use coord_array with a new grid to replace spatial coordinates"
            )

    removed = set(dynamic) | set(replacements)
    coordinates: dict[Hashable, Any] = {
        key: value.variable
        for key, value in signature.coords.items()
        if not removed.intersection(value.dims)
    }
    coordinates.update({key: np.asarray(value) for key, value in leading.items()})
    coordinates.update({key: np.asarray(value) for key, value in replacements.items()})
    sizes = {dim: size for dim, size in signature.sizes.items() if dim not in removed}
    template = coord_array(
        (*leading, *fixed),
        coordinates,
        sizes=sizes,
        dtype=signature.dtype,
        name=signature.name,
        attrs=signature.attrs,
    )
    _validate_template(template)
    return template


def plan_arrays(
    template: xr.DataArray, existing: Mapping[Hashable, xr.Variable] | None = None
) -> ArrayPlan:
    """Describe and validate the storage an IO template requires.

    Parameters
    ----------
    template : xr.DataArray
        Concrete template. Field values are never read.
    existing : Mapping[Hashable, xr.Variable], optional
        Coordinates already in the store.

    Returns
    -------
    ArrayPlan
        Array names, dimensions, missing coordinates, dtype and attributes.
        Coordinates depending on ``variable`` cannot be stored per array and are
        omitted.

    Raises
    ------
    ValueError
        If the template is not concrete, has repeated labels, names an array after a
        coordinate, or has coordinates that differ from the store.
    """
    _validate_template(template)
    existing = dict(existing or {})
    names = _array_names(template)
    dims = _field_dims(template)
    coords: dict[Hashable, xr.Variable] = {}
    for key, value in template.coords.items():
        if key == "variable" or "variable" in value.dims:
            continue
        variable = value.variable
        variable = xr.Variable(
            variable.dims,
            _normalize_labels(np.asarray(variable.values)),
            variable.attrs,
        )
        if key in existing:
            if not _same_variable(existing[key], variable):
                raise ValueError(f"Coordinate '{key}' differs from the store")
        else:
            coords[key] = variable
    clashes = set(names) & (set(template.coords) | set(existing))
    if clashes:
        raise ValueError(f"Array names collide with coordinates: {sorted(clashes)}")
    dtype = np.dtype(template.dtype)
    return ArrayPlan(
        names=names,
        dims=dims,
        shape=tuple(template.sizes[dim] for dim in dims),
        coords=coords,
        dtype=dtype,
        fill_value=np.nan if np.issubdtype(dtype, np.inexact) else 0,
        attrs={
            key: value
            for key, value in template.attrs.items()
            if key not in _DROPPED_ATTRS
        },
    )


def plan_write(
    x: xr.DataArray,
    arrays: Mapping[str, Mapping[Hashable, int]],
    coords: Mapping[Hashable, np.ndarray],
) -> list[tuple[str, dict[Hashable, Indexer], xr.DataArray]]:
    """Validate a write and locate it in a store before anything is written.

    Parameters
    ----------
    x : xr.DataArray
        Field to write. It is never modified.
    arrays : Mapping[str, Mapping[Hashable, int]]
        Stored array names and their dimension sizes, in dimension order.
    coords : Mapping[Hashable, np.ndarray]
        Stored dimension labels. Dimensions without labels must be written in
        full.

    Returns
    -------
    list[tuple[str, dict[Hashable, Indexer], xr.DataArray]]
        Per array: its name, positional indexers in its dimension order (slices
        for contiguous runs, otherwise ascending positions), and the field
        reordered to match. Fields keep the payload backing of ``x``.

    Raises
    ------
    ValueError
        If an array is unknown, dimensions differ, or a label is unknown or
        duplicated.
    """
    names = _array_names(x)
    dims = _field_dims(x)
    sizes = _check_arrays(names, dims, arrays, ordered=True)

    indexers: dict[Hashable, Indexer] = {}
    reorder: dict[Hashable, np.ndarray] = {}
    for dim, positions in _locate(x, dims, sizes, coords).items():
        if positions is None:
            indexers[dim] = slice(None)
            continue
        order = np.argsort(positions, kind="stable")
        indexers[dim] = _indexer(positions[order])
        if np.any(order != np.arange(order.size)):
            reorder[dim] = order

    field = x.isel(reorder) if reorder else x
    if "variable" not in x.dims:
        return [(names[0], indexers, field)]
    return [
        (name, indexers, field.isel(variable=index, drop=True))
        for index, name in enumerate(names)
    ]


def plan_read(
    selection: xr.DataArray | Mapping[Hashable, ArrayLike],
    arrays: Mapping[str, Mapping[Hashable, int]],
    coords: Mapping[Hashable, np.ndarray],
) -> tuple[xr.DataArray, tuple[str, ...], dict[Hashable, Indexer]]:
    """Validate a read and locate its labels in a store.

    Parameters
    ----------
    selection : xr.DataArray | Mapping[Hashable, ArrayLike]
        Template or field, or an ordered mapping from every dimension to its
        labels, in any dimension order. ``variable`` labels select arrays;
        without a ``variable`` dimension, a DataArray's name does. Field values
        are never read.
    arrays : Mapping[str, Mapping[Hashable, int]]
        Stored array names and their dimension sizes, in dimension order.
    coords : Mapping[Hashable, np.ndarray]
        Stored dimension labels. Dimensions without labels are read in full.

    Returns
    -------
    tuple[xr.DataArray, tuple[str, ...], dict[Hashable, Indexer]]
        The selection as a DataArray, the array names to read, and positional
        indexers in the selection's label order (slices for ascending contiguous
        runs).

    Raises
    ------
    ValueError
        If an array is unknown, dimensions differ, or a label is unknown or
        duplicated.
    """
    if not isinstance(selection, xr.DataArray):
        selection = coord_array(tuple(selection), dict(selection))
    names = _array_names(selection)
    dims = _field_dims(selection)
    sizes = _check_arrays(names, dims, arrays, ordered=False)
    indexers: dict[Hashable, Indexer] = {
        dim: slice(None) if positions is None else _indexer(positions)
        for dim, positions in _locate(selection, dims, sizes, coords).items()
    }
    return selection, names, indexers


def _validate_template(template: xr.DataArray) -> None:
    """Require concrete, nonempty dimensions with unique labels."""
    dynamic = tuple(template.attrs.get(E2S_DYNAMIC_DIMS, ()))
    if dynamic:
        raise ValueError(
            f"IO templates must be concrete; resolve dynamic dimensions {dynamic} first"
        )
    for dim, size in template.sizes.items():
        if size == 0:
            raise ValueError(f"IO template dimension '{dim}' must be nonempty")
        if dim in template.indexes and not template.indexes[dim].is_unique:
            raise ValueError(f"IO template labels along '{dim}' must be unique")


def _array_names(x: xr.DataArray) -> tuple[str, ...]:
    """Stored array names: ``variable`` labels verbatim, otherwise the name."""
    if "variable" in x.dims:
        if "variable" not in x.coords:
            raise ValueError("The variable dimension must have labels")
        names = tuple(str(name) for name in x.coords["variable"].values)
        if len(set(names)) != len(names):
            raise ValueError("Variable labels must be unique")
        return names
    if x.name is None:
        raise ValueError("Arrays without a variable dimension must be named")
    return (str(x.name),)


def _field_dims(x: xr.DataArray) -> tuple[Hashable, ...]:
    """Dimensions of the stored arrays for a template or field."""
    return tuple(dim for dim in x.dims if dim != "variable")


def _normalize_labels(values: np.ndarray) -> np.ndarray:
    """Store datetime and timedelta labels at nanosecond precision."""
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[ns]")
    return values


def _same_variable(left: xr.Variable, right: xr.Variable) -> bool:
    """Compare coordinate dimensions and values, treating NaNs as equal."""
    if left.dims != right.dims or left.shape != right.shape:
        return False
    a, b = np.asarray(left.values), np.asarray(right.values)
    if np.issubdtype(a.dtype, np.inexact):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _check_arrays(
    names: tuple[str, ...],
    dims: tuple[Hashable, ...],
    arrays: Mapping[str, Mapping[Hashable, int]],
    ordered: bool,
) -> Mapping[Hashable, int]:
    """Require stored arrays with the given dimensions; return their sizes."""
    for name in names:
        if name not in arrays:
            raise ValueError(f"Array '{name}' was not created with add_array")
        stored = tuple(arrays[name])
        same = stored == dims if ordered else set(stored) == set(dims)
        if not same or len(stored) != len(dims):
            raise ValueError(
                f"Dimensions {dims} differ from array '{name}' dimensions {stored}"
            )
    return arrays[names[0]]


def _locate(
    x: xr.DataArray,
    dims: tuple[Hashable, ...],
    sizes: Mapping[Hashable, int],
    coords: Mapping[Hashable, np.ndarray],
) -> dict[Hashable, np.ndarray | None]:
    """Map the labels of ``x`` to stored positions; None selects a whole axis.

    Labelled dimensions are matched label by label. Unlabelled dimensions must
    span the whole stored axis.
    """
    positions: dict[Hashable, np.ndarray | None] = {}
    for dim in dims:
        if dim not in x.coords:
            if x.sizes[dim] != sizes[dim]:
                raise ValueError(
                    f"Dimension '{dim}' has no labels, so it must span all "
                    f"{sizes[dim]} stored positions"
                )
            positions[dim] = None
        elif dim not in coords:
            raise ValueError(f"The store has no labels along '{dim}'")
        else:
            labels = _normalize_labels(np.asarray(x.coords[dim].values))
            positions[dim] = _label_positions(dim, labels, coords[dim])
    return positions


def _label_positions(
    dim: Hashable, labels: np.ndarray, store: np.ndarray
) -> np.ndarray:
    """Find the stored index of each label, e.g. lead times [12h, 6h] -> [2, 1].

    Raises if labels are empty, repeated or missing from the store.
    """
    if labels.size == 0:
        raise ValueError(f"Expected at least one label along '{dim}'")
    if len(pd.unique(labels)) != len(labels):
        raise ValueError(f"Labels along '{dim}' must be unique")
    positions = pd.Index(store).get_indexer(labels)
    if (positions < 0).any():
        unknown = [str(label) for label in pd.Index(labels[positions < 0])]
        raise ValueError(f"Labels along '{dim}' are not in the store: {unknown}")
    return positions


def _indexer(positions: np.ndarray) -> Indexer:
    """Use a slice for an ascending contiguous run of positions."""
    if np.all(np.diff(positions) == 1):
        return slice(int(positions[0]), int(positions[-1]) + 1)
    return positions
