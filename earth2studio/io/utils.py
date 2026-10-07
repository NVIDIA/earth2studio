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
import torch
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
DROPPED_ATTRS = (E2S_KIND, E2S_SCHEMA_VERSION, E2S_DYNAMIC_DIMS, E2S_STATISTICS)

Indexer: TypeAlias = slice | np.ndarray


@dataclass(frozen=True)
class ArrayPlan:
    """Storage layout described by an IO schema.

    Attributes
    ----------
    names : tuple[str, ...]
        Stored array names, one per ``variable`` label or the schema name.
    dims : tuple[Hashable, ...]
        Dimensions of each stored array, in schema order without ``variable``.
    shape : tuple[int, ...]
        Size of each dimension in ``dims``.
    coords : dict[Hashable, xr.Variable]
        Dimension and auxiliary coordinates to store, with their attributes.
    dtype : np.dtype
        Field dtype.
    attrs : dict[Hashable, Any]
        Attributes to persist on every stored array.
    """

    names: tuple[str, ...]
    dims: tuple[Hashable, ...]
    shape: tuple[int, ...]
    coords: dict[Hashable, xr.Variable]
    dtype: np.dtype
    attrs: dict[Hashable, Any]


def output_schema(
    signature: xr.DataArray,
    leading: Mapping[str, ArrayLike],
    coords: Mapping[str, ArrayLike] | None = None,
) -> CoordinateSystem:
    """Concretize a planning signature into an IO schema.

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
        Allocation-free schema with no dynamic dimensions, preserving grid
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
    schema = coord_array(
        (*leading, *fixed),
        coordinates,
        sizes=sizes,
        dtype=signature.dtype,
        name=signature.name,
        attrs=signature.attrs,
    )
    validate_schema(schema)
    return schema


def validate_schema(schema: xr.DataArray) -> None:
    """Require a concrete IO schema.

    Parameters
    ----------
    schema : xr.DataArray
        Schema to check. Only dimensions and attributes are inspected.

    Raises
    ------
    ValueError
        If a dimension is dynamic or empty.
    """
    dynamic = tuple(schema.attrs.get(E2S_DYNAMIC_DIMS, ()))
    if dynamic:
        raise ValueError(
            f"IO schemas must be concrete; resolve dynamic dimensions {dynamic} first"
        )
    for dim, size in schema.sizes.items():
        if size == 0:
            raise ValueError(f"IO schema dimension '{dim}' must be nonempty")


def array_names(x: xr.DataArray) -> tuple[str, ...]:
    """Name the stored arrays a schema or field maps to.

    Parameters
    ----------
    x : xr.DataArray
        Schema or field.

    Returns
    -------
    tuple[str, ...]
        ``variable`` labels verbatim, or the array name without a ``variable``
        dimension.

    Raises
    ------
    ValueError
        If ``x`` has neither a labelled ``variable`` dimension nor a name.
    """
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


def field_dims(x: xr.DataArray) -> tuple[Hashable, ...]:
    """Return the dimensions of the stored arrays for a schema or field."""
    return tuple(dim for dim in x.dims if dim != "variable")


def persisted_attrs(attrs: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
    """Drop signature markers and label-derived statistics from attributes."""
    return {key: value for key, value in attrs.items() if key not in DROPPED_ATTRS}


def _normalize_labels(values: np.ndarray) -> np.ndarray:
    """Store datetime and timedelta labels at nanosecond precision."""
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]")
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[ns]")
    return values


def plan_arrays(schema: xr.DataArray) -> ArrayPlan:
    """Describe the storage an IO schema requires.

    Parameters
    ----------
    schema : xr.DataArray
        Concrete schema. Field values are never read.

    Returns
    -------
    ArrayPlan
        Array names, dimensions, coordinates, dtype and attributes to store.
        Coordinates depending on ``variable`` cannot be stored per array and are
        omitted.
    """
    validate_schema(schema)
    names = array_names(schema)
    dims = field_dims(schema)
    coords: dict[Hashable, xr.Variable] = {}
    for key, value in schema.coords.items():
        if key == "variable" or "variable" in value.dims:
            continue
        variable = value.variable
        coords[key] = xr.Variable(
            variable.dims,
            _normalize_labels(np.asarray(variable.values)),
            variable.attrs,
        )
    return ArrayPlan(
        names=names,
        dims=dims,
        shape=tuple(schema.sizes[dim] for dim in dims),
        coords=coords,
        dtype=np.dtype(schema.dtype),
        attrs=persisted_attrs(schema.attrs),
    )


def merge_coords(
    existing: Mapping[Hashable, xr.Variable], new: Mapping[Hashable, xr.Variable]
) -> dict[Hashable, xr.Variable]:
    """Check new coordinates against a store and return those to create.

    Parameters
    ----------
    existing : Mapping[Hashable, xr.Variable]
        Coordinates already in the store.
    new : Mapping[Hashable, xr.Variable]
        Coordinates a schema requires.

    Returns
    -------
    dict[Hashable, xr.Variable]
        Coordinates missing from the store.

    Raises
    ------
    ValueError
        If a coordinate exists with different dimensions or values.
    """
    missing = {}
    for key, value in new.items():
        if key not in existing:
            missing[key] = value
            continue
        current = existing[key]
        same = current.dims == value.dims and current.shape == value.shape
        if same:
            left, right = np.asarray(current.values), np.asarray(value.values)
            same = bool(
                np.array_equal(left, right, equal_nan=True)
                if np.issubdtype(left.dtype, np.inexact)
                else np.array_equal(left, right)
            )
        if not same:
            raise ValueError(f"Coordinate '{key}' differs from the store")
    return missing


def fill_value(dtype: np.dtype) -> Any:
    """Return the value of unwritten positions: NaN for inexact dtypes, else zero."""
    return np.nan if np.issubdtype(dtype, np.inexact) else 0


def _positions(dim: Hashable, labels: np.ndarray, store: np.ndarray) -> np.ndarray:
    """Map unique, nonempty labels to their store positions."""
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


def _check_arrays(
    names: tuple[str, ...],
    dims: tuple[Hashable, ...],
    arrays: Mapping[str, tuple[Hashable, ...]],
    ordered: bool = True,
) -> None:
    """Require stored arrays with the given dimensions, in order if ``ordered``."""
    for name in names:
        if name not in arrays:
            raise ValueError(f"Array '{name}' was not created with add_array")
        stored = arrays[name]
        same = stored == dims if ordered else set(stored) == set(dims)
        if not same or len(stored) != len(dims):
            raise ValueError(
                f"Dimensions {dims} differ from array '{name}' dimensions {stored}"
            )


def _locate(
    x: xr.DataArray,
    dims: tuple[Hashable, ...],
    coords: Mapping[Hashable, np.ndarray],
    operation: str,
) -> dict[Hashable, np.ndarray | None]:
    """Map the labels of ``x`` to store positions; None selects a whole axis."""
    positions: dict[Hashable, np.ndarray | None] = {}
    for dim in dims:
        if dim not in coords:
            if dim in x.coords:
                raise ValueError(f"The store has no labels along '{dim}'")
            positions[dim] = None
        elif dim not in x.coords:
            if x.sizes[dim] != len(coords[dim]):
                raise ValueError(
                    f"Unlabelled dimension '{dim}' must be {operation} in full"
                )
            positions[dim] = None
        else:
            labels = _normalize_labels(np.asarray(x.coords[dim].values))
            positions[dim] = _positions(dim, labels, coords[dim])
    return positions


def plan_write(
    x: xr.DataArray,
    arrays: Mapping[str, tuple[Hashable, ...]],
    coords: Mapping[Hashable, np.ndarray],
) -> list[tuple[str, dict[Hashable, Indexer], xr.DataArray]]:
    """Validate a write and locate it in a store before anything is written.

    Parameters
    ----------
    x : xr.DataArray
        Field to write. It is never modified.
    arrays : Mapping[str, tuple[Hashable, ...]]
        Stored array names and their dimensions.
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
    names = array_names(x)
    dims = field_dims(x)
    _check_arrays(names, dims, arrays)

    indexers: dict[Hashable, Indexer] = {}
    reorder: dict[Hashable, np.ndarray] = {}
    for dim, positions in _locate(x, dims, coords, "written").items():
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


def read_selection(
    selection: xr.DataArray | Mapping[Hashable, ArrayLike],
) -> xr.DataArray:
    """Normalize a read selection to a coordinate signature.

    Parameters
    ----------
    selection : xr.DataArray | Mapping[Hashable, ArrayLike]
        Schema, field or ordered mapping from every dimension to its labels. A
        mapping selects arrays with its ``variable`` labels.

    Returns
    -------
    xr.DataArray
        The selection itself, or an allocation-free signature of the mapping.
    """
    if isinstance(selection, xr.DataArray):
        return selection
    return coord_array(tuple(selection), dict(selection))


def plan_read(
    selection: xr.DataArray,
    arrays: Mapping[str, tuple[Hashable, ...]],
    coords: Mapping[Hashable, np.ndarray],
) -> tuple[tuple[str, ...], dict[Hashable, Indexer]]:
    """Validate a read and locate its labels in a store.

    Parameters
    ----------
    selection : xr.DataArray
        Schema or field whose dimensions, labels and arrays to read, in any
        dimension order. Field values are never read.
    arrays : Mapping[str, tuple[Hashable, ...]]
        Stored array names and their dimensions.
    coords : Mapping[Hashable, np.ndarray]
        Stored dimension labels. Dimensions without labels are read in full.

    Returns
    -------
    tuple[tuple[str, ...], dict[Hashable, Indexer]]
        Array names to read, and positional indexers in the selection's label
        order (slices for ascending contiguous runs).

    Raises
    ------
    ValueError
        If an array is unknown, dimensions differ, or a label is unknown or
        duplicated.
    """
    names = array_names(selection)
    dims = field_dims(selection)
    _check_arrays(names, dims, arrays, ordered=False)
    indexers: dict[Hashable, Indexer] = {
        dim: slice(None) if positions is None else _indexer(positions)
        for dim, positions in _locate(selection, dims, coords, "read").items()
    }
    return names, indexers


def stack_fields(
    selection: xr.DataArray, names: tuple[str, ...], fields: list[xr.DataArray]
) -> xr.DataArray:
    """Assemble per-array reads into one field in the selection's layout.

    Parameters
    ----------
    selection : xr.DataArray
        Read selection, giving dimension order and the ``variable`` dimension.
    names : tuple[str, ...]
        Array names, aligned with ``fields``.
    fields : list[xr.DataArray]
        Selected stored arrays.

    Returns
    -------
    xr.DataArray
        Field with the selection's dimensions. Attributes shared by every array
        are kept.
    """
    if "variable" not in selection.dims:
        return fields[0].transpose(*selection.dims).rename(names[0])
    field = xr.concat(
        fields,
        dim=pd.Index(names, name="variable"),
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )
    return field.transpose(*selection.dims).rename(selection.name)


def to_device(x: xr.DataArray, device: torch.device | str = "cpu") -> xr.DataArray:
    """Place field values on a device: NumPy on CPU, CuPy on CUDA.

    Parameters
    ----------
    x : xr.DataArray
        NumPy-, CuPy- or Torch-backed field.
    device : torch.device | str, optional
        Destination, by default "cpu".

    Returns
    -------
    xr.DataArray
        Field on ``device``.
    """
    device = torch.device(device)
    if device.type == "cpu":
        return x.e2s.as_numpy()
    if device.type == "cuda":
        return x.e2s.as_cupy(device.index)
    raise ValueError(f"Unsupported device '{device}'")


def to_host(x: xr.DataArray) -> np.ndarray:
    """Return the field values as a NumPy array.

    NumPy payloads are returned without copying; CuPy and Torch payloads are
    transferred to the host.

    Parameters
    ----------
    x : xr.DataArray
        NumPy-, CuPy- or Torch-backed field.

    Returns
    -------
    np.ndarray
        Host values. Callers must not modify them in place.
    """
    return np.asarray(x.e2s.as_numpy().data)
