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

"""Private helpers shared by statistics and metrics"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.coords import E2S_DYNAMIC_DIMS, coord_array, handshake_coords
from earth2studio.utils.cupy import _get_cupy
from earth2studio.utils.type import CoordinateSystem


def _as_host_or_cupy(x: xr.DataArray) -> xr.DataArray:
    """Ensure a DataArray is backed by NumPy or CuPy (not Torch) data

    Parameters
    ----------
    x : xr.DataArray
        Input DataArray

    Returns
    -------
    xr.DataArray
        Input if already NumPy/CuPy backed, otherwise a converted DataArray
    """
    if isinstance(x.variable._data, np.ndarray) or x.e2s.is_cupy:
        return x
    return x.e2s.to_backend("auto")


def _array_module(x: xr.DataArray) -> Any:
    """Array module (numpy or cupy) matching the storage backing a DataArray"""
    return _get_cupy() if x.e2s.is_cupy else np


def _to_xp(a: Any, xp: Any, dtype: Any = None) -> Any:
    """Convert a torch tensor / numpy / cupy array (or array-like) into an array of the
    requested array module"""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    elif hasattr(a, "get") and not isinstance(a, np.ndarray):
        # cupy array
        a = a.get()
    out = xp.asarray(a)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def _axes(x: xr.DataArray, dims: Sequence[str]) -> tuple[int, ...]:
    """Axis indices of the named dimensions of x"""
    return tuple(x.dims.index(d) for d in dims)


def _validate_dims(x: xr.DataArray, dims: Sequence[str]) -> None:
    """Raise if any of the dimension names are not present in x"""
    if not all(d in x.dims for d in dims):
        raise ValueError(
            "Initialized reduction_dimensions do not appear in passed coords."
        )


def _broadcast_weights(weights: Any, rd: list[str], x: xr.DataArray, xp: Any) -> Any:
    """
    Broadcast weights to appropriate dimensions. This is meant for internal use.

    Parameters
    ----------
    weights : torch.Tensor | np.ndarray | None
        An array containing weights to broadcast to the shape of x. Can also
        be None, in which a 1.0 is returned.
    rd : list[str]
        A list of dimension names corresponding to the dimensions of weights.
    x : xr.DataArray
        Array (or coordinate system) the weights will be broadcasted to.
    xp : Any
        Array module (numpy or cupy) to place the weights in
    """
    if weights is None:
        shape = [x.sizes[d] if d in rd else 1 for d in x.dims]
        return xp.ones(shape, dtype=x.dtype)
    dim_shapes = [x.sizes[d] for d in x.dims if d in rd]
    if list(weights.shape) != dim_shapes:
        raise AssertionError(
            "Error, weights have don't have an appropriate shape.\n"
            + f"weights have shape {weights.shape} but should have shape {dim_shapes}."
        )
    weights = _to_xp(
        weights, xp, x.dtype if np.issubdtype(x.dtype, np.floating) else None
    )
    shape = [x.sizes[d] if d in rd else 1 for d in x.dims]
    return weights.reshape(shape)


def _output_signature(
    template: CoordinateSystem,
    remove: Sequence[str] = (),
    add: "Mapping[str, Any] | None" = None,
    add_first: bool = False,
    dtype: Any = None,
) -> CoordinateSystem:
    """Build the output coordinate system of a statistic

    Parameters
    ----------
    template : CoordinateSystem
        Input coordinate system (or data)
    remove : Sequence[str]
        Dimensions removed by the statistic. Coordinates depending on these
        dimensions are removed as well.
    add : Mapping[str, Any] | None
        New one dimensional dimensions / coordinates to add, by default None
    add_first : bool
        Place the added dimensions at the front of the output rather than the
        end, by default False
    dtype : Any
        Output dtype, defaults to template dtype

    Returns
    -------
    CoordinateSystem
        Allocation-free coordinate signature
    """
    for d in remove:
        if d not in template.dims:
            raise KeyError(f"Dimension {d} not found in input coordinates")
    dims = [d for d in template.dims if d not in remove]
    coords: dict[str, Any] = {
        key: val.variable
        for key, val in template.coords.items()
        if key not in remove and set(val.dims) <= set(dims)
    }
    sizes = {d: template.sizes[d] for d in dims}
    new_dims: list[str] = []
    for key, val in (add or {}).items():
        new_dims.append(key)
        coords[key] = np.asarray(val)
    dims = new_dims + dims if add_first else dims + new_dims
    dynamic = tuple(d for d in template.attrs.get(E2S_DYNAMIC_DIMS, ()) if d in dims)
    if tuple(dims[: len(dynamic)]) != dynamic:
        dynamic = ()
    return coord_array(
        dims,
        coords,
        dynamic=dynamic,
        sizes=sizes,
        dtype=template.dtype if dtype is None else dtype,
    )


def _wrap(data: Any, signature: CoordinateSystem) -> xr.DataArray:
    """Create a DataArray from raw data (numpy / cupy / torch) and an output
    coordinate signature"""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return xr.DataArray(
        data,
        dims=signature.dims,
        coords={k: v.variable for k, v in signature.coords.items()},
    )


def _spatial_dims_to_end(
    x: xr.DataArray, spatial_dimensions: Sequence[str]
) -> xr.DataArray:
    """Permute spatial dimensions to the last dimensions."""
    order = [d for d in x.dims if d not in spatial_dimensions] + list(
        spatial_dimensions
    )
    return x.transpose(*order)


def _align_xy(
    x: xr.DataArray, y: xr.DataArray, ensemble_dimension: str | None = None
) -> xr.DataArray:
    """Check that x (ignoring an optional ensemble dimension) and y share dimensions
    and coordinates, returning y transposed into the dimension order of x.

    Parameters
    ----------
    x : xr.DataArray
        Forecast / prediction
    y : xr.DataArray
        Observation / validation
    ensemble_dimension : str | None
        Dimension present only in x, by default None

    Raises
    ------
    ValueError
        If the ensemble dimension is missing in x or present in y or if the remaining
        dimensions / coordinates are incompatible
    """
    if ensemble_dimension is not None:
        if ensemble_dimension not in x.dims:
            raise ValueError(
                f"Ensemble dimension {ensemble_dimension} must be in x dimensions."
            )
        if ensemble_dimension in y.dims:
            raise ValueError(
                f"Ensemble dimension {ensemble_dimension} must not be in y dimensions."
            )
    x_dims = [d for d in x.dims if d != ensemble_dimension]
    if set(x_dims) != set(y.dims):
        raise ValueError(
            f"Dimensions of x {tuple(x_dims)} and y {tuple(y.dims)} are incompatible."
        )
    y = y.transpose(*x_dims)
    for d in x_dims:
        if d in x.coords and d in y.coords:
            try:
                handshake_coords(x, y, d)
            except (ValueError, KeyError) as e:
                raise ValueError("Coordinates are incompatible.") from e
        elif x.sizes[d] != y.sizes[d]:
            raise ValueError("Coordinates are incompatible.")
    return y
