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
from earth2studio.utils.type import CoordinateSystem


def _broadcast_weights(
    weights: torch.Tensor | np.ndarray | None,
    rd: list[str],
    x: xr.DataArray,
    device: torch.device,
) -> torch.Tensor:
    shape = [x.sizes[d] if d in rd else 1 for d in x.dims]
    if weights is None:
        return torch.ones(shape, device=device)
    dim_shapes = [x.sizes[d] for d in x.dims if d in rd]
    if list(weights.shape) != dim_shapes:
        raise AssertionError(
            "Error, weights have don't have an appropriate shape.\n"
            + f"weights have shape {weights.shape} but should have shape {dim_shapes}."
        )
    weights_t = torch.as_tensor(weights, device=device)
    if weights_t.dtype in (torch.float16, torch.bfloat16):
        weights_t = weights_t.float()
    return weights_t.reshape(shape)


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
    for coord in {"lat", "lon"} & x.coords.keys() & y.coords.keys():
        handshake_coords(x, y, coord)
    return y
