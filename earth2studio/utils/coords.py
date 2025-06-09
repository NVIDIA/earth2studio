# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Literal

import numpy as np
import torch

from earth2studio.utils.type import CoordSystem


def handshake_dim(
    input_coords: CoordSystem,
    required_dim: str,
    required_index: int | None = None,
) -> None:
    """Simple check to see if coordinate system has a dimension in a particular index

    Parameters
    ----------
    input_coords : CoordSystem
        Input coordinate system to validate
    required_dim : str
        Required dimension (name of coordinate)
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

    if required_dim not in input_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in input coordinates"
        )

    input_dims = list(input_coords.keys())

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
    input_coords: CoordSystem,
    target_coords: CoordSystem,
    required_dim: str,
) -> None:
    """Simple check to see if the required dimensions have the same coordinate system

    Parameters
    ----------
    input_coords : CoordSystem
        Input coordinate system to validate
    target_coords : CoordSystem
        Target coordinate system
    required_dim : str
        Required dimension (name of coordinate)
    Raises
    ------
    KeyError
        If required dim is not present in coordinate systems
    ValueError
        If coordinates of required dimensions don't match
    """
    if required_dim not in input_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in input coordinates"
        )

    if required_dim not in target_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in target coordinates"
        )

    if input_coords[required_dim].shape != target_coords[required_dim].shape:
        raise ValueError(
            f"Coordinate systems for required dim {required_dim} are not the same"
        )

    if not np.all(input_coords[required_dim] == target_coords[required_dim]):
        raise ValueError(
            f"Coordinate systems for required dim {required_dim} are not the same"
        )


def handshake_size(
    input_coords: CoordSystem,
    required_dim: str,
    required_size: int,
) -> None:
    """Simple check to see if a coordinate system of a given dimension is a required
    size

    Parameters
    ----------
    input_coords : CoordSystem
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

    if required_dim not in input_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in input coordinates"
        )

    if input_coords[required_dim].shape[0] != required_size:
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
                raise ValueError(
                    f"Coordinate dim {key} in input or mapped coords is \
                        two-dimensional; please use fetch_data or \
                        prep_data_array to regrid/interpolate first."
                )

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
