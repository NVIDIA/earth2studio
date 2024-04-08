# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Literal, Optional

import numpy as np
import torch

from earth2studio.utils.type import CoordSystem


def handshake_dim(
    input_coords: CoordSystem,
    required_dim: str,
    required_index: Optional[int] = None,
) -> None:
    """Simple check to see if coordinate system has a dimension in a particular index

    Parameters
    ----------
    input_coords : CoordSystem
        Input coordinate system to validate
    required_dim : str
        Required dimension (name of coordinate)
    required_index : optional[int], optional
        Required index of dimension if needed, by default None

    Raises
    ------
    KeyError
        If required dimension is not found in the input coordinate system
    ValueError
        If the required index is outside the dimensionality of the input coordinate system
    ValueError
        If dimension is not in the required index

    Returns
    -------
        None
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

    Returns
    -------
        None
    """
    if required_dim not in input_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in input coordinates"
        )

    if required_dim not in target_coords:
        raise KeyError(
            f"Required dimension {required_dim} not found in target coordinates"
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

    Returns
    -------
        None

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
    ignore_batch: bool = True,
) -> tuple[torch.Tensor, CoordSystem]:
    """A basic interpolation util to map between coordinate systems with common
    dimensions. Namely, `output_coords` should consist of keys are present in
    `input_coords`. Note that `output_coords` do not need have all the dimensions of the
    `input_coords`.

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
    ignore_batch: bool, optional
        Ignore batch dimension in output coordinate if present, by default True

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Mapped data and coordinate system.

    Raises
    ------
    KeyError:
        If output coordinate has a dimension not in the input coordinate
    ValueError
        If value in non-numeric output coordinate is not in input coordinate
    """
    mapped_coords = input_coords.copy()

    for key, value in output_coords.items():
        if key in [
            "batch",
            "time",
            "lead_time",
        ]:  # TODO: Need better solution, time is numeric
            continue

        if key not in input_coords:
            raise KeyError(f"Output coordinate dim {key} not found in input coords")

        outc = value
        inc = mapped_coords[key]
        dim = list(input_coords).index(key)

        if not np.issubdtype(value.dtype, np.number):
            if not np.all(np.isin(outc, inc)):
                raise ValueError(f"Error! Some elements of {outc} are not in {inc}.")
            # Not numerical just sub select

            # sort inputs and outputs before np.in1d
            indx_inc = inc.argsort()
            indx_outc = outc.argsort()
            indx_rev_outc = indx_outc.argsort()
            indx = np.where(
                np.in1d(inc[indx_inc], outc[indx_outc], assume_unique=True)
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

        else:

            # Method = nearest
            c1 = np.repeat(inc[:, np.newaxis], outc.shape[0], axis=1)
            c2 = np.repeat(outc[np.newaxis, :], inc.shape[0], axis=0)
            c = np.abs(c1 - c2)

            idx = np.argmin(c, axis=0)

            x = torch.index_select(
                x, dim, torch.tensor(idx, dtype=torch.int32, device=x.device)
            )
            mapped_coords[key] = outc

            # TODO: Linear
            # c = np.pad(array, pad_width=1, mode='edge')
            # idx2 = numpy.where(c[idx+2] < c[idx] , idx+1, idx-1)

            # a = torch.Tensor(input_coords[key][idx2] - input_coords[key][idx], device=x.device)

            # y0 = torch.index_select(x, i, torch.IntTensor(idx, device=x.device))
            # y1 = torch.index_select(x, i, torch.IntTensor(idx2, device=x.device))

            # x0 = torch.Tensor(value - input_coords[key][idx], device=x.device)
            # x1 = torch.Tensor(input_coords[key][idx2] - value, device=x.device)

            # x = torch.where(a == 0, y0, (x1*y0 + x0*y1)/a)

    return x, mapped_coords


def extract_coords(
    x: torch.Tensor, coords: CoordSystem, dim: str = "variable"
) -> tuple[list[torch.Tensor], CoordSystem, np.ndarray]:
    """
    A utility function to extract a dimension from a (x,coords) pair and convert it into a list of tensors, a CoordSystem, and
    the dimension that extract from coords.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    coords : CoordSystem
        Coordinates referring to the dimensions of x.
    dim : str
        Name of the dimension in coords to extract by.

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
