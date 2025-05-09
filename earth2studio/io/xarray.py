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
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.coords import convert_multidim_to_singledim
from earth2studio.utils.type import CoordSystem


class XarrayBackend:
    """An xarray backed IO object.

    Parameters
    ----------
    coords : CoordSystem
        Coordinates to initialize the xarray Dataset with. Must be a
        complete set of coordinates, i.e., the Dataset object should
        be viewed as (mostly) immutable with the given set of coordinates.
    xr_kwargs : dict
        Optional keyword arguments to pass to the xarray.Dataset constructor.

    """

    def __init__(self, coords: CoordSystem = OrderedDict({}), **xr_kwargs: Any) -> None:
        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
        if not mapping:
            self.root = xr.Dataset(
                data_vars=data_vars, coords=adjusted_coords, **xr_kwargs
            )
        else:
            for k in mapping:
                data_vars[k] = (mapping[k], coords[k])

            self.root = xr.Dataset(
                data_vars=data_vars, coords=adjusted_coords, **xr_kwargs
            )

        self.coords = adjusted_coords

    def __contains__(self, item: str) -> bool:
        """Checks if item in xarray Dataset.

        Parameters
        ----------
        item : str
        """
        return self.root.__contains__(item)

    def __getitem__(self, item: str) -> torch.Tensor | np.ndarray:
        """Gets item in xarray Dataset.

        Parameters
        ----------
        item : str
        """
        return self.root.__getitem__(item)

    def __len__(
        self,
    ) -> int:
        """Gets number of variables in xarray Dataset."""
        return self.root.__len__() + self.coords.__len__()

    def __iter__(
        self,
    ) -> Iterator:
        """Return an iterator over xarray DataSet variable names."""
        return self.root.__iter__()

    def add_array(
        self,
        coords: CoordSystem,
        array_name: str | list[str],
        data: torch.Tensor | list[torch.Tensor] = None,
        **xr_kwargs: Any,
    ) -> None:
        """Add an array to the existing xarray Dataset.

        Parameters
        ----------
        coords: CoordSystem
            Ordered dict of coordinate information.
        array_name : str
            Name to add to xarray Dataset for the new array.
        data: torch.Tensor | list[torch.Tensor], optional
            Optional data to initialize the array with. If None, then
            the array is NaN initialized (xarray default).
            Can also pass a list of tensors, which must match in length to the
            list of array_names passed. If a list of tensors is passed, it is assumed
            that each tensor share `coords`.
        xr_kwargs: Any
            Optional keyword arguments passed to xr.DataArray constructor.
        """
        if isinstance(array_name, str):
            array_name = [array_name]
        if isinstance(data, torch.Tensor):
            data = [data]
        elif data is None:
            data = [None] * len(array_name)

        if not (len(data) == len(array_name)):
            raise ValueError(
                f"The number of input tensors and array names must be the same but got {len(data)} and {len(array_name)}."
            )

        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        self.coords = self.coords | adjusted_coords

        for k in mapping:
            if k not in self.root:
                self.root[k] = xr.DataArray(
                    data=coords[k],
                    dims=mapping[k],
                    coords={adjusted_coords[ki] for ki in mapping[k]},
                    **xr_kwargs,
                )

        for name, di in zip(array_name, data):
            if name in self.root:
                raise AssertionError(f"Warning! {name} is already in xarray Dataset.")

            if di is not None:
                self.root[name] = xr.DataArray(
                    data=di.cpu().numpy(),
                    coords=adjusted_coords,
                    dims=list(adjusted_coords),
                    **xr_kwargs,
                )
            else:
                self.root[name] = xr.DataArray(
                    coords=adjusted_coords, dims=list(adjusted_coords), **xr_kwargs
                )

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """
        Write data to the current xarray Dataset using the passed array_name.

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Tensor(s) to be written to xarray dataset.
        coords : OrderedDict
            Coordinates of the passed data.
        array_name : str | list[str]
            Name(s) of the array(s) that will be written to.
        """

        # Input checking
        if isinstance(x, torch.Tensor):
            x = [x]
        if isinstance(array_name, str):
            array_name = [array_name]
        if not (len(x) == len(array_name)):
            raise ValueError(
                f"The number of input tensors and array names must be the same but got {len(x)} and {len(array_name)}."
            )

        # Reduce complex coordinates, if any multidimension coordinates exist
        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        for dim in adjusted_coords:
            if dim not in self.root:
                raise AssertionError("Coordinate dimension not in xarray dataset.")

        # Check to see if multidimensions are passed in full, otherwise error
        for key in mapping:
            if key not in self.root:
                raise AssertionError(
                    f"Multidimension coordinate {key} not in xarray store."
                )

            if coords[key].shape != self.root[key].shape:
                raise AssertionError(
                    "Currently writing data with multidimension arrays is only supported when"
                    + "the multidimension coordinates are passed in full."
                )

        for xi, name in zip(x, array_name):
            if name not in self.root:
                self.add_array(adjusted_coords, array_name, data=xi)

            else:
                # Get indices as list of arrays and set torch tensor
                self.root[name][
                    tuple(
                        [
                            np.where(np.isin(self.coords[dim], value))[0]
                            for dim, value in adjusted_coords.items()
                        ]
                    )
                ] = xi.to("cpu").numpy()

    def read(
        self,
        coords: CoordSystem,
        array_name: str,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Read data from the current xarray dataset using the passed array_name.

        Parameters
        ----------
        coords : OrderedDict
            Coordinates of the data to be read.
        array_name : str | list[str]
            Name(s) of the array(s) to read from.
        device : torch.device
            device to place the read data from, by default 'cpu'
        """

        # Reduce complex coordinates, if any multidimension coordinates exist
        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        for dim in adjusted_coords:
            if dim not in self.root:
                raise AssertionError(f"Coordinate dimension {dim} not in xarray store.")

        # Check to see if multidimensions are passed in full, otherwise error
        for key in mapping:
            if key not in self.root:
                raise AssertionError(
                    f"Multidimension coordinate {key} not in xarray store."
                )

            if coords[key].shape != self.root[key].shape:
                raise AssertionError(
                    "Currently reading data with multidimension arrays is only supported when"
                    + "the multidimension coordinates are passed in full."
                )

        x = self.root[array_name].values[
            np.ix_(
                *[
                    np.where(np.isin(self.coords[dim], value))[0]
                    for dim, value in adjusted_coords.items()
                ]
            )
        ]

        return torch.as_tensor(x, dtype=dtype, device=device), coords
