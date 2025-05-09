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
import xarray

from earth2studio.utils.coords import convert_multidim_to_singledim
from earth2studio.utils.type import CoordSystem


class KVBackend:
    """A key-value (dict) backend.

    Parameters
    ----------
    device: str = 'cpu'
        Device to keep array tensors.
    """

    def __init__(self, device: str = "cpu") -> None:

        self.device = device
        self.root: dict[str, torch.Tensor] = {}
        self.coords: CoordSystem = OrderedDict({})
        self.dims: dict[str, list[str]] = {}

    def __contains__(self, item: str) -> bool:
        """Checks if item in KV store.

        Parameters
        ----------
        item : str
        """
        return (self.root | self.coords).__contains__(item)

    def __getitem__(self, item: str) -> torch.Tensor | np.ndarray:
        """Gets item in KV store.

        Parameters
        ----------
        item : str
        """
        return (self.root | self.coords).__getitem__(item)

    def __len__(
        self,
    ) -> int:
        """Gets length of Dict Group."""
        return self.root.__len__() + self.coords.__len__()

    def __iter__(
        self,
    ) -> Iterator:
        """Return an iterator over KV store member names."""
        return (self.root | self.coords).__iter__()

    def add_array(
        self,
        coords: CoordSystem,
        array_name: str | list[str],
        data: torch.Tensor | list[torch.Tensor] = None,
    ) -> None:
        """Add an array to the existing KV store.

        Parameters
        ----------
        coords: CoordSystem
            Ordered dict of coordinate information.
        array_name : str
            Name to add to kv store for the new array. Can optionally
            be a list of array_names.
        data: torch.Tensor | list[torch.Tensor], optional
            Optional data to initialize the array with. If None, then
            the KV store array is initialized with torch.float32 zeros.
            Can also pass a list of tensors, which must match in length to the
            list of array_names passed. If a list of tensors is passed, it is assumed
            that each tensor share `coords`.
        """
        # Input checking

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

        # Add any multidim coordinates that were expelled above
        for k in mapping:
            if k not in self.root:
                values = coords[k]
                self.dims[k] = mapping[k]
                self.root[k] = torch.as_tensor(values, device="cpu")

        for name, di in zip(array_name, data):
            if name in self.root:
                raise AssertionError(f"Warning! {name} is already in KV Store.")

            self.dims[name] = list(adjusted_coords)

            if di is not None:
                self.root[name] = di.to(self.device)
            else:
                shape = [len(v) for v in adjusted_coords.values()]
                self.root[name] = torch.zeros(
                    shape, dtype=torch.float32, device=self.device
                )

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """
        Write data to the current KV store using the passed array_name.

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Tensor(s) to be written to KV store.
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
            if dim not in self.coords:
                raise AssertionError("Coordinate dimension not in KV store.")

        # Check to see if multidimensions are passed in full, otherwise error
        for key in mapping:
            if key not in self.root:
                raise AssertionError(
                    f"Multidimension coordinate {key} not in KV store."
                )

            if coords[key].shape != self.root[key].shape:
                raise AssertionError(
                    "Currently writing data with multidimension arrays is only supported when"
                    + "the multidimension coordinates are passed in full."
                )

        for xi, name in zip(x, array_name):
            if name not in self.root:
                self.add_array(coords, array_name, data=xi.to(self.device))

            else:
                # Get indices as list of arrays and set torch tensor
                self.root[name][
                    np.ix_(
                        *[
                            np.where(np.isin(self.coords[dim], value))[0]
                            for dim, value in adjusted_coords.items()
                        ]
                    )
                ] = xi.to(self.device)

    def to_xarray(self, **xr_kwargs: Any) -> xarray.Dataset:
        """
        Returns an xarray Dataset corresponding to the variables
        and coordinates in the KV Store.

        Parameters
        ----------
        xr_kwargs : dict[str, Any]
            Optional keyward arguments to pass to xarray Dataset constructor.
        Returns
        -------
        xarray.Dataset
        """

        return xarray.Dataset(
            data_vars={
                array_name: (self.dims[array_name], values.cpu().numpy())
                for array_name, values in self.root.items()
            },
            coords={
                coord_name: ([coord_name], values)
                for coord_name, values in self.coords.items()
            },
            **xr_kwargs,
        )

    def read(
        self, coords: CoordSystem, array_name: str, device: torch.device = "cpu"
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Read data from the current KV store using the passed array_name.

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
            if dim not in self.coords:
                raise AssertionError(f"Coordinate dimension {dim} not in KV store.")

        # Check to see if multidimensions are passed in full, otherwise error
        for key in mapping:
            if key not in self.root:
                raise AssertionError(
                    f"Multidimension coordinate {key} not in KV store."
                )

            if coords[key].shape != self.root[key].shape:
                raise AssertionError(
                    "Currently reading data with multidimension arrays is only supported when"
                    + "the multidimension coordinates are passed in full."
                )

        x = self.root[array_name][
            np.ix_(
                *[
                    np.where(np.isin(self.coords[dim], value))[0]
                    for dim, value in adjusted_coords.items()
                ]
            )
        ]

        return x.to(device), coords
