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

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
import zarr

from earth2studio.utils.type import CoordSystem


class ZarrBackend:
    """A backend that supports the zarr format.

    Parameters
    ----------
    file_name : str, optional
        Optional name to provide to the zarr store, if provided then this function
        will create a directory store with this file name. If not, will create a memory store.
    chunks : dict[str, int], optional
        An ordered dict of chunks to use with the data passed through data/coords.
    """

    def __init__(
        self,
        file_name: str = None,
        chunks: dict[str, int] = {},
    ) -> None:

        if file_name is None:
            self.store = zarr.storage.MemoryStore()
        else:
            self.store = zarr.storage.DirectoryStore(file_name)

        self.root = zarr.group(self.store)

        # Read data from file, if available
        self.coords: CoordSystem = OrderedDict({})
        self.chunks = chunks.copy()
        for array in self.root:
            dims = self.root[array].attrs["_ARRAY_DIMENSIONS"]
            for dim in dims:
                if dim not in self.coords:
                    self.coords[dim] = self.root[dim]

        for array in self.root:
            if array not in self.coords:
                dims = self.root[array].attrs["_ARRAY_DIMENSIONS"]
                for c, d in zip(self.root[array].chunks, dims):
                    self.chunks[d] = c

    def __contains__(self, item: str) -> bool:
        """Checks if item in Zarr Group.

        Parameters
        ----------
        item : str
        """
        return self.root.__contains__(item)

    def __getitem__(self, item: str) -> zarr.core.Array:
        """Gets item in Zarr Group.

        Parameters
        ----------
        item : str
        """
        return self.root.__getitem__(item)

    def __len__(
        self,
    ) -> int:
        """Gets length of Zarr Group."""
        return self.root.__len__()

    def __iter__(
        self,
    ) -> Iterator:
        """Return an iterator over Zarr Group member names."""
        return self.root.__iter__()

    def add_array(
        self,
        coords: CoordSystem,
        array_name: str | list[str],
        data: torch.Tensor | list[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        """Add an array to the existing zarr group.

        Parameters
        ----------
        coords: CoordSystem
            Ordered dict of coordinate information.
        array_name : str
            Name to add to zarr group for the new array.
        data: torch.Tensor | list[torch.Tensor], optional
            Optional data to initialize the array with. If None, then
            the array is NaN initialized (zarr default).
            Can also pass a list of tensors, which must match in length to the
            list of array_names passed. If a list of tensors is passed, it is assumed
            that each tensor share `coords`.
        kwargs: Any
            Optional keyword arguments passed to zarr dataset constructor.
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

        # Set fill value to None if not already given
        if "fill_value" not in kwargs:
            kwargs["fill_value"] = None

        for dim, values in coords.items():
            if dim not in self.coords:
                self.root.create_dataset(
                    dim,
                    shape=values.shape,
                    chunks=values.shape,
                    dtype=values.dtype,
                    **kwargs,
                )
                self.root[dim][:] = values
                self.root[dim].attrs["_ARRAY_DIMENSIONS"] = [dim]

        self.coords = self.coords | coords

        shape = [len(v) for v in coords.values()]
        chunks = [self.chunks.get(dim, len(coords[dim])) for dim in coords]

        for name, di in zip(array_name, data):
            if name in self.root and not kwargs.get("overwrite", False):
                raise AssertionError(f"Warning! {name} is already in zarr store.")

            di = di.cpu().numpy() if di is not None else None
            dtype = di.dtype if di is not None else "float32"
            self.root.create_dataset(
                name, shape=shape, chunks=chunks, dtype=dtype, **kwargs
            )
            if di is not None:
                self.root[name][:] = di

            self.root[name].attrs["_ARRAY_DIMENSIONS"] = list(coords)

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """
        Write data to the current zarr group using the passed array_name.

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Tensor(s) to be written to zarr store.
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

        for dim in coords:
            if dim not in self.root:
                raise AssertionError("Coordinate dimension not in zarr store.")

        for xi, name in zip(x, array_name):
            if name not in self.root:
                self.add_array(coords, array_name, data=xi)

            else:
                # Get indices as list of arrays and set torch tensor
                self.root[name][
                    np.ix_(
                        *[
                            np.where(np.in1d(self.coords[dim], value))[0]
                            for dim, value in coords.items()
                        ]
                    )
                ] = xi.to("cpu", non_blocking=False).numpy()

    def read(
        self, coords: CoordSystem, array_name: str, device: torch.device = "cpu"
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Read data from the current zarr group using the passed array_name.

        Parameters
        ----------
        coords : OrderedDict
            Coordinates of the data to be read.
        array_name : str | list[str]
            Name(s) of the array(s) to read from.
        device : torch.device
            device to place the read data from, by default 'cpu'
        """

        x = self.root[array_name][
            np.ix_(
                *[
                    np.where(np.in1d(self.coords[dim], value))[0]
                    for dim, value in coords.items()
                ]
            )
        ]

        return torch.as_tensor(x, device=device), coords
