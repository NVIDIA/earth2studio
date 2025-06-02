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
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
import zarr
from loguru import logger

# Dealing with zarr 3.0 API breaks and type checking
try:
    zarr_version = version("zarr")
    zarr_major_version = int(zarr_version.split(".")[0])
except Exception:
    zarr_major_version = 2

if TYPE_CHECKING:
    from typing import TypeAlias

    from zarr.core import Array
    from zarr.core.array import Array as Array3

    ZarrArray: TypeAlias = Union[Array, Array3]
else:
    if zarr_major_version >= 3:
        ZarrArray = zarr.core.array.Array
    else:
        ZarrArray = zarr.core.Array

from earth2studio.utils.coords import convert_multidim_to_singledim
from earth2studio.utils.type import CoordSystem


class ZarrBackend:
    """A backend that supports the zarr format.

    Parameters
    ----------
    file_name : str, optional
        Optional name to provide to the zarr store, if provided then this function
        will create a directory store with this file name. If none, will create a
        in-memory store., by default None
    chunks : dict[str, int], optional
        An ordered dict of chunks to use with the data passed through data/coords, by
        default {}
    backend_kwargs : dict[str, Any], optional
        Key word arguments for zarr.Group root object, by default {"overwrite": False}

    Note
    ----
    For keyword argument options see: https://zarr.readthedocs.io/en/stable/api/hierarchy.html
    """

    # sphinx - io zarr start
    def __init__(
        self,
        file_name: str = None,
        chunks: dict[str, int] = {},
        backend_kwargs: dict[str, Any] = {"overwrite": False},
    ) -> None:

        if file_name is None:
            self.store = zarr.storage.MemoryStore()
        else:
            if zarr_major_version >= 3:
                self.store = zarr.storage.LocalStore(file_name)
            else:
                self.store = zarr.storage.DirectoryStore(file_name)

        self.root = zarr.group(self.store, **backend_kwargs)

        # Read data from file, if available
        self.coords: CoordSystem = OrderedDict({})
        self.chunks = chunks.copy()
        for array in self.root:
            if zarr_major_version >= 3:
                # https://github.com/pydata/xarray/pull/9669
                dims = self.root[array].metadata.dimension_names
            else:
                dims = self.root[array].attrs["_ARRAY_DIMENSIONS"]
            for dim in dims:
                if dim not in self.coords:
                    self.coords[dim] = self.root[dim][:]

        for array in self.root:
            if array not in self.coords:
                if zarr_major_version >= 3:
                    # https://github.com/pydata/xarray/pull/9669
                    dims = self.root[array].metadata.dimension_names
                else:
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

    def __getitem__(self, item: str) -> "ZarrArray":
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

    # sphinx - io zarr end
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

        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        for dim, values in adjusted_coords.items():
            if dim not in self.coords:
                if zarr_major_version >= 3:
                    # Dates types not supported in zarr 3.0 at the moment
                    # https://github.com/zarr-developers/zarr-python/issues/2616
                    # TODO: Remove once fixed
                    if np.issubdtype(values.dtype, np.datetime64):
                        logger.warning(
                            "Datetime64 not supported in zarr 3.0, converting to int64 nanoseconds since epoch"
                        )
                        values = values.astype("datetime64[ns]").astype("int64")

                    if np.issubdtype(values.dtype, np.timedelta64):
                        logger.warning(
                            "Timedelta64 not supported in zarr 3.0, converting to int64 nanoseconds since epoch"
                        )
                        values = values.astype("timedelta64[ns]").astype("int64")

                    self.root.create_array(
                        dim,
                        shape=values.shape,
                        chunks=values.shape,
                        dtype=values.dtype,
                        dimension_names=[dim],
                        **kwargs,
                    )
                else:
                    self.root.create_dataset(
                        dim,
                        shape=values.shape,
                        chunks=values.shape,
                        dtype=values.dtype,
                        **kwargs,
                    )
                self.root[dim][:] = values
                if zarr_major_version < 3:
                    # https://github.com/pydata/xarray/pull/9669
                    self.root[dim].attrs["_ARRAY_DIMENSIONS"] = [dim]

        # Add any multidim coordinates that were expelled above
        for k in mapping:
            if k not in self.root:
                values = coords[k]
                if zarr_major_version >= 3:
                    # Dates types not supported in zarr 3.0 at the moment
                    # https://github.com/zarr-developers/zarr-python/issues/2616
                    # TODO: Remove once fixed
                    if np.issubdtype(values.dtype, np.datetime64):
                        logger.warning(
                            "Datetime64 not supported in zarr 3.0, converting to int64 nanoseconds since epoch"
                        )
                        values = values.astype("datetime64[ns]").astype("int64")

                    if np.issubdtype(values.dtype, np.timedelta64):
                        logger.warning(
                            "Timedelta64 not supported in zarr 3.0, converting to int64 nanoseconds since epoch"
                        )
                        values = values.astype("timedelta64[ns]").astype("int64")

                    self.root.create_array(
                        k,
                        shape=values.shape,
                        chunks=values.shape,
                        dtype=values.dtype,
                        dimension_names=mapping[k],
                        **kwargs,
                    )
                else:
                    self.root.create_dataset(
                        k,
                        shape=values.shape,
                        chunks=values.shape,
                        dtype=values.dtype,
                        **kwargs,
                    )
                self.root[k][:] = values
                if zarr_major_version < 3:
                    # https://github.com/pydata/xarray/pull/9669
                    self.root[k].attrs["_ARRAY_DIMENSIONS"] = [mapping[k]]

        self.coords = self.coords | adjusted_coords

        shape = [len(v) for v in adjusted_coords.values()]
        chunks = [
            self.chunks.get(dim, len(adjusted_coords[dim])) for dim in adjusted_coords
        ]

        for name, di in zip(array_name, data):
            if name in self.root and not kwargs.get("overwrite", False):
                raise RuntimeError(
                    f"{name} is already in Zarr Store. "
                    + "To overwrite Zarr array pass overwrite=True to this function"
                )

            di = di.cpu().numpy() if di is not None else None
            dtype = di.dtype if di is not None else "float32"
            if zarr_major_version >= 3:
                self.root.create_array(
                    name,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    dimension_names=list(adjusted_coords),
                    **kwargs,
                )
            else:
                self.root.create_dataset(
                    name, shape=shape, chunks=chunks, dtype=dtype, **kwargs
                )
            if di is not None:
                self.root[name][:] = di

            if zarr_major_version < 3:
                # https://github.com/pydata/xarray/pull/9669
                self.root[name].attrs["_ARRAY_DIMENSIONS"] = list(adjusted_coords)

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

        # Reduce complex coordinates, if any multidimension coordinates exist
        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        for dim in adjusted_coords:
            if dim not in self.root:
                raise AssertionError(f"Coordinate dimension {dim} not in zarr store.")

        # Check to see if multidimensions are passed in full, otherwise error
        for key in mapping:
            if key not in self.root:
                raise AssertionError(
                    f"Multidimension coordinate {key} not in zarr store."
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
                    np.ix_(
                        *[
                            np.where(np.isin(self.coords[dim], value))[0]
                            for dim, value in adjusted_coords.items()
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

        # Reduce complex coordinates, if any multidimension coordinates exist
        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        for dim in adjusted_coords:
            if dim not in self.root:
                raise AssertionError(f"Coordinate dimension {dim} not in zarr store.")

        # Check to see if multidimensions are passed in full, otherwise error
        for key in mapping:
            if key not in self.root:
                raise AssertionError(
                    f"Multidimension coordinate {key} not in zarr store."
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

        return torch.as_tensor(x, device=device), coords
