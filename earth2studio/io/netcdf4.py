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

import numpy as np
import torch
from cftime import date2num, num2date
from netCDF4 import Dataset, Variable

from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem


class NetCDF4Backend:
    """A backend that supports the NetCDF4 format.

    Parameters
    ----------
    file_name : str,
        File name to provide for creating the netcdf4 store.
    diskless : bool, optional
        Whether to store the Dataset in memory buffer
        Default value is False.
    persist : bool, optional
        Whether to save in-memory diskless buffer to disk upon close.
        Default value is False.
    """

    def __init__(
        self,
        file_name: str,
        diskless: bool = False,
        persist: bool = False,
    ) -> None:

        # Set persist to false if diskless is false
        self.root = Dataset(
            file_name,
            "r+",
            format="NETCDF4",
            diskless=diskless,
            persist=persist if diskless else False,
        )

        self.coords: CoordSystem = OrderedDict({})
        for dim in self.root.dimensions:
            if dim == "time":
                nums = self.root[dim]
                dates = num2date(
                    nums, units=self.root[dim].units, calendar=self.root[dim].calendar
                )
                self.coords[dim] = np.array(
                    [np.datetime64(d.isoformat()) for d in dates]
                )
            elif dim == "lead_time":
                nums = self.root[dim][:]
                self.coords[dim] = np.array(
                    [np.timedelta64(n, self.root[dim].units) for n in nums]
                )
            else:
                self.coords[dim] = self.root[dim][:]

    def __contains__(self, item: str) -> bool:
        """Checks if item in netCDF4 variables.

        Parameters
        ----------
        item : str
        """
        return (self.root.variables).__contains__(item)

    def __getitem__(self, item: str) -> Variable:
        """Gets variable in netCDF4 group.

        Parameters
        ----------
        item : str
        """
        return (self.root.variables).__getitem__(item)

    def __len__(
        self,
    ) -> int:
        """Gets number of netCDF4 variables."""
        return (self.root.variables).__len__()

    def __iter__(
        self,
    ) -> Iterator:
        """Return an iterator over netCDF4 variables."""
        return (self.root.variables).__iter__()

    def add_dimension(self, name: str, shape: tuple | list, data: np.ndarray) -> None:
        """Add a dimension to the existing netCDF4 group.

        Note that netCDF4 dimensions must be 1-dimensional.

        Parameters
        ----------
        name : str
            name of dimension
        shape : tuple
            Shape of dimension
        data : np.ndarray
            Data to save to dimension variable
        """

        if len(shape) == 1:
            self.root.createDimension(name, shape[0])

            # Check if data is datetime64
            if np.issubdtype(data.dtype, np.datetime64):

                var = self.root.createVariable(name, "f8", (name,))
                var.units = "hours since 0001-01-01 00:00:00.0"
                var.calendar = "gregorian"
                data = timearray_to_datetime(data)
                data = date2num(data, units=var.units, calendar=var.calendar)

            elif np.issubdtype(data.dtype, np.timedelta64):
                units, _ = np.datetime_data(data[0])
                data = data / np.timedelta64(1, units)
                var = self.root.createVariable(name, "i8", (name,))
                var.units = units
                var.calendar = "gregorian"
            else:
                var = self.root.createVariable(name, data.dtype, (name,))

            var[:] = data

        else:
            raise ValueError("Error, only 1-dimension coordinates are supported.")

    def add_array(
        self,
        coords: CoordSystem,
        array_name: str | list[str],
        data: torch.Tensor | list[torch.Tensor] = None,
    ) -> None:
        """Add an array to the existing netcdf Dataset.

        Parameters
        ----------
        coords: CoordSystem
            Ordered dict of coordinate information.
        array_name : str
            Name to add to netcdf Dataset for the new array. Can optionally
            be a list of array_names.
        data: torch.Tensor | list[torch.Tensor], optional
            Optional data to initialize the array with. If None, then
            the netcdf array is initialized with zeros.
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

        for c, v in coords.items():
            if c not in self.coords:
                self.add_dimension(c, v.shape, v)

        self.coords = self.coords | coords

        for name, di in zip(array_name, data):
            if name in self.root.variables:
                raise AssertionError(f"Warning! {name} is already in NetCDF Store.")

            di = di.cpu().numpy() if di is not None else None
            dtype = di.dtype if di is not None else "float32"
            self.root.createVariable(name, dtype, list(coords))

            if di is not None:
                self.root[name][:] = di

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """
        Write data to the current netCDF group using the passed array_name.

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Tensor(s) to be written to netCDF group.
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
            if dim not in self.coords:
                raise AssertionError("Coordinate dimension not in NetCDF store.")

        for xi, name in zip(x, array_name):
            if name not in self.root.variables:
                self.add_array(coords, array_name, data=xi)

            else:
                # Get indices as list of arrays and set torch tensor
                self.root[name][
                    tuple(
                        [
                            np.where(np.in1d(self.coords[dim], value))[0]
                            for dim, value in coords.items()
                        ]
                    )
                ] = xi.to("cpu", non_blocking=True).numpy()

    def read(
        self, coords: CoordSystem, array_name: str, device: torch.device = "cpu"
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Read data from the current netcdf store using the passed array_name.

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
            tuple(
                [
                    np.where(np.in1d(self.coords[dim], value))[0]
                    for dim, value in coords.items()
                ]
            )
        ]

        return torch.as_tensor(x, device=device), coords

    def close(
        self,
    ) -> None:
        """Close NetCDF4 group."""
        self.root.close()
