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
from cftime import date2num, num2date
from netCDF4 import Dataset, Variable

from earth2studio.utils.coords import convert_multidim_to_singledim
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

units_map = {
    "h": "hour",
    "D": "day",
    "s": "second",
    "m": "minute",
    "Y": "year",
}
rev_units_map = {v: k for k, v in units_map.items()}


class NetCDF4Backend:
    """A backend that supports the NetCDF4 format.

    Parameters
    ----------
    file_name : str
        _description_
    backend_kwargs : dict[str, Any], optional
        Key word arguments for netCDF.Dataset root object, by default
        {"mode": "r+", "diskless": False}

    Note
    ----
    For keyword argument options see: https://unidata.github.io/netcdf4-python/#netCDF4.Dataset
    """

    def __init__(
        self,
        file_name: str,
        backend_kwargs: dict[str, Any] = {"mode": "r+", "diskless": False},
    ) -> None:
        backend_kwargs["format"] = "NETCDF4"
        # Set persist to false if diskless is false
        self.root = Dataset(file_name, **backend_kwargs)

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
                    [
                        np.timedelta64(n, rev_units_map[self.root[dim].units])
                        for n in nums
                    ]
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
                var.calendar = "proleptic_gregorian"
                data = timearray_to_datetime(data)
                data = date2num(data, units=var.units, calendar=var.calendar)

            elif np.issubdtype(data.dtype, np.timedelta64):
                units, _ = np.datetime_data(data[0])
                out_units = units_map[units]
                data = data / np.timedelta64(1, units)
                var = self.root.createVariable(name, "i8", (name,))
                var.units = out_units
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

        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        for c, v in adjusted_coords.items():
            if c not in self.coords:
                self.add_dimension(c, v.shape, v)

        self.coords = self.coords | adjusted_coords

        # Add multidimensional coords
        for k in mapping:
            if k not in self.root.variables:
                dtype = coords[k].dtype
                self.root.createVariable(k, dtype, list(mapping[k]))
                self.root[k][:] = coords[k]

        for name, di in zip(array_name, data):
            if name in self.root.variables:
                raise RuntimeError(
                    f"{name} is already in NetCDF Store. "
                    + "NetCDF does not allow variables to be redefined. "
                    + r"To overwrite entire NetCDF, create object with backend_kwargs=\{'mode': 'w'\}"
                )

            di = di.cpu().numpy() if di is not None else None
            dtype = di.dtype if di is not None else "float32"
            self.root.createVariable(name, dtype, list(adjusted_coords))

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

        # Reduce complex coordinates, if any multidimension coordinates exist
        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        for dim in adjusted_coords:
            if dim not in self.coords:
                raise AssertionError("Coordinate dimension not in NetCDF store.")

        # Check to see if multidimensions are passed in full, otherwise error
        for key in mapping:
            if key not in self.root.variables:
                raise AssertionError(
                    f"Multidimension coordinate {key} not in NetCDF4 store."
                )

            if coords[key].shape != self.root[key].shape:
                raise AssertionError(
                    "Currently writing data with multidimension arrays is only supported when"
                    + "the multidimension coordinates are passed in full."
                )

        for xi, name in zip(x, array_name):
            if name not in self.root.variables:
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

        # Reduce complex coordinates, if any multidimension coordinates exist
        adjusted_coords, mapping = convert_multidim_to_singledim(coords)

        for dim in adjusted_coords:
            if dim not in self.coords:
                raise AssertionError(
                    f"Coordinate dimension {dim} not in NetCDF4 store."
                )

        # Check to see if multidimensions are passed in full, otherwise error
        for key in mapping:
            if key not in self.root.variables:
                raise AssertionError(
                    f"Multidimension coordinate {key} not in NetCDF4 store."
                )

            if coords[key].shape != self.root[key].shape:
                raise AssertionError(
                    "Currently reading data with multidimension arrays is only supported when"
                    + "the multidimension coordinates are passed in full."
                )

        x = self.root[array_name][
            tuple(
                [
                    np.where(np.isin(self.coords[dim], value))[0]
                    for dim, value in adjusted_coords.items()
                ]
            )
        ]

        return torch.as_tensor(x, device=device), coords

    def close(
        self,
    ) -> None:
        """Close NetCDF4 group."""
        self.root.close()
