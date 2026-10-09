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

import json
import os
from collections.abc import Hashable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from netCDF4 import Dataset, Variable
from numpy.typing import ArrayLike, DTypeLike

from earth2studio.io.utils import Indexer, plan_arrays, plan_read, plan_write

# CF encodings that round-trip nanosecond labels through Xarray without loss
TIME_UNITS = "nanoseconds since 1970-01-01 00:00:00"
TIME_CALENDAR = "proleptic_gregorian"
LEAD_TIME_UNITS = "nanoseconds"


class NetCDF4Backend:
    """A backend that supports the NetCDF4 format.

    Parameters
    ----------
    file_name : str
        Location of the netCDF4 file to open or create.
    backend_kwargs : dict[str, Any], optional
        Key word arguments for netCDF.Dataset root object, by default
        {"mode": "r+", "diskless": False}

    Note
    ----
    For keyword argument options see: https://unidata.github.io/netcdf4-python/#netCDF4.Dataset

    Note
    ----
    Attribute values netCDF cannot store natively, such as lists and
    dictionaries, are stored as JSON strings and are read back as strings.
    """

    def __init__(
        self,
        file_name: str,
        backend_kwargs: dict[str, Any] = {"mode": "r+", "diskless": False},
    ) -> None:
        backend_kwargs["format"] = "NETCDF4"
        # Set persist to false if diskless is false
        self.root = Dataset(file_name, **backend_kwargs)
        # Fill values are handled explicitly, so masked arrays are never needed
        self.root.set_auto_mask(False)
        self._file_name = os.fspath(file_name)
        self._closed = False
        self._coords: dict[Hashable, xr.Variable] = self._stored_coords()

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
        """Gets number of netCDF4 variables, including coordinates."""
        return (self.root.variables).__len__()

    def __iter__(
        self,
    ) -> Iterator:
        """Return an iterator over netCDF4 variables."""
        return (self.root.variables).__iter__()

    def add_array(self, template: xr.DataArray) -> None:
        """Create the arrays a template describes.

        Parameters
        ----------
        template : xr.DataArray
            Concrete coordinate signature; field values are never read.
        """
        self._check_open()
        plan = plan_arrays(template, self._coords)
        clashes = set(plan.coords).intersection(self._array_names())
        if clashes:
            raise ValueError(
                f"Coordinates collide with arrays: {sorted(map(str, clashes))}"
            )
        dims = tuple(str(dim) for dim in plan.dims)
        for name in plan.names:
            if name in self.root.variables:
                stored = tuple(self.root[name].dimensions)
                if stored != dims:
                    raise ValueError(
                        f"Array '{name}' exists with dimensions {stored}"
                    )

        for dim, size in zip(plan.dims, plan.shape, strict=True):
            self._create_dimension(dim, size)
        for name, variable in plan.coords.items():
            self._create_coord(name, variable)

        auxiliary = " ".join(
            str(name)
            for name, variable in self._coords.items()
            if variable.dims != (name,) and set(variable.dims).issubset(plan.dims)
        )
        for name in plan.names:
            if name in self.root.variables:
                continue
            # netCDF prefills unwritten positions with _FillValue; exact dtypes
            # keep the netCDF default fill rather than a masked sentinel
            fill = plan.fill_value if np.issubdtype(plan.dtype, np.inexact) else None
            array = self.root.createVariable(name, plan.dtype, dims, fill_value=fill)
            attrs = dict(plan.attrs)
            if auxiliary:
                attrs["coordinates"] = auxiliary
            for key, value in attrs.items():
                array.setncattr(str(key), _encode_attr(value))
        self.root.sync()

    def write(self, x: xr.DataArray) -> None:
        """Write a field at the positions its coordinate labels identify.

        Parameters
        ----------
        x : xr.DataArray
            NumPy-, CuPy- or Torch-backed field. It is never modified.
        """
        self._check_open()
        for name, indexers, field in plan_write(x, *self._layout()):
            dims = tuple(self.root[name].dimensions)
            self.root[name][tuple(indexers[dim] for dim in dims)] = (
                field.e2s.as_numpy().values
            )
        self.root.sync()

    def read(
        self,
        selection: xr.DataArray | Mapping[Hashable, ArrayLike],
        device: torch.device | str = "cpu",
        dtype: DTypeLike | None = None,
    ) -> xr.DataArray:
        """Read a field by coordinate label, the inverse of :meth:`write`.

        Parameters
        ----------
        selection : xr.DataArray | Mapping[Hashable, ArrayLike]
            Template or field, or an ordered mapping from every dimension to its
            labels. ``variable`` labels select arrays; without a ``variable``
            dimension, a DataArray's name does. Labels may be any subset, in any
            order; field values are never read.
        device : torch.device | str, optional
            Destination: NumPy-backed on CPU, CuPy-backed on CUDA, by default "cpu"
        dtype : DTypeLike, optional
            Output dtype, by default the stored dtype

        Returns
        -------
        xr.DataArray
            Field with the selection's dimensions and label order, stored
            coordinates and the attributes of the selected arrays.
        """
        self._check_open()
        selection, names, indexers = plan_read(selection, *self._layout())
        fields = [self._read_array(name, indexers) for name in names]
        if "variable" in selection.dims:
            field = xr.concat(
                fields,
                dim=pd.Index(names, name="variable"),
                coords="minimal",
                compat="override",
                combine_attrs="drop_conflicts",
            ).rename(selection.name)
        else:
            field = fields[0]
        field = field.transpose(*selection.dims)
        if dtype is not None:
            field = field.astype(dtype)
        if torch.device(device).type == "cuda":
            return field.e2s.as_cupy(torch.device(device).index)
        return field

    def to_xarray(self) -> xr.Dataset:
        """Open the store as an in-memory :class:`xarray.Dataset`.

        A closed store is reopened from its file, so a diskless store that was
        not persisted cannot be read after :meth:`close`.

        Returns
        -------
        xr.Dataset
            Decoded copy of the store, with dimension and auxiliary coordinates.
        """
        if self._closed:
            return xr.open_dataset(self._file_name).load()
        return xr.open_dataset(xr.backends.NetCDF4DataStore(self.root)).load()

    def flush(self) -> None:
        """Make every earlier write visible to readers of the file."""
        if not self._closed:
            self.root.sync()

    def close(self) -> None:
        """Flush and close the netCDF4 group; later writes raise."""
        if self._closed:
            return
        self._closed = True
        self.root.sync()
        self.root.close()

    def _check_open(self) -> None:
        """Reject operations on a closed store."""
        if self._closed:
            raise RuntimeError(f"NetCDF4Backend {self._file_name} is closed")

    def _array_names(self) -> tuple[str, ...]:
        """Names of the stored data arrays, excluding coordinates."""
        return tuple(
            str(name) for name in self.root.variables if name not in self._coords
        )

    def _layout(
        self,
    ) -> tuple[dict[str, Mapping[Hashable, int]], dict[Hashable, np.ndarray]]:
        """Stored array dimension sizes and dimension labels."""
        arrays: dict[str, Mapping[Hashable, int]] = {
            name: dict(
                zip(self.root[name].dimensions, self.root[name].shape, strict=True)
            )
            for name in self._array_names()
        }
        coords: dict[Hashable, np.ndarray] = {
            name: np.asarray(variable.values)
            for name, variable in self._coords.items()
            if variable.dims == (name,)
        }
        return arrays, coords

    def _stored_coords(self) -> dict[Hashable, xr.Variable]:
        """Read the store's coordinates: dimension variables and CF auxiliaries."""
        names = {name for name in self.root.dimensions if name in self.root.variables}
        for variable in self.root.variables.values():
            if "coordinates" in variable.ncattrs():
                names.update(str(variable.getncattr("coordinates")).split())
        return {
            name: _decode_variable(name, self.root[name])
            for name in sorted(names)
            if name in self.root.variables
        }

    def _create_dimension(self, dim: Hashable, size: int) -> None:
        """Create a fixed-size dimension, or require the stored size to match."""
        if dim not in self.root.dimensions:
            self.root.createDimension(str(dim), size)
        elif self.root.dimensions[dim].size != size:
            raise ValueError(
                f"Dimension '{dim}' has size {self.root.dimensions[dim].size} "
                f"in the store, not {size}"
            )

    def _create_coord(self, name: Hashable, variable: xr.Variable) -> None:
        """Store a dimension or auxiliary coordinate with its attributes."""
        encoded = _encode_variable(variable)
        for dim, size in zip(encoded.dims, encoded.shape, strict=True):
            self._create_dimension(dim, size)
        values = np.asarray(encoded.values)
        dtype = str if values.dtype.kind in "OSU" else values.dtype
        array = self.root.createVariable(
            str(name), dtype, tuple(str(dim) for dim in encoded.dims)
        )
        for key, value in encoded.attrs.items():
            array.setncattr(str(key), _encode_attr(value))
        array[:] = values
        self._coords[name] = variable

    def _read_array(self, name: str, indexers: dict[Hashable, Indexer]) -> xr.DataArray:
        """Read one stored array and its coordinates at the given positions."""
        dims = tuple(self.root[name].dimensions)
        values = self._fetch(name, dims, indexers)
        coords = {
            key: variable.isel({dim: indexers[dim] for dim in variable.dims})
            for key, variable in self._coords.items()
            if set(variable.dims).issubset(dims)
        }
        attrs = {
            key: self.root[name].getncattr(key)
            for key in self.root[name].ncattrs()
            if key not in ("_FillValue", "coordinates")
        }
        return xr.DataArray(values, dims=dims, coords=coords, name=name, attrs=attrs)

    def _fetch(
        self,
        name: str,
        dims: tuple[str, ...],
        indexers: Mapping[Hashable, Indexer],
    ) -> np.ndarray:
        """Select positions from a netCDF4 variable, which indexes in order."""
        selection: list[Indexer] = []
        restore: dict[int, np.ndarray] = {}
        for axis, dim in enumerate(dims):
            indexer = indexers[dim]
            if isinstance(indexer, slice):
                selection.append(indexer)
                continue
            order = np.argsort(indexer, kind="stable")
            selection.append(np.asarray(indexer)[order])
            if np.any(order != np.arange(order.size)):
                restore[axis] = np.argsort(order)
        values = np.asarray(self.root[name][tuple(selection)])
        for axis, positions in restore.items():
            values = np.take(values, positions, axis=axis)
        return values


def _encode_variable(variable: xr.Variable) -> xr.Variable:
    """Encode datetime and timedelta labels as CF nanosecond integers."""
    values = np.asarray(variable.values)
    attrs = dict(variable.attrs)
    if np.issubdtype(values.dtype, np.datetime64):
        values = values.astype("datetime64[ns]").astype("int64")
        attrs |= {"units": TIME_UNITS, "calendar": TIME_CALENDAR}
    elif np.issubdtype(values.dtype, np.timedelta64):
        values = values.astype("timedelta64[ns]").astype("int64")
        attrs |= {"units": LEAD_TIME_UNITS, "dtype": "timedelta64[ns]"}
    return xr.Variable(variable.dims, values, attrs)


def _decode_variable(name: Hashable, variable: Variable) -> xr.Variable:
    """Decode a stored coordinate, normalizing time labels to nanoseconds."""
    attrs = {
        key: variable.getncattr(key)
        for key in variable.ncattrs()
        if key != "_FillValue"
    }
    raw = xr.Variable(variable.dimensions, np.asarray(variable[:]), attrs)
    decoded = xr.conventions.decode_cf_variable(str(name), raw)
    values = np.asarray(decoded.values)
    if np.issubdtype(values.dtype, np.datetime64):
        values = values.astype("datetime64[ns]")
    elif np.issubdtype(values.dtype, np.timedelta64):
        values = values.astype("timedelta64[ns]")
    return xr.Variable(decoded.dims, values, dict(decoded.attrs))


def _encode_attr(value: Any) -> Any:
    """Store an attribute natively, or as a JSON string if netCDF cannot."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "iuf":
            return value
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return value
    return json.dumps(value, default=str)
