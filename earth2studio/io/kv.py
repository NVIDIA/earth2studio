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

from collections import OrderedDict
from collections.abc import Hashable, Iterator, Mapping
from typing import Any

import numpy as np
import torch
import xarray as xr
from numpy.typing import ArrayLike, DTypeLike

from earth2studio.io.utils import Indexer, plan_arrays, plan_read, plan_write
from earth2studio.utils.cupy import from_torch


class KVBackend:
    """A key-value (dict) backend holding Torch tensors.

    Stored arrays are Torch tensors on :attr:`device`, keyed by name in
    :attr:`root`. Dimension and auxiliary coordinates are kept in :attr:`coords`
    and the store is available as an :class:`xarray.Dataset` from
    :meth:`to_xarray`.

    Parameters
    ----------
    device : str, optional
        Device to keep array tensors on, by default "cpu"
    """

    def __init__(self, device: str = "cpu") -> None:

        self.device = device
        self.root: dict[str, torch.Tensor] = {}
        self.coords: dict[Hashable, xr.Variable] = {}
        self.dims: dict[str, tuple[Hashable, ...]] = {}
        self.attrs: dict[str, dict[Hashable, Any]] = {}
        self._closed = False

    def __contains__(self, item: str) -> bool:
        """Checks if item is an array or coordinate of the KV store.

        Parameters
        ----------
        item : str
        """
        return item in self.root or item in self.coords

    def __getitem__(self, item: str) -> torch.Tensor | xr.Variable:
        """Gets an array tensor or a coordinate of the KV store.

        Parameters
        ----------
        item : str
        """
        if item in self.root:
            return self.root[item]
        return self.coords[item]

    def __len__(
        self,
    ) -> int:
        """Gets the number of stored arrays."""
        return len(self.root)

    def __iter__(
        self,
    ) -> Iterator[str]:
        """Return an iterator over stored array names."""
        return iter(self.root)

    def add_array(self, template: xr.DataArray) -> None:
        """Create the arrays a template describes.

        Parameters
        ----------
        template : xr.DataArray
            Concrete coordinate signature; field values are never read.
        """
        plan = plan_arrays(template, self.coords)
        clashes = set(plan.coords).intersection(self.root)
        if clashes:
            raise ValueError(
                f"Coordinates collide with arrays: {sorted(map(str, clashes))}"
            )
        for name in plan.names:
            if name not in self.root:
                continue
            if self.dims[name] != plan.dims:
                raise ValueError(
                    f"Array '{name}' exists with dimensions {self.dims[name]}"
                )
            if tuple(self.root[name].shape) != plan.shape:
                raise ValueError(
                    f"Array '{name}' exists with shape {tuple(self.root[name].shape)}"
                )
        dtype = _torch_dtype(plan.dtype)
        self.coords.update(plan.coords)
        for name in plan.names:
            if name in self.root:
                continue
            self.dims[name] = plan.dims
            self.attrs[name] = dict(plan.attrs)
            self.root[name] = torch.full(
                plan.shape, plan.fill_value, dtype=dtype, device=self.device
            )

    def write(self, x: xr.DataArray) -> None:
        """Write a field at the positions its coordinate labels identify.

        Parameters
        ----------
        x : xr.DataArray
            NumPy-, CuPy- or Torch-backed field. It is never modified; its data
            is copied onto the store's device.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed KVBackend")
        for name, indexers, field in plan_write(x, *self._layout()):
            array = self.root[name]
            tensor, _ = field.e2s.to_torch()
            positions = self._positions(indexers, name)
            array[np.ix_(*[positions[dim] for dim in self.dims[name]])] = tensor.to(
                device=array.device, dtype=array.dtype
            )

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
            coordinates and the attributes shared by the selected arrays.
        """
        selection, names, indexers = plan_read(selection, *self._layout())
        dims = self.dims[names[0]]
        positions = self._positions(indexers, names[0])
        index: tuple[np.ndarray, ...] = np.ix_(*[positions[dim] for dim in dims])
        tensors = [self.root[name][index] for name in names]
        # Variables index orthogonally, so each dimension selects independently
        coords: dict[Hashable, Any] = {
            key: variable[tuple(positions[dim] for dim in variable.dims)]
            for key, variable in self.coords.items()
            if set(variable.dims).issubset(dims)
        }
        if "variable" in selection.dims:
            tensor = torch.stack(tensors)
            dims = ("variable", *dims)
            coords["variable"] = np.asarray(names)
            name = selection.name
        else:
            tensor = tensors[0]
            name = names[0]
        if dtype is not None:
            tensor = tensor.to(dtype=_torch_dtype(np.dtype(dtype)))
        tensor = tensor.to(device=torch.device(device))
        field = xr.DataArray(
            _payload(tensor),
            dims=dims,
            coords=coords,
            name=name,
            attrs=_shared_attrs([self.attrs[key] for key in names]),
        )
        return field.transpose(*selection.dims)

    def to_xarray(self, **xr_kwargs: Any) -> xr.Dataset:
        """Return the store as an in-memory, NumPy-backed dataset.

        Parameters
        ----------
        xr_kwargs : dict[str, Any]
            Optional keyword arguments passed to the :class:`xarray.Dataset`
            constructor, such as global ``attrs``.

        Returns
        -------
        xr.Dataset
            Dataset with one variable per stored array, the store's coordinates
            and the attributes :meth:`add_array` persisted.
        """
        return xr.Dataset(
            data_vars={
                name: xr.Variable(
                    self.dims[name], values.cpu().numpy(), dict(self.attrs[name])
                )
                for name, values in self.root.items()
            },
            coords=dict(self.coords),
            **xr_kwargs,
        )

    def _layout(
        self,
    ) -> tuple[dict[str, Mapping[Hashable, int]], dict[Hashable, np.ndarray]]:
        """Stored array dimension sizes and dimension labels."""
        arrays: dict[str, Mapping[Hashable, int]] = {
            name: OrderedDict(zip(self.dims[name], values.shape))
            for name, values in self.root.items()
        }
        coords = {
            key: np.asarray(variable.values)
            for key, variable in self.coords.items()
            if variable.dims == (key,)
        }
        return arrays, coords

    def _positions(
        self, indexers: Mapping[Hashable, Indexer], name: str
    ) -> dict[Hashable, np.ndarray]:
        """Map each dimension's indexer to an ascending integer position array."""
        positions: dict[Hashable, np.ndarray] = {}
        for dim, size in zip(self.dims[name], self.root[name].shape):
            indexer = indexers[dim]
            if isinstance(indexer, slice):
                start, stop, step = indexer.indices(size)
                positions[dim] = np.arange(start, stop, step)
            else:
                positions[dim] = np.asarray(indexer)
        return positions

    def flush(self) -> None:
        """Writes are synchronous; nothing to flush."""

    def close(self) -> None:
        """Reject later writes. The store remains readable."""
        self._closed = True


def _torch_dtype(dtype: np.dtype) -> torch.dtype:
    """Torch equivalent of a NumPy dtype."""
    try:
        return torch.from_numpy(np.empty(0, dtype=dtype)).dtype
    except TypeError as err:
        raise ValueError(f"KVBackend cannot store arrays of dtype {dtype}") from err


def _payload(tensor: torch.Tensor) -> Any:
    """NumPy or CuPy data for a tensor, following its device."""
    placeholder = OrderedDict(
        (f"dim_{index}", np.arange(size)) for index, size in enumerate(tensor.shape)
    )
    return from_torch(tensor, placeholder).data


def _shared_attrs(attrs: list[dict[Hashable, Any]]) -> dict[Hashable, Any]:
    """Attributes present with the same value in every mapping."""
    shared = dict(attrs[0])
    for other in attrs[1:]:
        shared = {
            key: value
            for key, value in shared.items()
            if key in other and _equal(value, other[key])
        }
    return shared


def _equal(left: Any, right: Any) -> bool:
    """Compare attribute values, tolerating array-valued attributes."""
    try:
        return bool(left == right)
    except ValueError:
        return bool(np.array_equal(left, right))
