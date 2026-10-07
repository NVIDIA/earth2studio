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

from collections.abc import Hashable, Iterator, Mapping
from typing import Any

import numpy as np
import torch
import xarray as xr
from numpy.typing import ArrayLike, DTypeLike

from earth2studio.io.utils import (
    fill_value,
    merge_coords,
    plan_arrays,
    plan_read,
    plan_write,
    read_selection,
    stack_fields,
    to_device,
    to_host,
)


class XarrayBackend:
    """An in-memory IO backend holding an :class:`xarray.Dataset`.

    Stored arrays are NumPy-backed regardless of the payload written. The dataset
    is available as :attr:`root`.

    Parameters
    ----------
    xr_kwargs : Any
        Optional keyword arguments passed to the :class:`xarray.Dataset`
        constructor, such as global ``attrs``.
    """

    def __init__(self, **xr_kwargs: Any) -> None:
        self.root = xr.Dataset(**xr_kwargs)
        self._closed = False

    def __contains__(self, item: str) -> bool:
        """Checks if item is an array or coordinate of the dataset.

        Parameters
        ----------
        item : str
        """
        return item in self.root

    def __getitem__(self, item: str) -> xr.DataArray:
        """Gets an array or coordinate of the dataset.

        Parameters
        ----------
        item : str
        """
        return self.root[item]

    def __len__(self) -> int:
        """Gets the number of stored arrays."""
        return len(self.root.data_vars)

    def __iter__(self) -> Iterator[Hashable]:
        """Return an iterator over stored array names."""
        return iter(self.root.data_vars)

    def add_array(self, schema: xr.DataArray) -> None:
        """Create the arrays a schema describes.

        Parameters
        ----------
        schema : xr.DataArray
            Concrete coordinate signature; field values are never read.
        """
        plan = plan_arrays(schema)
        existing = {key: value.variable for key, value in self.root.coords.items()}
        missing = merge_coords(existing, plan.coords)
        for name in plan.names:
            if name in self.root.data_vars and self.root[name].dims != plan.dims:
                raise ValueError(
                    f"Array '{name}' exists with dimensions {self.root[name].dims}"
                )
        self.root = self.root.assign_coords(missing)
        for name in plan.names:
            if name in self.root.data_vars:
                continue
            data = np.full(plan.shape, fill_value(plan.dtype), dtype=plan.dtype)
            self.root[name] = xr.Variable(plan.dims, data, dict(plan.attrs))

    def write(self, x: xr.DataArray) -> None:
        """Write a field at the positions its coordinate labels identify.

        Parameters
        ----------
        x : xr.DataArray
            NumPy-, CuPy- or Torch-backed field. It is never modified.
        """
        if self._closed:
            raise RuntimeError("Cannot write to a closed XarrayBackend")
        for name, indexers, field in plan_write(x, *self._layout()):
            self.root[name].variable[
                tuple(indexers[dim] for dim in self.root[name].dims)
            ] = to_host(field)

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
            Schema or field, or an ordered mapping from every dimension to its
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
        selection = read_selection(selection)
        names, indexers = plan_read(selection, *self._layout())
        fields = [self.root[name].isel(indexers) for name in names]
        field = stack_fields(selection, names, fields)
        if dtype is not None:
            field = field.astype(dtype)
        return to_device(field, device)

    def _layout(
        self,
    ) -> tuple[dict[str, tuple[Hashable, ...]], dict[Hashable, np.ndarray]]:
        """Stored array dimensions and dimension labels."""
        arrays = {str(name): self.root[name].dims for name in self.root.data_vars}
        coords = {
            dim: self.root.indexes[dim].values
            for dim in self.root.dims
            if dim in self.root.indexes
        }
        return arrays, coords

    def flush(self) -> None:
        """Writes are synchronous; nothing to flush."""

    def close(self) -> None:
        """Reject later writes. The dataset remains readable."""
        self._closed = True
