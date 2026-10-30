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
import pandas as pd
import torch
import xarray as xr
import zarr
from numpy.typing import ArrayLike, DTypeLike
from zarr.core.array import Array as ZarrArray
from zarr.core.array import CompressorsLike

from earth2studio.io.utils import Indexer, plan_arrays, plan_read, plan_write
from earth2studio.utils.type import CoordSystem

# CF attribute listing the auxiliary coordinates of a stored array
_COORDINATES = "coordinates"


class ZarrBackend:
    """A backend that supports the zarr format.

    One array is created per ``variable`` label of an :meth:`add_array` template,
    with the dimension and auxiliary coordinates stored alongside them so the group
    opens directly with :func:`xarray.open_zarr` (see :meth:`to_xarray`).

    Parameters
    ----------
    file_name : str, optional
        Optional name to provide to the zarr store, if provided then this function
        will create a directory store with this file name. If none, will create a
        in-memory store., by default None
    chunks : dict[str, int], optional
        An ordered dict of chunk sizes keyed by dimension. Dimensions that are not
        present in a stored array are ignored and dimensions without an entry are
        stored in a single chunk, by default {"ensemble": 1, "time": 1,
        "lead_time": 1}
    backend_kwargs : dict[str, Any], optional
        Key word arguments for zarr.Group root object, by default {"overwrite": False}
    zarr_codecs: CompressorsLike, optional
        Compression codec to use when creating any new arrays. Only effects Zarr 3.0.
        If None, will use no compressor, by default None

    Note
    ----
    For keyword argument options see: https://zarr.readthedocs.io/en/latest/api/zarr/index.html#zarr.group

    Note
    ----
    Variable labels name arrays verbatim, including statistic qualifiers such as
    ``tp:sum:6h``. A local store creates one directory per array, so such labels
    require a file system that permits ``:`` in names.
    """

    # --8<-- [start:zarr-backend-read]
    def __init__(
        self,
        file_name: str = None,
        chunks: dict[str, int] = {  # to avoid writing in the same chunk by default
            "ensemble": 1,  # dimensions not present in data are ignored
            "time": 1,
            "lead_time": 1,
        },
        backend_kwargs: dict[str, Any] = {"overwrite": False},
        zarr_codecs: CompressorsLike = None,
    ) -> None:

        if file_name is None:
            self.store = zarr.storage.MemoryStore()
        else:
            self.store = zarr.storage.LocalStore(file_name)

        self.root = zarr.group(self.store, **backend_kwargs)
        self.zarr_codecs = zarr_codecs
        self._closed = False
        self._read_store_state(chunks)

    def _read_store_state(self, chunks: dict[str, int]) -> None:
        """Populate coords, chunks and coordinate names from the store's arrays."""
        self.chunks = dict(chunks)
        self.coords: CoordSystem = OrderedDict({})
        self._coordinate_names: set[str] = set()

        arrays = {name: self.root[name] for name in self.root.array_keys()}
        dimensions: set[str] = set()
        for array in arrays.values():
            dimensions.update(self._dims(array))
            self._coordinate_names.update(self._auxiliary_names(array))
        self._coordinate_names.update(dimensions.intersection(arrays))

        for name, array in arrays.items():
            if name in self._coordinate_names:
                if self._dims(array) == (name,):
                    self.coords[name] = array[:]
            else:
                # https://github.com/pydata/xarray/pull/9669
                for dim, size in zip(self._dims(array), array.chunks):
                    self.chunks[dim] = size

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

    # --8<-- [end:zarr-backend-read]

    @property
    def write_blocks(self) -> dict[str, int]:
        """Largest chunk or shard extent per dimension over the stored data arrays.

        A block is the smallest unit two writers can update independently, so
        distributed writers must not share one.
        """
        blocks: dict[str, int] = {}
        for name in self.root.array_keys():
            if name in self._coordinate_names:
                continue
            array = self.root[name]
            extents = array.shards or array.chunks
            for dim, extent in zip(self._dims(array), extents):
                blocks[str(dim)] = max(blocks.get(str(dim), 0), int(extent))
        return blocks

    def add_array(self, template: xr.DataArray) -> None:
        """Create the arrays a template describes in the zarr group.

        Coordinates are written once, here, together with their attributes. Each
        stored array lists its auxiliary coordinates in a CF ``coordinates``
        attribute and carries the template's attributes.

        Parameters
        ----------
        template : xr.DataArray
            Concrete coordinate signature; field values are never read.
        """
        # Another process may have added arrays since this backend opened the store
        self._read_store_state(self.chunks)
        plan = plan_arrays(template, self._coordinate_variables())
        arrays, _ = self._layout()
        clashes = set(map(str, plan.coords)).intersection(arrays)
        if clashes:
            raise ValueError(f"Coordinates collide with arrays: {sorted(clashes)}")
        for name in plan.names:
            if name not in arrays:
                continue
            if tuple(arrays[name]) != plan.dims:
                raise ValueError(
                    f"Array '{name}' exists with dimensions {tuple(arrays[name])}"
                )
            if tuple(arrays[name].values()) != plan.shape:
                raise ValueError(
                    f"Array '{name}' exists with shape {tuple(arrays[name].values())}"
                )

        for key, variable in plan.coords.items():
            self._create(
                str(key),
                shape=variable.shape,
                chunks=variable.shape,
                dtype=variable.dtype,
                dims=variable.dims,
                fill_value=None,
                attrs=variable.attrs,
            )[...] = variable.values
            self._coordinate_names.add(str(key))
            if variable.dims == (key,):
                self.coords[str(key)] = np.asarray(variable.values)

        auxiliary = self._auxiliary_coords(template)
        attrs = dict(plan.attrs)
        if auxiliary:
            attrs[_COORDINATES] = " ".join(auxiliary)
        for name in plan.names:
            if name in arrays:
                continue
            self._create(
                name,
                shape=plan.shape,
                chunks=tuple(
                    min(self.chunks.get(str(dim), size), size)
                    for dim, size in zip(plan.dims, plan.shape)
                ),
                dtype=plan.dtype,
                dims=plan.dims,
                fill_value=plan.fill_value,
                attrs=attrs,
            )

    def write(self, x: xr.DataArray) -> None:
        """Write a field at the positions its coordinate labels identify.

        Parameters
        ----------
        x : xr.DataArray
            NumPy-, CuPy- or Torch-backed field. It is never modified.
        """
        if self._closed:
            raise RuntimeError(f"Cannot write to a closed {type(self).__name__}")
        arrays, coords = self._layout()
        for name, indexers, field in plan_write(x, arrays, coords):
            selection = tuple(indexers[dim] for dim in arrays[name])
            self._write_array(name, selection, field.e2s.as_numpy().values)

    def _write_array(self, name: str, selection: tuple, data: np.ndarray) -> None:
        """Assign `data` into `self.root[name][selection]`.

        Split out from `write` so subclasses can override just the storage step,
        e.g. to run it asynchronously.
        """
        array = self.root[name]
        if any(not isinstance(index, slice) for index in selection):
            # Orthogonal indexing: label subsets select each axis independently
            array.oindex[selection] = data
        else:
            # Basic slices, the common case, are roughly an order of magnitude
            # faster than zarr's orthogonal indexing path
            array[selection] = data

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
        self.flush()
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
            field = fields[0].rename(names[0])
        field = field.transpose(*selection.dims)
        if dtype is not None:
            field = field.astype(dtype)
        if torch.device(device).type == "cuda":
            return field.e2s.as_cupy(torch.device(device).index)
        return field

    def to_xarray(self, **xr_kwargs: Any) -> xr.Dataset:
        """Open the store as an Xarray dataset, flushing pending writes first.

        Parameters
        ----------
        xr_kwargs : Any
            Optional keyword arguments passed to :func:`xarray.open_zarr`.

        Returns
        -------
        xr.Dataset
            Lazily loaded dataset of the stored arrays and their coordinates.
        """
        self.flush()
        return xr.open_zarr(self.store, **{"consolidated": False, **xr_kwargs})

    def flush(self) -> None:
        """Writes are synchronous; nothing to flush."""

    def close(self) -> None:
        """Reject later writes. The store remains readable."""
        self.flush()
        self._closed = True

    def _create(
        self,
        name: str,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: np.dtype,
        dims: tuple[Hashable, ...],
        fill_value: Any,
        attrs: Mapping[Hashable, Any],
    ) -> "ZarrArray":
        """Create one zarr array with its dimension names and attributes."""
        return self.root.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            dimension_names=[str(dim) for dim in dims],
            fill_value=fill_value,
            compressors=self.zarr_codecs,
            attributes={str(key): value for key, value in attrs.items()},
        )

    def _read_array(self, name: str, indexers: dict[Hashable, Indexer]) -> xr.DataArray:
        """Read one stored array and its coordinates in the selection's order."""
        array = self.root[name]
        dims = self._dims(array)
        attrs = dict(array.attrs)
        auxiliary = self._auxiliary_names(array)
        attrs.pop(_COORDINATES, None)

        # zarr reads ascending positions only, so undo the selection order after
        ascending, undo = self._ascending(indexers)
        coords: dict[Hashable, Any] = {
            dim: xr.Variable(
                (dim,), self.coords[dim][ascending[dim]], dict(self.root[dim].attrs)
            )
            for dim in dims
            if dim in self.coords
        }
        for key in auxiliary:
            coordinate = self.root[key]
            coords[key] = xr.Variable(
                self._dims(coordinate),
                self._select(coordinate, ascending),
                dict(coordinate.attrs),
            )
        field = xr.DataArray(
            self._select(array, ascending), dims=dims, coords=coords, attrs=attrs
        )
        return field.isel(undo) if undo else field

    @staticmethod
    def _ascending(
        indexers: Mapping[Hashable, Indexer],
    ) -> tuple[dict[Hashable, Indexer], dict[Hashable, np.ndarray]]:
        """Sort each indexer, with the positions that restore the given order."""
        ascending: dict[Hashable, Indexer] = {}
        undo: dict[Hashable, np.ndarray] = {}
        for dim, index in indexers.items():
            if isinstance(index, slice):
                ascending[dim] = index
                continue
            order = np.argsort(index, kind="stable")
            ascending[dim] = index[order]
            if np.any(order != np.arange(order.size)):
                undo[dim] = np.argsort(order)
        return ascending, undo

    def _select(
        self, array: "ZarrArray", indexers: Mapping[Hashable, Indexer]
    ) -> np.ndarray:
        """Read the part of a stored array the indexers of its dimensions select."""
        selection = tuple(indexers.get(dim, slice(None)) for dim in self._dims(array))
        if any(not isinstance(index, slice) for index in selection):
            return array.oindex[selection]
        return array[selection]

    def _layout(
        self,
    ) -> tuple[dict[str, Mapping[Hashable, int]], dict[Hashable, np.ndarray]]:
        """Stored array dimension sizes and dimension labels."""
        arrays: dict[str, Mapping[Hashable, int]] = {}
        for name in self.root.array_keys():
            if name in self._coordinate_names:
                continue
            array = self.root[name]
            arrays[name] = dict(zip(self._dims(array), array.shape))
        coords: dict[Hashable, np.ndarray] = {}
        coords.update(self.coords)
        return arrays, coords

    def _coordinate_variables(self) -> dict[Hashable, xr.Variable]:
        """Stored coordinates, with their dimensions and attributes."""
        variables: dict[Hashable, xr.Variable] = {}
        for name in self._coordinate_names:
            array = self.root[name]
            variables[name] = xr.Variable(
                self._dims(array), array[...], dict(array.attrs)
            )
        return variables

    @staticmethod
    def _dims(array: "ZarrArray") -> tuple[str, ...]:
        """Dimension names of a stored array."""
        return tuple(str(dim) for dim in array.metadata.dimension_names or ())

    @staticmethod
    def _auxiliary_names(array: "ZarrArray") -> list[str]:
        """Auxiliary coordinates a stored array lists in its CF attribute."""
        return str(array.attrs.get(_COORDINATES, "")).split()

    @staticmethod
    def _auxiliary_coords(template: xr.DataArray) -> list[str]:
        """Coordinates of a template that are not dimension coordinates."""
        return [
            str(key)
            for key, value in template.coords.items()
            if key != "variable"
            and "variable" not in value.dims
            and value.dims != (key,)
        ]
