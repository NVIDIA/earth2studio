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

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from operator import mul
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.type import CoordSystem

_BATCH_METADATA_KEY = "_earth2studio_batch"


@dataclass(frozen=True)
class _BatchMetadata:
    batch_dim: Hashable
    batch_dims: tuple[Hashable, ...]
    batch_shape: tuple[int, ...]
    original_dims: tuple[Hashable, ...]
    coordinates: dict[Hashable, xr.Variable]


def _get_cupy() -> Any:
    try:
        cp = import_module("cupy")
    except ImportError as error:
        raise ImportError(
            "CuPy is required for GPU-backed Earth2Studio DataArrays."
        ) from error
    return cp


def _is_cupy_array(data: Any) -> bool:
    try:
        cp = _get_cupy()
    except ImportError:
        return False
    return isinstance(data, cp.ndarray)


def _replace_data(array: xr.DataArray, data: Any) -> xr.DataArray:
    result = xr.DataArray(
        data=data,
        coords=array.coords,
        dims=array.dims,
        name=array.name,
        attrs=array.attrs,
    )
    result.encoding = array.encoding.copy()
    return result


def _is_contiguous(data: Any) -> bool:
    flags = getattr(data, "flags", None)
    return bool(flags is not None and flags.c_contiguous)


def _shares_memory(first: Any, second: Any) -> bool:
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return bool(np.shares_memory(first, second))
    cp = _get_cupy()
    return bool(cp.shares_memory(first, second))


def _reshape(data: Any, shape: tuple[int, ...], contiguous: bool) -> Any:
    if not isinstance(data, np.ndarray) and not _is_cupy_array(data):
        raise TypeError("Batching supports only NumPy- or CuPy-backed DataArrays")

    source = data
    if contiguous and not _is_contiguous(source):
        if isinstance(source, np.ndarray):
            source = np.ascontiguousarray(source)
        else:
            source = _get_cupy().ascontiguousarray(source)

    result = source.reshape(shape)
    if not contiguous and not _shares_memory(source, result):
        raise ValueError(
            "Batching these dimensions requires a copy; set contiguous=True"
        )
    return result


def _coord_system(array: xr.DataArray) -> CoordSystem:
    coords: CoordSystem = OrderedDict()
    for dim, size in array.sizes.items():
        if dim in array.coords:
            coords[str(dim)] = np.asarray(array.coords[dim].to_numpy())
        else:
            coords[str(dim)] = np.arange(size)
    return coords


def from_torch(
    tensor: torch.Tensor,
    coords: CoordSystem,
    name: Hashable | None = None,
    attrs: Mapping[Any, Any] | None = None,
    preserve_grad: bool = False,
) -> xr.DataArray:
    """Wrap a Torch tensor and coordinate system in an xarray DataArray.

    CPU tensors share memory with a NumPy-backed DataArray. CUDA tensors share memory
    with a CuPy-backed DataArray through DLPack.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor containing the data.
    coords : CoordSystem
        Ordered coordinate mapping with one entry per tensor dimension.
    name : Hashable | None, optional
        DataArray name, by default None
    attrs : Mapping[Any, Any] | None, optional
        DataArray attributes, by default None
    preserve_grad : bool, optional
        Preserve the Torch autograd graph, by default False

    Returns
    -------
    xr.DataArray
        DataArray sharing memory with the input tensor.

    Raises
    ------
    ValueError
        If coordinate dimensions do not match the tensor shape.
    TypeError
        If the tensor is not on a CPU or CUDA device.
    ImportError
        If a CUDA tensor is provided without CuPy installed.
    NotImplementedError
        If ``preserve_grad`` is True.
    """
    if preserve_grad:
        raise NotImplementedError(
            "Gradient-preserving Torch conversion is not implemented"
        )

    if len(coords) != tensor.ndim:
        raise ValueError("Coordinate dimensions do not match the tensor rank")

    xr_coords: dict[str, np.ndarray] = {}
    for (dim, values), size in zip(coords.items(), tensor.shape, strict=True):
        coordinate = np.asarray(values)
        if coordinate.ndim != 1 or coordinate.shape[0] != size:
            raise ValueError(
                f"Coordinate '{dim}' does not match tensor dimension size {size}"
            )
        xr_coords[dim] = coordinate

    detached = tensor.detach()
    if detached.device.type == "cpu":
        data = detached.numpy()
    elif detached.device.type == "cuda":
        data = _get_cupy().from_dlpack(detached)
    else:
        raise TypeError(f"Unsupported Torch device type '{detached.device.type}'")

    return xr.DataArray(
        data=data,
        coords=xr_coords,
        dims=tuple(coords),
        name=name,
        attrs=dict(attrs) if attrs is not None else None,
    )


@xr.register_dataarray_accessor("e2s")
class Earth2StudioAccessor:
    """Earth2Studio conversions and batching for xarray DataArrays."""

    def __init__(self, array: xr.DataArray) -> None:
        self._array = array

    @property
    def is_cupy(self) -> bool:
        """Whether the DataArray is backed by a CuPy array."""
        return _is_cupy_array(self._array.data)

    def as_cupy(self, device: int | None = None) -> xr.DataArray:
        """Return a CuPy-backed DataArray.

        Parameters
        ----------
        device : int | None, optional
            CUDA device index used for conversion, by default None

        Returns
        -------
        xr.DataArray
            DataArray with GPU-resident CuPy data.
        """
        cp = _get_cupy()
        if device is None:
            data = cp.asarray(self._array.data)
        else:
            with cp.cuda.Device(device):
                data = cp.asarray(self._array.data)
        return _replace_data(self._array, data)

    def as_numpy(self) -> xr.DataArray:
        """Return a NumPy-backed DataArray.

        Returns
        -------
        xr.DataArray
            DataArray with host-resident NumPy data.
        """
        if self.is_cupy:
            return _replace_data(self._array, self._array.data.get())
        if isinstance(self._array.data, np.ndarray):
            return _replace_data(self._array, self._array.data)
        return self._array.as_numpy()

    def to_torch(self, preserve_grad: bool = False) -> tuple[torch.Tensor, CoordSystem]:
        """Convert to the legacy Torch tensor and coordinate representation.

        Parameters
        ----------
        preserve_grad : bool, optional
            Preserve the Torch autograd graph, by default False

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Tensor sharing memory with the DataArray data and its coordinates.

        Raises
        ------
        TypeError
            If the DataArray is not backed by NumPy or CuPy.
        NotImplementedError
            If ``preserve_grad`` is True.
        """
        if preserve_grad:
            raise NotImplementedError(
                "Gradient-preserving Torch conversion is not implemented"
            )

        data = self._array.data
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif self.is_cupy:
            tensor = torch.from_dlpack(data)
        else:
            raise TypeError(
                "Torch conversion supports only NumPy- or CuPy-backed DataArrays"
            )
        return tensor, _coord_system(self._array)

    def batch(
        self,
        dims: Sequence[Hashable],
        batch_dim: Hashable = "batch",
        contiguous: bool = True,
    ) -> xr.DataArray:
        """Flatten dimensions into a leading batch dimension.

        Parameters
        ----------
        dims : Sequence[Hashable]
            Dimensions to flatten, ordered within the new batch dimension.
        batch_dim : Hashable, optional
            Name of the flattened dimension, by default "batch"
        contiguous : bool, optional
            Make copied data contiguous when a view is not possible, by default True

        Returns
        -------
        xr.DataArray
            DataArray with a leading flattened batch dimension.

        Raises
        ------
        ValueError
            If dimensions are invalid, the array is already batched, or copying is
            required while ``contiguous`` is False.
        TypeError
            If the DataArray is not backed by NumPy or CuPy.
        """
        batch_dims = tuple(dims)
        if not batch_dims:
            raise ValueError("At least one batch dimension is required")
        if len(set(batch_dims)) != len(batch_dims):
            raise ValueError("Batch dimensions must be unique")
        missing = [dim for dim in batch_dims if dim not in self._array.dims]
        if missing:
            raise ValueError(f"Batch dimensions not found: {missing}")
        if batch_dim in self._array.dims and batch_dim not in batch_dims:
            raise ValueError(f"Batch dimension '{batch_dim}' already exists")
        if _BATCH_METADATA_KEY in self._array.attrs:
            raise ValueError("DataArray already contains Earth2Studio batch metadata")

        remaining_dims = tuple(dim for dim in self._array.dims if dim not in batch_dims)
        transposed = self._array.transpose(*(batch_dims + remaining_dims))
        batch_shape = tuple(self._array.sizes[dim] for dim in batch_dims)
        batch_size = reduce(mul, batch_shape, 1)
        data = _reshape(
            transposed.data,
            (batch_size,) + tuple(transposed.shape[len(batch_dims) :]),
            contiguous,
        )

        affected_coords = {
            name: coord.variable.copy(deep=False)
            for name, coord in self._array.coords.items()
            if not set(coord.dims).isdisjoint(batch_dims)
        }
        coords: dict[Hashable, Any] = {
            name: coord.variable
            for name, coord in self._array.coords.items()
            if set(coord.dims).isdisjoint(batch_dims)
        }
        coords[batch_dim] = np.arange(batch_size)
        attrs = self._array.attrs.copy()
        attrs[_BATCH_METADATA_KEY] = _BatchMetadata(
            batch_dim=batch_dim,
            batch_dims=batch_dims,
            batch_shape=batch_shape,
            original_dims=tuple(self._array.dims),
            coordinates=affected_coords,
        )

        result = xr.DataArray(
            data=data,
            coords=coords,
            dims=(batch_dim,) + remaining_dims,
            name=self._array.name,
            attrs=attrs,
        )
        result.encoding = self._array.encoding.copy()
        return result

    def unbatch(self, contiguous: bool = True) -> xr.DataArray:
        """Restore dimensions flattened by :meth:`batch`.

        Parameters
        ----------
        contiguous : bool, optional
            Make copied data contiguous when a view is not possible, by default True

        Returns
        -------
        xr.DataArray
            DataArray with its original batch dimensions restored.

        Raises
        ------
        ValueError
            If batch metadata is missing or incompatible with the DataArray.
        TypeError
            If the DataArray is not backed by NumPy or CuPy.
        """
        metadata = self._array.attrs.get(_BATCH_METADATA_KEY)
        if not isinstance(metadata, _BatchMetadata):
            raise ValueError("DataArray does not contain Earth2Studio batch metadata")
        if not self._array.dims or self._array.dims[0] != metadata.batch_dim:
            raise ValueError(
                "Earth2Studio batch dimension must be the leading dimension"
            )
        if self._array.shape[0] != reduce(mul, metadata.batch_shape, 1):
            raise ValueError("Batch dimension size does not match stored metadata")

        current_dims = tuple(self._array.dims[1:])
        raw_dims = metadata.batch_dims + current_dims
        data = _reshape(
            self._array.data,
            metadata.batch_shape + tuple(self._array.shape[1:]),
            contiguous,
        )
        sizes = dict(zip(raw_dims, data.shape, strict=True))
        coords: dict[Hashable, Any] = {
            name: coord.variable
            for name, coord in self._array.coords.items()
            if name != metadata.batch_dim
        }
        coords.update(
            {
                name: variable
                for name, variable in metadata.coordinates.items()
                if all(
                    dim in sizes and sizes[dim] == variable.sizes[dim]
                    for dim in variable.dims
                )
            }
        )

        attrs = self._array.attrs.copy()
        del attrs[_BATCH_METADATA_KEY]
        result = xr.DataArray(
            data=data,
            coords=coords,
            dims=raw_dims,
            name=self._array.name,
            attrs=attrs,
        )
        target_dims = tuple(
            dim for dim in metadata.original_dims if dim in result.dims
        ) + tuple(dim for dim in result.dims if dim not in metadata.original_dims)
        result = result.transpose(*target_dims)
        result.encoding = self._array.encoding.copy()
        return result
