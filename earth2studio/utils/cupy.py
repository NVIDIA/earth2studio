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

import os
import warnings
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from operator import mul
from typing import Any, Literal, cast

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.type import CoordSystem

_BATCH_METADATA_KEY = "_earth2studio_batch"
ArrayBackend = Literal["auto", "numpy", "cupy", "torch"]


def _validate_backend(value: str) -> ArrayBackend:
    if value not in ("auto", "numpy", "cupy", "torch"):
        raise ValueError(
            f"Invalid array backend {value!r}; expected auto, numpy, cupy, or torch"
        )
    return cast(ArrayBackend, value)


def _backend_from_environment() -> ArrayBackend:
    return _validate_backend(os.getenv("EARTH2STUDIO_ARRAY_BACKEND", "auto"))


_DEFAULT_BACKEND = _backend_from_environment()
_BACKEND_OVERRIDE: ContextVar[ArrayBackend | None] = ContextVar(
    "earth2studio_array_backend", default=None
)


@contextmanager
def backend(value: ArrayBackend) -> Iterator[None]:
    """Temporarily select the backend used by :func:`from_torch`.

    Explicit conversion arguments override this context-local scope, which overrides
    ``EARTH2STUDIO_ARRAY_BACKEND`` (read at import, default ``auto``). Existing arrays
    retain their backend. Nested scopes restore the previous policy on exit.

    Parameters
    ----------
    value : ArrayBackend
        ``auto``, ``numpy``, ``cupy``, or ``torch``. Auto uses NumPy on CPU and CuPy
        on CUDA. Torch preserves autograd history.

    Yields
    ------
    None
        The backend override is active within the context.

    Raises
    ------
    ValueError
        If the backend name is invalid.
    """
    token = _BACKEND_OVERRIDE.set(_validate_backend(value))
    try:
        yield
    finally:
        _BACKEND_OVERRIDE.reset(token)


def _resolve_backend(value: ArrayBackend | None) -> ArrayBackend:
    return (
        _validate_backend(value)
        if value is not None
        else (_BACKEND_OVERRIDE.get() or _DEFAULT_BACKEND)
    )


class _TorchArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        # Validate dtype without exporting the payload or severing its graph.
        self.dtype = np.dtype(torch.empty((), dtype=tensor.dtype).numpy().dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def size(self) -> int:
        return self.tensor.numel()

    @property
    def real(self) -> _TorchArray:
        return _TorchArray(self.tensor.real)

    @property
    def imag(self) -> _TorchArray:
        return _TorchArray(
            self.tensor.imag
            if self.tensor.is_complex()
            else torch.zeros_like(self.tensor)
        )

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        raise TypeError(
            "Implicit NumPy conversion of Torch-backed data is unsupported; "
            "use .e2s.as_numpy() to export explicitly"
        )

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        if func is np.result_type:
            return np.result_type(
                *(
                    value.dtype if isinstance(value, _TorchArray) else value
                    for value in args
                )
            )
        parameters = {
            np.zeros_like: (),
            np.sum: ("axis", "dtype", "out", "keepdims"),
            np.mean: ("axis", "dtype", "out", "keepdims"),
            np.nansum: ("axis", "dtype", "out", "keepdims"),
            np.nanmean: ("axis", "dtype", "out", "keepdims"),
            np.concatenate: ("axis",),
            np.stack: ("axis",),
            np.where: ("x", "y"),
            np.broadcast_to: ("shape",),
            np.reshape: ("shape", "order"),
            np.transpose: ("axes",),
            np.moveaxis: ("source", "destination"),
        }
        options = dict(kwargs)
        if func is np.reshape and "newshape" in options:
            options["shape"] = options.pop("newshape")
        names = parameters.get(func, ())
        positional = dict(zip(names, args[1:]))
        if (
            func not in parameters
            or len(args) > len(names) + 1
            or options.keys() - set(names)
            or options.keys() & positional.keys()
        ):
            raise NotImplementedError(
                f"Torch backend does not support {func.__name__} with these arguments"
            )
        options.update(positional)
        if func is np.zeros_like:
            return _TorchArray(torch.zeros_like(self._coerce(args[0])))
        if func in (np.sum, np.mean, np.nansum, np.nanmean):
            axis = options.get("axis")
            dtype = options.get("dtype")
            keepdims = options.get("keepdims", False)
            if options.get("out") is not None:
                raise NotImplementedError(
                    "Torch backend does not support reduction out"
                )
            value = self._coerce(args[0])
            dtype = _torch_dtype(dtype) if dtype is not None else None
            if (
                dtype is None
                and func in (np.mean, np.nanmean)
                and not (value.is_floating_point() or value.is_complex())
            ):
                dtype = torch.float64
            reductions: dict[Any, Callable[..., torch.Tensor]] = {
                np.sum: torch.sum,
                np.mean: torch.mean,
                np.nansum: torch.nansum,
                np.nanmean: torch.nanmean,
            }
            # Reduce singleton axes to retain NumPy's NaN and dtype semantics.
            if axis == ():
                value, axis, keepdims = value.unsqueeze(-1), -1, False
            return _TorchArray(
                reductions[func](value, dim=axis, keepdim=keepdims, dtype=dtype)
            )
        if func in (np.concatenate, np.stack):
            axis = options.get("axis", 0)
            values = [self._coerce(value) for value in args[0]]
            if func is np.concatenate and axis is None:
                values = [value.reshape(-1) for value in values]
                axis = 0
            return _TorchArray(
                (torch.cat if func is np.concatenate else torch.stack)(values, dim=axis)
            )
        if func is np.where:
            if len(args) != 3 or kwargs:
                raise NotImplementedError(
                    "Torch backend supports only three-argument where"
                )
            return _TorchArray(
                torch.where(
                    self._coerce(args[0]), self._coerce(args[1]), self._coerce(args[2])
                )
            )
        if func is np.broadcast_to:
            return _TorchArray(
                torch.broadcast_to(self._coerce(args[0]), options["shape"])
            )
        if func is np.reshape:
            if options.get("order", "C") != "C":
                raise NotImplementedError("Torch backend supports only C-order reshape")
            return _TorchArray(self._coerce(args[0]).reshape(options["shape"]))
        if func is np.transpose:
            return args[0].transpose(options.get("axes"))
        if func is np.moveaxis:
            source, destination = options["source"], options["destination"]
            if not isinstance(source, int):
                source = tuple(source)
                destination = tuple(destination)
            return _TorchArray(
                torch.movedim(self._coerce(args[0]), source, destination)
            )

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        operations: dict[Any, Callable[..., torch.Tensor]] = {
            np.add: torch.add,
            np.subtract: torch.sub,
            np.multiply: torch.mul,
            np.true_divide: torch.true_divide,
            np.floor_divide: torch.floor_divide,
            np.power: torch.pow,
            np.remainder: torch.remainder,
            np.negative: torch.neg,
            np.positive: torch.positive,
            np.absolute: torch.abs,
            np.square: torch.square,
            np.sqrt: torch.sqrt,
            np.exp: torch.exp,
            np.log: torch.log,
            np.equal: lambda a, b: a == b,
            np.not_equal: lambda a, b: a != b,
            np.less: lambda a, b: a < b,
            np.less_equal: lambda a, b: a <= b,
            np.greater: lambda a, b: a > b,
            np.greater_equal: lambda a, b: a >= b,
            np.isnan: torch.isnan,
            np.isfinite: torch.isfinite,
            np.logical_not: torch.logical_not,
            np.logical_and: torch.logical_and,
            np.logical_or: torch.logical_or,
            np.invert: torch.bitwise_not,
        }
        if method != "__call__" or kwargs or ufunc not in operations:
            raise NotImplementedError(
                f"Torch backend does not support {ufunc.__name__}.{method} with {tuple(kwargs)}"
            )
        # Python scalars retain Torch's weak scalar promotion; array constants
        # retain their dtype and are transferred to the payload's device.
        values = [
            value if np.isscalar(value) else self._coerce(value) for value in inputs
        ]
        return _TorchArray(operations[ufunc](*values))

    def _coerce(self, value: Any) -> torch.Tensor:
        if isinstance(value, _TorchArray):
            return value.tensor
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            value = (
                value.copy()
                if not value.flags.writeable or any(s < 0 for s in value.strides)
                else value
            )
        return torch.as_tensor(value, device=self.tensor.device)

    def __getitem__(self, key: Any) -> _TorchArray:
        keys = list(key if isinstance(key, tuple) else (key,))
        if any(item is Ellipsis for item in keys):
            position = next(i for i, item in enumerate(keys) if item is Ellipsis)
            count = self.ndim - sum(
                item is not None and item is not Ellipsis for item in keys
            )
            keys[position : position + 1] = [slice(None)] * count
        value = self.tensor
        axis = 0
        for index, item in enumerate(keys):
            if item is None:
                continue
            if isinstance(item, slice) and item.step is not None and item.step < 0:
                indices = torch.arange(
                    *item.indices(value.shape[axis]), device=value.device
                )
                value = torch.index_select(value, axis, indices)
                keys[index] = slice(None)
            elif isinstance(item, np.ndarray):
                keys[index] = self._coerce(item)
            axis += 1
        return _TorchArray(value[tuple(keys)])

    def transpose(self, axes: Sequence[int] | None = None) -> _TorchArray:
        return _TorchArray(
            self.tensor.permute(
                tuple(reversed(range(self.ndim))) if axes is None else tuple(axes)
            )
        )

    def reshape(self, shape: tuple[int, ...]) -> _TorchArray:
        return _TorchArray(self.tensor.reshape(shape))

    def astype(self, dtype: Any, **kwargs: Any) -> _TorchArray:
        copy = kwargs.pop("copy", True)
        if kwargs:
            raise NotImplementedError(
                f"Torch backend does not support astype options {tuple(kwargs)}"
            )
        return _TorchArray(self.tensor.to(dtype=_torch_dtype(dtype), copy=copy))

    def __deepcopy__(self, memo: dict[int, Any]) -> _TorchArray:
        result = _TorchArray(self.tensor.clone())
        memo[id(self)] = result
        return result


def _torch_dtype(dtype: Any) -> torch.dtype:
    return torch.from_numpy(np.empty((), dtype=dtype)).dtype


def _with_grad(tensor: torch.Tensor, requires_grad: bool) -> torch.Tensor:
    if requires_grad and not tensor.requires_grad:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            raise ValueError("Gradient tracking requires a floating or complex dtype")
        return tensor.detach().requires_grad_(True)
    return tensor


def _tensor_data(
    tensor: torch.Tensor,
    policy: ArrayBackend,
    device: str | torch.device | int | None = None,
) -> Any:
    if tensor.device.type not in ("cpu", "cuda"):
        raise TypeError(f"Unsupported Torch device type '{tensor.device.type}'")
    if policy == "auto":
        policy = "numpy" if tensor.device.type == "cpu" else "cupy"
    target = (
        torch.device("cuda", device)
        if isinstance(device, int)
        else torch.device(device) if device is not None else None
    )
    if target is not None and target.type not in ("cpu", "cuda"):
        raise TypeError(f"Unsupported Torch device type '{target.type}'")
    if policy == "torch":
        return _TorchArray(tensor if target is None else tensor.to(target))
    if policy == "numpy" and target is not None and target.type != "cpu":
        raise ValueError("NumPy backend requires a CPU device")
    if policy == "cupy":
        cp = _get_cupy()
        if target is not None and target.type != "cuda":
            raise ValueError("CuPy backend requires a CUDA device")
        if target is None:
            target = (
                tensor.device
                if tensor.is_cuda
                else torch.device("cuda", cp.cuda.Device().id)
            )
    if tensor.requires_grad:
        warnings.warn(
            f"Converting gradient-tracked data to the {policy} backend drops "
            "autograd history; use the torch backend to preserve gradients.",
            UserWarning,
            stacklevel=3,
        )
    detached = tensor.detach()
    if policy == "numpy":
        return detached.cpu().resolve_conj().resolve_neg().numpy()
    return cp.from_dlpack(detached.to(target).resolve_conj().resolve_neg())


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
    if isinstance(data, _TorchArray):
        if contiguous:
            return _TorchArray(data.tensor.contiguous().reshape(shape))
        try:
            return _TorchArray(data.tensor.view(shape))
        except RuntimeError as error:
            raise ValueError(
                "Batching these dimensions requires a copy; set contiguous=True"
            ) from error
    if not isinstance(data, np.ndarray) and not _is_cupy_array(data):
        raise TypeError(
            "Batching supports only NumPy- or CuPy-backed DataArrays or the Torch backend"
        )

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
    coords: CoordSystem | xr.DataArray,
    name: Hashable | None = None,
    attrs: Mapping[Any, Any] | None = None,
    requires_grad: bool = False,
    *,
    backend: ArrayBackend | None = None,
) -> xr.DataArray:
    """Wrap a Torch tensor and coordinate system in an xarray DataArray.

    By default, CPU tensors share memory with NumPy and CUDA tensors share memory
    with CuPy through DLPack. Select the Torch backend to retain the original tensor
    and its autograd history. NumPy/CuPy exports warn and detach tracked tensors.
    Explicit backend arguments override :func:`backend` scopes, which override
    ``EARTH2STUDIO_ARRAY_BACKEND`` (read at import, default ``auto``).

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor containing the data.
    coords : CoordSystem | xr.DataArray
        Ordered dimension mapping or DataArray signature. A signature supplies
        all coordinates and attributes; signature-only attributes are omitted.
    name : Hashable | None, optional
        DataArray name, by default None
    attrs : Mapping[Any, Any] | None, optional
        DataArray attributes, by default None
    requires_grad : bool, optional
        Enable tracking if needed for floating/complex tensors, by default False
        False preserves existing tracking. A non-Torch output still warns and
        detaches when tracking is enabled.
    backend : ArrayBackend | None, optional
        Output policy (auto, numpy, cupy, torch), using the configured default
        when None, by default None
        NumPy transfers data to CPU; CuPy transfers CPU inputs to the current
        CUDA device. Torch retains the input device.

    Returns
    -------
    xr.DataArray
        DataArray sharing memory where the selected backend permits. Torch-backed
        arrays support selection, transpose, broadcasting, concatenation, copies,
        arithmetic, sum/mean, and batching. Unsupported operations raise rather
        than silently exporting to NumPy. Use ``.e2s.as_numpy()`` for explicit export.

    Raises
    ------
    ValueError
        If coordinates, backend policy, or requested gradient dtype are invalid.
    TypeError
        If the tensor is not on a CPU or CUDA device.
    ImportError
        If the selected output requires CuPy and it is not installed.

    Warns
    -----
    UserWarning
        If a non-Torch backend drops autograd history.
    """
    policy = _resolve_backend(backend)

    if isinstance(coords, xr.DataArray):
        if tuple(tensor.shape) != coords.shape:
            raise ValueError("Coordinate dimensions do not match the tensor shape")
        dimensions = coords.dims
        xr_coords = dict(coords.coords)
        metadata = dict(coords.attrs)
        if attrs is not None:
            metadata.update(attrs)
        for key in (
            "earth2studio_kind",
            "earth2studio_schema_version",
            "earth2studio_dynamic_dims",
        ):
            metadata.pop(key, None)
        attrs = metadata
        if name is None:
            name = coords.name
    else:
        dimensions = tuple(coords)
        xr_coords = {}
        for (dim, values), size in zip(coords.items(), tensor.shape):
            coordinate = np.asarray(values)
            if coordinate.ndim != 1 or coordinate.shape[0] != size:
                raise ValueError(
                    f"Coordinate '{dim}' does not match tensor dimension size {size}"
                )
            xr_coords[dim] = coordinate

    if len(dimensions) != tensor.ndim:
        raise ValueError("Coordinate dimensions do not match the tensor rank")

    data = _tensor_data(_with_grad(tensor, requires_grad), policy)

    return xr.DataArray(
        data=data,
        coords=xr_coords,
        dims=dimensions,
        name=name,
        attrs=dict(attrs) if attrs is not None else None,
    )


@xr.register_dataarray_accessor("e2s")  # type: ignore[no-untyped-call]
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

        Torch-backed inputs warn and detach if gradient tracking is enabled.

        Parameters
        ----------
        device : int | None, optional
            CUDA device index used for conversion, by default None

        Returns
        -------
        xr.DataArray
            DataArray with GPU-resident CuPy data.
        """
        if isinstance(self._array.data, _TorchArray):
            return self.to_backend("cupy", device=device)
        cp = _get_cupy()
        if device is None:
            data = cp.asarray(self._array.data)
        else:
            with cp.cuda.Device(device):
                data = cp.asarray(self._array.data)
        return _replace_data(self._array, data)

    def as_numpy(self) -> xr.DataArray:
        """Return a NumPy-backed DataArray.

        Torch-backed inputs warn and detach if gradient tracking is enabled.

        Returns
        -------
        xr.DataArray
            DataArray with host-resident NumPy data.
        """
        if isinstance(self._array.data, _TorchArray):
            return self.to_backend("numpy")
        if self.is_cupy:
            return _replace_data(self._array, self._array.data.get())
        if isinstance(self._array.data, np.ndarray):
            return _replace_data(self._array, self._array.data)
        return self._array.as_numpy()

    def to_torch(self, requires_grad: bool = False) -> tuple[torch.Tensor, CoordSystem]:
        """Convert to the legacy Torch tensor and coordinate representation.

        Dispatch uses the actual payload, independently of backend configuration.
        Torch-backed arrays return their original tensor and autograd history.

        Parameters
        ----------
        requires_grad : bool, optional
            Enable tracking for floating/complex data if needed, by default False
            False preserves existing tracking. NumPy/CuPy inputs start new leaves;
            this cannot recover history dropped by an earlier export.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Tensor sharing memory with the DataArray data and its coordinates.

        Raises
        ------
        TypeError
            If the DataArray is not backed by NumPy, CuPy, or Torch.
        ValueError
            If gradient tracking is requested for a non-floating/non-complex dtype.
        """
        data = self._array.data
        if isinstance(data, _TorchArray):
            tensor = data.tensor
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif self.is_cupy:
            tensor = torch.from_dlpack(data)
        else:
            raise TypeError(
                "Torch conversion supports only NumPy- or CuPy-backed DataArrays "
                "or the Torch backend"
            )
        return _with_grad(tensor, requires_grad), _coord_system(self._array)

    def to_backend(
        self, backend: ArrayBackend, *, device: str | torch.device | int | None = None
    ) -> xr.DataArray:
        """Convert the numerical payload while retaining coordinates and metadata.

        Parameters
        ----------
        backend : ArrayBackend
            Explicit output backend: auto, numpy, cupy, or torch. Auto chooses
            NumPy for CPU sources and CuPy for CUDA sources.
        device : str | torch.device | int | None, optional
            Output device, preserving the source device when possible, by default None
            Integer values are CUDA device indices. NumPy requires CPU; CuPy
            requires CUDA. Torch supports both.

        Returns
        -------
        xr.DataArray
            Converted array. Torch transfers preserve autograd. Exports to other
            backends warn and detach when the source requires gradients.

        Raises
        ------
        ValueError
            If the backend or requested device is incompatible.
        TypeError
            If the input backend or device type is unsupported.
        ImportError
            If CuPy is needed but unavailable.
        """
        policy = _validate_backend(backend)
        tensor, _ = self.to_torch()
        return _replace_data(self._array, _tensor_data(tensor, policy, device))

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
        NotImplementedError
            If a coordinate spans batched and unbatched dimensions.
        TypeError
            If the DataArray is not backed by NumPy, CuPy, or Torch.
        """
        batch_dims = tuple(dims)
        if _BATCH_METADATA_KEY in self._array.attrs or batch_dim in self._array.dims:
            raise ValueError("Recursive batching is not supported")
        if batch_dim in self._array.coords:
            raise ValueError(f"Batch coordinate '{batch_dim}' already exists")
        if not batch_dims:
            raise ValueError("At least one batch dimension is required")
        if len(set(batch_dims)) != len(batch_dims):
            raise ValueError("Batch dimensions must be unique")
        missing = [dim for dim in batch_dims if dim not in self._array.dims]
        if missing:
            raise ValueError(f"Batch dimensions not found: {missing}")
        unsupported_coords = [
            name
            for name, coord in self._array.coords.items()
            if not set(coord.dims).isdisjoint(batch_dims)
            and not set(coord.dims).issubset(batch_dims)
        ]
        if unsupported_coords:
            raise NotImplementedError(
                "Batching coordinates that span batched and unbatched dimensions "
                f"is not supported: {unsupported_coords}"
            )

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
            If the DataArray is not backed by NumPy, CuPy, or Torch.
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
        coords: dict[Hashable, Any] = {
            name: coord.variable
            for name, coord in self._array.coords.items()
            if name != metadata.batch_dim
        }
        coords.update(metadata.coordinates)

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
