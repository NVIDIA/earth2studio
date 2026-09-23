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
import functools
import inspect
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterator
from itertools import chain, islice
from typing import Any, TypeVar

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.coords import (
    coord_array,
    handshake_coords,
    handshake_dim,
    handshake_nonempty,
    handshake_size,
    handshake_time,
)
from earth2studio.utils.cupy import _BATCH_METADATA_KEY
from earth2studio.utils.type import CoordinateSystem, CoordSystem

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


class batch_func:
    """Batch utility decorator which can be added to prognostic and diagnostic models
    to help enable support for automatic batching of data. This class contains a
    decorator function which should be added to calls where this functionality is
    desired.

    CoordinateSystem inputs remain allocation-free DataArrays paired with separate
    tensors. Leading dimensions are flattened into ``batch`` and restored on each
    output, including auxiliary coordinates. Dictionary coordinates remain supported
    for models whose input signatures are dictionaries.

    DataArray methods take one DataArray followed by optional pass-through arguments
    and return or yield a DataArray. The input signature must start with ``batch``;
    all remaining signature dimensions are fixed trailing dimensions. Leading input
    dimensions are packed with ``.e2s.batch()`` and restored with ``.e2s.unbatch()``.
    This preserves leading labels and auxiliary coordinates even if the model drops
    attributes. Output batch size must remain unchanged. Coordinates spanning both
    packed and fixed dimensions are rejected by the accessor.

    Note
    ----
    A model attributes `input_coords` and `output_coords` must have "batch" as the
    coordinate system of the first dimensions. I.e. first key entry needs to be "batch".

    Note
    ----
    When decorating a method of a model, such as `__call__`, the method is required to
    have a signature of `(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, CoordSystem]`.
    All positional arguments must be a sequence of (x, CoordSystem) pairs. All kwargs are
    passed through to the wrapped function without modification.

    Example
    -------
    ```python
    class Model():

        input_coords = OrderedDict([("batch", np.empty(0)), ...])
        output_coords = OrderedDict([("batch", np.empty(0)), ...])

        @batch_func()
        def __call__(
            self,
            x: torch.Tensor,
            coords: CoordSystem,
        ) -> tuple[torch.Tensor, CoordSystem]:
            ...
    ```
    """

    def __call__(self, func: F) -> Callable:
        if inspect.isgeneratorfunction(func):
            legacy = self._batch_wrap_generator(func)
            arrays = self._array_wrap_generator(func)
        else:
            legacy = self._batch_wrap(func)
            arrays = self._array_wrap(func)

        @functools.wraps(func)
        def _dispatch(model: Any, *args: Any, **kwargs: Any) -> Any:
            if args and isinstance(args[0], xr.DataArray):
                return arrays(model, *args, **kwargs)
            if not args and isinstance(kwargs.get("x"), xr.DataArray):
                return arrays(model, **kwargs)
            return legacy(model, *args, **kwargs)

        return _dispatch

    def _compress_array(
        self, model: Any, x: xr.DataArray
    ) -> tuple[xr.DataArray, Callable[[xr.DataArray], xr.DataArray]]:
        signature = model.input_coords()
        handshake_nonempty(x)
        for dim in ("time", "lead_time"):
            if dim in x.coords:
                handshake_time(x, dim, dimension=dim in x.dims)
        handshake_dim(signature, "batch", 0)
        fixed = signature.dims[1:]
        count = x.ndim - len(fixed)
        for index, dim in enumerate(fixed, start=-len(fixed)):
            try:
                handshake_dim(x, dim, index)
            except KeyError as error:
                raise ValueError(str(error)) from error
        leading = x.dims[:count]
        batch_coordinate = None
        if "batch" in x.coords and "batch" not in x.dims:
            batch_coordinate = x.coords["batch"].variable.copy(deep=True)
            x = x.drop_vars("batch")
        if not leading:
            packed = x.expand_dims(batch=[0])
            metadata = None
            temporary = "batch"
        else:
            temporary = "_model_batch"
            while temporary in x.dims or temporary in x.coords:
                temporary += "_"
            packed = x.e2s.batch(leading, batch_dim=temporary)
            metadata = packed.attrs[_BATCH_METADATA_KEY]
            packed = packed.rename({temporary: "batch"})
        size = packed.sizes["batch"]
        # Keep restoration state outside model metadata: model arithmetic may drop attrs.
        packed.attrs = {
            key: value
            for key, value in packed.attrs.items()
            if key != _BATCH_METADATA_KEY
        }

        def restore(out: xr.DataArray) -> xr.DataArray:
            if not isinstance(out, xr.DataArray):
                raise TypeError("Batched model must return a DataArray")
            handshake_dim(out, "batch", 0)
            handshake_size(out, "batch", size)
            handshake_coords(out, {"batch": np.arange(size)}, "batch")
            if metadata is None:
                out = out.isel(batch=0, drop=True)
            else:
                out = out.rename({"batch": temporary})
                out.attrs = {**out.attrs, _BATCH_METADATA_KEY: metadata}
                out = out.e2s.unbatch()
            if batch_coordinate is not None:
                out = out.assign_coords(batch=batch_coordinate)
            return out

        return packed, restore

    def _array_wrap(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def _wrapper(
            model: Any, x: xr.DataArray, *args: Any, **kwargs: Any
        ) -> xr.DataArray:
            packed, restore = self._compress_array(model, x)
            return restore(func(model, packed, *args, **kwargs))

        return _wrapper

    def _array_wrap_generator(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def _wrapper(
            model: Any, x: xr.DataArray, *args: Any, **kwargs: Any
        ) -> Iterator[xr.DataArray]:
            packed, restore = self._compress_array(model, x)
            gen = func(model, packed, *args, **kwargs)
            try:
                response = next(gen)
                while True:
                    output = restore(response)
                    try:
                        request = yield output
                    except GeneratorExit:
                        gen.close()
                        raise
                    except BaseException as exc:
                        response = gen.throw(exc)
                    else:
                        response = gen.send(request)
            except StopIteration as exc:
                return exc.value
            finally:
                gen.close()

        return _wrapper

    def _compress_batch(
        self, model: Any, x: torch.Tensor, coords: CoordSystem | CoordinateSystem
    ) -> tuple[
        torch.Tensor,
        CoordSystem | CoordinateSystem,
        CoordSystem | CoordinateSystem,
        torch.Size,
    ]:
        """Compresses dimensions into the models batch dimension

        Parameters
        ----------
        model : Any
            Any object, prognostic / diagnostic model that has a input_coords property
        x : torch.Tensor
            Input tensor to compress
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[ torch.Tensor, CoordSystem, CoordSystem, torch.Size, ]
            Returns batch compressed tensor, compressed coords, the coords of the batch
            dimensions and the shape of the batched dimensions. Later two are needed for
            decompression.

        Raises
        ------
        ValueError
            If model's input_coords do not contain the batch dimension
        """
        input_coords = model.input_coords()
        if isinstance(input_coords, xr.DataArray):
            return self._compress_coordinates(input_coords, x, coords)
        output_coords = model.output_coords
        if (
            next(iter(input_coords)) != "batch"
            or next(iter(output_coords(input_coords))) != "batch"
        ):
            raise ValueError(
                "Model coordinate systems not compatible with batch processing"
            )

        if len(x.shape) != len(coords):
            raise ValueError(
                "Input tensor shape does not match the provided coordinates"
            )
        flatten_coords: CoordSystem
        batched_coords: CoordSystem
        # If dims of input is one less than input coords, just prepend batch dim
        if len(x.shape) == len(input_coords) - 1:
            flatten_coords = coords.copy()
            flatten_coords.update({"batch": np.array([0])})
            flatten_coords.move_to_end("batch", last=False)
            return x.unsqueeze(0), flatten_coords, OrderedDict({}), torch.Size([])

        i = len(coords) - len(input_coords.keys()) + 1
        batched_shape = x.shape[:i]
        # Prep coordinate dicts
        batched_coords = OrderedDict(islice(coords.items(), 0, i))
        flatten_coords = OrderedDict(islice(coords.items(), i, None))
        flatten_coords.update({"batch": np.empty(0)})
        flatten_coords.move_to_end("batch", last=False)
        # Flatten batch dims
        x = torch.flatten(x, start_dim=0, end_dim=len(batched_coords) - 1)
        flatten_coords["batch"] = np.arange(x.shape[0])

        return x, flatten_coords, batched_coords, batched_shape

    def _compress_coordinates(
        self, signature: CoordinateSystem, x: torch.Tensor, coords: CoordinateSystem
    ) -> tuple[torch.Tensor, CoordinateSystem, CoordinateSystem, torch.Size]:
        if not isinstance(coords, xr.DataArray):
            raise TypeError("This model requires a CoordinateSystem coordinate array")
        if signature.dims[0] != "batch":
            raise ValueError("Model signature must start with batch")
        if tuple(x.shape) != coords.shape:
            raise ValueError("Input tensor shape does not match coordinates")
        fixed = len(signature.dims) - 1
        if coords.ndim < fixed:
            raise ValueError("Input has fewer dimensions than the model requires")
        n = coords.ndim - fixed
        leading = coords.dims[:n]
        shape = x.shape[:n]
        size = int(np.prod(shape))
        if size == 0:
            raise ValueError("Execution batch dimensions must be nonempty")
        coordinates: dict[Any, Any] = {}
        for name, coordinate in coords.coords.items():
            if name in leading or name == "batch":
                continue
            if any(d in leading for d in coordinate.dims):
                trailing = tuple(d for d in coordinate.dims if d not in leading)
                expanded = coordinate.variable.set_dims(
                    {d: coords.sizes[d] for d in (*leading, *trailing)}
                )
                coordinates[name] = (
                    ("batch", *trailing),
                    np.asarray(expanded).reshape(
                        size, *(coords.sizes[d] for d in trailing)
                    ),
                    dict(coordinate.attrs),
                )
            else:
                coordinates[name] = coordinate.variable
        coordinates["batch"] = np.arange(size)
        dims = ("batch", *coords.dims[n:])
        compressed = coord_array(
            dims,
            coordinates,
            sizes=dict(zip(dims, (size, *x.shape[n:]))),
            attrs=coords.attrs,
            name=coords.name,
            dtype=coords.dtype,
        )
        return x.reshape(size, *x.shape[n:]), compressed, coords, shape

    def _decompress_batch(
        self,
        out: torch.Tensor,
        out_coords: CoordSystem | CoordinateSystem,
        batched_coords: CoordSystem | CoordinateSystem,
        batched_shape: torch.Size,
    ) -> tuple[torch.Tensor, CoordSystem | CoordinateSystem]:
        """Decompresses the batch dimension of a tensor

        Parameters
        ----------
        out : torch.Tensor
            Batched tensor to decompress
        out_coords : CoordSystem
            Compressed coordinates
        batched_coords : CoordSystem
            The coords of the batch dimensions
        batched_shape : torch.Size
            The shape of the batched dimensions

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Uncompressed tensor and coordinates
        """

        if isinstance(batched_coords, xr.DataArray):
            if not isinstance(out_coords, xr.DataArray):
                raise TypeError("Model must return CoordinateSystem coordinates")
            size = int(np.prod(batched_shape))
            if (
                out_coords.dims[0] != "batch"
                or out.shape != out_coords.shape
                or out.shape[0] != size
                or not np.array_equal(out_coords.coords["batch"], np.arange(size))
            ):
                raise ValueError("Model changed batch shape, labels, or order")
            leading = batched_coords.dims[: len(batched_shape)]
            coordinates = {}
            for name, coordinate in out_coords.coords.items():
                if name == "batch":
                    continue
                if "batch" in coordinate.dims:
                    trailing = tuple(d for d in coordinate.dims if d != "batch")
                    values = np.asarray(
                        coordinate.transpose("batch", *trailing)
                    ).reshape(*batched_shape, *(out_coords.sizes[d] for d in trailing))
                    variable = xr.Variable(
                        (*leading, *trailing), values, attrs=coordinate.attrs
                    )
                    if name in batched_coords.coords:
                        original = batched_coords.coords[name]
                        variable = variable.isel(
                            {d: 0 for d in leading if d not in original.dims}
                        )
                        variable = variable.transpose(*original.dims)
                    coordinates[name] = variable
                else:
                    coordinates[name] = coordinate.variable
            for name, coordinate in batched_coords.coords.items():
                if name in leading or (name == "batch" and "batch" not in leading):
                    coordinates[name] = coordinate.variable
            dims = (*leading, *out_coords.dims[1:])
            shape = (*batched_shape, *out.shape[1:])
            restored = coord_array(
                dims,
                coordinates,
                sizes=dict(zip(dims, shape)),
                attrs=out_coords.attrs,
                name=out_coords.name,
                dtype=out_coords.dtype,
            )
            return out.reshape(shape), restored

        # Reconstruct batch dims
        out = out.reshape(batched_shape + out.shape[1:])
        out_coords = out_coords.copy()
        del out_coords["batch"]
        out_coords = OrderedDict(chain(batched_coords.items(), out_coords.items()))
        return out, out_coords

    def _batch_wrap(self, func: Callable) -> Callable:
        """Standard batch function decorator"""

        # TODO: Better typing for model object
        @functools.wraps(func)
        def _wrapper(
            model: Any, *args: Any, **kwargs: Any
        ) -> tuple[torch.Tensor, CoordSystem]:
            # Validate positional args are ONLY a sequence of (x, CoordSystem) pairs
            if len(args) == 0:
                raise ValueError(
                    "batch_func requires at least one positional (x, CoordSystem) pair"
                )
            if len(args) % 2 != 0:
                raise ValueError(
                    "Invalid positional arguments: expected (x, CoordSystem) pairs"
                )

            for i in range(0, len(args), 2):
                xi = args[i]
                ci = args[i + 1]
                if not isinstance(xi, torch.Tensor) or not isinstance(
                    ci, (OrderedDict, xr.DataArray)
                ):
                    raise ValueError(
                        "Invalid positional arguments: only (torch.Tensor, CoordSystem) pairs are supported"
                    )

            # Support any number of paired (x, CoordSystem) positional args; kwargs won't be batched
            new_args: list[Any] = list(args)
            ref_shape = None
            ref_coords = None
            for i in range(0, len(new_args), 2):
                xi = new_args[i]
                ci = new_args[i + 1]
                (
                    x_comp,
                    coords_comp,
                    batched_coords,
                    batched_shape,
                ) = self._compress_batch(model, xi, ci)
                new_args[i] = x_comp
                new_args[i + 1] = coords_comp
                if ref_shape is None and ref_coords is None:
                    ref_shape = batched_shape
                    ref_coords = batched_coords
                elif batched_shape != ref_shape:
                    # Guarantee that all batched_shape / batched_coords pairs match
                    raise ValueError(
                        "Mismatched batched dimensions across input (x, CoordSystem) pairs"
                    )
                elif isinstance(ref_coords, xr.DataArray):
                    leading = ref_coords.dims[: len(ref_shape)]
                    if ci.dims[: len(ref_shape)] != leading or any(
                        not ref_coords.coords[d].variable.equals(ci.coords[d].variable)
                        for d in leading
                    ):
                        raise ValueError("Mismatched batch labels across input pairs")

            # Model forward
            out, out_coords = func(model, *new_args, **kwargs)
            out, out_coords = self._decompress_batch(
                out, out_coords, ref_coords, ref_shape
            )
            return out, out_coords

        return _wrapper

    def _batch_wrap_generator(self, func: Callable) -> Callable:
        """Wrapper generator functions, for each output we need to decompress batches"""

        # Based on Pytorch: # https://github.com/pytorch/pytorch/pull/68617/files
        # TODO: Better typing for model object
        @functools.wraps(func)
        def _wrapper(
            model: Any, x: torch.Tensor, coords: CoordSystem
        ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:

            x, flatten_coords, batched_coords, batched_shape = self._compress_batch(
                model, x, coords
            )

            gen = func(model, x, flatten_coords)

            # Run the generator
            try:
                # Prime it
                response = gen.send(None)

                while True:
                    try:
                        # Forward the response to our caller and get its next request
                        out, out_coords = response
                        out, out_coords = self._decompress_batch(
                            out, out_coords, batched_coords, batched_shape
                        )
                        request = yield out, out_coords

                    except GeneratorExit:  # noqa: PERF203
                        # Inform the still active generator about its imminent closure
                        gen.close()
                        raise

                    except BaseException:
                        # Propagate the exception thrown at us by the caller
                        response = gen.throw(*sys.exc_info())

                    else:
                        # Get next response from generator
                        response = gen.send(request)

            except StopIteration as e:
                # The generator informed us that it is done
                return e.value

        return _wrapper
