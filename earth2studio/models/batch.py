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
import gc
import inspect
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from itertools import chain, islice
from typing import Any, TypeVar

import numpy as np
import torch
import xarray as xr
from loguru import logger

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

_OOM_TOKENS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError) or any(
        c.__name__ == "OutOfMemoryError" for c in type(exc).__mro__
    ):
        return True
    return isinstance(exc, RuntimeError) and any(
        t in str(exc).lower() for t in _OOM_TOKENS
    )


def _clear_gpu_memory(device: torch.device | str | None = None) -> None:
    gc.collect()
    target = torch.device(device) if device is not None else None
    if (target is None or target.type == "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        cp = sys.modules.get("cupy")
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()


def _infer_device(
    model: Any = None,
    x: tuple[xr.DataArray, ...] = (),
    device: torch.device | str | None = None,
) -> torch.device:
    if device is not None:
        return torch.device(device)
    for arr in x:
        if hasattr(arr, "e2s"):
            return arr.e2s.to_torch()[0].device
    if isinstance(model, torch.nn.Module):
        for p in chain(model.parameters(), model.buffers()):
            return p.device
    dev = getattr(model, "device", None)
    if isinstance(dev, (torch.device, str)):
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fixed_ndim(model: Any, x: xr.DataArray) -> int:
    ic_fn = getattr(model, "input_coords", None)
    if callable(ic_fn):
        try:
            ic = ic_fn()
            ic = ic[0] if isinstance(ic, tuple) else ic
            dims = list(ic.dims if isinstance(ic, xr.DataArray) else ic.keys())
            return max(0, len(dims) - (1 if dims and dims[0] == "batch" else 0))
        except Exception:  # noqa: S110
            pass
    return max(0, x.ndim - 1)


def _tensor_bytes(
    dims: Sequence[Any],
    sizes: dict[Any, int],
    itemsize: int,
    override: CoordSystem | None = None,
) -> int:
    shape = [
        len(override[d]) if (override and d in override) else max(1, int(sizes[d]))
        for d in dims
    ]
    return (int(np.prod(shape)) if shape else 1) * itemsize


def _estimate_sample_memory(
    *x: xr.DataArray,
    output_coords: CoordSystem | xr.DataArray | None = None,
    model: Any = None,
    activation_multiplier: float = 4.0,
) -> int:
    override = output_coords if isinstance(output_coords, OrderedDict) else None
    input_bytes = 0
    for xi in x:
        fixed = _fixed_ndim(model, xi)
        dims = xi.dims[-fixed:] if fixed > 0 else ()
        input_bytes += _tensor_bytes(dims, dict(xi.sizes), xi.dtype.itemsize)

    output_bytes = input_bytes
    if (
        model is not None
        and hasattr(model, "output_coords")
        and hasattr(model, "input_coords")
    ):
        try:
            ic = model.input_coords()
            oc = (
                model.output_coords(*ic)
                if isinstance(ic, tuple)
                else model.output_coords(ic)
            )
            slots = oc if isinstance(oc, tuple) else (oc,)
            slot_bytes = sum(
                _tensor_bytes(
                    [d for d in s.dims if d not in ("batch", "time", "ensemble")],
                    dict(s.sizes),
                    s.dtype.itemsize,
                )
                + _tensor_bytes(
                    [d for d in s.dims if d not in ("batch", "time", "ensemble")],
                    dict(s.sizes),
                    s.dtype.itemsize,
                    override,
                )
                for s in slots
                if isinstance(s, xr.DataArray)
            )
            if slot_bytes > 0:
                output_bytes = slot_bytes
        except Exception:  # noqa: S110
            pass
    elif override and x:
        xi = x[0]
        fixed = _fixed_ndim(model, xi)
        dims = xi.dims[-fixed:] if fixed > 0 else ()
        output_bytes = input_bytes + _tensor_bytes(
            dims, dict(xi.sizes), xi.dtype.itemsize, override
        )

    return max(1, int(input_bytes * activation_multiplier + output_bytes))


def _measure_model_memory(
    model: Any,
    *x: xr.DataArray,
    step_fn: Callable[..., Any] | None = None,
    output_coords: CoordSystem | None = None,
    device: torch.device | str | None = None,
) -> int:
    sample_x = tuple(
        xi.isel(
            {
                d: slice(0, 1)
                for d in xi.dims[: max(0, xi.ndim - _fixed_ndim(model, xi))]
            }
        ).copy(deep=True)
        for xi in x
    )
    analytic = _estimate_sample_memory(
        *sample_x, output_coords=output_coords, model=model
    )
    target = _infer_device(model=model, x=sample_x, device=device)
    fn = step_fn if step_fn is not None else model
    if target.type != "cuda" or not torch.cuda.is_available() or not callable(fn):
        return analytic

    idx = target.index if target.index is not None else torch.cuda.current_device()
    cuda_dev = torch.device(f"cuda:{idx}")
    gens = [
        (v, v.get_state())
        for v in getattr(model, "__dict__", {}).values()
        if isinstance(v, torch.Generator)
    ]
    _clear_gpu_memory(cuda_dev)
    baseline = int(torch.cuda.memory_allocated(cuda_dev))
    torch.cuda.reset_peak_memory_stats(cuda_dev)
    try:
        with torch.random.fork_rng(devices=[idx]), torch.no_grad():
            out = fn(*sample_x)
            torch.cuda.synchronize(cuda_dev)
            peak = int(torch.cuda.max_memory_allocated(cuda_dev))
            del out
    except BaseException as exc:
        if _is_oom_error(exc):
            _clear_gpu_memory(cuda_dev)
            return sys.maxsize
        raise
    finally:
        for gen, state in gens:
            gen.set_state(state)
        del sample_x
        _clear_gpu_memory(cuda_dev)

    return max(peak - baseline, analytic)


class AutoBatcher:
    """VRAM-aware automatic batch size estimator and execution helper with OOM backoff.

    Parameters
    ----------
    memory_fraction : float, optional
        Fraction of available free GPU memory to allocate per batch in ``(0.0, 1.0]``,
        by default 0.8
    max_batch_size : int | None, optional
        Optional upper bound on the selected batch size, by default None
    min_batch_size : int, optional
        Minimum batch size to try before raising an out-of-memory error, by default 1
    oom_backoff : bool, optional
        Whether to halve the batch size and retry on OOM, by default True
    per_sample_bytes : int | None, optional
        Optional user-supplied per-sample memory cost in bytes, by default None
    available_bytes : int | None, optional
        Optional explicit available memory in bytes, by default None
    """

    def __init__(
        self,
        memory_fraction: float = 0.8,
        max_batch_size: int | None = None,
        min_batch_size: int = 1,
        oom_backoff: bool = True,
        per_sample_bytes: int | None = None,
        available_bytes: int | None = None,
    ) -> None:
        if not (0.0 < memory_fraction <= 1.0):
            raise ValueError(
                f"memory_fraction must be in (0.0, 1.0], got {memory_fraction}"
            )
        if min_batch_size < 1:
            raise ValueError(f"min_batch_size must be >= 1, got {min_batch_size}")
        if max_batch_size is not None and max_batch_size < min_batch_size:
            raise ValueError(
                f"max_batch_size ({max_batch_size}) must be >= min_batch_size ({min_batch_size})"
            )
        if per_sample_bytes is not None and per_sample_bytes < 1:
            raise ValueError(f"per_sample_bytes must be >= 1, got {per_sample_bytes}")
        self.memory_fraction = memory_fraction
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.oom_backoff = oom_backoff
        self.per_sample_bytes = per_sample_bytes
        self.available_bytes = available_bytes

    @classmethod
    def from_config(
        cls, autobatch: "bool | AutoBatcher", batch_size: int | None = None
    ) -> "AutoBatcher | None":
        """Resolve an ``autobatch`` flag or instance paired with an optional ``batch_size`` cap.

        Parameters
        ----------
        autobatch : bool | AutoBatcher
            Autobatch configuration flag or instance.
        batch_size : int | None, optional
            Explicit batch size cap, by default None

        Returns
        -------
        AutoBatcher | None
            Configured :class:`AutoBatcher` or ``None`` when autobatching is disabled.
        """
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if isinstance(autobatch, cls):
            if batch_size is not None and autobatch.max_batch_size is None:
                return cls(
                    memory_fraction=autobatch.memory_fraction,
                    max_batch_size=batch_size,
                    min_batch_size=autobatch.min_batch_size,
                    oom_backoff=autobatch.oom_backoff,
                    per_sample_bytes=autobatch.per_sample_bytes,
                    available_bytes=autobatch.available_bytes,
                )
            return autobatch
        return cls(max_batch_size=batch_size) if autobatch else None

    def estimate_batch_size(
        self,
        model: Any,
        *x: xr.DataArray,
        total_samples: int | None = None,
        step_fn: Callable[..., Any] | None = None,
        output_coords: CoordSystem | None = None,
        device: torch.device | str | None = None,
    ) -> int:
        """Estimate the optimal batch size for ``model`` given input DataArrays ``x``.

        Parameters
        ----------
        model : Any
            Prognostic or diagnostic model to run.
        *x : xr.DataArray
            Sample or full input DataArrays.
        total_samples : int | None, optional
            Total number of batch items to process, by default None
        step_fn : Callable[..., Any] | None, optional
            Optional callable to execute during the 1-sample memory probe, by default None
        output_coords : CoordSystem | None, optional
            Optional output coordinate system override, by default None
        device : torch.device | str | None, optional
            Target execution device, by default None

        Returns
        -------
        int
            Selected batch size.
        """
        if total_samples is not None and total_samples <= self.min_batch_size:
            return max(1, total_samples)
        target = _infer_device(model=model, x=x, device=device)
        cost = self.per_sample_bytes or _measure_model_memory(
            model, *x, step_fn=step_fn, output_coords=output_coords, device=target
        )
        if self.available_bytes is not None:
            free_bytes = max(0, int(self.available_bytes))
        elif target.type == "cuda" and torch.cuda.is_available():
            _clear_gpu_memory(target)
            free_bytes = int(torch.cuda.mem_get_info(target)[0])
        else:
            free_bytes = 4 * (1024**3)

        size = max(self.min_batch_size, int(free_bytes * self.memory_fraction) // cost)
        if self.max_batch_size is not None:
            size = min(size, self.max_batch_size)
        if total_samples is not None:
            size = min(size, max(self.min_batch_size, total_samples))
        return int(size)

    def backoff(
        self,
        exc: BaseException,
        current_batch_size: int,
        device: torch.device | str | None = None,
    ) -> int:
        """Halve the batch size on OOM or re-raise if recovery is not possible.

        Parameters
        ----------
        exc : BaseException
            Exception raised during execution.
        current_batch_size : int
            Batch size that triggered the failure.
        device : torch.device | str | None, optional
            Target device for clearing GPU cache, by default None

        Returns
        -------
        int
            Reduced batch size to retry with.
        """
        if (
            not self.oom_backoff
            or not _is_oom_error(exc)
            or current_batch_size <= self.min_batch_size
        ):
            raise exc
        _clear_gpu_memory(device)
        next_size = max(self.min_batch_size, current_batch_size // 2)
        logger.warning(
            f"Out of memory with batch_size={current_batch_size}; "
            f"backing off to batch_size={next_size} and retrying."
        )
        return next_size

    def execute(
        self,
        total_samples: int,
        batch_size: int,
        fn: Callable[[int, int], Any],
        device: torch.device | str | None = None,
    ) -> int:
        """Execute ``fn(start, end)`` across ``[0, total_samples)`` with OOM backoff.

        Parameters
        ----------
        total_samples : int
            Total number of items along the batch dimension.
        batch_size : int
            Initial batch size to use.
        fn : Callable[[int, int], Any]
            Callback accepting ``(start_index, end_index)`` for each sub-batch.
        device : torch.device | str | None, optional
            Target device for clearing GPU cache on OOM, by default None

        Returns
        -------
        int
            Final effective batch size after any OOM backoff adjustments.
        """
        if total_samples <= 0:
            return max(1, batch_size)
        current = max(self.min_batch_size, min(batch_size, total_samples))
        start = 0
        while start < total_samples:
            end = min(start + current, total_samples)
            try:
                fn(start, end)
            except BaseException as exc:
                current = self.backoff(exc, end - start, device)
                continue
            start = end
        return current


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

    Parameters
    ----------
    max_batch_size : int | None, optional
        Optional maximum sub-batch size along the packed ``batch`` dimension for
        non-generator calls, by default None
    autobatch : bool | AutoBatcher, optional
        Enable VRAM-aware automatic sub-batching and OOM backoff for non-generator
        DataArray calls, by default False

    Note
    ----
    A model attributes `input_coords` and `output_coords` must have "batch" as the
    coordinate system of the first dimensions. I.e. first key entry needs to be "batch".

    Note
    ----
    Decorated model methods declare explicit, named parameters. The decorator's
    variadic forwarding is an internal implementation detail and preserves the
    wrapped method's signature. For legacy tensor methods, positional arguments
    are explicit (x, CoordSystem) pairs and the result is a
    tuple[torch.Tensor, CoordSystem]. Keyword arguments pass through unchanged.

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

    def __init__(
        self,
        max_batch_size: int | None = None,
        autobatch: bool | AutoBatcher = False,
    ) -> None:
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        self.max_batch_size = max_batch_size
        self.autobatch = autobatch

    def __call__(self, func: F) -> Callable:
        """Decorate a model method with automatic leading-dimension batch packing.

        Parameters
        ----------
        func : F
            Model method or generator function to wrap.

        Returns
        -------
        Callable
            Wrapped callable handling batch compression and restoration.
        """
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
            total = int(packed.sizes["batch"])
            if self.max_batch_size is None and not self.autobatch:
                return restore(func(model, packed, *args, **kwargs))

            batcher = AutoBatcher.from_config(
                self.autobatch, self.max_batch_size
            ) or AutoBatcher(max_batch_size=self.max_batch_size, oom_backoff=False)
            device = _infer_device(model=model, x=(packed,))
            step_size = (
                batcher.estimate_batch_size(
                    model,
                    packed,
                    total_samples=total,
                    step_fn=lambda s: func(model, s, *args, **kwargs),
                    device=device,
                )
                if (self.autobatch and total > 1)
                else (self.max_batch_size or total)
            )
            if step_size >= total and not batcher.oom_backoff:
                return restore(func(model, packed, *args, **kwargs))

            outputs: dict[int, xr.DataArray] = {}

            def _run_slice(start: int, end: int) -> None:
                sub = packed.isel(batch=slice(start, end)).assign_coords(
                    batch=np.arange(end - start)
                )
                sub.attrs = dict(packed.attrs)
                out_sub = func(model, sub, *args, **kwargs)
                if not isinstance(out_sub, xr.DataArray):
                    raise TypeError("Batched model must return a DataArray")
                outputs[start] = out_sub.assign_coords(batch=np.arange(start, end))

            batcher.execute(total, step_size, _run_slice, device=device)
            ordered = [outputs[k] for k in sorted(outputs)]
            combined = (
                ordered[0] if len(ordered) == 1 else xr.concat(ordered, dim="batch")
            ).assign_coords(batch=np.arange(total))
            combined.attrs = dict(ordered[0].attrs)
            return restore(combined)

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
