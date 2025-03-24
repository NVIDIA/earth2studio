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
import functools
import inspect
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterator
from itertools import chain, islice
from typing import Any, TypeVar

import numpy as np
import torch

from earth2studio.utils.type import CoordSystem

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


class batch_func:
    """Batch utility decorator which can be added to prognostic and diagnostic models
    to help enable support for automatic batching of data. This class contains a
    decorator function which should be added to calls where this functionality is
    desired.

    Note
    ----
    A model attributes `input_coords` and `output_coords` must have "batch" as the
    coordinate system of the first dimensions. I.e. first key entry needs to be "batch".

    Example
    -------
    .. highlight:: python
    .. code-block:: python

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
    """

    def __call__(self, func: F) -> Callable:
        if inspect.isgeneratorfunction(func):
            return self._batch_wrap_generator(func)
        return self._batch_wrap(func)

    def _compress_batch(
        self, model: Any, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem, CoordSystem, torch.Size]:
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
        if (
            next(iter(input_coords)) != "batch"
            and next(iter(model.output_coords(input_coords))) != "batch"
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

    def _decompress_batch(
        self,
        out: torch.Tensor,
        out_coords: CoordSystem,
        batched_coords: CoordSystem,
        batched_shape: torch.Size,
    ) -> tuple[torch.Tensor, CoordSystem]:
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
            model: Any,
            x: torch.Tensor,
            coords: CoordSystem,
        ) -> tuple[torch.Tensor, CoordSystem]:

            x, flatten_coords, batched_coords, batched_shape = self._compress_batch(
                model, x, coords
            )

            # Model forward
            out, out_coords = func(model, x, flatten_coords)
            out, out_coords = self._decompress_batch(
                out, out_coords, batched_coords, batched_shape
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


class batch_coords:
    """Batch utility decorator which can be added to prognostic and diagnostic
    output_coords methods to help enable support for automatic batching of data.
    This class contains a decorator function which should be added to output_coord
    calls where this functionality is desired.

    Note
    ----
    `input_coords` and `output_coords` must have "batch" as the
    coordinate system of the first dimensions. I.e. first key entry needs to be "batch".
    """

    def __call__(self, func: F) -> Callable:
        return self._batch_wrap(func)

    def _compress_batch(
        self, model: Any, coords: CoordSystem
    ) -> tuple[CoordSystem, CoordSystem]:
        """Compresses dimensions into the models batch dimension

        Parameters
        ----------
        model : Any
            Any object, prognostic / diagnostic model that has a input_coords property
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[ CoordSystem, CoordSystem ]
            Returns batch compressed coords and the coords of the batch
            dimensions.

        Raises
        ------
        ValueError
            If model's input_coords do not contain the batch dimension
        """
        input_coords = model.input_coords()
        if next(iter(input_coords)) != "batch":
            raise ValueError(
                "Model input coordinate systems not compatible with batch processing"
            )

        flatten_coords: CoordSystem
        batched_coords: CoordSystem
        # If dims of input is one less than input coords, just prepend batch dim
        if len(coords) == len(input_coords) - 1:
            flatten_coords = coords.copy()
            flatten_coords.update({"batch": np.array([0])})
            flatten_coords.move_to_end("batch", last=False)
            return flatten_coords, OrderedDict({})

        i = len(coords) - len(input_coords.keys()) + 1
        # Prep coordinate dicts
        batched_coords = OrderedDict(islice(coords.items(), 0, i))
        flatten_coords = OrderedDict(islice(coords.items(), i, None))
        flatten_coords.update({"batch": np.empty(0)})
        flatten_coords.move_to_end("batch", last=False)
        # Flatten batch dims
        flatten_coords["batch"] = np.arange(len(next(iter(coords.values()))))

        return flatten_coords, batched_coords

    def _decompress_batch(
        self,
        out_coords: CoordSystem,
        batched_coords: CoordSystem,
    ) -> CoordSystem:
        """Decompresses the batch dimension of a tensor

        Parameters
        ----------
        out_coords : CoordSystem
            Compressed coordinates
        batched_coords : CoordSystem
            The coords of the batch dimensions

        Returns
        -------
        CoordSystem
            Uncompressed coordinates
        """

        # Reconstruct batch dims
        out_coords = out_coords.copy()
        del out_coords["batch"]
        out_coords = OrderedDict(chain(batched_coords.items(), out_coords.items()))
        return out_coords

    def _batch_wrap(self, func: Callable) -> Callable:
        """Standard batch function decorator"""

        # TODO: Better typing for model object
        @functools.wraps(func)
        def _wrapper(model: Any, input_coords: CoordSystem) -> CoordSystem:

            flatten_coords, batched_coords = self._compress_batch(model, input_coords)

            # Model forward
            out_coords = func(model, flatten_coords)
            out_coords = self._decompress_batch(out_coords, batched_coords)
            return out_coords

        return _wrapper
