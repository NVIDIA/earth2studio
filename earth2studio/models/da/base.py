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

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd
import xarray as xr

from earth2studio.utils.type import CoordinateSystem, FrameSchema

if TYPE_CHECKING:
    from earth2studio.data.base import DataFrameSource, DataSource, ForecastSource

try:
    import cudf
except ImportError:
    cudf = None  # type: ignore[assignment, misc]


# --8<-- [start:assimilation-model-interface]
@runtime_checkable
class AssimilationModel(Protocol):
    """Data assimilation model interface

    Each initialization and input slot is a separate positional argument, in the
    order declared by ``init_coords()`` followed by ``input_coords()``. Observation
    slots are DataFrames described by a ``FrameSchema``; gridded slots are
    DataArrays described by a ``CoordinateSystem`` signature. Assimilation models define
    ``default_sources()``, returning a source for one slot, a tuple of sources or
    ``None`` per slot, or ``None`` for no recommendations. Drivers read it through
    ``earth2studio.models.utils.recommended_sources``.
    """

    def __call__(
        self,
        *args: pd.DataFrame | xr.DataArray | None,
    ) -> tuple[pd.DataFrame | xr.DataArray, ...]:
        """Stateless iteration for the data assimilation model.

        Processes observations and returns assimilated data without maintaining
        internal state between calls. This method is suitable for independent
        processing of observation batches.

        Parameters
        ----------
        *args : pd.DataFrame | xr.DataArray | None
            Variable number of observation arguments. Each argument can be a
            DataFrame (pandas or cudf DataFrame) or xarray DataArray
            containing observation data. None can be passed for optional
            arguments when no input data is available.

        Returns
        -------
        tuple[pd.DataFrame | xr.DataArray, ...]
            Assimilated data output. Can return a combination of DataFrames or
            xarray DataArrays depending on the particular model. Output is expect to be
            on the same device as the model.
        """

    def create_generator(
        self,
        *args: pd.DataFrame | xr.DataArray,
    ) -> Generator[
        tuple[pd.DataFrame | xr.DataArray, ...],
        tuple[pd.DataFrame | xr.DataArray | None, ...],
        None,
    ]:
        """Creates a generator which accepts collection of input observations and
        outputs a collection of assimilated data. Used for both stateless and stateful
        iterations of the data assimilation model

        The generator accepts observations (DataFrame or DataArray) via the send()
        method and yields assimilated data (DataFrame or DataArray) as output.
        Supports any number of arguments (variadic).

        Parameters
        ----------
        *args : pd.DataFrame | xr.DataArray
            Variable number of initialization arguments, if any are required by
            the model. Each argument can be a DataFrame (pandas or cudf
            DataFrame) or xarray DataArray containing initial state data.

        Yields
        ------
        tuple[pd.DataFrame | xr.DataArray, ...]
            Generator yields multiple arguments of assimilated data. Each argument
            can be a DataFrame (PyArrow Table or cudf DataFrame) or xarray DataArray.
            Supports any number of arguments.

        Receives
        --------
        tuple[pd.DataFrame | xr.DataArray | None, ...]
            Observations sent via generator.send() as multiple arguments. Each
            argument can be a DataFrame (PyArrow Table or cudf DataFrame) or xarray
            DataArray. None is sent initially to start the generator and can also be
            sent for iterations where no input data is available. Supports any number
            of arguments.

        Examples
        --------
        >>> generator = model.create_generator()
        >>> generator.send(None)  # Prime the generator
        >>> # Process observations over time
        >>> for obs in observations:
        ...     result = generator.send(obs)  # Send observations, receive assimilated data
        ...     # result is a tuple of DataFrames or DataArrays
        >>> generator.close()  # Clean up
        """
        pass

    def init_coords(self) -> tuple[FrameSchema | CoordinateSystem, ...] | None:
        """Initialization coordinate system of the assimilation model.

        Describes the initial state data the model needs before it can process
        observations, matching the first argument(s) passed to ``__call__`` or sent
        to ``create_generator`` when initializing the model state.

        Returns
        -------
        tuple[FrameSchema | CoordinateSystem, ...] | None
            Allocation-free DataArray initialization signatures, or frame schemas for
            DataFrame slots, in initialization-slot order. ``None`` for models
            without initialization data, such as stateless models.
        """
        pass

    def default_sources(
        self,
    ) -> (
        DataSource
        | ForecastSource
        | DataFrameSource
        | type[DataSource | ForecastSource | DataFrameSource]
        | tuple[
            DataSource
            | ForecastSource
            | DataFrameSource
            | type[DataSource | ForecastSource | DataFrameSource]
            | None,
            ...,
        ]
        | None
    ):
        """Recommended data sources for each initialization and input slot.

        Return one source directly for a single slot, or a tuple with one source
        per ``init_coords()`` slot, then one per ``input_coords()`` slot. Models
        without initialization slots (``init_coords()`` returns ``None``) only
        list their input slots. ``None`` means no recommendations for the model;
        a ``None`` tuple entry means no recommendation for that slot. Sources may
        be returned as classes rather than instances so the caller can construct
        them with its own arguments (for example a ``time_tolerance``). A source
        whose native grid differs from the slot's is returned composed with the
        recommended regridder; transforms intrinsic to the model, whatever the
        provider, stay inside the wrapper. Models never fetch from these
        themselves.

        Returns
        -------
        DataSource | ForecastSource | DataFrameSource | type[DataSource | ForecastSource | DataFrameSource] | tuple[DataSource | ForecastSource | DataFrameSource | type[DataSource | ForecastSource | DataFrameSource] | None, ...] | None
            A single source (instance or class), sources or ``None`` in
            initialization-then-input slot order, or no recommendations.
        """
        pass

    def input_coords(self) -> tuple[FrameSchema | CoordinateSystem, ...]:
        """Input coordinate system of the assimilation model.

        DataFrame inputs use frame schemas specifying fields and constraints.
        Gridded inputs use DataArray signatures.

        Returns
        -------
        tuple[FrameSchema | CoordinateSystem, ...]
            Allocation-free DataArray input signatures, or frame schemas for
            DataFrame slots, in input-slot order: one per input argument that
            ``__call__`` or ``create_generator`` accepts.
        """
        pass

    def output_coords(
        self,
        input_coords: tuple[FrameSchema | CoordinateSystem, ...],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[FrameSchema | CoordinateSystem, ...]:
        """Output coordinate system of the assimilation model.

        Parameters
        ----------
        input_coords : tuple[FrameSchema | CoordinateSystem, ...]
            Input signatures or frame schemas, one per ``input_coords()`` slot, in
            declared order.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments, typically including request metadata such as
            ``request_time`` and ``request_lead_time`` from DataFrame attributes.

        Returns
        -------
        tuple[FrameSchema | CoordinateSystem, ...]
            Allocation-free DataArray output signatures, or frame schemas for
            DataFrame slots, in output-slot order: one per output that ``__call__``
            or ``create_generator`` returns.

        Raises
        ------
        ValueError
            If the input coordinates are not valid.
        """
        pass

    def to(self, device: Any) -> AssimilationModel:
        """Moves assimilation model onto inference device, this is typically satisfied
        via `torch.nn.Module`.

        Parameters
        ----------
        device : Any
            Object representing the inference device, typically `torch.device` or str

        Returns
        -------
        AssimilationModel
            Returns instance of assimilation model
        """
        pass


# --8<-- [end:assimilation-model-interface]
