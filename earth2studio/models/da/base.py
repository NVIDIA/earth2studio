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
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import xarray as xr

from earth2studio.utils.type import CoordSystem, FrameSchema

try:
    import cudf
except ImportError:
    cudf = None  # type: ignore[assignment, misc]


@runtime_checkable
class AssimilationModel(Protocol):
    """Data assimilation model interface"""

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

    def init_coords(self) -> tuple[FrameSchema | CoordSystem, ...] | None:
        """Initialization coordinate system required by the assimilation model.

        Specifies the coordinate system(s) for initial state data that must be provided
        before the model can process observations. The returned coordinate systems should
        match the expected input format for the first argument(s) passed to ``__call__``
        or sent to ``create_generator`` when initializing the model state.

        Returns
        -------
        tuple[FrameSchema | CoordSystem, ...] | None
            Tuple of coordinate systems or frame schemas defining the structure of
            required initialization data. Returns ``None`` if the model does not require
            initialization data (e.g., stateless models).
        """
        pass

    def input_coords(self) -> tuple[FrameSchema | CoordSystem, ...]:
        """Input coordinate system of assimilation model.

        For DataFrame inputs, this should return a PyArrow schema (or a wrapper
        containing schema and constraints). For tensor inputs, this should return
        a CoordSystem.

        Returns
        -------
        tuple[FrameSchema | CoordSystem, ...]
            Tuple of coordinate systems or frame schemas, one for each input argument
            that __call__ or create_generator accepts
        """
        pass

    def output_coords(
        self,
        input_coords: tuple[FrameSchema | CoordSystem, ...],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[FrameSchema | CoordSystem, ...]:
        """Output coordinate system of the assimilation model given an input coordinate
        system.

        Parameters
        ----------
        input_coords : tuple[FrameSchema | CoordSystem, ...]
            Input coordinate system tuple. FrameSchema (OrderedDict mapping field names
            to numpy arrays) for DataFrame inputs, or CoordSystem (OrderedDict mapping
            dimension names to coordinate arrays) for tensor inputs
        *args
            Additional positional arguments
        **kwargs
            Additional keyword arguments, typically including request metadata such as
            request_time and request_lead_time from DataFrame attrs

        Returns
        -------
        tuple[FrameSchema | CoordSystem, ...]
            Tuple of coordinate systems or frame schemas, one for each output argument
            that __call__ or create_generator returns
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
            Returns instance of prognostic
        """
        pass
