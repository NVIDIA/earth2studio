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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd
import pyarrow as pa
import xarray as xr
from typing_extensions import Unpack

from earth2studio.utils.type import CoordSystem, LeadTimeArray, TimeArray

if TYPE_CHECKING:
    try:
        import cudf
    except ImportError:
        cudf = None  # type: ignore[assignment, misc]
else:
    try:
        import cudf
    except ImportError:
        cudf = None  # type: ignore[assignment, misc]


@dataclass
class AssimilationInput:
    """Assimilation model input
    This is a container input with various paramters that influence a data assimilation
    models task.
    """

    time: TimeArray = None
    lead_time: LeadTimeArray = None


# Type alias for DataFrame-like objects (PyArrow Table or cudf DataFrame)
if cudf is not None:
    DataFrame = pa.Table | cudf.DataFrame
else:
    DataFrame = pa.Table

@runtime_checkable
class AssimilationModel(Protocol):
    """Data assimilation model interface"""

    def __call__(
        self, x: AssimilationInput
    ) -> Generator[
        *DataFrame | xr.DataArray,
        *DataFrame | xr.DataArray,
        None,
    ]:
        """Creates a generator which accepts collection of input observations and
        outputs a collection of assimilated data.

        The generator accepts observations (DataFrame or DataArray) via the send()
        method and yields assimilated data (DataFrame or DataArray) as output.
        Supports any number of arguments (variadic).

        Parameters
        ----------
        x : AssimilationInput
            Input configuration for the assimilation model

        Yields
        ------
        *DataFrame | xr.DataArray
            Generator yields multiple arguments of assimilated data. Each argument
            can be a DataFrame (PyArrow Table or cudf DataFrame) or xarray DataArray.
            Supports any number of arguments.

        Receives
        --------
        *DataFrame | xr.DataArray
            Observations sent via generator.send() as multiple arguments. Each
            argument can be a DataFrame (PyArrow Table or cudf DataFrame) or xarray
            DataArray. None is sent initially to start the generator. Supports any
            number of arguments.
        """
        pass

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of assimilation model.

        For DataFrame inputs, this should return a PyArrow schema (or a wrapper
        containing schema and constraints). For tensor inputs, this should return
        a CoordSystem.

        Returns
        -------
        CoordSystem
            PyArrow schema for DataFrame inputs or coordinate system dictionary
            for tensor inputs
        """
        pass

    def output_coords(
        self, input_coords: CoordSystem, x: AssimilationInput
    ) -> pa.Schema | CoordSystem:
        """Output coordinate system of the assimilation model given an input coordinate
        system.

        Parameters
        ----------
        input_coords : pa.Schema | CoordSystem
            Input coordinate system (PyArrow schema for DataFrame inputs or CoordSystem
            for tensor inputs) to transform into output_coords
        x : AssimilationInput
            Input configuration for the assimilation model

        Returns
        -------
        pa.Schema | CoordSystem
            Output coordinate system (PyArrow schema for DataFrame outputs or CoordSystem
            for tensor outputs)

        Raises
        ------
        ValueError
            If input_coords are not valid
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