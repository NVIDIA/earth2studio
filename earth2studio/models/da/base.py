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
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import pyarrow as pa
import xarray as xr

from earth2studio.utils.type import CoordSystem, LeadTimeArray, TimeArray


@dataclass
class AssimilationInput:
    """Assimilation model input
    This is a container input with various paramters that influence a data assimilation
    models task.
    """

    time: TimeArray = None
    lead_time: LeadTimeArray = None


@runtime_checkable
class AssimilationModel(Protocol):
    """Data assimilation model interface"""

    def __call__(self, x: AssimilationInput) -> Generator[
        tuple[pd.DataFrame | xr.DataArray, ...],
        tuple[pd.DataFrame | xr.DataArray, ...],
        None,
    ]:
        """Creates a generator which accepts collection of input observations and
        outputs a collection of assimilated data.

        Parameters
        ----------
        x : AssimilationInput
            Input configuration for the assimilation model

        Yields
        ------
        tuple[*pd.DataFrame | xr.DataArray]
            Generator that yields assimilated data as tuples of pandas DataFrames
            or xarray DataArrays.
        """
        pass

    def input_coords(self) -> pa.Schema | CoordSystem:
        """Input coordinate system of assimilation model.

        For DataFrame inputs, this should return a PyArrow schema (or a wrapper
        containing schema and constraints). For tensor inputs, this should return
        a CoordSystem.

        Returns
        -------
        pa.Schema | CoordSystem
            PyArrow schema for DataFrame inputs or coordinate system dictionary
            for tensor inputs
        """
        pass

    def output_coords(
        self, input_coords: pa.Schema | CoordSystem, x: AssimilationInput
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
