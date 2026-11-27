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

"""Wind speed diagnostic model that computes wind speed from u/v components.

This is a simple diagnostic model that does not require any checkpoints.
It computes wind speed at 10m (ws10m) from u10m and v10m components using
the formula: ws10m = sqrt(u10m^2 + v10m^2).
"""

import torch
import xarray as xr

from earth2studio.models.batch import batch_func
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem

INPUT_VARIABLES = ["u10m", "v10m"]
OUTPUT_VARIABLES = ["ws10m"]


class WindSpeed(torch.nn.Module):
    """Wind speed diagnostic that computes ws10m from u10m and v10m.

    A simple diagnostic model that computes wind speed at 10m from the
    u and v wind components. No checkpoints or external weights needed.

    Note
    ----
    Wind speed is computed as: ws10m = sqrt(u10m^2 + v10m^2)

    Badges
    ------
    region:global product:wind provider:nvidia backend:pytorch
    """

    stochastic: bool = False

    def __init__(self) -> None:
        super().__init__()

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the diagnostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature with a dynamic leading batch dimension,
            the u10m/v10m variables and the 0.25 degree global lat/lon grid.
        """
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": INPUT_VARIABLES},
            dynamic=("batch",),
            grid="latlon-0.25deg",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate the input and plan the output coordinate system.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature with the ws10m variable.
        """
        handshake_dataarray(input_coords, self.input_coords())
        return coord_array_like(input_coords, {"variable": OUTPUT_VARIABLES})

    def default_sources(self) -> None:
        """Recommend no input data source.

        Returns
        -------
        None
            No source recommendation; the caller supplies the input field.
        """
        return None

    @torch.inference_mode()
    @batch_func()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Compute wind speed from u/v components.

        Parameters
        ----------
        x : xr.DataArray
            Input field with dims (..., variable, lat, lon) where the variable
            coordinate is [u10m, v10m].

        Returns
        -------
        xr.DataArray
            Wind speed field matching ``output_coords(x)``.
        """
        handshake_nonempty(x)
        signature = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        # tensor shape: (batch, 2, lat, lon) after batch_func compression
        u = tensor[:, 0:1]
        v = tensor[:, 1:2]
        return from_torch(torch.sqrt(u**2 + v**2), signature)
