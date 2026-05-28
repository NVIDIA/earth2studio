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

from collections import OrderedDict
from collections.abc import Generator, Iterator

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class ConstantModel(torch.nn.Module, PrognosticMixin):
    """Prognostic model that returns a constant value of 273.15 for all grid points
    and variables at every time step. Primarily useful for testing.

    The model takes variables ['t2m', 'u10m', 'v10m'] on a 0.25-degree global grid
    (721x1440) and ignores input data entirely, outputting a constant field.

    Parameters
    ----------
    constant_value : float, optional
        The constant value to fill the output with, by default 273.15
    """

    def __init__(self, constant_value: float = 273.15):
        super().__init__()
        self._dt = np.timedelta64(6, "h")
        self._constant_value = constant_value

        lat = np.linspace(90.0, -90.0, 721)
        lon = np.linspace(0.0, 360.0, 1440, endpoint=False)
        variables = np.array(["t2m", "u10m", "v10m"])

        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": variables,
                "lat": lat,
                "lon": lon,
            }
        )
        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([self._dt]),
                "variable": variables,
                "lat": lat,
                "lon": lon,
            }
        )

    def __str__(self) -> str:
        return "constant_model"

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self._input_coords.copy()

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        output_coords = self._output_coords.copy()

        if input_coords is None:
            return output_coords

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key != "batch":
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)

        output_coords["batch"] = input_coords["batch"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"][-1]
        )
        return output_coords

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)

        # Return constant value regardless of input
        batch_size = x.shape[0]
        n_vars = len(output_coords["variable"])
        n_lat = len(output_coords["lat"])
        n_lon = len(output_coords["lon"])

        out = torch.full(
            (batch_size, 1, n_vars, n_lat, n_lon),
            self._constant_value,
            dtype=x.dtype,
            device=x.device,
        )

        return out, output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system
        """
        return self._forward(x, coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        self.output_coords(coords.copy())
        coords_out = coords.copy()
        coords_out["lead_time"] = coords["lead_time"][-1:]
        yield x[:, -1:], coords_out

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward pass — return constant
            x_out, coords_out = self._forward(x, coords)

            # Rear hook
            x_out, coords_out = self.rear_hook(x_out, coords_out)

            coords["lead_time"] = np.concatenate(
                [coords["lead_time"][1:], coords_out["lead_time"]]
            )
            x = torch.cat([x[:, 1:], x_out], dim=1)

            yield x_out, coords_out

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)
