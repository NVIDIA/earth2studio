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

"""Persistence prognostic model that predicts future by returning current state.

A persistence forecast assumes conditions remain unchanged. Useful as a
simple baseline for weather prediction.
"""

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

VARIABLES = ["t2m", "msl", "z500", "t850", "u500", "v500"]


class PersistenceModel(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Persistence prognostic model.

    A baseline model that predicts the future by assuming the current state
    persists unchanged. Useful as a reference for forecast skill evaluation.

    Parameters
    ----------
    core_model : torch.nn.Module, optional
        Core model (not used, for interface compatibility).

    Note
    ----
    Persistence forecasts are a common baseline in weather prediction.
    """

    def __init__(self, core_model: torch.nn.Module | None = None) -> None:
        super().__init__()
        self.model = core_model
        self.register_buffer("device_buffer", torch.empty(0))
        self._time_step = np.timedelta64(24, "h")

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721, endpoint=True),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinates to validate and transform.

        Returns
        -------
        CoordSystem
            Output coordinates with updated lead_time.
        """
        target_input_coords = self.input_coords()

        handshake_dim(input_coords, "lead_time", 2)
        handshake_dim(input_coords, "variable", 3)
        handshake_dim(input_coords, "lat", 4)
        handshake_dim(input_coords, "lon", 5)

        handshake_coords(input_coords, target_input_coords, "variable")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "lon")

        output_coords = input_coords.copy()
        output_coords["lead_time"] = input_coords["lead_time"] + np.array(
            [self._time_step]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package.

        Returns
        -------
        Package
            Model package (empty for persistence model).
        """
        return Package(".")

    @classmethod
    def load_model(cls, package: Package) -> PrognosticModel:
        """Load prognostic model from package.

        Parameters
        ----------
        package : Package
            Model package (not used for persistence model).

        Returns
        -------
        PrognosticModel
            Loaded model instance.
        """
        return cls(core_model=None)

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor (persistence) and coordinates one time step ahead.
        """
        target_input_coords = self.input_coords()
        handshake_coords(coords, target_input_coords, "variable")
        handshake_dim(coords, "variable", 3)

        device = self.device_buffer.device
        x = x.to(device)

        out_coords = self.output_coords(coords)
        return x, out_coords

    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Batch-decorated generator for time integration.

        Parameters
        ----------
        x : torch.Tensor
            Initial condition tensor.
        coords : CoordSystem
            Initial coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and coordinates at each time step.
        """
        yield x, coords

        current_x = x
        current_coords = coords
        while True:
            current_x, current_coords = self.front_hook(current_x, current_coords)
            current_x, current_coords = self.__call__(current_x, current_coords)
            current_x, current_coords = self.rear_hook(current_x, current_coords)
            yield current_x, current_coords

    def create_iterator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Create time-integration iterator.

        Parameters
        ----------
        x : torch.Tensor
            Initial condition tensor.
        coords : CoordSystem
            Initial coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and coordinates at each time step.
        """
        yield from self._default_generator(x, coords)
