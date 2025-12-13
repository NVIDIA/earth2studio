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


class Persistence(torch.nn.Module, PrognosticMixin):
    """Persistence model that generates a forecast by applying the identity operator on
    the initial condition and indexing the lead time by 6 hours. Primarily used in
    testing.

    Parameters
    ----------
    variable : Union[str, List[str]]
        The variable or list of variables predicted by the model.
    domain_coords : CoordSystem
        The coordinates representing the domain for this model to operate on.
    history : int, optional
        Specifies the number of previous time steps to include as input, by default set
        to 1.
    dt : np.timedelta64, optional
        Time-step size of model between inputs and output, by default np.timedelta64(6, "h")
    """

    def __init__(
        self,
        variable: str | list[str],
        domain_coords: CoordSystem,
        history: int = 1,
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ):
        super().__init__()

        if isinstance(variable, str):
            variable = [variable]

        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array(
                    [np.timedelta64(-dt * i, "h") for i in reversed(range(history))]
                ),
                "variable": np.array(variable),
            }
        )
        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([dt]),
                "variable": np.array(variable),
            }
        )
        for key, value in domain_coords.items():
            self._input_coords[key] = value
            self._output_coords[key] = value

        self._history = history
        self._dt = dt

    def __str__(
        self,
    ) -> str:
        return "persistence"

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
        # Model is identity operator
        # Update coordinates
        output_coords = self.output_coords(coords)

        return x[:, -1:], output_coords

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
            Coordinate system, should have dimensions ``[time, lead_time, variable, *domain_dims]``

        Returns
        ------
        x : torch.Tensor
        coords : CoordSystem
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

            # Forward is identity operator
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
