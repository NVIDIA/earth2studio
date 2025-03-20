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
    """

    def __init__(
        self,
        variable: str | list[str],
        domain_coords: CoordSystem,
        history: int = 1,  # TODO
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ):
        super().__init__()

        if isinstance(variable, str):
            variable = [variable]

        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(variable),
            }
        )

        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([np.timedelta64(6, "h")]),
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
            output_coords["lead_time"] + input_coords["lead_time"]
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

        return x, output_coords

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
            Coordinate system, should have dimensions ``[time, variable, *domain_dims]``

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
        coords = coords.copy()

        self.output_coords(coords)
        yield x, coords

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward is identity operator
            x, coords = self._forward(x, coords)

            # Rear hook
            x, coords = self.rear_hook(x, coords)
            yield x, coords

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
