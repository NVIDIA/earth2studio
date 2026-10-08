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

"""Native packaged diagnostic methods; see skeleton-template.py for a simple model.

Generative outputs declare their sample axis and output grid explicitly. Follow
model-contract.md for set_rng and isolated RNG ownership. The multi-grid example
below demonstrates fixed parameters for execution and coordinate planning.
"""

import torch
import xarray as xr

from earth2studio.grids import GridDefinition
from earth2studio.models.batch import batch_func
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem


def output_coords_template(self, input_coords: CoordinateSystem) -> CoordinateSystem:
    """Plan output variables while preserving leading axes and grid metadata."""
    handshake_dataarray(input_coords, self.input_coords())
    return coord_array_like(input_coords, {"variable": self.output_variables})


@torch.inference_mode()
@batch_func()
def automodel_call_template(self, x: xr.DataArray) -> xr.DataArray:
    """Normalize, execute a Torch core, and restore labelled output metadata."""
    handshake_nonempty(x)
    signature = self.output_coords(x)
    tensor, _ = x.e2s.to_torch()
    tensor = (tensor.to(self.center.device) - self.center) / self.scale
    output = self.core_model(tensor)
    return from_torch(output, signature)


class MultiGridDiagnostic(torch.nn.Module):
    """Copy fields on two different grids, preserving each slot's leading axes.

    Parameters
    ----------
    fine_grid, coarse_grid : GridDefinition | str
        Distinct configured lat/lon grids for the two slots.
    """

    stochastic: bool = False

    def __init__(
        self, fine_grid: GridDefinition | str, coarse_grid: GridDefinition | str
    ) -> None:
        super().__init__()
        self.grids = (fine_grid, coarse_grid)

    def input_coords(self) -> tuple[CoordinateSystem, CoordinateSystem]:
        """Declare fine-grid and coarse-grid temperature inputs, in that order."""
        fine, coarse = (
            coord_array(
                ("batch", "variable", "lat", "lon"),
                {"variable": ["t2m"]},
                dynamic=("batch",),
                grid=grid,
            )
            for grid in self.grids
        )
        return fine, coarse

    def output_coords(
        self, fine: CoordinateSystem, coarse: CoordinateSystem
    ) -> tuple[CoordinateSystem, CoordinateSystem]:
        """Validate separate signatures and plan one output per distinct grid."""
        fine_signature, coarse_signature = self.input_coords()
        handshake_dataarray(fine, fine_signature)
        handshake_dataarray(coarse, coarse_signature)
        return coord_array_like(fine), coord_array_like(coarse)

    def default_sources(self) -> None:
        """Recommend no providers for either slot."""
        return None

    def __call__(
        self, fine: xr.DataArray, coarse: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Copy the two input fields on their respective devices.

        Parameters
        ----------
        fine, coarse : xr.DataArray
            Separate positional fields, in ``input_coords()`` order.

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Independent field copies on the fine and coarse grids.
        """
        self.output_coords(fine, coarse)
        handshake_nonempty(fine)
        handshake_nonempty(coarse)
        return fine.copy(deep=True), coarse.copy(deep=True)
