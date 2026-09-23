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

"""Native diagnostic skeleton; configure variables/grid before signature lookup.

Packaged diagnostics additionally inherit AutoModelMixin and define immutable
load_default_package/load_model methods. Load core weights on CPU, call eval(),
and use the optional-dependency decorator for the actual backend. See
PrecipitationAFNO for packaging and CorrDiff for a sample axis and changed grid.
"""

import torch
import xarray as xr

from earth2studio.grids import GridDefinition
from earth2studio.models.batch import batch_func
from earth2studio.utils.coords import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem


class SimpleDiagnostic(torch.nn.Module):
    """Compute wind speed on a configured grid without model weights."""

    def __init__(self, grid: GridDefinition | str = "latlon-0.25deg") -> None:
        super().__init__()
        self.grid = grid

    def input_coords(self) -> CoordinateSystem:
        """Declare fixed variables/grid and an explicit dynamic leading prefix."""
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": ["u10m", "v10m"]},
            dynamic=("batch",),
            grid=self.grid,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate geometry and replace variables without allocating field data."""
        handshake_dataarray(input_coords, self.input_coords())
        return coord_array_like(input_coords, {"variable": ["ws10m"]})

    @torch.inference_mode()
    @batch_func()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Return labelled wind speed on the input device."""
        signature = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        output = torch.sqrt(tensor[:, :1] ** 2 + tensor[:, 1:2] ** 2)
        return from_torch(output, signature)
