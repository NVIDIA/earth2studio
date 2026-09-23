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

"""Native one-frame prognostic template; adapt core shapes, grid and package assets.

For multi-frame history, retain the complete rolling state but yield only its last
frame initially. See Persistence and FCN for checkpoint continuation, and DLWP for
multi-output hook cadence. Keep optional imports behind OptionalDependencyFailure
and apply check_optional_dependencies to the packaged loader when appropriate.
"""

from collections.abc import Iterator

import numpy as np
import torch
import xarray as xr

from earth2studio.grids import GridDefinition
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils.coords import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem


class ModelName(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
    """Forecast one configured field frame with a Torch core.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core accepting and returning (batch, variable, lat, lon).
    grid : GridDefinition | str
        Configured lat/lon grid, fixed throughout a rollout.
    """

    def __init__(self, core_model: torch.nn.Module, grid: GridDefinition | str) -> None:
        super().__init__()
        self.model = core_model
        self.grid = grid
        self.register_buffer("device_buffer", torch.empty(0))
        self._time_step = np.timedelta64(6, "h")

    def input_coords(self) -> CoordinateSystem:
        """Declare the fixed frame and grid with arbitrary leading dimensions."""
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {"lead_time": np.array([0], dtype="timedelta64[h]"), "variable": ["t2m"]},
            dynamic=("batch",),
            grid=self.grid,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate relative history and plan the next absolute lead time."""
        if "lead_time" not in input_coords.coords:
            raise ValueError("lead_time is required")
        lead = input_coords.lead_time.values
        if (
            input_coords.lead_time.dims != ("lead_time",)
            or lead.size != 1
            or not np.issubdtype(lead.dtype, np.timedelta64)
            or np.isnat(lead).any()
        ):
            raise ValueError("lead_time must contain one finite timedelta")
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(
            input_coords, {"lead_time": lead[-1:] + self._time_step}
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Replace this URL with an immutable checkpoint revision."""
        return Package(
            "hf://organization/model@commit",
            cache_options={"cache_storage": Package.default_cache("model_name")},
        )

    @classmethod
    def load_model(cls, package: Package) -> "ModelName":
        """Load the actual core and its configured grid from package assets."""
        core = torch.load(
            package.resolve("model.pt"), map_location="cpu", weights_only=False
        )
        core.eval()
        return cls(core, grid="latlon-0.25deg")

    @torch.inference_mode()
    @batch_func()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Advance a field without invoking iterator hooks or modifying input."""
        signature = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        tensor = tensor.to(self.device_buffer.device).clone()
        output = self.model(tensor[:, 0]).unsqueeze(1)
        return from_torch(output, signature)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the initial frame, then forecasts with original-dimension hooks."""
        self.output_coords(x)
        state = x.copy(deep=True)
        yield state.isel(lead_time=slice(-1, None)).copy(deep=True)
        while True:
            state = self.front_hook(state.copy(deep=True))
            state = self.rear_hook(self(state))
            yield state.copy(deep=True)
