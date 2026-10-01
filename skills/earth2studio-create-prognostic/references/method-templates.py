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

"""Native history-method examples; see skeleton-template.py for a complete wrapper.

Keep loading methods model-specific: resolve immutable Package assets, load Torch
weights on CPU and call eval(). Override to() only for non-Torch state such as
ONNX sessions or JAX device placement. Never implement a tensor-pair public path.
"""

from collections.abc import Iterator
from copy import deepcopy

import numpy as np
import xarray as xr

from earth2studio.utils.coords import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.type import CoordinateSystem


def input_coords_with_history_template(self) -> CoordinateSystem:
    """Declare two frames on the instance's configured grid."""
    return coord_array(
        ("batch", "lead_time", "variable", "lat", "lon"),
        {"lead_time": np.array([-6, 0], dtype="timedelta64[h]"), "variable": ["t2m"]},
        dynamic=("batch",),
        grid=self.grid,
    )


def output_coords_template(self, x: CoordinateSystem) -> CoordinateSystem:
    """Validate history before planning a single forecast frame."""
    if "lead_time" not in x.coords:
        raise ValueError("lead_time is required")
    lead = x.lead_time.values
    if (
        x.lead_time.dims != ("lead_time",)
        or lead.size != 2
        or not np.issubdtype(lead.dtype, np.timedelta64)
        or np.isnat(lead).any()
    ):
        raise ValueError("Expected two finite timedelta history labels")
    handshake_dataarray(x.assign_coords(lead_time=lead - lead[-1]), self.input_coords())
    return coord_array_like(x, {"lead_time": lead[-1:] + np.timedelta64(6, "h")})


def create_iterator_template(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
    """Keep history internal and apply hooks in original leading dimensions."""
    self.output_coords(x)
    state = x.copy(deep=True)
    yield state.isel(lead_time=slice(-1, None)).copy(deep=True)
    while True:
        state = self.front_hook(state.copy(deep=True))
        output = self.rear_hook(self(state))
        history = state.isel(lead_time=slice(1, None)).drop_vars(
            [name for name in state.coords if name not in output.coords]
        )
        state = xr.concat(
            [history, output], dim="lead_time", coords="minimal", compat="override"
        )
        state = state.assign_coords(
            {
                name: coord.variable.copy(deep=True)
                for name, coord in output.coords.items()
                if "lead_time" not in coord.dims
            }
        )
        state.name = output.name
        for name, coord in output.coords.items():
            if "lead_time" in coord.dims:
                state.coords[name].attrs = deepcopy(coord.attrs)
        state.attrs = deepcopy(output.attrs)
        state.encoding = deepcopy(output.encoding)
        yield output.copy(deep=True)
