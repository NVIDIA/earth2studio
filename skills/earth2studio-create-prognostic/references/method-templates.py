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

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class HistoryState:
    """Serializable older frame, excluding the current public forecast.

    Parameters
    ----------
    history : xr.DataArray
        One older frame in original leading dimensions.
    """

    history: xr.DataArray


def initialize_template(self, x: xr.DataArray) -> tuple[xr.DataArray, HistoryState]:
    """Compute from the full initial window and retain only its latest input.

    Bind as ``initialize`` with the coordinate methods above and
    ``advance_history_template`` as ``_advance_history``.
    """
    self.output_coords(x)
    return self._advance_history(x), HistoryState(
        x.isel(lead_time=[-1]).copy(deep=True)
    )


def step_template(
    self, y: xr.DataArray, state: HistoryState
) -> tuple[xr.DataArray, HistoryState]:
    """Rebuild the two-frame input window without modifying the checkpoint pair.

    Select input variables by label. A model adding diagnostic channels must
    extend the expected output signature too. History and the new frame must
    continue at six-hour cadence.
    """
    expected = coord_array_like(
        state.history,
        {"lead_time": state.history.lead_time.values + np.timedelta64(6, "h")},
    )
    handshake_dataarray(y, expected)
    signature = self.input_coords()
    latest = y.sel(variable=signature.coords["variable"].values).isel(lead_time=[-1])
    # Retain history only where coordinates still apply to the current payload.
    history = state.history.drop_vars(
        [name for name in state.history.coords if name not in latest.coords]
    )
    window = xr.concat(
        [history, latest],
        dim="lead_time",
        coords="minimal",
        compat="override",
        join="exact",
    )
    window.attrs = latest.attrs.copy()
    window.name = latest.name
    window.encoding = latest.encoding.copy()
    self.output_coords(window)
    return self._advance_history(window), HistoryState(latest.copy(deep=True))


@torch.inference_mode()
@batch_func()
def advance_history_template(self, x: xr.DataArray) -> xr.DataArray:
    """Execute a core mapping two history frames to one forecast frame.

    Bind as ``_advance_history``. The core consumes (batch, history, variable,
    lat, lon) and returns (batch, variable, lat, lon). Batch only this numerical
    helper, so state and iterator hooks retain original leading dimensions.
    """
    handshake_nonempty(x)
    signature = self.output_coords(x)
    tensor, _ = x.e2s.to_torch()
    output = self.model(tensor.to(self.device_buffer.device).clone()).unsqueeze(1)
    return from_torch(output, signature)


def create_iterator_template(self, x: xr.DataArray) -> Generator[
    xr.DataArray | tuple[xr.DataArray, ...],
    xr.DataArray | tuple[xr.DataArray, ...] | None,
    None,
]:
    """Delegate forecasts-only iteration; history lives in returned state."""
    return self._default_create_iterator(x)
