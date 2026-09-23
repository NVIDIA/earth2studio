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

Generative outputs must declare their sample axis and output grid explicitly; use
CorrDiff's native implementation as the reference. When migrating an existing
wrapper, retain its seed API, sampler progression and RNG ownership; RNG contract
changes are separate follow-up work.
"""

import torch
import xarray as xr

from earth2studio.models.batch import batch_func
from earth2studio.utils.coords import coord_array_like, handshake_dataarray
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
    signature = self.output_coords(x)
    tensor, _ = x.e2s.to_torch()
    tensor = (tensor.to(self.center.device) - self.center) / self.scale
    output = self.core_model(tensor)
    return from_torch(output, signature)
