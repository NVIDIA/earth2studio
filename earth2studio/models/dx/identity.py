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

import torch
import xarray as xr

from earth2studio.utils import coord_array, coord_array_like, handshake_nonempty
from earth2studio.utils.type import CoordinateSystem


class Identity(torch.nn.Module):
    """Identity diagnostic that is coordinate insensitive. Typically used for testing.

    Badges
    ------
    region:global provider:nvidia backend:pytorch
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "identity"

    def input_coords(self) -> CoordinateSystem:
        """Return an allocation-free signature accepting arbitrary dimensions."""
        return coord_array(("batch",), dynamic=("batch",))

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Return the input coordinates without allocating field storage."""
        return coord_array_like(input_coords)

    @torch.inference_mode()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Return the labelled field, preserving its device and metadata."""
        handshake_nonempty(x)
        return x.copy(deep=False)
