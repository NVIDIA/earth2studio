# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np
import torch

from earth2studio.models.batch import batch_func
from earth2studio.utils.type import CoordSystem


class Identity(torch.nn.Module):
    """Identity diagnostic that is coordinate insensitive. Typically used for testing."""

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "identity"

    @property
    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model, time dimension should contain
        time-delta objects

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict({"batch": np.empty(1)})

    @property
    def output_coords(self) -> CoordSystem:
        """Ouput coordinate system of diagnostic model, time dimension should contain
        time-delta objects

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict({"batch": np.empty(1)})

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = coords.copy()
        return x, output_coords
