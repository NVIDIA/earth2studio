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

import numpy as np
import torch


class Zero:
    """No perturbation scheme

    Primarily used for deterministic runs in ensemble workflows
    """

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """Apply perturbation method

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply perturbation on
        coords : dict[str, np.ndarray]
            Ordered dict representing coordinate system that describes the tensor

        Returns
        -------
        tuple[torch.Tensor, dict[str, np.ndarray]]:
            Output tensor and respective coordinate system dictionary
        """
        return x, coords
