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

from typing import Optional

import numpy as np
import torch

from earth2studio.perturbation.base import PerturbationMethod
from earth2studio.utils import handshake_dim
from earth2studio.utils.type import CoordSystem


class Perturbation:
    """Applies a perturbation method to an input tensor. If supplied this will filter
    the variable dimension of the input to a subset to apply the specified perturbation.
    Additionally, normalization vectors can be supplied to normalize the data prior to
    applying perturbations.

    Note
    ----
    It is a core design principle of Earth2Studio to always move data with physical
    units between components. It is very likely users should provide normalization
    arrays.

    Note
    ----
    Presently this class is strict, requiring the last three dimensions of the input
    coordinate system to be ['variable', 'lat', 'lon']. Further generalization may be
    considered in the future as the use case arises.

    Parameters
    ----------
    method : PerturbationMethod
        Perturbation method
    variables : Optional[list[str]], optional
        List of variable id's to apply petrubation on. If None, perturbation will be
        applied to all variables, by default None
    center : Optional[np.ndarray], optional
        Variable center / mean array. If None, no center will be used, by default None
    scale : Optional[np.ndarray], optional
        Variable scale / std array. If None, no scale will be used,, by default None
    """

    def __init__(
        self,
        method: PerturbationMethod,
        variables: Optional[list[str]] = None,
        center: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ):
        self.method = method
        self.variables = variables

        if center is None and scale is None:
            center = np.array([0])
            scale = np.array([1])
        elif center is None:
            center = np.zeros_like(scale)
        elif scale is None:
            scale = np.ones_like(center)

        self.center = torch.Tensor(center)
        self.scale = torch.Tensor(scale)

        if self.center.ndim != 1 and self.scale.ndim != 1:
            raise ValueError("Only 1D normalization vectors supported for fields")
        if self.center.shape != self.scale.shape:
            raise ValueError("Center and scale arrays must be the same dimensionality")

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Applies perturbation method to input tensor

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Ordered dict representing coordinate system that describes the tensor

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Tensor with applied perturbation, coordinate system
        """
        # Check the required dimensions are present
        handshake_dim(coords, required_dim="variable", required_index=-3)
        handshake_dim(coords, required_dim="lat", required_index=-2)
        handshake_dim(coords, required_dim="lon", required_index=-1)
        # Filter variables
        if self.variables:
            vindex = torch.IntTensor(
                [coords["variable"].index(i) for i in self.variables]
            ).to(x.device)
            x0 = x[..., vindex, :, :].contiguous()
        else:
            x0 = x

        # Normalize
        center = self.center.to(x.device).unsqueeze(-1).unsqueeze(-1)
        scale = self.scale.to(x.device).unsqueeze(-1).unsqueeze(-1)
        x0 = (x0 - center) / scale

        # Compute noise
        noise, coords = self.method(x0, coords)
        # Apply noise and unnormalize
        x0 = scale * (x0 + noise) + center

        # Apply variable perturbation
        if self.variables:
            x[..., vindex, :, :] = x0
        else:
            x = x0

        return x, coords
