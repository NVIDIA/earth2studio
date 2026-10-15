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

from __future__ import annotations

from collections.abc import Callable

import torch
import xarray as xr
from loguru import logger

from earth2studio.perturbation._array import (
    amplitude_like,
    array_module,
    as_host_or_cupy,
)
from earth2studio.perturbation.base import Perturbation
from earth2studio.perturbation.brown import Brown


class BredVector:
    """Bred Vector perturbation method, a classical technique for pertubations in
    ensemble forecasting.

    Parameters
    ----------
    model : Callable[[xr.DataArray], xr.DataArray]
        Dynamical model that maps a DataArray to a DataArray of the same shape,
        typically this is the prognostic AI model.
        TODO: Update to prognostic looper
    noise_amplitude : float | Tensor, optional
        Noise amplitude, by default 0.05. If a tensor,
        this must be broadcastable with the input data.
    integration_steps : int, optional
        Number of integration steps to use in forward call, by default 20
    ensemble_perturb : bool, optional
        Perturb the ensemble in an interacting fashion, by default False
    seeding_perturbation_method : Perturbation, optional
        Method to seed the Bred Vector perturbation, by default Brown Noise

    Note
    ----
    For additional information:

    - https://journals.ametsoc.org/view/journals/bams/74/12/1520-0477_1993_074_2317_efantg_2_0_co_2.xml
    - https://en.wikipedia.org/wiki/Bred_vector
    """

    def __init__(
        self,
        model: Callable[[xr.DataArray], xr.DataArray],
        noise_amplitude: float | torch.Tensor = 0.05,
        integration_steps: int = 20,
        ensemble_perturb: bool = False,
        seeding_perturbation_method: Perturbation = Brown(),
    ):
        self.model = model
        self.noise_amplitude = (
            noise_amplitude
            if isinstance(noise_amplitude, torch.Tensor)
            else torch.Tensor([noise_amplitude])
        )
        self.ensemble_perturb = ensemble_perturb
        self.integration_steps = integration_steps
        self.seeding_perturbation_method = seeding_perturbation_method

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Apply perturbation method

        Parameters
        ----------
        x : xr.DataArray
            xarray DataArray intended to apply perturbation on

        Returns
        -------
        xr.DataArray
            Perturbed DataArray with the coordinates of the input
        """
        if "lead_time" in x.dims and x.sizes["lead_time"] > 1:
            logger.warning(
                "Input data / models that require multiple lead times may lead to unexpected behavior"
            )

        x = as_host_or_cupy(x)
        xp = array_module(x)
        x0 = x.data
        noise_amplitude = amplitude_like(self.noise_amplitude, xp, x0.dtype)

        dx = self.seeding_perturbation_method(x.copy(deep=True)).data - x0

        xd = self.model(x.copy(deep=True)).data
        # Run forward model
        for _ in range(self.integration_steps):
            x2 = self.model(x.copy(data=x0 + dx)).data
            if self.ensemble_perturb:
                dx1 = x2 - xd
                dx = dx1 + noise_amplitude * (dx - dx.mean(axis=0))
            else:
                dx = x2 - xd

        gamma = xp.linalg.norm(x0.ravel()) / xp.linalg.norm((x0 + dx).ravel())
        return x.copy(data=x0 + dx * noise_amplitude * gamma)
