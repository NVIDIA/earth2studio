# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from datetime import datetime

import numpy as np
import torch

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation.base import Perturbation
from earth2studio.utils import handshake_dim, handshake_size
from earth2studio.utils.time import to_time_array
from earth2studio.utils.type import CoordSystem


class HemisphericCentredBredVector:
    """Bred Vector perturbation method, following the approach introduced in
    'Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier
    Neural Operators'. The vector is bred by advancing in time. The bred vector is
    scaled seperately in the northern and southern extra-tropics and interpolated in the
    tropics. Additionally, it applies a centred perturbation, ie generating two
    perturbed stated by adding and subtracting the bred vector, respecitvely.


    Parameters
    ----------
    model : PrognosticModel
        Dynamical model, typically this is the prognostic AI model
    data : DataSource
        data source for obtaining warmup time steps
    time : datetime
        Time from which initial conditions are taken
    seeding_perturbation_method : Perturbation, optional
        Method to seed the Bred Vector perturbation
    noise_amplitude : float | Tensor, optional
        Noise amplitude, by default 0.05. If a tensor, this must be broadcastable with
        the input data
    integration_steps : int, optional
        Number of integration steps to use in forward call, by default 20

    Note
    ----
    For additional information:

    - https://arxiv.org/abs/2408.03100
    - https://journals.ametsoc.org/view/journals/bams/74/12/1520-0477_1993_074_2317_efantg_2_0_co_2.xml
    - https://en.wikipedia.org/wiki/Bred_vector
    """

    def __init__(
        self,
        model: PrognosticModel,
        data: DataSource,
        time: datetime,
        seeding_perturbation_method: Perturbation,
        noise_amplitude: float | torch.Tensor = 0.35,
        integration_steps: int = 3,
    ) -> None:
        self.model = model
        self.data = data
        self.noise_amplitude = (
            noise_amplitude
            if isinstance(noise_amplitude, torch.Tensor)
            else torch.Tensor(
                [noise_amplitude] * len(self.model.input_coords()["variable"])
            )
        )
        self.integration_steps = integration_steps
        self.seeding_perturbation_method = seeding_perturbation_method

        self.fetch_warmup_ics(time)  # make work with ic_block
        self.set_clip_indices()

    def fetch_warmup_ics(self, time: datetime) -> None:
        """Method for obtaining ICs for warmup steps

        Parameters
        ----------
        time : datetime
            Time from which initial conditions are taken
        """

        inco = self.model.input_coords()

        time = to_time_array([time])
        self.warmup_times = (
            time
            + np.arange(-self.integration_steps, 1)
            * self.model.output_coords(inco)["lead_time"]
        )

        self.ics, _ = fetch_data(
            source=self.data,
            time=self.warmup_times,
            variable=inco["variable"],
            lead_time=inco["lead_time"],
            device="cpu",
        )
        return

    def set_clip_indices(self) -> None:
        """If humidity and tcwv in variable set, add to list of variables to clip"""
        self.clip_idcs = []
        for ii, var in enumerate(self.model.input_coords()["variable"]):
            if var[0] == "q" or var == "tcwv" or var[0] == "r" or var[:2] == "tp":
                self.clip_idcs.append(ii)
        return

    def hemispheric_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate norm of fields speperately in northern and southern extra-tropics

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to apply norm to
        """
        nvar, nlat, nlon = x.shape[-3:]
        ex_tropic = int(
            70 / 180 * nlat
        )  # using 70deg for extra-tropics like in hens publication
        in_tropic = nlat - 2 * ex_tropic

        jac = torch.sin(torch.linspace(0, torch.pi, nlat))[:, None].to(self.device)
        weights = (torch.pi / nlat) * (2 * torch.pi / nlon) * jac
        weights = weights / weights.mean()

        # compute norms on northern and southern hemisphere
        north = torch.sqrt(
            torch.mean(
                x[..., :ex_tropic, :] ** 2 * weights[:ex_tropic,],
                dim=(-2, -1),
            )
        )
        south = torch.sqrt(
            torch.mean(x[..., -ex_tropic:, :] ** 2 * weights[-ex_tropic:], dim=(-2, -1))
        )

        # interpolate in tropics
        tropic = torch.zeros(north.shape[0], 1, 1, nvar, in_tropic).to(self.device)
        for batch in range(north.shape[0]):
            for var in range(nvar):
                tropic[batch, 0, 0, var] = torch.linspace(
                    north[batch, 0, 0, var], south[batch, 0, 0, var], in_tropic
                )

        # expand and concatenate
        north = north[..., None].repeat(1, 1, 1, 1, ex_tropic)
        south = south[..., None].repeat(1, 1, 1, 1, ex_tropic)

        return torch.cat([north, tropic, south], dim=-1)[..., None]

    def force_non_neg(self, x: torch.Tensor) -> torch.Tensor:
        """Clip variables to non-negative values

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        """
        nn = x[:, :, :, self.clip_idcs, ...]
        nn = torch.where(nn < 0.0, 0.0, nn)
        x[:, :, :, self.clip_idcs, ...] = nn
        return x

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Apply perturbation method

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply perturbation on
        coords : CoordSystem
            Ordered dict representing coordinate system that describes the tensor.
            Must contain coordinates (Any, "time", "lead_time", "variable", "lat",
            "lon"). Time and lead_time must have size 1.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]:
            Output tensor and respective coordinate system dictionary
        """
        shape = x.shape
        # Check the required dimensions are present
        handshake_dim(coords, required_dim="time", required_index=1)
        handshake_dim(coords, required_dim="lead_time", required_index=2)
        handshake_dim(coords, required_dim="variable", required_index=3)
        handshake_dim(coords, required_dim="lat", required_index=4)
        handshake_dim(coords, required_dim="lon", required_index=5)
        handshake_size(coords, required_dim="time", required_size=1)
        handshake_size(coords, required_dim="lead_time", required_size=1)
        if len(shape) != 6:
            raise ValueError("Input tensor and coords need 6 dimensions")
        if shape[0] % 2 != 0:
            raise ValueError(
                "batch size and ensemble size have to be even for centred perturbation"
            )

        self.device = x.device
        self.noise_amplitude = self.noise_amplitude.to(self.device)

        # get unperturbed intital state

        xunp = (
            self.ics[:1]
            .repeat(shape[0] // 2, *[1 for _ in range(len(shape) - 1)])
            .to(self.device)
        )

        # generate perturbed initial state
        xper, coords = self.seeding_perturbation_method(xunp, coords)

        for ii in range(self.integration_steps):
            xunp, _ = self.model(xunp, coords)
            xper, _ = self.model(xper, coords)

            dx = xper - xunp
            hem_norm = self.hemispheric_norm(dx)

            # if zero elements are requierd, replace NaNs in scaled dx with 0
            if (hem_norm == 0).any():
                raise ValueError(
                    "zero element in hemispheric norm, maybe noise amplification too small?"
                )

            # scale dx
            dx = self.noise_amplitude[:, None, None] * (dx / hem_norm)

            xunp = (
                self.ics[ii + 1]
                .unsqueeze(dim=0)
                .repeat(shape[0] // 2, 1, 1, 1, 1, 1)
                .to(self.device)
            )
            xper = xunp + dx

        xper = self.force_non_neg(xper)  # apply at every iteration?
        xper_neg = self.force_non_neg(xunp - dx)

        return torch.cat((xper, xper_neg)), coords
