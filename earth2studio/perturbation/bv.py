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


from collections.abc import Callable
from datetime import datetime

import numpy as np
import torch

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation.base import Perturbation
from earth2studio.perturbation.brown import Brown
from earth2studio.utils.time import to_time_array
from earth2studio.utils.type import CoordSystem


class BredVector:
    """Bred Vector perturbation method, a classical technique for pertubations in
    ensemble forecasting.

    Parameters
    ----------
    model : Callable[[torch.Tensor], torch.Tensor]
        Dynamical model, typically this is the prognostic AI model.
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
        model: Callable[
            [torch.Tensor, CoordSystem],
            tuple[torch.Tensor, CoordSystem],
        ],
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
            Ordered dict representing coordinate system that describes the tensor

        Returns
        -------
        Returns
        -------
        tuple[torch.Tensor, CoordSystem]:
            Output tensor and respective coordinate system dictionary
        """
        noise_amplitude = self.noise_amplitude.to(x.device)
        dx, coords = self.seeding_perturbation_method(x, coords)
        dx -= x

        xd = torch.clone(x)
        xd, _ = self.model(xd, coords)
        # Run forward model
        for k in range(self.integration_steps):
            x1 = x + dx
            x2, _ = self.model(x1, coords)
            if self.ensemble_perturb:
                dx1 = x2 - xd
                dx = dx1 + noise_amplitude * (dx - dx.mean(dim=0))
            else:
                dx = x2 - xd

        gamma = torch.norm(x) / torch.norm(x + dx)
        return x + dx * noise_amplitude * gamma, coords


class HemisphericCentredBredVector:
    """Bred Vector perturbation method, following the approach introduced in
    https://arxiv.org/abs/2408.03100. The vector is bred by advancing in time.
    The bred vector is scaled seperately in the northern and in the southern
    extra-tropics and interpolated in the tropics. Additionally, it applies a
    centred perturbation, ie generating two perturbed stated by adding and
    subtracting the bred vector, respecitvely.


    Parameters
    ----------
    model : Callable[[torch.Tensor], torch.Tensor]
        Dynamical model, typically this is the prognostic AI model.
    data : DataSource
        data source for obtaining warmup time steps.
    time : list[str] | list[datetime] | list[np.datetime64]
        time from which initial conditions are taken
    seeding_perturbation_method : Perturbation, optional
        Method to seed the Bred Vector perturbation
    noise_amplitude : float | Tensor, optional
        Noise amplitude, by default 0.05. If a tensor,
        this must be broadcastable with the input data.
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
        # model: Callable[
        #     [torch.Tensor, CoordSystem],
        #     tuple[torch.Tensor, CoordSystem],
        # ],
        data: DataSource,
        time: list[str] | list[datetime] | list[np.datetime64],
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

        return

    def fetch_warmup_ics(
        self, time: list[str] | list[datetime] | list[np.datetime64]
    ) -> None:
        """
        Method for obtaining ICs for warmup steps

        Parameters
        ----------
        time : list[str] | list[datetime] | list[np.datetime64]
            time from which initial conditions are taken
        """

        inco = self.model.input_coords()

        time = time if isinstance(time, list) else [time]
        self.warmup_times = (
            to_time_array(time)
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
        """
        if humidity and tcwv in variable set, add to list of variables to clip
        """
        self.clip_idcs = []
        for ii, var in enumerate(self.model.input_coords()["variable"]):
            if var[0] == "q" or var == "tcwv" or var[0] == "r" or var[:1] == "tp":
                self.clip_idcs.append(ii)
        return

    def hemispheric_norm(self, xx: torch.Tensor) -> torch.Tensor:
        """
        calculate norm of fields speperately in northern and southern extra-tropics

        Parameters
        ----------
        xx : torch.Tensor
            Input tensor
        """
        nvar, nlat, nlon = xx.shape[-3:]
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
                xx[..., :ex_tropic, :] ** 2
                * weights[
                    :ex_tropic,
                ],
                dim=(-2, -1),
            )
        )
        south = torch.sqrt(
            torch.mean(
                xx[..., -ex_tropic:, :] ** 2 * weights[-ex_tropic:], dim=(-2, -1)
            )
        )

        # interpolate in tropics
        tropic = torch.zeros(north.shape[0], 1, 1, nvar, in_tropic).to(self.device)
        for batch in range(north.shape[0]):
            for var in range(nvar):
                tropic[batch, 0, 0, var] = torch.linspace(
                    north[batch].squeeze()[var], south[batch].squeeze()[var], in_tropic
                )

        # expand and concatenate
        north = north[..., None].repeat(1, 1, 1, 1, ex_tropic)
        south = south[..., None].repeat(1, 1, 1, 1, ex_tropic)

        return torch.cat([north, tropic, south], dim=-1)[..., None]

    def force_non_neg(self, xx: torch.Tensor) -> torch.Tensor:
        """
        clip variables to non-negative values

        Parameters
        ----------
        xx : torch.Tensor
            Input tensor
        """
        nn = xx[:, :, :, self.clip_idcs, ...]
        nn = torch.where(nn < 0.0, 0.0, nn)
        xx[:, :, :, self.clip_idcs, ...] = nn
        return xx

    @torch.inference_mode()
    def __call__(
        self,
        xx: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Apply perturbation method

        Parameters
        ----------
        xx : torch.Tensor
            Input tensor intended to apply perturbation on
        coords : CoordSystem
            Ordered dict representing coordinate system that describes the tensor

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]:
            Output tensor and respective coordinate system dictionary
        """
        bs = xx.shape[0]
        self.device = xx.device
        self.noise_amplitude = self.noise_amplitude.to(self.device)
        if bs % 2 != 0:
            raise ValueError(
                "batch size and ensemble size have to be even for centred perturbation"
            )

        # get unperturbed intital state
        xunp = (
            self.ics[0].unsqueeze(dim=0).repeat(bs // 2, 1, 1, 1, 1, 1).to(self.device)
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
            dx = self.noise_amplitude * (dx / hem_norm)

            xunp = (
                self.ics[ii + 1]
                .unsqueeze(dim=0)
                .repeat(bs // 2, 1, 1, 1, 1, 1)
                .to(self.device)
            )
            xper = xunp + dx

        xper = self.force_non_neg(xper)  # apply at every iteration?
        xper_neg = self.force_non_neg(xunp - dx)

        return torch.cat((xper, xper_neg)), coords
