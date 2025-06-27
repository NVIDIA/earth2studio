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

from collections import OrderedDict
from collections.abc import Generator

import numpy as np
import torch
from loguru import logger

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation.base import Perturbation
from earth2studio.utils import handshake_dim, handshake_size
from earth2studio.utils.time import to_time_array
from earth2studio.utils.type import CoordSystem, TimeArray


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
    seeding_perturbation_method : Perturbation, optional
        Method to seed the Bred Vector perturbation
    noise_amplitude : float | Tensor, optional
        Noise amplitude, by default 0.05. If a tensor, this must be broadcastable with
        the input data
    integration_steps : int, optional
        Number of integration steps to use in forward call, by default 3

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
            )[:, None, None]
        )
        self.integration_steps = integration_steps
        self.seeding_perturbation_method = seeding_perturbation_method
        self.set_clip_indices()
        self._residual: list[torch.Tensor] = []

    @torch.inference_mode()
    def create_generator(
        self, time: TimeArray, generator_size: int = 1, device: torch.device = "cpu"
    ) -> Generator[torch.Tensor, None, None]:
        """Creates and initializes the perturbation generator"""
        # Initialize your IC or other necessary components
        batch_size = generator_size // 2
        input_coords = self.model.input_coords()

        time = to_time_array(time)
        warmup_times = (
            time
            + np.arange(-self.integration_steps, 1)
            * self.model.output_coords(input_coords)["lead_time"]
        )
        input_data, data_coords = fetch_data(
            source=self.data,
            time=warmup_times,
            variable=input_coords["variable"],
            lead_time=input_coords["lead_time"],
            device="cpu",
        )
        coords = OrderedDict(
            [("batch", np.arange(batch_size))] + list(data_coords.items())
        )

        if input_coords["lead_time"].shape[0] > 1:
            logger.warning(
                "Input data / models that require multiple lead times may lead to unexpected behavior"
            )

        # get unperturbed intital state, assuming tensor always has 6 dims
        xunp = input_data[:1].repeat(batch_size, 1, 1, 1, 1, 1).to(device)
        # Commenting here, not tested for when multiple lead times are needed
        # May work...
        coords["time"] = data_coords["time"][:1]

        # generate perturbed initial state
        xper, coords = self.seeding_perturbation_method(xunp, coords)

        for ii in range(self.integration_steps):
            xunp, _ = self.model(xunp, coords)
            xper, _ = self.model(xper, coords)

            dx = xper - xunp
            hem_norm = self.hemispheric_norm(dx, device)

            # if zero elements are requierd, replace NaNs in scaled dx with 0
            if (hem_norm == 0).any():
                raise ValueError(
                    "zero element in hemispheric norm, maybe noise amplification too small?"
                )

            # scale dx
            dx = self.noise_amplitude * (dx / hem_norm)

            xunp = (
                input_data[ii + 1]
                .unsqueeze(dim=0)
                .repeat(batch_size, 1, 1, 1, 1, 1)
                .to(device)
            )
            coords["time"] = data_coords["time"][ii + 1 : ii + 2]
            xper = xunp + dx
            # self.force_non_neg(xper[i : i + 1])

        # Yield single batches in alternating order to keep perturbation centered
        for i in range(batch_size):
            yield self.force_non_neg(xper[i : i + 1])
            yield self.force_non_neg(xunp[i : i + 1] - dx[i : i + 1])

    def set_clip_indices(self) -> None:
        """If humidity and tcwv in variable set, add to list of variables to clip"""
        self.clip_idcs = []
        for ii, var in enumerate(self.model.input_coords()["variable"]):
            if var[0] == "q" or var == "tcwv" or var[0] == "r" or var[:2] == "tp":
                self.clip_idcs.append(ii)
        return

    def hemispheric_norm(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
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

        jac = torch.sin(torch.linspace(0, torch.pi, nlat))[:, None].to(device)
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
        tropic = torch.zeros(north.shape[0], 1, 1, nvar, in_tropic).to(device)
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
            Input tensor intended to apply perturbation on, not used in this
            perturbation method
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
        # We should probably run the perturbation and then check the output noise can
        # match the size of the input tensor
        handshake_dim(coords, required_dim="time", required_index=-5)
        handshake_dim(coords, required_dim="lead_time", required_index=-4)
        handshake_dim(coords, required_dim="variable", required_index=-3)
        handshake_dim(coords, required_dim="lat", required_index=-2)
        handshake_dim(coords, required_dim="lon", required_index=-1)
        handshake_size(coords, required_dim="time", required_size=1)

        if len(shape) != 5 and len(shape) != 6:
            raise ValueError("Input tensor and coords need 5 or 6 dimensions")

        self.noise_amplitude = self.noise_amplitude.to(x.device)

        # Not the cleanest but works
        if len(shape) == 5:
            batch_size = 1
        if len(shape) == 6:
            batch_size = coords[list(coords.keys())[0]].shape[0]

        # This is some pretty annoying logic to deal with storing the centered
        # perturbation for odd batch sizes... could be worse probably can be better
        # TLDR: if odd batch, store the last generated perturbation for next call
        generator_size = 2 * int((batch_size + 1 - len(self._residual)) // 2)
        # Special case where we can skip the generation
        if generator_size == 0:
            noise = [self._residual.pop()]
        else:
            # Create generator if it doesn't exist
            generator = self.create_generator(
                coords["time"], generator_size=generator_size, device=x.device
            )
            # Generate noise, and concat in batch dimension
            noise = self._residual + [next(generator) for _ in range(generator_size)]
            # If we have an extra output (odd batch size) store it for next call
            try:
                self._residual = [noise.pop(batch_size)]
            except IndexError:
                self._residual = []

        # Sanity checkout for this perturbation
        if len(noise) != batch_size:
            raise ValueError(
                "Seems something went wrong in the perturbation, open an issue with your setup"
            )

        x = torch.cat(noise, dim=0)
        if len(shape) == 5:
            x = x.squeeze(0)

        return x, coords
