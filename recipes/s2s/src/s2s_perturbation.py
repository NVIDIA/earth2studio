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

from typing import Any

from numpy import asarray
from omegaconf import DictConfig
from torch import Tensor, inference_mode
from xarray import open_dataset

from earth2studio.data import DataSource
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import (
    CorrelatedSphericalGaussian,
    HemisphericCentredBredVector,
)
from earth2studio.utils.coords import CoordSystem


def initialize_perturbation(
    cfg: DictConfig, model: PrognosticModel, data_source: DataSource
) -> Any:
    """Initialize the perturbation method for the ensemble forecast

    Parameters
    ----------
    cfg : DictConfig
        The configuration object
    model : PrognosticModel
        The prognostic model to be used for ensemble forecasting
    data_source : DataSource
        Data source for obtaining initial conditions

    Returns
    -------
    perturbation : Any
        The perturbation method
    """
    if cfg.perturbation.type == "HENSPerturbation":
        return HENSPerturbation(
            model=model,
            data_source=data_source,
            skill_path=cfg.perturbation.skill_path,
            noise_amplification=cfg.perturbation.noise_amplification,
            perturbed_var=cfg.perturbation.perturbed_var,
            integration_steps=cfg.perturbation.integration_steps,
        )
    elif cfg.perturbation.type == "CorrelatedSphericalGaussian":
        return CorrelatedSphericalGaussian(
            noise_amplitude=cfg.perturbation.noise_amplitude,
            sigma=cfg.perturbation.sigma,
            length_scale=cfg.perturbation.length_scale,
            time_scale=cfg.perturbation.time_scale,
        )
    else:
        raise ValueError(f"Invalid perturbation type: {cfg.perturbation.type}")


class HENSPerturbation:
    """Initialize the HemisphericCentredBredVector perturbation method for obtaining a calibrated ensemble as described in Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators https://arxiv.org/abs/2408.03100.

    Parameters
    ----------
    model : PrognosticModel
        The prognostic model to be used for ensemble forecasting
    data_source : DataSource
        Data source for obtaining initial conditions. Uses the synchronous __call__ interface.
    skill_path : str
        Path to the file containing forecast skill scores for noise amplification
    noise_amplification : float
        Base amplification factor for the noise
    perturbed_var : str | list[str]
        Variable(s) to be perturbed in the ensemble
    integration_steps : int
        Number of integration steps for the bred vector calculation
    """

    def __init__(
        self,
        model: PrognosticModel,
        data_source: DataSource,
        skill_path: str,
        noise_amplification: float,
        perturbed_var: str | list[str],
        integration_steps: int,
    ) -> None:

        noise_amp_seed = self.get_noise_vector(
            model,
            skill_path=skill_path,
            noise_amplification=noise_amplification,
            perturbed_var=perturbed_var,
        )
        noise_amp_iter = self.get_noise_vector(
            model,
            skill_path=skill_path,
            noise_amplification=noise_amplification,
        )

        seed_perturbation = CorrelatedSphericalGaussian(
            noise_amplitude=noise_amp_seed,
            sigma=1.0,
            length_scale=5.0e5,
            time_scale=48.0,
        )

        self.perturbation = HemisphericCentredBredVector(
            model=model,
            data=data_source,
            noise_amplitude=noise_amp_iter,
            integration_steps=integration_steps,  # use cfg.breeding_steps
            seeding_perturbation_method=seed_perturbation,
        )

        return

    def get_noise_vector(
        self,
        model: PrognosticModel,
        skill_path: str | None = None,
        noise_amplification: float = 1.0,
        perturbed_var: str | list[str] | None = None,
        lead_time: int = 48,
    ) -> Tensor:
        """Generate a noise vector for the HemisphericCentredBredVector perturbation method.

        This function creates a variable-specific noise vector based on forecast skill scores.
        The noise amplitude for each variable is scaled according to its forecast skill at the
        specified lead time, with variables not in the perturbation list set to zero.

        Parameters
        ----------
        model : PrognosticModel
            The prognostic model used for forecasting
        skill_path : str, optional
            Path to the file containing model skill scores (RMSE/MSE), by default None
        noise_amplification : float, optional
            Base amplification factor for the noise vector, by default 1.0
        perturbed_var : str | list[str] | None, optional
            Variables to be perturbed. If None, all model variables are perturbed, by default None
        lead_time : int, optional
            Lead time at which to evaluate model skill, by default 48

        Returns
        -------
        torch.Tensor
            A noise vector with shape (1, 1, 1, n_vars, 1, 1) where n_vars is the number of model variables

        Raises
        ------
        ValueError
            If skill_path is not provided
        """
        if skill_path is None:
            raise ValueError(
                f"provide path to data set containing {lead_time}h deterministic [r]mse"
            )

        model_vars = model.input_coords()["variable"]
        if perturbed_var is None:
            perturbed_var = model_vars
        elif isinstance(perturbed_var, str):
            perturbed_var = [perturbed_var]

        # set noise for variables which shall not be perturbed to 0.
        skill = open_dataset(skill_path)
        scale_vec = Tensor(
            asarray(
                [
                    (
                        skill.sel(channel=var, lead_time=lead_time)["value"].item()
                        if var in perturbed_var
                        else 0.0
                    )
                    for var in model_vars
                ]
            )
        )

        return scale_vec.reshape(1, 1, 1, -1, 1, 1) * noise_amplification

    @inference_mode()
    def __call__(
        self,
        x: Tensor,
        coords: CoordSystem,
    ) -> tuple[Tensor, CoordSystem]:
        """Apply the perturbation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be perturbed
        coords : CoordSystem
            Coordinate system describing the input tensor

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            The perturbed tensor and its corresponding coordinate system
        """
        return self.perturbation(x, coords)
