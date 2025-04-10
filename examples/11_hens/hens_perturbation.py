from datetime import datetime

from numpy import asarray
from torch import Tensor, inference_mode
from xarray import open_dataset

from earth2studio.data import DataSource
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import (
    CorrelatedSphericalGaussian,
    HemisphericCentredBredVector,
)
from earth2studio.utils.coords import CoordSystem


class HENSPerturbation:
    """Initialize the HemisphericCentredBredVector perturbation method for obtaining a calibrated ensemble as described in Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators https://arxiv.org/abs/2408.03100.

    Parameters
    ----------
    model : PrognosticModel
        The prognostic model to be used for ensemble forecasting
    data : DataSource
        Data source for obtaining initial conditions
    start_time : datetime
        The initial time for the ensemble forecast
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
        data: DataSource,
        start_time: datetime,
        skill_path: str,
        noise_amplification: float,
        perturbed_var: str | list[str],
        integration_steps: int,
    ) -> None:

        noise_amp_seed = self.get_noise_vector(
            model,
            skill_path=skill_path,
            noise_amplification=noise_amplification,
            vars=perturbed_var,
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
            data=data,
            time=start_time,
            noise_amplitude=noise_amp_iter,
            integration_steps=integration_steps,  # use cfg.breeding_steps
            seeding_perturbation_method=seed_perturbation,
        )

        return

    def get_noise_vector(
        self,
        model: PrognosticModel,
        skill_path: str = None,
        noise_amplification: float = 1.0,
        vars: str | list[str] | None = None,
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
        vars : str | list[str] | None, optional
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
        if vars is None:
            vars = model_vars
        elif isinstance(vars, str):
            vars = [vars]

        # set noise for variables which shall not be perturbed to 0.
        skill = open_dataset(skill_path)
        scale_vec = Tensor(
            asarray(
                [
                    (
                        skill.sel(channel=var, lead_time=lead_time)["value"].item()
                        if var in vars
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
