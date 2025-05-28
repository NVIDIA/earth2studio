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

from typing import Any

import numpy as np
import torch

try:
    from torch_harmonics import InverseRealSHT
except ImportError:
    InverseRealSHT = None
from typing_extensions import Self

from earth2studio.utils import handshake_dim
from earth2studio.utils.imports import check_extra_imports
from earth2studio.utils.type import CoordSystem


class Gaussian:
    """Standard Gaussian peturbation

    Parameters
    ----------
    noise_amplitude : float | Tensor, optional
        Noise amplitude, by default 0.05. If a tensor,
        this must be broadcastable with the input data.
    """

    def __init__(self, noise_amplitude: float | torch.Tensor = 0.05):
        self.noise_amplitude = (
            noise_amplitude
            if isinstance(noise_amplitude, torch.Tensor)
            else torch.Tensor([noise_amplitude])
        )

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
        tuple[torch.Tensor, CoordSystem]:
            Output tensor and respective coordinate system dictionary
        """
        noise_amplitude = self.noise_amplitude.to(x.device)
        return x + noise_amplitude * torch.randn_like(x), coords


@check_extra_imports("perturbation", [InverseRealSHT])
class CorrelatedSphericalGaussian:
    """Produces Gaussian random field on the sphere with Matern
    covariance peturbation method output to a lat lon grid

    Warning
    -------
    Presently this method generates noise on equirectangular grid of size [N, 2*N] when
    N is even or [N+1, 2*N] when N is odd.

    Parameters
    ----------
    noise_amplitude : float | Tensor, optional
        Noise amplitude, by default 0.05. If a tensor,
        this must be broadcastable with the input data.
    alpha : float, optional
        Regularity parameter. Larger means smoother, by default 2.0
    tau : float, optional
        Length-scale parameter. Larger means more scales, by default 3.0
    sigma : Union[float, None], optional
        Scale parameter. If None, sigma = tau**(0.5*(2*alpha - 2.0)), by default None
    """

    def __init__(
        self,
        noise_amplitude: float | torch.Tensor | None = None,
        sigma: float = 1.0,
        length_scale: float = 5.0e5,
        time_scale: float = 48.0,
    ) -> None:
        if noise_amplitude is None:
            raise ValueError("pass noise amplitude")
        self.sigma = sigma
        self.length_scale = length_scale
        self.time_scale = time_scale
        self.noise_amplitude = (
            noise_amplitude
            if isinstance(noise_amplitude, torch.Tensor)
            else torch.Tensor([noise_amplitude])
        )

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
            Ordered dict representing coordinate system that describes the tensor, must
            contain "lat" and "lon" coordinates

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]:
            Output tensor and respective coordinate system dictionary
        """
        shape = x.shape
        # Check the required dimensions are present
        handshake_dim(coords, required_dim="lat", required_index=-2)
        handshake_dim(coords, required_dim="lon", required_index=-1)

        # Check the ratio
        if 2 * (shape[-2] // 2) != shape[-1] / 2:
            raise ValueError("Lat/lon aspect ration must be N:2N or N+1:2N")

        nlat = 2 * (shape[-2] // 2)  # Noise only support even lat count
        sampler = CorrelatedSphericalField(
            nlat=nlat,
            length_scale=self.length_scale,
            time_scale=self.time_scale,
            sigma=self.sigma,
            N=np.prod(shape[1:-2], dtype=int),
        )
        sampler = sampler.to(x.device)

        sample_noise = sampler(x, None)
        sample_noise = sample_noise.reshape(*shape[:-2], nlat, 2 * nlat)

        # Hack for odd lat coords
        if x.shape[-2] % 2 == 1:
            noise = torch.zeros_like(x)
            noise[..., :-1, :] = sample_noise
            noise[..., -1:, :] = noise[..., -2:-1, :]
        else:
            noise = sample_noise

        noise_amplitude = self.noise_amplitude.to(x.device)
        return x + noise_amplitude * noise, coords


class CorrelatedSphericalField(torch.nn.Module):
    """
    This class was taken from https://github.com/ankurmahesh/earth2mip-fork/blob/HENS/earth2mip/ensemble_utils.py#L392-L531.
    Reference publication: A.Mahesh et al. Huge Ensembles Part I: Design of Ensemble Weather Forecasts using Spherical Fourier Neural Operators https://arxiv.org/abs/2408.03100.

    This class can be used to create noise on the sphere
    with a given length scale (in m) and time scale (in hours).

    It mimics the implementation of the SPPT: Stochastic Perturbed
    Parameterized Tendency in this paper:

    https://www.ecmwf.int/sites/default/files/elibrary/2009/11577-stochastic-parametrization-and-model-uncertainty.pdf

    Parameters
    ----------
    nlat : int
        Number of latitudinal modes;
        longitudinal modes are 2*nlat.
    length_scale : float
        Correlation length scale in meters that determines the spatial decorrelation
        distance of the noise field on the sphere
    time_scale : int
        Time scale in hours for the AR(1) process, that governs
        the evolution of the coefficients
    sigma: float
        desired standard deviation of the field in grid point space
    N: int
        Number of latent dimensions
    grid : string, default is "equiangular"
        Grid type. Currently supports "equiangular" and
        "legendre-gauss".
    dtype : torch.dtype, default is torch.float32
        Numerical type for the calculations.
    """

    def __init__(
        self,
        nlat: int,
        length_scale: float,
        time_scale: float,
        sigma: float,
        N: int,
        grid: str = "equiangular",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        dt = 6.0
        self.phi = np.exp(-dt / time_scale)

        # Number of latitudinal modes.
        self.nlat = nlat

        # Inverse SHT
        self.isht = InverseRealSHT(
            self.nlat, 2 * self.nlat, grid=grid, norm="backward"
        ).to(dtype=dtype)

        r_earth = 6.371e6
        # kT is defined on slide 7
        self.kT = (length_scale / r_earth) ** 2 / 2
        F0 = self.calculateF0(self.sigma, self.phi, self.nlat, self.kT)

        prods = (
            torch.tensor([j * (j + 1) for j in range(0, self.nlat)])
            .view(self.nlat, 1)
            .repeat(1, self.nlat + 1)
        )

        sigma_n = torch.tril(torch.exp(-self.kT * prods / 2) * F0)
        self.register_buffer("sigma_n", sigma_n)

        # Save mean and var of the standard Gaussian.
        # Need these to re-initialize distribution on a new device.
        mean = torch.tensor([0.0]).to(dtype=dtype)
        var = torch.tensor([1.0]).to(dtype=dtype)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
        self.N = N

        # Standard normal noise sampler.
        # Why is this defined in the init? Not sure, would be a lot more flexible
        # in the call function
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)
        xi = self.gaussian_noise.sample(
            torch.Size((self.N, self.nlat, self.nlat + 1, 2))
        ).squeeze()
        xi = torch.view_as_complex(xi)

        # Set specrtral cofficients to this value at initial time
        # for stability in teh AR(1) process.  See link in description
        coeff: torch.tensor = ((1 - self.phi**2) ** (-0.5)) * self.sigma_n * xi
        coeff = coeff.unsqueeze(0)
        self.register_buffer("coeff", coeff)

    def calculateF0(
        self, sigma: float, phi: float, nlat: int, kT: float
    ) -> torch.Tensor:
        """
        This function scales the coefficients such that their
        grid-point standard deviation is sigma.
        sigma is the desired variance
        phi is a np.exp(-dt/time_scale)
        """
        numerator = sigma**2 * (1 - (phi**2))
        wavenumbers = torch.arange(1, nlat)
        denominator = (2 * wavenumbers + 1) * torch.exp(
            -kT * wavenumbers * (wavenumbers + 1)
        )
        denominator = 2 * torch.Tensor(denominator).sum()

        return (numerator / denominator) ** 0.5

    def forward(self, x: torch.Tensor, time: np.datetime64 = None) -> torch.Tensor:
        """
        Generate and return a field with a correlated length scale.

        Update the coefficients using an AR(1) process.
        """
        noises = []
        # iterate over samples in batch
        for _ in range(x.shape[0]):
            noise = self.isht(self.coeff) * 4 * np.pi  # type: ignore
            noises.append(noise.reshape(1, 1, 1, self.N, self.nlat, self.nlat * 2))

            # Sample Gaussian noise. # TODO why??? for next step maybe?
            xi = self.gaussian_noise.sample(
                torch.Size((self.N, self.nlat, self.nlat + 1, 2))
            ).squeeze()
            xi = torch.view_as_complex(xi)

            self.coeff = (self.phi * self.coeff) + (self.sigma_n * xi)  # type: ignore

        return torch.cat(noises)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Override cuda and to methods so sampler gets initialized with mean and
        variance on the correct device, to(*args, **kwargs)
        """
        super().to(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self
