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

import numpy as np
import torch
from torch_harmonics import InverseRealSHT

from earth2studio.utils import handshake_dim
from earth2studio.utils.type import CoordSystem


class SphericalGaussian:
    """Gaussian random field on the sphere with Matern covariance peturbation method
    output to a lat lon grid

    Warning
    -------
    Presently this method generates noise on equirectangular grid of size [N, 2*N] when
    N is even or [N+1, 2*N] when N is odd.

    Parameters
    ----------
    noise_amplitude : float, optional
        Noise amplitude, by default 0.05
    alpha : float, optional
        Regularity parameter. Larger means smoother, by default 2.0
    tau : float, optional
        Length-scale parameter. Larger means more scales, by default 3.0
    sigma : Union[float, None], optional
        Scale parameter. If None, sigma = tau**(0.5*(2*alpha - 2.0)), by default None
    """

    def __init__(
        self,
        noise_amplitude: float = 0.05,
        alpha: float = 2.0,
        tau: float = 3.0,
        sigma: float | None = None,
    ):
        self.noise_amplitude = noise_amplitude
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma

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
            Ouput tensor and respective coordinate system dictionary
        """
        shape = x.shape
        # Check the required dimensions are present
        handshake_dim(coords, required_dim="lat", required_index=-2)
        handshake_dim(coords, required_dim="lon", required_index=-1)
        # Check the ratio
        if 2 * (shape[-2] // 2) != shape[-1] / 2:
            raise ValueError("Lat/lon aspect ration must be N:2N or N+1:2N")

        nlat = 2 * (shape[-2] // 2)  # Noise only support even lat count
        sampler = GaussianRandomFieldS2(
            nlat=nlat,
            alpha=self.alpha,
            tau=self.tau,
            sigma=self.sigma,
            device=x.device,
        )
        sampler = sampler.to(x.device)

        sample_noise = sampler(np.array(shape[:-2]).prod()).reshape(
            *shape[:-2], nlat, 2 * nlat
        )

        # Hack for odd lat coords
        if x.shape[-2] % 2 == 1:
            noise = torch.zeros_like(x)
            noise[..., :-1, :] = sample_noise
            noise[..., -1:, :] = noise[..., -2:-1, :]
        else:
            noise = sample_noise

        return x + self.noise_amplitude * noise, coords


class GaussianRandomFieldS2(torch.nn.Module):
    """A mean-zero Gaussian Random Field on the sphere with Matern covariance:
    C = sigma^2 (-Lap + tau^2 I)^(-alpha).

    Lap is the Laplacian on the sphere, I the identity operator,
    and sigma, tau, alpha are scalar parameters.

    Note: C is trace-class on L^2 if and only if alpha > 1.

    Parameters
    ----------
    nlat : int
        Number of latitudinal modes;
        longitudinal modes are 2*nlat.
    alpha : float, default is 2
        Regularity parameter. Larger means smoother.
    tau : float, default is 3
        Lenght-scale parameter. Larger means more scales.
    sigma : float, default is None
        Scale parameter. Larger means bigger.
        If None, sigma = tau**(0.5*(2*alpha - 2.0)).
    radius : float, default is 1
        Radius of the sphere.
    grid : string, default is "equiangular"
        Grid type. Currently supports "equiangular" and
        "legendre-gauss".
    dtype : torch.dtype, default is torch.float32
        Numerical type for the calculations.

    Parameters
    ----------
    nlat : int
        Number of latitudinal modes; longitudinal modes are 2*nlat.
    alpha : float, optional
        Regularity parameter. Larger means smoother, by default 2.0
    tau : float, optional
        Lenght-scale parameter, by default 3.0
    sigma : Union[float, None], optional
        Scale parameter, by default None
    radius : float, optional
        Radius of the sphere, by default 1.0
    grid : str, optional
        Grid type. Currently supports "equiangular" and "legendre-gauss", by default
        "equiangular"
    dtype : torch.dtype, optional
        Numerical type for the calculations, by default torch.float32
    device : torch.device, optional
        Pytorch device, by default "cuda:0"
    """

    def __init__(
        self,
        nlat: int,
        alpha: float = 2.0,
        tau: float = 3.0,
        sigma: float | None = None,
        radius: float = 1.0,
        grid: str = "equiangular",
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda:0",
    ):
        super().__init__()
        # Number of latitudinal modes.
        self.nlat = nlat

        # Default value of sigma if None is given.
        if alpha < 1.0:
            raise ValueError(f"Alpha must be greater than one, got {alpha}.")

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - 2.0))

        # Inverse SHT
        self.isht = (
            InverseRealSHT(self.nlat, 2 * self.nlat, grid=grid, norm="backward")
            .to(dtype=dtype)
            .to(device=device)
        )

        # Square root of the eigenvalues of C.
        sqrt_eig = (
            torch.tensor([j * (j + 1) for j in range(self.nlat)], device=device)
            .view(self.nlat, 1)
            .repeat(1, self.nlat + 1)
        )
        sqrt_eig = torch.tril(
            sigma * (((sqrt_eig / radius**2) + tau**2) ** (-alpha / 2.0))
        )
        sqrt_eig[0, 0] = 0.0
        sqrt_eig = sqrt_eig.unsqueeze(0)
        self.register_buffer("sqrt_eig", sqrt_eig)

        # Save mean and var of the standard Gaussian.
        # Need these to re-initialize distribution on a new device.
        mean = torch.tensor([0.0], device=device).to(dtype=dtype)
        var = torch.tensor([1.0], device=device).to(dtype=dtype)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)

    def forward(self, N: int, xi: torch.Tensor | None = None) -> torch.Tensor:
        """Sample random functions from a spherical GRF.

        Parameters
        ----------
        N : int
            Number of functions to sample.
        xi : torch.Tensor, default is None
            Noise is a complex tensor of size (N, nlat, nlat+1).
            If None, new Gaussian noise is sampled.
            If xi is provided, N is ignored.

        Output
        -------
        u : torch.Tensor
           N random samples from the GRF returned as a
           tensor of size (N, nlat, 2*nlat) on a equiangular grid.
        """
        # Sample Gaussian noise.
        if xi is None:
            gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)
            xi = gaussian_noise.sample(
                torch.Size((N, self.nlat, self.nlat + 1, 2))
            ).squeeze()
            xi = torch.view_as_complex(xi)

        # Karhunen-Loeve expansion.
        u = self.isht(xi * self.sqrt_eig)

        return u
