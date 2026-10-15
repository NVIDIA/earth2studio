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

from typing import Any

import torch
import xarray as xr

from earth2studio.perturbation._array import (
    amplitude_like,
    array_module,
    as_host_or_cupy,
    match_backend,
    randn,
)
from earth2studio.utils import handshake_dim


class Brown:
    """Lat/Lon 2D brown noise

    Parameters
    ----------
    noise_amplitude : float | Tensor, optional
        Noise amplitude, by default 0.05. If a tensor,
        this must be broadcastable with the input data.
    reddening : int, optional
        Reddening in Fourier space, by default 2
    """

    def __init__(
        self, noise_amplitude: float | torch.Tensor = 0.05, reddening: int = 2
    ):
        self.reddening = reddening
        self.noise_amplitude = (
            noise_amplitude
            if isinstance(noise_amplitude, torch.Tensor)
            else torch.Tensor([noise_amplitude])
        )

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Apply perturbation method

        Parameters
        ----------
        x : xr.DataArray
            xarray DataArray intended to apply perturbation on, must contain "lat" and
            "lon" dimensions in the last two dims

        Returns
        -------
        xr.DataArray
            Perturbed DataArray with the coordinates of the input
        """
        # Check the required dimensions are present
        handshake_dim(x, required_dim="lat", required_index=-2)
        handshake_dim(x, required_dim="lon", required_index=-1)

        orig = x
        x = as_host_or_cupy(x)
        xp = array_module(x)
        data = x.data
        noise = self._generate_noise_correlated(xp, data.shape, data.dtype)
        noise_amplitude = amplitude_like(self.noise_amplitude, xp, data.dtype)
        return match_backend(x.copy(data=data + noise_amplitude * noise), orig)

    def _generate_noise_correlated(
        self, xp: Any, shape: tuple[int, ...], dtype: Any
    ) -> Any:
        """Utility class for producing brown noise."""
        noise = randn(xp, shape, dtype)
        x_white = xp.fft.rfft2(noise)
        S = (
            xp.abs(xp.fft.fftfreq(shape[-2]).reshape(-1, 1)) ** self.reddening
            + xp.fft.rfftfreq(shape[-1]) ** self.reddening
        )
        S = 1 / S
        S[..., 0, 0] = 0
        S = S / xp.sqrt(xp.mean(S**2))

        x_shaped = x_white * S.astype(noise.dtype)
        noise_shaped = xp.fft.irfft2(x_shaped, s=shape[-2:])
        return noise_shaped.astype(noise.dtype)
