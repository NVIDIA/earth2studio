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

"""Probabilistic super-resolution diagnostic model.

This diagnostic model performs probabilistic super-resolution similar to CorrDiff,
generating multiple samples from a coarse input grid to a fine output grid.
"""

import numpy as np
import torch
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils.coords import (
    coord_array,
    handshake_dataarray,
    handshake_nonempty,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem

INPUT_VARIABLES = ["t2m", "u10m", "v10m", "msl"]
OUTPUT_VARIABLES = ["t2m", "u10m", "v10m", "msl"]
INPUT_GRID = LatLonGrid(
    np.linspace(90, -90, 91, endpoint=True),
    np.linspace(0, 360, 180, endpoint=False),
)
OUTPUT_GRID = LatLonGrid(
    np.linspace(90, -90, 361, endpoint=True),
    np.linspace(0, 360, 720, endpoint=False),
)


class SuperResolution(torch.nn.Module, AutoModelMixin):
    """Probabilistic super-resolution diagnostic model.

    Performs probabilistic super-resolution from a coarse 2-degree grid
    to a fine 0.5-degree grid, generating multiple samples per input. Seed the
    sampler with ``set_rng``; the constructor and ``load_model`` take no seed.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core model for super-resolution
    in_center : torch.Tensor
        Input center normalization tensor
    in_scale : torch.Tensor
        Input scale normalization tensor
    out_center : torch.Tensor
        Output center normalization tensor
    out_scale : torch.Tensor
        Output scale normalization tensor
    number_of_samples : int, optional
        Number of samples to generate per input, by default 1

    Note
    ----
    Similar to CorrDiff, this model generates multiple high-resolution
    samples from a single low-resolution input using probabilistic methods.

    Badges
    ------
    region:global class:downscaling provider:nvidia backend:pytorch
    """

    stochastic: bool = True
    _rng: torch.Generator | None = None

    def __init__(
        self,
        core_model: torch.nn.Module,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        number_of_samples: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(number_of_samples, int) or number_of_samples < 1:
            raise ValueError("`number_of_samples` must be a positive integer.")
        self.core_model = core_model
        self.number_of_samples = number_of_samples

        # Register normalization buffers
        self.register_buffer("in_center", in_center)
        self.register_buffer("in_scale", in_scale)
        self.register_buffer("out_center", out_center)
        self.register_buffer("out_scale", out_scale)

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the sampler's local noise generator.

        Parameters
        ----------
        seed : int
            Seed for the noise stream.
        reset : bool, optional
            Replace an existing generator, by default True. When False the
            generator is only created if absent and the seed is otherwise ignored.
        """
        if reset or self._rng is None:
            self._rng = torch.Generator().manual_seed(seed)

    def default_sources(self) -> None:
        """Recommend no input data source.

        Returns
        -------
        None
            No source recommendation; the caller supplies the input field.
        """
        return None

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the diagnostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature on the coarse 2-degree grid.
        """
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": INPUT_VARIABLES},
            dynamic=("batch",),
            grid=INPUT_GRID,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system with a sample dimension on the fine grid.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature with the leading input dimensions, then
            ``sample``, ``variable`` and the fine lat/lon grid.
        """
        handshake_dataarray(input_coords, self.input_coords())
        leading = input_coords.dims[:-3]
        coords = {
            dim: input_coords.coords[dim].values
            for dim in leading
            if dim in input_coords.coords
        }
        coords["sample"] = np.arange(self.number_of_samples)
        coords["variable"] = OUTPUT_VARIABLES
        dynamic = tuple(input_coords.attrs.get("earth2studio_dynamic_dims", ()))
        return coord_array(
            (*leading, "sample", "variable", "lat", "lon"),
            coords,
            dynamic=dynamic,
            sizes={dim: input_coords.sizes[dim] for dim in leading},
            dtype=input_coords.dtype,
            grid=OUTPUT_GRID,
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package.

        Returns
        -------
        Package
            Model package with checkpoint files.
        """
        return Package(
            "ngc://models/nvidia/modulus/super_resolution@v1.0",
            cache_options={
                "cache_storage": Package.default_cache("super_resolution"),
                "same_names": True,
            },
        )

    @classmethod
    def load_model(
        cls,
        package: Package,
        number_of_samples: int = 1,
    ) -> DiagnosticModel:
        """Load generative diagnostic from package.

        Parameters
        ----------
        package : Package
            Model package with checkpoint files.
        number_of_samples : int, optional
            Number of samples to generate, by default 1

        Returns
        -------
        DiagnosticModel
            Loaded model instance.
        """
        # Load model on CPU; the caller moves it with .to(device)
        core_model = torch.load(
            package.resolve("model.pt"), map_location="cpu", weights_only=False
        )
        core_model.eval()
        core_model.requires_grad_(False)

        # Load normalization parameters
        in_center = torch.from_numpy(np.load(package.resolve("in_center.npy")))
        in_scale = torch.from_numpy(np.load(package.resolve("in_scale.npy")))
        out_center = torch.from_numpy(np.load(package.resolve("out_center.npy")))
        out_scale = torch.from_numpy(np.load(package.resolve("out_scale.npy")))

        return cls(
            core_model,
            in_center=in_center.view(4, 1, 1),
            in_scale=in_scale.view(4, 1, 1),
            out_center=out_center.view(4, 1, 1),
            out_scale=out_scale.view(4, 1, 1),
            number_of_samples=number_of_samples,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate samples for a single input.

        Parameters
        ----------
        x : torch.Tensor
            Single input tensor [var, lat, lon].

        Returns
        -------
        torch.Tensor
            Generated samples [samples, var, lat, lon].
        """
        x_norm = ((x - self.in_center) / self.in_scale).unsqueeze(0)
        samples = []
        for _ in range(self.number_of_samples):
            # Draw from the local generator so global RNG state is untouched;
            # unseeded models fall back to the global RNG.
            if self._rng is None:
                noise = torch.randn(1, 4, 361, 720, device=x.device)
            else:
                noise = torch.randn(1, 4, 361, 720, generator=self._rng).to(x.device)
            samples.append(self.core_model(x_norm, noise))
        out = torch.cat(samples, dim=0)
        return out * self.out_scale + self.out_center

    @batch_func()
    def _generate(self, x: xr.DataArray) -> xr.DataArray:
        """Generate samples for a batch-packed input.

        Parameters
        ----------
        x : xr.DataArray
            Batch-packed input with dims (batch, variable, lat, lon).

        Returns
        -------
        xr.DataArray
            Samples with dims (batch, sample, variable, lat, lon).
        """
        handshake_nonempty(x)
        signature = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        out = torch.stack([self._forward(tensor[i]) for i in range(tensor.shape[0])])
        return from_torch(out, signature)

    @torch.inference_mode()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Generate high-resolution samples.

        Parameters
        ----------
        x : xr.DataArray
            Input field with dims (..., variable, lat, lon) matching
            ``input_coords()``.

        Returns
        -------
        xr.DataArray
            Samples matching ``output_coords(x)`` with dims
            (..., sample, variable, lat, lon).
        """
        signature = self.output_coords(x)
        # batch_func restores the original leading dims first and appends new
        # ones, so a new sample axis must be transposed into its planned place.
        return self._generate(x).transpose(*signature.dims)
