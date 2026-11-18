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

"""Precipitation estimator diagnostic model with AutoModel support.

This diagnostic model estimates total precipitation from atmospheric variables
using a neural network loaded from a model package.
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
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem

INPUT_VARIABLES = ["t2m", "u10m", "v10m", "msl", "tcwv"]
OUTPUT_VARIABLES = ["tp"]
GRID = LatLonGrid(
    np.linspace(90, -90, 361, endpoint=True),
    np.linspace(0, 360, 720, endpoint=False),
)


class PrecipEstimator(torch.nn.Module, AutoModelMixin):
    """Precipitation estimator diagnostic model.

    Estimates total precipitation from atmospheric variables using
    a neural network. Loads weights from a model package.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model for precipitation estimation
    center : torch.Tensor
        Model input center normalization tensor of size [5, 1, 1]
    scale : torch.Tensor
        Model input scale normalization tensor of size [5, 1, 1]

    Note
    ----
    This model estimates 6-hourly total precipitation from atmospheric
    state variables on a 0.5-degree grid.

    Badges
    ------
    region:global product:precipitation provider:nvidia backend:pytorch
    """

    stochastic: bool = False

    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.core_model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the diagnostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature with a dynamic leading batch dimension,
            the five input variables and the 0.5 degree global grid.
        """
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": INPUT_VARIABLES},
            dynamic=("batch",),
            grid=GRID,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate the input and plan the output coordinate system.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature with the tp variable.
        """
        handshake_dataarray(input_coords, self.input_coords())
        return coord_array_like(input_coords, {"variable": OUTPUT_VARIABLES})

    def default_sources(self) -> None:
        """Recommend no input data source.

        Returns
        -------
        None
            No source recommendation; the caller supplies the input field.
        """
        return None

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package.

        Returns
        -------
        Package
            Model package with checkpoint files.
        """
        return Package(
            "hf://nvidia/precip-estimator@0123456789abcdef0123456789abcdef01234567",
            cache_options={
                "cache_storage": Package.default_cache("precip_estimator"),
                "same_names": True,
            },
        )

    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load diagnostic model from package.

        Parameters
        ----------
        package : Package
            Model package with checkpoint files.

        Returns
        -------
        DiagnosticModel
            Loaded model instance.
        """
        # Resolve checkpoint files
        model_path = package.resolve("model.pt")
        center_path = package.resolve("center.npy")
        scale_path = package.resolve("scale.npy")

        # Load model on CPU; the caller moves it with .to(device)
        core_model = torch.load(model_path, map_location="cpu", weights_only=False)
        core_model.eval()
        core_model.requires_grad_(False)

        # Load normalization parameters
        center = torch.from_numpy(np.load(center_path)).view(5, 1, 1)
        scale = torch.from_numpy(np.load(scale_path)).view(5, 1, 1)

        return cls(core_model, center=center, scale=scale)

    @torch.inference_mode()
    @batch_func()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Estimate total precipitation.

        Parameters
        ----------
        x : xr.DataArray
            Input field with dims (..., variable, lat, lon) matching
            ``input_coords()``.

        Returns
        -------
        xr.DataArray
            Precipitation field matching ``output_coords(x)``.
        """
        handshake_nonempty(x)
        signature = self.output_coords(x)
        # tensor shape: (batch, 5, lat, lon) after batch_func compression
        tensor, _ = x.e2s.to_torch()
        out = self.core_model((tensor - self.center) / self.scale)
        return from_torch(out, signature)
