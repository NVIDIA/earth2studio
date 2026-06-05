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

"""Diagnostic Model Wrapper Skeleton Template.

This file demonstrates the required structure for Earth2Studio diagnostic model
wrappers. Copy this template and replace TODO comments with implementations.

Diagnostic models transform data at a single time point (no time integration).
Key differences from prognostic models:
- NO create_iterator method
- NO PrognosticMixin inheritance
- NO lead_time coordinate

Method ordering is CANONICAL and must be preserved:
1. __init__ — constructor
2. input_coords — input coordinate system
3. output_coords — output coordinate system (decorated @batch_coords())
4. load_default_package — classmethod returning default Package (AutoModel only)
5. load_model — classmethod loading model from package (AutoModel only)
6. to — device management (optional)
7. Private/support methods (e.g., _normalize, _forward)
8. __call__ — single-step forward (decorated @batch_func())

See real examples:
- earth2studio/models/dx/precipitation_afno.py (deterministic)
- earth2studio/models/dx/corrdiff.py (generative)
- earth2studio/models/dx/identity.py (simple)
"""

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

# ---------------------------------------------------------------------------
# Optional dependency imports (try/except pattern)
# ---------------------------------------------------------------------------
try:
    # TODO: Import optional packages here
    # import model_package
    pass
except ImportError:
    OptionalDependencyFailure("model-name")  # TODO: Use your dependency group name
    # model_package = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TODO: Define input/output variable lists
INPUT_VARIABLES = [
    "u10m",
    "v10m",
    "t2m",
    "msl",
]

OUTPUT_VARIABLES = [
    "tp",  # Total precipitation, or whatever the model outputs
]


# ===========================================================================
# TEMPLATE A: Simple Deterministic Diagnostic (with AutoModelMixin)
# ===========================================================================


class DeterministicDiagnostic(torch.nn.Module, AutoModelMixin):
    """One-line description of the diagnostic model.

    Extended description of the model, its source, architecture,
    and any relevant details.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model for the diagnostic transformation
    center : torch.Tensor
        Model input center normalization tensor
    scale : torch.Tensor
        Model input scale normalization tensor

    Note
    ----
    For more information see: <link to paper/repo>

    Badges
    ------
    region:global class:dx product:precip year:2024 gpu:40gb
    """

    # =========================================================================
    # 1. CONSTRUCTOR
    # =========================================================================
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

    # =========================================================================
    # 2. INPUT COORDINATES
    # =========================================================================
    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),  # MUST be first, MUST be np.empty(0)
                "variable": np.array(INPUT_VARIABLES),
                # TODO: Set grid dimensions from reference
                "lat": np.linspace(90, -90, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    # =========================================================================
    # 3. OUTPUT COORDINATES
    # =========================================================================
    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinates to validate and transform.

        Returns
        -------
        CoordSystem
            Output coordinates (typically with different variables).

        Raises
        ------
        ValueError
            If input coordinates are invalid.
        """
        target_input_coords = self.input_coords()

        # Validate dimensions exist at correct indices
        # For diagnostics: batch=0, variable=1, lat=2, lon=3
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)

        # Validate coordinate values match expected
        handshake_coords(input_coords, target_input_coords, "variable")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "lon")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(OUTPUT_VARIABLES)
        return output_coords

    # =========================================================================
    # 4. LOAD DEFAULT PACKAGE
    # =========================================================================
    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package on HuggingFace/NGC/S3.

        Returns
        -------
        Package
            Model package with checkpoint files.

        Warning
        -------
        NEVER commit credentials, API keys, or secrets to this file.
        """
        # TODO: Replace with actual checkpoint URL
        return Package(
            "ngc://models/nvidia/modulus/model_name@v1.0",
            cache_options={
                "cache_storage": Package.default_cache("model_name"),
                "same_names": True,
            },
        )

    # =========================================================================
    # 5. LOAD MODEL
    # =========================================================================
    @classmethod
    @check_optional_dependencies()
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
        # Resolve checkpoint files from package
        model_path = package.resolve("model.pt")
        center_path = package.resolve("center.npy")
        scale_path = package.resolve("scale.npy")

        # Load model (always to CPU first)
        core_model = torch.load(model_path, map_location="cpu", weights_only=False)
        core_model.eval()
        core_model.requires_grad_(False)

        # Load normalization parameters
        center = torch.from_numpy(np.load(center_path))
        scale = torch.from_numpy(np.load(scale_path))

        return cls(core_model, center=center, scale=scale)

    # =========================================================================
    # 6. DEVICE MANAGEMENT (optional)
    # =========================================================================
    # NOTE: Only override .to() if there is non-PyTorch state to manage.
    # For pure PyTorch wrappers, inherited .to() is sufficient.

    # =========================================================================
    # 7. PRIVATE/SUPPORT METHODS
    # =========================================================================
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor using single center/scale pair."""
        return (x - self.center) / self.scale

    # =========================================================================
    # 8. SINGLE-STEP FORWARD
    # =========================================================================
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, variable, lat, lon).
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinates.
        """
        output_coords = self.output_coords(coords)

        with torch.no_grad():
            # Move to device
            device = next(self.parameters()).device
            x = x.to(device)

            # Normalize
            x = self._normalize(x)

            # Run forward pass
            out = self.core_model(x)

        return out, output_coords


# ===========================================================================
# TEMPLATE B: Generative Diagnostic (with samples, e.g., CorrDiff-like)
# ===========================================================================


class GenerativeDiagnostic(torch.nn.Module, AutoModelMixin):
    """Generative diagnostic model producing multiple samples.

    This template is for models like CorrDiff that use diffusion or VAE
    to generate multiple samples from a single input. The output has an
    additional "sample" dimension.

    Parameters
    ----------
    residual_model : torch.nn.Module
        Core diffusion/residual model
    regression_model : torch.nn.Module
        Optional regression model for mean prediction
    in_center : torch.Tensor
        Input center normalization
    in_scale : torch.Tensor
        Input scale normalization
    out_center : torch.Tensor
        Output center normalization
    out_scale : torch.Tensor
        Output scale normalization
    lat_input : torch.Tensor
        Input latitude grid
    lon_input : torch.Tensor
        Input longitude grid
    lat_output : torch.Tensor
        Output latitude grid
    lon_output : torch.Tensor
        Output longitude grid
    number_of_samples : int
        Number of samples to generate per input
    seed : int | None
        Random seed for reproducibility

    Note
    ----
    See CorrDiff for a complete generative diagnostic implementation.

    Badges
    ------
    region:global class:sr product:wind year:2024 gpu:80gb
    """

    # =========================================================================
    # 1. CONSTRUCTOR
    # =========================================================================
    def __init__(
        self,
        residual_model: torch.nn.Module,
        regression_model: torch.nn.Module | None,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        lat_input: torch.Tensor,
        lon_input: torch.Tensor,
        lat_output: torch.Tensor,
        lon_output: torch.Tensor,
        input_variables: list[str],
        output_variables: list[str],
        number_of_samples: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.residual_model = residual_model
        self.regression_model = regression_model
        self.number_of_samples = number_of_samples
        self.seed = seed
        self.input_variables = input_variables
        self.output_variables = output_variables

        # Register buffers
        self.register_buffer("in_center", in_center)
        self.register_buffer("in_scale", in_scale)
        self.register_buffer("out_center", out_center)
        self.register_buffer("out_scale", out_scale)
        self.register_buffer("lat_input", lat_input)
        self.register_buffer("lon_input", lon_input)
        self.register_buffer("lat_output", lat_output)
        self.register_buffer("lon_output", lon_output)

    # =========================================================================
    # 2. INPUT COORDINATES
    # =========================================================================
    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(self.input_variables),
                "lat": self.lat_input.cpu().numpy(),
                "lon": self.lon_input.cpu().numpy(),
            }
        )

    # =========================================================================
    # 3. OUTPUT COORDINATES
    # =========================================================================
    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system with sample dimension.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinates to validate and transform.

        Returns
        -------
        CoordSystem
            Output coordinates including sample dimension.
        """
        target_input_coords = self.input_coords()

        # Validate input coordinates
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target_input_coords, "variable")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "lon")

        # Output includes sample dimension
        output_coords = OrderedDict(
            {
                "batch": input_coords["batch"],
                "sample": np.arange(self.number_of_samples),
                "variable": np.array(self.output_variables),
                "lat": self.lat_output.cpu().numpy(),
                "lon": self.lon_output.cpu().numpy(),
            }
        )
        return output_coords

    # =========================================================================
    # 4. LOAD DEFAULT PACKAGE
    # (Same pattern as DeterministicDiagnostic but with generative model URL)
    # =========================================================================
    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained generative model package on NGC.

        Returns
        -------
        Package
            Model package with diffusion/VAE checkpoint files.

        Note
        ----
        Generative models typically have larger checkpoints and may include
        both a regression model and a residual/diffusion model.
        """
        # TODO: Replace with actual generative model checkpoint URL
        return Package(
            "ngc://models/nvidia/modulus/generative_model@v1.0",
            cache_options={
                "cache_storage": Package.default_cache("generative_model"),
                "same_names": True,
            },
        )

    # =========================================================================
    # 5. LOAD MODEL
    # =========================================================================
    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load generative diagnostic from package."""
        # TODO: Implement actual loading logic
        # See CorrDiff.load_model for a complete example
        raise NotImplementedError

    # =========================================================================
    # 6. DEVICE MANAGEMENT
    # =========================================================================
    def to(self, device: torch.device | str) -> DiagnosticModel:
        """Move model to device.

        Parameters
        ----------
        device : torch.device | str
            Target device.

        Returns
        -------
        DiagnosticModel
            Model on target device.
        """
        super().to(device)
        if self.residual_model is not None:
            self.residual_model.to(device)
        if self.regression_model is not None:
            self.regression_model.to(device)
        return self

    # =========================================================================
    # 7. PRIVATE/SUPPORT METHODS
    # =========================================================================
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor using separate input center/scale (generative pattern)."""
        return (x - self.in_center) / self.in_scale

    def _denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor using separate output center/scale (generative pattern)."""
        return x * self.out_scale + self.out_center

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward pass generating samples.

        Parameters
        ----------
        x : torch.Tensor
            Single input tensor [var, lat, lon].

        Returns
        -------
        torch.Tensor
            Generated samples [samples, var, lat, lon].
        """
        # Normalize input
        x = self._normalize_input(x.unsqueeze(0))

        # Generate samples
        samples = []
        for i in range(self.number_of_samples):
            seed = self.seed + i if self.seed is not None else None  # noqa: F841
            # TODO: Implement actual sample generation
            # For diffusion: run sampler with seed
            sample = x  # Placeholder
            samples.append(sample)

        # Stack samples
        out = torch.cat(samples, dim=0)

        # Denormalize
        out = self._denormalize_output(out)

        return out

    # =========================================================================
    # 8. SINGLE-STEP FORWARD
    # =========================================================================
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass generating samples.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch, variable, lat, lon).
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor (batch, sample, variable, lat, lon) and coordinates.
        """
        output_coords = self.output_coords(coords)

        with torch.no_grad():
            # Allocate output tensor
            out = torch.zeros(
                [len(v) for v in output_coords.values()],
                device=x.device,
                dtype=torch.float32,
            )

            # Generate samples for each batch element
            for i in range(out.shape[0]):
                out[i] = self._forward(x[i])

        return out, output_coords


# ===========================================================================
# TEMPLATE C: Simple Diagnostic (no AutoModel, no checkpoints)
# ===========================================================================


class SimpleDiagnostic(torch.nn.Module):
    """Simple diagnostic that computes a derived quantity.

    Use this template for diagnostics that don't need checkpoints,
    such as computing wind speed from u/v components.

    Note: No AutoModelMixin inheritance needed.
    """

    def __init__(self) -> None:
        super().__init__()

    def input_coords(self) -> CoordSystem:
        """Input coordinate system."""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(["u10m", "v10m"]),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system."""
        target = self.input_coords()
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(["ws10m"])  # Wind speed
        return output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Compute wind speed from u/v components."""
        output_coords = self.output_coords(coords)

        with torch.no_grad():
            # x shape: (batch, 2, lat, lon) where variable=["u10m", "v10m"]
            u = x[:, 0:1, :, :]
            v = x[:, 1:2, :, :]
            ws = torch.sqrt(u**2 + v**2)

        return ws, output_coords
