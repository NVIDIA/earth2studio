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

"""
Method Implementation Templates for Prognostic Model Wrappers

This file contains fully-documented method templates that can be adapted for
specific model implementations. Each method includes complete docstrings with
Parameters, Returns, and Raises sections following NumPy style.
"""

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import torch

from earth2studio.models.auto import Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.type import CoordSystem

VARIABLES = [...]  # Replace with actual variable list


# =============================================================================
# COORDINATE SYSTEM METHODS
# =============================================================================


def input_coords(self) -> CoordSystem:
    """Input coordinate system of the prognostic model.

    Returns
    -------
    CoordSystem
        Coordinate system dictionary
    """
    return OrderedDict(
        {
            "batch": np.empty(0),  # MUST be first, MUST be np.empty(0)
            "time": np.empty(0),  # Dynamic time dimension
            "lead_time": np.array([np.timedelta64(0, "h")]),  # Initial lead time (0h)
            "variable": np.array(VARIABLES),
            "lat": np.linspace(90, -90, num_lat, endpoint=...),  # From reference
            "lon": np.linspace(0, 360, num_lon, endpoint=False),  # From reference
        }
    )


@batch_coords()
def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
    """Output coordinate system.

    Parameters
    ----------
    input_coords : CoordSystem
        Input coordinates to validate and transform

    Returns
    -------
    CoordSystem
        Output coordinates with updated lead_time

    Raises
    ------
    ValueError
        If input coordinates are invalid
    """
    target_input_coords = self.input_coords()

    # Validate dimensions exist at correct indices
    handshake_dim(input_coords, "lead_time", 2)
    handshake_dim(input_coords, "variable", 3)
    handshake_dim(input_coords, "lat", 4)
    handshake_dim(input_coords, "lon", 5)

    # Validate coordinate values match
    handshake_coords(input_coords, target_input_coords, "variable")
    handshake_coords(input_coords, target_input_coords, "lat")
    handshake_coords(input_coords, target_input_coords, "lon")

    output_coords = input_coords.copy()
    output_coords["lead_time"] = input_coords["lead_time"] + np.array([self._time_step])
    return output_coords


# =============================================================================
# FORWARD PASS METHODS
# =============================================================================


@batch_func()
def __call__(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    """Run prognostic model 1 step.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    coords : CoordSystem
        Input coordinate system

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Output tensor and coordinates one time step ahead
    """
    target_input_coords = self.input_coords()
    handshake_coords(coords, target_input_coords, "variable")
    handshake_dim(coords, "variable", 3)

    # Move to device
    device = self.device_buffer.device
    x = x.to(device)

    # Run forward pass
    # TODO: Reshape input tensor for the core model
    # TODO: Call core model
    # TODO: Reshape output tensor back to earth2studio format

    out_coords = self.output_coords(coords)
    return output, out_coords


@batch_func()
def _default_generator(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
    """Batch-decorated generator for time integration.

    Parameters
    ----------
    x : torch.Tensor
        Initial condition tensor
    coords : CoordSystem
        Initial coordinate system

    Yields
    ------
    tuple[torch.Tensor, CoordSystem]
        Predicted state and coordinates at each time step
    """
    # MUST yield initial condition first (step 0)
    yield x, coords

    # Time integration loop (runs indefinitely)
    current_x = x
    current_coords = coords
    while True:
        # Apply front hook (for perturbation injection, etc.)
        current_x, current_coords = self.front_hook(current_x, current_coords)

        # Forward step
        current_x, current_coords = self.__call__(current_x, current_coords)

        # Apply rear hook (for post-processing, etc.)
        current_x, current_coords = self.rear_hook(current_x, current_coords)

        yield current_x, current_coords


def create_iterator(
    self,
    x: torch.Tensor,
    coords: CoordSystem,
) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
    """Create time-integration iterator.

    Parameters
    ----------
    x : torch.Tensor
        Initial condition tensor
    coords : CoordSystem
        Initial coordinate system

    Yields
    ------
    tuple[torch.Tensor, CoordSystem]
        Predicted state and coordinates at each time step
    """
    yield from self._default_generator(x, coords)


# =============================================================================
# MODEL LOADING METHODS
# =============================================================================


@classmethod
def load_default_package(cls) -> Package:
    """Default pre-trained model package on <source>.

    Returns
    -------
    Package
        Model package
    """
    return Package(
        "hf://org/repo@commit",  # or ngc://, s3://, local path
        cache_options={
            "cache_storage": Package.default_cache("model_name"),
            "same_names": True,
        },
    )


@classmethod
@check_optional_dependencies()
def load_model(
    cls,
    package: Package,
) -> PrognosticModel:
    """Load prognostic model from package.

    Parameters
    ----------
    package : Package
        Model package with checkpoint files

    Returns
    -------
    PrognosticModel
        Loaded model instance
    """
    # Resolve checkpoint files
    checkpoint_path = package.resolve("model.pt")

    # Load model
    core_model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    core_model.eval()

    return cls(core_model)


def to(self, device: torch.device | str) -> PrognosticModel:
    """Move model to device.

    Note: Only override this method if there is non-PyTorch state to manage
    (e.g., ONNX Runtime sessions, JAX device placement). For pure PyTorch
    wrappers, super().to(device) is sufficient.

    Parameters
    ----------
    device : torch.device | str
        Target device

    Returns
    -------
    PrognosticModel
        Model on target device
    """
    super().to(device)
    # If using ONNX Runtime, destroy and recreate session on new device
    # If using PyTorch, super().to(device) handles it
    return self
