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
from collections.abc import Generator, Iterator
from typing import Protocol

import numpy as np
import torch
from loguru import logger

from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import handshake_coords, handshake_dim, map_coords
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.type import CoordSystem


def _convert_to_2d(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (lat.ndim == 1) and (lon.ndim == 1):
        return np.meshgrid(lat, lon, indexing="ij")
    else:
        return (lat, lon)


def _can_concat_directly(px_coords: CoordSystem, dx_coords: CoordSystem) -> bool:
    try:
        for i, key in enumerate(dx_coords.keys()):
            handshake_dim(px_coords, key, i)
            if key != "variable":
                handshake_coords(px_coords, dx_coords, key)
        return True
    except (KeyError, ValueError):
        return False


def _can_concat_with_subregion(px_coords: CoordSystem, dx_coords: CoordSystem) -> bool:
    try:
        for i, key in enumerate(dx_coords.keys()):
            handshake_dim(px_coords, key, i)
            if key not in ["variable", "lat", "lon"]:
                handshake_coords(px_coords, dx_coords, key)

        # Check if dx lat/lon is a subregion of px lat/lon (1D grids only)
        if not px_coords["lat"].ndim == 1 or not px_coords["lon"].ndim == 1:
            return False

        lat_idx0 = np.where(px_coords["lat"] == dx_coords["lat"][0])[0][0]
        lat_idx1 = lat_idx0 + dx_coords["lat"].shape[0]
        lon_idx0 = np.where(px_coords["lon"] == dx_coords["lon"][0])[0][0]
        lon_idx1 = lon_idx0 + dx_coords["lon"].shape[0]

        return np.all(
            dx_coords["lat"] == px_coords["lat"][lat_idx0:lat_idx1]
        ) and np.all(dx_coords["lon"] == px_coords["lon"][lon_idx0:lon_idx1])

    except (KeyError, ValueError, IndexError):
        return False


class PrepareInputCoordsDefault:
    """Prepares output coords from prognostic model for diagnostic models"""

    def __call__(self, px_coords: CoordSystem, dx_coords: CoordSystem) -> CoordSystem:
        """Prepare coordinates for diagnostic model input.

        Parameters
        ----------
        px_coords : CoordSystem
            Output coordinates from the prognostic model
        dx_coords : CoordSystem
            Diagnostic model input coordinate system

        Returns
        -------
        CoordSystem
            Prepared coordinate system for diagnostic model
        """
        # Handling np.empty (free coordinate system)
        if dx_coords["lat"].shape[0] == 0:
            dx_coords["lat"] = px_coords["lat"]
        if dx_coords["lon"].shape[0] == 0:
            dx_coords["lon"] = px_coords["lon"]

        for key, value in dx_coords.items():
            if key in ["variable", "lat", "lon"] and key in px_coords:
                px_coords[key] = value

        return px_coords


class PrepareInputTensorDefault:
    """Prepares output from prognostic model for diagnostic"""

    def __init__(self) -> None:
        super().__init__()
        self.interp: torch.nn.Module | None = None

    @torch.inference_mode()
    def __call__(
        self, x: torch.Tensor, px_coords: CoordSystem, dx_coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepare tensor for diagnostic model input with interpolation.

        Parameters
        ----------
        x : torch.Tensor
            Output of prognostic model from a single step
        px_coords : CoordSystem
            Output coordinates from the prognostic model
        dx_coords : CoordSystem
            Diagnostic model input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Prepared tensor and coordinate system for diagnostic model
        """
        if "lat" not in px_coords:
            raise KeyError("'lat' not found in prognostic model output coordinates")
        if "lon" not in px_coords:
            raise KeyError("'lon' not found in prognostic model output coordinates")
        if "lat" not in dx_coords:
            raise KeyError("'lat' not found in diagnostic model input coordinates")
        if "lon" not in dx_coords:
            raise KeyError("'lon' not found in diagnostic model input coordinates")

        # Handling np.empty (free coordinate system)
        if dx_coords["lat"].shape[0] == 0:
            dx_coords["lat"] = px_coords["lat"]
        if dx_coords["lon"].shape[0] == 0:
            dx_coords["lon"] = px_coords["lon"]

        if self.interp is None:
            (lat0, lon0) = _convert_to_2d(px_coords["lat"], px_coords["lon"])
            (lat1, lon1) = _convert_to_2d(dx_coords["lat"], dx_coords["lon"])

            # Check if coordinates are identical - if so, skip interpolation
            if np.array_equal(lat0, lat1) and np.array_equal(lon0, lon1):
                self.interp = lambda x: x
            else:
                self.interp = LatLonInterpolation(lat0, lon0, lat1, lon1).to(x.device)

        x = self.interp(x)
        coords = px_coords.copy()
        coords["lat"] = dx_coords["lat"]
        coords["lon"] = dx_coords["lon"]
        # Map remaining coords
        try:
            x, coords = map_coords(x, coords, dx_coords)
        except Exception as e:
            logger.error(
                "Mapping between prognostic output and diagnostic input failed. Check coordinate systems are compatible."
            )
            raise e

        return x, coords


class PrepareOutputCoordsDefault:
    """Preparing output coordinates of the diagnostic wrapper"""

    def __call__(
        self, px_coords: CoordSystem, dx_coords: list[CoordSystem]
    ) -> CoordSystem:
        """Returns the output coordinates of the diagnostic wrapper

        Parameters
        ----------
        px_coords : CoordSystem
            Prognostic coords
        dx_coords : list[CoordSystem]
            Diagnostic coords

        Returns
        -------
        CoordSystem
            Expected output coords from model for a given time-step
        """
        dx_target = dx_coords[-1]
        if _can_concat_directly(px_coords, dx_target):
            variables = [px_coords["variable"]] + [c["variable"] for c in dx_coords]
        elif _can_concat_with_subregion(px_coords, dx_target):
            variables = [px_coords["variable"]] + [c["variable"] for c in dx_coords]
        else:
            variables = [c["variable"] for c in dx_coords]

        coords = dx_coords[-1].copy()
        coords["variable"] = np.concatenate(variables)
        return coords


class PrepareOutputTensorDefault(torch.nn.Module):
    """Preparing output tensor / coords of the diagnostic wrapper. This default
    implementation offers the following three strategies for preparing the output:

    1. Attempt to concat px outputs and all dx outputs
    2. Attempt to concat lat/lon sub-region px outputs and all dx outputs
    3. Concat just dx outputs
    """

    @torch.inference_mode()
    def forward(
        self,
        px_x: torch.Tensor,
        px_coords: CoordSystem,
        dx_x: list[torch.Tensor],
        dx_coords: list[CoordSystem],
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepare outputs for diagnostic wrapper

        Parameters
        ----------
        px_x : torch.Tensor
            Output of prognostic model from a single step
        px_coords : CoordSystem
            Output coordinates from the prognostic model
        dx_x: list[torch.Tensor]
            Output of diagnostic model
        dx_coords : list[CoordSystem]
            Diagnostic model input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Outputs to be returned by the wrapper
        """
        dx_target = dx_coords[-1]

        # Attempt various concat strategies
        if _can_concat_directly(px_coords, dx_target):
            x = [px_x] + dx_x
            variables = [px_coords["variable"]] + [c["variable"] for c in dx_coords]
        elif _can_concat_with_subregion(px_coords, dx_target):
            # Find the dimension positions of lat and lon in px_coords
            lat_dim = list(px_coords.keys()).index("lat")
            lon_dim = list(px_coords.keys()).index("lon")

            # Find the slice indices
            lat_idx0 = np.where(px_coords["lat"] == dx_target["lat"][0])[0][0]
            lat_idx1 = lat_idx0 + dx_target["lat"].shape[0]
            lon_idx0 = np.where(px_coords["lon"] == dx_target["lon"][0])[0][0]
            lon_idx1 = lon_idx0 + dx_target["lon"].shape[0]

            # Slice px_x along the lat and lon dimensions
            slices = [slice(None)] * px_x.ndim
            slices[lat_dim] = slice(lat_idx0, lat_idx1)
            slices[lon_dim] = slice(lon_idx0, lon_idx1)
            px_x_sliced = px_x[tuple(slices)]

            x = [px_x_sliced] + dx_x
            variables = [px_coords["variable"]] + [c["variable"] for c in dx_coords]
        else:
            x = dx_x
            variables = [c["variable"] for c in dx_coords]

        try:
            x = torch.concat(x, dim=list(dx_target).index("variable"))
            coords = dx_target.copy()
            coords["variable"] = np.concatenate(variables)
        except RuntimeError as e:
            logger.error(
                "Failed to concatenate outputs of diagnostic models. "
                "The outputs of the models cannot be concatenated."
            )
            raise e
        return x, coords


class PrepareDxInputCoords(Protocol):
    """Protocol for preparing diagnostic model input coordinates."""

    def __call__(
        self, px_coords: CoordSystem, dx_coords: CoordSystem
    ) -> CoordSystem: ...


class PrepareDxInputTensor(Protocol):
    """Protocol for preparing diagnostic model input tensor."""

    def __call__(
        self, x: torch.Tensor, px_coords: CoordSystem, dx_coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]: ...


class PrepareOutputCoords(Protocol):
    """Protocol for preparing output coordinates."""

    def __call__(
        self, px_coords: CoordSystem, dx_coords: list[CoordSystem]
    ) -> CoordSystem: ...


class PrepareOutputTensor(Protocol):
    """Protocol for preparing output tensor."""

    def __call__(
        self,
        px_x: torch.Tensor,
        px_coords: CoordSystem,
        dx_x: list[torch.Tensor],
        dx_coords: list[CoordSystem],
    ) -> tuple[torch.Tensor, CoordSystem]: ...


class DiagnosticWrapper(torch.nn.Module, PrognosticMixin):
    """Wraps a prognostic model and one or more diagnostic models into a single
    prognostic model. The micro-pipeline this wrapper encapsulates has the following
    four steps:

    1. Execute one step of the prognostic model
    2. Prepare output of prognostic model for each diagnostic model
    3. Execute forward pass each diagnostic model using the prepare prognostic data
    4. Prepare outputs of prognostic/diagnostic for final return

    The wrapper provides customizable methods for preparing diagnostic model inputs and
    outputs. If not provided, default methods are have the following requirements:

    - All diagnostics must have the same output coordinate systems with the exception
    of the variable dimension
    - Both the prognostic and diagnostic models must have lat/lon grid systems.

    Note
    ----
    Custom callables or classes implementing the Protocol interfaces can be provided to
    override default behavior such as skipping interpolation or changing concatenation
    logic. This will be required for many diagnostic models. The prepare functions must
    implement the appropriate Protocol (__call__ method with matching signature):

    - PrepareDxInputCoords: Prepares coordinate systems
    - PrepareDxInputTensor: Prepares tensors with optional interpolation
    - PrepareOutputCoords: Prepares final output coordinate systems
    - PrepareOutputTensor: Prepares final output tensors

    Parameters
    ----------
    px_model : PrognosticModel
        The prognostic model to use as the base model.
    dx_model : DiagnosticModel | list[DiagnosticModel]
        Single diagnostic model or list of diagnostic models whose outputs are
        concatenated to the prognostic model output.
    prepare_dx_input_coords : PrepareDxInputCoords | list[PrepareDxInputCoords] | None, optional
        Callable or Protocol-implementing object to prepare coordinate system for
        diagnostic model input. Can be a single instance (applied to all diagnostics)
        or a list (one per diagnostic). If None, uses PrepareInputCoordsDefault for
        each diagnostic, by default None
    prepare_dx_input_tensor : PrepareDxInputTensor | list[PrepareDxInputTensor] | None, optional
        Callable or Protocol-implementing object to prepare tensor for diagnostic model
        input. Can be a single instance (applied to all diagnostics) or a list (one per
        diagnostic). If None, uses PrepareInputTensorDefault with interpolation for
        each diagnostic, by default None
    prepare_output_coords : PrepareOutputCoords | None, optional
        Callable or Protocol-implementing object to prepare output coordinate system.
        If None, uses PrepareOutputCoordsDefault which concatenates all variables,
        by default None
    prepare_output_tensor : PrepareOutputTensor | None, optional
        Callable or Protocol-implementing object to prepare output tensor. If None,
        uses PrepareOutputTensorDefault which concatenates all outputs, by default None
    """

    def __init__(
        self,
        px_model: PrognosticModel,
        dx_model: DiagnosticModel | list[DiagnosticModel],
        prepare_dx_input_coords: (
            PrepareDxInputCoords | list[PrepareDxInputCoords] | None
        ) = None,
        prepare_dx_input_tensor: (
            PrepareDxInputTensor | list[PrepareDxInputTensor] | None
        ) = None,
        prepare_output_coords: PrepareOutputCoords | None = None,
        prepare_output_tensor: PrepareOutputTensor | None = None,
    ):
        super().__init__()

        self.px_model = px_model
        if not isinstance(dx_model, list):
            dx_model = [dx_model]
        self.dx_model = torch.nn.ModuleList(dx_model)

        # Set up the prepare / map functions if not provided
        # prepare px -> dx coordinates
        if prepare_dx_input_coords is None:
            prepare_dx_input_coords = [
                PrepareInputCoordsDefault() for _ in self.dx_model
            ]
        elif not isinstance(prepare_dx_input_coords, list):
            prepare_dx_input_coords = [prepare_dx_input_coords]

        # prepare px -> dx input tensors
        if prepare_dx_input_tensor is None:
            prepare_dx_input_tensor = [
                PrepareInputTensorDefault() for _ in self.dx_model
            ]
        elif not isinstance(prepare_dx_input_tensor, list):
            prepare_dx_input_tensor = [prepare_dx_input_tensor]

        # prepare final output tensors
        if prepare_output_coords is None:
            prepare_output_coords = PrepareOutputCoordsDefault()
        if prepare_output_tensor is None:
            prepare_output_tensor = PrepareOutputTensorDefault()

        self.prepare_dx_input_coords = prepare_dx_input_coords
        self.prepare_dx_input_tensor = prepare_dx_input_tensor
        self.prepare_output_coords = prepare_output_coords
        self.prepare_output_tensor = prepare_output_tensor

        # Validate lengths match number of diagnostic models
        if len(self.prepare_dx_input_coords) != len(self.dx_model):
            raise ValueError(
                f"Length of prepare_dx_input_coords ({len(self.prepare_dx_input_coords)}) "
                f"must match number of diagnostic models ({len(self.dx_model)})"
            )
        if len(self.prepare_dx_input_tensor) != len(self.dx_model):
            raise ValueError(
                f"Length of prepare_dx_input_tensor ({len(self.prepare_dx_input_tensor)}) "
                f"must match number of diagnostic models ({len(self.dx_model)})"
            )

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        # Common dim we should always request
        input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.empty(0),
                "variable": np.empty(0),
            }
        )
        for key, value in self.px_model.input_coords().items():
            input_coords[key] = value

        return input_coords

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        px_coords = self.px_model.output_coords(input_coords)
        dx_coords = []
        for model, prepare_dx_input in zip(self.dx_model, self.prepare_dx_input_coords):
            # This is kinda annnoying at the moment, but I'm not sure of a better way yet
            # I wish we could just use prepare_dx_input_tensor but we have no tensors
            coords = prepare_dx_input(px_coords.copy(), model.input_coords())
            dx_coords.append(model.output_coords(coords))
        out_coords = self.prepare_output_coords(px_coords, dx_coords)
        return out_coords

    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        ------
        x : torch.Tensor
        coords : CoordSystem
        """
        px_x, px_coords = self.px_model(x, coords)
        dx_x = []
        dx_coords = []
        for model, prepare_dx_input in zip(self.dx_model, self.prepare_dx_input_tensor):
            dx_x0, dx_coords0 = prepare_dx_input(
                px_x, px_coords.copy(), model.input_coords()
            )
            dx_x0, dx_coords0 = model(dx_x0, dx_coords0)
            dx_x.append(dx_x0)
            dx_coords.append(dx_coords0)
        x, coords = self.prepare_output_tensor(px_x, px_coords, dx_x, dx_coords)
        return x, coords

    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        for px_x, px_coords in self.px_model.create_iterator(x, coords):
            dx_x = []
            dx_coords = []
            for model, prepare_dx_input in zip(
                self.dx_model, self.prepare_dx_input_tensor
            ):
                dx_x0, dx_coords0 = prepare_dx_input(
                    px_x, px_coords.copy(), model.input_coords()
                )
                dx_x0, dx_coords0 = model(dx_x0, dx_coords0)
                dx_x.append(dx_x0)
                dx_coords.append(dx_coords0)
            x, coords = self.prepare_output_tensor(px_x, px_coords, dx_x, dx_coords)
            yield x, coords

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)
