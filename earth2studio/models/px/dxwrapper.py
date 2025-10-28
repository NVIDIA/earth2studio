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

from collections.abc import Callable, Generator, Iterator

import numpy as np
import torch
from loguru import logger

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import handshake_coords, handshake_dim, map_coords
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.type import CoordSystem

PrepareDxInputCoords = Callable[[CoordSystem, CoordSystem], CoordSystem]
PrepareDxInputTensor = Callable[
    [torch.Tensor, CoordSystem, CoordSystem], tuple[torch.Tensor, CoordSystem]
]
PrepareOutputCoords = Callable[[CoordSystem, list[CoordSystem]], CoordSystem]
PrepareOutputTensor = Callable[
    [torch.Tensor, CoordSystem, list[torch.Tensor], list[CoordSystem]],
    tuple[torch.Tensor, CoordSystem],
]


class DiagnosticWrapper(torch.nn.Module, PrognosticMixin):
    """Wraps a prognostic model and one or more diagnostic models into a single
    prognostic model. The micro-pipeline this wrapper encapsulates has the following
    four steps:

    1. Execute one step of the prognostic model
    2. Prepare output of prognostic model for each diagnostic model
    3. Execute forward pass each diagnostic model
    4. Prepare outputs of prognostic/diagnostic for final return

    The wrapper provides customizable methods for preparing diagnostic model inputs and
    outputs. If not provided, default methods are have the following requirements:

    - All diagnostics must have the same output coordinate systems with the exception
    of the variable dimension
    - Both the prognostic and diagnostic models must have lat/lon grid systems.

    Note
    ----
    Custom callables can be provided to override default behavior such as skipping
    interpolation or change concatenation logic. This will be required for many
    diagnostic models. Refer to this classes internal default functions to understand
    the required function signature.

    Parameters
    ----------
    px_model : PrognosticModel
        The prognostic model to use as the base model.
    dx_model : DiagnosticModel | list[DiagnosticModel]
        Single diagnostic model or list of diagnostic models whose outputs are
        concatenated to the prognostic model output.
    prepare_dx_input_coords : PrepareDxInputCoords | None, optional
        Callable to prepare coordinate system for diagnostic model input. If None,
        uses default method, by default None
    prepare_dx_input_tensor : PrepareDxInputTensor | None, optional
        Callable to prepare tensor for diagnostic model input. If None, uses default
        method with interpolation, by default None
    prepare_output_coords : PrepareOutputCoords | None, optional
        Callable to prepare output coordinate system. If None, uses default method
        which concatenates all variables, by default None
    prepare_output_tensor : PrepareOutputTensor | None, optional
        Callable to prepare output tensor. If None, uses default method which
        concatenates all outputs, by default None
    """

    def __init__(
        self,
        px_model: PrognosticModel,
        dx_model: DiagnosticModel | list[DiagnosticModel],
        prepare_dx_input_coords: PrepareDxInputCoords | None = None,
        prepare_dx_input_tensor: PrepareDxInputTensor | None = None,
        prepare_output_coords: PrepareOutputCoords | None = None,
        prepare_output_tensor: PrepareOutputTensor | None = None,
    ):
        super().__init__()

        self.px_model = px_model
        if not isinstance(dx_model, list):
            dx_model = [dx_model]
        self.dx_model = torch.nn.ModuleList(dx_model)

        if prepare_dx_input_coords is None:
            prepare_dx_input_coords = self._default_prepare_dx_input_coords
        if prepare_dx_input_tensor is None:
            prepare_dx_input_tensor = self._default_prepare_dx_input_tensor
        if prepare_output_coords is None:
            prepare_output_coords = self._default_prepare_output_coords
        if prepare_output_tensor is None:
            prepare_output_tensor = self._default_prepare_output_tensor

        self.prepare_dx_input_coords = prepare_dx_input_coords
        self.prepare_dx_input_tensor = prepare_dx_input_tensor
        self.prepare_output_coords = prepare_output_coords
        self.prepare_output_tensor = prepare_output_tensor

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self.px_model.input_coords()

    @batch_coords()
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
        for model in self.dx_model:
            # This is kinda annnoying at the moment, but I'm not sure of a better way yet
            coords = self.prepare_dx_input_coords(
                px_coords.copy(), model.input_coords()
            )
            dx_coords.append(model.output_coords(coords))
        out_coords = self.prepare_output_coords(px_coords, dx_coords)
        return out_coords

    @torch.inference_mode()
    def _default_prepare_dx_input_coords(
        self, px_coords: CoordSystem, dx_coords: CoordSystem
    ) -> CoordSystem:
        """Default coordinate preparation for diagnostic model.
        Just naively replaces variable, lat, lon coords with those required by the
        diagnostic model

        Parameters
        ----------
        px_coords : CoordSystem
            Output coordinates from the prognostic model
        dx_coords : CoordSystem
            Diagnostic model input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Inputs to prognostic model
        """
        for key, value in dx_coords.items():
            if key in ["variable", "lat", "lon"] and key in px_coords:
                px_coords[key] = value

        return px_coords

    @torch.inference_mode()
    def _default_prepare_dx_input_tensor(
        self, x: torch.Tensor, px_coords: CoordSystem, dx_coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """This default method will attempt to interpolate / extrapolate lat lon
        coordinates

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
            Inputs to prognostic model
        """
        if "lat" not in px_coords:
            raise KeyError("'lat' not found in prognostic model output coordinates")
        if "lon" not in px_coords:
            raise KeyError("'lon' not found in prognostic model output coordinates")
        if "lat" not in dx_coords:
            raise KeyError("'lat' not found in diagnostic model input coordinates")
        if "lon" not in dx_coords:
            raise KeyError("'lon' not found in diagnostic model input coordinates")

        def _convert_to_2d(
            lat: np.ndarray, lon: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if (lat.ndim == 1) and (lon.ndim == 1):
                return np.meshgrid(lat, lon, indexing="ij")
            else:
                return (lat, lon)

        (lat0, lon0) = _convert_to_2d(px_coords["lat"], px_coords["lon"])
        (lat1, lon1) = _convert_to_2d(dx_coords["lat"], dx_coords["lon"])

        if not hasattr(self, "interp") or self.interp is None:  # type: ignore
            self.interp = LatLonInterpolation(lat0, lon0, lat1, lon1).to(x.device)

        x = self.interp(x)
        coords = px_coords.copy()
        coords["lat"] = dx_coords["lat"]
        coords["lon"] = dx_coords["lon"]
        # Map remaining coords
        x, coords = map_coords(x, coords, dx_coords)

        return x, coords

    def _default_prepare_output_coords(
        self, px_coords: CoordSystem, dx_coords: list[CoordSystem]
    ) -> CoordSystem:
        """Returns the output coordinates of the diagnostic wrapper

        Parameters
        ----------
        px_coords : CoordSystem
            Prognostic coords
        dx_coords : CoordSystem
            Diagnostic coords

        Returns
        -------
        CoordSystem
            Expected output coords from model for a given time-step
        """
        try:
            for i, key in enumerate(dx_coords[-1].keys()):
                handshake_dim(px_coords, key, i)
                if not key == "variable":
                    handshake_coords(px_coords, dx_coords[-1], key)
            # Concat all variables
            variables = [px_coords["variable"]] + [
                coord["variable"] for coord in dx_coords
            ]
        except (KeyError, ValueError):
            variables = [coord["variable"] for coord in dx_coords]

        coords = dx_coords[-1].copy()
        coords["variable"] = np.concatenate(variables)
        return coords

    @torch.inference_mode()
    def _default_prepare_output_tensor(
        self,
        px_x: torch.Tensor,
        px_coords: CoordSystem,
        dx_x: list[torch.Tensor],
        dx_coords: list[CoordSystem],
    ) -> tuple[torch.Tensor, CoordSystem]:
        """This default method will attempt concat the prognostic and diagnostic
        outputs if the coordinate systems will allow it

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
        try:
            for i, key in enumerate(dx_coords[-1].keys()):
                handshake_dim(px_coords, key, i)
                if not key == "variable":
                    handshake_coords(px_coords, dx_coords[-1], key)

            x = [px_x] + dx_x
            variables = [px_coords["variable"]] + [
                coord["variable"] for coord in dx_coords
            ]
        except (KeyError, ValueError):
            x = dx_x
            variables = [coord["variable"] for coord in dx_coords]

        try:
            x = torch.concat(x, dim=list(dx_coords[-1]).index("variable"))
            coords = dx_coords[-1].copy()
            coords["variable"] = np.concatenate(variables)
        except RuntimeError as e:
            logger.error(
                "Failed to concatenate outputs of diagnostic models. "
                "The outputs of the models cannot be concatenated."
            )
            raise e
        return x, coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Coordinate system, should have dimensions ``[time, variable, *domain_dims]``

        Returns
        ------
        x : torch.Tensor
        coords : CoordSystem
        """
        px_x, px_coords = self.px_model(x, coords)
        dx_x = []
        dx_coords = []
        for model in self.dx_model:
            dx_x0, dx_coords0 = self.prepare_dx_input_tensor(
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
            for model in self.dx_model:
                dx_x0, dx_coords0 = self.prepare_dx_input_tensor(
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
