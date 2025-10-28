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

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import handshake_coords, map_coords
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.type import CoordSystem

PrepareDxInputCoords = Callable[[CoordSystem, CoordSystem], CoordSystem]
PrepareDxInputTensor = Callable[[torch.Tensor, CoordSystem, CoordSystem], torch.Tensor]
PrepareOutputCoords = Callable[[CoordSystem, CoordSystem], CoordSystem]
PrepareOutputTensor = Callable[
    [torch.Tensor, CoordSystem, torch.Tensor, CoordSystem], torch.Tensor
]


class DiagnosticWrapper(torch.nn.Module, PrognosticMixin):
    """Wraps a prognostic model and one or more diagnostic models into a single
    prognostic model. This allows diagnostic model outputs to be included in workflows
    that expect a prognostic model.

    The outputs of the diagnostic models are concatenated the output of the prognostic
    in the order given in `dx_models`.

    Results will be returned in the coordinate system of the last diagnostic model.

    Model compatibility requirements:
    - Input variables of each model in the chain
      [px_model, dx_models[0], dx_models[1], ...]
      must be available in the outputs of one of the previous models.
    - If interpolate_coords == False, the coordinates of each model in the chain
      must be mappable to the next model using `earth2studio.utils.coords.map_coords`.
    - If interpolate_coords == True, the coordinates of each model in the chain must
      be possible to interpolate to the next model using
      `earth2studio.utils.interp.LatLonInterpolation`.

    Parameters
    ----------
    px_model : PrognosticModel
        The prognostic model to use as the base model.
    dx_model : DiagnosticModel
        The diagnostic models whose outputs are concatenated to the output of px_model.
    interpolate_coords : bool, default False
        Whether to use bilinear interpolation to map spatial coordinates. If False,
        nearest neighbor interpolation will be used. Must be set to True if any models
        have 2D latitude/longitude coordinates.
    keep_px_output : bool, default True
        Whether to include output of px_model in the input.
    """

    def __init__(
        self,
        px_model: PrognosticModel,
        dx_model: DiagnosticModel,
        prepare_dx_input_coords: PrepareDxInputCoords | None = None,
        prepare_dx_input_tensor: PrepareDxInputTensor | None = None,
        prepare_output_coords: PrepareOutputCoords | None = None,
        prepare_output_tensor: PrepareOutputTensor | None = None,
    ):
        super().__init__()

        self.px_model = px_model
        self.dx_model = dx_model

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
        # This is kinda annnoying at the moment, but I'm not sure of a better way yet
        dx_coords = self.prepare_dx_input_coords(
            px_coords, self.dx_model.input_coords()
        )
        dx_coords = self.dx_model.output_coords(dx_coords)
        out_coords = self.prepare_output_coords(px_coords, dx_coords)
        return out_coords

    @torch.inference_mode()
    def _default_prepare_dx_input_coords(
        self, px_coords: CoordSystem, dx_coords: CoordSystem
    ) -> CoordSystem:
        """Default coordinate preparation for diagnostic model.
        Just naively replaces variable, lat, lon coords with thos required by the
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
        self, px_coords: CoordSystem, dx_coords: CoordSystem
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
            for key in dx_coords.keys():
                if key == "variable":
                    continue
                handshake_coords(px_coords, dx_coords, key)
            coords = dx_coords
            coords["variable"] = np.concatenate(
                [px_coords["variable"], dx_coords["variable"]]
            )
        except (KeyError, ValueError):
            coords = dx_coords
        return coords

    @torch.inference_mode()
    def _default_prepare_output_tensor(
        self,
        px_x: torch.Tensor,
        px_coords: CoordSystem,
        dx_x: torch.Tensor,
        dx_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """This default method will attempt concat the prognostic and diagnostic
        outputs if the coordinate systems will allow it

        Parameters
        ----------
        px_x : torch.Tensor
            Output of prognostic model from a single step
        px_coords : CoordSystem
            Output coordinates from the prognostic model
        dx_x:
            Output of diagnostic model
        dx_coords : CoordSystem
            Diagnostic model input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Outputs to be returned by the wrapper
        """
        try:
            for key in dx_coords.keys():
                if not key == "variable":
                    handshake_coords(px_coords, dx_coords, key)

            x = torch.concat([px_x, dx_x], dim=list(dx_coords).index("variable"))
            coords = dx_coords
            coords["variable"] = np.concatenate(
                [px_coords["variable"], dx_coords["variable"]]
            )
        except (KeyError, ValueError):
            x = dx_x
            coords = dx_coords

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
        dx_x, dx_coords = self.prepare_dx_input_tensor(
            px_x, px_coords, self.dx_model.input_coords()
        )
        dx_x, dx_coords = self.dx_model(dx_x, dx_coords)
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
            dx_x, dx_coords = self.prepare_dx_input_tensor(
                px_x, px_coords, self.dx_model.input_coords()
            )
            dx_x, dx_coords = self.dx_model(dx_x, dx_coords)
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
