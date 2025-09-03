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

from collections.abc import Iterator, Sequence

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import handshake_coords, map_coords
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.type import CoordSystem


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
    dx_models : DiagnosticModel | Sequence[DiagnosticModel]
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
        dx_models: DiagnosticModel | Sequence[DiagnosticModel],
        interpolate_coords: bool = False,
        keep_px_output: bool = True,
    ):
        super().__init__()

        # initialize models
        if not isinstance(dx_models, Sequence):
            dx_models = [dx_models]
        self.dx_model = dx_models[-1]
        if len(dx_models) < 0:
            raise ValueError("At least 1 diagnostic model must be provided.")
        elif len(dx_models) == 1:
            self.px_model = px_model
        else:
            # support multiple diagnostic models through recursion,
            # keeping the rest of the implementation cleaner
            self.px_model = DiagnosticWrapper(
                px_model,
                dx_models[:-1],
                interpolate_coords=interpolate_coords,
                keep_px_output=keep_px_output,
            )

        self.interpolate_coords = interpolate_coords
        if interpolate_coords:
            self.interp: LatLonInterpolation | None = None
            self.interp_lat0: np.ndarray | None = None
            self.interp_lon0: np.ndarray | None = None
            self.interp_lat1: np.ndarray | None = None
            self.interp_lon1: np.ndarray | None = None
        self.keep_px_output = keep_px_output or (len(dx_models) > 1)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self.px_model.input_coords()

    def _check_dx_grid(
        self,
        lat_px_out: np.ndarray,
        lon_px_out: np.ndarray,
        lat_dx_in: np.ndarray,
        lon_dx_in: np.ndarray,
    ) -> None:
        """Check that the diagnostic input lat/lon grids are a subset of the prognostic
        output lat/lon grids.
        """
        lat_px_out = torch.as_tensor(lat_px_out)
        lon_px_out = torch.as_tensor(lon_px_out)
        lat_dx_in = torch.as_tensor(lat_dx_in)
        lon_dx_in = torch.as_tensor(lon_dx_in)
        grids_ok = (
            (lat_dx_in >= lat_px_out.min()).all()
            and (lat_dx_in <= lat_px_out.max()).all()
            and (lon_dx_in >= lon_px_out.min()).all()
            and (lon_dx_in <= lon_px_out.max()).all()
        )
        if not grids_ok:
            raise ValueError(
                "Output lat/lon grids must be a subset of the input lat/lon grids."
            )

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

        out_coords_px = self.px_model.output_coords(input_coords)
        in_coords_dx = self.dx_model.input_coords()
        if len(in_coords_dx["lat"]) == 0:
            # diagnostic model has no specific input grid -> use grid from px model
            in_coords_dx["lat"] = out_coords_px["lat"]
            in_coords_dx["lon"] = out_coords_px["lon"]
        out_coords_dx = self.dx_model.output_coords(in_coords_dx)
        out_coords_dx["lead_time"] = out_coords_px["lead_time"]

        # check that diagnostic input grid is contained in the prognostic output grid
        self._check_dx_grid(
            out_coords_px["lat"],
            out_coords_px["lon"],
            in_coords_dx["lat"],
            in_coords_dx["lon"],
        )

        if self.keep_px_output:
            variables = np.concatenate(
                [out_coords_px["variable"], out_coords_dx["variable"]]
            )
            out_coords_dx["variable"] = variables

        return out_coords_dx

    @torch.inference_mode()
    def _interpolate(
        self, x: torch.Tensor, in_coords: CoordSystem, out_coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Interpolate spatial coordinates."""

        def _convert_to_2d(
            lat: np.ndarray, lon: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            if (lat.ndim == 1) and (lon.ndim == 1):
                return np.meshgrid(lat, lon, indexing="ij")
            else:
                return (lat, lon)

        def _coords_match(coord0: np.ndarray, coord1: np.ndarray) -> bool:
            return (coord0.shape == coord1.shape) and (coord0 == coord1).all()

        (lat0, lon0) = _convert_to_2d(in_coords["lat"], in_coords["lon"])
        (lat1, lon1) = _convert_to_2d(out_coords["lat"], out_coords["lon"])

        reuse_interp = (
            (self.interp is not None)
            and _coords_match(lat0, self.interp_lat0)
            and _coords_match(lon0, self.interp_lon0)
            and _coords_match(lat1, self.interp_lat1)
            and _coords_match(lon1, self.interp_lon1)
        )
        if not reuse_interp:
            self.interp = LatLonInterpolation(lat0, lon0, lat1, lon1).to(
                device=x.device
            )
            self.interp_lat0 = lat0
            self.interp_lon0 = lon0
            self.interp_lat1 = lat1
            self.interp_lon1 = lon1

        if self.interp is not None:  # needed to keep mypy happy
            x = self.interp(x)

        coords = in_coords.copy()
        coords["lat"] = out_coords["lat"]
        coords["lon"] = out_coords["lon"]

        return (x, coords)

    @torch.inference_mode()
    def _concat_diagnostic(
        self, x_px: torch.Tensor, coords_px: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run diagnostic model and concatenate output to prognostic model."""

        in_coords_dx = self.dx_model.input_coords()

        # convert prognostic model output to diagnostic model grid without dropping
        # variables (necessary for concatenating to the diagnostic model output)
        coords_dx_spatial = in_coords_dx.copy()
        coords_dx_spatial["variable"] = coords_px["variable"]
        map_coords_func = self._interpolate if self.interpolate_coords else map_coords
        (x_px, coords_px) = map_coords_func(x_px, coords_px, coords_dx_spatial)

        # select diagnostic input variables and inference diagnostic model
        (x, coords) = self.dx_model(*map_coords(x_px, coords_px, in_coords_dx))

        # concatenate prognostic and diagnostic outputs and coordinates
        if self.keep_px_output:
            try:
                for dim, dim_name in enumerate(coords):
                    if dim_name != "variable" and (len(coords[dim_name]) > 0):
                        handshake_coords(coords_px, coords, dim_name)
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"If keep_px_output==True, the outputs of the diagnostic model must be concatenatable to the inputs. Original error message: '{str(e)}'"
                )

            variable_dim = list(coords).index("variable")
            x = torch.concat([x_px, x], dim=variable_dim)
            variables = np.concatenate([coords_px["variable"], coords["variable"]])
            coords["variable"] = variables

        return (x, coords)

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
        return self._concat_diagnostic(*self.px_model(x, coords))

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
        for x, coords in self.px_model.create_iterator(x, coords):
            yield self._concat_diagnostic(x, coords)
