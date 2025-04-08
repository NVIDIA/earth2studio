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
from collections import OrderedDict
from collections.abc import Generator, Iterator
from datetime import datetime

import numpy as np
import torch
import xarray as xr
from physicsnemo import Module
from physicsnemo.utils.zenith_angle import cos_zenith_angle

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import CoordSystem, map_coords

VARIABLES = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
]


class ForecastInterpolation(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """ModAFNO interpolation for global prognostic models.
    Interpolates a forecast model to a shorter time-step size (by default from 6 h to 1 h)
    Operates on 0.25 degree lat-lon equirectangular grid with 73 variables.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core PyTorch model with loaded weights
    center : torch.Tensor
        Model center normalization tensor
    scale : torch.Tensor
        Model scale normalization tensor
    variables : np.array, optional
        Variables associated with model, by default 73 variable model.
    """

    def __init__(
        self,
        interp_model: torch.nn.Module,
        fc_model: PrognosticModel,
        center: torch.Tensor,
        scale: torch.Tensor,
        geop: torch.Tensor,
        lsm: torch.Tensor,
        num_interp_steps=6,
        variables: np.array = np.array(VARIABLES),
    ):
        super().__init__()
        self.fc_model = fc_model
        self.interp_model = interp_model
        self.num_interp_steps = num_interp_steps
        self.variables = variables
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.register_buffer("geop", geop)
        self.register_buffer("lsm", lsm)

        # compute sin/cos of lat/lon
        coords = self.output_coords(self.input_coords())
        lat = np.deg2rad(coords["lat"])
        lon = np.deg2rad(coords["lon"])
        (lat, lon) = np.meshgrid(lat, lon, indexing="ij")
        sincos_latlon = torch.Tensor(
            np.stack([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)], axis=0)
        ).unsqueeze(0)
        self.register_buffer("lat", torch.Tensor(lat))
        self.register_buffer("lon", torch.Tensor(lon))
        self.register_buffer("sincos_latlon", torch.Tensor(sincos_latlon))

    def __str__(self) -> str:
        return "modafno_interp_73ch"

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model
        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self.fc_model.input_coords()

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model
        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.
        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(1, "h")]),
                "variable": np.array(self.variables),
                "lat": np.linspace(90.0, -90.0, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        if input_coords is None:
            return output_coords
        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        pass

    @classmethod
    def load_model(
        cls, package: Package, *, fc_model: PrognosticModel, variables: list = VARIABLES
    ) -> PrognosticModel:
        """Load prognostic from package"""
        model = Module.from_checkpoint(package.resolve("fcinterp-modafno-2x2.mdlus"))
        model.eval()

        # Load center and std normalizations
        local_center = torch.Tensor(np.load(package.resolve("global_means.npy")))[
            :, : len(variables)
        ]
        local_std = torch.Tensor(np.load(package.resolve("global_stds.npy")))[
            :, : len(variables)
        ]

        # load static variables
        geop = _load_feature_from_file(package.resolve("orography.nc"), "Z")[
            0, :, :, :720, :
        ]
        geop = (geop - geop.mean()) / geop.std()
        lsm = _load_feature_from_file(package.resolve("land_sea_mask.nc"), "LSM")[
            0, :, :, :720, :
        ]

        return cls(
            model,
            fc_model,
            center=local_center,
            scale=local_std,
            geop=geop,
            lsm=lsm,
            variables=np.array(variables),
        )

    def _cos_zenith(self, times):
        times = (datetime.fromisoformat(str(t)[:19]) for t in times)
        cos_zen = [
            cos_zenith_angle(t, self.lon.cpu().numpy(), self.lat.cpu().numpy())
            for t in times
        ]
        return torch.Tensor(np.stack(cos_zen, axis=0)).unsqueeze(0)

    @torch.inference_mode()
    def _interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        x0 = (x0 - self.center) / self.scale
        x1 = (x1 - self.center) / self.scale

        out = torch.zeros_like(x0)
        batch_size = len(coords["batch"])

        # expand constants
        sincos_latlon = self.sincos_latlon.expand(batch_size, -1, -1, -1)
        geop = self.geop.expand(batch_size, -1, -1, -1)
        lsm = self.lsm.expand(batch_size, -1, -1, -1)

        t0 = coords["time"][:, None] + coords["lead_time"][None, :]
        coords_end = coords
        for _ in range(self.num_interp_steps):
            coords_end = self.output_coords(coords_end)
        t1 = coords_end["time"][:, None] + coords_end["lead_time"][None, :]

        for interp_step in range(1, self.num_interp_steps):
            coords = self.output_coords(coords)

            for ti, t in enumerate(coords["time"]):
                for lti, lt in enumerate(coords["lead_time"]):
                    t_ip = t + lt
                    cos_zen = self._cos_zenith([t0[ti, lti], t1[ti, lti], t_ip]).to(
                        device=x0.device
                    )

                    x = torch.concat(
                        [
                            x0[:, ti, lti],
                            x1[:, ti, lti],
                            cos_zen.expand(batch_size, -1, -1, -1),
                            sincos_latlon,
                            geop,
                            lsm,
                        ],
                        dim=1,
                    )

                    t_norm = torch.Tensor([interp_step / self.num_interp_steps]).to(
                        device=x0.device
                    )
                    out[:, ti, lti] = self.interp_model(x, t_norm)

            out *= self.scale
            out += self.center
            yield (out, coords)

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
            Input coordinate system
        Returns
        ------
        x : torch.Tensor
        coords : CoordSystem
        """
        gen = self._default_generator(x, coords)
        return next(gen)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        for fc_step, (x, coords) in enumerate(self.fc_model.create_iterator(x, coords)):
            (x, coords) = map_coords(x, coords, self.output_coords(coords))
            if fc_step == 0:
                x0 = x
                coords0 = coords
                yield (x, coords)
            else:
                x1 = x
                for (x, coords) in self._interpolate(x0, x1, coords0):
                    yield (x, coords)
                coords = self.output_coords(coords)
                yield (x1, coords)
                x0 = x1
                coords0 = coords

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


def _load_feature_from_file(fn, var):
    with xr.open_dataset(fn) as ds:
        x = np.array(ds[var])
    return torch.Tensor(x).unsqueeze(0).unsqueeze(0)
