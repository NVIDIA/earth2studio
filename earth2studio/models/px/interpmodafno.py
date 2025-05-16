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
from datetime import datetime

import numpy as np
import torch
import xarray as xr

try:
    from physicsnemo import Module as PhysicsNemoModule
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    PhysicsNemoModule = None
    cos_zenith_angle = None

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import check_extra_imports
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


@check_extra_imports("interp-modafno", [PhysicsNemoModule, cos_zenith_angle])
class InterpModAFNO(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """ModAFNO interpolation for global prognostic models. Interpolates a forecast model
    to a shorter time-step size (by default from 6 to 1 hour). Operates on 0.25 degree
    lat-lon equirectangular grid with 73 variables.

    Note
    ----
    For more information on the model, please refer to:

    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/afno_dx_fi-v1-era5
    - https://arxiv.org/abs/2410.18904

    Warning
    -------
    The model requires a base forecast model to be set before execution. This can be
    done by setting the `px_model` attribute or using the `load_model` method.

    Parameters
    ----------
    interp_model : torch.nn.Module
        The interpolation model that performs the time interpolation
    center : torch.Tensor
        Model center normalization tensor
    scale : torch.Tensor
        Model scale normalization tensors
    geop : torch.Tensor
        Geopotential height data used as a static feature
    lsm : torch.Tensor
        Land-sea mask data used as a static feature
    px_model : PrognosticModel, optional
        The base forecast model that produces the coarse time resolution forecasts. If
        not provide, should be set by the user before executing the model, by default
        None.
    num_interp_steps : int, optional
        Number of interpolation steps to perform between forecast steps, by default 6
    """

    def __init__(
        self,
        interp_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
        geop: torch.Tensor,
        lsm: torch.Tensor,
        px_model: PrognosticModel | None = None,
        num_interp_steps: int = 6,
    ) -> None:
        super().__init__()
        self.px_model = px_model
        self.interp_model = interp_model
        self.num_interp_steps = num_interp_steps
        self.variables = np.array(VARIABLES)
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.register_buffer("geop", geop)
        self.register_buffer("lsm", lsm)

    @staticmethod
    def _load_feature_from_file(fn: str, var: str) -> torch.Tensor:
        """Load a feature from a NetCDF file.

        Parameters
        ----------
        fn : str
            Path to the NetCDF file
        var : str
            Variable name to load from the file

        Returns
        -------
        torch.Tensor
            Loaded feature as a tensor with shape (1, 1, H, W)
        """
        with xr.open_dataset(fn) as ds:
            x = np.array(ds[var])
        return torch.Tensor(x).unsqueeze(0).unsqueeze(0)

    def _compute_latlon(self) -> None:
        # compute sin/cos of lat/lon
        coords = self.output_coords(self.input_coords())
        lat = np.deg2rad(coords["lat"])
        lon = np.deg2rad(coords["lon"])
        (lat, lon) = np.meshgrid(lat, lon, indexing="ij")
        sincos_latlon = torch.Tensor(
            np.stack([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)], axis=0)
        ).unsqueeze(0)
        self.register_buffer("lat", torch.as_tensor(lat, device=self.center.device))
        self.register_buffer("lon", torch.as_tensor(lon, device=self.center.device))
        self.register_buffer(
            "sincos_latlon", torch.as_tensor(sincos_latlon, device=self.center.device)
        )

    def __str__(self) -> str:
        return "InterpModAFNO"

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model
        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        # Getter / Setters don't work with torch.nn.Module, need to check manually here
        if self.px_model is None:
            raise ValueError("Base forecast model, px_model, must be set")
        return self.px_model.input_coords()

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
        # Sometime prognostics don't explicitly have a time coord and thats okay
        # as long as the data does
        output_coords["batch"] = input_coords.get("batch", np.empty(0))
        output_coords["time"] = input_coords.get("time", np.empty(0))
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "ngc://models/nvidia/earth-2/afno_dx_fi-v1-era5@v0.1.0",
            cache_options={
                "cache_storage": Package.default_cache("modafno_interpolation"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_extra_imports("interp-modafno", [PhysicsNemoModule, cos_zenith_angle])
    def load_model(
        cls, package: Package, px_model: PrognosticModel | None = None
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        px_model : PrognosticModel | None, optional
            The base forecast model that produces the coarse time resolution forecasts.
            If None, should be set manually, by default None

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        model = PhysicsNemoModule.from_checkpoint(
            package.resolve("fcinterp-modafno-2x2.mdlus")
        )
        model.eval()

        # Load center and std normalizations
        local_center = torch.Tensor(np.load(package.resolve("global_means.npy")))[
            :, :73
        ]
        local_std = torch.Tensor(np.load(package.resolve("global_stds.npy")))[:, :73]

        # load static variables
        geop = cls._load_feature_from_file(package.resolve("orography.nc"), "Z")[
            0, :, :, :720, :
        ]
        geop = (geop - geop.mean()) / geop.std()
        lsm = cls._load_feature_from_file(package.resolve("land_sea_mask.nc"), "LSM")[
            0, :, :, :720, :
        ]

        return cls(
            model,
            center=local_center,
            scale=local_std,
            geop=geop,
            lsm=lsm,
            px_model=px_model,
        )

    def _cos_zenith(self, times: list) -> torch.Tensor:
        """Calculate cosine of zenith angle for given times.

        Parameters
        ----------
        times : list
            List of times to calculate cosine of zenith angle for

        Returns
        -------
        torch.Tensor
            Cosine of zenith angle for each time
        """
        # Convert generator to list to fix type incompatibility
        times_list = list(datetime.fromisoformat(str(t)[:19]) for t in times)
        cos_zen = [
            cos_zenith_angle(t, self.lon.cpu().numpy(), self.lat.cpu().numpy())
            for t in times_list
        ]
        return torch.Tensor(np.stack(cos_zen, axis=0)).unsqueeze(0)

    @torch.inference_mode()
    def _interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Interpolate between two forecast steps.

        Parameters
        ----------
        x0 : torch.Tensor
            First forecast step
        x1 : torch.Tensor
            Second forecast step
        coords : CoordSystem
            Coordinate system for the forecast steps

        Yields
        ------
        Generator[tuple[torch.Tensor, CoordSystem], None, None]
            Generator yielding interpolated forecast steps and their coordinate systems
        """
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
        """Runs prognostic model 1 step

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 1 hour in the future
        """
        gen = self._default_generator(x, coords)
        return next(gen)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        if self.px_model is None:
            raise ValueError(
                "Base forecast model, px_model, must be set before executing the model."
            )

        if not hasattr(self, "sincos_latlon"):
            self._compute_latlon()

        for fc_step, (x, coords) in enumerate(self.px_model.create_iterator(x, coords)):
            # Make sure prognostic model has all 73 required variables
            (x, coords) = map_coords(x, coords, self.output_coords(coords))
            if fc_step == 0:
                x0 = x
                coords0 = coords
                yield (x, coords)
            else:
                x1 = x
                for x, coords in self._interpolate(x0, x1, coords0):
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
