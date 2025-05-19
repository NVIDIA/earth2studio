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

import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    check_extra_imports,
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

try:
    from physicsnemo.utils.zenith_angle import cos_zenith_angle

    from earth2studio.models.nn.afno_ssrd import SolarRadiationNet
except ImportError:
    SolarRadiationNet = None
    cos_zenith_angle = None

VARIABLES = [
    "t2m",
    "sp",
    "tcwv",
    "z50",
    "z300",
    "z500",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t300",
    "t500",
    "t700",
    "t850",
    "t925",
    "t1000",
    "q50",
    "q300",
    "q500",
    "q700",
    "q850",
    "q925",
    "q1000",
]


@check_extra_imports("solarradiation-afno", [SolarRadiationNet, cos_zenith_angle])
class SolarRadiationAFNO(torch.nn.Module, AutoModelMixin):
    """Solar Radiation AFNO diagnostic model. Predicts the accumulated global surface solar
    radiation over 6 hours [Jm^-2]. The model uses 31 variables as input and outputs
    one on a 0.25 degree lat-lon grid (south-pole excluding) [720 x 1440].

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model
    freq : str
        Frequency of the model (e.g. "6h")
    era5_mean : torch.Tensor
        Model mean normalization tensor for ERA5 variables
    era5_std : torch.Tensor
        Model standard deviation normalization tensor for ERA5 variables
    ssrd_mean : torch.Tensor
        Model mean normalization tensor for SSRD output
    ssrd_std : torch.Tensor
        Model standard deviation normalization tensor for SSRD output
    orography : torch.Tensor
        Surface geopotential (orography)
    landsea_mask : torch.Tensor
        Land sea mask
    sincos_latlon : torch.Tensor
        4 fields embedding location information (cos(lat), sin(lat), cos(lon), sin(lon))
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        freq: str,
        era5_mean: torch.Tensor,
        era5_std: torch.Tensor,
        ssrd_mean: torch.Tensor,
        ssrd_std: torch.Tensor,
        orography: torch.Tensor,
        landsea_mask: torch.Tensor,
        sincos_latlon: torch.Tensor,
    ):
        super().__init__()
        self.core_model = core_model
        self.freq = freq
        self.register_buffer("era5_mean", era5_mean)
        self.register_buffer("era5_std", era5_std)
        self.register_buffer("ssrd_mean", ssrd_mean)
        self.register_buffer("ssrd_std", ssrd_std)
        self.register_buffer("orography", orography)
        self.register_buffer("landsea_mask", landsea_mask)
        self.register_buffer("sincos_latlon", sincos_latlon)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.empty(0),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model

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
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 5)
        handshake_dim(input_coords, "lat", 4)
        handshake_dim(input_coords, "variable", 3)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")
        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(["ssrd"])
        return output_coords

    def __str__(self) -> str:
        return "solarnet"

    @classmethod
    def load_default_package(cls, cache=False) -> Package:
        """Default pre-trained solar radiation model package from Nvidia model registry"""
        return Package(
            "ngc://models/nvidian/onboarding/afno_dx_solarradiation@0.0.0",
            cache=cache,
            cache_options={
                "cache_storage": Package.default_cache("ssrd_afno"),
                "same_names": True,
            },
        )

    @classmethod
    @check_extra_imports("solarradiation-afno", [SolarRadiationNet, cos_zenith_angle])
    def load_model(cls, package: Package, freq: str = "6h") -> DiagnosticModel:
        """Load diagnostic from package"""
        if SolarRadiationNet is None or cos_zenith_angle is None:
            raise ImportError(
                "Additional SolarRadiationAFNO model dependencies are not installed. See install documentation for details."
            )

        model = SolarRadiationNet.from_checkpoint(
            str(
                Path(
                    package.resolve(
                        f"ssrd_{freq}_afno/solarradiation_afno/ssrd_{freq}_afno.mdlus"
                    )
                )
            )
        )
        model.eval()
        era5_mean = torch.Tensor(
            np.load(
                str(
                    Path(
                        package.resolve(
                            f"ssrd_{freq}_afno/solarradiation_afno/global_means.npy"
                        )
                    )
                )
            )
        )
        era5_std = torch.Tensor(
            np.load(
                str(
                    Path(
                        package.resolve(
                            f"ssrd_{freq}_afno/solarradiation_afno/global_stds.npy"
                        )
                    )
                )
            )
        )
        ssrd_mean = torch.Tensor(
            np.load(
                str(
                    Path(
                        package.resolve(
                            f"ssrd_{freq}_afno/solarradiation_afno/ssrd_means.npy"
                        )
                    )
                )
            )
        )
        ssrd_std = torch.Tensor(
            np.load(
                str(
                    Path(
                        package.resolve(
                            f"ssrd_{freq}_afno/solarradiation_afno/ssrd_stds.npy"
                        )
                    )
                )
            )
        )
        z = torch.Tensor(
            np.load(
                str(
                    Path(
                        package.resolve(
                            f"ssrd_{freq}_afno/solarradiation_afno/orography.npy"
                        )
                    )
                )
            )
        ).permute(1, 0, 2, 3)
        z = (z - z.mean()) / z.std()

        lsm = torch.Tensor(
            np.load(
                str(
                    Path(
                        package.resolve(
                            f"ssrd_{freq}_afno/solarradiation_afno/land_sea_mask.npy"
                        )
                    )
                )
            )
        ).permute(1, 0, 2, 3)

        sincos_latlon = torch.Tensor(
            np.load(
                str(
                    Path(
                        package.resolve(
                            f"ssrd_{freq}_afno/solarradiation_afno/sincos_latlon.npy"
                        )
                    )
                )
            )
        ).permute(1, 0, 2, 3)
        return cls(
            model, freq, era5_mean, era5_std, ssrd_mean, ssrd_std, z, lsm, sincos_latlon
        )

    def get_sza_lonlat(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        grid = np.meshgrid(lon, lat)
        return (grid[0].reshape(-1), grid[1].reshape(-1))

    def compute_sza(self, output_coords):
        lon, lat = self.get_sza_lonlat(output_coords["lon"], output_coords["lat"])
        t = output_coords["time"] + output_coords["lead_time"]
        t = datetime.fromtimestamp(
            t.astype("datetime64[s]").astype("int")[0], tz=timezone.utc
        )
        return torch.Tensor(cos_zenith_angle(t, lon, lat)).to(self.device)

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        x = x.view(x.shape[2:])
        x = (x - self.era5_mean) / self.era5_std
        output_coords = self.output_coords(coords)

        # compute solar zenith angle and concatenate
        sza = self.compute_sza(output_coords).reshape((1, 1, *x.shape[2:]))
        x = torch.cat(
            (x, sza, self.sincos_latlon, self.orography, self.landsea_mask), dim=1
        )

        out = self.core_model(x) * self.ssrd_std + self.ssrd_mean

        # filter out negative values
        out[out < 0] = 0
        return out[None, None], output_coords
