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
    """Base class for Solar Radiation AFNO diagnostic models. Predicts the accumulated global surface solar
    radiation [Jm^-2]. The model uses 31 variables as input and outputs one on a 0.25 degree lat-lon grid
    (south-pole excluding) [720 x 1440].

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
    ) -> None:
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
                "lat": np.linspace(90, -90, 721),
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
        handshake_dim(input_coords, "lead_time", 2)
        handshake_dim(input_coords, "time", 1)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")
        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(["ssrd"])
        return output_coords

    def __str__(self) -> str:
        return "SolarRadiationNet"

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        raise NotImplementedError("Use specific frequency subclass")

    @classmethod
    @check_extra_imports("solarradiation-afno", [SolarRadiationNet, cos_zenith_angle])
    def load_model(cls, package: Package) -> DiagnosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from

        Returns
        -------
        PrognosticModel
            Prognostic model
        """

        model = SolarRadiationNet(
            dim=2,
            in_channels=31,
            out_channels=1,
            patch_size=[4, 4],
            pad_size=[[3, 4], [2, 2]],
            embed_dim=256,
            depth=12,
            mlp_ratio=4.0,
            drop_rate=0.0,
            num_blocks=8,
            sparsity_threshold=0.01,
        )
        model.load(package.resolve(f"ssrd_{cls.freq}_afno/ssrd_{cls.freq}_afno.mdlus"))
        model.eval()
        era5_mean = torch.Tensor(
            np.load(
                str(Path(package.resolve(f"ssrd_{cls.freq}_afno/global_means.npy")))
            )
        )
        era5_std = torch.Tensor(
            np.load(str(Path(package.resolve(f"ssrd_{cls.freq}_afno/global_stds.npy"))))
        )
        ssrd_mean = torch.Tensor(
            np.load(str(Path(package.resolve(f"ssrd_{cls.freq}_afno/ssrd_means.npy"))))
        )
        ssrd_std = torch.Tensor(
            np.load(str(Path(package.resolve(f"ssrd_{cls.freq}_afno/ssrd_stds.npy"))))
        )
        z = torch.Tensor(
            np.load(str(Path(package.resolve(f"ssrd_{cls.freq}_afno/orography.npy"))))
        ).permute(1, 0, 2, 3)
        z = (z - z.mean()) / z.std()

        lsm = torch.Tensor(
            np.load(
                str(Path(package.resolve(f"ssrd_{cls.freq}_afno/land_sea_mask.npy")))
            )
        ).permute(1, 0, 2, 3)

        sincos_latlon = torch.Tensor(
            np.load(
                str(Path(package.resolve(f"ssrd_{cls.freq}_afno/sincos_latlon.npy")))
            )
        ).permute(1, 0, 2, 3)
        return cls(
            model,
            cls.freq,
            era5_mean,
            era5_std,
            ssrd_mean,
            ssrd_std,
            z,
            lsm,
            sincos_latlon,
        )

    def get_sza_lonlat(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get longitude and latitude arrays for solar zenith angle calculation.

        Parameters
        ----------
        lon : np.ndarray
            Longitude array
        lat : np.ndarray
            Latitude array

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (longitude, latitude) arrays
        """
        grid = np.meshgrid(lon, lat)
        return (grid[0].reshape(-1), grid[1].reshape(-1))

    def compute_sza(self, output_coords: CoordSystem) -> torch.Tensor:
        """Compute solar zenith angle for given coordinates.

        Parameters
        ----------
        output_coords : CoordSystem
            Output coordinate system

        Returns
        -------
        torch.Tensor
            Solar zenith angle tensor
        """
        lon, lat = self.get_sza_lonlat(output_coords["lon"], output_coords["lat"])
        t = output_coords["time"] + output_coords["lead_time"]
        t = datetime.fromtimestamp(
            t.astype("datetime64[s]").astype("int")[0], tz=timezone.utc
        )
        return torch.Tensor(cos_zenith_angle(t, lon, lat))

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        # Normalize input
        x = (x - self.era5_mean) / self.era5_std
        output_coords = self.output_coords(coords)

        # Initialize output tensor
        out = torch.zeros_like(x[..., :1, :, :])

        # Get lon/lat grid for SZA computation
        grid_x, grid_y = torch.meshgrid(
            torch.tensor(coords["lat"]), torch.tensor(coords["lon"])
        )

        for j, _ in enumerate(coords["batch"]):
            for k, t in enumerate(coords["time"]):
                for lt, dt in enumerate(coords["lead_time"]):
                    # Compute SZA for this specific time and lead time
                    sza = (
                        self.compute_sza(
                            OrderedDict(
                                {
                                    "time": np.array([t]),
                                    "lead_time": np.array([dt]),
                                    "lat": coords["lat"],
                                    "lon": coords["lon"],
                                }
                            )
                        )
                        .reshape((1, 1, *x.shape[4:]))
                        .to(x.device)
                    )

                    # Get the current time/lead time slice
                    current_x = x[j, k, lt : lt + 1]

                    # Concatenate all inputs
                    in_ = torch.cat(
                        (
                            current_x,
                            sza,
                            self.sincos_latlon,
                            self.orography,
                            self.landsea_mask,
                        ),
                        dim=1,
                    )

                    # Run model and denormalize
                    out[j, k, lt : lt + 1] = (
                        self.core_model(in_) * self.ssrd_std + self.ssrd_mean
                    )

        # filter out negative values
        out = torch.clamp(out, min=0)

        return out, output_coords


class SolarRadiationAFNO1H(SolarRadiationAFNO):
    """Solar Radiation AFNO diagnostic model for 1-hour frequency predictions.
    Predicts the accumulated global surface solar radiation over 1 hour [Jm^-2].
    """

    freq = "1h"

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "ngc://models/nvidia/earth-2/afno_dx_sr-v1-era5@1.0.0",
            cache_options={
                "cache_storage": Package.default_cache("ssrd_afno_1h"),
                "same_names": True,
            },
        )
        return package


class SolarRadiationAFNO6H(SolarRadiationAFNO):
    """Solar Radiation AFNO diagnostic model for 6-hour frequency predictions.
    Predicts the accumulated global surface solar radiation over 6 hours [Jm^-2].
    """

    freq = "6h"

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "ngc://models/nvidia/earth-2/afno_dx_sr-v1-era5@1.0.0",
            cache_options={
                "cache_storage": Package.default_cache("ssrd_afno_6h"),
                "same_names": True,
            },
        )
        return package
