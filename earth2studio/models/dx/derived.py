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

import numpy as np
import torch

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem


class DerivedWS(torch.nn.Module):
    """Calculates the Wind Speed (WS) magnitude from eastward and northward wind
    components for specified levels. The calculation is based on the formula:
    ws = sqrt(u^2 + v^2)

    Parameters
    ----------
    levels : list[int  |  str], optional
        Pressure / height levels to compute WS for. The resulting expected input fields
        are u and v wind components pairs for each level. E.g. for level 100 the input
        fields should be [u100, v100], by default [100]
    """

    def __init__(self, levels: list[int | str] = [100]) -> None:
        super().__init__()
        self.levels = levels
        self.in_variables = []
        for input_level in [[f"u{level}", f"v{level}"] for level in levels]:
            self.in_variables.extend(input_level)
        self.out_variables = [f"ws{level}" for level in levels]

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
                "variable": np.array(self.in_variables),
                "lat": np.empty(0),
                "lon": np.empty(0),
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
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(self.out_variables)
        return output_coords

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)
        # This function expects [u, v] pairs
        u = x[..., ::2, :, :]
        v = x[..., 1::2, :, :]
        out_tensor = torch.sqrt(u**2 + v**2)
        return out_tensor, output_coords


class DerivedRH(torch.nn.Module):
    """Calculates the relative humidity (RH) from specific humidity and temperature
    for specified pressure levels. Based on the calculations ECMWF uses in the IFS
    numerical simulator which accounts for estimating the water vapor and ice present
    in the atmosphere.

    Note
    ----
    See reference, equation 7.98 onwards:

    - https://www.ecmwf.int/en/elibrary/81370-ifs-documentation-cy48r1-part-iv-physical-processes

    Parameters
    ----------
    levels : list[int  |  str], optional
        hPa Pressure levels to compute RH. The resulting expected input fields
        are specific humidity and temperature pairs for each pressure level. E.g. for
        level 100 hPa the input fields should be [t100, q100], by default [100]
    """

    def __init__(self, levels: list[int | str] = [100]) -> None:
        super().__init__()
        self.levels = levels
        self.in_variables = []
        for input_level in [[f"t{level}", f"q{level}"] for level in levels]:
            self.in_variables.extend(input_level)
        self.out_variables = [f"r{level}" for level in levels]
        # Set up pressure levels tensor
        pressure_levels = [100 * float(level) for level in levels]
        self.pressure_levels = torch.tensor(pressure_levels)[:, None, None]

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
                "variable": np.array(self.in_variables),
                "lat": np.empty(0),
                "lon": np.empty(0),
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
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(self.out_variables)
        return output_coords

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)

        epsilon = 0.621981
        t = x[..., ::2, :, :]  # K
        q = x[..., 1::2, :, :]  # g/kg
        p = self.pressure_levels.to(x.device)

        e = (p * q * (1.0 / epsilon)) / (1 + q * (1.0 / (epsilon) - 1))

        es_w = 611.21 * torch.exp(17.502 * (t - 273.16) / (t - 32.19))
        es_i = 611.21 * torch.exp(22.587 * (t - 273.16) / (t + 0.7))

        alpha = torch.clip((t - 250.16) / (273.16 - 250.16), 0, 1.2) ** 2
        es = alpha * es_w + (1 - alpha) * es_i
        out_tensor = 100 * e / es
        out_tensor = torch.clamp(out_tensor, 0, 100)

        return out_tensor, output_coords


class DerivedRHDewpoint(torch.nn.Module):
    """Calculates the surface relative humidity (RH) from dewpoint temperature and air
    temperature. This calculation is based on the August-Roche-Magnus approximation.

    Note
    ----
    For more details see the following references:

    - https://doi.org/10.5194/gmd-9-523-2016 (Eq. B3)
    - https://doi.org/10.5194/tc-2023-8 (Eq. 1 and 5)
    - https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2 (Eq. 21)
    - https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#August%E2%80%93Roche%E2%80%93Magnus_formula
    """

    def __init__(self) -> None:
        super().__init__()

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
                "variable": np.array(["t2m", "d2m"]),
                "lat": np.empty(0),
                "lon": np.empty(0),
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
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(["r2m"])
        return output_coords

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)

        t = x[..., ::2, :, :] - 273.16  # K -> C
        d = x[..., 1::2, :, :] - 273.16  # K -> C

        # Calculate saturation vapor pressure (es) and vapor pressure (e)
        e = 6.11 * torch.exp((17.62 * d) / (d + 243.12))
        es = 6.11 * torch.exp((17.62 * t) / (t + 243.12))

        # Improved fit for cold temperatures (Sonntag 1990)
        e_cold = 6.11 * torch.exp((22.46 * d) / (d + 272.62))
        es_cold = 6.11 * torch.exp((22.46 * t) / (t + 272.62))

        out_tensor = torch.where(t < 0, e_cold / es_cold, e / es) * 100
        # Clamp to 0-100%
        out_tensor = torch.clamp(out_tensor, 0, 100)
        return out_tensor, output_coords


class DerivedVPD(torch.nn.Module):
    """Calculates the Vapor Pressure Deficit (VPD) in hPa from relative humidity
    and temperature fields. The calculation is based on the formula:

    VPD = es * ((100 - rh) / 100)

    The variable es is the saturation vapor pressure calculated using the formula:

    es = 6.11 * exp((L / Rv) * ((1 / 273) - (1 / T)))

    where L is the latent heat of vaporization (2.26e6 J/kg), Rv is the gas constant for
    water vapor (461 J/kg/K).

    Note
    ----
    For additional information on the VPD calculation in millibars, please refer to:

    - Dennis Hartman "Global Physical Climatology" (p 350)
    - https://www.sciencedirect.com/book/9780123285317/global-physical-climatology

    Parameters
    ----------
    levels : list[int  |  str], optional
        Pressure / height levels to compute VPD for. The resulting expected input fields
        are temperature and relative humidity pairs for each level. E.g. for level 100
        the input fields should be [t100, r100], by default [100]
    """

    def __init__(self, levels: list[int | str] = [100]) -> None:
        super().__init__()
        self.levels = levels
        self.in_variables = []
        for input_level in [[f"t{level}", f"r{level}"] for level in levels]:
            self.in_variables.extend(input_level)
        self.out_variables = [f"vpd{level}" for level in levels]

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
                "variable": np.array(self.in_variables),
                "lat": np.empty(0),
                "lon": np.empty(0),
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
        handshake_dim(input_coords, "variable", 1)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "lon", 3)
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(self.out_variables)
        return output_coords

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)
        # This function expects [temp, rh] pairs
        t = x[..., ::2, :, :]
        r = x[..., 1::2, :, :]
        L = 2.26e6
        Rv = 461
        es = 6.11 * torch.exp((L / Rv) * ((1.0 / 273.16) - (1.0 / t)))
        out_tensor = es * ((100.0 - r) / 100.0)

        return out_tensor, output_coords
