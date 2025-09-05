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


class DerivedSurfacePressure(torch.nn.Module):
    """Interpolates the surface pressure in hPa from pressure level geopotential,
    surface geopotential and optionally temperature. The calculation is based on
    linear interpolation of the logarithm of pressure, as well as an optional
    second-order correction based on temperature and an empirical adjustment.

    Note
    ----
    A more detailed description of the method is available at
    https://nvidia.github.io/earth2studio/userguide/notes/surface_pressure.md.

    Parameters
    ----------
    p_levels : list[int]
        Pressure levels (hPa) that are used as input for the surface pressure
        interpolation. At least 2 levels are needed. They must be ordered from
        highest to lowest pressure.
    temperature_correction : bool, optional, default True
        Whether to apply the second-order correction to surface pressure accounting
        for temperature.
    corr_adjustment : tuple[float, float], optional, default (3.4257e-5, 1.5224)
        An empirical adjustment to the temperature correction. The correction is modified
        as correction = corr_adjustment[0] + correction * corr_adjustment[1]. The default
        value was derived empirically optimizing the prediction of surface pressure in
        ERA5 data. The adjustment can be effectively disabled by passing
        `corr_adjustment=(0.0, 1.0)`.
    """

    Rs = 287.053  # gas constant for dry air (J/kg/K)
    g = 9.8067  # Earth's gravitational constant (m/s**2)
    L = -6.5e-3  # average temperature lapse rate (K/m)

    def __init__(
        self,
        p_levels: list[int] | np.ndarray | torch.Tensor,
        surface_geopotential: torch.Tensor,
        surface_geopotential_coords: CoordSystem,
        temperature_correction: bool = True,
        corr_adjustment: tuple[float, float] = (3.4257e-5, 1.5224),
    ) -> None:
        super().__init__()
        self.temperature_correction = temperature_correction
        self.corr_adjustment = corr_adjustment

        p_levels = torch.as_tensor(p_levels, dtype=torch.float32)
        (p_levels, _) = torch.sort(p_levels, descending=True)
        self.in_variables = [f"z{int(level)}" for level in p_levels]
        if self.temperature_correction:
            self.in_variables.extend(f"t{int(level)}" for level in p_levels)
        self.in_variables = np.array(self.in_variables)
        self.out_variables = np.array(["sp"])

        self.register_buffer(
            "log_p_levels",
            torch.log(p_levels * 100),
        )

        self.register_buffer("surface_geopotential", surface_geopotential)
        self.surface_geopotential_coords = surface_geopotential_coords.copy()
        if self.surface_geopotential.ndim != 2:
            raise ValueError("The surface geopotential must be two-dimensional.")
        coords_shape = tuple(len(v) for v in self.surface_geopotential_coords.values())
        if self.surface_geopotential.shape != coords_shape:
            raise ValueError(
                "The surface geopotential coordinates must match the size of the tensor."
            )

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
                "variable": self.in_variables,
                "lat": self.surface_geopotential_coords["lat"],
                "lon": self.surface_geopotential_coords["lon"],
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
        output_coords["variable"] = self.out_variables
        return output_coords

    @torch.inference_mode()
    def _find_pressure_level_below(
        self, z_target: torch.Tensor, z_levels: torch.Tensor
    ) -> torch.Tensor:
        """Find the index of the nearest pressure level in z_levels where z is lower than
        the corresponding index in z_target.
        """
        # Create a mask where z_levels are smaller than z
        mask = z_levels < z_target.unsqueeze(0)
        # Find the largest index where the condition is True along the last dimension
        indices = torch.sum(mask, dim=0) - 1
        indices = indices.clamp(min=0, max=z_levels.shape[0] - 2)
        return indices

    @torch.inference_mode()
    def _interpolate_pressure(
        self,
        z_target: torch.Tensor,
        z_levels: torch.Tensor,
        t_levels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform pressure level interpolation.

        Parameters
        ----------
        z_target : torch.Tensor
            Tensor of geopotentials (m**2 / s**2) at the target level (e.g. surface).
            Can be of any shape of one or more dimensions.
        z_levels : torch.Tensor
            Tensor of geopotentials (m**2 / s**2) at the given pressure levels
            corresponding to the `levels` attribute of the constructor.
            Shape `(N_levels, *data_shape)` where `N_levels` is the number of pressure
            levels and `data_shape` is identical to the shape of `z_target`.
        t_levels: torch.Tensor or None, optional, default None
            Can be None if `temperature_correction == False`, otherwise a Tensor of
            temperatures (K) at the given pressure levels. Shape must be identical to
            that of `z_levels`.

        Returns
        -------
        torch.Tensor
            Tensor of estimated surface pressures (Pa) with shape identical to that of
            `z_target`.
        """

        # find index of pressure level below z_target
        plevel_indices = self._find_pressure_level_below(z_target, z_levels)

        # flatten data for analysis
        data_shape = z_target.shape
        n = np.prod(data_shape)
        all_indices = torch.arange(n, device=z_target.device)
        z_target = z_target.flatten()
        z_levels = z_levels.reshape(z_levels.shape[0], n)
        plevel_indices = plevel_indices.flatten()
        if t_levels is not None:
            t_levels = t_levels.reshape(t_levels.shape[0], n)

        # select pressure and geopotential values above and below target level
        z0 = z_levels[plevel_indices, all_indices]
        z1 = z_levels[plevel_indices + 1, all_indices]
        dz = z_target - z0
        log_p0 = self.log_p_levels[plevel_indices]
        log_p1 = self.log_p_levels[plevel_indices + 1]

        # linear interpolation of log_p
        log_p = log_p0 + dz / (z1 - z0) * (log_p1 - log_p0)

        if self.temperature_correction:
            # apply second-order correction based on temperature
            t0 = t_levels[plevel_indices, all_indices]
            t1 = t_levels[plevel_indices + 1, all_indices]
            tm = 0.5 * (t0 + t1)

            c = self.L / (2 * self.g * self.Rs) / tm**2

            log_p_corr = c * dz * (z_target - z1)

            log_p_corr = self.corr_adjustment[0] + self.corr_adjustment[1] * log_p_corr
            log_p = log_p + log_p_corr

        return torch.exp(log_p).reshape(*data_shape)

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)
        num_levels = len(self.log_p_levels)

        # unpack inputs, swap variable dimension to dim 0, flatten batch dims to dim 1
        variable_dim = list(coords).index("variable")
        z_levels = x[:, :num_levels].transpose(0, variable_dim)
        shape = z_levels.shape
        flat_shape = (shape[0], np.prod(shape[1:-2]), *shape[-2:])
        z_levels = z_levels.reshape(*flat_shape)
        if self.temperature_correction:
            t_levels = x[:, num_levels:].transpose(0, variable_dim).reshape(flat_shape)
        z_target = self.surface_geopotential.expand(flat_shape[1], *flat_shape[2:])
        output_shape = list(x.shape)
        output_shape[variable_dim] = 1

        sp_pred = self._interpolate_pressure(
            z_target,
            z_levels,
            t_levels=t_levels if self.temperature_correction else None,
        ).reshape(output_shape)

        return sp_pred, output_coords
