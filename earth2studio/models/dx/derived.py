# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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

from collections.abc import Sequence

import numpy as np
import torch
import xarray as xr

from earth2studio.grids import GridDefinition, LatLonGrid, resolve_grid
from earth2studio.models.batch import batch_func
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.interp import hybrid_pressure, interpolate_vertical
from earth2studio.utils.type import CoordinateSystem, CoordSystem


class _DerivedDiagnostic(torch.nn.Module):
    stochastic = False

    def __init__(self, grid: str | GridDefinition) -> None:
        super().__init__()
        definition = resolve_grid(grid) if isinstance(grid, str) else grid
        if len(definition.dims) != 2:
            raise ValueError("Derived diagnostics require a two-dimensional grid")
        self.grid = grid
        self._spatial_dims = definition.dims

    def default_sources(self) -> None:
        """Return no recommended input data source.

        Returns
        -------
        None
            Input data sources are chosen by the caller.
        """
        return None

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the diagnostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature on the configured
            spatial grid.
        """
        return coord_array(
            ("batch", "variable", *self._spatial_dims),
            {"variable": np.asarray(self.in_variables)},
            dynamic=("batch",),
            grid=self.grid,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the diagnostic model.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform,
            on the configured grid.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature for the derived
            variables, preserving leading dimensions and unaffected metadata
            and coordinates.
        """
        handshake_dataarray(input_coords, self.input_coords())
        output = coord_array_like(
            input_coords, {"variable": np.asarray(self.out_variables)}
        )
        output.encoding = input_coords.encoding.copy()
        return output

    @torch.inference_mode()
    @batch_func()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Compute derived fields on the input device.

        Parameters
        ----------
        x : xr.DataArray
            NumPy- or CuPy-backed input with the declared trailing dimensions.

        Returns
        -------
        xr.DataArray
            Derived fields with the input name, encoding and unaffected metadata.
        """
        output_coords = self.output_coords(x)
        tensor, _ = x.e2s.to_torch()
        output = from_torch(self._compute(tensor), output_coords)
        output.encoding = x.encoding.copy()
        return output

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DerivedWS(_DerivedDiagnostic):
    """Calculates the Wind Speed (WS) magnitude from eastward and northward wind
    components for specified levels. The calculation is based on the formula:
    ws = sqrt(u^2 + v^2)

    Parameters
    ----------
    levels : list[int  |  str], optional
        Pressure / height levels to compute WS for. The resulting expected input fields
        are u and v wind components pairs for each level. E.g. for level 100 the input
        fields should be [u100, v100], by default [100]
    grid : str | GridDefinition, optional
        Spatial grid configured before coordinate planning, by default "latlon-0.25deg"

    Badges
    ------
    region:global product:wind product:atmos
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        levels: list[int | str] = [100],
        grid: str | GridDefinition = "latlon-0.25deg",
    ) -> None:
        super().__init__(grid)
        self.levels = levels
        self.in_variables = []
        for input_level in [[f"u{level}", f"v{level}"] for level in levels]:
            self.in_variables.extend(input_level)
        self.out_variables = [f"ws{level}" for level in levels]

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        # This function expects [u, v] pairs
        u = x[..., ::2, :, :]
        v = x[..., 1::2, :, :]
        out_tensor = torch.sqrt(u**2 + v**2)
        return out_tensor


class DerivedRH(_DerivedDiagnostic):
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
    grid : str | GridDefinition, optional
        Spatial grid configured before coordinate planning, by default "latlon-0.25deg"

    Badges
    ------
    region:global product:atmos
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        levels: list[int | str] = [100],
        grid: str | GridDefinition = "latlon-0.25deg",
    ) -> None:
        super().__init__(grid)
        self.levels = levels
        self.in_variables = []
        for input_level in [[f"t{level}", f"q{level}"] for level in levels]:
            self.in_variables.extend(input_level)
        self.out_variables = [f"r{level}" for level in levels]
        # Set up pressure levels tensor
        pressure_levels = [100 * float(level) for level in levels]
        # Keep Pa constants in full precision even when the module is cast to half.
        self.pressure_levels = torch.tensor(pressure_levels, dtype=torch.float32)[
            :, None, None
        ]

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        output_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        epsilon = 0.621981
        t = x[..., ::2, :, :]  # K
        q = x[..., 1::2, :, :]  # g/kg
        p = self.pressure_levels.to(x.device)

        e = (p * q * (1.0 / epsilon)) / (1 + q * (1.0 / (epsilon) - 1))

        es_w = 611.21 * torch.exp(17.502 * (t - 273.16) / (t - 32.19))
        es_i = 611.21 * torch.exp(22.587 * (t - 273.16) / (t + 0.7))

        alpha = torch.clip((t - 250.16) / (273.16 - 250.16), 0, 1.0) ** 2
        es = alpha * es_w + (1 - alpha) * es_i
        out_tensor = 100 * e / es
        out_tensor = torch.clamp(out_tensor, 0, 100)

        return out_tensor.to(output_dtype)


class DerivedRHDewpoint(_DerivedDiagnostic):
    """Calculates the surface relative humidity (RH) from dewpoint temperature and air
    temperature. This calculation is based on the August-Roche-Magnus approximation.

    Note
    ----
    For more details see the following references:

    - https://doi.org/10.5194/gmd-9-523-2016 (Eq. B3)
    - https://doi.org/10.5194/tc-2023-8 (Eq. 1 and 5)
    - https://doi.org/10.1175/1520-0450%281996%29035%3C0601%3AIMFAOS%3E2.0.CO%3B2 (Eq. 21)
    - https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#August%E2%80%93Roche%E2%80%93Magnus_formula

    Parameters
    ----------
    grid : str | GridDefinition, optional
        Spatial grid configured before coordinate planning, by default "latlon-0.25deg"

    Badges
    ------
    region:global product:atmos
    provider:nvidia backend:pytorch
    """

    def __init__(self, grid: str | GridDefinition = "latlon-0.25deg") -> None:
        super().__init__(grid)
        self.in_variables = ["t2m", "d2m"]
        self.out_variables = ["r2m"]

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
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
        return out_tensor


class DerivedVPD(_DerivedDiagnostic):
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
    grid : str | GridDefinition, optional
        Spatial grid configured before coordinate planning, by default "latlon-0.25deg"

    Badges
    ------
    region:global product:veg product:atmos
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        levels: list[int | str] = [100],
        grid: str | GridDefinition = "latlon-0.25deg",
    ) -> None:
        super().__init__(grid)
        self.levels = levels
        self.in_variables = []
        for input_level in [[f"t{level}", f"r{level}"] for level in levels]:
            self.in_variables.extend(input_level)
        self.out_variables = [f"vpd{level}" for level in levels]

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        # This function expects [temp, rh] pairs
        t = x[..., ::2, :, :]
        r = x[..., 1::2, :, :]
        L = 2.26e6
        Rv = 461
        es = 6.11 * torch.exp((L / Rv) * ((1.0 / 273.16) - (1.0 / t)))
        out_tensor = es * ((100.0 - r) / 100.0)

        return out_tensor


class DerivedSurfacePressure(_DerivedDiagnostic):
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

    Badges
    ------
    region:global product:atmos
    provider:nvidia backend:pytorch
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
        if tuple(surface_geopotential_coords) != ("lat", "lon"):
            raise ValueError(
                "Surface geopotential coordinates must be ordered lat, lon"
            )
        super().__init__(
            LatLonGrid(
                surface_geopotential_coords["lat"], surface_geopotential_coords["lon"]
            )
        )
        self.temperature_correction = temperature_correction
        self.corr_adjustment = corr_adjustment

        p_levels = torch.as_tensor(p_levels, dtype=torch.float32)
        p_levels, _ = torch.sort(p_levels, descending=True)
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

        if self.temperature_correction and t_levels is not None:
            # apply second-order correction based on temperature
            t0 = t_levels[plevel_indices, all_indices]
            t1 = t_levels[plevel_indices + 1, all_indices]
            tm = 0.5 * (t0 + t1)

            c = self.L / (2 * self.g * self.Rs) / tm**2

            log_p_corr = c * dz * (z_target - z1)

            log_p_corr = self.corr_adjustment[0] + self.corr_adjustment[1] * log_p_corr
            log_p = log_p + log_p_corr

        return torch.exp(log_p).reshape(*data_shape)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        num_levels = len(self.log_p_levels)

        # unpack inputs, swap variable dimension to dim 0, flatten batch dims to dim 1
        variable_dim = 1
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

        return sp_pred


class DerivedTCWV(_DerivedDiagnostic):
    """Calculates the Total Column Water Vapor (TCWV) from specific humidity at
    pressure levels and surface pressure. The calculation is based on the vertical
    integration of specific humidity using the trapezoidal rule:

    TCWV = (1/g) * ∫(p_top to p_surface) q dp

    Discretized as:
    TCWV ≈ (1/g) * Σ [(q_i + q_{i+1})/2 * (p_i - p_{i+1})]

    Note
    ----
    The integration includes the layer between the surface pressure and the lowest
    pressure level, assuming the surface specific humidity equals the lowest level
    specific humidity.

    Parameters
    ----------
    levels : list[int], optional
        Pressure levels (hPa) to use for the integration. They will be sorted
        internally from highest to lowest pressure. Default is
        [1000, 850, 700, 500, 300, 200, 100].
    grid : str | GridDefinition, optional
        Spatial grid configured before coordinate planning, by default "latlon-0.25deg"

    Badges
    ------
    region:global product:atmos
    provider:nvidia backend:pytorch
    """

    g = 9.8067  # Earth's gravitational constant (m/s**2)

    def __init__(
        self,
        levels: list[int] = [1000, 850, 700, 500, 300, 200, 100],
        grid: str | GridDefinition = "latlon-0.25deg",
    ) -> None:
        super().__init__(grid)
        # Sort levels from highest to lowest pressure (descending)
        levels = sorted(levels, reverse=True)
        self.in_variables = [f"q{level}" for level in levels] + ["sp"]
        self.out_variables = ["tcwv"]

        # Store pressure levels in Pa as a buffer
        plevels = torch.tensor(
            [100.0 * float(level) for level in levels], dtype=torch.float32
        )
        self.register_buffer("plevels", plevels)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        q_levels = x[..., :-1, :, :]
        sp = x[..., -1, :, :]
        tcwv = torch.zeros_like(sp)

        # Create mask for pressure levels below surface pressure
        # we will zero anywhere this is not true
        surface_mask = self.plevels.view(1, -1, 1, 1).expand_as(
            q_levels
        ) <= sp.unsqueeze(-3)

        # Integrate from surface to top of atmosphere using trapezoidal rule
        for i, p_level in enumerate(self.plevels):
            if i == 0:
                # First layer: from surface pressure to first pressure level
                # Use q at first level for both endpoints (assume constant in this layer)
                dp = torch.where(
                    surface_mask[..., i, :, :], sp - p_level, torch.zeros_like(sp)
                )
                tcwv = tcwv + q_levels[..., i, :, :] * dp
            else:
                # Subsequent layers: use trapezoidal rule between levels
                level_mask = surface_mask[..., i, :, :] & surface_mask[..., i - 1, :, :]
                dp = self.plevels[i - 1] - p_level

                # Trapezoidal contribution: (q_{i-1} + q_i) / 2 * dp
                contribution = (
                    0.5 * (q_levels[..., i - 1, :, :] + q_levels[..., i, :, :]) * dp
                )
                tcwv = tcwv + torch.where(
                    level_mask, contribution, torch.zeros_like(contribution)
                )

                # Handle layer where lower bound is below the surface
                partial_mask = (
                    surface_mask[..., i, :, :] & ~surface_mask[..., i - 1, :, :]
                )
                if partial_mask.any():
                    dp_partial = sp - p_level
                    tcwv = tcwv + torch.where(
                        partial_mask,
                        q_levels[..., i, :, :] * dp_partial,
                        torch.zeros_like(sp),
                    )

        # Divide by gravity to get TCWV in kg/m^2
        out_tensor = tcwv.unsqueeze(-3) / self.g

        return out_tensor


class DerivedPressureLevels(_DerivedDiagnostic):
    """Interpolates fields from model or pressure levels onto pressure levels.

    Interpolation is linear in the logarithm of pressure, column by column. Levels
    are read from the lexicon's variable names: ``{variable}{p}`` on pressure
    level ``p`` hPa, or ``{variable}{k}k`` on model level ``k``. Outputs are
    ``{variable}{p}`` on each target level. Inputs come in one of three forms:

    - ``input_levels``: pressure levels, such as ``t850``.
    - ``model_levels`` with ``a`` and ``b``: hybrid sigma-pressure model levels,
      ``p = a + b * sp``, plus surface pressure ``sp`` in Pa as the last input.
    - ``model_levels`` with ``pressure``: model levels of any kind whose pressure
      the model publishes as a field on each level, such as ``pres{k}k`` in Pa.

    Used between coupled models, it maps one model's levels onto another's.

    Parameters
    ----------
    variables : list[str]
        Variables to interpolate, such as ``["t", "u", "v"]``.
    levels : list[float]
        Target pressure levels in hPa.
    input_levels : list[float] | None, optional
        Pressure levels of the inputs in hPa, by default None
    model_levels : list[int] | None, optional
        Model level numbers of the inputs, as named in the lexicon (e.g. 1-137 for
        ERA5), from the top of the model down, by default None
    a : Sequence[float] | None, optional
        Hybrid pressure coefficients in Pa: one per model level, or one per
        interface (``len(model_levels) + 1``, as ECMWF publishes them), in which
        case each level takes the mean pressure of its interfaces, by default None
    b : Sequence[float] | None, optional
        Hybrid surface-pressure coefficients, shaped like ``a``, by default None
    pressure : str | None, optional
        Variable holding each model level's pressure in Pa, by default None
    grid : str | GridDefinition, optional
        Spatial grid configured before coordinate planning, by default
        "latlon-0.25deg"
    fill_value : float | None, optional
        Value for target levels outside a column, such as below the surface, by
        default None, which takes the nearest level's value.

    Badges
    ------
    region:global product:atmos
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        variables: list[str],
        levels: list[float],
        input_levels: list[float] | None = None,
        model_levels: list[int] | None = None,
        a: Sequence[float] | None = None,
        b: Sequence[float] | None = None,
        pressure: str | None = None,
        grid: str | GridDefinition = "latlon-0.25deg",
        fill_value: float | None = None,
    ) -> None:
        super().__init__(grid)
        if not variables:
            raise ValueError("DerivedPressureLevels needs at least one variable")
        if (input_levels is None) == (model_levels is None):
            raise ValueError("Pass exactly one of input_levels and model_levels")
        hybrid = a is not None or b is not None
        if input_levels is not None and (hybrid or pressure is not None):
            raise ValueError("a, b and pressure apply to model_levels only")
        if model_levels is not None and hybrid == (pressure is not None):
            raise ValueError("model_levels need either a and b, or pressure")
        _check_pressure_levels(levels)
        self.variables = list(variables)
        self.levels = [float(level) for level in levels]
        self.fill_value = fill_value
        self.pressure = pressure
        self.a = self.b = None
        self.interfaces = False
        if input_levels is not None:
            _check_pressure_levels(input_levels)
            self.input_levels: list[float] | None = [float(p) for p in input_levels]
            names = [_level_name(level) for level in self.input_levels]
        else:
            self.input_levels = None
            model_levels = [int(k) for k in model_levels]  # type: ignore[union-attr]
            if len(model_levels) < 2 or len(set(model_levels)) != len(model_levels):
                raise ValueError("model_levels needs at least two distinct levels")
            names = [f"{k}k" for k in model_levels]
            if hybrid:
                self.a, self.b, self.interfaces = _hybrid_coefficients(
                    a, b, len(model_levels)
                )
        self.nlevel = len(names)
        self.in_variables = [f"{v}{name}" for v in self.variables for name in names]
        if pressure is not None:
            self.in_variables += [f"{pressure}{name}" for name in names]
        if self.a is not None:
            self.in_variables.append("sp")
        self.out_variables = [
            f"{v}{_level_name(level)}" for v in self.variables for level in self.levels
        ]
        # Pressures stay out of module buffers so a half-precision cast of the
        # model cannot overflow them; they are built per call in float32 or wider.

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        nvar, nlevel = len(self.variables), self.nlevel
        work = torch.promote_types(x.dtype, torch.float32)
        values = x[..., : nvar * nlevel, :, :]
        values = values.reshape(*x.shape[:-3], nvar, nlevel, *x.shape[-2:])
        if self.a is not None:
            surface = x[..., -1, :, :].to(work)
            pressure = hybrid_pressure(self.a, self.b, surface, self.interfaces)  # type: ignore[arg-type]
            pressure = pressure.movedim(-1, -3).unsqueeze(-4)
        elif self.pressure is not None:
            pressure = x[..., nvar * nlevel : (nvar + 1) * nlevel, :, :].to(work)
            pressure = pressure.unsqueeze(-4)
        else:
            pressure = torch.tensor(self.input_levels, device=x.device, dtype=work)
            pressure = pressure * 100.0
        target = torch.tensor(self.levels, device=x.device, dtype=work) * 100.0
        output = interpolate_vertical(
            values, pressure, target, dim=-3, log=True, fill_value=self.fill_value
        )
        return output.reshape(*x.shape[:-3], -1, *x.shape[-2:])


def _check_pressure_levels(levels: Sequence[float]) -> None:
    levels = [float(level) for level in levels]
    if not levels or min(levels) <= 0:
        raise ValueError("Pressure levels must be positive")
    if len(set(levels)) != len(levels):
        raise ValueError(f"Pressure levels {levels} repeat a level")


def _hybrid_coefficients(
    a: Sequence[float] | None, b: Sequence[float] | None, nlevel: int
) -> tuple[tuple[float, ...], tuple[float, ...], bool]:
    """Validate hybrid coefficients given per level or per interface."""
    if a is None or b is None or len(a) != len(b):
        raise ValueError("Hybrid coefficients a and b must have equal length")
    if len(a) not in (nlevel, nlevel + 1):
        raise ValueError(
            f"{len(a)} hybrid coefficients for {nlevel} model levels; pass one per "
            "level or one per interface"
        )
    interfaces = len(a) == nlevel + 1
    for surface in (50000.0, 110000.0):
        pressure = hybrid_pressure(
            a, b, torch.tensor(surface, dtype=torch.float64), interfaces
        )
        if (pressure <= 0).any() or (pressure.diff() <= 0).any():
            raise ValueError(
                "Hybrid coefficients must give positive pressure increasing "
                "from the top of the model down"
            )
    return tuple(map(float, a)), tuple(map(float, b)), interfaces


def _level_name(level: float) -> str:
    """Lexicon suffix of a pressure level: ``850`` for 850 hPa, ``2.5`` for 2.5 hPa."""
    return f"{level:g}"
