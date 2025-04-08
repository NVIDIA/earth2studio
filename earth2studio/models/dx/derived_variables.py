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

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

# if possible variable names follow the Earth2Studio vocabolary defined in earth2studio.lexicon.base.E2STUDIO_VOCAB
IN_VARIABLES = {
    "vpd_from_r_t": ["r", "t"],
    "ws_from_u_v": ["u", "v"],
    "r_from_q_t_sp": ["q", "t", "sp"],
    "r_from_d_t": ["d", "t"],
}
single_level_variables = [
    "sp",
    "msl",
    "tcwv",
    "tp",
    "tpp",
    "tpi",
    "tp06",
    "lsm",
    "zsl",
    "uvcossza",
    "refc",
    "csnow",
    "cicep",
    "cfrzr",
    "crain",
]


class DerivedVariables(torch.nn.Module, AutoModelMixin):
    """
    A derived variable computation module for Earth system models.

    The `DerivedVariables` class is designed to compute various derived meteorological and atmospheric variables
    from input variables provided in a tensor format. The derived variables are calculated based on specific
    meteorological formulas and input variables such as relative humidity (r), temperature (t), wind components (u, v),
    specific humidity (q), and more. The class supports multiple vertical levels and handles the coordinate
    transformations required for the input and output tensors.

    Parameters
    ----------
    calculation_name : str
        The name of the derived variable calculation to perform. Supported calculations are defined in the `IN_VARIABLES`
        dictionary and include:
        - 'vpd_from_r_t': Vapor Pressure Deficit from relative humidity and temperature.
        - 'ws_from_u_v': Wind Speed from u and v wind components.
        - 'r_from_q_t_sp': Relative Humidity from specific humidity, temperature, and pressure.
        - 'r_from_d_t': Relative Humidity from dew point temperature and air temperature.

    levels : list[int | str]
        A list of vertical levels for which to compute the derived variables. Levels can be specified as integers (e.g., 850)
        or strings (e.g., '10m') to handle different pressure or height levels.

    Attributes
    ----------
    calculation_name : str
        The name of the derived variable calculation to perform.

    var : str
        The base name of the derived variable calculated, extracted from `calculation_name`.

    levels : list[int | str]
        The list of vertical levels for which the derived variables are computed.

    in_variables : list[str]
        The list of input variable names required for the derive variable calculation, adjusted to include level information.

    Methods
    -------
    input_coords() -> CoordSystem:
        Returns the input coordinate system required for the derived variable calculation.

    output_coords(input_coords: CoordSystem) -> CoordSystem:
        Transforms the input coordinate system into the output coordinate system for the derived variables.

    get_variable(x: torch.Tensor, var: str) -> torch.Tensor:
        Extracts a specific variable from the input tensor based on the variable name.

    get_vpd_from_r_t(x: torch.Tensor, levels) -> torch.Tensor:
        Calculates the Vapor Pressure Deficit (VPD) from relative humidity and temperature for specified levels.

    get_ws_from_u_v(x: torch.Tensor, levels: list) -> torch.Tensor:
        Calculates the Wind Speed (WS) from u and v wind components for specified levels.

    get_r_from_q_t_sp(x: torch.Tensor, levels: list) -> torch.Tensor:
        Calculates the Relative Humidity (R) from specific humidity, temperature, and pressure for specified levels.

    get_r_from_d_t(x: torch.Tensor, levels: list) -> torch.Tensor:
        Calculates the Relative Humidity (R) from dew point temperature and air temperature for specified levels.

    __call__(x: torch.Tensor, coords: CoordSystem) -> tuple[torch.Tensor, CoordSystem]:
        Performs the forward pass to compute the derived variables, transforming input tensors and coordinates as needed.

    Notes
    -----
    The `DerivedVariables` class integrates with other modules in the Earth2Studio framework, including data handling,
    model configurations, and coordinate transformations. It supports batch processing of input tensors and ensures that
    the input and output coordinates are correctly aligned, facilitating seamless integration with larger Earth system models.
    """

    def __init__(self, calculation_name: str, levels: list[int | str]) -> None:

        super().__init__()
        self.calculation_name = calculation_name
        self.var = self.calculation_name.split("_")[0]
        self.levels = levels

        in_variables = []
        for var in IN_VARIABLES[self.calculation_name]:
            if var not in single_level_variables:
                in_variables_sub = [var + str(level) for level in self.levels]
            else:
                in_variables_sub = [var]
            in_variables.extend(in_variables_sub)

        self.in_variables = in_variables
        pass

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
                "lat": np.linspace(90, -90, 721, endpoint=True),
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
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        out_variables = [self.var + str(x) for x in self.levels]
        output_coords["variable"] = np.array(out_variables)

        return output_coords

    def get_variable(self, x: torch.Tensor, var: str) -> torch.Tensor:
        """
        Extracts a specific variable from the input tensor based on the variable name.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing all variables.
        var : str
            Name of the variable to extract.

        Returns
        -------
        torch.Tensor
            The extracted variable as a tensor.
        """
        index = self.in_variables.index(var)
        return x[:, index]

    def get_vpd_from_r_t(self, x: torch.Tensor, levels: list) -> torch.Tensor:
        """
        Calculates the vapor pressure deficit (VPD) for multiple levels from the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing all variables.
        levels : list
            List of levels for which to calculate the VPD.

        Returns
        -------
        torch.Tensor
            Vapor pressure deficit (VPD) for each level as a tensor.

        Notes
        -----
        The calculation is based on the formula: VPD = es * ((100 - rh) / 100),
        where es is the saturation vapor pressure calculated using the formula:
        es = 6.11 * exp((L / Rv) * ((1 / 273) - (1 / t))),
        where L is the latent heat of vaporization (2.5e6 J/kg), Rv is the gas constant for water vapor (461 J/kg/K).
        The function iterates over the specified levels, calculates the VPD for each level, and returns the results as a tensor.
        """
        out_list = []
        for level in levels:
            r = self.get_variable(x, "r" + str(level))
            t = self.get_variable(x, "t" + str(level))
            L = 2.5e6
            Rv = 461
            es = 6.11 * torch.exp((L / Rv) * ((1 / 273) - (1 / t)))
            vpd = es * ((100 - r) / 100)
            out_list.append(vpd)
        out_tensor = torch.stack(out_list, dim=1)
        return out_tensor

    def get_ws_from_u_v(self, x: torch.Tensor, levels: list) -> torch.Tensor:
        """
        Calculates the wind speed (WS) for multiple levels from the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing all variables.
        levels : list
            List of levels for which to calculate the WS.

        Returns
        -------
        torch.Tensor
            Wind speed (WS) for each level as a tensor.

        Notes
        -----
        The calculation is based on the formula: WS = sqrt(u^2 + v^2),
        where u and v are the zonal and meridional wind components, respectively.
        The function iterates over the specified levels, calculates the WS for each level, and returns the results as a tensor.
        """
        out_list = []
        for level in levels:
            u = self.get_variable(x, "u" + str(level))
            v = self.get_variable(x, "v" + str(level))
            ws = torch.sqrt(u**2 + v**2)
            out_list.append(ws)
        out_tensor = torch.stack(out_list, dim=1)
        return out_tensor

    def get_r_from_q_t_sp(self, x: torch.Tensor, levels: list) -> torch.Tensor:
        """
        Calculates the relative humidity (R) for multiple levels from the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing all variables.
        levels : list
            List of levels for which to calculate the R.

        Returns
        -------
        torch.Tensor
            Relative humidity (R) for each level as a tensor.

        Notes
        -----
        The calculation is based on the formula: R = 0.263 * p * q * (1 / exp((17.67 * (t - 273.16)) / (t - 29.65))),
        where p is the pressure, q is the specific humidity, and t is the temperature in Kelvin.
        The function iterates over the specified levels, calculates the R for each level, and returns the results as a tensor.
        """
        out_list = []
        for level in levels:
            q = self.get_variable(x, "q" + str(level))
            t = self.get_variable(x, "t" + str(level))

            if isinstance(level, (int, float)):
                p = level * 100
            else:
                p = self.get_variable(
                    x, "sp"
                )  # use surface pressure for variables not specified on pressure levels
            r = 0.263 * p * q * (1 / (torch.exp((17.67 * (t - 273.16)) / (t - 29.65))))
            out_list.append(r)
        out_tensor = torch.stack(out_list, dim=1)
        return out_tensor

    def get_r_from_d_t(self, x: torch.Tensor, levels: list) -> torch.Tensor:
        """
        Calculates the relative humidity (R) from dewpoint temperature (d) and air temperature (a) for multiple levels.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing all variables.
        levels : list
            List of levels for which to calculate the R.

        Returns
        -------
        torch.Tensor
            Relative humidity (R) for each level as a tensor.
        """
        out_list = []
        for level in levels:
            d = self.get_variable(x, "d" + str(level))
            t = self.get_variable(x, "t" + str(level))
            # Applying August–Roche–Magnus formula to calculate the saturation vapor pressure (es) and vapor pressure (e)
            es = 6.11 * np.exp((17.67 * d) / (d + 243.04))
            e = 6.11 * np.exp((17.67 * t) / (t + 243.04))
            r = (e / es) * 100
            out_list.append(r)
        out_tensor = torch.stack(out_list, dim=1)
        return out_tensor

    calculation_functions = {
        "vpd_from_r_t": get_vpd_from_r_t,
        "ws_from_u_v": get_ws_from_u_v,
        "r_from_q_t_sp": get_r_from_q_t_sp,
        "r_from_d_t": get_r_from_d_t,
    }

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Forward pass of diagnostic.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinates.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinates.
        """

        output_coords = self.output_coords(coords)

        if self.calculation_name not in self.calculation_functions:
            raise ValueError(
                f"Unsupported calculation_function: {self.calculation_name}"
            )

        calculation_function = self.calculation_functions[self.calculation_name]
        out_tensor = calculation_function(self, x, self.levels)

        return out_tensor, output_coords


def run_example() -> None:
    """
    Demonstrate the usage of the DerivedVariables class with an example.

    This function sets up the necessary components, including the DerivedVariables diagnostic model,
    the prognostic model SFNO, data source (GFS), and input/output backend (ZarrBackend).
    It then runs a diagnostic to compute derived variables for a specific date and time.
    """
    from earth2studio import run
    from earth2studio.data import GFS
    from earth2studio.io import ZarrBackend
    from earth2studio.models.px import SFNO

    dx = DerivedVariables("ws_from_u_v", levels=[850, 500, "10m"])
    px = SFNO.load_model(SFNO.load_default_package())
    data = GFS()
    io = ZarrBackend()
    io = run.diagnostic(["2022-01-14 00:00:00"], 4, px, dx, data=data, io=io)


if __name__ == "__main__":
    run_example()
