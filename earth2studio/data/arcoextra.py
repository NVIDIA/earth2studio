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

import asyncio
import os
import pathlib
import shutil
from datetime import datetime
from importlib.metadata import version

import fsspec
import gcsfs
import numpy as np
import xarray as xr
import zarr
from fsspec.implementations.cached import WholeFileCacheFileSystem
from loguru import logger
from tqdm import tqdm

from earth2studio.data.utils import (
    datasource_cache_root,
    prep_data_inputs,
    unordered_generator,
)
from earth2studio.lexicon import ARCOLexicon
from earth2studio.utils.type import TimeArray, VariableArray
from earth2studio.data.arco import ARCO

class ARCOExtraLexicon(ARCOLexicon):
    VOCAB = ARCOLexicon.VOCAB | {
        "w50": "vertical_velocity::50",
        "w100": "vertical_velocity::100",
        "w150": "vertical_velocity::150",
        "w200": "vertical_velocity::200",
        "w250": "vertical_velocity::250",
        "w300": "vertical_velocity::300",
        "w400": "vertical_velocity::400",
        "w500": "vertical_velocity::500",
        "w600": "vertical_velocity::600",
        "w700": "vertical_velocity::700",
        "w850": "vertical_velocity::850",
        "w925": "vertical_velocity::925",
        "w1000": "vertical_velocity::1000",
        "tp06": "total_precipitation_6hr::",
        "z": "geopotential_at_surface::",
        "lsm": "land_sea_mask::",
    }

    INV_VOCAB = {v: k for k, v in VOCAB.items()}

class ARCOExtra:
    """Custom ARCOExtra datasource
    https://nvidia.github.io/earth2studio/examples/extend/03_custom_datasource.html#custom-data-source
    """

    relative_humidity_ids = [
        "r50",
        "r100",
        "r150",
        "r200",
        "r250",
        "r300",
        "r400",
        "r500",
        "r600",
        "r700",
        "r850",
        "r925",
        "r1000",
    ]
    tp_accumulated_ids = ["tp06"]
    static = ["z", "lsm"]

    def __init__(self, cache: bool = True, verbose: bool = True):
        self.arco = ARCO(cache, verbose)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in IFS lexicon.

        Returns
        -------
        xr.DataArray
        """
        time, variable = prep_data_inputs(time, variable)

        # Replace relative humidity with respective temperature
        # and specifc humidity fields
        variable_expanded = []
        for v in variable:
            if v in self.relative_humidity_ids:
                level = int(v[1:])
                variable_expanded.extend([f"t{level}", f"q{level}"])
            elif v in self.tp_accumulated_ids:
                variable_expanded.extend(["tp"])
            else:
                variable_expanded.append(v)
        variable_expanded = list(set(variable_expanded))

        # Fetch
        da_exp = self.arco(time, variable_expanded)

        # Calculate relative humidity when needed
        arrays = []
        for v in variable:
            if v in self.relative_humidity_ids:
                level = int(v[1:])
                t = da_exp.sel(variable=f"t{level}").values
                q = da_exp.sel(variable=f"q{level}").values
                rh = self.calc_relative_humdity(t, q, 100 * level)
                arrays.append(rh)
            elif v in self.tp_accumulated_ids:
                # Accumulate every 6 hours, sometime >1 time
                tp06_singles = []
                for t in time:
                    start_accumul = t - timedelta(hours=5)
                    accum_da = xr.concat(
                        [
                            self.arco(d, ["tp"])
                            for d in pd.date_range(start=start_accumul, end=t, freq="H")
                        ],
                        dim="time",
                    )
                    tp06_singles.append(accum_da)
                all_tp = (
                    xr.concat(tp06_singles, dim="time")
                    .resample(time="6h", closed="right", label="right")
                    .sum("time")
                    .sel(time=time)
                )
                arrays.append(all_tp.sel(variable="tp").values)
            elif v in self.static:
                arrays.append(da_exp.sel(variable=v).values)
            else:
                arrays.append(da_exp.sel(variable=v).values)

        da = xr.DataArray(
            data=np.stack(arrays, axis=1),
            dims=["time", "variable", "lat", "lon"],
            coords=dict(
                time=da_exp.coords["time"].values,
                variable=np.array(variable),
                lat=da_exp.coords["lat"].values,
                lon=da_exp.coords["lon"].values,
            ),
        )
        return da

    def calc_relative_humdity(
        self, temperature: np.array, specific_humidity: np.array, pressure: float
    ) -> np.array:
        """Relative humidity calculation

        Parameters
        ----------
        temperature : np.array
            Temperature field (K)
        specific_humidity : np.array
            Specific humidity field (g.kg-1)
        pressure : float
            Pressure (Pa)

        Returns
        -------
        np.array
        """
        epsilon = 0.621981
        p = pressure
        q = specific_humidity
        t = temperature

        e = (p * q * (1.0 / epsilon)) / (1 + q * (1.0 / (epsilon) - 1))

        es_w = 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))
        es_i = 611.21 * np.exp(22.587 * (t - 273.16) / (t + 0.7))

        alpha = np.clip((t - 250.16) / (273.16 - 250.16), 0, 1.2) ** 2
        es = alpha * es_w + (1 - alpha) * es_i
        rh = 100 * e / es

        return rh
