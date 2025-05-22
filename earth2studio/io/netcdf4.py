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
import collections
from collections.abc import Iterator
import os
from string import Template
from typing import Any

import xarray as xr
import fsspec
import numpy as np
import torch
from cftime import date2num, num2date
from netCDF4 import Dataset, Variable

from earth2studio.utils.coords import convert_multidim_to_singledim
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

units_map = {
    "h": "hour",
    "D": "day",
    "s": "second",
    "m": "minute",
    "Y": "year",
}
rev_units_map = {v: k for k, v in units_map.items()}



class NetCDFBackend:

    _index: int = 0

    def __init__(
        self,
        fs: fsspec.spec.AbstractFilesystem,
        ft: Template = Template("earth2studio_$index.nc"),
        blocking: bool = False,
        pool_size: int = 1,
    ):
        self.fs = fs
        self.ft = ft
        self.blocking = blocking
        self.pool_size = pool_size
        self.engine = "netcdf4"
        

    def write(self, x: torch.tensor, coords: CoordSystem, ft_kwargs: dict[str, Any] = {}) -> None:


    async def write_async(self, x: torch.tensor, coords: CoordSystem, ft_kwargs: dict[str, Any] = {} ) -> None:


        # potential not thread safe, todo
        self._index += 1

    def consolidate(self, files: Iterator[dict[str, Any]] | "NestedIterator[dict[str, Any]]", file_name: str = "earth2studio_consolidated.nc"):

        index = 0
        def for_each_nested(nested_iter: Iterator[Any]) -> None:
            output = []
            for item in nested_iter:
                if isinstance(item, collections.Iterator):
                    output.append(for_each_nested(item))
                else:
                    # TODO: Check file exists
                    file = self.fs.open(self.ft.safe_substitute(index=index, **item))
                    output.append(xr.open_dataset(file, engine=self.engine ))

        consolidated_ds = xr.combine_nested(for_each_nested(files))
        consolidated_ds.to_netcdf("temp.nc", engine=self.engine)
        # Move to remote file store (TODO async)
        with self.fs.open(file_name, "wb") as f:
            with open("temp.nc", "rb") as temp:
                f.write(temp.read())

        if os.path.exists("temp.nc"):
            os.remove("temp.nc")


    # IDK if this is needed
    async def consolidate_async(self, files: Iterator[dict[str, Any]] | "NestedIterator[dict[str, Any]]"):