#!/usr/bin/env python3
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
import os
from collections import OrderedDict
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Literal

import torch
import xarray as xr

from api_server.workflow import Earth2Workflow, workflow_registry
from earth2studio import run
from earth2studio.data import GFS, HRRR, InferenceOutputSource
from earth2studio.io import IOBackend, NetCDF4Backend, XarrayBackend
from earth2studio.models.dx import DerivedSurfacePressure
from earth2studio.models.px import FCN3, DiagnosticWrapper, InterpModAFNO, StormCast


@workflow_registry.register
class StormCastFCN3Workflow(Earth2Workflow):
    name = "stormcast_fcn3_workflow"
    description = "StormCast + FCN3 workflow"

    def __init__(
        self,
        fcn3_result_storage: Literal["memory", "file"] = "memory",
        device: str = "cuda",
    ):
        super().__init__()

        self.fcn3_result_storage = fcn3_result_storage
        self.device = device

        # models
        ## FourCastNet 3
        fcn3_package = FCN3.load_default_package()
        fcn3 = FCN3.load_model(fcn3_package)

        ## Surface pressure interpolation (needed by downstream models)
        orography_fn = fcn3_package.resolve("orography.nc")
        with xr.open_dataset(orography_fn) as ds:
            z_surface = torch.as_tensor(ds["Z"][0].values)
        z_surf_coords = OrderedDict({d: fcn3.input_coords()[d] for d in ["lat", "lon"]})
        sp_model = DerivedSurfacePressure(
            p_levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
            surface_geopotential=z_surface,
            surface_geopotential_coords=z_surf_coords,
        )

        ## Bundle surface pressure to FCN3
        fcn3_sp = DiagnosticWrapper(px_model=fcn3, dx_model=sp_model)

        ## Temporal interpolation to 1h resolution
        self.fcn3_interp = InterpModAFNO.from_pretrained()
        self.fcn3_interp.px_model = fcn3_sp
        self.fcn3_interp.to(device=self.device)

        ## Pretrained StormCast model
        self.stormcast = StormCast.from_pretrained()

        # initial conditions
        self.gfs_ic = GFS()
        self.hrrr_ic = HRRR()

    def __call__(
        self,
        io: IOBackend,
        start_time: datetime = datetime(2024, 1, 1, 0),
        num_hours: int = 10,
        run_stormcast: bool = True,
    ) -> None:
        if not run_stormcast:
            fcn3_results: IOBackend = io
        elif self.fcn3_result_storage == "memory":
            fcn3_results = XarrayBackend()
        else:
            tmp_dir = TemporaryDirectory()
            tmp_file = os.path.join(tmp_dir.name, "fcn3_output.nc")
            fcn3_results = NetCDF4Backend(  # type: ignore[assignment]
                tmp_file, backend_kwargs={"mode": "w", "diskless": False}
            )  # when using temporary file

        # run suface pressure interpolated FCN3
        run.deterministic(
            [start_time],
            num_hours,
            self.fcn3_interp,
            self.gfs_ic,
            fcn3_results,
            device=self.device,
        )

        if not run_stormcast:  # return FCN3 result without StormCast
            return

        # set StormCast to use FCN3 output
        if self.fcn3_result_storage == "memory":
            # XarrayBackend has a root attribute, but IOBackend doesn't
            source = InferenceOutputSource(fcn3_results.root)  # type: ignore[attr-defined]
        else:
            source = InferenceOutputSource(tmp_file)  # when using temporary file
        self.stormcast.conditioning_data_source = source

        # run StormCast
        run.deterministic(
            [start_time],
            num_hours,
            self.stormcast,
            self.hrrr_ic,
            io,
            device=self.device,
        )
