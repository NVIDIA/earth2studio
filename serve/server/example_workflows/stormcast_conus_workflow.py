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

from earth2studio import run
from earth2studio.data import GFS, GFS_FX, HRRR, InferenceOutputSource
from earth2studio.io import IOBackend, NetCDF4Backend, XarrayBackend
from earth2studio.models.auto import Package
from earth2studio.models.dx import DerivedSurfacePressure, DerivedTCWV
from earth2studio.models.px import (
    AIFS,
    FCN3,
    DiagnosticWrapper,
    InterpModAFNO,
    StormCastCONUS,
)
from earth2studio.serve.server import Earth2Workflow, WorkflowRegistry


@WorkflowRegistry.instance().register
class StormCastCONUSWorkflow(Earth2Workflow):
    """StormCast-CONUS deterministic forecast workflow.

    Runs a global conditioning model produces large-scale
    atmospheric state at 1-hour resolution over the forecast window, then
    StormCastCONUS downscales that output to high-resolution CONUS fields using
    HRRR initial conditions.

    The global conditioning stage can be driven by FourCastNet v3 (FCN3), AIFS,
    or GFS operational forecasts. FCN3 and AIFS outputs are temporally
    interpolated to 1-hour resolution via InterpModAFNO before being passed to
    StormCastCONUS.
    """

    name = "stormcast_conus_workflow"
    description = "StormCastCONUS workflow"

    def __init__(
        self,
        conditioning_model: Literal["fcn3", "aifs", "gfs"] = "fcn3",
        conditioning_result_storage: Literal["memory", "file"] = "memory",
        device: str = "cuda",
    ):
        """Initialize the StormCastCONUS workflow.

        Parameters
        ----------
        conditioning_model : {"fcn3", "aifs", "gfs"}, optional
            Global model used to produce large-scale conditioning fields.
            ``"fcn3"`` uses FourCastNet v3 with surface-pressure derivation and
            temporal interpolation; ``"aifs"`` uses ECMWF AIFS with TCWV
            derivation and temporal interpolation; ``"gfs"`` streams GFS
            operational forecasts directly without additional processing.
            Defaults to ``"fcn3"``.
        conditioning_result_storage : {"memory", "file"}, optional
            Where to cache the conditioning model output before passing it to
            StormCastCONUS.  ``"memory"`` keeps the data in an in-memory
            ``XarrayBackend``; ``"file"`` writes a temporary NetCDF4 file to
            disk.  Only used when ``conditioning_model`` is ``"fcn3"`` or
            ``"aifs"``.  Defaults to ``"memory"``.
        device : str, optional
            PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``) passed to all
            loaded models.  Defaults to ``"cuda"``.
        """
        super().__init__()

        self.conditioning_result_storage = conditioning_result_storage
        self.device = device

        # models
        ## FourCastNet 3
        if conditioning_model == "fcn3":
            self.conditioning_model = self._setup_fcn3()
        elif conditioning_model == "aifs":
            self.conditioning_model = self._setup_aifs()
        elif conditioning_model == "gfs":
            self.conditioning_model = GFS_FX()
        else:
            raise ValueError('conditioning_model must be one of "fcn3", "aifs", "gfs".')

        ## Pretrained StormCast model
        # TODO: replace with from_pretrained once model is public
        # self.stormcast_conus = StormCastCONUS.from_pretrained()
        model_path = os.environ.get("STORMCAST_CONUS_MODEL_PATH", "stormcast-conus")
        package = Package(
            model_path,
            cache_options={
                "cache_storage": Package.default_cache("stormcast-conus"),
                "same_names": True,
            },
        )
        self.stormcast_conus = StormCastCONUS.load_model(package)

        # initial conditions
        self.gfs_ic = GFS()
        self.hrrr_ic = HRRR()

    def _setup_fcn3(self):
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
        fcn3_interp = InterpModAFNO.from_pretrained()
        fcn3_interp.px_model = fcn3_sp
        fcn3_interp.to(device=self.device)

        return fcn3_interp

    def _setup_aifs(self):
        aifs = AIFS.from_pretrained()

        # Total column water vapor (TCWV), missing from AIFS
        tcwv_model = DerivedTCWV(
            levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        )

        ## Bundle TCWV to AIFS
        aifs_tcwv = DiagnosticWrapper(px_model=aifs, dx_model=tcwv_model)

        ## Temporal interpolation to 1h resolution
        fcn3_interp = InterpModAFNO.from_pretrained()
        fcn3_interp.px_model = aifs_tcwv
        fcn3_interp.to(device=self.device)

        return fcn3_interp

    def __call__(
        self,
        io: IOBackend,
        start_time: datetime = datetime(2025, 10, 15, 0),
        num_hours: int = 12,
    ) -> None:
        """Execute the StormCastCONUS forecast.

        If the conditioning model is FCN3 or AIFS, first runs a deterministic
        global forecast initialized from GFS analysis, stores the results
        according to ``conditioning_result_storage``, and attaches them to
        StormCastCONUS as a data source.  If the conditioning model is GFS, the
        GFS_FX forecast source is used directly.  StormCastCONUS then runs a
        deterministic downscaling forecast initialized from HRRR analysis and
        writes all outputs to ``io``.

        Parameters
        ----------
        io : IOBackend
            Output backend where StormCastCONUS forecast fields are written.
        start_time : datetime, optional
            Forecast initialization time.  Defaults to
            ``datetime(2025, 10, 15, 0)``.
        num_hours : int, optional
            Number of forecast hours to produce.  Defaults to ``12``.
        """
        if isinstance(self.conditioning_model, GFS_FX):
            conditioning_source = self.conditioning_model
        else:
            if self.conditioning_result_storage == "memory":
                conditioning_results = XarrayBackend()
            else:
                tmp_dir = TemporaryDirectory()
                tmp_file = os.path.join(tmp_dir.name, "fcn3_output.nc")
                conditioning_results = NetCDF4Backend(  # type: ignore[assignment]
                    tmp_file, backend_kwargs={"mode": "w", "diskless": False}
                )  # when using temporary file

            # run conditioning model
            run.deterministic(
                [start_time],
                num_hours,
                self.conditioning_model,
                self.gfs_ic,
                conditioning_results,
                device=self.device,
            )

            # set StormCastCONUS to use conditioning output
            if self.conditioning_result_storage == "memory":
                # XarrayBackend has a root attribute, but IOBackend doesn't
                conditioning_source = InferenceOutputSource(conditioning_results.root)  # type: ignore[attr-defined]
            else:
                conditioning_source = InferenceOutputSource(
                    tmp_file
                )  # when using temporary file

        self.stormcast_conus.conditioning_data_source = conditioning_source

        # run StormCastCONUS
        run.deterministic(
            [start_time],
            num_hours,
            self.stormcast_conus,
            self.hrrr_ic,
            io,
            device=self.device,
        )
