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

from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import torch
import xarray as xr
import zarr
from cftime import date2num
from loguru import logger

from earth2studio.data import (
    GOES,
    InferenceOutputSource,
    PlanetaryComputerECMWFOpenDataIFS,
    PlanetaryComputerGOES,
    fetch_data,
)
from earth2studio.io import IOBackend, NetCDF4Backend, XarrayBackend, ZarrBackend
from earth2studio.models.dx import DerivedSurfacePressure
from earth2studio.models.px import FCN3, DiagnosticWrapper, InterpModAFNO
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
)
from earth2studio.run import _map_field
from earth2studio.serve.server import (
    Earth2Workflow,
    WorkflowParameters,
    WorkflowProgress,
    WorkflowRegistry,
)
from earth2studio.utils.coords import CoordSystem, coord_array_like, split_coords
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.time import timearray_to_datetime, to_time_array

GOES_MODEL_NAME = "6km_60min_natten_cos_zenith_input_eoe_v2"

_MAX_FORECAST_STEPS = 32
_MAX_ENSEMBLE_SAMPLES = 32


@WorkflowRegistry.instance().register
class FoundryFCN3StormScopeGOESWorkflow(Earth2Workflow):
    """FCN3 (with interpolation) plus StormScope GOES diagnostic ensemble for Foundry."""

    name = "foundry_fcn3_stormscope_goes_workflow"
    description = "FCN3+StormScopeGOES ensemble workflow for Foundry"

    def __init__(
        self,
        device: str = "cuda",
        init_seed: int = 1234,
    ):
        super().__init__()

        self.device = torch.device(device)

        self.fcn3_interp = self.load_fcn3_interp()
        self.stormscope = self.load_stormscope()
        self.rng = np.random.default_rng(init_seed)

        self.data_fcn3 = PlanetaryComputerECMWFOpenDataIFS(verbose=False, cache=False)

        scan_mode = "C"
        self.data_stormscope = {
            satellite: PlanetaryComputerGOES(
                satellite=satellite, scan_mode=scan_mode, verbose=False, cache=False
            )
            for satellite in ["goes16", "goes19"]
        }

        # GOES-16 and GOES19 have the same grid
        goes_lat, goes_lon = GOES.grid(satellite="goes16", scan_mode=scan_mode)
        coords_out = self.fcn3_interp.output_coords(self.fcn3_interp.input_coords())
        self.stormscope.build_input_interpolator(goes_lat, goes_lon)
        self.stormscope.build_conditioning_interpolator(
            coords_out["lat"].values, coords_out["lon"].values
        )

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | WorkflowParameters
    ) -> WorkflowParameters:
        """Validate request parameters, geo_catalog/container_url/output_format, and FCN3/StormScope limits."""
        validated = super().validate_parameters(parameters)
        if not 1 <= validated.n_steps <= _MAX_FORECAST_STEPS:
            raise ValueError(
                f"n_steps must be between 1 and {_MAX_FORECAST_STEPS}, "
                f"got {validated.n_steps}"
            )
        if not 1 <= validated.n_samples_fcn3 <= _MAX_ENSEMBLE_SAMPLES:
            raise ValueError(
                f"n_samples_fcn3 must be between 1 and {_MAX_ENSEMBLE_SAMPLES}, "
                f"got {validated.n_samples_fcn3}"
            )
        if not 1 <= validated.n_samples_stormscope <= _MAX_ENSEMBLE_SAMPLES:
            raise ValueError(
                f"n_samples_stormscope must be between 1 and {_MAX_ENSEMBLE_SAMPLES}, "
                f"got {validated.n_samples_stormscope}"
            )
        if validated.geo_catalog_url is not None:
            if validated.container_url is None:
                raise ValueError(
                    "container_url is required when geo_catalog_url is set."
                )
            if validated.output_format != "netcdf4":
                raise ValueError(
                    "output_format must be 'netcdf4' when geo_catalog_url is set."
                )
        return validated

    def load_fcn3_interp(self) -> InterpModAFNO:
        """Load FCN3 with surface pressure diagnostics and hourly ``InterpModAFNO`` wrapping."""
        logger.info("Loading FCN3")
        package = FCN3.load_default_package()
        fcn3 = FCN3.load_model(package)

        # Surface pressure interpolation
        orography_fn = package.resolve("orography.nc")
        with xr.open_dataset(orography_fn) as ds:
            z_surface = torch.as_tensor(ds["Z"][0].values)
        z_surf_coords = OrderedDict(
            {d: fcn3.input_coords()[d].values for d in ["lat", "lon"]}
        )
        sp_model = DerivedSurfacePressure(
            p_levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
            surface_geopotential=z_surface,
            surface_geopotential_coords=z_surf_coords,
        )

        # Bundle surface pressure with FCN3
        fcn3_sp = DiagnosticWrapper(px_model=fcn3, dx_model=sp_model)

        # Add temporal interpolation to 1 hour
        fcn3_interp = InterpModAFNO.from_pretrained()
        fcn3_interp.px_model = fcn3_sp
        # Diagnose the interpolation endpoint, not the public initial condition.
        fcn3_interp.prepare_endpoint = fcn3_sp._diagnose
        fcn3_interp.to(device=self.device)
        fcn3_interp.eval()
        return fcn3_interp

    def load_stormscope(self) -> StormScopeGOES:
        """Load the StormScope GOES model package and move it to the workflow device."""
        logger.info("Loading StormScope")
        package = StormScopeBase.load_default_package()
        stormscope = StormScopeGOES.load_model(
            package=package,
            conditioning_data_source=None,  # set later
            model_name=GOES_MODEL_NAME,
        )
        stormscope.to(self.device)
        stormscope.eval()
        return stormscope

    def get_seeds(self, n_seeds: int) -> list[int]:
        """Sample ``n_seeds`` distinct integer RNG seeds for ensemble members."""
        seeds = self.rng.choice(2**32, size=n_seeds, replace=False)
        return [int(s) for s in seeds]

    def validate_start_times(
        self, time_stormscope: datetime, time_fcn3: datetime
    ) -> None:
        """Check StormScope (1 h) and FCN3 (6 h) start times and their relative ordering."""
        ref = datetime(1900, 1, 1)
        if (time_stormscope - ref).total_seconds() % (1 * 60 * 60) != 0:
            raise ValueError(
                f"Start time for StormScope must be 1-hour interval: {time_stormscope}"
            )
        if (time_fcn3 - ref).total_seconds() % (6 * 60 * 60) != 0:
            raise ValueError(
                f"Start time for FCN3 must be 6-hour interval: {time_fcn3}"
            )
        if time_stormscope < time_fcn3:
            raise ValueError(
                "Start time for StormScope cannot preceed start time for FCN3"
            )
        if time_stormscope - time_fcn3 > timedelta(hours=12):
            logger.warning(
                "Start times for StormScope and FCN3 should not be more than 12 hours apart but got '%s' and '%s'",
                time_stormscope,
                time_fcn3,
            )

    def validate_samples(
        self, n_samples: int, seeds: Sequence[int] | None
    ) -> list[int]:
        """Return ensemble seeds of length ``n_samples``, generating them if missing."""
        if not seeds:
            return self.get_seeds(n_samples)
        if len(seeds) != n_samples:
            logger.warning(
                "Ignoring requested number of samples because it does not match number of seeds"
            )
        return list(seeds)

    def validate_variables(self, variables: Sequence[str] | None) -> np.ndarray:
        """Resolve StormScope output variables, defaulting to the model's variables."""
        if variables is None:
            variables = self.stormscope.variables
        else:
            unknown_variables = set(variables) - set(self.stormscope.variables)
            if len(unknown_variables):
                raise ValueError(f"Unknown variable(s) {', '.join(unknown_variables)}")
            variables = np.array(variables)
        return variables

    def setup_io(
        self,
        io: IOBackend,
        output_coords: CoordSystem,
        seeds_fcn3: Sequence[int],
        seeds_stormscope: Sequence[int],
    ) -> None:
        """Define IO arrays, CRS metadata, and per-model seeds for ensemble outputs."""
        io.add_array(
            {k: v for k, v in output_coords.items() if k != "variable"},
            output_coords["variable"],
        )

        # Storing seeds separately makes it easier to filter with Titiler
        e_coords = {"ensemble": output_coords["ensemble"]}
        n_stormscope_per_fcn3 = len(seeds_stormscope) // len(seeds_fcn3)
        tiled_seeds_fcn3 = np.repeat(seeds_fcn3, n_stormscope_per_fcn3)
        io.add_array(e_coords, "seed_fcn3", data=torch.tensor(tiled_seeds_fcn3))
        io.add_array(e_coords, "seed_stormscope", data=torch.tensor(seeds_stormscope))

        # Add CRS definition
        io.add_array({}, "crs")
        io.root["crs"].grid_mapping_name = "lambert_conformal_conic"
        io.root["crs"].standard_parallel = 38.5
        io.root["crs"].longitude_of_central_meridian = 262.5
        io.root["crs"].latitude_of_projection_origin = 38.5
        io.root["crs"].semi_major_axis = 6371229
        io.root["crs"].semi_minor_axis = 6371229

        for var in output_coords["variable"]:
            io.root[var].grid_mapping = "crs"

        # Set attributes for automatic parsing of dimensions
        io.root["ensemble"].standard_name = "realization"
        io.root["time"].standard_name = "time"
        io.root["time"].axis = "T"
        io.root["y"].standard_name = "projection_y_coordinate"
        io.root["y"].units = "m"
        io.root["y"].axis = "Y"
        io.root["x"].standard_name = "projection_x_coordinate"
        io.root["x"].units = "m"
        io.root["x"].axis = "X"

        # Unwrap BackendProgress (serve API)
        e2io = (
            io
            if isinstance(io, (NetCDF4Backend, ZarrBackend))
            else getattr(io, "io", None)
        )

        if isinstance(e2io, ZarrBackend):
            zarr.consolidate_metadata(e2io.store)

        if isinstance(e2io, NetCDF4Backend):
            # Planetary Computer does not like the original time format (hours since 0001-01-01).
            # Re-encode from the same datetimes as add_dimension so values match units, then
            # sync so the coordinate is flushed.
            ref_time = np.datetime_as_string(output_coords["time"][0], unit="s")
            units = f"hours since {ref_time.replace('T', ' ')}"
            tv = e2io.root["time"]
            tv.units = units
            tv[:] = date2num(
                timearray_to_datetime(output_coords["time"]),
                units=units,
                calendar=tv.calendar,
            )
            e2io.root.sync()

        return io

    def get_fcn3_input(self, time: datetime) -> xr.DataArray:
        """Fetch FCN3 branch input from Planetary Computer ECMWF IFS."""
        signature = self.fcn3_interp.input_coords()
        field = fetch_data(
            self.data_fcn3,
            time=to_time_array([time]),
            variable=signature["variable"].values,
            lead_time=signature.lead_time.values,
            device=self.device,
        )
        return _map_field(field, signature)

    def get_stormscope_input(self, time: datetime) -> xr.DataArray:
        """Fetch GOES inputs for StormScope (GOES-16 vs GOES-19 by date) and preprocess."""
        coords_in = self.stormscope.input_coords()
        if time < datetime(2025, 4, 7):
            data = self.data_stormscope["goes16"]
        else:
            data = self.data_stormscope["goes19"]
        x, coords = fetch_data(
            data,
            time=to_time_array([time]),
            variable=coords_in["variable"].values,
            lead_time=coords_in["lead_time"].values,
            device=self.device,
        ).e2s.to_torch()

        batch_size = 1
        if x.dim() == 5:
            x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
            coords["batch"] = np.arange(batch_size)
            coords.move_to_end("batch", last=False)

        x, coords = self.stormscope.prep_input(x, coords)
        x = torch.where(self.stormscope.valid_mask, x, torch.nan)

        signature = coord_array_like(
            coords_in, {"batch": coords["batch"], "time": coords["time"]}
        )
        return from_torch(x, signature)

    def run_fcn3(
        self,
        io: IOBackend,
        x: xr.DataArray,
        seed_fcn3: int,
        start_time_stormscope: datetime,
        lead_times: np.ndarray,
        sample: int,
        total_samples: int,
    ) -> None:
        """Run FCN3 to produce conditioning fields for StormScope up to the given horizon."""
        # Create z500 conditioning with FCN3
        coords_in = self.stormscope.input_coords()
        start_time_stormscope = to_time_array([start_time_stormscope])
        variables = self.stormscope.conditioning_variables
        # Start time and lead times are shifted to StormScope start time
        output_coords = {
            "time": start_time_stormscope,
            "lead_time": lead_times,
            "variable": variables,
            "y": coords_in["y"].values,
            "x": coords_in["x"].values,
        }
        io.add_array(
            {k: v for k, v in output_coords.items() if k != "variable"}, variables
        )

        model_gap = int(
            ((start_time_stormscope - x.time.values) / np.timedelta64(1, "h")).item()
        )

        self.fcn3_interp.px_model.px_model.set_rng(seed=seed_fcn3)
        iterator = self.fcn3_interp.create_iterator(x.copy(deep=True))
        conditioning_grid = self.fcn3_interp.output_coords(
            self.fcn3_interp.input_coords()
        )

        n_steps = model_gap + len(lead_times)
        for step, x in enumerate(iterator):
            # Update progress for FCN3 step
            msg = (
                f"Processing FCN3 for sample {sample + 1}/{total_samples} "
                f"(seed_fcn3={seed_fcn3}) "
                f"step {step + 1}/{n_steps}"
            )
            progress = WorkflowProgress(
                progress=msg,
                current_step=step + 1,
                total_steps=n_steps,
            )
            self.update_progress(progress)
            logger.info(msg)

            if step < model_gap:
                # Skip initial steps leading up to StormScope start time
                continue

            tensor, coords_x = x.sel(
                variable=variables,
                lat=conditioning_grid.lat.values,
                lon=conditioning_grid.lon.values,
            ).e2s.to_torch()
            tensor, coords_x = self.stormscope.prep_input(
                tensor, coords_x, conditioning=True
            )
            coords_x["time"] = start_time_stormscope
            coords_x["lead_time"] = coords_x["lead_time"] - np.timedelta64(
                model_gap, "h"
            )
            io.write(*split_coords(tensor, coords_x))

            if step == (n_steps - 1):
                break

    def run_stormscope(
        self,
        io: IOBackend,
        y: xr.DataArray,
        seed_fcn3: int,
        seed_stormscope: int,
        lead_times: np.ndarray,
        variables: np.ndarray,
        sample: int,
        total_samples: int,
    ) -> None:
        """Run StormScope autoregressively and write outputs to ``io``."""
        n_steps = len(lead_times)

        def log_progress(step: int) -> None:
            msg = (
                f"Processing sample {sample + 1}/{total_samples} "
                f"(seed_fcn3={seed_fcn3}, seed_stormscope={seed_stormscope}), "
                f"step {step + 1}/{n_steps}"
            )
            progress = WorkflowProgress(
                progress=msg,
                current_step=step + 1,
                total_steps=n_steps,
            )
            self.update_progress(progress)
            logger.info(msg)

        def prep_output(
            y_pred: xr.DataArray,
        ) -> xr.DataArray:
            y_out = y_pred.sel(variable=variables).rename(batch="ensemble")
            y_out = y_out.assign_coords(ensemble=[sample])
            valid_time = y_out.time.values + y_out.lead_time.values[0]
            return y_out.isel(lead_time=0, drop=True).assign_coords(time=valid_time)

        # Update progress for step within sample
        log_progress(0)

        # Store initial GOES data (identical across seeds)
        y_out = prep_output(y.isel(lead_time=slice(-1, None)))
        io.write(*split_coords(*y_out.e2s.to_torch()))

        self.stormscope.set_rng(seed_stormscope)

        for step in range(1, n_steps):
            y_pred = self.stormscope(y)

            # Update progress for step within sample
            log_progress(step)

            y_out = prep_output(y_pred)
            io.write(*split_coords(*y_out.e2s.to_torch()))

            if step == (n_steps - 1):
                break

            y = self.stormscope.next_input(y_pred, y)

    def __call__(
        self,
        io: IOBackend,
        start_time_fcn3: datetime = datetime(2025, 1, 1, 18),
        start_time_stormscope: datetime = datetime(2025, 1, 1, 18),
        n_steps: int = 12,
        n_samples_fcn3: int = 1,
        n_samples_stormscope: int = 1,
        seeds_fcn3: Sequence[int] | None = None,
        seeds_stormscope: Sequence[int] | None = None,
        variables: Sequence[str] | None = ("abi01c", "abi02c", "abi03c"),
        output_format: Literal["zarr", "netcdf4"] = "netcdf4",
        container_url: str | None = None,
        geo_catalog_url: str | None = None,
        collection_id: str | None = None,
    ) -> None:
        self.validate_start_times(start_time_stormscope, start_time_fcn3)
        lead_times = np.array([np.timedelta64(i, "h") for i in range(n_steps + 1)])
        # Different StormScope seed for every trajectory
        if n_samples_stormscope % n_samples_fcn3 != 0:
            raise ValueError(
                "'n_samples_stormscope' must be divisible by 'n_samples_fcn3'"
            )
        seeds_fcn3 = self.validate_samples(n_samples_fcn3, seeds_fcn3)
        seeds_stormscope = self.validate_samples(n_samples_stormscope, seeds_stormscope)
        n_stormscope_per_fcn3 = len(seeds_stormscope) // len(seeds_fcn3)
        variables = self.validate_variables(variables)

        x_ori = self.get_fcn3_input(start_time_fcn3)
        y_ori = self.get_stormscope_input(start_time_stormscope)

        coords_out = self.stormscope.output_coords(self.stormscope.input_coords())
        output_coords = {
            "ensemble": np.arange(len(seeds_stormscope)),
            # Planetary Computer does not like separate 'lead_time'
            "time": to_time_array([start_time_stormscope]) + lead_times,
            "variable": variables,
            "y": coords_out["y"].values,
            "x": coords_out["x"].values,
        }
        self.setup_io(io, output_coords, seeds_fcn3, seeds_stormscope)

        total_samples = len(seeds_stormscope)
        sample = 0
        for seed_fcn3 in seeds_fcn3:
            # Generate FCN3 conditioning (z500)
            logger.info("Starting FCN3 inference")
            io_fcn3 = XarrayBackend()
            self.run_fcn3(
                io=io_fcn3,
                x=x_ori.copy(deep=True),
                seed_fcn3=seed_fcn3,
                start_time_stormscope=start_time_stormscope,
                lead_times=lead_times,
                sample=sample,
                total_samples=total_samples,
            )
            self.stormscope.conditioning_data_source = InferenceOutputSource(
                io_fcn3.root
            )

            # Run StormScope forecast conditioned on FCN3
            logger.info("Starting StormScope inference")
            for _ in range(n_stormscope_per_fcn3):
                self.run_stormscope(
                    io=io,
                    y=y_ori.copy(deep=True),
                    seed_fcn3=seed_fcn3,
                    seed_stormscope=seeds_stormscope[sample],
                    lead_times=lead_times,
                    variables=variables,
                    sample=sample,
                    total_samples=total_samples,
                )
                sample += 1
