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

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import torch
import zarr
from cftime import date2num

from earth2studio.data import PlanetaryComputerECMWFOpenDataIFS, fetch_data
from earth2studio.io import IOBackend, NetCDF4Backend, ZarrBackend
from earth2studio.models.px import FCN3
from earth2studio.serve.server import (
    Earth2Workflow,
    WorkflowParameters,
    WorkflowProgress,
    workflow_registry,
)
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import timearray_to_datetime, to_time_array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("foundry_fcn3_workflow")

_MAX_FORECAST_STEPS = 400
_MAX_ENSEMBLE_SAMPLES = 32


@workflow_registry.register
class FoundryFCN3Workflow(Earth2Workflow):
    """FCN3 ensemble inference workflow for Foundry using ECMWF IFS initial conditions."""

    name = "foundry_fcn3_workflow"
    description = "FCN3 ensemble workflow for Foundry"

    def __init__(
        self,
        device: str = "cuda",
        init_seed: int | None = None,
    ):
        super().__init__()

        self.device = torch.device(device)

        self.fcn3 = self.load_fcn3()
        self.rng = np.random.default_rng(init_seed)

        self.data = PlanetaryComputerECMWFOpenDataIFS(verbose=False, cache=False)

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | WorkflowParameters
    ) -> WorkflowParameters:
        """Validate request parameters and enforce FCN3 ensemble limits for ``n_steps`` and ``n_samples``."""
        validated = super().validate_parameters(parameters)
        if not 1 <= validated.n_steps <= _MAX_FORECAST_STEPS:
            raise ValueError(
                f"n_steps must be between 1 and {_MAX_FORECAST_STEPS}, "
                f"got {validated.n_steps}"
            )
        if not 1 <= validated.n_samples <= _MAX_ENSEMBLE_SAMPLES:
            raise ValueError(
                f"n_samples must be between 1 and {_MAX_ENSEMBLE_SAMPLES}, "
                f"got {validated.n_samples}"
            )
        return validated

    def load_fcn3(self) -> FCN3:
        """Load the default FCN3 package, move it to the workflow device, and set eval mode."""
        logger.info("Loading FCN3")
        package = FCN3.load_default_package()
        fcn3 = FCN3.load_model(package)
        fcn3.to(self.device)
        fcn3.eval()
        return fcn3

    def get_seeds(self, n_seeds: int) -> list[int]:
        """Sample ``n_seeds`` distinct integer RNG seeds for ensemble members."""
        seeds = self.rng.choice(2**32, size=n_seeds, replace=False)
        return [int(s) for s in seeds]

    def validate_start_time(self, start_time: datetime) -> None:
        """Require ``start_time`` to fall on a 6-hour boundary (FCN3 / IFS cadence)."""
        if (start_time - datetime(1900, 1, 1)).total_seconds() % 21600 != 0:
            raise ValueError(f"Start time needs to be 6-hour interval: {start_time}")

    def validate_samples(
        self, n_samples: int, seeds: Sequence[int] | None
    ) -> list[int]:
        """Return ensemble seeds of length ``n_samples``, generating them if omitted."""
        if seeds is None:
            seeds = self.get_seeds(n_samples)
        elif len(seeds) != n_samples:
            logger.warning(
                "Ignoring requested number of samples because it does not match number of seeds"
            )
        return list(seeds)

    def validate_variables(self, variables: Sequence[str] | None) -> np.ndarray:
        """Resolve output variables, defaulting to the model's variables and checking names."""
        if variables is None:
            variables = self.fcn3.variables
        else:
            unknown_variables = set(variables) - set(self.fcn3.variables)
            if len(unknown_variables):
                raise ValueError(f"Unknown variable(s) {', '.join(unknown_variables)}")
            variables = np.array(variables)
        return variables

    def setup_io(
        self, io: IOBackend, output_coords: CoordSystem, seeds: Sequence[int]
    ) -> None:
        """Define Zarr/NetCDF arrays, CRS metadata, and time encoding for ensemble outputs."""
        io.add_array(
            {k: v for k, v in output_coords.items() if k != "variable"},
            output_coords["variable"],
        )

        # Storing seeds separately makes it easier to filter with Titiler
        e_coords = {"ensemble": output_coords["ensemble"]}
        io.add_array(e_coords, "seed", data=torch.tensor(seeds))

        # Add CRS definition
        io.add_array({}, "crs")
        io.root["crs"].grid_mapping_name = "latitude_longitude"
        io.root["crs"].longitude_of_prime_meridian = 0.0
        io.root["crs"].semi_major_axis = 6378137.0
        io.root["crs"].inverse_flattening = 298.257223563

        for var in output_coords["variable"]:
            io.root[var].grid_mapping = "crs"

        # Set attributes for automatic parsing of dimensions
        io.root["ensemble"].standard_name = "realization"
        io.root["time"].standard_name = "time"
        io.root["time"].axis = "T"
        io.root["lat"].standard_name = "latitude"
        io.root["lat"].units = "degrees_north"
        io.root["lat"].axis = "Y"
        io.root["lon"].standard_name = "longitude"
        io.root["lon"].units = "degrees_east"
        io.root["lon"].axis = "X"

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

    def get_fcn3_input(self, time: datetime) -> tuple[torch.Tensor, CoordSystem]:
        """Fetch FCN3 input tensors and coordinates from Planetary Computer ECMWF IFS."""
        x, coords = fetch_data(
            self.data,
            time=to_time_array([time]),
            variable=self.fcn3.input_coords()["variable"],
            device=self.device,
        )
        return x, coords

    def __call__(
        self,
        io: IOBackend,
        start_time: datetime = datetime(2025, 1, 1),
        n_steps: int = 12,
        n_samples: int = 4,
        seeds: Sequence[int] | None = None,
        variables: Sequence[str] | None = ("t2m", "u10m", "v10m"),
        collection_id: str | None = None,
    ) -> None:
        self.validate_start_time(start_time)
        lead_times = np.array([np.timedelta64(i * 6, "h") for i in range(n_steps + 1)])
        seeds = self.validate_samples(n_samples, seeds)
        variables = self.validate_variables(variables)

        x_ori, coords_ori = self.get_fcn3_input(start_time)

        output_coords = CoordSystem(
            {
                "ensemble": np.arange(len(seeds)),
                # Combine 'time' and 'lead_time' into single dimension
                "time": to_time_array([start_time]) + lead_times,
                "variable": variables,
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(-180, 180, 1440, endpoint=False),
            }
        )
        self.setup_io(io, output_coords, seeds)

        logger.info("Starting inference")
        total_samples = len(seeds)
        n_steps += 1  # add 1 for step 0 (initial conditions)
        for sample, seed in enumerate(seeds):

            self.fcn3.set_rng(seed=seed)
            iterator = self.fcn3.create_iterator(x_ori.clone(), coords_ori.copy())
            for step, (x, coords) in enumerate(iterator):
                # Update progress for step within sample
                msg = (
                    f"Processing sample {sample + 1}/{total_samples} "
                    f"(seed={seed}), step {step + 1}/{len(lead_times)}"
                )
                progress = WorkflowProgress(
                    progress=msg,
                    current_step=step + 1,
                    total_steps=n_steps,
                )
                self.update_progress(progress)
                logger.info(msg)

                # Select variables
                x_out, coords_out = map_coords(
                    x, coords, CoordSystem({"variable": output_coords["variable"]})
                )
                # Roll longitudes (for raster visualization)
                x_out = torch.roll(x_out, 720, dims=-1)
                coords_out["lon"] = np.linspace(-180, 180, 1440, endpoint=False)
                # Add ensemble dimension
                x_out = x_out.unsqueeze(0)
                coords_out["ensemble"] = np.array([sample])
                coords_out.move_to_end("ensemble", last=False)
                # Combine time and lead_time
                lead_time_dim = list(coords_out).index("lead_time")
                x_out = x_out.squeeze(lead_time_dim)
                coords_out["time"] = coords_out["time"] + coords_out["lead_time"]
                del coords_out["lead_time"]
                # Write to disk
                io.write(*split_coords(x_out, coords_out))

                if step == (n_steps - 1):
                    break
