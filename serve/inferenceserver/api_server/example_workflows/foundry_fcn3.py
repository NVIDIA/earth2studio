import logging
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import torch
import zarr

from api_server.workflow import Earth2Workflow, WorkflowProgress, workflow_registry
from earth2studio.data import PlanetaryComputerECMWFOpenDataIFS, fetch_data
from earth2studio.io import IOBackend, NetCDF4Backend, ZarrBackend
from earth2studio.models.px import FCN3
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("foundry_fcn3_workflow")


@workflow_registry.register
class FoundryFCN3Workflow(Earth2Workflow):
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

    def load_fcn3(self) -> FCN3:
        logger.info("Loading FCN3")
        package = FCN3.load_default_package()
        fcn3 = FCN3.load_model(package)
        fcn3.to(self.device)
        fcn3.eval()
        return fcn3

    def get_seeds(self, n_seeds: int) -> list[int]:
        seeds = self.rng.choice(2**32, size=n_seeds, replace=False)
        return [int(s) for s in seeds]

    def validate_start_time(self, start_time: datetime) -> None:
        if (start_time - datetime(1900, 1, 1)).total_seconds() % 21600 != 0:
            raise ValueError(f"Start time needs to be 6-hour interval: {start_time}")

    def validate_samples(
        self, n_samples: int, seeds: Sequence[int] | None
    ) -> list[int]:
        if seeds is None:
            seeds = self.get_seeds(n_samples)
        elif len(seeds) != n_samples:
            logger.warning(
                "Ignoring requested number of samples because it does not match number of seeds"
            )
        return list(seeds)

    def validate_variables(self, variables: Sequence[str] | None) -> np.ndarray:
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

        if isinstance(io, ZarrBackend):
            zarr.consolidate_metadata(io.store)

        if isinstance(io, NetCDF4Backend):
            # Planetary Computer does not like the original time format
            ref_time = np.datetime_as_string(output_coords["time"][0], unit="s")
            io["time"].units = f"hours since {ref_time.replace('T', ' ')}"
            io["time"][:] = np.arange(len(io["time"])) * 6

        return io

    def get_fcn3_input(self, time: datetime) -> tuple[torch.Tensor, CoordSystem]:
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
        n_steps: int = 20,
        n_samples: int = 16,
        seeds: Sequence[int] | None = None,
        variables: Sequence[str] | None = None,
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
