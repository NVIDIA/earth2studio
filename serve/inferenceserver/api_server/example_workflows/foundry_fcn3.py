import logging
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import torch
import zarr

from api_server.workflow import Earth2Workflow, workflow_registry
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

        # TODO: Do not use data cache
        self.data = PlanetaryComputerECMWFOpenDataIFS(verbose=False)

    def load_fcn3(self) -> FCN3:
        logger.info("Loading FCN3")
        # Assuming AZUREML_MODEL_DIR == EARTH2STUDIO_CACHE
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
        if isinstance(io, ZarrBackend):
            io.chunks = {
                "sample": 1,
                "time": 1,
                "variable": 1,
            }

        io.add_array(
            {k: v for k, v in output_coords.items() if k != "variable"},
            output_coords["variable"],
        )

        # Storing seeds separately makes it easier to filer with Titiler
        sample_coords = {"sample": output_coords["sample"]}
        io.add_array(sample_coords, "seed", data=torch.tensor(seeds))

        # Set attributes for automatic parsing of dimensions
        io.root["lat"].standard_name = "latitude"
        io.root["lat"].axis = "Y"
        io.root["lon"].standard_name = "longitude"
        io.root["lon"].axis = "X"
        # io.root["time"].standard_name = "time"
        # io.root["time"].axis = "T"

        # Planetary Computer does not like grid_mapping, so we skip adding it

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
                "sample": np.arange(len(seeds)),
                # Planetary Computer does not like separate 'lead_time'
                "time": to_time_array([start_time]) + lead_times,
                "variable": variables,
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        self.setup_io(io, output_coords, seeds)

        logger.info("Starting inference")
        for si, seed in enumerate(seeds):
            self.fcn3.set_rng(seed=seed)
            iterator = self.fcn3.create_iterator(x_ori.clone(), coords_ori.copy())
            for ii, (x, coords) in enumerate(iterator):
                logger.info(
                    "Processing FCN3 (seed=%d) step %d/%d",
                    seed,
                    ii + 1,
                    len(lead_times),
                )

                x_out, coords_out = map_coords(
                    x, coords, CoordSystem({"variable": output_coords["variable"]})
                )
                # Add sample dimension
                x_out = x_out.unsqueeze(0)
                coords_out["sample"] = np.array([si])
                coords_out.move_to_end("sample", last=False)
                # Combine time and lead_time
                lead_time_dim = list(coords_out).index("lead_time")
                x_out = x_out.squeeze(lead_time_dim)
                coords_out["time"] = coords_out["time"] + coords_out["lead_time"]
                del coords_out["lead_time"]
                # Write to disk
                io.write(*split_coords(x_out, coords_out))

                if ii == n_steps:
                    break
