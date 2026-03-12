import logging
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
import zarr
from api_server.workflow import Earth2Workflow, WorkflowProgress, workflow_registry

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
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array

logger = logging.getLogger("foundry_fcn3_stormscope_goes_workflow")
logger.setLevel(logging.INFO)

GOES_MODEL_NAME = "6km_60min_natten_cos_zenith_input_eoe_v2"


@workflow_registry.register
class FoundryFCN3StormScopeGOESWorkflow(Earth2Workflow):
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
            coords_out["lat"], coords_out["lon"]
        )

    def load_fcn3_interp(self) -> InterpModAFNO:
        logger.info("Loading FCN3")
        package = FCN3.load_default_package()
        fcn3 = FCN3.load_model(package)

        # Surface pressure interpolation
        orography_fn = package.resolve("orography.nc")
        with xr.open_dataset(orography_fn) as ds:
            z_surface = torch.as_tensor(ds["Z"][0].values)
        z_surf_coords = OrderedDict({d: fcn3.input_coords()[d] for d in ["lat", "lon"]})
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
        fcn3_interp.to(device=self.device)
        fcn3_interp.eval()
        return fcn3_interp

    def load_stormscope(self) -> StormScopeGOES:
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
        seeds = self.rng.choice(2**32, size=n_seeds, replace=False)
        return [int(s) for s in seeds]

    def validate_start_times(
        self, time_stormscope: datetime, time_fcn3: datetime
    ) -> None:
        ref = datetime(1900, 1, 1)
        if (time_stormscope - ref).total_seconds() % (1 * 60 * 60) != 0:
            raise ValueError(
                f"Start time for StormScope must be 6-hour interval: {time_stormscope}"
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
        if not seeds:
            return self.get_seeds(n_samples)
        if len(seeds) != n_samples:
            logger.warning(
                "Ignoring requested number of samples because it does not match number of seeds"
            )
        return list(seeds)

    def validate_variables(self, variables: Sequence[str] | None) -> np.ndarray:
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

        if isinstance(io, ZarrBackend):
            zarr.consolidate_metadata(io.store)

        if isinstance(io, NetCDF4Backend):
            # Planetary Computer does not like the original time format
            ref_time = np.datetime_as_string(output_coords["time"][0], unit="s")
            io["time"].units = f"hours since {ref_time.replace('T', ' ')}"
            io["time"][:] = np.arange(len(io["time"]))

        return io

    def get_fcn3_input(self, time: datetime) -> tuple[torch.Tensor, CoordSystem]:
        x, coords = fetch_data(
            self.data_fcn3,
            time=to_time_array([time]),
            variable=self.fcn3_interp.input_coords()["variable"],
            device=self.device,
        )
        return x, coords

    def get_stormscope_input(self, time: datetime) -> tuple[torch.Tensor, CoordSystem]:
        coords_in = self.stormscope.input_coords()
        if time < datetime(2025, 4, 7):
            data = self.data_stormscope["goes16"]
        else:
            data = self.data_stormscope["goes19"]
        x, coords = fetch_data(
            data,
            time=to_time_array([time]),
            variable=coords_in["variable"],
            lead_time=coords_in["lead_time"],
            device=self.device,
        )

        batch_size = 1
        if x.dim() == 5:
            x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
            coords["batch"] = np.arange(batch_size)
            coords.move_to_end("batch", last=False)

        x, coords = self.stormscope.prep_input(x, coords)
        x = torch.where(self.stormscope.valid_mask, x, torch.nan)

        return x, coords

    def run_fcn3(
        self,
        io: IOBackend,
        x: torch.Tensor,
        coords_x: CoordSystem,
        seed_fcn3: int,
        start_time_stormscope: datetime,
        lead_times: np.ndarray,
        sample: int,
        total_samples: int,
    ) -> None:
        # Create z500 conditioning with FCN3
        coords_in = self.stormscope.input_coords()
        start_time_stormscope = to_time_array([start_time_stormscope])
        variables = self.stormscope.conditioning_variables
        # Start time and lead times are shifted to StormScope start time
        output_coords = {
            "time": start_time_stormscope,
            "lead_time": lead_times,
            "variable": variables,
            "y": coords_in["y"],
            "x": coords_in["x"],
        }
        io.add_array(
            {k: v for k, v in output_coords.items() if k != "variable"}, variables
        )

        model_gap = int(
            (start_time_stormscope - coords_x["time"]) / np.timedelta64(1, "h")
        )

        self.fcn3_interp.px_model.px_model.set_rng(seed=seed_fcn3)
        iterator = self.fcn3_interp.create_iterator(x.clone(), coords_x.copy())

        n_steps = model_gap + len(lead_times)
        for step, (x, coords_x) in enumerate(iterator):
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

            x, coords_x = map_coords(x, coords_x, OrderedDict({"variable": variables}))
            x, coords_x = self.stormscope.prep_input(x, coords_x, conditioning=True)
            coords_x["time"] = start_time_stormscope
            coords_x["lead_time"] = coords_x["lead_time"] - np.timedelta64(
                model_gap, "h"
            )
            io.write(*split_coords(x, coords_x))

            if step == (n_steps - 1):
                break

    def run_stormscope(
        self,
        io: IOBackend,
        y: torch.Tensor,
        coords_y: CoordSystem,
        seed_fcn3: int,
        seed_stormscope: int,
        lead_times: np.ndarray,
        variables: np.ndarray,
        sample: int,
        total_samples: int,
    ) -> None:
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
            y_pred: torch.Tensor, coords_pred: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            y_out, coords_out = map_coords(
                y_pred, coords_pred, CoordSystem({"variable": variables})
            )
            del coords_out["batch"]
            # Reuse batch dimension as ensemble dimension (squeeze/unsqueeze)
            coords_out["ensemble"] = np.array([sample])
            coords_out.move_to_end("ensemble", last=False)
            # Combine time and lead_time
            lead_time_dim = list(coords_out).index("lead_time")
            y_out = y_out.squeeze(lead_time_dim)
            coords_out["time"] = coords_out["time"] + coords_out["lead_time"]
            del coords_out["lead_time"]
            return y_out, coords_out

        # Update progress for step within sample
        log_progress(0)

        # Store initial GOES data (identical across seeds)
        y_out, coords_out = prep_output(y, coords_y)
        io.write(*split_coords(y_out, coords_out))

        # Cannot use seeded Generator before torch==2.10
        # Use self.stormscope.sampler_args["randn_like"] once updated
        torch.manual_seed(seed_stormscope)

        for step in range(1, n_steps):
            y_pred, coords_pred = self.stormscope(y, coords_y)

            # Update progress for step within sample
            log_progress(step)

            y_out, coords_out = prep_output(y_pred, coords_pred)
            io.write(*split_coords(y_out, coords_out))

            if step == (n_steps - 1):
                break

            y, coords_y = self.stormscope.next_input(y_pred, coords_pred, y, coords_y)

    def __call__(
        self,
        io: IOBackend,
        start_time_fcn3: datetime = datetime(2025, 1, 1, 18),
        start_time_stormscope: datetime = datetime(2025, 1, 1, 18),
        n_steps: int = 12,
        n_samples_fcn3: int = 4,
        n_samples_stormscope: int = 16,
        seeds_fcn3: Sequence[int] | None = None,
        seeds_stormscope: Sequence[int] | None = None,
        variables: Sequence[str] | None = None,
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

        x_ori, coords_x_ori = self.get_fcn3_input(start_time_fcn3)
        y_ori, coords_y_ori = self.get_stormscope_input(start_time_stormscope)

        coords_out = self.stormscope.output_coords(self.stormscope.input_coords())
        output_coords = {
            "ensemble": np.arange(len(seeds_stormscope)),
            # Planetary Computer does not like separate 'lead_time'
            "time": to_time_array([start_time_stormscope]) + lead_times,
            "variable": variables,
            "y": coords_out["y"],
            "x": coords_out["x"],
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
                x=x_ori.clone(),
                coords_x=coords_x_ori.copy(),
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
                    y=y_ori.clone(),
                    coords_y=coords_y_ori.copy(),
                    seed_fcn3=seed_fcn3,
                    seed_stormscope=seeds_stormscope[sample],
                    lead_times=lead_times,
                    variables=variables,
                    sample=sample,
                    total_samples=total_samples,
                )
                sample += 1
