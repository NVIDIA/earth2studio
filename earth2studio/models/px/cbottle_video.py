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
from collections.abc import Generator, Iterator
from datetime import datetime
from enum import IntEnum, StrEnum

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.lexicon import CBottleLexicon
from earth2studio.models.auto import Package
from earth2studio.models.auto.mixin import AutoModelMixin
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem, TimeArray

try:
    import earth2grid
    from cbottle.checkpointing import Checkpoint
    from cbottle.datasets.dataset_2d import encode_sst
    from cbottle.inference import CBottle3d, MixtureOfExpertsDenoiser
except ImportError:
    OptionalDependencyFailure("cbottle")
    earth2grid = None
    Checkpoint = None
    TimeUnit = None
    encode_sst = None
    CBottle3d = None
    MixtureOfExpertsDenoiser = None

HPX_LEVEL = 6


class DatasetModality(IntEnum):
    """Dataset label"""

    ICON = 0
    ERA5 = 1


class TimeStepperFunction(StrEnum):
    """Supported time-stepper functions"""

    HEUN = "heun"
    EULER = "euler"


@check_optional_dependencies()
class CBottleVideo(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Climate in a bottle video prognostic
    Climate in a Bottle (cBottle) is an AI model for emulating global km-scale climate
    simulations and reanalysis on the equal-area HEALPix grid. The cBottle video
    prognostic model uses the video diffusion checkpoint of CBottle trained to predict
    12 frames (initial state including) at a time.

    Note
    ----
    This wrapper allows users to provide an input condition for the first frame of the
    model. If this tensor is all NaNs no variable conditioning will be used running the
    network outside of time-stamp and respective SST.

    Warning
    -------
    Default model package has SST data from January 1940 to December 2022, expanded SST
    data should be provided out of this range.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        import numpy as np
        import torch
        from datetime import datetime

        from earth2studio.models.px import CBottleVideo

        # Load the default model package and instantiate model
        package = CBottleVideo.load_default_package()
        model = CBottleVideo.load_model(package)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Prepare input coordinates
        time = np.array([datetime(2022, 1, 1)], dtype="datetime64[ns]")
        coords = model.input_coords()
        coords["time"] = time
        coords["batch"] = np.array([0])

        # Unconditional generation: create NaN tensor for unconditional sampling
        x = torch.full(
            (1, 1, 1, len(model.VARIABLES), 721, 1440),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

        # Run inference - generates 12 frames (0-66 hours in 6-hour steps)
        iterator = model.create_iterator(x, coords)
        for step, (output, output_coords) in enumerate(iterator):
            print(f"Step {step}: {output.shape}, lead_time: {output_coords['lead_time']}")
            if step >= 11:  # Get first 12 frames (0-66 hours)
                break

    Note
    ----
    For conditional generation using ERA5 data, use :py:class:`earth2studio.models.dx.CBottleInfill`
    to generate all required variables first, then condition CBottleVideo on the infilled state.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2505.06474v1
    - https://github.com/NVlabs/cBottle
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/cbottle

    Parameters
    ----------
    core_model : torch.nn.Module
        Core Pytorch model
    sst_ds : xr.Dataset
        Sea surface temperature xarray dataset
    lat_lon : bool, optional
        Lat/lon toggle, if true data source will return output on a 0.25 deg lat/lon
        grid. If false, the native nested HealPix grid will be returned, by default True
    sampler_steps : int, optional
        Number of diffusion steps, by default 18
    sigma_max : float, optional
        Maximum supported noise level during sampling, by default 1000
    sigma_min : float, optional
        Minimum supported noise level during sampling, by default 0.02
    seed : int | None, optional
        If set, will fix the seed of the random generator for latent variables, by
        default None
    dataset_modality: DatasetModality, optional
        Dataset modality label to use when sampling (0=ICON, 1=ERA5), by default
        DatasetModality.ERA5
    time_stepper : TimeStepperFunction, optional
        Sampler function used to denoise, by default TimeStepperFunction.HEUN
    """

    VARIABLES = np.array(list(CBottleLexicon.VOCAB.keys()))
    torch_compile = False

    def __init__(
        self,
        core_model: torch.nn.Module,
        sst_ds: xr.Dataset,
        lat_lon: bool = True,
        sampler_steps: int = 18,
        sigma_max: float = 1000.0,
        sigma_min: float = 0.02,
        seed: int | None = None,
        dataset_modality: DatasetModality = DatasetModality.ERA5,
        time_stepper: TimeStepperFunction = TimeStepperFunction.HEUN,
    ):
        super().__init__()

        self.sst = sst_ds
        self.lat_lon = lat_lon
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sampler_steps = sampler_steps
        self.time_stepper = time_stepper
        self.seed = seed
        self.dataset_modality = dataset_modality
        self._mixture_model = core_model
        self.core_model = CBottle3d(core_model)

        self._time_length = 12
        self._time_step = np.timedelta64(6, "h")
        # ["rlut", "rsut", "rsds"]
        self._nan_channels = [38, 39, 42]

        # Set up SST Lat Lon to HPX regridder
        target_grid = earth2grid.healpix.Grid(
            HPX_LEVEL, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        lon_center = self.sst.lon.values
        # need to workaround bug where earth2grid fails to interpolate in circular manner
        # if lon[0] > 0
        # hack: rotate both src and target grids by the same amount so that src_lon[0] == 0
        # See https://github.com/NVlabs/earth2grid/issues/21
        src_lon = lon_center - lon_center[0]
        target_lon = (target_grid.lon - lon_center[0]) % 360
        grid = earth2grid.latlon.LatLonGrid(self.sst.lat.values, src_lon)
        self.sst_regridder = grid.get_bilinear_regridder_to(
            target_grid.lat, lon=target_lon
        )

        nlat, nlon = 721, 1440
        latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
            nlat, nlon, includes_south_pole=True
        )
        self.condition_regridder = earth2grid.get_regridder(latlon_grid, target_grid)
        self.output_regridder = earth2grid.get_regridder(target_grid, latlon_grid)

        # Empty tensor just to make tracking current device easier
        self.register_buffer("device_buffer", torch.empty(0))

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        if self.lat_lon:
            return OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": np.array(self.VARIABLES),
                    "lat": np.linspace(90, -90, 721),
                    "lon": np.linspace(0, 360, 1440, endpoint=False),
                }
            )
        else:
            return OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": np.array(self.VARIABLES),
                    "hpx": np.arange(4**HPX_LEVEL * 12),
                }
            )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "variable", 3)
        handshake_dim(input_coords, "lead_time", 2)
        handshake_dim(input_coords, "time", 1)
        handshake_coords(input_coords, target_input_coords, "variable")

        if self.lat_lon:
            handshake_dim(input_coords, "lon", -1)
            handshake_dim(input_coords, "lat", -2)
            handshake_coords(input_coords, target_input_coords, "lon")
            handshake_coords(input_coords, target_input_coords, "lat")
        else:
            handshake_dim(input_coords, "hpx", -1)
            handshake_coords(input_coords, target_input_coords, "hpx")

        output_coords = input_coords.copy()
        output_coords["lead_time"] = input_coords["lead_time"] + np.array(
            [self._time_step]
        )
        return output_coords

    def _forward(self, x: torch.Tensor, times: TimeArray) -> torch.Tensor:
        """Executes forward sample of the model given conditional tensor and time array

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to condition video model. Of size [time,1,45,721,1440] if
            lat_lon or [time,1,45,49152] if healpix
        times : TimeArray
            Time stamp array of size [time]

        Returns
        -------
        torch.Tensor
            12 forecast steps (initial step including) [time, 12, 45, 721, 1440] if
            lat_lon or [time, 1, 45, 49152] if healpix
        """
        # Small check to make sure

        device = self.device_buffer.device
        self.core_model.sigma_min = self.sigma_min
        self.core_model.sigma_max = self.sigma_max
        self.core_model.num_steps = self.sampler_steps
        self.core_model.time_stepper = TimeStepperFunction(self.time_stepper).value

        if self.lat_lon:
            x = self.condition_regridder(x.double())

        # CBottle video expects [time, vars, time-step, hpx]
        x = x.transpose(1, 2)
        input_batch = self.get_cbottle_input(
            x, times, dataset_modality=self.dataset_modality, device=device
        )
        out, _ = self.core_model.sample(input_batch, seed=self.seed)
        # Regrid if needed
        if self.lat_lon:
            out = self.output_regridder(out.contiguous().double())
        # [time, vars, lead, ...] -> [time, lead, vars, ...]
        out = out.transpose(1, 2)

        return out

    def get_cbottle_input(
        self,
        conditions: torch.Tensor,
        times: TimeArray,
        dataset_modality: DatasetModality = DatasetModality.ERA5,
        device: torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Creates batch input for cbottle

        Parameters
        ----------
        conditions : torch.Tensor
            HPX conditional tensor for first time-step of size
            [time, 45, 1, 4**HPX_LEVEL*12]. If all NaNs no condition will be used.
        times : TimeArray
            Array of np.datetime64 time stamps to be samples of size [time], must have
            SST that can be sampled from self.sst
        dataset_modality : DatasetModality, optional
            Dataset modality label, by default DatasetModality.ERA5
        device : torch.device, optional
            Torch device, by default "cpu"

        Returns
        -------
        dict[str, torch.Tensor]
            Input batch dictionary used in the CBottle repo
        """
        # Known support range for SST
        for time in times:
            if time < np.datetime64("1940-01-01") or time >= np.datetime64(
                "2022-12-12T12:00"
            ):
                logger.warning(
                    f"Requeedst time {time} is outside of the default SST support range"
                )

        time_steps = [i * self._time_step for i in range(self._time_length)]
        times = times[:, None] + np.array(time_steps, dtype=np.timedelta64)[None, :]

        time_arr = np.array(times, dtype="datetime64[ns]").reshape(-1)
        sst_data = torch.from_numpy(
            self.sst["tosbcs"].interp(time=time_arr, method="linear").values + 273.15
        ).to(self.device_buffer.device)
        sst_data = self.sst_regridder(sst_data)

        # TODO: Fix on off device in efficiency here
        cond = torch.zeros(
            times.shape[0],
            47,
            times.shape[1],
            4**HPX_LEVEL * 12,
            dtype=torch.double,
            device=device,
        )
        cond[:, -2, :, :] = torch.tensor(
            encode_sst(sst_data.cpu()).reshape(times.shape[0], times.shape[1], -1),
            device=device,
        )
        cond[:, self._nan_channels, :, :] = torch.nan
        # If initial state to condition the model
        if (
            not torch.isnan(conditions).all()
            and self.core_model.batch_info.center
            and self.core_model.batch_info.scales
        ):
            means = (
                torch.tensor(self.core_model.batch_info.center)
                .to(device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            stds = (
                torch.tensor(self.core_model.batch_info.scales)
                .to(device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            conditions = (conditions.to(device) - means) / stds
            cond[:, :-2, :1, :] = conditions.to(device)
            cond[:, -1, :1, :] = 1  # Frame mask to 1

        def reorder(x: torch.Tensor) -> torch.Tensor:
            x = torch.as_tensor(x)
            x = earth2grid.healpix.reorder(
                x, earth2grid.healpix.PixelOrder.NEST, earth2grid.healpix.HEALPIX_PAD_XY
            )
            return x

        # Set up time tensors
        times0 = [
            datetime.fromtimestamp(t.astype("datetime64[s]").astype(int))
            for t in times.reshape(-1)
        ]
        second_of_day = np.array(
            [(t.hour * 3600) + (t.minute * 60) + t.second for t in times0]
        ).reshape(times.shape)
        day_of_year = np.array(
            [(t - datetime(t.year, 1, 1)).total_seconds() / (86400.0) for t in times0]
        ).reshape(times.shape)
        second_of_day = torch.tensor(second_of_day.astype(np.float32), device=device)
        day_of_year = torch.tensor(day_of_year.astype(np.float32), device=device)

        # Target tensor, not needed for video model since no infill
        # target = torch.zeros(
        #     (len(times), self.VARIABLES.shape[0], 1, 4**HPX_LEVEL * 12),
        #     dtype=torch.float32,
        #     device=device,
        # )
        # target[:, self._nan_channels, ...] = torch.nan
        # target = target.repeat(1, 1, self._time_length, 1)
        target = torch.empty(1, device=device)

        # Label tensor
        dataset_modality = DatasetModality(dataset_modality)
        labels = torch.nn.functional.one_hot(
            torch.tensor(dataset_modality.value, device=device), num_classes=1024
        )
        labels = labels.unsqueeze(0).repeat(len(times), 1)

        out = {
            "target": target,
            "labels": labels,
            "condition": reorder(cond),
            "second_of_day": second_of_day,
            "day_of_year": day_of_year,
        }
        return out

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained CBottle3D model package from Nvidia model registry"""
        return Package(
            "ngc://models/nvidia/earth-2/cbottle@1.2",
            cache_options={
                "cache_storage": Package.default_cache("cbottle"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        lat_lon: bool = True,
        sampler_steps: int = 18,
        sigma_max: float = 1000,
        seed: int | None = None,
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            CBottle AI model package
        lat_lon : bool, optional
            Lat/lon toggle, if true prognostic input/output on a 0.25 deg lat/lon
            grid. If false, the native nested HealPix grid will be returned, by default
            True
        sampler_steps : int, optional
            Number of diffusion steps, by default 18
        sigma_max : float, optional
            Noise amplitude used to generate latent variables, by default 200
        seed : int, optional
            Random generator seed for latent variables. If None, no seed will be used,
            by default None

        Returns
        -------
        PrognosticModel
            Prognostic Model
        """
        checkpoints = [
            package.resolve("cBottle-video.zip"),
        ]

        # https://github.com/NVlabs/cBottle/blob/4f44c125398896fad1f4c9df3d80dc845758befa/src/cbottle/inference.py#L810
        experts = []
        batch_info = None
        for path in checkpoints:
            with Checkpoint(path) as c:
                model = c.read_model().eval()
                experts.append(model)
                batch_info = c.read_batch_info()
        core_model = MixtureOfExpertsDenoiser(
            experts, sigma_thresholds=(), batch_info=batch_info
        )

        sst_ds = xr.open_dataset(
            package.resolve("amip_midmonth_sst.nc"),
            engine="netcdf4",
            cache=False,
        ).load()

        return cls(
            core_model,
            sst_ds,
            lat_lon=lat_lon,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
            seed=seed,
        )

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input conditional tensor for first frame, if all NaNs model will not use
            any conditioning.
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 6 hours in the future
        """

        output_coords = self.output_coords(coords)

        times = coords["time"].repeat(coords["batch"].shape[0])

        domain_shape = list(x.shape)[3:]  # Auto handle lat/lon vs healpix
        x = x.reshape(-1, coords["lead_time"].shape[0], *domain_shape)
        x = self._forward(x, times)
        x = x.reshape(
            coords["batch"].shape[0], coords["time"].shape[0], -1, *domain_shape
        )
        return x[:, :, 1:2], output_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        times = coords["time"].repeat(coords["batch"].shape[0])
        coords = self.output_coords(coords)
        domain_shape = list(x.shape)[3:]  # Auto handle lat/lon vs healpix
        start_frame = True
        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)
            x = x.reshape(-1, 1, *domain_shape)
            x = self._forward(x, times)
            x = x.reshape(
                coords["batch"].shape[0], coords["time"].shape[0], -1, *domain_shape
            )
            # Note that the input just conditions the model, so we need to run forward
            # even for the initial time step unlike the auto regressive models
            if start_frame:
                start_frame = False
                coords["lead_time"] = np.array([np.timedelta64(0)])
                output_tensor, coords_out = self.rear_hook(x[:, :, 0:1], coords)
                yield output_tensor, coords_out

            # Yield the 12 generated frames
            for i in range(1, self._time_length):
                coords["lead_time"] = coords["lead_time"] + np.array([self._time_step])
                # Rear hook
                output_tensor, coords_out = self.rear_hook(x[:, :, i : i + 1], coords)
                yield output_tensor, coords_out

            # Use last generated frame as the first input one (has not been formally verified for accuracy)
            times = times + 11 * np.array([self._time_step])
            x = x[:, :, -1:, ...]

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)
