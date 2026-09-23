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

from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime, timezone
from enum import IntEnum, StrEnum

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.lexicon import CBottleLexicon
from earth2studio.models.auto import Package
from earth2studio.models.auto.mixin import AutoModelMixin
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem, TimeArray

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
class CBottleVideo(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
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

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2505.06474v1
    - https://github.com/NVlabs/cBottle
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/cbottle
    - https://huggingface.co/nvidia/cbottle

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

    Badges
    ------
    region:global class:climate product:wind product:precip product:temp product:atmos
    product:solar year:2025 gpu:40gb
    provider:nvidia backend:pytorch
    """

    VARIABLES = np.array(list(CBottleLexicon.VOCAB.keys()))
    torch_compile = False
    front_hook_interval = 11

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

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of prognostic model

        Returns
        -------
        CoordinateSystem
            Allocation-free input coordinate signature
        """
        return coord_array(
            (
                "batch",
                "time",
                "lead_time",
                "variable",
                *(("lat", "lon") if self.lat_lon else ("hpx",)),
            ),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": self.VARIABLES,
            },
            dynamic=("batch", "time"),
            grid="latlon-0.25deg" if self.lat_lon else "healpix-l6-nested",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of prognostic model

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordinateSystem
            Allocation-free output coordinate signature
        """
        handshake_dataarray(input_coords, self.input_coords(), relative_lead_time=True)
        lead = input_coords.lead_time
        return coord_array_like(
            input_coords, {"lead_time": lead.values + self._time_step}
        )

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
            datetime.fromtimestamp(
                t.astype("datetime64[s]").astype(int), tz=timezone.utc
            )
            for t in times.reshape(-1)
        ]
        second_of_day = np.array(
            [(t.hour * 3600) + (t.minute * 60) + t.second for t in times0]
        ).reshape(times.shape)
        day_of_year = np.array(
            [
                (t - datetime(t.year, 1, 1, tzinfo=timezone.utc)).total_seconds()
                / (86400.0)
                for t in times0
            ]
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
            "hf://nvidia/cbottle@eebd93c85b3cd3a5a8f79c546ed917b0b80438f4",
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
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        # https://github.com/NVlabs/cBottle/blob/4f44c125398896fad1f4c9df3d80dc845758befa/src/cbottle/inference.py#L810
        experts = []
        batch_info = None
        for path in [package.resolve("cBottle-video.zip")]:
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
    def _advance(self, x: xr.DataArray) -> xr.DataArray:
        self.output_coords(x)
        times = np.tile(x.time.values + x.lead_time.values[-1], x.sizes["batch"])
        tensor = x.e2s.to_torch()[0].to(self.device_buffer.device).clone()
        domain = tensor.shape[3:]
        out = self._forward(tensor.reshape(-1, 1, *domain), times)
        out = out.reshape(x.sizes["batch"], x.sizes["time"], self._time_length, *domain)
        signature = coord_array_like(
            x,
            {
                "lead_time": x.lead_time.values[-1]
                + np.arange(1, self._time_length) * self._time_step
            },
        )
        return from_torch(out[:, :, 1:].clone(), signature)

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict the next six-hour field from labelled conditioning data."""
        return self._advance(x).isel(lead_time=slice(0, 1)).copy(deep=True)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the initial condition, then eleven forecasts per video advance."""
        handshake_dataarray(x, runtime=True)
        handshake_time(x)
        self.output_coords(x)
        state = x.copy(deep=True)
        yield state.copy(deep=True)
        while True:
            state = self.front_hook(state.copy(deep=True))
            frames = self._advance(state)
            for i in range(self._time_length - 1):
                frame = frames.isel(lead_time=slice(i, i + 1)).copy(deep=True)
                signature = coord_array_like(
                    state, {"lead_time": frame.lead_time.values}
                )
                # Rebuild from current hook metadata so deleted auxiliaries cannot
                # reappear from the cached video on the next yield.
                frame = xr.DataArray(
                    frame.data,
                    dims=signature.dims,
                    coords=deepcopy(signature.coords),
                    name=state.name,
                    attrs=deepcopy(state.attrs),
                )
                frame.encoding = deepcopy(state.encoding)
                state = self.rear_hook(frame).copy(deep=True)
                yield state.copy(deep=True)
