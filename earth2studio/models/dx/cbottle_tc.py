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
from datetime import datetime, timedelta

import cftime
import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.lexicon import CBottleLexicon
from earth2studio.models.auto import Package
from earth2studio.models.auto.mixin import AutoModelMixin
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import to_time_array
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

HPX_LEVEL = 6
TC_HPX_LEVEL = 3

VARIABLES = np.array(list(CBottleLexicon.VOCAB.keys()))


@check_optional_dependencies()
class CBottleTCGuidance(torch.nn.Module, AutoModelMixin):
    """Climate in a Bottle tropical cyclone guidance diagnostic.
    This model for Climate in a Bottle (cBottle) allows users to provide an cyclone
    guidance map on a lat-lon grid and synthesis global climate realizations at that
    given time. The tropical cyclone guidance field is regridded to HPX Level 3, which
    is then used during the sampling process.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2505.06474v1
    - https://github.com/NVlabs/cBottle
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/cbottle

    Note
    ----
    This model provides the function :py:func:`CBottleTCGuidance.create_guidance_tensor`
    as a utility to create the input guidance tensor.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core Pytorch diffusion model
    classifier_model : torch.nn.Module
        Pytorch classifier model
    sst_ds : xr.Dataset
        Sea surface temperature xarray dataset
    lat_lon : bool, optional
        Lat/lon toggle, if true data source will return output on a 0.25 deg lat/lon
        grid. If false, the native nested HealPix grid will be returned, by default True
    sampler_steps : int, optional
        Number of diffusion steps, by default 18
    sigma_max : float, optional
        Noise amplitude used to generate latent variables, by default 200
    batch_size : int, optional
        Batch size to generate time samples at, consider adjusting based on hardware
        being used, by default 4
    seed : int, optional
        Random generator seed for latent variables, by default 0
    """

    output_variables = VARIABLES

    def __init__(
        self,
        core_model: torch.nn.Module,
        classifier_model: torch.nn.Module,
        sst_ds: xr.Dataset,
        lat_lon: bool = True,
        sampler_steps: int = 18,
        sigma_max: float = 200.0,
        batch_size: int = 4,
        seed: int | None = None,
    ):
        super().__init__()

        self.sst = sst_ds
        self.lat_lon = lat_lon
        self.sigma_max = sigma_max
        self.sampler_steps = sampler_steps
        self.batch_size = batch_size
        self.seed = seed
        self._core_model = core_model
        self._class_model = classifier_model
        self.core_model = CBottle3d(core_model, separate_classifier=classifier_model)

        # Set up SST Lat Lon to HPX regridder
        lon_center = self.sst.lon.values
        # need to workaround bug where earth2grid fails to interpolate in circular manner
        # if lon[0] > 0
        # hack: rotate both src and target grids by the same amount so that src_lon[0] == 0
        # See https://github.com/NVlabs/earth2grid/issues/21
        src_lon = lon_center - lon_center[0]
        target_lon = (self.core_model.output_grid.lon - lon_center[0]) % 360
        grid = earth2grid.latlon.LatLonGrid(self.sst.lat.values, src_lon)
        self.sst_regridder = grid.get_bilinear_regridder_to(
            self.core_model.output_grid.lat, lon=target_lon
        )

        self.register_buffer("lat_grid", torch.tensor(np.linspace(90, -90, 721)))
        self.register_buffer(
            "lon_grid", torch.tensor(np.linspace(0, 360, 1440, endpoint=False))
        )
        # Empty tensor just to make tracking current device easier
        self.register_buffer("device_buffer", torch.empty(0))

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model

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
                    "variable": np.array(["tc_guidance"]),
                    "lat": self.lat_grid.cpu().numpy(),
                    "lon": self.lon_grid.cpu().numpy(),
                }
            )
        else:
            return OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": np.array(["tc_guidance"]),
                    "hpx": np.arange(4**TC_HPX_LEVEL * 12),
                }
            )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.

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

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(self.output_variables)

        if self.lat_lon:
            handshake_dim(input_coords, "lon", -1)
            handshake_dim(input_coords, "lat", -2)
            handshake_coords(input_coords, target_input_coords, "lon")
            handshake_coords(input_coords, target_input_coords, "lat")
        else:
            handshake_dim(input_coords, "hpx", -1)
            handshake_coords(input_coords, target_input_coords, "hpx")
            output_coords["hpx"] = np.arange(4**HPX_LEVEL * 12)

        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained cBottle model package from Nvidia model registry"""
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
        sigma_max: float = 200,
        seed: int = 0,
    ) -> DiagnosticModel:
        """Load AI datasource from package

        Parameters
        ----------
        package : Package
            CBottle AI model package
        lat_lon : bool, optional
            Lat/lon toggle, if true prognostic input/output on a 0.25 deg lat/lon
            grid. If false, the native nested HealPix grid will be returned, by default
            True
        input_variables: list[str] | VariableArray
            List of input variables that will be provided for conditioning the output
            generation, by default ["u10m", "v10m"]
        sampler_steps : int, optional
            Number of diffusion steps, by default 18
        sigma_max : float, optional
            Noise amplitude used to generate latent variables, by default 80
        seed : int, optional
            Random generator seed for latent variables, by default 0

        Returns
        -------
        DiagnosticModel
            Diagnostic model
        """
        checkpoints = [
            package.resolve("cBottle-3d/training-state-000512000.checkpoint"),
            package.resolve("cBottle-3d/training-state-002048000.checkpoint"),
            package.resolve("cBottle-3d/training-state-009856000.checkpoint"),
        ]
        # https://github.com/NVlabs/cBottle/blob/4f44c125398896fad1f4c9df3d80dc845758befa/src/cbottle/inference.py#L106
        core_model = MixtureOfExpertsDenoiser.from_pretrained(
            checkpoints, (100.0, 10.0)
        )

        classifier_model = None
        with Checkpoint(
            package.resolve("cBottle-3d-tc/training-state-002176000.checkpoint")
        ) as c:
            classifier_model = c.read_model().eval()

        sst_ds = xr.open_dataset(
            package.resolve("amip_midmonth_sst.nc"),
            engine="h5netcdf",
            storage_options=None,
            cache=False,
        ).load()

        return cls(
            core_model,
            classifier_model,
            sst_ds,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
            seed=seed,
        )

    @staticmethod
    def create_guidance_tensor(
        lat_coords: torch.Tensor,
        lon_coords: torch.Tensor,
        times: list[datetime] | TimeArray,
    ) -> tuple[torch.Tensor, OrderedDict]:
        """Creates a TC guidance tensor from lat/lon coordinates.

        Parameters
        ----------
        lat_coords : torch.Tensor
            Latitude coordinates where TC guidance should be set
        lon_coords : torch.Tensor
            Longitude coordinates where TC guidance should be set
        times: list[datetime] | TimeArray
            List of datetime objects or numpy datetime64 array specifying the times for
            the guidance tensor's coordinate system

        Returns
        -------
        tuple[torch.Tensor, OrderedDict]
            Tuple containing:
            - Guidance tensor with shape (1,n,1,721,1440) where values are 1 at the
                specified coordinates
            - OrderedDict with coordinate dimensions and values
        """
        times = to_time_array(times)
        # Convert any longitudes in -180 to 180 range to 0 to 360 range
        lon_coords = torch.where(lon_coords < 0, lon_coords + 360, lon_coords)

        lat_grid = np.linspace(90, -90, 721)
        lon_grid = np.linspace(0, 360, 1440, endpoint=False)
        guidance = torch.full((times.shape[0], 1, 1, 721, 1440), 0).float()

        lat_idx = torch.searchsorted(torch.from_numpy(-lat_grid), -lat_coords)
        lon_idx = torch.searchsorted(torch.from_numpy(lon_grid), lon_coords)
        guidance[:, :, :, lat_idx, lon_idx] = 1

        coords = OrderedDict(
            {
                "time": times,
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["tc_guidance"]),
                "lat": lat_grid,
                "lon": lon_grid,
            }
        )

        return guidance, coords

    def _latlon_guidance_to_hpx(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates HPX guidance pixel IDs if model has lat/lon grid inputs

        Parameters
        ----------
        x : torch.Tensor
            Input lat/lon array of tc guidance, where non-zero indicates a region to
            guide a topical cyclone. Dimensions [batch, 1, lat, lon] or [batch, 1, hpx]

        Returns
        -------
        torch.Tensor
            guidance pixel array for the classifier model
        """
        if not self.lat_lon:
            return x.unsqueeze(-2)

        guidance_data = torch.full(
            (x.shape[0], 1, 1, *self.core_model.classifier_grid.shape),
            torch.nan,
            device=x.device,
        )

        for batch in range(x.shape[0]):
            idx = torch.nonzero(x[batch, 0])
            idx = self.core_model.classifier_grid.ang2pix(
                self.lon_grid[idx[:, 1]], self.lat_grid[idx[:, 0]]
            )
            guidance_data[batch, :, :, idx] = 1

        return guidance_data

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)

        n_batch = x.shape[0]
        times = output_coords["time"][:, None]
        leads = output_coords["lead_time"][None, :]
        times = n_batch * [pd.to_datetime(t) for t in (times + leads).reshape(-1)]

        domain_shape = list(x.shape)[3:]
        x = x.reshape(-1, *domain_shape)

        input = self.get_cbottle_input(times)

        device = self.device_buffer.device
        condition = input["condition"].to(device)
        labels = input["labels"].to(device)
        images = input["target"].to(device)
        second_of_day = input["second_of_day"].to(device)
        day_of_year = input["day_of_year"].to(device)

        self.core_model.sigma_max = torch.Tensor([self.sigma_max]).to(device)
        self.core_model.num_steps = self.sampler_steps
        # Process in batches with progress bar if verbose is enabled
        n_samples = len(times)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        outputs = []
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)

            # Get batch slices
            batch = {}
            batch["target"] = images[start_idx:end_idx]
            batch["labels"] = labels[start_idx:end_idx]
            batch["condition"] = condition[start_idx:end_idx]
            batch["second_of_day"] = second_of_day[start_idx:end_idx]
            batch["day_of_year"] = day_of_year[start_idx:end_idx]

            indices_where_tc = self._latlon_guidance_to_hpx(x[start_idx:end_idx])
            output, cb_coords = self.core_model.sample(
                batch, guidance_pixels=indices_where_tc, seed=self.seed
            )
            outputs.append(output)

        # Concatenate all batches
        output = torch.cat(outputs, dim=0)
        if self.lat_lon:
            # Convert back into lat lon
            nlat, nlon = 721, 1440
            latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
                nlat, nlon, includes_south_pole=True
            )
            regridder = earth2grid.get_regridder(cb_coords.grid, latlon_grid).to(device)
            output = regridder(output).squeeze(2)

            output = output.reshape(
                output_coords["batch"].shape[0],
                output_coords["time"].shape[0],
                output_coords["lead_time"].shape[0],
                output_coords["variable"].shape[0],
                output_coords["lat"].shape[0],
                output_coords["lon"].shape[0],
            )
        else:
            output = output.reshape(
                output_coords["batch"].shape[0],
                output_coords["time"].shape[0],
                output_coords["lead_time"].shape[0],
                output_coords["variable"].shape[0],
                output_coords["hpx"].shape[0],
            )

        return output, output_coords

    def get_cbottle_input(
        self,
        times: list[datetime],
        n_batch: int = 1,
        label: int = 1,  # 0 for ICON, 1 for ERA5
    ) -> dict[str, torch.Tensor]:
        """Prepares the CBottle inputs

        Adopted from:

        - https://github.com/NVlabs/cBottle/blob/ed96dfe35d87ecefa4846307807e8241c4b24e71/src/cbottle/datasets/amip_sst_loader.py#L55
        - https://github.com/NVlabs/cBottle/blob/ed96dfe35d87ecefa4846307807e8241c4b24e71/src/cbottle/datasets/dataset_3d.py#L247

        Parameters
        ----------
        time : list[datetime]
            List of times for inference
        label : int, optional
            Label ID, by default 1

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of input tensors for CBottle
        """
        self._validate_sst_time(times)
        time_arr = np.array(times, dtype="datetime64[ns]")
        sst_data = torch.from_numpy(
            self.sst["tosbcs"].interp(time=time_arr, method="linear").values + 273.15
        ).to(self.device_buffer.device)
        sst_data = self.sst_regridder(sst_data)

        cond = encode_sst(sst_data.cpu())

        def reorder(x: torch.Tensor) -> torch.Tensor:
            x = torch.as_tensor(x)
            x = earth2grid.healpix.reorder(
                x, earth2grid.healpix.PixelOrder.NEST, earth2grid.healpix.HEALPIX_PAD_XY
            )
            return torch.permute(x, (2, 0, 1, 3))

        times = [
            cftime.DatetimeGregorian(t.year, t.month, t.day, t.hour, t.minute, t.second)
            for t in times
        ]
        day_start = np.array([t.replace(hour=0, minute=0, second=0) for t in times])
        year_start = np.array([d.replace(month=1, day=1) for d in day_start])
        second_of_day = (times - day_start) / timedelta(seconds=1)
        day_of_year = (times - year_start) / timedelta(seconds=86400)

        # ["rlut", "rsut", "rsds"]
        nan_channels = [38, 39, 42]
        target = np.zeros(
            (len(times), self.output_variables.shape[0], 1, 4**HPX_LEVEL * 12),
            dtype=np.float32,
        )
        target[:, nan_channels, ...] = np.nan

        labels = torch.nn.functional.one_hot(torch.tensor(label), num_classes=1024)
        labels = labels.unsqueeze(0).repeat(len(times), 1)

        out = {
            "target": torch.tensor(target),
            "labels": labels,
            "condition": reorder(cond),
            "second_of_day": torch.tensor(second_of_day.astype(np.float32)).unsqueeze(
                1
            ),
            "day_of_year": torch.tensor(day_of_year.astype(np.float32)).unsqueeze(1),
        }
        return out

    def _validate_sst_time(self, times: list[datetime]) -> None:
        """Verify if date time is valid for use with the default AMIP mid-month SST data

        Parameters
        ----------
        times : list[datetime]
            list of date times of input data
        """
        for time in times:
            if time < datetime(year=1940, month=1, day=1):
                raise ValueError(
                    f"Input data at {time} needs to be after January 1st, 1940 for CBottle infill if no input SST fields are provided"
                )

            if time >= datetime(year=2022, month=12, day=16, hour=12):
                raise ValueError(
                    f"Input data at {time} needs to be before December 16th, 2022 for CBottle infill if no input SST fields are provided"
                )
