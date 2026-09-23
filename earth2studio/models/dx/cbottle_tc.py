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

from datetime import datetime, timedelta
from enum import IntEnum

import cftime
import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.grids import HEALPixGrid, LatLonGrid
from earth2studio.lexicon import CBottleLexicon
from earth2studio.models.auto import Package
from earth2studio.models.auto.mixin import AutoModelMixin
from earth2studio.models.batch import batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_size,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import to_time_array
from earth2studio.utils.type import TimeArray

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


class DatasetModality(IntEnum):
    """Dataset label"""

    ICON = 0
    ERA5 = 1


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
    - https://huggingface.co/nvidia/cbottle

    Note
    ----
    This model provides the function :py:func:`model.create_guidance_tensor`
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
        Lat/lon toggle, if true the model will return output on a 0.25 deg lat/lon
        grid. If false, guidance uses flat XY HEALPix (north origin, clockwise)
        and generated fields use nested HEALPix, by default True
    sampler_steps : int, optional
        Number of diffusion steps, by default 18
    sigma_max : float, optional
        Noise amplitude used to generate latent variables, by default 200
    batch_size : int, optional
        Batch size to generate time samples at, consider adjusting based on hardware
        being used, by default 4
    seed : int, optional
        Random generator seed for latent variables. If None will use no seed, by default
        None
    dataset_modality: DatasetModality, optional
        Dataset modality label to use when sampling (0=ICON, 1=ERA5), by default
        DatasetModality.ERA5

    Badges
    ------
    region:global class:climate product:wind product:precip product:temp product:atmos
    year:2025 gpu:80gb
    provider:nvidia backend:pytorch
    """

    output_variables = VARIABLES
    guidance_scale = 0.005  # 0.03 = strong, 0 = no guidance

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
        dataset_modality: DatasetModality = DatasetModality.ERA5,
    ):
        super().__init__()

        self.sst = sst_ds
        self.lat_lon = lat_lon
        self.sigma_max = sigma_max
        self.sampler_steps = sampler_steps
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_modality = dataset_modality
        self._core_model = core_model
        self._class_model = classifier_model
        self.core_model = CBottle3d(
            core_model,
            sigma_max=sigma_max,
            num_steps=sampler_steps,
            separate_classifier=classifier_model,
        )

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

        self.register_buffer("lat_grid", torch.tensor(np.linspace(90, -90, 721)))
        self.register_buffer(
            "lon_grid", torch.tensor(np.linspace(0, 360, 1440, endpoint=False))
        )
        # Empty tensor just to make tracking current device easier
        self.register_buffer("device_buffer", torch.empty(0))

    def input_coords(self) -> xr.DataArray:
        """Input coordinate system of diagnostic model

        Returns
        -------
        xr.DataArray
            Allocation-free input coordinate signature
        """
        grid = (
            LatLonGrid(self.lat_grid.cpu().numpy(), self.lon_grid.cpu().numpy())
            if self.lat_lon
            else HEALPixGrid(
                TC_HPX_LEVEL,
                ordering="xy",
                layout="flat",
                xy_origin="north",
                xy_clockwise=True,
            )
        )
        return coord_array(
            ("batch", "time", "lead_time", "variable", *grid.dims),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": ["tc_guidance"],
            },
            dynamic=("batch", "time"),
            grid=grid,
        )

    def output_coords(self, input_coords: xr.DataArray) -> xr.DataArray:
        """Output coordinate system of diagnostic model

        Finite one-dimensional lead times are preserved. Each frame conditions
        the model at its initialization time plus its lead time.

        Parameters
        ----------
        input_coords : xr.DataArray
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.

        Returns
        -------
        xr.DataArray
            Allocation-free output coordinate signature
        """
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = input_coords.coords["lead_time"]
        # Each guidance frame is independent, conditioned at time + lead_time.
        signature = self.input_coords()
        handshake_dataarray(
            input_coords,
            coord_array_like(signature, {"lead_time": lead.values}),
        )
        if self.lat_lon:
            return coord_array_like(input_coords, {"variable": self.output_variables})
        leading = input_coords.dims[:-2]
        return coord_array(
            (*leading, "variable", "hpx"),
            {
                **{
                    k: v.variable
                    for k, v in input_coords.coords.items()
                    if set(v.dims).issubset(leading)
                },
                "variable": self.output_variables,
            },
            sizes={d: input_coords.sizes[d] for d in leading},
            dynamic=input_coords.attrs.get("earth2studio_dynamic_dims", ()),
            grid="healpix-l6-nested",
            dtype=input_coords.dtype,
            name=input_coords.name,
            attrs={
                k: v
                for k, v in input_coords.attrs.items()
                if k not in signature.attrs and k != "earth2studio_grid_id"
            },
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained cBottle model package from Nvidia model registry"""
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
        sigma_max: float = 200,
        seed: int | None = None,
        allow_second_order_derivatives: bool = False,
    ) -> DiagnosticModel:
        """Load diagnostic from package

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
            Noise amplitude used to generate latent variables, by default 80
        seed : int, optional
            Random generator seed for latent variables. If None, no seed will be used,
            by default None
        allow_second_order_derivatives : bool, optional
            Enable checkpoint/model loading path required for second-order autodiff
            (needed for odds-ratio computations). Keep False for faster standard
            guided inference, by default False.

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
            checkpoints,
            (100.0, 10.0),
            allow_second_order_derivatives=allow_second_order_derivatives,
        )

        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        classifier_model = None
        with Checkpoint(
            package.resolve("cBottle-3d-tc/training-state-002176000.checkpoint")
        ) as c:
            classifier_model = c.read_model(
                allow_second_order_derivatives=allow_second_order_derivatives,
            ).eval()

        sst_ds = xr.open_dataset(
            package.resolve("amip_midmonth_sst.nc"),
            engine="netcdf4",
            cache=False,
        ).load()

        return cls(
            core_model,
            classifier_model,
            sst_ds,
            lat_lon=lat_lon,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
            seed=seed,
        )

    def create_guidance_tensor(
        self,
        lat_coords: torch.Tensor,
        lon_coords: torch.Tensor,
        times: list[datetime] | TimeArray,
    ) -> xr.DataArray:
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
        xr.DataArray
            Labelled guidance with time, lead_time, variable and spatial dimensions;
            values are one at the specified coordinates and NaN elsewhere.
        """
        times = to_time_array(times)
        device = self.device_buffer.device
        guidance = None

        lat_coords = lat_coords.to(device)
        lon_coords = lon_coords.to(device)

        if self.lat_lon:
            # Convert any longitudes in -180 to 180 range to 0 to 360 range
            lon_coords = torch.where(lon_coords < 0, lon_coords + 360, lon_coords)

            lat_grid = np.linspace(90, -90, 721)
            lon_grid = np.linspace(0, 360, 1440, endpoint=False)
            guidance = torch.full(
                (times.shape[0], 1, 1, 721, 1440), torch.nan, device=device
            ).float()

            lat_idx = torch.searchsorted(
                -torch.tensor(lat_grid).to(device), -lat_coords
            )
            lon_idx = torch.searchsorted(torch.tensor(lon_grid).to(device), lon_coords)
            guidance[:, :, :, lat_idx, lon_idx] = 1

        else:
            guidance = torch.full(
                (times.shape[0], 1, 1, *self.core_model.classifier_grid.shape),
                torch.nan,
                device=device,
            )
            idx = self.core_model.classifier_grid.ang2pix(lon_coords, lat_coords)
            guidance[:, :, :, idx] = 1

        signature = coord_array_like(self.input_coords(), {"batch": [0], "time": times})
        signature = signature.isel(batch=0, drop=True)
        return from_torch(guidance, signature)

    def _prepare_guidance_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Preparies HPX guidance tensor for model. If inputs are lat lon, will convert
        to HPX, otherwise just expanded required dims for model inference

        Parameters
        ----------
        x : torch.Tensor
            Input lat/lon array of tc guidance, where non-nan indicates a region to
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
            idx = torch.nonzero(~torch.isnan(x[batch, 0]))
            idx = self.core_model.classifier_grid.ang2pix(
                self.lon_grid[idx[:, 1]], self.lat_grid[idx[:, 0]]
            )
            guidance_data[batch, :, :, idx] = 1

        return guidance_data

    @batch_func()
    def __call__(
        self,
        x: xr.DataArray,
    ) -> xr.DataArray:
        """Generate labelled fields from cyclone guidance."""
        output_coords = self.output_coords(x)
        x = x.e2s.to_torch()[0].to(self.device_buffer.device).clone()

        n_batch = x.shape[0]
        times = output_coords["time"].values[:, None]
        leads = output_coords["lead_time"].values[None, :]
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

        self.core_model.sigma_max = float(self.sigma_max)
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

            indices_where_tc = self._prepare_guidance_tensor(x[start_idx:end_idx])
            output, cb_coords = self.core_model.sample(
                batch,
                guidance_pixels=indices_where_tc,
                seed=self.seed,
                guidance_scale=self.guidance_scale,
            )

            # If ICON, translate
            if DatasetModality(self.dataset_modality) == DatasetModality.ICON:
                output = self.core_model._normalize(output)
                output = self.core_model._reorder(output)
                batch["target"] = output
                output, _ = self.core_model.translate(batch, dataset="icon")

            outputs.append(output)

        # Concatenate all batches
        output = torch.cat(outputs, dim=0)

        if self.lat_lon:
            # Convert back into lat lon
            output = self.regrid_hpx_to_latlon(output, grid=cb_coords.grid).squeeze(2)

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

        return from_torch(output, output_coords)

    def calculate_odds_ratio(
        self,
        x: xr.DataArray,
        guidance_scale: float = 128,
        compute_forward_divergences: bool = False,
    ) -> tuple[float | torch.Tensor, xr.DataArray]:
        """Compute classifier-guided log-odds ratio for one guidance sample.

        Parameters
        ----------
        x : xr.DataArray
            Labelled guidance with the same layout expected by :meth:`__call__`.
        guidance_scale : float, optional
            Guidance scale forwarded to cBottle odds-ratio evaluation. Defaults to 128.
        compute_forward_divergences : bool, optional
            If True, compute forward-phase Hutchinson divergence terms. These are
            not required for ``log_odds_ratio`` and increase runtime; by default False.

        Returns
        -------
        tuple[float | torch.Tensor, xr.DataArray]
            Log-odds ratio and labelled forward latents on the output grid.

        Note
        ----
        This method requires the underlying cBottle model to be loaded with
        ``allow_second_order_derivatives=True`` via :meth:`load_model`. The default
        ``allow_second_order_derivatives=False`` path is optimized for standard guided
        sampling and fails for odds-ratio computations.
        """

        handshake_nonempty(x)
        handshake_time(x)
        for dim in x.dims[: -(3 if self.lat_lon else 2)]:
            handshake_size(x, dim, 1)
        output_coords = self.output_coords(x)
        times = output_coords["time"].values[:, None]
        leads = output_coords["lead_time"].values[None, :]
        sample_times = [pd.to_datetime(t) for t in (times + leads).reshape(-1)]
        spatial_rank = 2 if self.lat_lon else 1
        tensor = x.e2s.to_torch()[0].to(self.device_buffer.device).clone()
        x = tensor.reshape(-1, 1, *tensor.shape[-spatial_rank:])
        times = sample_times * (x.shape[0] // len(sample_times))

        cb_input = self.get_cbottle_input(times)
        device = self.device_buffer.device
        batch = {
            "target": cb_input["target"].to(device),
            "labels": cb_input["labels"].to(device),
            "condition": cb_input["condition"].to(device),
            "second_of_day": cb_input["second_of_day"].to(device),
            "day_of_year": cb_input["day_of_year"].to(device),
        }

        guidance_hpx = self._prepare_guidance_tensor(x.to(device))
        guidance_pixels = torch.nonzero(
            ~torch.isnan(guidance_hpx[0, 0, 0]), as_tuple=False
        ).squeeze(-1)
        if guidance_pixels.numel() == 0:
            raise ValueError("No guidance pixels set (all guidance entries are NaN).")

        self.core_model.sigma_max = float(self.sigma_max)
        log_odds_ratio, forward_latents = self.core_model.calculate_odds_ratio(
            batch=batch,
            guidance_pixels=guidance_pixels,
            guidance_scale=guidance_scale,
            compute_forward_divergences=compute_forward_divergences,
            num_steps=self.sampler_steps,
        )  # forward_latents shape: (1, 45, 1, 49152)

        if self.lat_lon:
            domain_grid = getattr(
                self.core_model.net.domain, "_grid", self.core_model.net.domain
            )
            forward_latents = self.regrid_hpx_to_latlon(
                forward_latents, domain_grid
            ).squeeze(2)
        else:
            domain_grid = getattr(
                self.core_model.net.domain, "_grid", self.core_model.net.domain
            )
            forward_latents = domain_grid.reorder(
                earth2grid.healpix.PixelOrder.NEST, forward_latents
            )
            forward_latents = forward_latents.squeeze(2)

        return log_odds_ratio, from_torch(
            forward_latents.reshape(output_coords.shape), output_coords
        )

    def regrid_hpx_to_latlon(
        self,
        x: torch.Tensor,
        grid: "earth2grid.base.Grid",
    ) -> torch.Tensor:
        """Regrid an HPX tensor to lat/lon grid.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose last dimension is HPX pixels.
        grid : earth2grid.base.Grid
            Input HPX grid

        Returns
        -------
        torch.Tensor
            Regridded tensor with trailing dimensions ``(..., nlat, nlon)``.
        """
        latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
            721, 1440, includes_south_pole=True
        )
        regridder = earth2grid.get_regridder(grid, latlon_grid).to(
            self.device_buffer.device
        )
        return regridder(x)

    def get_cbottle_input(
        self,
        times: list[datetime],
        dataset_modality: DatasetModality = DatasetModality.ERA5,
    ) -> dict[str, torch.Tensor]:
        """Prepares the CBottle inputs

        Adopted from:

        - https://github.com/NVlabs/cBottle/blob/ed96dfe35d87ecefa4846307807e8241c4b24e71/src/cbottle/datasets/amip_sst_loader.py#L55
        - https://github.com/NVlabs/cBottle/blob/ed96dfe35d87ecefa4846307807e8241c4b24e71/src/cbottle/datasets/dataset_3d.py#L247

        Parameters
        ----------
        time : list[datetime]
            List of times for inference
        dataset_modality : DatasetModality, optional
            Dataset modality label, by default DatasetModality.ERA5

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of input tensors for CBottle
        """
        self._validate_sst_time(times)

        device = self.device_buffer.device
        time_arr = np.array(times, dtype="datetime64[ns]")
        sst_data = torch.from_numpy(
            self.sst["tosbcs"].interp(time=time_arr, method="linear").values + 273.15
        ).to(device)
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

        dataset_modality = DatasetModality(dataset_modality)
        labels = torch.nn.functional.one_hot(
            torch.tensor(dataset_modality.value, device=device), num_classes=1024
        )
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
        handshake_time(
            {"time": np.asarray(times, dtype="datetime64[us]")},
            minimum=np.datetime64("1940-01-01"),
            maximum=np.datetime64("2022-12-16T12:00"),
        )
