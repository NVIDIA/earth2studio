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

import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils.coords import handshake_coords, handshake_dim

try:
    import earth2grid
    from cbottle.checkpointing import Checkpoint
    from cbottle.datasets.dataset_3d import get_mean, get_std
    from cbottle.denoiser_factories import DenoiserType, get_denoiser
    from cbottle.diffusion_samplers import edm_sampler_from_sigma
except ImportError:
    earth2grid = None
    Checkpoint = None
    edm_sampler_from_sigma = None
    get_denoiser = None
    get_mean = None
    get_std = None

from earth2studio.lexicon import CBottleLexicon
from earth2studio.models.auto import Package
from earth2studio.models.auto.mixin import AutoModelMixin
from earth2studio.utils.imports import check_extra_imports
from earth2studio.utils.type import CoordSystem, VariableArray

HPX_LEVEL = 6

VARIABLES = np.array(list(CBottleLexicon.VOCAB.keys()))


@check_extra_imports("cbottle", ["cbottle", "earth2grid"])
class CBottleInfill(torch.nn.Module, AutoModelMixin):
    """Climate in a bottle infill diagnostic
    Climate in a Bottle (cBottle) is an AI model for emulating global km-scale climate
    simulations and reanalysis on the equal-area HEALPix grid. The cBottle infill
    diagnostic enables users to generate all variables supported by cBottle from just
    a subset ontop of existing monthly average sea surface temperatures and solar
    conditioning. The cBottle infill model uses the a globally-trained coarse-resolution
    image diffusion model that generates 100km (50k-pixel) fields

    Note
    ----
    Unlike other diagnostics that have a fixed input, this diagnostic allows users to
    specify the input to be any subset of cBottle's output variables.
    Namely, the model can be adapted to the data present in the inference pipeline and
    will always generate all output fields based on the information provided.

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
    input_variables: list[str] | VariableArray
        List of input variables that will be provided for conditioning the output
        generation. Must be a subset of the cBottle output / supported variables.
        See cBottle lexicon for full list.
    sampler_steps : int, optional
        Number of diffusion steps, by default 18
    sigma_max : float, optional
        Noise amplitude used to generate latent variables, by default 80
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
        sst_ds: xr.Dataset,
        input_variables: list[str] | VariableArray,
        sampler_steps: int = 18,
        sigma_max: float = 80,
        seed: int = 0,
    ):
        super().__init__()

        self.core_model = core_model
        self.sst = sst_ds
        self.sigma_max = sigma_max
        self.sampler_steps = sampler_steps
        self.batch_size = 4
        self.rng = torch.Generator().manual_seed(seed)

        self.input_variables = input_variables

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

        # Set up regridder for input
        grid = earth2grid.latlon.LatLonGrid(
            self.input_coords()["lat"].tolist(), self.input_coords()["lon"].tolist()
        )
        self.input_regridder = grid.get_bilinear_regridder_to(
            self.core_model.domain._grid.lat, lon=self.core_model.domain._grid.lon
        )

        # Empty tensor just to make tracking current device easier
        self.register_buffer("device_buffer", torch.empty(0))
        self.register_buffer(
            "input_means", torch.Tensor(get_mean()[self.input_variable_idx, None])
        )
        self.register_buffer(
            "input_stds", torch.Tensor(get_std()[self.input_variable_idx, None])
        )
        self.register_buffer("output_means", torch.Tensor(get_mean()[:, None, None]))
        self.register_buffer("output_stds", torch.Tensor(get_std()[:, None, None]))
        # Set seed of random generator
        self.set_seed(seed=seed)

    @property
    def input_variables(self) -> VariableArray:
        """List of input variables expected for conditioning"""
        return self._input_variables

    @input_variables.setter
    def input_variables(self, value: list[str] | VariableArray) -> None:
        """Set input variables, validating they exist in VARIABLES

        Parameters
        ----------
        value : list[str] | VariableArray
            List of variable names to validate and set

        Raises
        ------
        ValueError
            If any variable name is not found in VARIABLES
        """
        for var in value:
            if var not in VARIABLES:
                raise ValueError(
                    f"Variable {var} not found in CBottle supported variables"
                )
        self._input_variables = np.array(value)

    @property
    def input_variable_idx(self) -> np.ndarray:
        """List of input variables expected for conditioning"""
        varidx = []
        for var in self.input_variables:
            idx = np.where(self.output_variables == var)[0]
            varidx.append(idx[0])
        return np.array(varidx)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.empty(0),
                "variable": np.array(self.input_variables),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
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
        handshake_dim(input_coords, "lon", -1)
        handshake_dim(input_coords, "lat", -2)
        handshake_dim(input_coords, "variable", -3)
        handshake_dim(input_coords, "lead_time", -4)
        handshake_dim(input_coords, "time", -5)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = input_coords.copy()
        output_coords["variable"] = np.array(self.output_variables)
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained cBottle model package from Nvidia model registry"""
        return Package(
            "ngc://models/nvidia/earth-2/cbottle@1.1",
            cache_options={
                "cache_storage": Package.default_cache("cbottle"),
                "same_names": True,
            },
        )

    @classmethod
    @check_extra_imports("cbottle", ["cbottle", "earth2grid"])
    def load_model(
        cls,
        package: Package,
        input_variables: list[str] | VariableArray = ["u10m", "v10m"],
        sampler_steps: int = 18,
        sigma_max: float = 80,
        seed: int = 0,
    ) -> DiagnosticModel:
        """Load AI datasource from package

        Parameters
        ----------
        package : Package
            CBottle AI model package
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

        with Checkpoint(package.resolve("cBottle-3d.zip")) as checkpoint:
            core_model = checkpoint.read_model()

        core_model.eval()
        core_model.requires_grad_(False)
        core_model.float()

        sst_ds = xr.open_dataset(
            package.resolve("amip_midmonth_sst.nc"),
            engine="h5netcdf",
            storage_options=None,
            cache=False,
        ).load()

        return cls(
            core_model,
            sst_ds,
            input_variables=input_variables,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
            seed=seed,
        )

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)

        time = output_coords["time"][:, None]
        lead = output_coords["lead_time"][None, :]
        time = [pd.to_datetime(t) for t in (time + lead).reshape(-1)]
        x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])

        input = self.get_cbottle_input(time, x)

        device = self.device_buffer.device
        condition = input["condition"].to(device)
        labels = input["labels"].to(device)
        images = input["images"].to(device)
        second_of_day = input["second_of_day"].to(device)
        day_of_year = input["day_of_year"].to(device)
        sigma_max = torch.Tensor([self.sigma_max]).to(device)

        # Process in batches with progress bar if verbose is enabled
        batch_size = self.batch_size
        n_samples = images.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        outputs = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)

            # Get batch slices
            batch_images = images[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            batch_condition = condition[start_idx:end_idx]
            batch_second_of_day = second_of_day[start_idx:end_idx]
            batch_day_of_year = day_of_year[start_idx:end_idx]

            # Generate latents
            batch_latents = torch.randn(
                (
                    end_idx - start_idx,
                    self.core_model.img_channels,
                    self.core_model.time_length,
                    self.core_model.domain.numel(),
                ),
                generator=self.rng,
            ).to(device)

            batch_xT = batch_latents * sigma_max

            # Gets appropriate denoiser based on config
            batch_D = get_denoiser(
                net=self.core_model,
                images=batch_images,
                labels=batch_labels,
                condition=batch_condition,
                second_of_day=batch_second_of_day,
                day_of_year=batch_day_of_year,
                denoiser_type=DenoiserType.infill,  # 'mask_filling', 'infill', 'standard'
                sigma_max=sigma_max,
                labels_when_nan=None,
            )

            batch_out = edm_sampler_from_sigma(
                batch_D,
                batch_xT,
                num_steps=self.sampler_steps,
                randn_like=torch.randn_like,
                sigma_max=int(sigma_max),  # Convert to int for type compatibility
            )

            batch_out = batch_out * self.output_stds + self.output_means
            outputs.append(batch_out)

        # Concatenate all batches
        x = torch.cat(outputs, dim=0)
        # Regrid as needed
        output = self._regrid_outputs(x)

        output = output.reshape(
            output_coords["batch"].shape[0],
            output_coords["time"].shape[0],
            output_coords["lead_time"].shape[0],
            output_coords["variable"].shape[0],
            output_coords["lat"].shape[0],
            output_coords["lon"].shape[0],
        )

        return output, output_coords

    def get_cbottle_input(
        self,
        time: list[datetime],
        x: torch.Tensor,
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
        x : torch.Tensor
            Input lat/lon variables used to condition infill
        label : int, optional
            Label ID, by default 1

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of input tensors for CBottle
        """
        # If SST is part of our inputs, use this for the condition
        if "sst" in self.input_coords()["variable"]:
            sst_idx = np.where(self.input_coords()["variable"] == "sst")[0]
            sst_data = x[..., sst_idx, :, :].clone()
            sst_data = self.input_regridder(sst_data.double()).squeeze()
        # If not we will use the AMIP SST the model was trained with to condition
        else:
            self._validate_sst_time(time)
            time_arr = np.array(time, dtype="datetime64")
            sst_data = torch.from_numpy(
                self.sst["tosbcs"].interp(time=time_arr, method="linear").values
                + 273.15
            ).to(self.device_buffer.device)
            sst_data = self.sst_regridder(sst_data)

        # Regrid in known variables and normalize, the BatchInfo will handle the denorm
        # for all outputs in the call function
        # I'd like to formally apologize for the inefficiencies in this part of the code
        x_data = self.input_regridder(x.double())
        x_data = (x_data - self.input_means) / self.input_stds

        def encode_sst(sstk: torch.Tensor, offset: float = 0.0) -> torch.Tensor:
            # https://github.com/NVlabs/cBottle/blob/ed96dfe35d87ecefa4846307807e8241c4b24e71/src/cbottle/datasets/dataset_2d.py#L925
            SST_LAND_FILL_VALUE = 290
            SST_MEAN = 287.6897  # K
            SST_SCALE = 15.5862  # K
            is_land = torch.isnan(sstk)
            monthly_sst = sstk + offset
            monthly_sst = torch.where(is_land, SST_LAND_FILL_VALUE, monthly_sst)
            condition = (monthly_sst - SST_MEAN) / SST_SCALE
            condition = condition[None, None, :]
            return condition

        def reorder(x: torch.Tensor) -> torch.Tensor:
            x = earth2grid.healpix.reorder(
                x, earth2grid.healpix.PixelOrder.NEST, earth2grid.healpix.HEALPIX_PAD_XY
            )
            return torch.permute(x, (2, 0, 1, 3))

        cond = reorder(encode_sst(sst_data))

        day_start = np.array([t.replace(hour=0, minute=0, second=0) for t in time])
        year_start = np.array([d.replace(month=1, day=1) for d in day_start])
        second_of_day = (time - day_start) / timedelta(seconds=1)
        day_of_year = (time - year_start) / timedelta(seconds=86400)

        # For infill we set everything to NaNs except the known fields
        # The denoiser will then fill in anything thats NaN
        images = torch.full(
            (len(time), self.output_variables.shape[0], 1, 4**HPX_LEVEL * 12),
            float("nan"),
            dtype=torch.double,  # Important for reproducibility
            device=self.device_buffer.device,
        )
        # Add known channels
        images[:, self.input_variable_idx] = x_data.unsqueeze(-2)

        labels = torch.nn.functional.one_hot(torch.tensor(label), num_classes=1024)
        labels = labels.unsqueeze(0).repeat(len(time), 1)

        out = {
            "images": images,
            "labels": labels,
            "condition": cond,
            "second_of_day": torch.tensor(second_of_day.astype(np.float32)).unsqueeze(
                1
            ),
            "day_of_year": torch.tensor(day_of_year.astype(np.float32)).unsqueeze(1),
        }

        return out

    def _regrid_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Regrids output tensor from model as needed (in this case to lat / lon)"""
        # Convert back into lat lon
        nlat, nlon = 721, 1440
        device = self.device_buffer.device
        latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
            nlat, nlon, includes_south_pole=True
        )
        regridder = earth2grid.get_regridder(
            self.core_model.domain._grid, latlon_grid
        ).to(device)

        return regridder(x).squeeze().float()

    def set_seed(self, seed: int) -> None:
        """Set seed of CBottle latent variable generator, this is not sufficient for
        exact reproducibility due to the sampler. Also set torch.manual_seed and or
        torch.cuda.manual_seed before executing model.

        Parameters
        ----------
        seed : int
            Seed value
        """
        self.rng.manual_seed(seed)

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
