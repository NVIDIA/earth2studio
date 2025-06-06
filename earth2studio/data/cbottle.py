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

import os
import pathlib
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

try:
    import earth2grid
    from cbottle.checkpointing import Checkpoint
    from cbottle.datasets.base import TimeUnit
    from cbottle.datasets.dataset_2d import encode_sst
    from cbottle.datasets.dataset_3d import get_batch_info
    from cbottle.denoiser_factories import DenoiserType, get_denoiser
    from cbottle.diffusion_samplers import edm_sampler_from_sigma
except ImportError:
    earth2grid = None
    Checkpoint = None
    edm_sampler_from_sigma = None
    get_batch_info = None
    TimeUnit = None
    get_denoiser = None

from earth2studio.data.base import DataSource
from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import CBottleLexicon
from earth2studio.models.auto import Package
from earth2studio.models.auto.mixin import AutoModelMixin
from earth2studio.utils.imports import check_extra_imports
from earth2studio.utils.type import TimeArray, VariableArray

HPX_LEVEL = 6


@check_extra_imports("cbottle", ["cbottle", "earth2grid"])
class CBottle3D(torch.nn.Module, AutoModelMixin):
    """Climate in a bottle data source
    Climate in a Bottle (cBottle) is an AI model for emulating global km-scale climate
    simulations and reanalysis on the equal-area HEALPix grid. The cBottle data source
    uses the a globally-trained coarse-resolution image diffusion model that generates
    100km (50k-pixel) fields given monthly average sea surface temperatures and solar
    conditioning.

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
        Noise amplitude used to generate latent variables, by default 80
    batch_size : int, optional
        Batch size to generate time samples at, consider adjusting based on hardware
        being used, by default 4
    seed : int, optional
        Random generator seed for latent variables, by default 0
    cache : bool, optional
        Does nothing at the moment, by default False
    verbose : bool, optional
        Print generation progress, by default True
    """

    VARIABLES = np.array(list(CBottleLexicon.VOCAB.keys()))

    def __init__(
        self,
        core_model: torch.nn.Module,
        sst_ds: xr.Dataset,
        lat_lon: bool = True,
        sampler_steps: int = 18,
        sigma_max: float = 80,
        batch_size: int = 4,
        seed: int = 0,
        cache: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        self.core_model = core_model
        self.sst = sst_ds
        self.lat_lon = lat_lon
        self.sigma_max = sigma_max
        self.sampler_steps = sampler_steps
        self.batch_size = batch_size
        self.rng = torch.Generator()

        self._cache = cache
        self._verbose = verbose

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

        # Empty tensor just to make tracking current device easier
        self.register_buffer("device_buffer", torch.empty(0))
        # Set seed of random generator
        self.set_seed(seed=seed)

    @torch.inference_mode()
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for (UTC).
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to variables to
            return. Must be in CBottle3D lexicon.

        Returns
        -------
        xr.DataArray
            Generated data from CBottle
        """
        time, variable = prep_data_inputs(time, variable)
        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        input = self.get_cbottle_input(time)
        batch_info = get_batch_info(time_step=0, time_unit=TimeUnit.HOUR)

        varidx = []
        for var in variable:
            idx = np.where(self.VARIABLES == var)[0]
            if len(idx) == 0:
                raise ValueError(
                    f"Variable {var} not found in CBottle3D lexicon variables"
                )
            varidx.append(idx[0])
        varidx = np.array(varidx)

        device = self.device_buffer.device
        condition = input["condition"].to(device)
        labels = input["labels"].to(device)
        images = input["target"].to(device)
        second_of_day = input["second_of_day"].to(device)
        day_of_year = input["day_of_year"].to(device)
        sigma_max = torch.Tensor([self.sigma_max]).to(device)

        # Process in batches with progress bar if verbose is enabled
        batch_size = self.batch_size
        n_samples = images.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        outputs = []
        for i in tqdm(
            range(n_batches),
            desc="Generating cBottle Data",
            disable=(not self._verbose),
        ):
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
                denoiser_type=DenoiserType.standard,  # 'mask_filling', 'infill', 'standard'
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

            batch_x = batch_info.denormalize(batch_out)
            batch_x = batch_x[:, varidx]
            outputs.append(batch_x)

        # Concatenate all batches
        x = torch.cat(outputs, dim=0)

        if self.lat_lon:
            # Convert back into lat lon
            nlat, nlon = 721, 1440
            latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(
                nlat, nlon, includes_south_pole=True
            )
            regridder = earth2grid.get_regridder(
                self.core_model.domain._grid, latlon_grid
            ).to(device)
            field_regridded = regridder(x).squeeze(2)

            return xr.DataArray(
                data=field_regridded.cpu().numpy(),
                dims=["time", "variable", "lat", "lon"],
                coords={
                    "time": np.array(time),
                    "variable": np.array(variable),
                    "lat": np.linspace(90, -90, nlat, endpoint=False),
                    "lon": np.linspace(0, 360, nlon, endpoint=False),
                },
            )
        else:
            return xr.DataArray(
                data=x[:, :, 0].cpu().numpy(),
                dims=["time", "variable", "hpx"],
                coords={
                    "time": np.array(time),
                    "variable": np.array(variable),
                    "hpx": np.arange(x.shape[-1]),
                },
            )

    def get_cbottle_input(
        self,
        time: list[datetime],
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
        time_arr = np.array(time, dtype="datetime64[ns]")
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

        day_start = np.array([t.replace(hour=0, minute=0, second=0) for t in time])
        year_start = np.array([d.replace(month=1, day=1) for d in day_start])
        second_of_day = (time - day_start) / timedelta(seconds=1)
        day_of_year = (time - year_start) / timedelta(seconds=86400)

        # ["rlut", "rsut", "rsds"]
        nan_channels = [38, 39, 42]
        target = np.zeros(
            (len(time), self.VARIABLES.shape[0], 1, 4**HPX_LEVEL * 12), dtype=np.float32
        )
        target[:, nan_channels, ...] = np.nan

        labels = torch.nn.functional.one_hot(torch.tensor(label), num_classes=1024)
        labels = labels.unsqueeze(0).repeat(len(time), 1)

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

    def set_seed(self, seed: int) -> None:
        """Set seed of CBottle latent variable generator

        Parameters
        ----------
        seed : int
            Seed value
        """
        self.rng.manual_seed(seed)

    @property
    def cache(self) -> str:
        """Get the appropriate cache location."""
        cache_location = os.path.join(datasource_cache_root(), "cbottle")
        if not self._cache:
            cache_location = os.path.join(cache_location, "tmp_cbottle")
        return cache_location

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Verify if date time is valid for CBottle3D, governed but the CMIP SST data
        used to train it

        Parameters
        ----------
        times : list[datetime]
            list of date times to fetch data
        """
        for time in times:

            if time < datetime(year=1940, month=1, day=1):
                raise ValueError(
                    f"Requested date time {time} needs to be after January 1st, 1940 for CBottle3D"
                )

            if time >= datetime(year=2022, month=12, day=16, hour=12):
                raise ValueError(
                    f"Requested date time {time} needs to be before December 16th, 2022 for CBottle3D"
                )

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained CBottle3D model package from Nvidia model registry"""
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
        lat_lon: bool = True,
        batch_size: int = 4,
        seed: int = 0,
        verbose: bool = True,
    ) -> DataSource:
        """Load AI datasource from package

        Parameters
        ----------
        package : Package
            CBottle AI model package
        lat_lon : bool, optional
            Lat/lon toggle, if true data source will return output on a 0.25 deg lat/lon
            grid. If false, the native nested HealPix grid will be returned, by default
            True
        batch_size : int, optional
            Batch size to generate time samples at, consider adjusting based on hardware
            being used, by default 4
        seed : int, optional
            Random generator seed for latent variables, by default 0
        verbose : bool, optional
            Print generation progress, by default True

        Returns
        -------
        DataSource
            Data source
        """

        with Checkpoint(package.resolve("cBottle-3d.zip")) as checkpoint:
            core_model = checkpoint.read_model()

        core_model.eval()
        core_model.requires_grad_(False)
        core_model.float()

        # The following code is left here for reference of how to access the AMIP SST
        # data from the original data store. NGC is faster and cleaner so it is also
        # provided there.
        # sst_url = "https://esgf.ceda.ac.uk/thredds/dodsC/esg_cmip6/input4MIPs/CMIP6Plus/CMIP/PCMDI/PCMDI-AMIP-1-1-9/ocean/mon/tosbcs/gn/v20230512/"
        # sst_file = (
        #     "tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc"
        # )
        # sst_ds = xr.open_dataset(
        #     urljoin(sst_url, sst_file),
        #     engine="netcdf4",
        #     cache=False,
        # ).load()

        sst_ds = xr.open_dataset(
            package.resolve("amip_midmonth_sst.nc"),
            engine="netcdf4",
            cache=False,
        ).load()

        return cls(
            core_model,
            sst_ds,
            lat_lon=lat_lon,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
        )
