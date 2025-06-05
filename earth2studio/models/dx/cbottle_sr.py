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

import zipfile
from collections import OrderedDict
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import zarr

try:
    import earth2grid
    from earth2grid import healpix
    from earth2grid.latlon import equiangular_lat_lon_grid
    from cbottle import patchify
    from cbottle.diffusion_samplers import edm_sampler
    from cbottle.checkpointing import Checkpoint
    from cbottle.datasets.base import TimeUnit
    from cbottle.datasets.dataset_2d import encode_sst
    from cbottle.datasets.dataset_3d import get_batch_info
    from cbottle.denoiser_factories import get_denoiser
    from cbottle.diffusion_samplers import (
        StackedRandomGenerator,
        edm_sampler_from_sigma,
    )
except ImportError:
    earth2grid = None
    healpix = None
    equiangular_lat_lon_grid = None
    patchify = None
    edm_sampler = None
    Checkpoint = None
    StackedRandomGenerator = None
    edm_sampler_from_sigma = None
    get_batch_info = None
    TimeUnit = None
    get_denoiser = None

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    check_extra_imports,
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.interp import latlon_interpolation_regular
from earth2studio.utils.type import CoordSystem

VARIABLES = [
    "tclw",
    "tciw",
    "t2m",
    "u10m",
    "v10m",
    "rlut",
    "rsut",
    "msl",
    "tpf",
    "rsds",
    "sst",
    "sic",
]

HPX_LEVEL_LR = 6
HPX_LEVEL_HR = 10


@check_extra_imports("cbottle", ["cbottle", "earth2grid"])
class CBottleSR(torch.nn.Module, AutoModelMixin):
    """Climate in a Bottle Super-Resolution (CBottleSR) model.

    CBottleSR is a diffusion-based super-resolution model that learns mappings between
    low- and high-resolution climate data with high fidelity. This model generates
    results at 5km resolution on a healpix grid with 10 levels of resolution (1024x1024).
    The results are then regridded to a lat/lon grid. Suggested output dimensions are
    (2161, 4320) which corresponds to 10km resolution at the equator or (4321, 8640)
    which corresponds to 5km resolution at the equator.

    Note
    ----
    For more information see the following references:
    - https://arxiv.org/abs/2505.06474v1
    - https://github.com/NVlabs/cBottle
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/cbottle

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model implementing the diffusion-based super-resolution
    hr_latlon : Tuple[int, int], optional
        High-resolution output dimensions (lat, lon), by default (2161, 4320)
    num_steps : int, optional
        Number of diffusion steps, by default 18
    sigma_max : int, optional
        Maximum noise level for diffusion process, by default 800
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        hr_latlon: Tuple[int, int] = (2161, 4320),
        num_steps: int = 18,
        sigma_max: int = 800,
    ):
        super().__init__()

        # Model
        self.core_model = core_model

        # Output shape
        self.hr_latlon = hr_latlon

        # Sampler
        self.num_steps = num_steps
        self.sigma_max = sigma_max

        # Make in and out regridders
        lat_lon_low_res_grid = equiangular_lat_lon_grid(721, 1440, includes_south_pole=False)
        self.hpx_high_res_grid = healpix.Grid(level=HPX_LEVEL_HR, pixel_order=healpix.PixelOrder.NEST)
        lat_lon_high_res_grid = equiangular_lat_lon_grid(self.hr_latlon[0], self.hr_latlon[1], includes_south_pole=False)
        self.regrid_latlon_low_res_to_hpx_high_res = earth2grid.get_regridder(lat_lon_low_res_grid, self.hpx_high_res_grid)
        self.regrid_latlon_low_res_to_hpx_high_res.double()
        self.regrid_hpx_high_res_to_latlon_high_res = earth2grid.get_regridder(self.hpx_high_res_grid, lat_lon_high_res_grid)
        self.regrid_hpx_high_res_to_latlon_high_res.double()

        # Make global lat lon regridder
        lat = torch.linspace(-90, 90, 128)[:, None].double()
        lon = torch.linspace(0, 360, 128)[None, :].double()
        self.regrid_to_latlon = self.hpx_high_res_grid.get_bilinear_regridder_to(lat, lon).to(torch.float64)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 721, endpoint=False),
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

        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, self.hr_latlon[0], endpoint=False),
                "lon": np.linspace(0, 360, self.hr_latlon[1], endpoint=False),
            }
        )

        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained CBottle3D model package from Nvidia model registry"""
        return Package(
            "ngc://models/nvidia/earth-2/cbottle@1.0",
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
        num_steps: int = 18,
        sigma_max: int = 800,
        hr_latlon: Tuple[int, int] = (2161, 4320),
    ) -> DiagnosticModel:
        """Load AI datasource from package"""

        with Checkpoint(package.resolve("cBottle-SR.zip")) as checkpoint:
            core_model = checkpoint.read_model()

        core_model.eval()
        core_model.requires_grad_(False)
        core_model.float()

        return cls(core_model, num_steps=num_steps, sigma_max=sigma_max, hr_latlon=hr_latlon)

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:

        # Get high res
        lr_hpx = self.regrid_latlon_low_res_to_hpx_high_res(x.double())

        # Get global lat lon
        global_lr = self.regrid_to_latlon(lr_hpx)

        # Run denoiser
        latents = torch.randn_like(lr_hpx, device=x.device, dtype=torch.float64)

        # Add 1 batch dimension
        latents = latents[None,]
        lr_hpx = lr_hpx[None,]
        global_lr = global_lr[None,]

        with torch.no_grad():
            # scope with global_lr and other inputs present
            def denoiser(x, t):
                return (
                    patchify.apply_on_patches(
                        self.core_model,
                        patch_size=128,
                        overlap_size=32,
                        x_hat=x,
                        x_lr=lr_hpx,
                        t_hat=t,
                        class_labels=None,
                        batch_size=128,
                        global_lr=global_lr,
                        inbox_patch_index=None,
                        device=x.device,
                    )
                    .to(torch.float64)
                    .to(x.device)
                )

            denoiser.sigma_max = self.core_model.sigma_max
            denoiser.sigma_min = self.core_model.sigma_min
            denoiser.round_sigma = self.core_model.round_sigma

            pred = edm_sampler(
                denoiser,
                latents,
                num_steps=self.num_steps,
                sigma_max=self.sigma_max,
            )

        # Reorder to ring order
        #pred = self.hpx_high_res_grid.reorder(healpix.PixelOrder.RING, pred)

        # Get lat lon high res
        hr_latlon = torch.zeros(x.shape[0], self.hr_latlon[0], self.hr_latlon[1], device=x.device, dtype=torch.float32)
        for i in range(x.shape[0]):
            hr_latlon[i:i+1] = self.regrid_hpx_high_res_to_latlon_high_res(pred[0, i:i+1]).to(torch.float32)

        return hr_latlon

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""
        output_coords = self.output_coords(coords)

        out = torch.zeros(
            [len(v) for v in output_coords.values()],
            device=x.device,
            dtype=torch.float32,
        )
        for i in range(out.shape[0]):
            out[i] = self._forward(x[i])

        return out, output_coords