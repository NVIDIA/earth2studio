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

import numpy as np
import torch

try:
    import earth2grid
    from cbottle import patchify
    from cbottle.checkpointing import Checkpoint
    from cbottle.datasets.base import TimeUnit
    from cbottle.datasets.dataset_3d import get_batch_info
    from cbottle.denoiser_factories import get_denoiser
    from cbottle.diffusion_samplers import (
        StackedRandomGenerator,
        edm_sampler,
        edm_sampler_from_sigma,
    )
    from earth2grid import healpix
except ImportError:
    earth2grid = None
    healpix = None
    patchify = None
    Checkpoint = None
    TimeUnit = None
    get_batch_info = None
    edm_sampler = None
    StackedRandomGenerator = None
    edm_sampler_from_sigma = None
    get_denoiser = None

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    check_extra_imports,
    handshake_coords,
    handshake_dim,
)
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
    which corresponds to 5km resolution at the equator. The model can also be used
    to generate results on a smaller region of the globe by specifying a super-resolution
    window. This can often be desirable as full global results are extremely
    expensive to generate.

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
    output_resolution : Tuple[int, int], optional
        High-resolution output dimensions, by default (2161, 4320).
    super_resolution_window : Union[None, Tuple[int, int, int, int]], optional
        Super-resolution window. If None, super-resolution is done
        on the entire global grid. If provided, the super-resolution window is a tuple
        of (lat south, lon west, lat north, lon east) and will return results for the
        specified window, by default None
    sampler_steps : int, optional
        Number of diffusion steps, by default 18
    sigma_max : int, optional
        Maximum noise level for diffusion process, by default 800
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        output_resolution: tuple[int, int] = (2161, 4320),
        super_resolution_window: None | tuple[int, int, int, int] = None,
        sampler_steps: int = 18,
        sigma_max: int = 800,
    ):
        super().__init__()

        # Model
        self.core_model = core_model

        # Output shape
        self.output_resolution = output_resolution

        # Sampler
        self.sampler_steps = sampler_steps
        self.sigma_max = sigma_max

        # Super resolution window
        self.super_resolution_window = super_resolution_window
        if super_resolution_window is not None:
            self.output_lat = np.linspace(
                super_resolution_window[0],
                super_resolution_window[2],
                self.output_resolution[0],
            )
            self.output_lon = np.linspace(
                super_resolution_window[1],
                super_resolution_window[3],
                self.output_resolution[1],
            )
            self.inbox_patch_index = patchify.patch_index_from_bounding_box(
                HPX_LEVEL_HR, super_resolution_window, 128, 32, "cpu"
            )
        else:
            self.output_lat = np.linspace(90, -90, self.output_resolution[0])
            self.output_lon = np.linspace(
                0, 360, self.output_resolution[1], endpoint=False
            )
            self.inbox_patch_index = None

        # Make in and out regridders
        lat_lon_low_res_grid = earth2grid.latlon.equiangular_lat_lon_grid(
            721, 1440, includes_south_pole=False
        )
        self.hpx_high_res_grid = healpix.Grid(
            level=HPX_LEVEL_HR, pixel_order=healpix.PixelOrder.NEST
        )
        if super_resolution_window is not None:
            lat_lon_high_res_grid = earth2grid.latlon.LatLonGrid(
                self.output_lat, self.output_lon
            )
        else:
            lat_lon_high_res_grid = earth2grid.latlon.equiangular_lat_lon_grid(
                self.output_resolution[0],
                self.output_resolution[1],
                includes_south_pole=False,
            )
        self.regrid_latlon_low_res_to_hpx_high_res = earth2grid.get_regridder(
            lat_lon_low_res_grid, self.hpx_high_res_grid
        )
        self.regrid_latlon_low_res_to_hpx_high_res.double()
        self.regrid_hpx_high_res_to_latlon_high_res = earth2grid.get_regridder(
            self.hpx_high_res_grid, lat_lon_high_res_grid
        )
        self.regrid_hpx_high_res_to_latlon_high_res.double()

        # Make global lat lon regridder
        lat = torch.linspace(-90, 90, 128)[:, None].double()
        lon = torch.linspace(0, 360, 128)[None, :].double()
        self.regrid_to_latlon = self.hpx_high_res_grid.get_bilinear_regridder_to(
            lat, lon
        ).to(torch.float64)

        # Hard set scale and center
        scale = torch.tensor(
            [
                1.4847e-01,
                2.7247e-02,
                1.5605e01,
                5.1746e00,
                4.6485e00,
                4.1996e01,
                1.2832e02,
                1.1094e03,
                1.0940e-04,
                3.0466e02,
                8.5142e00,
                1.4541e-01,
            ],
            dtype=torch.float64,
        )[:, None]
        center = torch.tensor(
            [
                5.4994e-02,
                1.1090e-02,
                2.8609e02,
                -1.5407e-01,
                -3.8198e-01,
                2.4358e02,
                8.8927e01,
                1.0116e05,
                1.7416e-05,
                2.1382e02,
                2.9097e02,
                2.5404e-02,
            ],
            dtype=torch.float64,
        )[:, None]

        self.register_buffer("scale", scale)
        self.register_buffer("center", center)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES),
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

        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES),
                "lat": self.output_lat,
                "lon": self.output_lon,
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
        output_resolution: tuple[int, int] = (2161, 4320),
        super_resolution_window: None | tuple[int, int, int, int] = None,
        sampler_steps: int = 18,
        sigma_max: int = 800,
    ) -> DiagnosticModel:
        """Load AI datasource from package

        Parameters
        ----------
        package : Package
            CBottle AI model package
        output_resolution : tuple[int, int], optional
            High-resolution output dimensions, by default (2161, 4320)
        super_resolution_window : None | tuple[int, int, int, int], optional
            Super-resolution window, by default None
        sampler_steps : int, optional
            Number of diffusion steps, by default 18
        sigma_max : float, optional
            Noise amplitude used to generate latent variables, by default 800

        Returns
        -------
        DiagnosticModel
            Diagnostic model
        """

        with Checkpoint(package.resolve("cBottle-SR.zip")) as checkpoint:
            core_model = checkpoint.read_model()

        core_model.eval()
        core_model.requires_grad_(False)
        core_model.float()

        return cls(
            core_model,
            output_resolution=output_resolution,
            super_resolution_window=super_resolution_window,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
        )

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:

        # Get high res
        lr_hpx = self.regrid_latlon_low_res_to_hpx_high_res(x.double())

        # Normalize
        lr_hpx = (lr_hpx - self.center.to(x.device)) / self.scale.to(x.device)

        # Get global lat lon
        global_lr = self.regrid_to_latlon(lr_hpx)

        # Run denoiser
        latents = torch.randn_like(lr_hpx, device=x.device, dtype=torch.float64)

        # Add 1 batch dimension
        latents = latents[None,].float()
        lr_hpx = lr_hpx[None,].float()
        global_lr = global_lr[None,]

        # Set device inbox patch index
        if self.inbox_patch_index is not None:
            self.inbox_patch_index = self.inbox_patch_index.to(x.device)

        with torch.no_grad():
            # scope with global_lr and other inputs present
            def denoiser(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
                        inbox_patch_index=self.inbox_patch_index,
                        device=x.device,
                    )
                    .to(torch.float64)
                    .to(x.device)
                )

            denoiser.sigma_max = self.core_model.sigma_max  # type: ignore[attr-defined]
            denoiser.sigma_min = self.core_model.sigma_min  # type: ignore[attr-defined]
            denoiser.round_sigma = self.core_model.round_sigma  # type: ignore[attr-defined]

            pred = edm_sampler(
                denoiser,
                latents,
                num_steps=self.sampler_steps,
                sigma_max=self.sigma_max,
            )

        # Unnormalize
        pred = pred[0] * self.scale.to(x.device) + self.center.to(x.device)

        # Get lat lon high res
        output = torch.zeros(
            x.shape[0],
            self.output_lat.shape[0],
            self.output_lon.shape[0],
            device=x.device,
            dtype=torch.float32,
        )
        for i in range(x.shape[0]):
            output[i : i + 1] = self.regrid_hpx_high_res_to_latlon_high_res(
                pred[i : i + 1]
            ).to(torch.float32)

        return output

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
