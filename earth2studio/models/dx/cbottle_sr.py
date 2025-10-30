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
from dataclasses import replace

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    import earth2grid
    from cbottle.inference import (
        Coords,
        DistilledSuperResolutionModel,
        SuperResolutionModel,
    )
    from earth2grid import healpix
except ImportError:
    OptionalDependencyFailure("cbottle")
    earth2grid = None
    healpix = None
    SuperResolutionModel = None
    DistilledSuperResolutionModel = None
    Coords = None

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

VARIABLE_TO_CHANNEL = {
    "tclw": "cllvi",
    "tciw": "clivi",
    "t2m": "tas",
    "u10m": "uas",
    "v10m": "vas",
    "rlut": "rlut",
    "rsut": "rsut",
    "msl": "pres_msl",
    "tpf": "pr",
    "rsds": "rsds",
    "sst": "sst",
    "sic": "sic",
}
CHANNEL_TO_VARIABLE = {v: k for k, v in VARIABLE_TO_CHANNEL.items()}

HPX_LEVEL_LR = 6
HPX_LEVEL_HR = 10


@check_optional_dependencies()
class CBottleSR(torch.nn.Module, AutoModelMixin):
    """Climate in a Bottle Super-Resolution (CBottleSR) model.

    CBottleSR is a diffusion-based super-resolution model that learns mappings between
    low- and high-resolution climate data with high fidelity. This model generates
    results at 5km resolution on a healpix grid with 10 levels of resolution (1024x1024).
    The results can be output in either HEALPix format or regridded to a lat/lon grid.
    If lat/lon is used for output, the results will be regridded to the specified output resolution.
    Suggested output resolutions are (2161, 4320) for ~10km equatorial resolution and (4321, 8640) for ~5km equatorial resolution.
    The model can also be used to generate results for a smaller region of the globe by specifying a super-resolution window.
    This is often desirable as full global results are computationally expensive.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2505.06474v1
    - https://github.com/NVlabs/cBottle
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/cbottle
    - HEALPix: https://healpix.sourceforge.io/

    Parameters
    ----------
    sr_model : torch.nn.Module
        Core cBottle super-resolution helper module implementing the diffusion process
    lat_lon : bool, optional, by default True
        Lat/lon toggle, if true the model will expect a lat/lon grid as input and output a lat/lon
        grid. If false, the native nested HealPix grid will be used for input and output.
        Input HEALPix is level 6 and output HEALPix is level 10 with NEST pixel ordering
    output_resolution : Tuple[int, int], optional
        High-resolution output dimensions for lat/lon output. Only used when
        lat_lon=True, by default (2161, 4320)
    super_resolution_window : Tuple[int, int, int, int] | None, optional
        Super-resolution window. If None, super-resolution is done on the entire global grid
        If provided, the super-resolution window is a tuple of (lat_south, lon_west, lat_north, lon_east)
        and will only apply super-resolution to the specified window. For lat/lon output, the result will
        just be returned for the specified window with the specified output resolution,
        by default None
    sampler_steps : int, optional
        Number of diffusion steps, by default 18
    sigma_max : int, optional
        Maximum noise level for diffusion process, by default 800
    seed : int, optional
        Random generator seed for latent variables, by default None
    """

    def __init__(
        self,
        sr_model: torch.nn.Module,
        lat_lon: bool = True,
        output_resolution: tuple[int, int] = (2161, 4320),
        super_resolution_window: tuple[int, int, int, int] | None = None,
        sampler_steps: int = 18,
        sigma_max: int = 800,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.sr_model = sr_model
        self.seed = seed
        self.sampler_steps = sampler_steps
        self.sigma_max = sigma_max
        self._sample_index = 0

        self.register_buffer(
            "_device_buffer",
            torch.empty(0),
            persistent=False,
        )

        # Configure grid types
        if lat_lon:
            self.input_type = "latlon"
            self.output_type = "latlon"
        else:
            self.input_type = "healpix"
            self.output_type = "healpix"

        # Validate batch info and create channel reorder indices
        # NOTE: This should never fail but keep it here for safety
        sr_channels = list(self.sr_model.batch_info.channels)
        missing = [ch for ch in sr_channels if ch not in CHANNEL_TO_VARIABLE]
        if missing:
            raise ValueError(
                "CBottleSR received unexpected channels from cBottle model: "
                + ", ".join(missing)
            )

        self.register_buffer(
            "_to_sr_index",
            torch.tensor(
                [VARIABLES.index(CHANNEL_TO_VARIABLE[ch]) for ch in sr_channels],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "_from_sr_index",
            torch.tensor(
                [sr_channels.index(VARIABLE_TO_CHANNEL[var]) for var in VARIABLES],
                dtype=torch.long,
            ),
        )

        # Setup grids and regridders
        self.hpx_low_res_grid = self.sr_model.low_res_grid
        self.hpx_high_res_grid = self.sr_model.high_res_grid

        # Setup grids and regridders based on input/output types
        # Input grids
        if self.input_type == "latlon":
            self.input_grid = earth2grid.latlon.equiangular_lat_lon_grid(
                721, 1440, includes_south_pole=False
            )
            self.regrid_input_to_hpx_low_res = earth2grid.get_regridder(
                self.input_grid, self.hpx_low_res_grid
            )
            if hasattr(self.regrid_input_to_hpx_low_res, "to"):
                self.regrid_input_to_hpx_low_res = self.regrid_input_to_hpx_low_res.to(
                    self.device
                ).double()
        else:
            self.input_grid = self.hpx_low_res_grid
            self.regrid_input_to_hpx_low_res = None

        self.output_resolution = output_resolution

        # Setup super resolution window (this will initialize output coords and patch index)
        self.set_super_resolution_window(super_resolution_window, output_resolution)

        self._coords = Coords(self.sr_model.batch_info, self.hpx_low_res_grid)

    @property
    def device(self) -> torch.device:
        """Current device for model buffers"""
        return self._device_buffer.device

    def set_super_resolution_window(
        self,
        super_resolution_window: tuple[int, int, int, int] | None = None,
        output_resolution: tuple[int, int] = (2161, 4320),
    ) -> None:
        """Set or update the super-resolution window

        Parameters
        ----------
        super_resolution_window : None | tuple[int, int, int, int], optional
            Super-resolution window. If None, super-resolution is done
            on the entire global grid. If provided, the super-resolution window is a tuple
            of (lat_south, lon_west, lat_north, lon_east) and will only apply super-resolution
            to the specified window. By default None.
        output_resolution : tuple[int, int] | None, optional
            High-resolution output dimensions for lat/lon output. Only used when
            lat_lon=True, by default (2161, 4320)
        """
        # Update super resolution window
        self.super_resolution_window = super_resolution_window
        self.super_resolution_extents = None
        if super_resolution_window is not None:
            lat_south, lon_west, lat_north, lon_east = super_resolution_window
            self.super_resolution_extents = (lon_west, lon_east, lat_south, lat_north)

        # Update output coordinates and grids based on output type
        if self.output_type == "latlon":
            if super_resolution_window is not None:
                self.output_lat = np.linspace(
                    super_resolution_window[0],
                    super_resolution_window[2],
                    output_resolution[0],
                )
                self.output_lon = np.linspace(
                    super_resolution_window[1],
                    super_resolution_window[3],
                    output_resolution[1],
                )
                # Update output grid
                self.output_grid = earth2grid.latlon.LatLonGrid(
                    self.output_lat, self.output_lon
                )
            else:
                self.output_lat = np.linspace(90, -90, output_resolution[0])
                self.output_lon = np.linspace(
                    0, 360, output_resolution[1], endpoint=False
                )
                # Update output grid
                self.output_grid = earth2grid.latlon.equiangular_lat_lon_grid(
                    output_resolution[0],
                    output_resolution[1],
                    includes_south_pole=False,
                )

            # Update regridder for high-res HEALPix to output
            self.regrid_hpx_high_res_to_output = earth2grid.get_regridder(
                self.hpx_high_res_grid, self.output_grid
            )
            if hasattr(self.regrid_hpx_high_res_to_output, "to"):
                self.regrid_hpx_high_res_to_output = (
                    self.regrid_hpx_high_res_to_output.to(self.device).double()
                )
        else:  # healpix output
            # For healpix output, the output grid is always the high-res healpix grid
            self.output_grid = self.hpx_high_res_grid
            self.regrid_hpx_high_res_to_output = None

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        if self.input_type == "latlon":
            return OrderedDict(
                {
                    "batch": np.empty(0),
                    "variable": np.array(VARIABLES),
                    "lat": np.linspace(90, -90, 721),
                    "lon": np.linspace(0, 360, 1440, endpoint=False),
                }
            )
        else:  # healpix
            # HEALPix level 6: nside = 2^6 = 64, npix = 64^2 * 12 = 49,152 pixels
            nside = 2**HPX_LEVEL_LR
            npix = nside**2 * 12
            return OrderedDict(
                {
                    "batch": np.empty(0),
                    "variable": np.array(VARIABLES),
                    "hpx": np.arange(npix),
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
        # Validate input coordinates against expected input coords
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "variable", 1)
        handshake_coords(input_coords, target_input_coords, "variable")

        if self.input_type == "latlon":
            handshake_dim(input_coords, "lon", 3)
            handshake_dim(input_coords, "lat", 2)
            handshake_coords(input_coords, target_input_coords, "lon")
            handshake_coords(input_coords, target_input_coords, "lat")
        else:  # healpix input
            handshake_dim(input_coords, "hpx", 2)
            handshake_coords(input_coords, target_input_coords, "hpx")

        # Build output coordinate system based on output type
        if self.output_type == "latlon":
            output_coords = OrderedDict(
                {
                    "batch": np.empty(0),
                    "variable": np.array(VARIABLES),
                    "lat": self.output_lat,
                    "lon": self.output_lon,
                }
            )
        else:  # healpix output
            # HEALPix level 10: nside = 2^10 = 1024, npix = 1024^2 * 12 = 12,582,912 pixels
            nside = 2**HPX_LEVEL_HR
            npix = nside**2 * 12
            output_coords = OrderedDict(
                {
                    "batch": np.empty(0),
                    "variable": np.array(VARIABLES),
                    "hpx": np.arange(npix),
                }
            )

        output_coords["batch"] = input_coords["batch"]
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
        output_resolution: tuple[int, int] = (2161, 4320),
        super_resolution_window: tuple[int, int, int, int] | None = None,
        sampler_steps: int = 18,
        sigma_max: int = 800,
        seed: int | None = None,
        distilled_model: bool = False,
        device: str = "cpu",
    ) -> DiagnosticModel:
        """Load diagnostic model from package

        Parameters
        ----------
        package : Package
            CBottle AI model package
        lat_lon : bool, optional, by default True
            Lat/lon toggle, if true the model will expect a lat/lon grid as input and output a lat/lon
            grid. If false, the native nested HEALPix grid will be used for input and output
            Input HEALPix is level 6 and output HEALPix is level 10 with NEST pixel ordering
        output_resolution : tuple[int, int], optional
            High-resolution output dimensions for lat/lon output, by default (2161, 4320)
        super_resolution_window : tuple[int, int, int, int] | None, optional
            Super-resolution window for lat/lon output, by default None
        sampler_steps : int, optional
            Number of diffusion steps, by default 18
        sigma_max : float, optional
            Noise amplitude used to generate latent variables, by default 800
        seed : int, optional
            Random generator seed for latent variables, by default None
        distilled_model : bool, optional
            Whether to use the distilled model, If True, the distilled helper is used,
            enabling generation with fewer sampler steps, by default False
        device : str
            Device to load model onto, by default cpu

        Returns
        -------
        DiagnosticModel
            Diagnostic model
        """
        checkpoint_name = (
            "cBottle-SR-Distill.zip" if distilled_model else "cBottle-SR.zip"
        )
        model_cls = (
            DistilledSuperResolutionModel if distilled_model else SuperResolutionModel
        )
        state_path = package.resolve(checkpoint_name)

        if distilled_model:
            sr_model = model_cls.from_pretrained(
                state_path, window_function="KBD", window_alpha=1.0, device=device
            )
        else:
            sr_model = model_cls.from_pretrained(state_path, device=device)

        return cls(
            sr_model,
            lat_lon=lat_lon,
            output_resolution=output_resolution,
            super_resolution_window=super_resolution_window,
            sampler_steps=sampler_steps,
            sigma_max=sigma_max,
            seed=seed,
        )

    def _reorder_to_sr_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder channels to the super resolution model"""
        return torch.index_select(x, 0, self._to_sr_index)

    def _reorder_from_sr_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder channels from the super resolution model"""
        return torch.index_select(x, 0, self._from_sr_index)

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super resolve the input tensor"""
        x = self._reorder_to_sr_channels(x)

        if self.input_type == "latlon":
            x = self.regrid_input_to_hpx_low_res(x.double()).to(torch.float32)
        else:
            x = x.to(torch.float32)

        self.sr_model.num_steps = self.sampler_steps
        self.sr_model.sigma_max = self.sigma_max
        self.sr_model.device = self.device  # Terrible software

        x = x.unsqueeze(0).unsqueeze(2)

        if self.seed is not None:
            rng_devices: list[torch.device] = []
            if self.device.type == "cuda":
                rng_devices.append(self.device)

            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                seed = self.seed + self._sample_index
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                out, _ = self.sr_model(
                    x,
                    coords=replace(self._coords),
                    extents=self.super_resolution_extents,
                )
        else:
            out, _ = self.sr_model(
                x,
                coords=replace(self._coords),
                extents=self.super_resolution_extents,
            )

        self._sample_index += 1

        out = self._reorder_from_sr_channels(out[0, :, 0])
        if self.output_type == "healpix":
            return out.to(torch.float32)

        out = out[None,].double()
        out = self.regrid_hpx_high_res_to_output(out).to(torch.float32)
        return out[0]

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
