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

from collections import OrderedDict

import numpy as np
import torch
import yaml  # type: ignore

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
    from climate_learn.data.precipmodule import LogTransform
    from climate_learn.data.processing.era5_constants import PRECIP_VARIABLES
    from climate_learn.models.hub import Res_Slim_ViT
    from climate_learn.utils.fused_attn import FusedAttn
    from climate_learn.utils.visualize import TileCoordinates, TileProcessor
except ImportError:
    OptionalDependencyFailure("orbit")
    LogTransform = None
    PRECIP_VARIABLES = None
    Res_Slim_ViT = None
    FusedAttn = None
    TileCoordinates = None
    TileProcessor = None

VARIABLES = [
    "t2m",
    "t200",
    "t500",
    "t850",
    "u10m",
    "u200",
    "u500",
    "u850",
    "v10m",
    "v200",
    "v500",
    "v850",
    "q200",
    "q500",
    "q850",
    "swvl1",
]

ORBIT_VARIABLE_MAPPING = [
    "2m_temperature",
    "temperature_200",
    "temperature_500",
    "temperature_850",
    "10m_u_component_of_wind",
    "u_component_of_wind_200",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "10m_v_component_of_wind",
    "v_component_of_wind_200",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
    "specific_humidity_200",
    "specific_humidity_500",
    "specific_humidity_850",
    "volumetric_soil_water_layer_1",
    "total_precipitation_24hr",
    "2m_temperature_max",
    "2m_temperature_min",
]

STATIC_VARIABLES = [
    "land_sea_mask",
    "orography",
    "lattitude",
    "landcover",
]

OUT_VARIABLES = [
    "total_precipitation_24hr",
]

IN_HEIGHT = 720
IN_WIDTH = 1440
OUT_HEIGHT = 2880
OUT_WIDTH = 5760


@check_optional_dependencies()
class OrbitGlobalPrecip(torch.nn.Module, AutoModelMixin):
    """ORBIT-2 precipitation downscaling model

    Note
    ----
    This model and checkpoint are from Wang et al. 2025. For more information see the
    following references:

    - https://dl.acm.org/doi/10.1145/3712285.3771989
    - https://dl.acm.org/doi/10.1145/3712285.3771989

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model
    land_sea_mask : np.ndarray
        Binary land-sea mask at 0.25° resolution, shape ``(720, 1440)``.
        Values are 1 over land, 0 over ocean.
    orography : np.ndarray
        Surface geopotential height (meters) at 0.25° resolution,
        shape ``(720, 1440)``.
    lattitude : np.ndarray
        Latitude values broadcast to grid shape ``(720, 1440)``, used as a
        positional encoding input to the model.
    landcover : np.ndarray
        Land-use / land-cover classification at 0.25° resolution,
        shape ``(720, 1440)``.
    normalize_mean_lowres : np.lib.npyio.NpzFile
        Per-variable mean values for input normalization. Keys are variable
        names, values are single-element arrays.
    normalize_std_lowres : np.lib.npyio.NpzFile
        Per-variable standard deviation values for input normalization. Keys
        are variable names, values are single-element arrays.
    normalize_mean_highres : np.lib.npyio.NpzFile
        Per-variable mean values for output denormalization. Keys are variable
        names, values are single-element arrays.
    normalize_std_highres : np.lib.npyio.NpzFile
        Per-variable standard deviation values for output denormalization. Keys
        are variable names, values are single-element arrays.
    do_tiling : bool
        Boolean to indicate whether tiled inference is performed
    div : int
        If performing tiling, number of tiles to divide input into
    overlap : int
        If performing tiling, number of overlap pixels to use during tiled inference

    Badges
    ------
    region:global class:mrf product:precip year:2025 gpu:60gb
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        land_sea_mask: np.ndarray,
        orography: np.ndarray,
        lattitude: np.ndarray,
        landcover: np.ndarray,
        normalize_mean_lowres: np.lib.npyio.NpzFile,
        normalize_std_lowres: np.lib.npyio.NpzFile,
        normalize_mean_highres: np.lib.npyio.NpzFile,
        normalize_std_highres: np.lib.npyio.NpzFile,
        do_tiling: bool,
        div: int,
        overlap: int,
    ) -> None:
        super().__init__()

        self.model = core_model

        self.in_variables = ORBIT_VARIABLE_MAPPING + STATIC_VARIABLES
        self.out_variables = OUT_VARIABLES

        # Register static variables as buffers (auto-move with .to(device))
        # Shape: (1, 1, H, W) so only batch expand is needed at forward time
        self.register_buffer(
            "land_sea_mask",
            torch.from_numpy(land_sea_mask).float().unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "orography",
            torch.from_numpy(orography).float().unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "lattitude",
            torch.from_numpy(lattitude).float().unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "landcover",
            torch.from_numpy(landcover).float().unsqueeze(0).unsqueeze(0),
        )

        # Build input normalization: mean/std vectors as buffers, precip mask
        # For precip channels, we use LogTransform instead of mean/std normalization
        normalize_mean_dict = dict(normalize_mean_lowres)
        normalize_std_dict = dict(normalize_std_lowres)
        n_channels = len(self.in_variables)
        norm_mean = torch.zeros(n_channels)
        norm_std = torch.ones(n_channels)
        precip_mask = torch.zeros(n_channels, dtype=torch.bool)
        for i, var in enumerate(self.in_variables):
            if var in PRECIP_VARIABLES:
                precip_mask[i] = True
            else:
                norm_mean[i] = float(normalize_mean_dict[var][0])
                norm_std[i] = float(normalize_std_dict[var][0])
        # Shape (1, C, 1, 1) for broadcasting over (B, C, H, W)
        self.register_buffer("norm_mean", norm_mean.view(1, -1, 1, 1))
        self.register_buffer("norm_std", norm_std.view(1, -1, 1, 1))
        self.register_buffer("precip_mask", precip_mask)
        self.log_transform = LogTransform(m2mm=True, LOG1P=True, thres_mm_per_day=0.25)

        # Build output denormalization: mean/std buffers
        denorm_mean_dict = dict(normalize_mean_highres)
        denorm_std_dict = dict(normalize_std_highres)
        n_out = len(self.out_variables)
        denorm_mean = torch.zeros(n_out)
        denorm_std = torch.ones(n_out)
        for i, var in enumerate(self.out_variables):
            if var not in PRECIP_VARIABLES:
                denorm_mean[i] = float(denorm_mean_dict[var][0])
                denorm_std[i] = float(denorm_std_dict[var][0])
        # Invert: denorm(x) = x * (1/std) - mean/std
        inv_std = 1.0 / denorm_std
        inv_mean = -denorm_mean * inv_std
        self.register_buffer("denorm_mean", inv_mean.view(1, -1, 1, 1))
        self.register_buffer("denorm_std", inv_std.view(1, -1, 1, 1))
        # Track which output channels are precip (no denorm, identity)
        out_precip_mask = torch.zeros(n_out, dtype=torch.bool)
        for i, var in enumerate(self.out_variables):
            if var in PRECIP_VARIABLES:
                out_precip_mask[i] = True
        self.register_buffer("out_precip_mask", out_precip_mask)

        self.do_tiling = do_tiling
        self.div = div
        self.overlap = overlap

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
                "variable": np.array(OUT_VARIABLES),
                "lat": np.linspace(90, -90, OUT_HEIGHT),
                "lon": np.linspace(0, 360, OUT_WIDTH, endpoint=False),
            }
        )

        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key != "batch":
                handshake_dim(input_coords, key, i)
                handshake_coords(input_coords, target_input_coords, key)
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "hf://jychoi-hpc/ORBIT-2",
            cache_options={
                "cache_storage": Package.default_cache("orbit-2"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    def load_model(
        cls, package: Package, model_type: str, model_size: str, model_variable: str
    ) -> DiagnosticModel:
        """Load diagnostic from package"""
        # Load YAML configuration
        with open(
            package.resolve(
                model_type
                + "-finetune"
                + "/"
                + model_type
                + "_"
                + model_size
                + "_"
                + model_variable
                + ".yaml"
            ),
        ) as f:
            conf = yaml.safe_load(f)

        try:
            do_tiling = conf["tiling"]["do_tiling"]
            if do_tiling:
                div = conf["tiling"]["div"]
                overlap = conf["tiling"]["overlap"]
            else:
                div = 1
                overlap = 0
        except Exception:
            do_tiling = False
            div = 1
            overlap = 0

        default_vars = conf["data"]["default_vars"]
        spatial_resolution = conf["data"]["spatial_resolution"]

        superres_mag = conf["model"]["superres_mag"]
        cnn_ratio = conf["model"]["cnn_ratio"]
        patch_size = conf["model"]["patch_size"]
        embed_dim = conf["model"]["embed_dim"]
        depth = conf["model"]["depth"]
        decoder_depth = conf["model"]["decoder_depth"]
        num_heads = conf["model"]["num_heads"]
        mlp_ratio = conf["model"]["mlp_ratio"]
        drop_path = conf["model"]["drop_path"]
        drop_rate = conf["model"]["drop_rate"]

        in_channels = len(ORBIT_VARIABLE_MAPPING + STATIC_VARIABLES)
        if do_tiling:
            if overlap % 2 != 0:
                raise ValueError("Only handling even overlapping for now")
            top = bottom = overlap // 2
            left = right = overlap // 2 * 2
            in_height = int(IN_HEIGHT / div + top + bottom)
            in_width = int(IN_WIDTH / div + left + right)
        else:
            in_height = IN_HEIGHT
            in_width = IN_WIDTH

        out_channels = 1

        model = Res_Slim_ViT(
            default_vars,
            (in_height, in_width),
            in_channels,
            out_channels,
            superres_mag=superres_mag,
            history=1,
            patch_size=patch_size,
            cnn_ratio=cnn_ratio,
            learn_pos_emb=True,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            FusedAttn_option=FusedAttn.DEFAULT,
        )
        model.data_config(
            spatial_resolution["ERA5_2"],
            (in_height, in_width),
            in_channels,
            out_channels,
        )

        map_location = "cpu"
        checkpoint = torch.load(
            package.resolve(
                model_type
                + "-finetune"
                + "/"
                + model_type
                + "_"
                + model_size
                + "_"
                + model_variable
                + ".ckpt"
            ),
            map_location=map_location,
            weights_only=True,
        )

        pretrain_model = checkpoint["model_state_dict"]
        del checkpoint

        model.load_state_dict(pretrain_model)

        land_sea_mask = np.load(
            package.resolve("static_variables/land_sea_mask_0.25deg.npy")
        )
        orography = np.load(package.resolve("static_variables/orography_0.25deg.npy"))
        lattitude = np.load(package.resolve("static_variables/lattitude_0.25deg.npy"))
        landcover = np.load(package.resolve("static_variables/landcover_0.25deg.npy"))

        normalize_mean_lowres = np.load(
            package.resolve("mean_std/era5/0.25_deg/normalize_mean.npz")
        )
        normalize_std_lowres = np.load(
            package.resolve("mean_std/era5/0.25_deg/normalize_std.npz")
        )

        normalize_mean_highres = np.load(
            package.resolve("mean_std/era5/0.25_deg/normalize_mean.npz")
        )
        normalize_std_highres = np.load(
            package.resolve("mean_std/era5/0.25_deg/normalize_std.npz")
        )

        return cls(
            model,
            land_sea_mask,
            orography,
            lattitude,
            landcover,
            normalize_mean_lowres,
            normalize_std_lowres,
            normalize_mean_highres,
            normalize_std_highres,
            do_tiling,
            div,
            overlap,
        )

    def _denormalize_output(self, yhat: torch.Tensor) -> torch.Tensor:
        """Denormalize model output using precomputed buffers.

        For non-precip channels: x_denorm = x * inv_std + inv_mean
        For precip channels (identity): mean=0, std=1 so no-op.
        """
        return yhat * self.denorm_std + self.denorm_mean

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess Earth2Studio data to match ORBIT-2 DATA"""
        # Input lat is [90, ..., -90] (721 points, descending)
        # Flip lat axis to get [-90, ..., 90] (ascending)
        x = torch.flip(x, dims=(2,))

        # Remove the last row (was 90° before flip, now at end) to get (720, 1440)
        x = x[:, :, :-1, :]

        # Add static variables (buffers are already on correct device)
        # Buffers are (1, 1, H, W), expand to (batch, 1, H, W)
        batch = x.shape[0]
        land_sea_mask = self.land_sea_mask.expand(batch, -1, -1, -1)
        orography = self.orography.expand(batch, -1, -1, -1)
        lattitude = self.lattitude.expand(batch, -1, -1, -1)
        landcover = self.landcover.expand(batch, -1, -1, -1)

        x = torch.cat((x, land_sea_mask, orography, lattitude, landcover), dim=1)

        # Normalize: apply log transform to precip channels, mean/std to rest
        for i in range(x.shape[1]):
            if self.precip_mask[i]:
                x[:, i] = self.log_transform(x[:, i])
        # Vectorized mean/std normalization (precip channels have mean=0, std=1)
        x = (x - self.norm_mean) / self.norm_std

        return x

    @staticmethod
    def clip_replace_constant(
        y: torch.Tensor, out_variables: list[str]
    ) -> torch.Tensor:
        """Postprocess Precipitation Data to get rid of unphysical values"""

        prcp_index = out_variables.index("total_precipitation_24hr")
        for i in range(y.shape[1]):
            if i == prcp_index:
                torch.clamp_(y[:, prcp_index, :, :], min=0.0)

        return y

    @staticmethod
    def adjust_coords_for_flip(
        coords: TileCoordinates, processor: "TileProcessor"
    ) -> TileCoordinates:
        """Adjust coordinates after vertical image flip."""
        # Adjust input tile extraction coordinates after flip
        # Calculate flipped y-coordinates for tile extraction
        tile_height_with_overlap = processor.yinp // processor.div + (
            processor.top + processor.bottom
        )
        yi2tp = tile_height_with_overlap - coords.yi1t
        yi1tp = tile_height_with_overlap - coords.yi2t
        coords.yi1t = yi1tp
        coords.yi2t = yi2tp

        # Calculate flipped y-coordinates for result placement
        yi2rp = processor.yinp - coords.yi1r
        yi1rp = processor.yinp - coords.yi2r
        coords.yi1r = yi1rp
        coords.yi2r = yi2rp

        # Adjust output tile extraction coordinates after flip
        output_tile_height_with_overlap = (
            processor.yout // processor.div
            + (processor.top + processor.bottom) * processor.vmul
        )
        yo2tp = output_tile_height_with_overlap - coords.yo1t
        yo1tp = output_tile_height_with_overlap - coords.yo2t
        coords.yo1t = yo1tp
        coords.yo2t = yo2tp

        yo2rp = processor.yout - coords.yo1r
        yo1rp = processor.yout - coords.yo2r
        coords.yo1r = yo1rp
        coords.yo2r = yo2rp

        return coords

    @staticmethod
    def stitch_tiles(
        tiles: list, tile_coords: list, processor: TileProcessor, batch_size: int
    ) -> torch.Tensor:
        """Stitch tiles together into complete images.

        Reconstructs full image from processed tiles, handling overlap regions.
        """
        preds = torch.zeros(
            (batch_size, 1, processor.yout, processor.xout),
            dtype=torch.float32,
            device=tiles[0].device,
        )

        # Place each tile in the correct position
        for i in range(len(tiles)):
            coords = tile_coords[i]

            # Place prediction
            preds[:, :, coords.yo1r : coords.yo2r, coords.xo1r : coords.xo2r] = tiles[
                i
            ][:, :, coords.yo1t : coords.yo2t, coords.xo1t : coords.xo2t]

        return preds

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        if self.do_tiling:
            x = self.preprocess_input(x)
            processor = TileProcessor(
                self.div,
                self.overlap,
                (IN_HEIGHT, IN_WIDTH),
                (OUT_HEIGHT, OUT_WIDTH),
                1,
            )
            tiles = []
            tile_coords = []
            for vindex in range(self.div):  # Vertical tile index
                for hindex in range(self.div):  # Horizontal tile index
                    # Get tile coordinates with overlap handling
                    coords = processor.get_tile_coordinates(hindex, vindex)

                    # Extract tile data from full tensors
                    # x_tile: [batch, channels, height, width] for input
                    x_tile = x[:, :, coords.yi1 : coords.yi2, coords.xi1 : coords.xi2]
                    yhat = self.model.forward(
                        x_tile, self.in_variables, self.out_variables
                    )
                    yhat = self.clip_replace_constant(yhat, self.out_variables)
                    yhat = self._denormalize_output(yhat)
                    yhat = torch.flip(yhat, dims=(2,))
                    tile_coords.append(self.adjust_coords_for_flip(coords, processor))
                    tiles.append(yhat)
            yhat = self.stitch_tiles(tiles, tile_coords, processor, yhat.shape[0])
        else:
            x = self.preprocess_input(x)
            yhat = self.model.forward(x, self.in_variables, self.out_variables)
            yhat = self.clip_replace_constant(yhat, self.out_variables)
            yhat = self._denormalize_output(yhat)
        return yhat

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""

        output_coords = self.output_coords(coords)

        with torch.no_grad():
            out = self._forward(x)

        return out, output_coords
